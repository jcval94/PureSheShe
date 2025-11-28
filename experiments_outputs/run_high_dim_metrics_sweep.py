"""Ejecuta un sweep de parámetros en el dataset high-dim y exporta métricas.

El script reutiliza el pipeline simplificado usado en los A/B tests pero lo
aplica sobre el dataset sintético de 30k muestras y 25 dimensiones empleado en
los benchmarks ``high_dim_run_*``.  Para cada configuración de
``find_low_dim_spaces`` se calcula el Top-K por clase y se vuelca a CSV la F1 y
el lift de precisión mediante :func:`deldel.describe_regions_metrics`.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from time import perf_counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deldel import (  # noqa: E402
    describe_regions_metrics,
    find_low_dim_spaces,
    make_high_dim_classification_dataset,
)
from deldel.experiments import _build_demo_selection  # noqa: E402


ParamGrid = Sequence[Mapping[str, object]]


def _summaries_by_class(rows: Iterable[Mapping[str, object]]) -> Dict[int, Dict[str, float]]:
    best: Dict[int, Dict[str, float]] = {}
    for row in rows:
        cls = int(row.get("class_id", -1))
        best.setdefault(cls, dict(best_f1=0.0, best_lift=0.0))
        f1 = float(row.get("f1") or 0.0)
        lift = float(row.get("lift_precision") or 0.0)
        if f1 > best[cls]["best_f1"]:
            best[cls]["best_f1"] = f1
        if lift > best[cls]["best_lift"]:
            best[cls]["best_lift"] = lift
    return best


def _run_single(
    params: Mapping[str, object],
    X,
    y,
    sel,
    *,
    feature_names: Sequence[str],
    dataset_size: int,
    top_per_class: int = 5,
) -> Tuple[List[Dict[str, object]], float]:
    start = perf_counter()
    valuable = find_low_dim_spaces(
        X,
        y,
        sel,
        feature_names=list(feature_names),
        **params,
    )
    duration = perf_counter() - start
    rows = describe_regions_metrics(
        valuable,
        top_per_class=top_per_class,
        dataset_size=dataset_size,
    )
    return rows, duration


def main() -> None:
    dataset_kwargs = dict(
        n_samples=20_000,
        n_features=25,
        n_informative=18,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        class_sep=2.1,
        random_state=11,
    )
    X, y, feature_names = make_high_dim_classification_dataset(**dataset_kwargs)
    sel = _build_demo_selection(X, y, feature_names)
    dataset_size = int(y.shape[0])

    # Barrido mucho más agresivo: combinaciones que tensan soporte mínimo,
    # profundidad de las reglas y ganancias relativas. Se limita
    # consider_dims_up_to a 3 para que el runtime siga siendo asumible con un dataset de 20k×25.
    param_grid: ParamGrid = [
        # Base conservadora (para comparar)
        dict(
            max_planes_in_rule=2,
            max_planes_per_pair=3,
            min_support=25,
            min_rel_gain_f1=0.03,
            min_lift_prec=1.20,
            consider_dims_up_to=3,
            rng_seed=0,
        ),
        # Más planos por regla y soporte mínimo (recall agresivo)
        dict(
            max_planes_in_rule=4,
            max_planes_per_pair=5,
            min_support=15,
            min_rel_gain_f1=0.02,
            min_lift_prec=1.05,
            consider_dims_up_to=3,
            rng_seed=1,
        ),
        # Reglas largas con soporte medio-alto y lift duro
        dict(
            max_planes_in_rule=5,
            max_planes_per_pair=6,
            min_support=45,
            min_rel_gain_f1=0.08,
            min_lift_prec=2.20,
            consider_dims_up_to=3,
            rng_seed=2,
        ),
        # Apilamiento profundo con soporte laxo (stress-test)
        dict(
            max_planes_in_rule=6,
            max_planes_per_pair=6,
            min_support=12,
            min_rel_gain_f1=0.015,
            min_lift_prec=1.10,
            consider_dims_up_to=3,
            rng_seed=3,
        ),
        # Búsqueda amplia con soporte extremo y lift exigente
        dict(
            max_planes_in_rule=4,
            max_planes_per_pair=5,
            min_support=60,
            min_rel_gain_f1=0.10,
            min_lift_prec=2.50,
            consider_dims_up_to=3,
            rng_seed=4,
        ),
    ]

    summary_rows: List[Dict[str, object]] = []
    region_rows: List[Dict[str, object]] = []

    for run_idx, params in enumerate(param_grid):
        metrics_rows, elapsed = _run_single(
            params,
            X,
            y,
            sel,
            feature_names=feature_names,
            dataset_size=dataset_size,
        )
        summaries = _summaries_by_class(metrics_rows)
        summary_rows.append(
            dict(
                run_index=run_idx,
                runtime_s=f"{elapsed:.6f}",
                params_json=str(dict(params)),
                best_f1_c0=summaries.get(0, {}).get("best_f1", 0.0),
                best_lift_c0=summaries.get(0, {}).get("best_lift", 0.0),
                best_f1_c1=summaries.get(1, {}).get("best_f1", 0.0),
                best_lift_c1=summaries.get(1, {}).get("best_lift", 0.0),
                best_f1_c2=summaries.get(2, {}).get("best_f1", 0.0),
                best_lift_c2=summaries.get(2, {}).get("best_lift", 0.0),
            )
        )
        for row in metrics_rows:
            region_rows.append(
                dict(
                    run_index=run_idx,
                    params_json=str(dict(params)),
                    **row,
                )
            )

    summary_path = ROOT / "experiments_outputs" / "high_dim_metrics_summary.csv"
    region_path = ROOT / "experiments_outputs" / "high_dim_metrics_by_region.csv"

    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(summary_rows[0].keys()) if summary_rows else [],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    if region_rows:
        fieldnames = list(region_rows[0].keys())
        with region_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(region_rows)

    print(f"Resúmenes guardados en {summary_path}")
    print(f"Métricas por región guardadas en {region_path}")


if __name__ == "__main__":
    main()
