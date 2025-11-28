"""Ejecuta un experimento grande en el dataset corner usando un criterio
"brújula" conservador basado en la mejor región por clase.

El pipeline genera un dataset grande con ``make_corner_class_dataset``, extrae
planos frontera con :class:`DelDel`, los poda con
:func:`prune_and_orient_planes_unified_globalmaj` y explora reglas de baja
dimensión mediante :func:`find_low_dim_spaces`.  Finalmente, ``nueva_funcion``
resume **todas** las regiones, identifica la mejor por clase y calcula un
``compass_score`` que combina F1, lift de precisión y el mínimo lift observado
entre todas las clases cubiertas por la región.  Esa métrica es la que debe
maximizarse a partir de ahora para comparar mejoras o regresiones.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deldel import (  # noqa: E402
    ChangePointConfig,
    DelDel,
    DelDelConfig,
    compute_frontier_planes_all_modes,
    find_low_dim_spaces,
    make_corner_class_dataset,
    prune_and_orient_planes_unified_globalmaj,
)


def _pairs_from_records(records: Sequence[Any]) -> List[Tuple[int, int]]:
    pairs = set()
    for rec in records:
        if getattr(rec, "success", True) and getattr(rec, "y0", None) is not None and getattr(rec, "y1", None) is not None:
            a, b = int(rec.y0), int(rec.y1)
            if a != b:
                pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def _min_precision_all_classes(region: Mapping[str, Any]) -> float | None:
    metrics_pc = region.get("metrics_per_class") or {}
    precisions: List[float] = []
    for entry in metrics_pc.values():
        try:
            precisions.append(float(entry.get("precision", 0.0)))
        except (TypeError, ValueError):
            continue
    return min(precisions) if precisions else None


def _min_lift_all_classes(region: Mapping[str, Any]) -> float | None:
    metrics_pc = region.get("metrics_per_class") or {}
    lifts: List[float] = []
    for entry in metrics_pc.values():
        try:
            lifts.append(float(entry.get("lift_precision", 0.0)))
        except (TypeError, ValueError):
            continue
    return min(lifts) if lifts else None


def _geo_mean(values: Sequence[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and float(v) > 0.0]
    if not vals:
        return None
    return float(np.exp(np.mean(np.log(vals))))


def nueva_funcion(valuable: Dict[int, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Flatten regions, score them conservadoramente y elegir la mejor por clase."""

    flattened: List[Dict[str, Any]] = []
    best_by_class: Dict[int, Dict[str, Any]] = {}

    for dim_k in sorted(valuable.keys()):
        for region in valuable.get(dim_k, []) or []:
            metrics = region.get("metrics", {}) or {}
            min_lift = _min_lift_all_classes(region)
            compass_score = _geo_mean([metrics.get("f1"), metrics.get("lift_precision"), min_lift])
            flattened.append(
                dict(
                    dim_k=dim_k,
                    target_class=region.get("target_class"),
                    region_id=region.get("region_id"),
                    dims=tuple(region.get("dims") or ()),
                    is_pareto=bool(region.get("is_pareto", False)),
                    f1=metrics.get("f1"),
                    precision=metrics.get("precision"),
                    recall=metrics.get("recall"),
                    lift_precision=metrics.get("lift_precision"),
                    support=metrics.get("size"),
                    min_lift_precision_all_classes=min_lift,
                    min_precision_all_classes=_min_precision_all_classes(region),
                    compass_score=compass_score,
                )
            )

            cls = region.get("target_class")
            if cls is None:
                continue

            def _as_key(entry: Mapping[str, Any]) -> Tuple[float, float, float, int]:
                return (
                    float(entry.get("compass_score") or -np.inf),
                    float(entry.get("f1") or -np.inf),
                    float(entry.get("lift_precision") or -np.inf),
                    int(entry.get("support") or 0),
                )

            current_best = best_by_class.get(int(cls))
            candidate = flattened[-1]
            if current_best is None or _as_key(candidate) > _as_key(current_best):
                best_by_class[int(cls)] = candidate

    flattened.sort(
        key=lambda r: (
            r["compass_score"] is None,
            -(r["compass_score"] or 0.0),
            -(r["f1"] or 0.0),
            -(r["lift_precision"] or 0.0),
        )
    )
    best_class_list = sorted(best_by_class.values(), key=lambda r: int(r["target_class"]))
    return flattened, best_class_list


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # Dataset grande (4 dims, miles de muestras por clase)
    dataset_kwargs = dict(
        n_per_cluster=1500,
        std_class1=0.45,
        std_other=0.85,
        a=2.8,
        random_state=0,
    )

    t0 = perf_counter()
    X, y, feature_names = make_corner_class_dataset(**dataset_kwargs)
    dataset_time = perf_counter() - t0

    rf = RandomForestClassifier(n_estimators=50, random_state=0)
    t1 = perf_counter()
    rf.fit(X, y)
    model_time = perf_counter() - t1

    cfg = DelDelConfig(segments_target=900, random_state=0)
    cp_cfg = ChangePointConfig(
        enabled=True,
        mode="treefast",
        per_record_max_points=6,
        max_candidates=144,
        max_bisect_iters=6,
    )
    engine = DelDel(cfg, cp_cfg)
    t2 = perf_counter()
    engine.fit(X, rf)
    deldel_time = perf_counter() - t2

    records = list(engine.records_)
    pairs = _pairs_from_records(records)

    frontier_kwargs = dict(mode="C", seed=0, verbosity=0)
    t3 = perf_counter()
    res_c = compute_frontier_planes_all_modes(records, pairs=pairs, **frontier_kwargs)
    frontier_time = perf_counter() - t3

    selection_kwargs = dict(
        feature_names=[f"f{i}" for i in range(X.shape[1])],
        max_k=8,
        min_improve=1e-3,
        min_region_size=25,
        min_abs_diff=0.02,
        min_rel_lift=0.05,
    )
    t4 = perf_counter()
    sel = prune_and_orient_planes_unified_globalmaj(
        res_c,
        X,
        y,
        **selection_kwargs,
    )
    selection_time = perf_counter() - t4

    finder_kwargs = dict(
        feature_names=[f"x{i}" for i in range(X.shape[1])],
        max_planes_in_rule=3,
        max_planes_per_pair=4,
        min_support=40,
        min_rel_gain_f1=0.05,
        min_lift_prec=1.40,
        consider_dims_up_to=X.shape[1],
        rng_seed=0,
    )
    t5 = perf_counter()
    valuable = find_low_dim_spaces(
        X,
        y,
        sel,
        **finder_kwargs,
    )
    finder_time = perf_counter() - t5

    ranked, best_by_class = nueva_funcion(valuable)
    output_dir = ROOT / "experiments_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "corner_compass_by_region.csv"
    _write_csv(metrics_path, ranked)

    best_by_class_path = output_dir / "corner_compass_best_by_class.csv"
    _write_csv(best_by_class_path, best_by_class)

    timings_path = output_dir / "corner_compass_stage_timings.csv"
    stage_rows = [
        dict(stage="dataset", seconds=f"{dataset_time:.6f}"),
        dict(stage="model_fit", seconds=f"{model_time:.6f}"),
        dict(stage="deldel_fit", seconds=f"{deldel_time:.6f}"),
        dict(stage="frontier", seconds=f"{frontier_time:.6f}"),
        dict(stage="selection", seconds=f"{selection_time:.6f}"),
        dict(stage="finder", seconds=f"{finder_time:.6f}"),
    ]
    _write_csv(timings_path, stage_rows)

    summary = dict(
        dataset=dict(name="make_corner_class_dataset", params=dataset_kwargs),
        stages=dict(
            dataset_s=dataset_time,
            model_fit_s=model_time,
            deldel_fit_s=deldel_time,
            frontier_s=frontier_time,
            selection_s=selection_time,
            finder_s=finder_time,
        ),
        selection=selection_kwargs,
        finder=finder_kwargs,
        compass=dict(
            description=(
                "compass_score = media geométrica de F1, lift_precision y min_lift_precision "
                "entre clases para la región; se maximiza por clase y de forma macro"
            ),
            best_by_class_csv=str(best_by_class_path),
            macro_geo_mean=_geo_mean([r.get("compass_score") for r in best_by_class]),
        ),
        outputs=dict(
            metrics_csv=str(metrics_path),
            best_by_class_csv=str(best_by_class_path),
            timings_csv=str(timings_path),
        ),
    )
    summary_path = output_dir / "corner_compass_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
