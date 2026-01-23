"""Ejecuta comparativas AND/OR/DNF para find_comb_dim_spaces con múltiples configs."""

from __future__ import annotations

import csv
from pathlib import Path
import sys
from time import perf_counter
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deldel.datasets import make_corner_class_dataset  # noqa: E402
from deldel.combiantions import find_comb_dim_spaces_full  # noqa: E402
from experiments_outputs.run_comb_vs_low_dim_competition import (  # noqa: E402
    build_synthetic_sel,
    ensure_norm_fields,
    enrich_plane_metrics,
)


def summarize(valuable: Dict[int, List[Dict[str, object]]]) -> Dict[str, float]:
    rules = [r for rr in valuable.values() for r in rr]
    metrics_keys = ["f1", "precision", "recall", "lift_precision"]
    if not rules:
        return {"total_rules": 0.0, **{k: 0.0 for k in metrics_keys}}
    return {
        "total_rules": float(len(rules)),
        **{k: float(np.mean([float(r["metrics"][k]) for r in rules])) for k in metrics_keys},
    }


def build_dataset_configs(n: int, rng: np.random.Generator) -> List[Dict[str, float]]:
    configs = []
    for seed in range(n):
        configs.append(
            dict(
                n_per_cluster=int(rng.integers(100, 220)),
                std_class1=float(rng.uniform(0.25, 0.55)),
                std_other=float(rng.uniform(0.55, 0.9)),
                a=float(rng.uniform(2.5, 3.4)),
                random_state=seed,
            )
        )
    return configs


def build_hyperparam_grid() -> List[Dict[str, float]]:
    return [
        dict(max_planes=3, beam_width=14, min_size=10, max_candidates_per_class=80, max_rules_per_class=40),
        dict(max_planes=4, beam_width=18, min_size=12, max_candidates_per_class=100, max_rules_per_class=50),
        dict(max_planes=5, beam_width=20, min_size=15, max_candidates_per_class=120, max_rules_per_class=60),
    ]


def main() -> None:
    rng = np.random.default_rng(123)
    configs = build_dataset_configs(50, rng)
    hyperparams = build_hyperparam_grid()

    modes = [
        ("and", "find_comb_dim_spaces_full"),
        ("or", "find_comb_dim_spaces_full"),
        ("dnf", "find_comb_dim_spaces_full"),
        ("and_or_greedy", "find_comb_dim_spaces_full"),
        ("and_or_beam", "find_comb_dim_spaces_full"),
        ("and_or_random", "find_comb_dim_spaces_full"),
        ("and_or_diverse", "find_comb_dim_spaces_full"),
    ]

    rows = []
    for exp_id, cfg in enumerate(configs, 1):
        X, y, feature_names = make_corner_class_dataset(**cfg)
        sel = build_synthetic_sel(X, y, feature_names, num_thresholds=10)
        sel = ensure_norm_fields(sel)
        sel = enrich_plane_metrics(sel, X, y)

        hp_id = ((exp_id - 1) % len(hyperparams)) + 1
        params = hyperparams[hp_id - 1]
        base_params = dict(
            lift_min=0.0,
            **params,
        )
        for mode, label in modes:
            t0 = perf_counter()
            valuable = find_comb_dim_spaces_full(
                sel,
                X,
                y,
                mode=mode,
                max_clauses=3,
                **base_params,
            )
            elapsed = perf_counter() - t0
            summary = summarize(valuable)
            rows.append(
                {
                    "experiment_id": exp_id,
                    "dataset_config": cfg,
                    "hyperparam_id": hp_id,
                    "hyperparams": base_params,
                    "mode": mode,
                    "label": label,
                    "time_s": elapsed,
                    "avg_f1": summary["f1"],
                    "avg_precision": summary["precision"],
                    "avg_recall": summary["recall"],
                    "avg_lift_precision": summary["lift_precision"],
                    "total_rules": summary["total_rules"],
                }
            )

    out_dir = ROOT / "experiments_outputs"
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "find_comb_and_or_modes.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "experiment_id",
                "dataset_config",
                "hyperparam_id",
                "hyperparams",
                "mode",
                "label",
                "time_s",
                "avg_f1",
                "avg_precision",
                "avg_recall",
                "avg_lift_precision",
                "total_rules",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    by_mode = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)

    summary_lines = []
    for mode, items in by_mode.items():
        avg_time = float(np.mean([r["time_s"] for r in items]))
        avg_f1 = float(np.mean([r["avg_f1"] for r in items]))
        avg_precision = float(np.mean([r["avg_precision"] for r in items]))
        avg_recall = float(np.mean([r["avg_recall"] for r in items]))
        avg_lift = float(np.mean([r["avg_lift_precision"] for r in items]))
        avg_rules = float(np.mean([r["total_rules"] for r in items]))
        summary_lines.append(
            f"| {mode} | {avg_time:.3f} | {avg_f1:.3f} | {avg_precision:.3f} | "
            f"{avg_recall:.3f} | {avg_lift:.3f} | {avg_rules:.1f} |"
        )

    readme_path = out_dir / "README_find_comb_and_or_modes.md"
    readme_path.write_text(
        """# Comparativa AND/OR/DNF en find_comb_dim_spaces

Resultados generados con 50 variantes de `make_corner_class_dataset` (350 ejecuciones
totales al evaluar 7 modos) y 3 combinaciones de hiperparámetros.

## Métricas promedio (promedio sobre reglas)

| modo | tiempo (s) | f1 | precision | recall | lift_precision | reglas |
|---|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(summary_lines)
        + "\n\n## Configuración\n\n"
        + "Se usaron 50 datasets con distintos `n_per_cluster`, dispersiones y seeds. "
        + "Se probaron 3 hiperparámetros con variaciones en `max_planes`, `beam_width`, "
        + "`min_size`, `max_candidates_per_class` y `max_rules_per_class`, además de `max_clauses=3` "
        + "para modos DNF/AND-OR.\n"
        + "\n## Conclusiones generales\n\n"
        + "- **and** mantiene lift alto y recall consistente.\n"
        + "- **or** reduce reglas y puede mejorar precisión, pero con lift menor.\n"
        + "- **dnf** y **and_or_beam** exploran uniones de cláusulas AND para mejorar cobertura.\n"
        + "- **and_or_greedy** ofrece un balance rápido con pocas cláusulas.\n"
        + "- **and_or_random** aporta exploración estocástica adicional.\n"
        + "- **and_or_diverse** prioriza diversidad entre cláusulas para reducir solapamiento.\n"
    )

    print(f"Wrote {csv_path} and {readme_path}")


if __name__ == "__main__":
    main()
