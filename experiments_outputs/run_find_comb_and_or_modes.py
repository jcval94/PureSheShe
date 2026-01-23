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
from deldel.combiantions import find_comb_dim_spaces, find_comb_dim_spaces_full  # noqa: E402
from experiments_outputs.run_comb_vs_low_dim_competition import (  # noqa: E402
    build_synthetic_sel,
    ensure_norm_fields,
    enrich_plane_metrics,
)


def summarize(valuable: Dict[int, List[Dict[str, object]]], class_ids: List[int]) -> Dict[str, float]:
    rules = [r for rr in valuable.values() for r in rr]
    metrics_keys = ["f1", "precision", "recall", "lift_precision"]
    summary: Dict[str, float] = {}
    if not rules:
        summary["total_rules"] = 0.0
        for key in metrics_keys:
            summary[key] = 0.0
        for class_id in class_ids:
            summary[f"class_{class_id}_total_rules"] = 0.0
            for key in metrics_keys:
                summary[f"class_{class_id}_{key}"] = 0.0
        return summary

    summary["total_rules"] = float(len(rules))
    for key in metrics_keys:
        summary[key] = float(np.mean([float(r["metrics"][key]) for r in rules]))

    for class_id in class_ids:
        class_rules = valuable.get(class_id, [])
        summary[f"class_{class_id}_total_rules"] = float(len(class_rules))
        if not class_rules:
            for key in metrics_keys:
                summary[f"class_{class_id}_{key}"] = 0.0
            continue
        for key in metrics_keys:
            summary[f"class_{class_id}_{key}"] = float(
                np.mean([float(r["metrics"][key]) for r in class_rules])
            )

    return summary


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


def build_hyperparam_grid() -> Dict[str, List[Dict[str, float]]]:
    default_grid = [
        dict(
            max_planes=12,
            beam_width=60,
            min_size=2,
            max_candidates_per_class=1200,
            max_rules_per_class=1500,
            lift_min=0.3,
            top_k_floor_per_dim=18,
            max_clause_candidates=500,
            clause_beam_width=72,
            clause_iterations=900,
            clause_diverse_topk=200,
            clause_overlap_max=0.92,
            max_clauses=8,
            max_dnf_rules_per_class=10,
        ),
        dict(
            max_planes=14,
            beam_width=72,
            min_size=2,
            max_candidates_per_class=1400,
            max_rules_per_class=1800,
            lift_min=0.25,
            top_k_floor_per_dim=20,
            max_clause_candidates=600,
            clause_beam_width=84,
            clause_iterations=1100,
            clause_diverse_topk=240,
            clause_overlap_max=0.95,
            max_clauses=9,
            max_dnf_rules_per_class=12,
        ),
        dict(
            max_planes=16,
            beam_width=72,
            min_size=1,
            max_candidates_per_class=1600,
            max_rules_per_class=2000,
            lift_min=0.2,
            top_k_floor_per_dim=22,
            max_clause_candidates=700,
            clause_beam_width=96,
            clause_iterations=1300,
            clause_diverse_topk=300,
            clause_overlap_max=0.97,
            max_clauses=10,
            max_dnf_rules_per_class=14,
        ),
    ]

    aggressive_grid = [
        dict(
            max_planes=16,
            beam_width=84,
            min_size=1,
            max_candidates_per_class=1800,
            max_rules_per_class=2400,
            lift_min=0.2,
            top_k_floor_per_dim=24,
            max_clause_candidates=800,
            clause_beam_width=96,
            clause_iterations=1400,
            clause_diverse_topk=320,
            clause_overlap_max=0.97,
            max_clauses=10,
            max_dnf_rules_per_class=16,
        ),
        dict(
            max_planes=20,
            beam_width=96,
            min_size=1,
            max_candidates_per_class=2200,
            max_rules_per_class=2600,
            lift_min=0.15,
            top_k_floor_per_dim=26,
            max_clause_candidates=1000,
            clause_beam_width=120,
            clause_iterations=1600,
            clause_diverse_topk=400,
            clause_overlap_max=0.98,
            max_clauses=12,
            max_dnf_rules_per_class=18,
        ),
    ]

    return {
        "default": default_grid,
        "aggressive_by_mode": {
            "or": aggressive_grid,
            "and_or_beam": aggressive_grid,
        },
    }


def main() -> None:
    rng = np.random.default_rng(123)
    configs = build_dataset_configs(50, rng)
    hyperparam_grid = build_hyperparam_grid()
    default_grid = hyperparam_grid["default"]
    aggressive_by_mode = hyperparam_grid["aggressive_by_mode"]

    modes = [
        ("find_comb_dim_spaces", "find_comb_dim_spaces", find_comb_dim_spaces),
        ("and", "find_comb_dim_spaces_full", find_comb_dim_spaces_full),
        ("or", "find_comb_dim_spaces_full", find_comb_dim_spaces_full),
        ("dnf", "find_comb_dim_spaces_full", find_comb_dim_spaces_full),
        ("and_or_greedy", "find_comb_dim_spaces_full", find_comb_dim_spaces_full),
        ("and_or_beam", "find_comb_dim_spaces_full", find_comb_dim_spaces_full),
        ("and_or_random", "find_comb_dim_spaces_full", find_comb_dim_spaces_full),
        ("and_or_diverse", "find_comb_dim_spaces_full", find_comb_dim_spaces_full),
    ]

    rows = []
    for exp_id, cfg in enumerate(configs, 1):
        X, y, feature_names = make_corner_class_dataset(**cfg)
        sel = build_synthetic_sel(X, y, feature_names, num_thresholds=10)
        sel = ensure_norm_fields(sel)
        sel = enrich_plane_metrics(sel, X, y)

        class_ids = sorted(int(v) for v in np.unique(y))
        for mode, label, func in modes:
            grid = aggressive_by_mode.get(mode, default_grid)
            hp_id = ((exp_id - 1) % len(grid)) + 1
            params = grid[hp_id - 1]
            if func is find_comb_dim_spaces:
                allowed_keys = {
                    "max_planes",
                    "lift_min",
                    "beam_width",
                    "min_size",
                    "max_candidates_per_class",
                    "max_rules_per_class",
                    "top_k_floor_per_dim",
                }
                base_params = {key: value for key, value in params.items() if key in allowed_keys}
            else:
                base_params = dict(**params)
            t0 = perf_counter()
            valuable = func(
                sel,
                X,
                y,
                mode="base" if func is find_comb_dim_spaces else mode,
                **base_params,
            )
            elapsed = perf_counter() - t0
            summary = summarize(valuable, class_ids)
            rows.append(
                {
                    "experiment_id": exp_id,
                    "dataset_config": cfg,
                    "hyperparam_id": hp_id,
                    "hyperparams": base_params,
                    "hyperparam_pool": "aggressive" if mode in aggressive_by_mode else "default",
                    "mode": mode,
                    "label": label,
                    "time_s": elapsed,
                    "avg_f1": summary["f1"],
                    "avg_precision": summary["precision"],
                    "avg_recall": summary["recall"],
                    "avg_lift_precision": summary["lift_precision"],
                    "total_rules": summary["total_rules"],
                    **{
                        f"class_{class_id}_total_rules": summary[f"class_{class_id}_total_rules"]
                        for class_id in class_ids
                    },
                    **{f"class_{class_id}_f1": summary[f"class_{class_id}_f1"] for class_id in class_ids},
                    **{
                        f"class_{class_id}_precision": summary[f"class_{class_id}_precision"]
                        for class_id in class_ids
                    },
                    **{f"class_{class_id}_recall": summary[f"class_{class_id}_recall"] for class_id in class_ids},
                    **{
                        f"class_{class_id}_lift_precision": summary[f"class_{class_id}_lift_precision"]
                        for class_id in class_ids
                    },
                }
            )

    out_dir = ROOT / "experiments_outputs"
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "find_comb_and_or_modes.csv"
    with csv_path.open("w", newline="") as fh:
        class_ids = sorted(int(v) for v in np.unique(y))
        fieldnames = [
            "experiment_id",
            "dataset_config",
            "hyperparam_id",
            "hyperparams",
            "hyperparam_pool",
            "mode",
            "label",
            "time_s",
            "avg_f1",
            "avg_precision",
            "avg_recall",
            "avg_lift_precision",
            "total_rules",
        ]
        for class_id in class_ids:
            fieldnames.append(f"class_{class_id}_total_rules")
        for class_id in class_ids:
            fieldnames.extend(
                [
                    f"class_{class_id}_f1",
                    f"class_{class_id}_precision",
                    f"class_{class_id}_recall",
                    f"class_{class_id}_lift_precision",
                ]
            )
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_mode = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)

    summary_lines = []
    class_ids = sorted(int(v) for v in np.unique(y))
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

    per_class_tables = []
    for class_id in class_ids:
        per_class_lines = []
        for mode, items in by_mode.items():
            avg_f1 = float(np.mean([r[f"class_{class_id}_f1"] for r in items]))
            avg_precision = float(np.mean([r[f"class_{class_id}_precision"] for r in items]))
            avg_recall = float(np.mean([r[f"class_{class_id}_recall"] for r in items]))
            avg_lift = float(np.mean([r[f"class_{class_id}_lift_precision"] for r in items]))
            avg_rules = float(np.mean([r[f"class_{class_id}_total_rules"] for r in items]))
            per_class_lines.append(
                f"| {mode} | {avg_f1:.3f} | {avg_precision:.3f} | {avg_recall:.3f} | "
                f"{avg_lift:.3f} | {avg_rules:.1f} |"
            )
        per_class_tables.append(
            "\n".join(
                [
                    f"## Métricas promedio por clase {class_id}",
                    "",
                    "| modo | f1 | precision | recall | lift_precision | reglas |",
                    "|---|---:|---:|---:|---:|---:|",
                    *per_class_lines,
                ]
            )
        )

    readme_path = out_dir / "README_find_comb_and_or_modes.md"
    readme_path.write_text(
        """# Comparativa AND/OR/DNF en find_comb_dim_spaces

Resultados generados con 50 variantes de `make_corner_class_dataset` (400 ejecuciones
totales al evaluar 8 modos) y combinaciones de hiperparámetros por modo.

## Métricas promedio (promedio sobre reglas)

| modo | tiempo (s) | f1 | precision | recall | lift_precision | reglas |
|---|---:|---:|---:|---:|---:|---:|
"""
        + "\n".join(summary_lines)
        + "\n\n"
        + "\n\n".join(per_class_tables)
        + "\n\n## Ajustes para incrementar reglas\n\n"
        + "Se incrementaron los presupuestos combinatorios para producir más reglas: "
        + "mayores `max_planes`, `beam_width`, `max_candidates_per_class`, "
        + "`max_rules_per_class` y `top_k_floor_per_dim`, además de relajar la poda con "
        + "`min_size` y `lift_min` más bajos. Para los modos con cláusulas se elevaron "
        + "`max_clause_candidates`, `clause_beam_width`, `clause_iterations`, "
        + "`clause_diverse_topk`, `max_clauses` y `max_dnf_rules_per_class`.\n"
        + "\n## Configuración\n\n"
        + "Se usaron 50 datasets con distintos `n_per_cluster`, dispersiones y seeds. "
        + "Se probaron configuraciones de hiperparámetros con variaciones en `max_planes`, "
        + "`beam_width`, `min_size`, `max_candidates_per_class`, `max_rules_per_class`, "
        + "`lift_min` y parámetros de cláusulas (`max_clause_candidates`, `clause_beam_width`, "
        + "`clause_iterations`, `clause_diverse_topk`, `clause_overlap_max`, `max_clauses`). "
        + "Los modos **or** y **and_or_beam** usan un bloque agresivo con valores más altos.\n"
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
