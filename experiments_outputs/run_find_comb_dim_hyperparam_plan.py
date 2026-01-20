"""Ejecuta el plan agresivo de hiperparametrización para find_comb_dim_spaces.

Genera un CSV con combos/seg y un resumen con recomendaciones (Parte 2).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deldel import find_comb_dim_spaces  # noqa: E402


@dataclass
class RunResult:
    mode: str
    max_planes: int
    beam_width: int
    max_candidates_per_class: int
    max_rules_per_class: int
    min_size: int
    lift_min: float
    elapsed_s: float
    total_rules: int
    combos_per_sec: float
    classes: int
    dims: int


def _flatten(valuable: Dict[int, List[Dict[str, object]]]) -> Iterable[Dict[str, object]]:
    for rules in valuable.values():
        for r in rules or []:
            yield r


def build_synthetic_sel(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    *,
    num_thresholds: int = 8,
) -> Dict[str, object]:
    classes = sorted(int(c) for c in np.unique(y))
    n_samples, n_features = X.shape
    qs = np.linspace(0.2, 0.8, num_thresholds)

    winning_planes: List[Dict[str, object]] = []
    by_pair: Dict[Tuple[int, int], Dict[str, object]] = {}

    for pair_idx, (a, b) in enumerate(_pairs(classes)):
        planes_pair: List[Dict[str, object]] = []
        for j in range(n_features):
            col = X[:, j]
            for q in qs:
                t = float(np.quantile(col, q))
                n_vec = np.zeros(n_features, dtype=float)
                n_vec[j] = 1.0
                b_val = -t

                metrics_by_class: Dict[int, Dict[str, float]] = {}
                mask = col <= t
                size = int(mask.sum())
                for c in classes:
                    c_mask = y == c
                    c_in = int(np.logical_and(mask, c_mask).sum())
                    total_c = int(c_mask.sum())
                    prec = (c_in / size) if size > 0 else 0.0
                    rec = (c_in / total_c) if total_c > 0 else 0.0
                    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                    baseline = (total_c / n_samples) if n_samples > 0 else 0.0
                    lift = (prec / baseline) if baseline > 0 else 0.0
                    metrics_by_class[c] = {
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "lift": lift,
                        "lift_precision": lift,
                        "region_frac": (size / n_samples) if n_samples > 0 else 0.0,
                    }

                plane_id = f"syn{pair_idx:02d}_f{j}_q{int(q * 100):02d}"
                plane = dict(
                    plane_id=plane_id,
                    oriented_plane_id=f"{plane_id}:≤",
                    origin_pair=(int(a), int(b)),
                    side=+1,
                    dims=(j,),
                    n=n_vec,
                    b=b_val,
                    n_norm=n_vec,
                    b_norm=b_val,
                    inequality={"general": f"{feature_names[j]} ≤ {t:.3f}"},
                    family_id="synthetic",
                    metrics_by_class=metrics_by_class,
                )
                planes_pair.append(plane)
        by_pair[(int(a), int(b))] = {"winning_planes": planes_pair}
        winning_planes.extend(planes_pair)

    return {
        "by_pair_augmented": by_pair,
        "winning_planes": winning_planes,
        "regions_global": {"per_plane": [], "per_class": {c: [] for c in classes}},
    }


def _pairs(classes: List[int]) -> Iterable[Tuple[int, int]]:
    for i, a in enumerate(classes):
        for b in classes[i + 1 :]:
            yield a, b


def run_config(
    sel: Dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    *,
    mode: str,
    max_planes: int,
    beam_width: int,
    max_candidates_per_class: int,
    max_rules_per_class: int,
    min_size: int,
    lift_min: float,
) -> RunResult:
    start = perf_counter()
    valuable = find_comb_dim_spaces(
        sel,
        X,
        y,
        mode=mode,
        metric="f1",
        max_planes=max_planes,
        beam_width=beam_width,
        max_candidates_per_class=max_candidates_per_class,
        max_rules_per_class=max_rules_per_class,
        min_size=min_size,
        lift_min=lift_min,
    )
    elapsed = perf_counter() - start
    total_rules = sum(len(v) for v in valuable.values())
    combos_per_sec = total_rules / elapsed if elapsed > 0 else 0.0
    return RunResult(
        mode=mode,
        max_planes=max_planes,
        beam_width=beam_width,
        max_candidates_per_class=max_candidates_per_class,
        max_rules_per_class=max_rules_per_class,
        min_size=min_size,
        lift_min=lift_min,
        elapsed_s=elapsed,
        total_rules=total_rules,
        combos_per_sec=combos_per_sec,
        classes=len(np.unique(y)),
        dims=X.shape[1],
    )


def recommend_tweaks(best: RunResult) -> List[str]:
    tips: List[str] = []
    saturation_ratio = best.total_rules / max(best.max_rules_per_class, 1)
    if best.elapsed_s > 0 and best.combos_per_sec < 50:
        tips.append("Tiempo alto: baja max_planes o beam_width para reducir costo.")
    if saturation_ratio < 0.5:
        tips.append("Pocas reglas: baja lift_min y min_size, o sube beam_width.")
    if saturation_ratio > 0.9:
        tips.append("Salida saturada: sube max_rules_per_class o filtra con lift_min.")
    if best.mode == "hessian_rank" and best.total_rules < 100:
        tips.append("Prueba mode=base para explorar sin sesgo Hessiano.")
    if not tips:
        tips.append("Config estable: prueba subir max_planes o beam_width para más combinaciones.")
    return tips


def main() -> None:
    X, y = make_classification(
        n_samples=800,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=3,
        random_state=17,
    )
    X = X.astype(float)
    y = y.astype(int)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    sel = build_synthetic_sel(X, y, feature_names, num_thresholds=8)

    modes = ["hessian_rank", "base"]
    max_planes_opts = [10, 12]
    beam_width_opts = [36, 48]
    max_candidates_opts = [600, 900]
    max_rules_opts = [720]
    min_size_opts = [2]
    lift_min_opts = [0.5]

    configs = [
        dict(
            mode=mode,
            max_planes=max_planes,
            beam_width=beam_width,
            max_candidates_per_class=max_candidates_per_class,
            max_rules_per_class=max_rules_per_class,
            min_size=min_size,
            lift_min=lift_min,
        )
        for mode in modes
        for max_planes in max_planes_opts
        for beam_width in beam_width_opts
        for max_candidates_per_class in max_candidates_opts
        for max_rules_per_class in max_rules_opts
        for min_size in min_size_opts
        for lift_min in lift_min_opts
    ]

    results: List[RunResult] = []
    for cfg in configs:
        results.append(run_config(sel, X, y, **cfg))

    results.sort(key=lambda r: r.combos_per_sec, reverse=True)
    best = results[0]

    csv_path = ROOT / "experiments_outputs" / "find_comb_dim_hyperparam_plan.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "max_planes",
                "beam_width",
                "max_candidates_per_class",
                "max_rules_per_class",
                "min_size",
                "lift_min",
                "elapsed_s",
                "total_rules",
                "combos_per_sec",
                "classes",
                "dims",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.mode,
                    r.max_planes,
                    r.beam_width,
                    r.max_candidates_per_class,
                    r.max_rules_per_class,
                    r.min_size,
                    f"{r.lift_min:.2f}",
                    f"{r.elapsed_s:.4f}",
                    r.total_rules,
                    f"{r.combos_per_sec:.2f}",
                    r.classes,
                    r.dims,
                ]
            )

    md_path = ROOT / "experiments_outputs" / "README_find_comb_dim_hyperparam_plan.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Resultados del plan de hiperparametrización\n\n")
        f.write("## Top configs (ordenado por combos/seg)\n\n")
        f.write("| rank | mode | max_planes | beam_width | max_candidates_per_class | max_rules_per_class | min_size | lift_min | elapsed_s | total_rules | combos_per_sec |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for i, r in enumerate(results, start=1):
            f.write(
                "| {rank} | {mode} | {max_planes} | {beam_width} | {max_candidates_per_class} "
                "| {max_rules_per_class} | {min_size} | {lift_min:.2f} | {elapsed_s:.3f} "
                "| {total_rules} | {combos_per_sec:.2f} |\n".format(
                    rank=i,
                    mode=r.mode,
                    max_planes=r.max_planes,
                    beam_width=r.beam_width,
                    max_candidates_per_class=r.max_candidates_per_class,
                    max_rules_per_class=r.max_rules_per_class,
                    min_size=r.min_size,
                    lift_min=r.lift_min,
                    elapsed_s=r.elapsed_s,
                    total_rules=r.total_rules,
                    combos_per_sec=r.combos_per_sec,
                )
            )

        f.write("\n## Recomendaciones (Parte 2)\n\n")
        f.write(f"Mejor config: **mode={best.mode}**, max_planes={best.max_planes}, beam_width={best.beam_width}.\n\n")
        for tip in recommend_tweaks(best):
            f.write(f"- {tip}\n")


if __name__ == "__main__":
    main()
