"""Hiperparametrización de find_comb_dim_spaces_full en el corner dataset.

Evalúa múltiples combinaciones y reporta cuáles generan más candidatos
(según len(valuable[4]) y el total de reglas).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter
from typing import Dict, Iterable, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deldel import find_comb_dim_spaces_full  # noqa: E402
from deldel.datasets import make_corner_class_dataset  # noqa: E402
from deldel.experiments import _build_demo_selection  # noqa: E402


@dataclass
class RunResult:
    mode: str
    max_planes: int
    beam_width: int
    max_candidates_per_class: int
    max_rules_per_class: int
    max_clauses: int
    clause_beam_width: int
    clause_iterations: int
    clause_diverse_topk: int
    clause_overlap_max: float
    elapsed_s: float
    rules_class4: int
    total_rules: int


def _total_rules(valuable: Dict[int, List[Dict[str, object]]]) -> int:
    return sum(len(v) for v in valuable.values())


def _compute_metrics_by_class(mask: np.ndarray, y: np.ndarray) -> Dict[int, Dict[str, float]]:
    classes = sorted(int(c) for c in np.unique(y))
    size = int(mask.sum())
    metrics_by_class: Dict[int, Dict[str, float]] = {}
    for c in classes:
        c_mask = y == c
        c_in = int(np.logical_and(mask, c_mask).sum())
        total_c = int(c_mask.sum())
        prec = (c_in / size) if size > 0 else 0.0
        rec = (c_in / total_c) if total_c > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        baseline = (total_c / len(y)) if len(y) > 0 else 0.0
        lift = (prec / baseline) if baseline > 0 else 0.0
        metrics_by_class[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "lift": lift,
            "lift_precision": lift,
            "region_frac": (size / len(y)) if len(y) > 0 else 0.0,
        }
    return metrics_by_class


def _enrich_selection_metrics(
    sel: Dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, object]:
    def _attach(entry: Dict[str, object]) -> None:
        if entry.get("metrics_by_class"):
            return
        n = np.asarray(entry.get("n") or entry.get("n_norm"), dtype=float).reshape(-1)
        if n.size == 0:
            return
        b = float(entry.get("b") if entry.get("b") is not None else entry.get("b_norm", 0.0))
        side = int(entry.get("side", 1))
        dims = entry.get("dims")
        if dims:
            dims_idx = tuple(int(dd) for dd in dims)
            n_vec = n
            if n_vec.size != len(dims_idx):
                return
            expr = X[:, dims_idx] @ n_vec + b
        else:
            if n.size != X.shape[1]:
                return
            expr = X @ n + b
        mask = expr <= 1e-12 if side >= 0 else expr >= -1e-12
        entry["metrics_by_class"] = _compute_metrics_by_class(mask, y)

    by_pair = sel.get("by_pair_augmented", {}) or {}
    for payload in by_pair.values():
        for entry in payload.get("winning_planes", []) or []:
            _attach(entry)

    regions = sel.get("regions_global", {}).get("per_plane", []) or []
    for entry in regions:
        _attach(entry)

    for entry in sel.get("winning_planes", []) or []:
        _attach(entry)

    return sel


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
    max_clauses: int,
    clause_beam_width: int,
    clause_iterations: int,
    clause_diverse_topk: int,
    clause_overlap_max: float,
) -> RunResult:
    start = perf_counter()
    valuable = find_comb_dim_spaces_full(
        sel,
        X,
        y,
        mode=mode,
        max_planes=max_planes,
        metric="f1",
        beam_width=beam_width,
        max_candidates_per_class=max_candidates_per_class,
        max_rules_per_class=max_rules_per_class,
        max_clauses=max_clauses,
        clause_beam_width=clause_beam_width,
        clause_iterations=clause_iterations,
        clause_diverse_topk=clause_diverse_topk,
        clause_overlap_max=clause_overlap_max,
    )
    elapsed = perf_counter() - start
    rules_class4 = len(valuable.get(4, []))
    total_rules = _total_rules(valuable)
    return RunResult(
        mode=mode,
        max_planes=max_planes,
        beam_width=beam_width,
        max_candidates_per_class=max_candidates_per_class,
        max_rules_per_class=max_rules_per_class,
        max_clauses=max_clauses,
        clause_beam_width=clause_beam_width,
        clause_iterations=clause_iterations,
        clause_diverse_topk=clause_diverse_topk,
        clause_overlap_max=clause_overlap_max,
        elapsed_s=elapsed,
        rules_class4=rules_class4,
        total_rules=total_rules,
    )


def main() -> None:
    X, y, feature_names = make_corner_class_dataset(
        n_per_cluster=150,
        std_class1=0.4,
        std_other=0.7,
        a=3.0,
        random_state=42,
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)
    print(f"Accuracy en el dataset sintético: {clf.score(X, y):.3f}")

    sel = _build_demo_selection(X, y, feature_names)
    sel = _enrich_selection_metrics(sel, X, y)

    model_types = ["base", "and_or_beam", "and_or_diverse"]

    max_planes_opts = [10, 12, 14, 16]
    beam_width_opts = [24, 36, 48]
    max_candidates_opts = [300, 600]
    max_rules_opts = [480, 720]

    max_clauses_opts = [2, 3, 4]
    clause_beam_width_opts = [12, 16]
    clause_iterations_opts = [120, 180]
    clause_diverse_topk_opts = [30, 40]
    clause_overlap_max_opts = [0.6, 0.7, 0.8]

    configs = [
        dict(
            mode=mode,
            max_planes=max_planes,
            beam_width=beam_width,
            max_candidates_per_class=max_candidates_per_class,
            max_rules_per_class=max_rules_per_class,
            max_clauses=max_clauses,
            clause_beam_width=clause_beam_width,
            clause_iterations=clause_iterations,
            clause_diverse_topk=clause_diverse_topk,
            clause_overlap_max=clause_overlap_max,
        )
        for mode in model_types
        for max_planes in max_planes_opts
        for beam_width in beam_width_opts
        for max_candidates_per_class in max_candidates_opts
        for max_rules_per_class in max_rules_opts
        for max_clauses in max_clauses_opts
        for clause_beam_width in clause_beam_width_opts
        for clause_iterations in clause_iterations_opts
        for clause_diverse_topk in clause_diverse_topk_opts
        for clause_overlap_max in clause_overlap_max_opts
    ]

    results: List[RunResult] = []
    for cfg in configs:
        results.append(run_config(sel, X, y, **cfg))

    results.sort(key=lambda r: (r.rules_class4, r.total_rules), reverse=True)

    csv_path = ROOT / "experiments_outputs" / "find_comb_dim_spaces_full_hyperparam.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mode",
                "max_planes",
                "beam_width",
                "max_candidates_per_class",
                "max_rules_per_class",
                "max_clauses",
                "clause_beam_width",
                "clause_iterations",
                "clause_diverse_topk",
                "clause_overlap_max",
                "elapsed_s",
                "rules_class4",
                "total_rules",
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
                    r.max_clauses,
                    r.clause_beam_width,
                    r.clause_iterations,
                    r.clause_diverse_topk,
                    f"{r.clause_overlap_max:.2f}",
                    f"{r.elapsed_s:.4f}",
                    r.rules_class4,
                    r.total_rules,
                ]
            )

    md_path = ROOT / "experiments_outputs" / "README_find_comb_dim_spaces_full_hyperparam.md"
    top_n = 20
    best = results[0]
    mode_summary: Dict[str, Dict[str, float]] = {}
    for r in results:
        stats = mode_summary.setdefault(
            r.mode,
            {
                "count": 0,
                "best_rules_class4": 0,
                "avg_rules_class4": 0.0,
                "best_total_rules": 0,
                "avg_total_rules": 0.0,
            },
        )
        stats["count"] += 1
        stats["best_rules_class4"] = max(stats["best_rules_class4"], r.rules_class4)
        stats["best_total_rules"] = max(stats["best_total_rules"], r.total_rules)
        stats["avg_rules_class4"] += r.rules_class4
        stats["avg_total_rules"] += r.total_rules

    for stats in mode_summary.values():
        count = max(int(stats["count"]), 1)
        stats["avg_rules_class4"] /= count
        stats["avg_total_rules"] /= count

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Hiperparametrización find_comb_dim_spaces_full\n\n")
        f.write("Dataset: make_corner_class_dataset (4D, 3 clases).\n\n")
        f.write("## Conclusiones\n\n")
        f.write(
            "- Configuración líder: "
            f"mode={best.mode}, max_planes={best.max_planes}, beam_width={best.beam_width}, "
            f"max_candidates_per_class={best.max_candidates_per_class}, "
            f"max_rules_per_class={best.max_rules_per_class}, max_clauses={best.max_clauses}, "
            f"clause_beam_width={best.clause_beam_width}, clause_iterations={best.clause_iterations}, "
            f"clause_diverse_topk={best.clause_diverse_topk}, clause_overlap_max={best.clause_overlap_max:.2f}.\n"
        )
        f.write(
            f"- Máximo len(valuable[4]) observado: {best.rules_class4} (total reglas {best.total_rules}).\n"
        )
        f.write(
            "- Tabla completa disponible en find_comb_dim_spaces_full_hyperparam.csv.\n\n"
        )

        f.write("## Resumen por modo\n\n")
        f.write("| mode | runs | best rules_class4 | avg rules_class4 | best total_rules | avg total_rules |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for mode, stats in sorted(mode_summary.items()):
            f.write(
                "| {mode} | {count} | {best_rules_class4} | {avg_rules_class4:.2f} | "
                "{best_total_rules} | {avg_total_rules:.2f} |\n".format(
                    mode=mode,
                    count=int(stats["count"]),
                    best_rules_class4=int(stats["best_rules_class4"]),
                    avg_rules_class4=float(stats["avg_rules_class4"]),
                    best_total_rules=int(stats["best_total_rules"]),
                    avg_total_rules=float(stats["avg_total_rules"]),
                )
            )

        f.write("\n## Top {top_n} configuraciones\n\n".format(top_n=top_n))
        f.write(
            "| rank | mode | max_planes | beam_width | max_candidates_per_class | "
            "max_rules_per_class | max_clauses | clause_beam_width | "
            "clause_iterations | clause_diverse_topk | clause_overlap_max | "
            "elapsed_s | rules_class4 | total_rules |\n"
        )
        f.write(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        for i, r in enumerate(results[:top_n], start=1):
            f.write(
                "| {rank} | {mode} | {max_planes} | {beam_width} | "
                "{max_candidates_per_class} | {max_rules_per_class} | {max_clauses} | "
                "{clause_beam_width} | {clause_iterations} | {clause_diverse_topk} | "
                "{clause_overlap_max:.2f} | {elapsed_s:.3f} | {rules_class4} | {total_rules} |\n".format(
                    rank=i,
                    mode=r.mode,
                    max_planes=r.max_planes,
                    beam_width=r.beam_width,
                    max_candidates_per_class=r.max_candidates_per_class,
                    max_rules_per_class=r.max_rules_per_class,
                    max_clauses=r.max_clauses,
                    clause_beam_width=r.clause_beam_width,
                    clause_iterations=r.clause_iterations,
                    clause_diverse_topk=r.clause_diverse_topk,
                    clause_overlap_max=r.clause_overlap_max,
                    elapsed_s=r.elapsed_s,
                    rules_class4=r.rules_class4,
                    total_rules=r.total_rules,
                )
            )


if __name__ == "__main__":
    main()
