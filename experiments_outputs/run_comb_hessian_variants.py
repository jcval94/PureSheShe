"""Comparar variantes Hessianas de `find_comb_dim_spaces` con la versión base.

Se generan reportes Top-5 por clase usando `describe_regions_report`, además de
un CSV con métricas (f1, precisión, recall) de los 5 mejores candidatos por clase
junto con los tiempos de ejecución.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
import sys
from time import perf_counter
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.datasets import make_classification
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
    describe_regions_report,
    find_comb_dim_spaces,
    find_comb_dim_spaces_hessian_filter,
    find_comb_dim_spaces_hessian_rank,
    find_comb_dim_spaces_hessian_seed,
    prune_and_orient_planes_unified_globalmaj,
)
from deldel.reporting_plotting import _group_by_class  # noqa: E402


def _flatten(valuable: Dict[int, List[Dict[str, object]]]) -> Iterable[Dict[str, object]]:
    for rules in valuable.values():
        for r in rules or []:
            yield r


def build_synthetic_sel(
    X: np.ndarray, y: np.ndarray, feature_names: List[str], *, num_thresholds: int = 15
) -> Dict[str, object]:
    classes = sorted(int(c) for c in np.unique(y))
    n_samples, n_features = X.shape
    qs = np.linspace(0.2, 0.8, num_thresholds)

    winning_planes: List[Dict[str, object]] = []
    by_pair: Dict[Tuple[int, int], Dict[str, object]] = {}

    for pair_idx, (a, b) in enumerate(combinations(classes, 2)):
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


def ensure_norm_fields(sel: Dict[str, object]) -> Dict[str, object]:
    planes = sel.get("winning_planes") or []
    regions_global = sel.get("regions_global") or {}
    per_plane = regions_global.get("per_plane") or []
    region_lookup = {r.get("region_id"): r for r in per_plane if isinstance(r, dict)}
    clean_planes: List[Dict[str, object]] = []
    for p in planes:
        if "n_norm" not in p and "n" not in p:
            region_id = p.get("region_id")
            region = region_lookup.get(region_id) if region_id else None
            geometry = region.get("geometry") if isinstance(region, dict) else None
            if geometry and "n" in geometry and "b" in geometry:
                p["n_norm"] = np.asarray(geometry["n"], float)
                p["b_norm"] = float(geometry["b"])
                p.setdefault("side", int(geometry.get("side", p.get("side", 1))))
        if "n_norm" not in p and "n" not in p:
            continue
        if "n_norm" not in p and "n" in p:
            p["n_norm"] = np.asarray(p["n"], float)
        if "b_norm" not in p and "b" in p:
            p["b_norm"] = float(p["b"])
        if "dims" not in p or not p.get("dims"):
            n_vec = p.get("n_norm") if "n_norm" in p else p.get("n", [])
            p["dims"] = tuple(i for i, v in enumerate(np.asarray(n_vec)) if v != 0)
        n_vec = p.get("n_norm")
        dims = p.get("dims") or ()
        if isinstance(n_vec, np.ndarray) and dims and n_vec.shape[0] != len(dims):
            p["n_norm"] = n_vec[list(dims)]
        clean_planes.append(p)
    sel["winning_planes"] = clean_planes
    return sel


def enrich_plane_metrics(
    sel: Dict[str, object], X: np.ndarray, y: np.ndarray
) -> Dict[str, object]:
    classes = sorted(int(c) for c in np.unique(y))
    planes = sel.get("winning_planes") or []
    for p in planes:
        metrics_by_class = p.get("metrics_by_class") or {}
        if metrics_by_class:
            continue
        n_norm = p.get("n_norm")
        n_raw = p.get("n")
        n_vec = np.asarray(n_norm if n_norm is not None else n_raw, float)
        if n_vec.size == 0:
            continue
        b_val = float(p.get("b_norm", p.get("b", 0.0)))
        dims = tuple(p.get("dims") or tuple(range(n_vec.shape[0])))
        if dims:
            expr = X[:, dims] @ n_vec[list(dims)] + b_val
        else:
            expr = X @ n_vec + b_val
        oriented = str(p.get("oriented_plane_id", ""))
        if oriented.endswith("≤"):
            mask = expr <= 1e-12
        elif oriented.endswith("≥"):
            mask = expr >= -1e-12
        else:
            side = int(p.get("side", 1))
            mask = expr <= 1e-12 if side < 0 else expr >= -1e-12
        size = int(mask.sum())
        metrics_by_class = {}
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
        p["metrics_by_class"] = metrics_by_class
    sel["winning_planes"] = planes
    return sel


def augment_with_synthetic_planes(
    sel: Dict[str, object],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    *,
    min_planes: int = 200,
) -> Dict[str, object]:
    planes = sel.get("winning_planes") or []
    if len(planes) >= min_planes:
        return sel
    synthetic = build_synthetic_sel(X, y, feature_names)
    synthetic_planes = synthetic.get("winning_planes") or []
    existing_ids = {p.get("oriented_plane_id") for p in planes}
    for sp in synthetic_planes:
        if sp.get("oriented_plane_id") not in existing_ids:
            planes.append(sp)
    sel["winning_planes"] = planes
    return sel


def run_variant(
    name: str,
    finder,
    X: np.ndarray,
    y: np.ndarray,
    sel: Dict[str, object],
    report_dir: Path,
    *,
    finder_kwargs: Dict[str, object],
) -> Tuple[Dict[int, List[Dict[str, object]]], float, Path]:
    start = perf_counter()
    valuable = finder(sel, X, y, **finder_kwargs)
    runtime = perf_counter() - start
    report_text = describe_regions_report(
        valuable,
        top_per_class=5,
        dataset_size=int(X.shape[0]),
    )
    report_path = report_dir / f"{name}_top5.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return valuable, runtime, report_path


def write_csv(
    rows: List[Dict[str, object]],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "variant",
                "runtime_s",
                "class_id",
                "rank",
                "region_id",
                "precision",
                "recall",
                "f1",
                "report",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    summary_rows: List[Dict[str, object]],
    csv_path: Path,
    readme_path: Path,
) -> None:
    lines = [
        "# Comparativa Hessiana para find_comb_dim_spaces",
        "",
        "Resumen de tiempos y métricas (media de top-5 por clase).",
        "",
        "| Variante | Runtime (s) | F1 mean | Prec mean | Recall mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {variant} | {runtime_s:.3f} | {f1:.3f} | {precision:.3f} | {recall:.3f} |".format(
                variant=row["variant"],
                runtime_s=row["runtime_s"],
                f1=row["f1_mean"],
                precision=row["precision_mean"],
                recall=row["recall_mean"],
            )
        )
    lines.extend(
        [
            "",
            f"CSV generado: `{csv_path.relative_to(ROOT)}`",
        ]
    )
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    reports_dir = ROOT / "experiments_outputs" / "ab_test_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / "experiments_outputs" / "find_comb_hessian_variants.csv"
    readme_path = ROOT / "experiments_outputs" / "README_find_comb_hessian_variants.md"

    X, y = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=6,
        class_sep=1.0,
        random_state=0,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]

    model = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)

    deldel_cfg = DelDelConfig(segments_target=180, random_state=0)
    records = DelDel(deldel_cfg, ChangePointConfig(enabled=False)).fit(X, model).records_

    frontiers = compute_frontier_planes_all_modes(
        records,
        mode="C",
        min_cluster_size=12,
        max_models_per_round=6,
        seed=0,
    )

    sel = prune_and_orient_planes_unified_globalmaj(
        frontiers,
        X,
        y,
        feature_names=feature_names,
        max_k=8,
        min_improve=1e-3,
        min_region_size=25,
        min_abs_diff=0.02,
        min_rel_lift=0.05,
    )

    if not sel.get("winning_planes"):
        sel = prune_and_orient_planes_unified_globalmaj(
            frontiers,
            X,
            y,
            feature_names=feature_names,
            max_k=10,
            min_improve=0.0,
            min_region_size=12,
            min_abs_diff=0.0,
            min_rel_lift=0.0,
        )

    if not sel.get("winning_planes"):
        sel = build_synthetic_sel(X, y, feature_names)

    sel = ensure_norm_fields(sel)
    sel = enrich_plane_metrics(sel, X, y)
    sel = augment_with_synthetic_planes(sel, X, y, feature_names)
    sel = ensure_norm_fields(sel)
    sel = enrich_plane_metrics(sel, X, y)

    comb_kwargs = dict(
        max_planes=9,
        metric="precision",
        lift_min=0.2,
        beam_width=256,
        min_size=3,
        max_candidates_per_class=4000,
        max_rules_per_class=3000,
        top_k_floor_per_dim=200,
        projection_ref="model_space",
    )

    variants = [
        ("find_comb_dim_spaces", find_comb_dim_spaces),
        ("find_comb_dim_spaces_hessian_seed", find_comb_dim_spaces_hessian_seed),
        ("find_comb_dim_spaces_hessian_rank", find_comb_dim_spaces_hessian_rank),
        ("find_comb_dim_spaces_hessian_filter", find_comb_dim_spaces_hessian_filter),
    ]

    rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for name, finder in variants:
        valuable, runtime, report_path = run_variant(
            name,
            finder,
            X,
            y,
            sel,
            reports_dir,
            finder_kwargs=comb_kwargs,
        )
        grouped = _group_by_class(valuable)
        metrics_collect: List[Tuple[float, float, float]] = []
        for class_id, regions in grouped.items():
            top_regions = regions[:5]
            for rank, region in enumerate(top_regions, 1):
                metrics = region.get("metrics", {}) or {}
                rows.append(
                    {
                        "variant": name,
                        "runtime_s": runtime,
                        "class_id": class_id,
                        "rank": rank,
                        "region_id": region.get("region_id"),
                        "precision": float(metrics.get("precision", 0.0) or 0.0),
                        "recall": float(metrics.get("recall", 0.0) or 0.0),
                        "f1": float(metrics.get("f1", 0.0) or 0.0),
                        "report": str(report_path.relative_to(ROOT)),
                    }
                )
                metrics_collect.append(
                    (
                        float(metrics.get("precision", 0.0) or 0.0),
                        float(metrics.get("recall", 0.0) or 0.0),
                        float(metrics.get("f1", 0.0) or 0.0),
                    )
                )
        if metrics_collect:
            precision_mean = float(np.mean([m[0] for m in metrics_collect]))
            recall_mean = float(np.mean([m[1] for m in metrics_collect]))
            f1_mean = float(np.mean([m[2] for m in metrics_collect]))
        else:
            precision_mean = 0.0
            recall_mean = 0.0
            f1_mean = 0.0
        summary_rows.append(
            {
                "variant": name,
                "runtime_s": runtime,
                "precision_mean": precision_mean,
                "recall_mean": recall_mean,
                "f1_mean": f1_mean,
            }
        )

    write_csv(rows, csv_path)
    write_readme(summary_rows, csv_path, readme_path)

    print(f"Resultados guardados en {csv_path}")
    for row in summary_rows:
        print(
            f"{row['variant']}: runtime {row['runtime_s']:.3f}s | "
            f"F1={row['f1_mean']:.3f} | Prec={row['precision_mean']:.3f} | "
            f"Rec={row['recall_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
