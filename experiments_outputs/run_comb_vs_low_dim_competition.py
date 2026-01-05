"""Comparar `find_comb_dim_spaces` vs `find_low_dim_spaces` siguiendo el protocolo.

El experimento replica el flujo de `docs/EXPERIMENT_PROTOCOL.md` para generar un
dataset de referencia, construir la selección de planos y ejecutar ambos
buscadores. Se guardan los reportes Top-3 por clase, los promedios de métricas
(`describe_regions_report(..., return_average_metrics=True)`) y los tiempos de
ejecución en un CSV de resultados.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Tuple

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
    find_low_dim_spaces,
    prune_and_orient_planes_unified_globalmaj,
)
from deldel.combiantions import find_comb_dim_spaces  # noqa: E402

Finder = Callable[[np.ndarray, np.ndarray, Dict[str, object]], Dict[int, List[Dict[str, object]]]]


def _flatten(valuable: Dict[int, List[Dict[str, object]]]) -> Iterable[Dict[str, object]]:
    for rules in valuable.values():
        for r in rules or []:
            yield r


def merge_valuable_sets(
    values: Iterable[Dict[int, List[Dict[str, object]]]]
) -> Dict[int, List[Dict[str, object]]]:
    merged: Dict[int, Dict[str, Dict[str, object]]] = {}
    for valuable in values:
        for k, rules in valuable.items():
            bucket = merged.setdefault(int(k), {})
            for r in rules or []:
                region_id = str(r.get("region_id"))
                if region_id not in bucket:
                    bucket[region_id] = r
    return {k: list(bucket.values()) for k, bucket in merged.items()}


def summarise_averages(valuable: Dict[int, List[Dict[str, object]]], dataset_size: int) -> Dict[str, float]:
    averages = describe_regions_report(
        valuable,
        top_per_class=3,
        dataset_size=dataset_size,
        return_average_metrics=True,
    )
    global_mean = averages.get("global_mean", {}) if isinstance(averages, dict) else {}
    return {
        "global_f1": float(global_mean.get("f1", 0.0) or 0.0),
        "global_lift_precision": float(global_mean.get("lift_precision", 0.0) or 0.0),
    }


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


def run_finder(
    name: str,
    finder: Finder,
    X: np.ndarray,
    y: np.ndarray,
    sel: Dict[str, object],
    feature_names: List[str],
    report_dir: Path,
    *,
    finder_kwargs: Dict[str, object],
) -> Tuple[Dict[int, List[Dict[str, object]]], float, Path, Dict[str, float]]:
    start = perf_counter()
    if name.startswith("find_comb_dim_spaces"):
        valuable = finder(sel, X, y, **finder_kwargs)
    else:
        valuable = finder(X, y, sel, feature_names=feature_names, **finder_kwargs)
    duration = perf_counter() - start

    report_text = describe_regions_report(
        valuable,
        top_per_class=3,
        dataset_size=int(X.shape[0]),
    )
    report_path = report_dir / f"{name}_top3.txt"
    report_path.write_text(report_text, encoding="utf-8")

    averages = summarise_averages(valuable, dataset_size=int(X.shape[0]))
    return valuable, duration, report_path, averages


def run_comb_multi(
    finder: Finder,
    X: np.ndarray,
    y: np.ndarray,
    sel: Dict[str, object],
    feature_names: List[str],
    report_dir: Path,
    *,
    finder_kwargs: Dict[str, object],
    metrics: List[str],
) -> Tuple[Dict[int, List[Dict[str, object]]], float, Path, Dict[str, float]]:
    valuables = []
    total_runtime = 0.0
    for metric in metrics:
        kwargs = dict(finder_kwargs)
        kwargs["metric"] = metric
        start = perf_counter()
        valuable = finder(sel, X, y, **kwargs)
        duration = perf_counter() - start
        valuables.append(valuable)
        total_runtime += duration

    merged = merge_valuable_sets(valuables)
    report_text = describe_regions_report(
        merged,
        top_per_class=3,
        dataset_size=int(X.shape[0]),
    )
    report_path = report_dir / "find_comb_dim_spaces_top3.txt"
    report_path.write_text(report_text, encoding="utf-8")
    averages = summarise_averages(merged, dataset_size=int(X.shape[0]))
    return merged, total_runtime, report_path, averages


def main() -> None:
    reports_dir = ROOT / "experiments_outputs" / "ab_test_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / "experiments_outputs" / "find_comb_vs_low_dim_competition.csv"

    X, y = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=6,
        class_sep=1.0,
        random_state=0,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    dataset_size = int(y.shape[0])

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
        print("Pruning sin planos; relajando umbrales para garantizar regiones válidas")
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
        print("No hubo planos tras prune; creando selección sintética eje-alineada")
        sel = build_synthetic_sel(X, y, feature_names)

    sel = ensure_norm_fields(sel)
    sel = enrich_plane_metrics(sel, X, y)
    sel = augment_with_synthetic_planes(sel, X, y, feature_names)
    sel = ensure_norm_fields(sel)
    sel = enrich_plane_metrics(sel, X, y)

    baseline_kwargs = dict(
        max_planes_in_rule=2,
        max_planes_per_pair=2,
        min_support=15,
        min_rel_gain_f1=0.05,
        min_lift_prec=1.20,
        consider_dims_up_to=6,
        rng_seed=0,
    )

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

    rows: List[Dict[str, object]] = []

    baseline_val, baseline_runtime, baseline_report, baseline_avg = run_finder(
        "find_low_dim_spaces",
        find_low_dim_spaces,
        X,
        y,
        sel,
        feature_names,
        reports_dir,
        finder_kwargs=baseline_kwargs,
    )

    comb_val, comb_runtime, comb_report, comb_avg = run_comb_multi(
        find_comb_dim_spaces,
        X,
        y,
        sel,
        feature_names,
        reports_dir,
        finder_kwargs=comb_kwargs,
        metrics=["precision", "recall", "f1", "lift_precision"],
    )

    rows.append(
        dict(
            variant="find_low_dim_spaces",
            runtime_s=baseline_runtime,
            global_f1=baseline_avg["global_f1"],
            global_lift_precision=baseline_avg["global_lift_precision"],
            total_regions=sum(1 for _ in _flatten(baseline_val)),
            report=str(baseline_report.relative_to(ROOT)),
        )
    )
    rows.append(
        dict(
            variant="find_comb_dim_spaces",
            runtime_s=comb_runtime,
            global_f1=comb_avg["global_f1"],
            global_lift_precision=comb_avg["global_lift_precision"],
            total_regions=sum(1 for _ in _flatten(comb_val)),
            report=str(comb_report.relative_to(ROOT)),
        )
    )

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "variant",
                "runtime_s",
                "global_f1",
                "global_lift_precision",
                "total_regions",
                "report",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resultados guardados en {csv_path}")
    for row in rows:
        print(
            f"{row['variant']}: runtime {row['runtime_s']:.3f}s | "
            f"F1={row['global_f1']:.3f} | lift={row['global_lift_precision']:.3f} | "
            f"regiones={row['total_regions']}"
        )


if __name__ == "__main__":
    main()
