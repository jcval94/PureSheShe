"""Run AB testing between different find_low_dim_spaces strategies.

The script builds a synthetic dataset, executes the baseline finder and three
custom strategies, measures runtimes, captures their top-2-per-class reports
(using :func:`describe_regions_report`) and stores a CSV comparison table.
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deldel import (
    describe_regions_report,
    find_low_dim_spaces,
    find_low_dim_spaces_deterministic,
    find_low_dim_spaces_precision_boost,
    find_low_dim_spaces_support_first,
    make_corner_class_dataset,
)
from deldel.experiments import _build_demo_selection

VariantFunc = Callable[[np.ndarray, np.ndarray, Dict[str, object]], Dict[int, List[Dict[str, object]]]]


def _flatten(valuable: Dict[int, List[Dict[str, object]]]) -> Iterable[Dict[str, object]]:
    for rules in valuable.values():
        for r in rules or []:
            yield r


def summarise_regions(valuable: Dict[int, List[Dict[str, object]]]) -> Dict[str, float]:
    flattened = list(_flatten(valuable))
    if not flattened:
        return dict(total_regions=0, best_f1=0.0, best_precision=0.0)
    best_f1 = max(float(r.get("metrics", {}).get("f1", 0.0)) for r in flattened)
    best_precision = max(float(r.get("metrics", {}).get("precision", 0.0)) for r in flattened)
    return dict(total_regions=len(flattened), best_f1=best_f1, best_precision=best_precision)


def run_variant(
    name: str,
    func: VariantFunc,
    X: np.ndarray,
    y: np.ndarray,
    sel: Dict[str, object],
    feature_names: List[str],
    dataset_size: int,
    reports_dir: Path,
) -> Tuple[Dict[int, List[Dict[str, object]]], float, Path]:
    sel_payload = copy.deepcopy(sel)
    start = perf_counter()
    valuable = func(
        X,
        y,
        sel_payload,
        feature_names=feature_names,
        rng_seed=0,
    )
    duration = perf_counter() - start
    report_text = describe_regions_report(valuable, top_per_class=2, dataset_size=dataset_size)
    report_path = reports_dir / f"{name}_top2.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return valuable, duration, report_path


def main() -> None:
    root = ROOT
    reports_dir = root / "experiments_outputs" / "ab_test_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = root / "experiments_outputs" / "find_low_dim_spaces_ab_test_results.csv"

    X, y, feature_names = make_corner_class_dataset(
        n_per_cluster=600,
        std_class1=0.45,
        std_other=0.85,
        a=2.8,
        random_state=0,
    )
    dataset_size = int(y.shape[0])
    selection = _build_demo_selection(X, y, feature_names)

    baseline_val, baseline_runtime, baseline_report = run_variant(
        "baseline",
        find_low_dim_spaces,
        X,
        y,
        selection,
        feature_names,
        dataset_size,
        reports_dir,
    )
    baseline_summary = summarise_regions(baseline_val)

    variants: List[Tuple[str, VariantFunc]] = [
        ("deterministic", find_low_dim_spaces_deterministic),
        ("support_first", find_low_dim_spaces_support_first),
        ("precision_boost", find_low_dim_spaces_precision_boost),
    ]

    rows: List[Dict[str, object]] = []

    for variant_name, variant_func in variants:
        variant_val, variant_runtime, variant_report = run_variant(
            variant_name,
            variant_func,
            X,
            y,
            selection,
            feature_names,
            dataset_size,
            reports_dir,
        )
        variant_summary = summarise_regions(variant_val)
        rows.append(
            dict(
                variant=variant_name,
                baseline_runtime_s=baseline_runtime,
                variant_runtime_s=variant_runtime,
                runtime_ratio=variant_runtime / baseline_runtime if baseline_runtime > 0 else float("nan"),
                baseline_total_regions=baseline_summary["total_regions"],
                variant_total_regions=variant_summary["total_regions"],
                baseline_best_f1=baseline_summary["best_f1"],
                variant_best_f1=variant_summary["best_f1"],
                baseline_best_precision=baseline_summary["best_precision"],
                variant_best_precision=variant_summary["best_precision"],
                baseline_report=str(baseline_report.relative_to(root)),
                variant_report=str(variant_report.relative_to(root)),
            )
        )

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "variant",
                "baseline_runtime_s",
                "variant_runtime_s",
                "runtime_ratio",
                "baseline_total_regions",
                "variant_total_regions",
                "baseline_best_f1",
                "variant_best_f1",
                "baseline_best_precision",
                "variant_best_precision",
                "baseline_report",
                "variant_report",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved AB testing results to {csv_path}")
    for row in rows:
        print(
            f"Variant {row['variant']}: runtime {row['variant_runtime_s']:.3f}s, "
            f"best F1={row['variant_best_f1']:.3f}, regions={row['variant_total_regions']}"
        )


if __name__ == "__main__":
    main()
