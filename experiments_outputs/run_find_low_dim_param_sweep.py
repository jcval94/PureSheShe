"""Run a parameter sweep for :func:`find_low_dim_spaces` and export metrics.

This helper relies on the synthetic corner dataset pipeline defined in
:mod:`deldel.experiments` to evaluate several configurations of
:func:`find_low_dim_spaces`.  Results are written to
``finder_runs_params.csv`` inside the ``experiments_outputs`` folder so they
can be inspected alongside other experiment artifacts.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping

# Ensure the package sources are available without requiring an installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deldel.experiments import run_corner_pipeline_with_low_dim  # noqa: E402


def _write_finder_runs_csv(
    finder_runs: Iterable[Mapping[str, object]],
    *,
    n_dims: int,
    csv_path: Path,
) -> None:
    """Persist finder runs with flattened parameter columns."""
    finder_runs = list(finder_runs)
    param_keys = sorted({k for run in finder_runs for k in run.get("params", {})})

    fieldnames = [
        "run_index",
        "runtime_s",
        "regions_total",
        "params_json",
    ]
    fieldnames.extend(param_keys)
    for dim in range(1, n_dims + 1):
        fieldnames.append(f"regions_dim{dim}")
        fieldnames.append(f"best_f1_dim{dim}")
        fieldnames.append(f"best_precision_dim{dim}")

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in finder_runs:
            params = dict(entry.get("params", {}))
            row = {
                "run_index": entry.get("run_index"),
                "runtime_s": f"{float(entry.get('runtime_s', 0.0)):.6f}",
                "regions_total": entry.get("regions_total"),
                "params_json": json.dumps(params, sort_keys=True),
            }
            for key in param_keys:
                row[key] = params.get(key)
            for dim in range(1, n_dims + 1):
                row[f"regions_dim{dim}"] = entry.get("regions_by_dim", {}).get(dim, 0)
                row[f"best_f1_dim{dim}"] = (
                    f"{float(entry.get('best_f1_by_dim', {}).get(dim, 0.0)):.6f}"
                )
                row[f"best_precision_dim{dim}"] = (
                    f"{float(entry.get('best_precision_by_dim', {}).get(dim, 0.0)):.6f}"
                )
            writer.writerow(row)


def main() -> None:
    param_grid = [
        dict(
            max_planes_in_rule=2,
            max_planes_per_pair=3,
            min_support=20,
            min_rel_gain_f1=0.02,
            min_lift_prec=1.20,
            consider_dims_up_to=3,
            rng_seed=0,
        ),
        dict(
            max_planes_in_rule=3,
            max_planes_per_pair=4,
            min_support=30,
            min_rel_gain_f1=0.05,
            min_lift_prec=1.35,
            consider_dims_up_to=4,
            rng_seed=1,
        ),
        dict(
            max_planes_in_rule=4,
            max_planes_per_pair=5,
            min_support=25,
            min_rel_gain_f1=0.08,
            min_lift_prec=1.50,
            consider_dims_up_to=4,
            rng_seed=2,
            heuristic_merge_enable=True,
            p_merge=0.25,
        ),
        dict(
            max_planes_in_rule=3,
            max_planes_per_pair=6,
            min_support=18,
            min_rel_gain_f1=0.04,
            min_lift_prec=1.25,
            consider_dims_up_to=5,
            rng_seed=3,
            heuristic_merge_enable=True,
            p_merge=0.35,
        ),
    ]

    summary = run_corner_pipeline_with_low_dim(
        dataset_kwargs=dict(n_per_cluster=180, random_state=0),
        finder_param_grid=param_grid,
        csv_dir=None,
    )

    csv_path = Path(__file__).resolve().parent / "finder_runs_params.csv"
    _write_finder_runs_csv(
        summary.get("finder_runs", []),
        n_dims=int(summary.get("dataset", {}).get("n_dims", 0) or 0),
        csv_path=csv_path,
    )
    print(f"Exported finder runs to {csv_path}")


if __name__ == "__main__":
    main()
