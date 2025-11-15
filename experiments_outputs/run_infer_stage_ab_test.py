"""Persist A/B results for the stage inference helper to CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deldel import stage_infer_ab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[10_000],
        help="One or more iteration counts to benchmark each implementation.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of repeated timing runs per iteration count.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=ROOT / "experiments_outputs" / "infer_stage_ab_results.csv",
        help="Destination CSV path for per-scenario outputs.",
    )
    parser.add_argument(
        "--timings-path",
        type=Path,
        default=ROOT / "experiments_outputs" / "infer_stage_ab_timings.csv",
        help="Destination CSV path for runtime measurements.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    iterations = args.iterations
    if len(iterations) == 1:
        iterations = iterations[0]

    summary = stage_infer_ab.write_infer_stage_ab_csvs(
        iterations=iterations,
        repeat=args.repeat,
        results_path=str(args.results_path),
        timings_path=str(args.timings_path),
    )

    print(
        "Wrote "
        f"{len(list(summary['results']))} scenario rows to {args.results_path}"
    )
    print(
        "Wrote "
        f"{len(list(summary['timings']))} timing rows to {args.timings_path}"
    )


if __name__ == "__main__":
    main()
