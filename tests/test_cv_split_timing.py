import csv
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from deldel.subspace_change_detector import _evaluate_subspace


def _time_evaluation(splitter, X, y, classes) -> float:
    start = perf_counter()
    _evaluate_subspace(X, y, splitter, classes)
    return perf_counter() - start


def _write_csv(output_path: Path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["version", "run_seconds", "mean_seconds", "runs"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_single_split_path_is_faster():
    X, y = make_classification(
        n_samples=2500,
        n_features=40,
        n_informative=20,
        n_redundant=10,
        n_classes=3,
        random_state=0,
    )
    classes = np.unique(y)

    runs = 5
    legacy_splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    current_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.25, random_state=0
    )

    legacy_times = [_time_evaluation(legacy_splitter, X, y, classes) for _ in range(runs)]
    current_times = [_time_evaluation(current_splitter, X, y, classes) for _ in range(runs)]

    legacy_mean = statistics.mean(legacy_times)
    current_mean = statistics.mean(current_times)

    output = []
    for t in legacy_times:
        output.append(
            {
                "version": "previous_2fold",
                "run_seconds": f"{t:.4f}",
                "mean_seconds": f"{legacy_mean:.4f}",
                "runs": runs,
            }
        )
    for t in current_times:
        output.append(
            {
                "version": "current_shuffle_holdout",
                "run_seconds": f"{t:.4f}",
                "mean_seconds": f"{current_mean:.4f}",
                "runs": runs,
            }
        )

    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "experiments_outputs" / "cv_split_timing.csv"
    _write_csv(csv_path, output)

    assert current_mean < legacy_mean, (
        f"Expected shuffle holdout path to be faster than previous; "
        f"got {current_mean:.4f}s vs {legacy_mean:.4f}s"
    )

