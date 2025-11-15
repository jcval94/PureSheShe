"""AB testing script for ScoreAdaptor.scores_dedup optimisations.

The script compares the previous implementation (baseline) with the
optimised version that consults the cache before re-evaluating repeated
rows.  It generates two CSV reports in ``experiments_outputs``:

``scores_dedup_ab_timing.csv``
    Per dataset timings for each repeated call and adaptor version.

``scores_dedup_ab_consistency.csv``
    Verification that both implementations return identical scores.
"""

from __future__ import annotations

import copy
import csv
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:  # pragma: no cover - import side effect
    sys.path.insert(0, str(_ROOT / "src"))

from deldel.engine import ScoreAdaptor


class CountingModel:
    """Proxy that counts how many times ``predict_proba`` is invoked."""

    def __init__(self, model):
        self._model = model
        self.calls = 0

    def predict_proba(self, X):  # pragma: no cover - simple delegation
        self.calls += 1
        return self._model.predict_proba(X)

    def __getattr__(self, name):  # pragma: no cover - delegation helper
        return getattr(self._model, name)


class BaselineScoreAdaptor(ScoreAdaptor):
    """ScoreAdaptor with the historical ``scores_dedup`` implementation."""

    def scores_dedup(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        X = np.asarray(X, float)
        if X.ndim == 1:
            return self.scores(X)
        Q = np.round(X, getattr(self, "cache_decimals", 6))
        uniq, idx, inv = np.unique(Q, axis=0, return_index=True, return_inverse=True)
        Suniq = self._scores_raw(uniq)
        keys = [u.tobytes() for u in uniq]
        for k, val in zip(keys, Suniq):
            self._cache[k] = val
        return Suniq[inv]


def build_dataset(
    *,
    n_samples: int,
    n_features: int,
    random_state: int,
    n_classes: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create training and evaluation matrices with repeated rows."""

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 2,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        random_state=random_state,
    )

    # Evaluation matrix with repeated rows to exercise the cache between calls.
    rng = np.random.default_rng(random_state)
    dup_idx = rng.choice(n_samples, size=max(1, n_samples // 5), replace=False)
    repeats = rng.integers(2, 5, size=dup_idx.size)
    duplicated = np.repeat(X[dup_idx], repeats=repeats, axis=0)
    X_eval = np.vstack([X, duplicated])

    return X, y, X_eval


def run_version(
    adaptor_cls,
    *,
    model,
    X_eval: np.ndarray,
    n_calls: int,
) -> Tuple[List[float], List[np.ndarray], List[int]]:
    adaptor = adaptor_cls(model, cache_enabled=True, cache_decimals=6)
    times: List[float] = []
    outputs: List[np.ndarray] = []
    call_counts: List[int] = []
    for _ in range(n_calls):
        prev_calls = getattr(model, "calls", 0)
        t0 = perf_counter()
        scores = adaptor.scores_dedup(X_eval)
        times.append((perf_counter() - t0) * 1000.0)
        outputs.append(scores.copy())
        call_counts.append(getattr(model, "calls", 0) - prev_calls)
    return times, outputs, call_counts


def main() -> None:
    datasets = [
        ("small", dict(n_samples=600, n_features=12, random_state=7)),
        ("medium", dict(n_samples=2400, n_features=18, random_state=17)),
        ("large", dict(n_samples=7200, n_features=24, random_state=27)),
    ]
    n_calls = 5
    decimals = 6

    timing_rows: List[Dict[str, object]] = []
    consistency_rows: List[Dict[str, object]] = []

    for name, params in datasets:
        X_train, y_train, X_eval = build_dataset(**params)

        model = LogisticRegression(max_iter=1000, multi_class="multinomial")
        model.fit(X_train, y_train)

        baseline_model = CountingModel(copy.deepcopy(model))
        optim_model = CountingModel(copy.deepcopy(model))

        n_rows, n_features = X_eval.shape
        n_unique = np.unique(np.around(X_eval, decimals=decimals), axis=0).shape[0]

        baseline_times, baseline_outputs, baseline_calls = run_version(
            BaselineScoreAdaptor, model=baseline_model, X_eval=X_eval, n_calls=n_calls
        )
        optim_times, optim_outputs, optim_calls = run_version(
            ScoreAdaptor, model=optim_model, X_eval=X_eval, n_calls=n_calls
        )

        for call_idx in range(n_calls):
            timing_rows.append(
                dict(
                    dataset=name,
                    call=call_idx + 1,
                    version="baseline",
                    n_rows=n_rows,
                    n_unique=n_unique,
                    elapsed_ms=baseline_times[call_idx],
                    model_calls=baseline_calls[call_idx],
                )
            )
            timing_rows.append(
                dict(
                    dataset=name,
                    call=call_idx + 1,
                    version="optimised",
                    n_rows=n_rows,
                    n_unique=n_unique,
                    elapsed_ms=optim_times[call_idx],
                    model_calls=optim_calls[call_idx],
                )
            )

            base = baseline_outputs[call_idx]
            opt = optim_outputs[call_idx]
            diff = base - opt
            max_abs_diff = float(np.max(np.abs(diff))) if diff.size else 0.0
            consistency_rows.append(
                dict(
                    dataset=name,
                    call=call_idx + 1,
                    n_rows=n_rows,
                    n_unique=n_unique,
                    consistent=bool(np.allclose(base, opt)),
                    max_abs_diff=max_abs_diff,
                )
            )

    out_dir = Path(__file__).resolve().parent
    timing_path = out_dir / "scores_dedup_ab_timing.csv"
    consistency_path = out_dir / "scores_dedup_ab_consistency.csv"

    with timing_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "call",
                "version",
                "n_rows",
                "n_unique",
                "elapsed_ms",
                "model_calls",
            ],
        )
        writer.writeheader()
        writer.writerows(timing_rows)

    with consistency_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dataset", "call", "n_rows", "n_unique", "consistent", "max_abs_diff"],
        )
        writer.writeheader()
        writer.writerows(consistency_rows)


if __name__ == "__main__":
    main()
