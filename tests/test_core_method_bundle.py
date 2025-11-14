"""Pruebas para el bundle de métodos núcleo."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from sklearn.datasets import make_classification

from subspaces.experiments.core_method_bundle import (
    CORE_METHOD_KEYS,
    _adaptive_sample_size,
    run_core_method_bundle,
    run_single_method,
)


def test_bundle_matches_single_methods() -> None:
    X, y = make_classification(
        n_samples=220,
        n_features=12,
        n_informative=8,
        n_classes=4,
        random_state=7,
    )

    bundle = run_core_method_bundle(
        X,
        y,
        records=[],
        max_sets=2,
        combo_sizes=(2,),
        random_state=7,
        cv_splits=2,
    )

    for method_key in CORE_METHOD_KEYS:
        single = run_single_method(
            X,
            y,
            records=[],
            method_key=method_key,
            max_sets=2,
            combo_sizes=(2,),
            random_state=7,
            cv_splits=2,
        )
        assert bundle.explorer.method_candidate_sets_[method_key] == single.method_candidate_sets_[method_key]

    timings = bundle.explorer.method_timings_
    for friendly_name, seconds in timings.items():
        assert seconds >= 0.0, friendly_name


def test_adaptive_sample_size_behaviour() -> None:
    assert _adaptive_sample_size(50, 8) == 50
    assert _adaptive_sample_size(100, 8) == 100
    assert _adaptive_sample_size(150, 8) == 100
    assert _adaptive_sample_size(1200, 8) == 300
    assert _adaptive_sample_size(12000, 8) == 840
    assert _adaptive_sample_size(120000, 8) == 3345
    assert _adaptive_sample_size(120000, 32) > _adaptive_sample_size(120000, 8)
    assert _adaptive_sample_size(120000, 2) < _adaptive_sample_size(120000, 8)
    assert _adaptive_sample_size(150, 6, row_col_product=900) == 150


def test_run_core_method_bundle_applies_sampling() -> None:
    n_samples = 5000
    X = np.arange(n_samples * 4.0).reshape(n_samples, 4)
    y = np.arange(n_samples)
    records = list(range(n_samples))

    class DummyExplorer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X_fit, y_fit, record_fit):
            self.sample_count = len(y_fit)
            self.records_len = len(record_fit)
            self.shape = getattr(X_fit, "shape", (len(X_fit), None))
            return self

        def get_report(self):
            return []

    with patch(
        "subspaces.experiments.core_method_bundle.MultiClassSubspaceExplorer",
        DummyExplorer,
    ):
        result = run_core_method_bundle(
            X,
            y,
            records,
            max_sets=1,
            combo_sizes=(2,),
            random_state=13,
            cv_splits=2,
            adaptive_sampling=True,
        )

    expected = _adaptive_sample_size(
        n_samples,
        4,
        total_size=n_samples * 4,
        row_col_product=n_samples * 4,
    )
    assert result.explorer.sample_count == expected
    assert result.explorer.records_len == expected
    assert result.explorer.shape[0] == expected


def test_run_core_method_bundle_sampling_disabled_uses_full_dataset() -> None:
    n_samples = 750
    X = np.arange(n_samples * 2.0).reshape(n_samples, 2)
    y = np.arange(n_samples)
    records = list(range(n_samples))

    class DummyExplorer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X_fit, y_fit, record_fit):
            self.sample_count = len(y_fit)
            self.records_len = len(record_fit)
            self.shape = getattr(X_fit, "shape", (len(X_fit), None))
            return self

        def get_report(self):
            return []

    with patch(
        "subspaces.experiments.core_method_bundle.MultiClassSubspaceExplorer",
        DummyExplorer,
    ):
        result = run_core_method_bundle(
            X,
            y,
            records,
            max_sets=1,
            combo_sizes=(2,),
            random_state=17,
            cv_splits=2,
            adaptive_sampling=False,
        )

    assert result.explorer.sample_count == n_samples
    assert result.explorer.records_len == n_samples
    assert result.explorer.shape[0] == n_samples


def test_run_core_method_bundle_skips_sampling_when_rowcol_small() -> None:
    n_samples = 150
    X = np.arange(n_samples * 6.0).reshape(n_samples, 6)
    y = np.arange(n_samples)

    class DummyExplorer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X_fit, y_fit, record_fit):
            self.sample_count = len(y_fit)
            self.shape = getattr(X_fit, "shape", (len(X_fit),))
            return self

        def get_report(self):
            return []

    with patch(
        "subspaces.experiments.core_method_bundle.MultiClassSubspaceExplorer",
        DummyExplorer,
    ):
        result = run_core_method_bundle(
            X,
            y,
            records=[],
            max_sets=1,
            combo_sizes=(2,),
            random_state=11,
            cv_splits=2,
            adaptive_sampling=True,
        )

    assert result.explorer.sample_count == n_samples
    assert result.explorer.shape[0] == n_samples


def test_run_core_method_bundle_handles_multidimensional_inputs() -> None:
    n_samples = 3200
    X = np.arange(n_samples * 20.0).reshape(n_samples, 4, 5)
    y = np.arange(n_samples)

    class DummyExplorer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X_fit, y_fit, record_fit):
            self.sample_count = len(y_fit)
            self.shape = getattr(X_fit, "shape", (len(X_fit),))
            return self

        def get_report(self):
            return []

    with patch(
        "subspaces.experiments.core_method_bundle.MultiClassSubspaceExplorer",
        DummyExplorer,
    ):
        result = run_core_method_bundle(
            X,
            y,
            records=[],
            max_sets=1,
            combo_sizes=(2,),
            random_state=23,
            cv_splits=2,
            adaptive_sampling=True,
        )

    expected = _adaptive_sample_size(
        n_samples,
        20,
        total_size=n_samples * 20,
        row_col_product=n_samples * 20,
    )
    assert result.explorer.sample_count == expected
