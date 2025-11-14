"""Pruebas para el bundle de métodos núcleo."""

from __future__ import annotations

from sklearn.datasets import make_classification

from subspaces.experiments.core_method_bundle import (
    CORE_METHOD_KEYS,
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
