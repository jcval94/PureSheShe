"""Pruebas para el bundle de métodos núcleo."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.ensemble import RandomForestClassifier

from deldel import (
    ChangePointConfig,
    DelDel,
    DelDelConfig,
    compute_frontier_planes_all_modes,
    prune_and_orient_planes_unified_globalmaj,
)
from subspaces.experiments.core_method_bundle import (
    CORE_METHOD_KEYS,
    AdaptiveSamplingInfo,
    run_core_method_bundle,
    run_single_method,
    summarize_core_bundle,
    _maybe_adaptive_sample,
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
        max_sets=None,
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
            max_sets=None,
            combo_sizes=(2,),
            random_state=7,
            cv_splits=2,
        )
        assert bundle.explorer.method_candidate_sets_[method_key] == single.method_candidate_sets_[method_key]

    timings = bundle.explorer.method_timings_
    for friendly_name, seconds in timings.items():
        assert seconds >= 0.0, friendly_name

    info = bundle.sampling_info
    assert isinstance(info, AdaptiveSamplingInfo)
    assert info.original_size == len(y)
    assert info.sampled_size <= info.original_size


def test_bundle_summary_matches_explorer_maps() -> None:
    X, y = make_classification(
        n_samples=180,
        n_features=10,
        n_informative=6,
        n_classes=3,
        random_state=3,
    )

    result = run_core_method_bundle(
        X,
        y,
        records=[],
        max_sets=None,
        combo_sizes=(2,),
        random_state=3,
        cv_splits=2,
    )

    summaries = summarize_core_bundle(result)
    explorer = result.explorer

    expected_keys = list(explorer.method_name_map_.keys())
    assert [summary.method_key for summary in summaries] == expected_keys

    for summary in summaries:
        assert summary.method_name == explorer.method_name_map_[summary.method_key]
        assert summary.candidate_sets == len(explorer.method_candidate_sets_.get(summary.method_key, set()))
        assert summary.selected_reports >= 0
        assert summary.elapsed_seconds >= 0.0
    assert result.sampling_info.original_size == len(y)


def test_reports_include_method_metadata() -> None:
    X, y = make_classification(
        n_samples=140,
        n_features=9,
        n_informative=6,
        n_classes=4,
        random_state=5,
    )

    result = run_core_method_bundle(
        X,
        y,
        records=[],
        max_sets=None,
        combo_sizes=(2,),
        random_state=5,
        cv_splits=2,
    )

    explorer = result.explorer
    reports = result.reports
    method_keys = list(explorer.method_name_map_.keys())

    # Conteos esperados por método
    expected_global = {
        key: len(explorer.method_candidate_sets_.get(key, set())) for key in method_keys
    }
    top_hits = {
        key: 0 for key in method_keys
    }
    for report in sorted(reports, key=lambda r: r.mean_macro_f1, reverse=True)[:50]:
        combo = tuple(sorted(report.features))
        for method in explorer.candidate_sources_.get(combo, set()):
            if method in top_hits:
                top_hits[method] += 1

    assert reports, "los reportes no deben estar vacíos"
    for report in reports:
        combo = tuple(sorted(report.features))
        expected_origins = tuple(sorted(explorer.candidate_sources_.get(combo, set())))
        assert report.method_keys == expected_origins
        assert report.method_key == (expected_origins[0] if expected_origins else None)
        assert report.global_yes == sum(expected_global.get(key, 0) for key in expected_origins)
        assert report.top50_yes == sum(top_hits.get(key, 0) for key in expected_origins)


def test_extract_top_subspaces_by_method() -> None:
    X, y = make_classification(
        n_samples=160,
        n_features=10,
        n_informative=6,
        n_classes=3,
        random_state=13,
    )

    bundle = run_core_method_bundle(
        X,
        y,
        records=[],
        max_sets=None,
        combo_sizes=(2,),
        random_state=13,
        cv_splits=2,
    )

    top_subspaces = {}
    for key in CORE_METHOD_KEYS:
        candidates = [r for r in bundle.reports if key in r.method_keys]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: r.mean_macro_f1)
        top_subspaces[key] = {
            "features": best.features,
            "f1": best.mean_macro_f1,
            "planes": best.planes,
        }

    assert top_subspaces, "al menos un método debe aportar subespacios"
    assert set(top_subspaces).issubset(CORE_METHOD_KEYS)

    for method_key, payload in top_subspaces.items():
        ranked = sorted(
            (r for r in bundle.reports if method_key in r.method_keys),
            key=lambda r: r.mean_macro_f1,
            reverse=True,
        )
        assert payload["features"] == ranked[0].features
        assert payload["f1"] == ranked[0].mean_macro_f1
        assert payload["planes"] == ranked[0].planes


def test_records_to_selection_pipeline() -> None:
    X, y = make_classification(
        n_samples=140,
        n_features=8,
        n_informative=6,
        n_classes=3,
        random_state=11,
    )
    model = RandomForestClassifier(n_estimators=50, random_state=11).fit(X, y)

    deldel_cfg = DelDelConfig(segments_target=60, random_state=11)
    cp_cfg = ChangePointConfig(enabled=False)
    records = DelDel(deldel_cfg, cp_cfg).fit(X, model).records_

    res_c = compute_frontier_planes_all_modes(
        records,
        mode="C",
        min_cluster_size=6,
        max_models_per_round=3,
        max_depth=1,
        seed=11,
    )
    assert res_c, "res_c debe contener al menos un par de clases"

    selection = prune_and_orient_planes_unified_globalmaj(
        res_c,
        X,
        y,
        feature_names=[f"f{i}" for i in range(X.shape[1])],
        max_k=3,
        min_improve=1e-4,
        min_region_size=6,
        min_abs_diff=0.0,
        min_rel_lift=0.0,
        min_recall=0.0,
        diversity_additions=False,
    )

    assert "winning_planes" in selection
    assert "regions_global" in selection
    assert set(selection["by_pair_augmented"].keys()).issubset(res_c.keys())


def test_adaptive_sampling_skips_small_dataset() -> None:
    X = np.arange(90).reshape(30, 3)
    y = np.arange(30)

    sampled_X, sampled_y, sampled_records, info = _maybe_adaptive_sample(
        X,
        y,
        records=[],
        adaptive_sampling=True,
        random_state=123,
    )

    assert sampled_X is X
    assert sampled_y is y
    assert sampled_records == []
    assert not info.sampling_enabled
    assert info.sampled_size == len(y)


def test_adaptive_sampling_reduces_large_dataset_and_is_reproducible() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(2000, 48))
    y = rng.integers(0, 5, size=2000)
    records = list(range(2000))

    first = _maybe_adaptive_sample(
        X,
        y,
        records=records,
        adaptive_sampling=True,
        random_state=99,
    )
    second = _maybe_adaptive_sample(
        X,
        y,
        records=records,
        adaptive_sampling=True,
        random_state=99,
    )

    (_, first_y, first_records, first_info) = first
    (_, second_y, second_records, second_info) = second

    assert first_info.sampling_enabled
    assert first_info.sampled_size < len(y)
    assert first_info.sampled_size >= 500
    assert first_info.sampled_size == len(first_y)
    assert first_info.sampled_size == len(first_records)
    assert first_records == sorted(first_records)
    assert np.array_equal(first_y, second_y)
    assert first_records == second_records
    assert first_info.sampled_size == second_info.sampled_size


def test_adaptive_sampling_digits_dataset_has_representative_size() -> None:
    digits = load_digits()
    X = digits.data
    y = digits.target

    sampled_X, sampled_y, sampled_records, info = _maybe_adaptive_sample(
        X,
        y,
        records=[],
        adaptive_sampling=True,
        random_state=42,
    )

    assert info.sampling_enabled
    assert info.sampled_size >= 500
    assert info.sampled_size < len(y)
    assert sampled_X.shape[0] == info.sampled_size
    assert sampled_y.shape[0] == info.sampled_size
    assert sampled_records == []
