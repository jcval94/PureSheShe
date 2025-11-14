"""Pruebas para el bundle de métodos núcleo."""

from __future__ import annotations

from sklearn.datasets import make_classification
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
    run_core_method_bundle,
    run_single_method,
    summarize_core_bundle,
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
        max_sets=3,
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
