from __future__ import annotations

import numpy as np

from deldel.subspace_change_detector import MultiClassSubspaceExplorer, SubspaceReport


class DummyTable:
    def __init__(self, columns, arrays):
        self.columns = list(columns)
        self._mapping = {col: np.asarray(arr) for col, arr in zip(self.columns, arrays)}

    def __getitem__(self, key):
        return self._mapping[key]


def _make_table(n_samples: int = 60) -> tuple[DummyTable, np.ndarray]:
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, 6)
    y = rng.randint(0, 3, size=n_samples)
    columns = []
    arrays = []
    for j in range(X.shape[1]):
        name = f"x{j}"
        if j < 2:
            arrays.append(X[:, j] > np.median(X[:, j]))
        else:
            arrays.append(X[:, j])
        columns.append(name)
    return DummyTable(columns, arrays), y


def test_explorer_tracks_candidate_metadata() -> None:
    table, y = _make_table()
    explorer = MultiClassSubspaceExplorer(
        max_sets=5,
        combo_sizes=(2,),
        filter_top_k=6,
        chi2_pool=6,
        random_samples=20,
        corr_threshold=0.2,
        rf_estimators=5,
        rf_max_depth=3,
        cv_splits=2,
        random_state=0,
    )

    explorer.fit(table, y, records=[])
    reports = explorer.get_report()

    expected_methods = {
        "method_1_topk_random",
        "method_1b_topk_guided",
        "method_2_chi2",
        "method_2b_chi2_guided",
        "method_3_corr_groups",
        "method_3b_corr_guided",
        "method_4_rf_paths",
        "method_4b_rf_weighted",
        "method_5_leverage",
        "method_5b_leverage_class",
        "method_6_sparse_proj",
        "method_6b_sparse_proj_guided",
        "method_7_lazy_greedy",
        "method_7b_lazy_greedy_refined",
        "method_8_extratrees",
        "method_8b_extratrees_refined",
        "method_9_countsketch",
        "method_9b_countsketch_refined",
        "method_10_gradient_synergy",
        "method_10b_gradient_hessian",
        "method_11_minhash_lsh",
        "method_11b_minhash_refined",
    }
    assert expected_methods.issubset(set(explorer.method_name_map_.keys()))
    assert explorer.feature_ranking_, "feature ranking should be captured"
    assert explorer.mi_scores_, "mutual information scores should be stored"
    assert explorer.chi2_scores_, "chi2 scores should be stored"
    assert explorer.stump_scores_, "stump scores should be stored"
    assert explorer.all_candidate_sets_, "candidate sets should not be empty"
    assert set(explorer.method_candidate_sets_.keys()) == set(explorer.method_name_map_.keys())
    assert set(explorer.method_timings_.keys()) == set(explorer.method_name_map_.values())
    assert len(explorer.evaluated_reports_) == len(explorer.candidate_reports_)
    if explorer.evaluated_reports_:
        assert explorer.evaluated_reports_[0].features in explorer.candidate_reports_
    report_keys = set(explorer.candidate_reports_.keys())
    assert report_keys.issubset(set(explorer.all_candidate_sets_))

    if reports:
        first = tuple(sorted(reports[0].features))
        assert first in explorer.candidate_sources_

    kept = set(map(tuple, explorer.all_candidate_sets_))
    for combos in explorer.method_candidate_sets_.values():
        assert combos.issubset(kept)


def _make_report(size: int, score: float) -> SubspaceReport:
    features = tuple(f"x{i}" for i in range(size))
    return SubspaceReport(
        features=features,
        mean_macro_f1=score,
        std_macro_f1=0.0,
        lift_vs_majority=score,
        coverage_ratio=1.0,
        per_class_f1={0: score},
        support=10,
        variance_ratio=1.0,
        l1_importance=0.0,
    )


def test_pick_diverse_by_size_cycles_groupings() -> None:
    explorer = MultiClassSubspaceExplorer(max_sets=4, combo_sizes=(2, 3, 4), random_state=0)
    ordered = [
        _make_report(3, 0.96),
        _make_report(2, 0.94),
        _make_report(4, 0.92),
        _make_report(3, 0.9),
        _make_report(2, 0.88),
    ]

    selected = explorer._pick_diverse_by_size(ordered)

    assert len(selected) <= explorer.max_sets
    sizes = [len(report.features) for report in selected]
    assert len(set(sizes)) >= 2
    assert sizes[0] == 3  # highest score retained
    assert 4 in sizes  # larger combos also represented
