from time import perf_counter
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from numpy.testing import assert_allclose
from sklearn.linear_model import LogisticRegression

from deldel import (
    ChangePointConfig,
    DelDel,
    DelDelConfig,
    DeltaRecord,
    DeltaRecordLite,
    ModelCall,
    PCA3D,
    ScoreAdaptor,
    build_weighted_frontier,
    compute_frontier_planes_all_modes,
    compute_frontier_planes_weighted,
    find_low_dim_spaces,
    fit_cubic_from_records_weighted,
    fit_quadric_svd_weighted,
    fit_quadrics_from_records_weighted,
    fit_tls_plane_weighted,
    plot_planes_with_point_lines,
    plot_frontiers_implicit_interactive_v2,
    make_corner_class_dataset,
    run_corner_pipeline_experiments,
    run_low_dim_spaces_demo,
    SubspaceReport,
)


def run_and_time(label, func, *args, **kwargs):
    start = perf_counter()
    result = func(*args, **kwargs)
    duration = perf_counter() - start
    print(f"{label}: {duration:.6f}s")
    return result


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def corner_dataset():
    X, y, feature_names = run_and_time(
        "make_corner_class_dataset",
        make_corner_class_dataset,
        n_per_cluster=1150,
        std_class1=0.4,
        std_other=0.8,
        a=3.0,
        random_state=0,
    )
    return dict(X=X, y=y, feature_names=feature_names)


@pytest.fixture(scope="module")
def dataset(corner_dataset):
    X = corner_dataset["X"]
    y = corner_dataset["y"]
    model = LogisticRegression(
        max_iter=500,
        random_state=0,
        solver="lbfgs",
    )
    model = run_and_time("LogisticRegression.fit (corner dataset)", model.fit, X, y)
    return dict(**corner_dataset, model=model)


@pytest.fixture(scope="module")
def adaptor(dataset):
    X = dataset["X"]
    model = dataset["model"]
    adaptor = ScoreAdaptor(model, mode="auto", cache_enabled=True)
    # warm-up to build cache metadata for feature names
    run_and_time("ScoreAdaptor.scores (warm-up)", adaptor.scores, X[:1])
    return adaptor


@pytest.fixture(scope="module")
def sample_records(rng):
    records = []
    dim = 4
    for pair in [(0, 1), (1, 2)]:
        a, b = pair
        for _ in range(6):
            x0 = rng.normal(size=dim)
            direction = rng.normal(size=dim) * 0.2
            x1 = x0 + direction + (b - a) * 0.1
            delta = x1 - x0
            S0 = np.full(dim, 0.05)
            S1 = np.full(dim, 0.05)
            S0[a] = 0.9
            S1[b] = 0.9
            record = DeltaRecord(
                index_a=0,
                index_b=1,
                method="test",
                success=True,
                y0=a,
                y1=b,
                delta_norm_l2=float(np.linalg.norm(delta, ord=2)),
                delta_norm_linf=float(np.linalg.norm(delta, ord=np.inf)),
                score_change=0.75,
                distance_term=0.1,
                change_term=0.2,
                final_score=0.8,
                time_ms=0.5,
                x0=x0,
                x1=x1,
                delta=delta,
                S0=S0,
                S1=S1,
                prob_swing=0.1,
                margin_gain=0.2,
                jsd_change=0.0,
            )
            records.append(record)
    return records


def test_make_corner_class_dataset(corner_dataset):
    X = corner_dataset["X"]
    y = corner_dataset["y"]
    assert X.shape[1] == 4
    assert X.shape[0] == 8 * 1150
    assert y.shape == (X.shape[0],)


def test_dataclasses_instantiation():
    cfg = DelDelConfig(segments_target=4, random_state=1)
    cp_cfg = ChangePointConfig(enabled=False)
    record_lite = DeltaRecordLite(
        y0=0,
        y1=1,
        x0=np.zeros(2),
        x1=np.ones(2),
        cp_x=np.zeros((0, 0)),
        cp_count=0,
    )
    model_call = ModelCall(
        ts=0.0,
        stage=None,
        source="test",
        fn="predict",
        batch=1,
        n_features=2,
        cache_enabled=False,
        cache_hits=0,
        cache_misses=1,
        duration_ms=0.1,
    )

    assert cfg.segments_target == 4
    assert cp_cfg.enabled is False
    assert record_lite.y1 == 1
    assert model_call.batch == 1


def test_score_adaptor_scores_and_cache(dataset, adaptor):
    X = dataset["X"]
    y = dataset["y"]
    sample = X[:5]
    scores = run_and_time("ScoreAdaptor.scores (first call)", adaptor.scores, sample)
    assert scores.shape == (5, len(np.unique(y)))

    cached = run_and_time("ScoreAdaptor.scores (cached)", adaptor.scores, sample)
    assert_allclose(scores, cached, rtol=1e-6)

    dup = np.vstack([sample[0], sample[0]])
    dedup_scores = run_and_time("ScoreAdaptor.scores_dedup", adaptor.scores_dedup, dup)
    assert dedup_scores.shape == (2, scores.shape[1])
    assert_allclose(dedup_scores[0], dedup_scores[1], rtol=1e-6)


def test_pca3d_fit_transform(dataset):
    X = dataset["X"]
    pca = PCA3D()
    run_and_time("PCA3D.fit", pca.fit, X)
    transformed = run_and_time("PCA3D.transform", pca.transform, X[:4])
    assert transformed.shape == (4, 3)


def test_build_weighted_frontier_and_plane(sample_records):
    dim = sample_records[0].x0.shape[0]
    F_by, B_by, W_by = run_and_time(
        "build_weighted_frontier",
        build_weighted_frontier,
        sample_records,
        prefer_cp=False,
    )
    assert F_by
    for key, F in F_by.items():
        w = W_by[key]
        assert F.shape[1] == dim
        assert np.isclose(w.sum(), 1.0)

        n, b, mu = run_and_time("fit_tls_plane_weighted", fit_tls_plane_weighted, F, w)
        assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-6)
        assert mu.shape == (F.shape[1],)


def test_compute_frontier_planes_weighted(sample_records):
    planes = run_and_time(
        "compute_frontier_planes_weighted",
        compute_frontier_planes_weighted,
        sample_records,
        prefer_cp=False,
    )
    assert planes
    expected_dim = sample_records[0].x0.shape[0]
    for payload in planes.values():
        assert payload["n"].shape[0] == expected_dim
        assert "weights" in payload


def test_fit_quadric_svd_weighted_runtime_error(sample_records):
    start = perf_counter()
    with pytest.raises(RuntimeError):
        fit_quadric_svd_weighted(np.vstack([r.x1 for r in sample_records]), lambda Z: (Z, {}))
    duration = perf_counter() - start
    print(f"fit_quadric_svd_weighted (expected RuntimeError): {duration:.6f}s")


def test_fit_quadrics_and_cubics(sample_records):
    quadrics = run_and_time(
        "fit_quadrics_from_records_weighted",
        fit_quadrics_from_records_weighted,
        sample_records,
        prefer_cp=False,
    )
    assert quadrics
    for model in quadrics.values():
        assert set(model.keys()) >= {"Q", "r", "c", "weights"}


def test_fit_quadrics_parallel_matches_serial(sample_records):
    serial = run_and_time(
        "fit_quadrics_from_records_weighted (serial)",
        fit_quadrics_from_records_weighted,
        sample_records,
        prefer_cp=False,
        n_jobs=1,
    )

    parallel = run_and_time(
        "fit_quadrics_from_records_weighted (parallel)",
        fit_quadrics_from_records_weighted,
        sample_records,
        prefer_cp=False,
        n_jobs=2,
    )

    assert serial.keys() == parallel.keys()
    for key in serial:
        a = serial[key]
        b = parallel[key]
        assert set(a.keys()) == set(b.keys())
        assert_allclose(a["Q"], b["Q"], rtol=1e-8, atol=1e-10)
        assert_allclose(a["r"], b["r"], rtol=1e-8, atol=1e-10)
        assert np.isclose(a["c"], b["c"], rtol=1e-8, atol=1e-10)
        assert_allclose(a["weights"], b["weights"], rtol=1e-8, atol=1e-10)

    cubics = run_and_time(
        "fit_cubic_from_records_weighted",
        fit_cubic_from_records_weighted,
        sample_records,
        prefer_cp=False,
    )
    assert cubics
    for model in cubics.values():
        assert "w" in model
        assert "catalog" in model


def test_compute_frontier_planes_all_modes(sample_records):
    res = run_and_time(
        "compute_frontier_planes_all_modes",
        compute_frontier_planes_all_modes,
        sample_records,
        min_cluster_size=2,
    )
    assert res
    for payload in res.values():
        assert "planes_by_label" in payload
        assert "meta" in payload


def test_compute_frontier_with_explorer(sample_records):
    feature_names = [f"x{i}" for i in range(sample_records[0].x0.size)]
    report = SubspaceReport(
        features=("x0", "x1"),
        mean_macro_f1=0.5,
        std_macro_f1=0.0,
        lift_vs_majority=0.1,
        coverage_ratio=1.0,
        per_class_f1={0: 0.5},
        support=10,
        variance_ratio=0.0,
        l1_importance=0.0,
    )

    res = run_and_time(
        "compute_frontier_planes_all_modes (explorer)",
        compute_frontier_planes_all_modes,
        sample_records,
        min_cluster_size=1,
        explorer_reports=[report],
        explorer_feature_names=feature_names,
        explorer_top_k=1,
    )

    payload = next(iter(res.values()))
    assert payload.get("meta", {}).get("dimension") == sample_records[0].x0.size
    assert payload.get("meta", {}).get("dim_names")

    sub_variants = payload.get("subspace_variants") or {}
    if sub_variants:
        first_variant = next(iter(sub_variants.values()))
        assert first_variant.get("meta", {}).get("dims") is not None
    else:
        assert payload.get("meta", {}).get("subspace_error")


def test_plot_helpers(sample_records):
    plotly = pytest.importorskip("plotly")
    res = run_and_time(
        "compute_frontier_planes_all_modes (plot)",
        compute_frontier_planes_all_modes,
        sample_records,
        min_cluster_size=2,
    )
    try:
        fig = run_and_time(
            "plot_planes_with_point_lines",
            plot_planes_with_point_lines,
            res,
            records=sample_records,
            pair=None,
            show_planes=False,
            show_points=False,
            show=False,
            return_fig=True,
        )
    except ValueError as exc:
        if "planes_by_label vacío" in str(exc):
            pytest.skip("no planes available to plot for sample records")
        raise
    assert fig is not None

    X = np.vstack([r.x0 for r in sample_records])
    y = np.array([r.y0 for r in sample_records])
    fig2 = run_and_time(
        "plot_frontiers_implicit_interactive_v2",
        plot_frontiers_implicit_interactive_v2,
        sample_records,
        X,
        y=y,
        planes=None,
        planes_multi=None,
        quadrics=None,
        cubic_models=None,
        dims=(0, 1, 2),
        show=False,
        return_fig=True,
    )
    assert fig2 is not None

    fig3 = run_and_time(
        "plot_planes_with_point_lines (cloud)",
        plot_planes_with_point_lines,
        res,
        records=sample_records,
        X=X,
        y=y,
        pair=None,
        show_planes=False,
        show_points=False,
        show_cloud=True,
        show=False,
        return_fig=True,
    )
    assert fig3 is not None


def test_plot_cloud_bounds_include_negative_values():
    plotly = pytest.importorskip("plotly")
    import plotly.graph_objects as go

    class DummyRecord:
        def __init__(self):
            self.x0 = np.array([-1.0, -1.0, -1.0])
            self.x1 = np.array([-0.5, -0.25, -0.1])
            self.y0 = 0
            self.y1 = 1
            self.cp_x = np.empty((0, 0), float)
            self.cp_count = 0
            self.success = True
            self.final_score = 1.0

    record = DummyRecord()
    res = {
        (0, 1): {
            "planes_by_label": {
                0: [
                    {
                        "n": [0.5, 0.5, 0.7],
                        "b": 0.0,
                        "mu": [1.0, 1.0, 1.0],
                        "fit_error": {"inlier_rmse": 0.0},
                    }
                ]
            },
            "assignment": {
                "rec_indices": [0],
                "assigned_label": [0],
                "assigned_plane": [0],
            },
        }
    }

    X = np.array(
        [
            [-5.0, -4.0, -3.0],
            [-4.5, -3.5, -2.5],
            [-4.0, -3.0, -2.0],
        ]
    )
    y = np.array([0, 1, 0])

    fig = plot_planes_with_point_lines(
        res,
        records=[record],
        X=X,
        y=y,
        pair=(0, 1),
        show_planes=True,
        show_points=False,
        show_cloud=True,
        show=False,
        return_fig=True,
    )

    assert any(isinstance(tr, go.Surface) for tr in fig.data), "No se generó el plano esperado"

    def _finite_vals(data):
        arr = np.asarray(data, float).ravel()
        return arr[np.isfinite(arr)]

    all_x = np.concatenate([_finite_vals(tr.x) for tr in fig.data])
    all_y = np.concatenate([_finite_vals(tr.y) for tr in fig.data])
    all_z = np.concatenate([_finite_vals(tr.z) for tr in fig.data])

    assert all_x.min() <= X[:, 0].min()
    assert all_y.min() <= X[:, 1].min()
    assert all_z.min() <= X[:, 2].min()


def test_plot_normal_lines_use_unit_norm():
    plotly = pytest.importorskip("plotly")

    class DummyRecord:
        def __init__(self):
            self.x0 = np.array([-2.0, -1.0, 0.0])
            self.x1 = np.array([2.0, 1.0, 0.0])
            self.y0 = 0
            self.y1 = 1
            self.cp_x = np.empty((0, 0), float)
            self.cp_count = 0
            self.success = True
            self.final_score = 1.0

    record = DummyRecord()
    res = {
        (0, 1): {
            "planes_by_label": {
                0: [
                    {
                        "n": [2.0, 0.0, 0.0],
                        "b": -1.0,
                        "mu": [0.0, 0.0, 0.0],
                        "fit_error": {"inlier_rmse": 0.0},
                    }
                ]
            },
            "assignment": {
                "rec_indices": [0],
                "assigned_label": [0],
                "assigned_plane": [0],
            },
        }
    }

    fig = plot_planes_with_point_lines(
        res,
        records=[record],
        pair=(0, 1),
        show_planes=False,
        show_points=False,
        line_kind="normal",
        show=False,
        return_fig=True,
    )

    normal_traces = [tr for tr in fig.data if tr.name.startswith("Normal")]
    assert normal_traces, "No se generó la línea normal"
    xs = list(normal_traces[0].x)
    ys = list(normal_traces[0].y)
    v = record.x1 - record.x0
    u = v / np.linalg.norm(v)
    expected_frontier = record.x1 + 1e-3 * u
    assert_allclose([xs[0], xs[1]], [expected_frontier[0], 0.5], atol=1e-6)
    assert_allclose([ys[0], ys[1]], [expected_frontier[1], expected_frontier[1]], atol=1e-6)


def test_full_pipeline(dataset):
    X = dataset["X"]
    model = dataset["model"]
    cfg = DelDelConfig(
        segments_target=6,
        near_frac=0.6,
        k_near_base=2,
        k_far_per_i=1,
        q_near=0.2,
        q_far=0.8,
        margin_quantile=0.2,
        secant_iters=1,
        final_bisect=2,
        min_logit_gain=0.0,
        min_pair_margin_end=0.0,
        prob_swing_weight=0.5,
        use_jsd=False,
        random_state=0,
    )
    engine = DelDel(cfg)
    run_and_time("DelDel.fit", engine.fit, X, model)
    assert engine.records_

    res = run_and_time(
        "compute_frontier_planes_all_modes (pipeline)",
        compute_frontier_planes_all_modes,
        engine.records_,
        min_cluster_size=2,
    )
    assert res

    planes = run_and_time(
        "compute_frontier_planes_weighted (pipeline)",
        compute_frontier_planes_weighted,
        engine.records_,
        prefer_cp=False,
    )
    if not planes:
        pytest.skip("no weighted planes produced for this configuration")


def test_run_corner_pipeline_experiments(tmp_path):
    result = run_corner_pipeline_experiments(
        param_grid=[dict(min_cluster_size=2, max_models_per_round=2, max_depth=1)],
        dataset_kwargs=dict(n_per_cluster=80, random_state=1),
        csv_dir=tmp_path,
    )

    assert "valuable_" in result
    by_dim = result["valuable_"]["by_dim"]
    assert set(by_dim.keys()) == {1, 2, 3, 4}

    for dim in range(1, 5):
        classes = [entry["target_class"] for entry in by_dim[dim]]
        print(f"dim {dim}: {classes}")

    assert len(result["experiments"]) == 1
    exp = result["experiments"][0]
    assert "runtime_s" in exp and exp["runtime_s"] >= 0.0
    assert "total_pairs" in exp

    outputs = result.get("csv_outputs", {})
    expected_keys = {"experiments", "planes", "dimension_distributions"}
    assert expected_keys.issubset(outputs.keys())
    for key in expected_keys:
        path = Path(outputs[key])
        assert path.exists()
        with path.open() as fh:
            lines = [line for line in fh.readlines() if line.strip()]
        assert lines, f"CSV {key} should contain at least a header"


def test_run_low_dim_spaces_demo(tmp_path):
    artefacts = run_low_dim_spaces_demo(csv_dir=tmp_path)
    assert "valuable" in artefacts
    valuable = artefacts["valuable"]
    assert any(valuable.values()), "at least one rule should be discovered"

    csv_path = artefacts["csv_path"]
    assert csv_path is not None
    csv_file = Path(csv_path)
    assert csv_file.exists()
    with csv_file.open() as fh:
        rows = [line for line in fh.readlines() if line.strip()]
    assert len(rows) > 1, "CSV should contain header and data rows"
