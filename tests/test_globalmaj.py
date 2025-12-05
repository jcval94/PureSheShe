import pytest

np = pytest.importorskip("numpy")

from deldel.globalmaj import prune_and_orient_planes_unified_globalmaj


def test_prune_and_orient_basic_family_collapse():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
            [0.8, 0.0],
            [1.2, 0.0],
            [1.5, 0.0],
        ]
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    res = {
        (0, 1): {
            "planes_by_label": {
                0: [
                    {"n": [1.0, 0.0], "b": -0.75},
                    {"n": [1.0, 0.0], "b": -0.7499},
                ]
            }
        }
    }

    result = prune_and_orient_planes_unified_globalmaj(
        res,
        X,
        y,
        feature_names=["x0", "x1"],
        min_region_size=1,
        min_abs_diff=0.0,
        min_rel_lift=0.0,
        min_purity=0.0,
        max_k=1,
        min_recall=0.2,
        min_region_frac=0.1,
        diversity_additions=False,
        global_sides="good",
    )

    assert result["meta"]["total_families"] == 1
    candidates = result["candidates_global"]
    assert candidates, "se esperan candidatos globales"
    assert all("target_class" in c for c in candidates)
    assert {c["target_class"] for c in candidates} == {0, 1}

    per_plane = result["regions_global"]["per_plane"]
    assert per_plane, "se espera al menos una región global emitida"
    inequalities = {reg["inequality"]["pretty2D"] for reg in per_plane}
    assert inequalities == {"x0 ≤ 0.75", "x0 ≥ 0.75"}

    pair_payload = result["by_pair_augmented"][(0, 1)]
    assert pair_payload["winning_planes"], "el par debe conservar al menos un plano ganador"
    assert pair_payload["metrics_overall"]["balacc"] >= 0.9


def test_prune_and_orient_pair_filter_limits_pairs():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )
    y = np.array([0, 0, 1, 1, 2, 2])
    res = {
        (0, 1): {"planes_by_label": {0: [{"n": [1.0, 0.0], "b": -0.5}]}}
    }
    res[(1, 2)] = {"planes_by_label": {1: [{"n": [1.0, 0.0], "b": -1.5}]}}

    result = prune_and_orient_planes_unified_globalmaj(
        res,
        X,
        y,
        feature_names=["x0", "x1"],
        min_region_size=1,
        min_abs_diff=0.0,
        min_rel_lift=0.0,
        min_purity=0.0,
        max_k=1,
        min_recall=0.2,
        min_region_frac=0.1,
        diversity_additions=False,
        global_sides="good",
        pair_filter=[(1, 2)],
    )

    assert result["meta"]["total_pairs"] == 1
    assert list(result["by_pair_augmented"].keys()) == [(1, 2)]
    assert result["by_pair_augmented"][(1, 2)]["winning_planes"], "el filtro debe conservar planos para el par indicado"
    assert all(plane["origin_pair"] == (1, 2) for plane in result["winning_planes"])


def test_multi_class_candidates_can_be_disabled():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [1.2, 0.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    res = {
        (0, 1): {
            "planes_by_label": {
                0: [
                    {"n": [1.0, 0.0], "b": -0.5},
                    {"n": [0.0, 1.0], "b": 0.0},
                ],
                1: [{"n": [1.0, 0.0], "b": -1.1}],
            }
        }
    }

    common_kwargs = dict(
        X=X,
        y=y,
        feature_names=["x0", "x1"],
        min_region_size=1,
        min_abs_diff=-1.0,
        min_rel_lift=-1.0,
        min_purity=0.0,
        max_k=1,
        min_recall=0.0,
        min_region_frac=0.0,
        diversity_additions=False,
        global_sides="good",
    )

    result_multi = prune_and_orient_planes_unified_globalmaj(res, multi_class_candidates=True, **common_kwargs)
    result_single = prune_and_orient_planes_unified_globalmaj(res, multi_class_candidates=False, **common_kwargs)

    assert len(result_multi["candidates_global"]) >= len(result_single["candidates_global"])
    assert any(c.get("target_class") != c.get("best_class") for c in result_multi["candidates_global"])
    assert all(c.get("target_class") == c.get("best_class") for c in result_single["candidates_global"])
