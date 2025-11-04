# -*- coding: utf-8 -*-
"""Utilities to discover low-dimensional regions defined by affine half-spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import hashlib
import itertools
import math
import numpy as np


def _class_baseline(y: np.ndarray) -> Dict[int, float]:
    y = np.asarray(y, int).ravel()
    labels, counts = np.unique(y, return_counts=True)
    n = float(y.size) if y.size else 1.0
    return {int(lbl): float(cnt) / n for lbl, cnt in zip(labels, counts)}


def _feat_label(idx: int, feature_names: Optional[Sequence[str]]) -> str:
    if feature_names and 0 <= idx < len(feature_names):
        return str(feature_names[idx])
    return f"x{idx}"


def _format_interval_text(
    planes: Sequence[Tuple[np.ndarray, float, int]],
    dims: Tuple[int, ...],
    feature_names: Optional[Sequence[str]],
    tol: float = 1e-12,
) -> Tuple[str, List[str]]:
    if not planes:
        return "", []

    if len(dims) == 1:
        idx = dims[0]
        coef_low, coef_high = -math.inf, math.inf
        for n_eff, b_eff, side in planes:
            coef = float(n_eff[idx])
            if abs(coef) < tol:
                continue
            thr = -float(b_eff) / coef
            leq = (side >= 0 and coef > 0) or (side < 0 and coef < 0)
            if leq:
                coef_high = min(coef_high, thr)
            else:
                coef_low = max(coef_low, thr)
        name = _feat_label(idx, feature_names)
        if np.isfinite(coef_low) and np.isfinite(coef_high):
            text = f"{coef_low:.2f} ≤ {name} ≤ {coef_high:.2f}"
            return text, [text]
        if np.isfinite(coef_high):
            text = f"{name} ≤ {coef_high:.2f}"
            return text, [text]
        text = f"{name} ≥ {coef_low:.2f}"
        return text, [text]

    pieces: List[str] = []
    for n_eff, b_eff, side in planes:
        lhs = []
        for idx in dims:
            coef = float(n_eff[idx])
            if abs(coef) < tol:
                continue
            lhs.append(f"{coef:.2f}·{_feat_label(idx, feature_names)}")
        if not lhs:
            lhs_expr = f"{float(b_eff):.2f}"
        else:
            lhs_expr = " + ".join(lhs)
            if abs(b_eff) > tol:
                lhs_expr = f"{lhs_expr} {float(b_eff):.2f}"
        op = "≤" if side >= 0 else "≥"
        pieces.append(f"{lhs_expr} {op} 0")
    pieces = sorted(set(pieces))
    return "  AND  ".join(pieces), pieces


def _project_plane(
    n: np.ndarray,
    b: float,
    side: int,
    dims: Tuple[int, ...],
    mu: np.ndarray,
) -> Tuple[np.ndarray, float, int]:
    n = np.asarray(n, float).reshape(-1)
    mu = np.asarray(mu, float).reshape(-1)
    eff = np.zeros_like(n)
    eff[list(dims)] = n[list(dims)]
    complement = [idx for idx in range(n.size) if idx not in dims]
    b_eff = float(b)
    if complement:
        b_eff += float(n[complement] @ mu[complement])
    return eff, float(b_eff), int(side)


def _rule_id(cls: int, dims: Tuple[int, ...], plane_ids: Tuple[Any, ...], rule_text: str) -> str:
    payload = repr((int(cls), tuple(dims), tuple(map(str, plane_ids)), rule_text)).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:16]


def _mask_signature(mask: np.ndarray) -> str:
    return hashlib.md5(np.asarray(mask, bool).ravel().tobytes()).hexdigest()[:16]


@dataclass
class _PlaneRecord:
    plane_id: Any
    n: np.ndarray
    b: float
    side_for_cls: int
    origin_pair: Tuple[int, int]
    source: str


def _gather_planes(sel: Mapping[str, Any], cls: int) -> List[_PlaneRecord]:
    gathered: List[_PlaneRecord] = []
    by_pair = sel.get("by_pair_augmented", {}) or {}
    for (a, b), payload in by_pair.items():
        a, b = int(a), int(b)
        for idx, raw in enumerate(payload.get("winning_planes", []) or []):
            n = np.asarray(raw.get("n"), float)
            b0 = float(raw.get("b", 0.0))
            side = int(raw.get("side", +1))
            if cls == a:
                final_side = side
            elif cls == b:
                final_side = -side
            else:
                continue
            gathered.append(
                _PlaneRecord(
                    plane_id=raw.get("plane_id", f"pl{a}_{b}_{idx}"),
                    n=n,
                    b=b0,
                    side_for_cls=final_side,
                    origin_pair=(a, b),
                    source="pair",
                )
            )
    regions = sel.get("regions_global", {}).get("per_plane", []) or []
    for block in regions:
        if int(block.get("class_id", -999)) != int(cls):
            continue
        geom = block.get("geometry", {}) or {}
        n = np.asarray(geom.get("n"), float)
        b0 = float(geom.get("b", 0.0))
        side = int(geom.get("side", +1))
        gathered.append(
            _PlaneRecord(
                plane_id=block.get("plane_id"),
                n=n,
                b=b0,
                side_for_cls=side,
                origin_pair=tuple(block.get("origin_pair", (None, None))),
                source="global",
            )
        )
    return gathered


def _per_class_metrics(mask: np.ndarray, y: np.ndarray, labels: Sequence[int], baseline: Dict[int, float]):
    mask = np.asarray(mask, bool)
    y = np.asarray(y, int)
    sel = int(mask.sum())
    per_class: Dict[int, Dict[str, float]] = {}
    for cls in labels:
        cls = int(cls)
        pos = int(((y == cls) & mask).sum())
        total = int((y == cls).sum())
        prec = float(pos / sel) if sel else 0.0
        rec = float(pos / max(1, total))
        f1 = (2 * prec * rec) / (prec + rec + 1e-12)
        base = float(baseline.get(cls, 0.0))
        lift = (prec / base) if base > 0 else (math.inf if prec > 0 else 0.0)
        per_class[cls] = dict(
            pos=pos,
            total_pos=total,
            precision=prec,
            recall=rec,
            f1=f1,
            baseline=base,
            lift_precision=lift,
        )
    summary = dict(size=sel, frac=float(sel / max(1, y.size)))
    return per_class, summary


def _should_accept(
    metrics: Dict[str, float],
    baseline: float,
    *,
    size: int,
    min_support: int,
    min_rel_gain_f1: float,
    min_abs_gain_f1: float,
    min_lift_prec: float,
    min_abs_gain_prec: float,
    min_pos_in_region: int,
    pos_count: int,
) -> bool:
    if int(size) < int(min_support):
        return False
    if min_pos_in_region > 0 and pos_count < int(min_pos_in_region):
        return False
    base = float(baseline)
    f1_req = max(base * (1.0 + float(min_rel_gain_f1)), base + float(min_abs_gain_f1))
    prec_req = max(base * float(min_lift_prec), base + float(min_abs_gain_prec))
    return metrics["f1"] >= f1_req or metrics["precision"] >= prec_req


def find_low_dim_spaces(
    X: np.ndarray,
    y: np.ndarray,
    sel: Mapping[str, Any],
    *,
    feature_names: Optional[Sequence[str]] = None,
    max_planes_in_rule: int = 3,
    max_planes_per_pair: int = 2,
    max_rules_per_dim: int = 50,
    min_support: int = 30,
    min_rel_gain_f1: float = 0.25,
    min_abs_gain_f1: float = 0.05,
    min_lift_prec: float = 1.25,
    min_abs_gain_prec: float = 0.05,
    min_norm_in_dims: float = 1e-8,
    drop_vacuous_in_legend: bool = True,
    per_class_floor_topk: int = 3,
    consider_dims_up_to: Optional[int] = None,
    rng_seed: int = 0,
    projection_ref: str | np.ndarray = "class_mean",
    enable_logs: bool = False,
    max_log_records: int = 0,
    return_logs: bool = False,
    include_masks: bool = False,
    compute_relations: bool = False,
    use_global_regions: bool = False,
    global_force_accept_original: bool = False,
    global_lower_dim_enable: bool = False,
    heuristic_merge_enable: bool = False,
    p_merge: float = 0.0,
    heuristic_topk_per_region: int = 0,
    delta_min_f1: float = 0.0,
    delta_min_prec: float = 0.0,
    size_floor_ratio: float = 0.0,
    max_heuristic_trials_per_region: int = 0,
    max_heuristic_accepts_per_bucket: int = 0,
    min_pos_in_region: int = 0,
) -> Union[Dict[int, List[Dict[str, Any]]], Tuple[Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]]:
    X = np.asarray(X, float)
    y = np.asarray(y, int).ravel()
    n_samples, n_dims = X.shape

    if consider_dims_up_to is None:
        consider_dims = n_dims
    else:
        consider_dims = int(max(1, min(n_dims, consider_dims_up_to)))

    baseline = _class_baseline(y)
    labels = sorted(baseline.keys())
    mu_by_class = {int(cls): X[y == int(cls)].mean(axis=0) for cls in labels}
    mu_global = X.mean(axis=0)
    mu_zero = np.zeros(n_dims, float)

    def _mu_for(cls: int) -> np.ndarray:
        if isinstance(projection_ref, np.ndarray):
            return np.asarray(projection_ref, float).reshape(-1)
        if projection_ref == "global_mean":
            return mu_global
        if projection_ref == "zero":
            return mu_zero
        return mu_by_class[int(cls)]

    valuable: Dict[int, List[Dict[str, Any]]] = {k: [] for k in range(1, consider_dims + 1)}

    for cls in labels:
        planes_all = _gather_planes(sel, int(cls))
        by_pair: Dict[Tuple[int, int], List[_PlaneRecord]] = {}
        for p in planes_all:
            by_pair.setdefault(p.origin_pair, []).append(p)
        planes: List[_PlaneRecord] = []
        for pair, lst in by_pair.items():
            planes.extend(lst[: int(max_planes_per_pair)])

        if not planes:
            continue

        plane_masks_by_dims: Dict[Tuple[int, ...], List[Tuple[np.ndarray, Tuple[np.ndarray, float, int], _PlaneRecord]]] = {}
        mu_cls = _mu_for(int(cls))
        for dim_count in range(1, consider_dims + 1):
            for dims in itertools.combinations(range(n_dims), dim_count):
                dims = tuple(int(d) for d in dims)
                plane_data: List[Tuple[np.ndarray, Tuple[np.ndarray, float, int], _PlaneRecord]] = []
                for plane in planes:
                    n_eff, b_eff, side_eff = _project_plane(plane.n, plane.b, plane.side_for_cls, dims, mu_cls)
                    if np.linalg.norm(n_eff[list(dims)]) < float(min_norm_in_dims):
                        continue
                    mask = (X @ n_eff + b_eff) <= 0.0 if side_eff >= 0 else (X @ n_eff + b_eff) >= 0.0
                    if drop_vacuous_in_legend and mask.all():
                        continue
                    plane_data.append((mask, (n_eff, b_eff, side_eff), plane))
                if plane_data:
                    plane_masks_by_dims[dims] = plane_data

        rng = np.random.default_rng(int(rng_seed))
        for dims, plane_data in plane_masks_by_dims.items():
            indices = list(range(len(plane_data)))
            rng.shuffle(indices)
            for r in range(1, int(max_planes_in_rule) + 1):
                for combo in itertools.combinations(indices, r):
                    mask = np.ones(n_samples, dtype=bool)
                    planes_used = []
                    for idx in combo:
                        mask &= plane_data[idx][0]
                        planes_used.append(plane_data[idx][1])
                        if not mask.any():
                            break
                    if not mask.any():
                        continue

                    per_class, summary = _per_class_metrics(mask, y, labels, baseline)
                    metrics = per_class[int(cls)]
                    if not _should_accept(
                        metrics,
                        baseline[int(cls)],
                        size=summary["size"],
                        min_support=min_support,
                        min_rel_gain_f1=min_rel_gain_f1,
                        min_abs_gain_f1=min_abs_gain_f1,
                        min_lift_prec=min_lift_prec,
                        min_abs_gain_prec=min_abs_gain_prec,
                        min_pos_in_region=min_pos_in_region,
                        pos_count=int(per_class[int(cls)]["pos"]),
                    ):
                        continue

                    text, pieces = _format_interval_text(planes_used, dims, feature_names)
                    plane_ids = tuple(plane_data[idx][2].plane_id for idx in combo)
                    rec = dict(
                        region_id=f"rg_{len(dims)}d_c{int(cls)}_{_rule_id(int(cls), dims, plane_ids, text)}",
                        target_class=int(cls),
                        dims=dims,
                        plane_ids=plane_ids,
                        sources=tuple(
                            dict(
                                plane_id=plane_data[idx][2].plane_id,
                                origin_pair=plane_data[idx][2].origin_pair,
                                source=plane_data[idx][2].source,
                            )
                            for idx in combo
                        ),
                        rule_text=text,
                        rule_pieces=pieces,
                        metrics=dict(
                            size=int(summary["size"]),
                            precision=float(metrics["precision"]),
                            recall=float(metrics["recall"]),
                            f1=float(metrics["f1"]),
                            baseline=float(metrics["baseline"]),
                            lift_precision=float(metrics["lift_precision"]),
                        ),
                        metrics_per_class=per_class,
                        region_summary=summary,
                        projection_ref=str(projection_ref),
                        complexity=dict(num_dims=len(dims), num_planes=len(combo)),
                        is_floor=False,
                        generalizes=[],
                        specializes=[],
                        is_pareto=False,
                        family_id=None,
                        parent_id=None,
                        deltas_to_parent=None,
                        planes_used=[
                            dict(n=plane_data[idx][1][0].tolist(), b=float(plane_data[idx][1][1]), side=int(plane_data[idx][1][2]))
                            for idx in combo
                        ],
                    )
                    rec["mask_signature"] = _mask_signature(mask)
                    if include_masks:
                        rec["_mask"] = mask.astype(bool)
                    valuable[len(dims)].append(rec)

    for dim, rules in valuable.items():
        rules.sort(
            key=lambda r: (
                -r["metrics"]["f1"],
                -r["metrics"]["lift_precision"],
                -r["metrics"]["precision"],
                -r["metrics"]["size"],
            )
        )
        if len(rules) > int(max_rules_per_dim):
            valuable[dim] = rules[: int(max_rules_per_dim)]

    if return_logs:
        return valuable, []
    return valuable


__all__ = ["find_low_dim_spaces"]
