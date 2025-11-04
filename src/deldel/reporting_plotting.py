"""Utilities for textual reports and interactive plotting of DelDel regions.

This module groups two high level helpers that are commonly used after running
``find_low_dim_spaces`` and related routines:

``describe_regions_report``
    Builds a human friendly textual report for the discovered regions.

``plot_selected_regions_interactive``
    Generates an interactive Plotly figure that visualises planes and regions
    selected by the user.

The implementations included here are adaptations of internal notebooks that
were frequently re-used across experiments.  Centralising them in the package
helps to keep downstream projects cleaner while providing a single place to fix
bugs or extend features.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers for textual reports
# ---------------------------------------------------------------------------


def _flatten_valuable(valuable: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for _, records in (valuable or {}).items():
        out.extend(records or [])
    return out


def _cost_rec(region: Dict[str, Any]) -> float:
    complexity = region.get("complexity", {}) or {}
    return float(complexity.get("num_dims", 0)) + 0.5 * float(complexity.get("num_planes", 0))


def _fmt_float(value: Optional[float], nd: int = 3) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"
    return f"{float(value):.{nd}f}"


def _rank_key(region: Dict[str, Any]) -> Tuple[Any, ...]:
    metrics = region.get("metrics", {}) or {}
    return (
        0 if region.get("is_pareto", False) else 1,
        -float(metrics.get("f1", 0.0)),
        -float(metrics.get("lift_precision", 0.0)),
        -int(metrics.get("size", 0)),
        _cost_rec(region),
    )


def _fmt_sources(region: Dict[str, Any]) -> str:
    items: List[str] = []
    for src in (region.get("sources") or []):
        plane_id = src.get("plane_id", "—")
        origin_pair = tuple(src.get("origin_pair") or (None, None))
        source = src.get("source", "—")
        items.append(f"[{plane_id}] origen={origin_pair} src={source}")
    return "; ".join(items) if items else "—"


def _fmt_family(region: Dict[str, Any]) -> str:
    family: List[str] = []
    if region.get("is_pareto") is not None:
        family.append(f"Pareto: {'sí' if region.get('is_pareto') else 'no'}")
    if region.get("family_id"):
        family.append(f"family_id={region.get('family_id')}")
    if region.get("parent_id"):
        family.append(f"parent_id={region.get('parent_id')}")
    deltas = region.get("deltas_to_parent") or {}
    if deltas:
        family.append(
            "Δ vs parent (F1={} , Prec={} , Rec={})".format(
                _fmt_float(deltas.get("dF1")),
                _fmt_float(deltas.get("dPrecision")),
                _fmt_float(deltas.get("dRecall")),
            )
        )
    if region.get("generalizes"):
        family.append(f"generaliza={len(region.get('generalizes'))}")
    if region.get("specializes"):
        family.append(f"especializa={len(region.get('specializes'))}")
    return " | ".join(family) if family else "—"


_RE_INEQ = re.compile(
    r"x\s*(\d+)\s*(≥|<=|≤|>=|<|>)\s*([+\-−]?(?:\d+(?:\.\d*)?|\.\d+))",
    flags=re.IGNORECASE,
)


def _clean_ineq_piece(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"\s+", " ", text.strip())
    cleaned = re.sub(r"\+\s*-\s*", "- ", cleaned)
    cleaned = re.sub(r"(x\d)\s+([0-9]+(?:\.[0-9]+)?)\s*(≥|≤)\s*0", r"\1 + \2 \3 0", cleaned)
    cleaned = re.sub(r"(\d)\s+([0-9]+(?:\.[0-9]+)?)\s*(≥|≤)\s*0", r"\1 + \2 \3 0", cleaned)
    return cleaned


def _clean_rule_text_and_pieces(region: Dict[str, Any]) -> Tuple[str, List[str], bool]:
    changed = False
    raw_pieces = list(region.get("rule_pieces") or [])
    cleaned_pieces: List[str] = []
    for piece in raw_pieces:
        cleaned_piece = _clean_ineq_piece(piece)
        cleaned_pieces.append(cleaned_piece)
        if cleaned_piece != piece:
            changed = True

    raw_text = region.get("rule_text") or "  AND  ".join(raw_pieces)
    cleaned_text = _clean_ineq_piece(raw_text)
    if cleaned_text != raw_text:
        changed = True

    joined = "  AND  ".join(cleaned_pieces) if cleaned_pieces else cleaned_text
    final_text = joined if cleaned_pieces else cleaned_text

    return final_text, cleaned_pieces, changed


def _fmt_rule(
    region: Dict[str, Any],
    *,
    fix_rule_text: bool = True,
    show_original_if_changed: bool = True,
) -> str:
    dims = region.get("dims")
    plane_ids = region.get("plane_ids")
    parts = [
        f"Dimensiones: {tuple(dims) if dims is not None else '—'}",
        f"Planos: {tuple(plane_ids) if plane_ids is not None else '—'}",
    ]

    if fix_rule_text:
        clean_text, clean_pieces, changed = _clean_rule_text_and_pieces(region)
        parts.append(f"Regla: {clean_text}")
        if clean_pieces:
            parts.append(f"Cláusulas: {', '.join(clean_pieces)}")
        if changed and show_original_if_changed:
            raw_text = region.get("rule_text") or ""
            if raw_text and raw_text != clean_text:
                parts.append(f"Regla (original): {raw_text}")
    else:
        text = region.get("rule_text") or ""
        pieces = region.get("rule_pieces") or []
        if text:
            parts.append(f"Regla: {text}")
        if pieces:
            parts.append(f"Cláusulas: {', '.join(pieces)}")

    return "\n".join(parts)


def _region_size_and_frac(region: Dict[str, Any], dataset_size: Optional[int]) -> Tuple[int, Optional[float]]:
    summary = region.get("region_summary") or {}
    size = int(summary.get("size", region.get("metrics", {}).get("size", 0)))
    frac = summary.get("frac")
    if frac is None and dataset_size and dataset_size > 0:
        frac = float(size) / float(dataset_size)
    return size, frac if frac is None else float(frac)


def _fmt_metrics_header(region: Dict[str, Any], dataset_size: Optional[int] = None) -> str:
    size, frac = _region_size_and_frac(region, dataset_size)
    pct = f" ({frac*100:.2f}%)" if isinstance(frac, float) else ""
    metrics = region.get("metrics", {}) or {}
    return (
        f"Tamaño región: {size}{pct}\n"
        f"Precisión (clase objetivo): {_fmt_float(metrics.get('precision'))} | "
        f"Recall: {_fmt_float(metrics.get('recall'))} | "
        f"F1: {_fmt_float(metrics.get('f1'))}\n"
        f"Baseline clase: {_fmt_float(metrics.get('baseline'))} | "
        f"Lift precisión: {_fmt_float(metrics.get('lift_precision'))}"
    )


def _normalize_metrics_per_class(region: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    raw = region.get("metrics_per_class") or {}
    normalised: Dict[int, Dict[str, Any]] = {}
    for key, value in raw.items():
        try:
            class_id = int(key)
        except Exception:
            continue
        normalised[class_id] = value or {}
    return normalised


def _fmt_metrics_per_class(region: Dict[str, Any], nd: int = 3) -> str:
    per_class = _normalize_metrics_per_class(region)
    if not per_class:
        return "(sin métricas por clase)"
    lines = ["— Métricas por clase —"]
    for class_id in sorted(per_class.keys()):
        metrics = per_class[class_id] or {}
        positives = int(metrics.get("pos", 0))
        total_pos = int(metrics.get("total_pos", 0))
        lines.append(
            f"  Clase {class_id}: "
            f"Prec={_fmt_float(metrics.get('precision'), nd)}  |  "
            f"Rec={_fmt_float(metrics.get('recall'), nd)}  |  "
            f"F1={_fmt_float(metrics.get('f1'), nd)}  |  "
            f"lift={_fmt_float(metrics.get('lift_precision'), nd)}  |  "
            f"pos={positives}/{total_pos}  |  baseline={_fmt_float(metrics.get('baseline'), nd)}"
        )
    return "\n".join(lines)


def _class_mix_stats(region: Dict[str, Any]) -> Tuple[float, float, List[Tuple[int, float, float, float]]]:
    per_class = _normalize_metrics_per_class(region)
    size, _ = _region_size_and_frac(region, dataset_size=None)
    if not per_class or size <= 0:
        return 0.0, 0.0, []

    mix: List[Tuple[int, float, float, float]] = []
    for class_id, metrics in per_class.items():
        positives = float(metrics.get("pos", 0))
        share = positives / float(size) if size > 0 else 0.0
        mix.append(
            (
                class_id,
                share,
                float(metrics.get("recall") or 0.0),
                float(metrics.get("lift_precision") or 0.0),
            )
        )

    shares = [share for _, share, _, _ in mix if share > 0]
    if not shares:
        return 0.0, 0.0, []

    purity = max(shares)
    entropy = -sum(p * math.log(p, 2) for p in shares)
    mix.sort(key=lambda item: item[1], reverse=True)
    return purity, entropy, mix[:3]


def _fmt_quality_block(region: Dict[str, Any]) -> str:
    purity, entropy, top_classes = _class_mix_stats(region)
    if not top_classes:
        return "(sin distribución por clase)"
    lines = [
        f"Pureza: {_fmt_float(purity)}  |  Entropía: {_fmt_float(entropy)} bits",
        "Top clases en la región (por proporción dentro de la región):",
    ]
    for class_id, share, recall, lift in top_classes:
        lines.append(
            f"  Clase {class_id}: share={_fmt_float(share)}  |  recall={_fmt_float(recall)}  |  lift={_fmt_float(lift)}"
        )
    return "\n".join(lines)


def _find_by_region_id(valuable: Dict[int, List[Dict[str, Any]]], region_id: str) -> Optional[Dict[str, Any]]:
    for _, records in (valuable or {}).items():
        for region in (records or []):
            if region.get("region_id") == region_id:
                return region
    return None


def _group_by_class(valuable: Dict[int, List[Dict[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
    by_class: Dict[int, List[Dict[str, Any]]] = {}
    for region in _flatten_valuable(valuable):
        class_id = int(region.get("target_class"))
        by_class.setdefault(class_id, []).append(region)
    for class_id in list(by_class.keys()):
        by_class[class_id].sort(key=_rank_key)
    return by_class


def describe_regions_report(
    valuable: Dict[int, List[Dict[str, Any]]],
    *,
    region_id: Optional[str] = None,
    top_per_class: int = 2,
    dataset_size: Optional[int] = None,
    max_rule_text_chars: int = 220,
    show_per_class_in_top: bool = False,
    fix_rule_text: bool = True,
    show_original_if_changed: bool = True,
) -> str:
    """Generate a textual report describing the discovered regions."""

    if region_id:
        region = _find_by_region_id(valuable, region_id)
        if region is None:
            return f"▶ No se encontró la región con id '{region_id}'."
        lines: List[str] = [
            "====== FICHA DE REGIÓN ======",
            f"ID: {region.get('region_id', '—')}",
            f"Clase objetivo: {region.get('target_class', '—')}",
            f"Proyección (ref): {region.get('projection_ref', '—')}",
            (
                "Complejidad: num_dims={num_dims}, num_planes={num_planes}, coste={coste}".format(
                    num_dims=region.get("complexity", {}).get("num_dims", "—"),
                    num_planes=region.get("complexity", {}).get("num_planes", "—"),
                    coste=_fmt_float(_cost_rec(region)),
                )
            ),
            (
                "¿Pareto?: {pareto}  |  ¿Floor?: {floor}".format(
                    pareto="sí" if region.get("is_pareto") else "no",
                    floor="sí" if region.get("is_floor") else "no",
                )
            ),
            _fmt_metrics_header(region, dataset_size),
            _fmt_quality_block(region),
            _fmt_metrics_per_class(region),
            _fmt_rule(region, fix_rule_text=fix_rule_text, show_original_if_changed=show_original_if_changed),
            f"Fuentes: {_fmt_sources(region)}",
            f"Relaciones/Familia: {_fmt_family(region)}",
        ]
        return "\n".join(lines)

    grouped = _group_by_class(valuable)
    if not grouped:
        return "No hay regiones disponibles."

    k_top = max(1, int(top_per_class))
    lines = ["====== TOP REGIONES POR CLASE ======"]
    for class_id in sorted(grouped.keys()):
        lines.append(f"\n--- Clase {class_id} ---")
        top_regions = grouped[class_id][:k_top]
        if not top_regions:
            lines.append("  (sin regiones)")
            continue

        for idx, region in enumerate(top_regions, 1):
            metrics = region.get("metrics", {}) or {}
            size, frac = _region_size_and_frac(region, dataset_size)
            pct = f"{frac*100:.2f}%" if isinstance(frac, float) else "—"
            lines.append(
                "  #{idx}  id={region_id}  |  dims={dims}  |  coste={coste}  |  Pareto={pareto}".format(
                    idx=idx,
                    region_id=region.get("region_id", "—"),
                    dims=tuple(region.get("dims") or []),
                    coste=_fmt_float(_cost_rec(region)),
                    pareto="sí" if region.get("is_pareto") else "no",
                )
            )
            lines.append(
                "      F1={f1}  |  Prec={precision}  |  Rec={recall}  |  LiftPrec={lift}  |  size={size} ({pct})".format(
                    f1=_fmt_float(metrics.get("f1")),
                    precision=_fmt_float(metrics.get("precision")),
                    recall=_fmt_float(metrics.get("recall")),
                    lift=_fmt_float(metrics.get("lift_precision")),
                    size=size,
                    pct=pct,
                )
            )
            purity, entropy, top_mix = _class_mix_stats(region)
            if top_mix:
                short = ", ".join([f"c{cls}:sh={_fmt_float(share)}" for cls, share, _, _ in top_mix[:2]])
                lines.append(
                    f"      Calidad: pureza={_fmt_float(purity)} | H={_fmt_float(entropy)} | {short}"
                )
            if fix_rule_text:
                clean_text, _, _ = _clean_rule_text_and_pieces(region)
                rule_text = clean_text or (region.get("rule_text") or "")
            else:
                rule_text = region.get("rule_text") or ""
            if not rule_text:
                pieces = region.get("rule_pieces") or []
                rule_text = " AND ".join(pieces[:3]) + (" ..." if len(pieces) > 3 else "")
            if max_rule_text_chars and len(rule_text) > max_rule_text_chars:
                rule_text = rule_text[: max_rule_text_chars - 3] + "..."
            lines.append(f"      Regla: {rule_text}")
            if show_per_class_in_top:
                block = _fmt_metrics_per_class(region)
                indented = "\n".join(("      " + row) for row in block.splitlines())
                lines.append(indented)

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Interactive plotting helper
# ---------------------------------------------------------------------------


def plot_selected_regions_interactive(
    sel_aug: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    selected_plane_ids: Optional[List[str]] = None,
    selected_region_ids: Optional[List[str]] = None,
    rules_to_show: Optional[List[Dict[str, Any]]] = None,
    valuable: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    dims: Tuple[int, ...] = (0, 1, 2),
    dims_options: Optional[List[Tuple[int, ...]]] = None,
    feature_names: Optional[List[str]] = None,
    top_k_planes_per_dims: int = 6,
    min_dim_alignment: float = 0.45,
    renderer: Optional[str] = None,
    title: str = "Regiones y Planos — Interactivo",
    show: bool = True,
    return_fig: bool = False,
    extend: float = 0.06,
    points_opacity: float = 0.6,
    point_size: int = 3,
    scatter_sample_per_class: int = 900,
    region_opacity_3d: float = 0.18,
    region_opacity_2d: float = 0.30,
    region_opacity_1d: float = 0.40,
    grid_res_2d: int = 220,
    grid_res_1d: int = 1200,
    volume_res_3d: int = 36,
    color_planes: str = "#1f77b4",
    color_regions: str = "#9467bd",
    color_rules: str = "#2ca02c",
    slice_method: str = "mean",
    slice_values: Optional[Dict[Any, float]] = None,
    rng_seed: Optional[int] = 1337,
):
    """Interactive Plotly visualisation of planes and regions."""

    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    if renderer:
        try:
            pio.renderers.default = renderer
        except Exception:
            pass

    X = np.asarray(X, float)
    y = np.asarray(y, int).ravel()
    num_samples, num_dims = X.shape
    eps = 1e-9

    if rng_seed is not None:
        rng = np.random.default_rng(int(rng_seed))

        def _choice(index: np.ndarray, size: int) -> np.ndarray:
            return rng.choice(index, size=size, replace=False)

    else:

        def _choice(index: np.ndarray, size: int) -> np.ndarray:
            return np.random.choice(index, size=size, replace=False)

    def _feat(dim_index: int) -> str:
        if feature_names and 0 <= dim_index < len(feature_names):
            return feature_names[dim_index]
        return f"x{dim_index}"

    def _fmt2(value: float) -> str:
        return f"{float(value):.2f}"

    def _ensure_rgb(color: str) -> str:
        color = str(color)
        if color.startswith("#") and len(color) == 7:
            red = int(color[1:3], 16)
            green = int(color[3:5], 16)
            blue = int(color[5:7], 16)
            return f"rgb({red},{green},{blue})"
        if color.startswith("rgba"):
            inner = color[color.find("(") + 1 : color.find(")")]
            parts = [component.strip() for component in inner.split(",")][0:3]
            if len(parts) >= 3:
                red, green, blue = parts[:3]
                return f"rgb({red},{green},{blue})"
        return color

    def _colorscale(color: str) -> List[List[Any]]:
        rgb = _ensure_rgb(color)
        return [[0.0, rgb], [1.0, rgb]]

    def _parse_1d_rule_text(rule_text: Optional[str], expected_dim: Optional[int] = None) -> Optional[Dict[str, Any]]:
        if not isinstance(rule_text, str):
            return None
        content = rule_text.replace("−", "-")
        found = _RE_INEQ.findall(content)
        if not found:
            return None
        lows: List[float] = []
        ups: List[float] = []
        dim_seen: List[int] = []
        for dim_str, op, value_str in found:
            dim = int(dim_str)
            value = float(value_str.replace("−", "-"))
            if dim not in dim_seen:
                dim_seen.append(dim)
            if op in ("≥", ">=", ">"):
                lows.append(value)
            elif op in ("≤", "<=", "<"):
                ups.append(value)
        low = max(lows) if lows else None
        up = min(ups) if ups else None
        if (low is not None) and (up is not None) and (up < low):
            low, up = up, low
        dim_final = dim_seen[0] if dim_seen else expected_dim
        return {"dim": dim_final, "low": low, "up": up}

    by_pair = sel_aug.get("by_pair_augmented", {}) or {}
    plane_index: Dict[str, Dict[str, Any]] = {}
    for (a, b), entry in by_pair.items():
        for plane in (entry.get("winning_planes", []) or []):
            if plane.get("plane_id") is not None:
                plane_index[plane["plane_id"]] = dict(plane, origin_pair=(int(a), int(b)))
        for plane in (entry.get("other_planes", []) or []):
            if plane.get("plane_id") is not None:
                plane_index[plane["plane_id"]] = dict(plane, origin_pair=(int(a), int(b)))

    region_index: Dict[str, Dict[str, Any]] = {}
    for region in sel_aug.get("regions_global", {}).get("per_plane", []) or []:
        region_id = region.get("region_id")
        if region_id:
            region_index[region_id] = dict(region)

    valuable_region_index: Dict[str, Dict[str, Any]] = {}
    if isinstance(valuable, dict):
        for _, region_list in valuable.items():
            for valuable_region in region_list or []:
                region_id = valuable_region.get("region_id")
                if region_id:
                    valuable_region_index[region_id] = valuable_region

    selected_plane_ids = list(selected_plane_ids or [])
    selected_region_ids = list(selected_region_ids or [])
    rules_to_show = list(rules_to_show or [])

    plane_cls_from_valuable: Dict[str, int] = {}
    if isinstance(valuable, dict):
        for _, region_list in valuable.items():
            for valuable_region in region_list or []:
                cls = int(valuable_region.get("target_class", -1))
                for source in valuable_region.get("sources", []):
                    plane_id = source.get("plane_id")
                    if plane_id is not None:
                        plane_cls_from_valuable.setdefault(str(plane_id), cls)

    def _sign_plane(plane: Dict[str, Any]) -> int:
        ineq = (plane.get("inequality", {}) or {}).get("general", "")
        content = str(ineq).replace(" ", "")
        if ("<=0" in content) or ("≤0" in content):
            return +1
        if (">=0" in content) or ("≥0" in content):
            return -1
        return +1 if int(plane.get("side", +1)) >= 0 else -1

    def _global_cls(plane: Dict[str, Any]) -> Optional[int]:
        global_for = plane.get("global_for")
        if global_for is None:
            opposite = plane.get("opposite_side_eval", {})
            if isinstance(opposite, dict) and opposite.get("meets_majority_rule", False):
                global_for = opposite.get("majority_class")
        if global_for is None:
            return None
        return int(global_for)

    def _plane_target_class(plane_id: str, plane: Dict[str, Any]) -> Optional[int]:
        if plane_id in plane_cls_from_valuable:
            return int(plane_cls_from_valuable[plane_id])
        if "label" in plane and plane["label"] is not None:
            try:
                return int(plane["label"])
            except Exception:
                pass
        global_class = _global_cls(plane)
        if global_class is not None:
            return int(global_class)
        return None

    def _restrict(normal: np.ndarray, bias: float, mu_plane: Optional[np.ndarray], dims_sel: Tuple[int, ...]) -> Tuple[np.ndarray, float]:
        dims_sel = tuple(int(i) for i in dims_sel)
        if slice_method == "median":
            mu = np.nanmedian(X, axis=0)
        elif slice_method == "plane_mu" and mu_plane is not None:
            mu = np.asarray(mu_plane, float)
        else:
            mu = np.nanmean(X, axis=0)
        mu = np.asarray(mu, float).reshape(-1)
        if isinstance(slice_values, dict):
            for key, value in slice_values.items():
                if isinstance(key, str) and key.startswith("x"):
                    try:
                        dim_idx = int(key[1:])
                    except Exception:
                        continue
                else:
                    dim_idx = int(key)
                if dim_idx not in dims_sel and 0 <= dim_idx < num_dims:
                    mu[dim_idx] = float(value)
        other_dims = [idx for idx in range(num_dims) if idx not in dims_sel]
        normal_full = np.asarray(normal, float).reshape(-1)
        bias_eff = float(bias) + (
            float(normal_full[other_dims] @ mu[other_dims]) if other_dims else 0.0
        )
        normal_eff = np.zeros(num_dims)
        for idx in dims_sel:
            normal_eff[idx] = normal_full[idx]
        return normal_eff, bias_eff

    def _top_dims_from_n(normal: np.ndarray, k: int) -> Tuple[int, ...]:
        normal = np.asarray(normal, float).ravel()
        idx = np.argsort(-np.abs(normal))[: max(1, min(k, len(normal)))]
        return tuple(int(i) for i in np.sort(idx))

    def _dims_from_rule_sources(region: Dict[str, Any], k: int = 3) -> Tuple[int, ...]:
        acc = np.zeros(num_dims, float)
        for source in region.get("sources", []) or []:
            plane_id = source.get("plane_id")
            plane = plane_index.get(plane_id)
            if plane is None:
                continue
            acc += np.abs(np.asarray(plane["n"], float).ravel())
        if not np.any(acc):
            dims_region = tuple(region.get("dims", ()))
            return tuple(int(i) for i in sorted(dims_region[:k]))
        idx = np.argsort(-acc)[: max(1, min(k, num_dims))]
        return tuple(int(i) for i in np.sort(idx))

    def _orient_sign_to_class(normal_eff: np.ndarray, bias_eff: float, dims_sel: Tuple[int, ...], target_class: int) -> int:
        class_idx = np.flatnonzero(y == int(target_class))
        if class_idx.size == 0:
            return +1
        points = X[class_idx][:, list(dims_sel)]
        values = bias_eff + np.sum(
            np.array([normal_eff[d] * points[:, i] for i, d in enumerate(dims_sel)]),
            axis=0,
        )
        good_pos = np.count_nonzero(+values <= 0)
        good_neg = np.count_nonzero(-values <= 0)
        return +1 if good_pos >= good_neg else -1

    def _absorb_sign(normal_eff: np.ndarray, bias_eff: float, sign: int) -> Tuple[np.ndarray, float]:
        normal_new = np.array(normal_eff, float)
        normal_new *= float(sign)
        bias_new = float(bias_eff) * float(sign)
        return normal_new, bias_new

    def _restrict_1d(normal: np.ndarray, bias: float, mu: Optional[np.ndarray], primary_dim: int) -> Tuple[np.ndarray, float]:
        normal = np.asarray(normal, float).reshape(-1)
        if slice_method == "median":
            mu_base = np.nanmedian(X, axis=0)
        elif slice_method == "plane_mu" and mu is not None:
            mu_base = np.asarray(mu, float)
        else:
            mu_base = np.nanmean(X, axis=0)
        other_dims = [j for j in range(num_dims) if j != int(primary_dim)]
        bias_eff = float(bias) + (
            float(normal[other_dims] @ mu_base[other_dims]) if other_dims else 0.0
        )
        normal_eff = np.zeros(num_dims, float)
        normal_eff[int(primary_dim)] = float(normal[int(primary_dim)])
        return normal_eff, bias_eff

    def _force_1d_on_primary(normal_eff: np.ndarray, dims_sel: Tuple[int, ...], primary_dim: int) -> np.ndarray:
        normal_new = np.array(normal_eff, float)
        for dim in dims_sel:
            if int(dim) != int(primary_dim):
                normal_new[int(dim)] = 0.0
        return normal_new

    def _pick_filler_dim(primary: int) -> int:
        others = [j for j in range(num_dims) if j != primary]
        if not others:
            return primary
        ranges = [
            (j, float(np.nanmax(X[:, j]) - np.nanmin(X[:, j])))
            for j in others
        ]
        filler = max(ranges, key=lambda pair: pair[1])[0]
        return int(filler)

    palette = px.colors.qualitative.Plotly
    classes_all = sorted(np.unique(y))
    class_colors = {int(cls): palette[i % len(palette)] for i, cls in enumerate(classes_all)}

    user_dims_given = dims is not None
    base_k = len(dims) if user_dims_given else min(3, num_dims)
    blocks: List[Tuple[int, ...]] = []

    def _maybe_extend_1d_to_2d(dims_tuple: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(dims_tuple) == 1:
            primary = int(dims_tuple[0])
            filler = _pick_filler_dim(primary)
            if filler == primary and num_dims >= 2:
                filler = (primary + 1) % num_dims
            return (primary, filler)
        return tuple(dims_tuple)

    if selected_region_ids:
        for region_id in selected_region_ids:
            if region_id in region_index:
                region_global = region_index[region_id]
                normal = (region_global.get("geometry", {}) or {}).get("n")
                if normal is None:
                    continue
                dims_region = _top_dims_from_n(normal, base_k)
            elif region_id in valuable_region_index:
                valuable_region = valuable_region_index[region_id]
                dims_region = tuple(int(i) for i in valuable_region.get("dims", ()))
                if not dims_region:
                    continue
                if len(dims_region) > 3:
                    dims_region = _dims_from_rule_sources(valuable_region, k=3)
            else:
                continue
            dims_region = _maybe_extend_1d_to_2d(dims_region)
            if dims_region not in blocks:
                blocks.append(dims_region)
        if not blocks:
            base = tuple(int(i) for i in dims) if user_dims_given else tuple(range(min(3, num_dims)))
            blocks = [_maybe_extend_1d_to_2d(base)]
    elif selected_plane_ids:
        for plane_id in selected_plane_ids:
            plane = plane_index.get(plane_id)
            if not plane:
                continue
            dims_plane = _maybe_extend_1d_to_2d(_top_dims_from_n(plane["n"], base_k))
            if dims_plane not in blocks:
                blocks.append(dims_plane)
        if dims_options:
            for option in dims_options:
                option_tuple = _maybe_extend_1d_to_2d(tuple(int(i) for i in option))
                if option_tuple not in blocks:
                    blocks.append(option_tuple)
    else:
        if dims_options:
            for option in dims_options:
                option_tuple = _maybe_extend_1d_to_2d(tuple(int(i) for i in option))
                if option_tuple not in blocks:
                    blocks.append(option_tuple)
        elif user_dims_given:
            blocks = [_maybe_extend_1d_to_2d(tuple(int(i) for i in dims))]
        else:
            blocks = [_maybe_extend_1d_to_2d(tuple(range(min(3, num_dims))))]

    def _rules_for_dims_from_valuable(dims_sel: Tuple[int, ...], top: int = 3) -> List[Dict[str, Any]]:
        if not isinstance(valuable, dict):
            return []
        dim_k = len(dims_sel)
        pool: List[Dict[str, Any]] = []
        if dim_k == 2:
            primary = dims_sel[0]
            for region in valuable.get(2, []) or []:
                if tuple(region.get("dims", ())) == tuple(dims_sel):
                    pool.append(region)
            for region in valuable.get(1, []) or []:
                dims_region = tuple(region.get("dims", ()))
                if len(dims_region) == 1 and int(dims_region[0]) == int(primary):
                    pool.append(region)
        else:
            for region in valuable.get(dim_k, []) or []:
                if tuple(region.get("dims", ())) == tuple(dims_sel):
                    pool.append(region)
        if not pool:
            return []
        pool.sort(
            key=lambda region: (
                region.get("metrics", {}).get("f1", 0.0),
                region.get("metrics", {}).get("precision", 0.0),
                region.get("metrics", {}).get("size", 0),
            ),
            reverse=True,
        )
        return pool[:top]

    grid_cache: Dict[Tuple[Any, ...], Any] = {}

    def _grid_xy(lo: np.ndarray, hi: np.ndarray, res: int):
        key = ("xy", float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), int(res))
        if key not in grid_cache:
            xs = np.linspace(lo[0], hi[0], res)
            ys = np.linspace(lo[1], hi[1], res)
            Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
            grid_cache[key] = (xs, ys, Xg, Yg)
        return grid_cache[key]

    def _grid_xyz(lo: np.ndarray, hi: np.ndarray, res: int):
        key = (
            "xyz",
            float(lo[0]),
            float(hi[0]),
            float(lo[1]),
            float(hi[1]),
            float(lo[2]),
            float(hi[2]),
            int(res),
        )
        if key not in grid_cache:
            xs = np.linspace(lo[0], hi[0], res)
            ys = np.linspace(lo[1], hi[1], res)
            zs = np.linspace(lo[2], hi[2], res)
            Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")
            grid_cache[key] = (xs, ys, zs, Xg, Yg, Zg)
        return grid_cache[key]

    def _bounds(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.nanmin(points, axis=0)
        hi = np.nanmax(points, axis=0)
        span = np.maximum(hi - lo, 1e-12)
        padding = float(extend) * span
        return lo - padding, hi + padding

    def _clip_line_to_rect_2d(a: float, b: float, c: float, lo: np.ndarray, hi: np.ndarray):
        eps_line = 1e-12
        points: List[Tuple[float, float]] = []
        if abs(b) > eps_line:
            y0 = -(a * lo[0] + c) / b
            if lo[1] - 1e-9 <= y0 <= hi[1] + 1e-9:
                points.append((lo[0], y0))
            y1 = -(a * hi[0] + c) / b
            if lo[1] - 1e-9 <= y1 <= hi[1] + 1e-9:
                points.append((hi[0], y1))
        if abs(a) > eps_line:
            x0 = -(b * lo[1] + c) / a
            if lo[0] - 1e-9 <= x0 <= hi[0] + 1e-9:
                points.append((x0, lo[1]))
            x1 = -(b * hi[1] + c) / a
            if lo[0] - 1e-9 <= x1 <= hi[0] + 1e-9:
                points.append((x1, hi[1]))
        unique: List[Tuple[float, float]] = []
        for point in points:
            if all((abs(point[0] - other[0]) > 1e-7 or abs(point[1] - other[1]) > 1e-7) for other in unique):
                unique.append(point)
        if len(unique) < 2:
            return None
        import itertools as _itertools

        best = (None, None, -1.0)
        for u, v in _itertools.combinations(unique, 2):
            dist = (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2
            if dist > best[2]:
                best = (u, v, dist)
        return best[0], best[1]

    def _add_mask_heatmap_2d(data: List[Any], xs: np.ndarray, ys: np.ndarray, mask_bool: np.ndarray, color: str, name: str):
        z = np.where(mask_bool, 1.0, np.nan)
        data.append(
            go.Heatmap(
                x=xs,
                y=ys,
                z=z,
                colorscale=_colorscale(color),
                opacity=float(region_opacity_2d),
                showscale=False,
                hoverinfo="skip",
                zmin=0.0,
                zmax=1.0,
                name=name,
                showlegend=True,
                zsmooth="best",
            )
        )

    def _render_halfspace_3d(
        data: List[Any],
        dims_sel: Tuple[int, ...],
        normal_eff: np.ndarray,
        bias_eff: float,
        sign: int,
        color: str,
        name: str,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> None:
        def _box_vertices(lo3: np.ndarray, hi3: np.ndarray) -> np.ndarray:
            x0, y0, z0 = lo3
            x1, y1, z1 = hi3
            return np.array(
                [
                    [x0, y0, z0],
                    [x1, y0, z0],
                    [x1, y1, z0],
                    [x0, y1, z0],
                    [x0, y0, z1],
                    [x1, y0, z1],
                    [x1, y1, z1],
                    [x0, y1, z1],
                ],
                float,
            )

        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        vertices = _box_vertices(lo, hi)
        points: List[np.ndarray] = []
        normal_3d = np.array([normal_eff[dims_sel[0]], normal_eff[dims_sel[1]], normal_eff[dims_sel[2]]], float)
        for idx0, idx1 in edges:
            p0 = vertices[idx0]
            p1 = vertices[idx1]
            d0 = float(normal_3d @ p0 + bias_eff)
            d1 = float(normal_3d @ p1 + bias_eff)
            denominator = d1 - d0
            if abs(denominator) < 1e-12:
                if abs(d0) < 1e-12 and abs(d1) < 1e-12:
                    points += [p0, p1]
                continue
            t = -d0 / denominator
            if -1e-9 <= t <= 1 + 1e-9:
                t = min(max(t, 0.0), 1.0)
                points.append(p0 + t * (p1 - p0))
        unique: List[np.ndarray] = []
        for point in points:
            if all(np.linalg.norm(point - other) > 1e-7 for other in unique):
                unique.append(point)
        polygon = np.array(unique, float)
        if polygon.shape[0] >= 3:
            i_idx: List[int] = []
            j_idx: List[int] = []
            k_idx: List[int] = []
            for tri in range(1, polygon.shape[0] - 1):
                i_idx.append(0)
                j_idx.append(tri)
                k_idx.append(tri + 1)
            data.append(
                go.Mesh3d(
                    x=polygon[:, 0],
                    y=polygon[:, 1],
                    z=polygon[:, 2],
                    i=i_idx,
                    j=j_idx,
                    k=k_idx,
                    color=_ensure_rgb(color),
                    opacity=0.35,
                    showlegend=False,
                    name=name,
                    hoverinfo="skip",
                )
            )

        xs, ys, zs, Xg, Yg, Zg = _grid_xyz(lo, hi, int(volume_res_3d))
        field = sign * (
            normal_eff[dims_sel[0]] * Xg
            + normal_eff[dims_sel[1]] * Yg
            + normal_eff[dims_sel[2]] * Zg
            + bias_eff
        )
        mask = (field <= eps).astype(np.uint8)
        data.append(
            go.Volume(
                x=Xg.flatten(),
                y=Yg.flatten(),
                z=Zg.flatten(),
                value=mask.flatten(),
                isomin=0.5,
                isomax=1.0,
                surface_count=1,
                opacity=float(region_opacity_3d),
                colorscale=_colorscale(color),
                showscale=False,
                name=name,
                showlegend=True,
                hovertemplate=(
                    f"{_fmt2(normal_eff[dims_sel[0]])}·{_feat(dims_sel[0])} + "
                    f"{_fmt2(normal_eff[dims_sel[1]])}·{_feat(dims_sel[1])} + "
                    f"{_fmt2(normal_eff[dims_sel[2]])}·{_feat(dims_sel[2])} + "
                    f"{_fmt2(bias_eff)} ≤ 0<extra></extra>"
                ),
            )
        )

    def _render_halfspace_2d(
        data: List[Any],
        dims_sel: Tuple[int, ...],
        normal_eff: np.ndarray,
        bias_eff: float,
        sign: int,
        color: str,
        name: str,
        lo: np.ndarray,
        hi: np.ndarray,
        draw_boundary: bool = True,
    ) -> None:
        xs, ys, Xg, Yg = _grid_xy(lo, hi, int(grid_res_2d))
        field = sign * (normal_eff[dims_sel[0]] * Xg + normal_eff[dims_sel[1]] * Yg + bias_eff)
        mask = field <= eps
        _add_mask_heatmap_2d(data, xs, ys, mask, color, name)
        if draw_boundary:
            a = float(normal_eff[dims_sel[0]])
            b = float(normal_eff[dims_sel[1]])
            c = float(bias_eff)
            segment = _clip_line_to_rect_2d(a, b, c, lo, hi)
            if segment is not None:
                (x0, y0), (x1, y1) = segment
                data.append(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(width=2, color=_ensure_rgb(color)),
                        name=name,
                        showlegend=False,
                        hovertemplate=(
                            f"{_fmt2(a)}·{_feat(dims_sel[0])} + "
                            f"{_fmt2(b)}·{_feat(dims_sel[1])} + "
                            f"{_fmt2(c)} ≤ 0<extra></extra>"
                        ),
                    )
                )

    def _render_rule_mask_and_2d(
        data: List[Any],
        dims_sel: Tuple[int, ...],
        planes: List[Tuple[np.ndarray, float, int]],
        color: str,
        name: str,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> None:
        xs, ys, Xg, Yg = _grid_xy(lo, hi, int(grid_res_2d))
        field_max = None
        for normal_eff, bias_eff, sign in planes:
            current = sign * (normal_eff[dims_sel[0]] * Xg + normal_eff[dims_sel[1]] * Yg + bias_eff)
            field_max = current if field_max is None else np.maximum(field_max, current)
        mask = field_max <= eps
        _add_mask_heatmap_2d(data, xs, ys, mask, color, name)
        for normal_eff, bias_eff, _ in planes:
            a = float(normal_eff[dims_sel[0]])
            b = float(normal_eff[dims_sel[1]])
            c = float(bias_eff)
            segment = _clip_line_to_rect_2d(a, b, c, lo, hi)
            if segment is not None:
                (x0, y0), (x1, y1) = segment
                data.append(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(width=2, color=_ensure_rgb(color)),
                        name=f"{name}-edge",
                        showlegend=False,
                    )
                )

    def _render_rule_1d_band_from_text(
        data: List[Any],
        dims_sel: Tuple[int, ...],
        rule_text: Optional[str],
        color: str,
        name: str,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> bool:
        primary = int(dims_sel[0])
        parsed = _parse_1d_rule_text(rule_text, expected_dim=primary)
        if not parsed:
            return False
        low = parsed.get("low")
        up = parsed.get("up")
        xs, ys, Xg, Yg = _grid_xy(lo, hi, int(grid_res_2d))
        mask = np.ones_like(Xg, dtype=bool)
        if low is not None:
            mask &= Xg >= low - eps
        if up is not None:
            mask &= Xg <= up + eps
        _add_mask_heatmap_2d(data, xs, ys, mask, color, name)
        if low is not None:
            data.append(
                go.Scatter(
                    x=[low, low],
                    y=[lo[1], hi[1]],
                    mode="lines",
                    line=dict(width=2, color=_ensure_rgb(color)),
                    name=f"{name} · x{primary}={_fmt2(low)}",
                    showlegend=False,
                )
            )
        if up is not None:
            data.append(
                go.Scatter(
                    x=[up, up],
                    y=[lo[1], hi[1]],
                    mode="lines",
                    line=dict(width=2, color=_ensure_rgb(color)),
                    name=f"{name} · x{primary}={_fmt2(up)}",
                    showlegend=False,
                )
            )
        return True

    fig = go.Figure()
    all_traces: List[Any] = []
    masks_by_block: List[List[bool]] = []

    for dims_sel in blocks:
        kdim = len(dims_sel)
        points_block = X[:, dims_sel]
        lo, hi = _bounds(points_block)
        block_traces: List[Any] = []

        for cls in classes_all:
            idx = np.flatnonzero(y == int(cls))
            if idx.size > scatter_sample_per_class:
                idx = _choice(idx, size=scatter_sample_per_class)
            pts = X[idx][:, dims_sel] if kdim >= 2 else X[idx][:, [dims_sel[0]]]
            if kdim == 3:
                block_traces.append(
                    go.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        mode="markers",
                        name=f"Clase {cls}",
                        legendgroup=f"class-{cls}",
                        showlegend=True,
                        marker=dict(
                            size=int(point_size),
                            opacity=float(points_opacity),
                            color=class_colors[int(cls)],
                            line=dict(width=0),
                        ),
                    )
                )
            elif kdim == 2:
                block_traces.append(
                    go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode="markers",
                        name=f"Clase {cls}",
                        legendgroup=f"class-{cls}",
                        showlegend=True,
                        marker=dict(
                            size=int(point_size),
                            opacity=float(points_opacity),
                            color=class_colors[int(cls)],
                        ),
                    )
                )
            else:
                block_traces.append(
                    go.Scatter(
                        x=pts[:, 0],
                        y=np.zeros_like(pts[:, 0]),
                        mode="markers",
                        name=f"Clase {cls}",
                        legendgroup=f"class-{cls}",
                        showlegend=True,
                        marker=dict(
                            size=int(point_size),
                            opacity=float(points_opacity),
                            color=class_colors[int(cls)],
                        ),
                    )
                )

        if selected_plane_ids:
            for plane_id in selected_plane_ids:
                plane = plane_index.get(plane_id)
                if not plane:
                    continue
                dims_plane = _maybe_extend_1d_to_2d(_top_dims_from_n(plane["n"], min(3, num_dims)))
                if dims_sel != dims_plane:
                    continue
                normal_eff, bias_eff = _restrict(plane["n"], plane["b"], plane.get("mu", np.nanmean(X, axis=0)), dims_sel)
                sign = _sign_plane(plane)
                target_cls = _plane_target_class(plane_id, plane)
                if target_cls is not None:
                    sign = _orient_sign_to_class(normal_eff, bias_eff, dims_sel, target_cls)
                normal_draw, bias_draw = _absorb_sign(normal_eff, bias_eff, sign)
                color = class_colors.get(int(target_cls) if target_cls is not None else -999, color_planes)
                global_tag = ""
                global_cls = _global_cls(plane)
                if global_cls is not None:
                    global_tag = f" · GLOBAL y={int(global_cls)}"
                name = (
                    f"[{plane_id}] "
                    f"{_fmt2(normal_draw[dims_sel[0]])}·{_feat(dims_sel[0])} + "
                    f"{_fmt2(normal_draw[dims_sel[1]])}·{_feat(dims_sel[1])} "
                    f"{(' + ' + _fmt2(normal_draw[dims_sel[2]]) + '·' + _feat(dims_sel[2])) if kdim == 3 else ''}"
                    f" + {_fmt2(bias_draw)}  (lado pintado: ≤ 0){global_tag}"
                )
                if kdim == 3:
                    _render_halfspace_3d(block_traces, dims_sel, normal_draw, bias_draw, +1, color, name, lo, hi)
                else:
                    _render_halfspace_2d(block_traces, dims_sel, normal_draw, bias_draw, +1, color, name, lo, hi)

        if selected_region_ids:
            for region_id in selected_region_ids:
                if region_id in region_index:
                    region_global = region_index[region_id]
                    geometry = region_global.get("geometry", {})
                    normal = geometry.get("n")
                    bias0 = geometry.get("b")
                    side = int(geometry.get("side", +1))
                    if normal is None:
                        continue
                    dims_region = _maybe_extend_1d_to_2d(_top_dims_from_n(normal, min(3, num_dims)))
                    if dims_sel != dims_region:
                        continue
                    normal_eff, bias_eff = _restrict(normal, bias0, geometry.get("mu", np.nanmean(X, axis=0)), dims_sel)
                    class_region = region_global.get("class_id")
                    if class_region is not None and kdim == 3:
                        sign_oriented = _orient_sign_to_class(normal_eff, bias_eff, dims_sel, int(class_region))
                    else:
                        sign_oriented = side
                    normal_draw, bias_draw = _absorb_sign(normal_eff, bias_eff, sign_oriented)
                    color = class_colors.get(int(class_region) if class_region is not None else -999, color_regions)
                    name = f"[{region_id}] región ★ y={int(class_region) if class_region is not None else '?'} (lado pintado: ≤ 0)"
                    if kdim == 3:
                        _render_halfspace_3d(block_traces, dims_sel, normal_draw, bias_draw, +1, color, name, lo, hi)
                    else:
                        _render_halfspace_2d(block_traces, dims_sel, normal_draw, bias_draw, +1, color, name, lo, hi)
                elif region_id in valuable_region_index:
                    valuable_region = valuable_region_index[region_id]
                    dims_region = tuple(valuable_region.get("dims", ()))
                    if len(dims_region) == 1 and len(dims_sel) == 2:
                        if int(dims_region[0]) != int(dims_sel[0]):
                            continue
                    else:
                        if not set(dims_sel).issubset(set(dims_region)):
                            continue
                    target_cls = int(valuable_region.get("target_class", -1))
                    color = class_colors.get(target_cls, color_rules)
                    if len(dims_region) == 1 and len(dims_sel) == 2 and valuable_region.get("rule_text"):
                        name = f"[{region_id}] rule y={target_cls}"
                        rendered = _render_rule_1d_band_from_text(
                            block_traces,
                            dims_sel,
                            valuable_region["rule_text"],
                            color,
                            name,
                            lo,
                            hi,
                        )
                        if rendered:
                            continue
                    planes: List[Tuple[np.ndarray, float, int]] = []
                    for source in valuable_region.get("sources", []):
                        plane_id = source.get("plane_id")
                        plane = plane_index.get(plane_id)
                        if not plane:
                            continue
                        if len(dims_region) == 1 and len(dims_sel) == 2:
                            primary = int(dims_region[0])
                            normal_1d, bias_1d = _restrict_1d(plane["n"], plane["b"], plane.get("mu", np.nanmean(X, axis=0)), primary)
                            sign = _orient_sign_to_class(normal_1d, bias_1d, (primary,), target_cls)
                            normal_1d = _force_1d_on_primary(normal_1d, dims_sel, primary)
                            planes.append((normal_1d, bias_1d, sign))
                        else:
                            normal_eff, bias_eff = _restrict(plane["n"], plane["b"], plane.get("mu", np.nanmean(X, axis=0)), dims_sel)
                            sign = _orient_sign_to_class(normal_eff, bias_eff, dims_sel, target_cls)
                            normal_eff, bias_eff = _absorb_sign(normal_eff, bias_eff, sign)
                            planes.append((normal_eff, bias_eff, +1))
                    if not planes:
                        continue
                    if kdim == 3:
                        xs, ys, zs, Xg, Yg, Zg = _grid_xyz(lo, hi, int(volume_res_3d))
                        field_max = None
                        for normal_eff, bias_eff, sign in planes:
                            current = normal_eff[dims_sel[0]] * Xg + normal_eff[dims_sel[1]] * Yg + normal_eff[dims_sel[2]] * Zg + bias_eff
                            field_max = current if field_max is None else np.maximum(field_max, current)
                        mask = (field_max <= eps).astype(np.uint8)
                        name = f"[{region_id}] rule y={target_cls}"
                        block_traces.append(
                            go.Volume(
                                x=Xg.flatten(),
                                y=Yg.flatten(),
                                z=Zg.flatten(),
                                value=mask.flatten(),
                                isomin=0.5,
                                isomax=1.0,
                                surface_count=1,
                                opacity=float(region_opacity_3d),
                                colorscale=_colorscale(color),
                                showscale=False,
                                name=name,
                                showlegend=True,
                            )
                        )
                    else:
                        name = f"[{region_id}] rule y={target_cls}"
                        _render_rule_mask_and_2d(block_traces, dims_sel, planes, color, name, lo, hi)

        if not selected_region_ids:
            rules_dim = rules_to_show if rules_to_show else _rules_for_dims_from_valuable(dims_sel)
            for rule in rules_dim:
                dims_region = tuple(rule.get("dims", ()))
                if len(dims_region) == 1 and len(dims_sel) == 2:
                    if int(dims_region[0]) != int(dims_sel[0]):
                        continue
                else:
                    if not set(dims_sel).issubset(set(dims_region)):
                        continue
                target_cls = int(rule.get("target_class", -1))
                color = class_colors.get(target_cls, color_rules)
                if len(dims_region) == 1 and len(dims_sel) == 2 and rule.get("rule_text"):
                    metrics = rule.get("metrics", {})
                    tag = "F1={:.2f}, P={:.2f}, R={:.2f}, n={}".format(
                        metrics.get("f1", 0.0),
                        metrics.get("precision", 0.0),
                        metrics.get("recall", 0.0),
                        metrics.get("size", 0),
                    )
                    name = f"[rule y={target_cls}] · {tag}"
                    rendered = _render_rule_1d_band_from_text(block_traces, dims_sel, rule.get("rule_text"), color, name, lo, hi)
                    if rendered:
                        continue
                planes: List[Tuple[np.ndarray, float, int]] = []
                for source in rule.get("sources", []):
                    plane_id = source.get("plane_id")
                    plane = plane_index.get(plane_id)
                    if not plane:
                        continue
                    if len(dims_region) == 1 and len(dims_sel) == 2:
                        primary = int(dims_region[0])
                        normal_1d, bias_1d = _restrict_1d(plane["n"], plane["b"], plane.get("mu", np.nanmean(X, axis=0)), primary)
                        sign = _orient_sign_to_class(normal_1d, bias_1d, (primary,), target_cls)
                        normal_1d = _force_1d_on_primary(normal_1d, dims_sel, primary)
                        planes.append((normal_1d, bias_1d, sign))
                    else:
                        normal_eff, bias_eff = _restrict(plane["n"], plane["b"], plane.get("mu", np.nanmean(X, axis=0)), dims_sel)
                        sign = _orient_sign_to_class(normal_eff, bias_eff, dims_sel, target_cls)
                        normal_eff, bias_eff = _absorb_sign(normal_eff, bias_eff, sign)
                        planes.append((normal_eff, bias_eff, +1))
                if not planes:
                    continue
                metrics = rule.get("metrics", {})
                tag = "F1={:.2f}, P={:.2f}, R={:.2f}, n={}".format(
                    metrics.get("f1", 0.0),
                    metrics.get("precision", 0.0),
                    metrics.get("recall", 0.0),
                    metrics.get("size", 0),
                )
                if kdim == 3:
                    xs, ys, zs, Xg, Yg, Zg = _grid_xyz(lo, hi, int(volume_res_3d))
                    field_max = None
                    for normal_eff, bias_eff, sign in planes:
                        current = normal_eff[dims_sel[0]] * Xg + normal_eff[dims_sel[1]] * Yg + normal_eff[dims_sel[2]] * Zg + bias_eff
                        field_max = current if field_max is None else np.maximum(field_max, current)
                    mask = (field_max <= eps).astype(np.uint8)
                    name = (
                        f"[rule y={target_cls}] AND({', '.join(str(src.get('plane_id')) for src in rule.get('sources', []))}) · {tag}"
                    )
                    block_traces.append(
                        go.Volume(
                            x=Xg.flatten(),
                            y=Yg.flatten(),
                            z=Zg.flatten(),
                            value=mask.flatten(),
                            isomin=0.5,
                            isomax=1.0,
                            surface_count=1,
                            opacity=float(region_opacity_3d),
                            colorscale=_colorscale(color),
                            showscale=False,
                            name=name,
                            showlegend=True,
                        )
                    )
                else:
                    name = (
                        f"[rule y={target_cls}] AND({', '.join(str(src.get('plane_id')) for src in rule.get('sources', []))}) · {tag}"
                    )
                    _render_rule_mask_and_2d(block_traces, dims_sel, planes, color, name, lo, hi)

        all_traces += block_traces
        masks_by_block.append([True] * len(block_traces))

    for trace in all_traces:
        fig.add_trace(trace)

    if not blocks or not masks_by_block or len(fig.data) == 0:
        fig.update_layout(title=title, legend=dict(itemsizing="constant", groupclick="togglegroup"))
        if show:
            fig.show(config={"scrollZoom": True, "responsive": True})
            return None if not return_fig else fig
        return fig

    n0 = len(masks_by_block[0])
    for idx, trace in enumerate(fig.data):
        trace.visible = idx < n0

    starts = [0]
    for mask in masks_by_block[:-1]:
        starts.append(starts[-1] + len(mask))
    buttons = []
    for block_index, dims_sel in enumerate(blocks):
        visibility = [False] * len(fig.data)
        start = starts[block_index]
        end = start + len(masks_by_block[block_index])
        for idx in range(start, end):
            visibility[idx] = True
        labels = [_feat(j) for j in dims_sel]
        if len(dims_sel) == 3:
            scene_args = {
                "scene": dict(
                    xaxis_title=labels[0],
                    yaxis_title=labels[1],
                    zaxis_title=labels[2],
                    aspectmode="cube",
                )
            }
        elif len(dims_sel) == 2:
            scene_args = {"xaxis": dict(title=labels[0]), "yaxis": dict(title=labels[1])}
        else:
            scene_args = {"xaxis": dict(title=labels[0]), "yaxis": dict(visible=False)}
        buttons.append(
            dict(
                label=f"ejes {tuple(dims_sel)}",
                method="update",
                args=[{"visible": visibility}, scene_args],
            )
        )
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.12, buttons=buttons)],
        title=title,
        legend=dict(itemsizing="constant", groupclick="togglegroup"),
    )

    if blocks:
        labels = [_feat(j) for j in blocks[0]]
        if len(blocks[0]) == 3:
            fig.update_scenes(
                xaxis_title=labels[0],
                yaxis_title=labels[1],
                zaxis_title=labels[2],
                aspectmode="cube",
            )
        elif len(blocks[0]) == 2:
            fig.update_layout(xaxis_title=labels[0], yaxis_title=labels[1])
        else:
            fig.update_layout(xaxis_title=labels[0], yaxis=dict(visible=False))

    if show:
        fig.show(config={"scrollZoom": True, "responsive": True})
        return None if not return_fig else fig
    return fig
