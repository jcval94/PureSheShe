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

import logging
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from time import perf_counter

import numpy as np

from ._logging_utils import verbosity_to_level


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
    return_average_metrics: bool = False,
    verbosity: int = 0,
) -> Union[str, Dict[str, Any]]:
    """Generate a textual report describing the discovered regions.

    When ``return_average_metrics`` is ``True`` the function returns a dictionary
    with per-class and global means for F1 and lift_precision across the
    Top-K (``top_per_class``) regions ranked within each class.
    """

    logger = logging.getLogger(__name__)
    level = verbosity_to_level(max(1, verbosity))
    t0 = perf_counter()
    logger.log(level, "describe_regions_report: inicio | region_id=%s", region_id)

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
        logger.log(level, "describe_regions_report: ficha generada en %.6fs", perf_counter() - t0)
        return "\n".join(lines)

    grouped = _group_by_class(valuable)
    if not grouped:
        return "No hay regiones disponibles."

    k_top = max(1, int(top_per_class))

    def _as_float(value: Any) -> Optional[float]:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(value_f) or math.isinf(value_f):
            return None
        return value_f

    def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
        valid = [v for v in values if v is not None]
        return float(sum(valid) / len(valid)) if valid else None

    if return_average_metrics:
        per_class: Dict[int, Dict[str, Any]] = {}
        for class_id, regions in grouped.items():
            top_regions = regions[:k_top]
            f1_vals: List[Optional[float]] = []
            lift_vals: List[Optional[float]] = []
            for region in top_regions:
                metrics = region.get("metrics", {}) or {}
                f1_vals.append(_as_float(metrics.get("f1")))
                lift_vals.append(_as_float(metrics.get("lift_precision")))

            per_class[class_id] = {
                "mean_f1": _mean_or_none(f1_vals),
                "mean_lift_precision": _mean_or_none(lift_vals),
                "count": len(top_regions),
            }

        global_f1_vals = [vals.get("mean_f1") for vals in per_class.values() if vals.get("mean_f1") is not None]
        global_lift_vals = [
            vals.get("mean_lift_precision") for vals in per_class.values() if vals.get("mean_lift_precision") is not None
        ]

        logger.log(
            level,
            "describe_regions_report: fin (promedios) en %.6fs | clases=%d",
            perf_counter() - t0,
            len(grouped),
        )
        return {
            "per_class": per_class,
            "global_mean": {
                "f1": _mean_or_none(global_f1_vals),
                "lift_precision": _mean_or_none(global_lift_vals),
            },
        }

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
                metrics_pc = _normalize_metrics_per_class(region)
                if metrics_pc:
                    lines.append(
                        "      Métricas por clase: "
                        + ", ".join(
                            [
                                f"c{cls}:prec={_fmt_float(vals.get('precision'))} rec={_fmt_float(vals.get('recall'))} f1={_fmt_float(vals.get('f1'))}"
                                for cls, vals in metrics_pc.items()
                            ]
                        )
                    )

    if show_per_class_in_top:
        lines.append("\nNota: se muestran métricas por clase en el top.")

    logger.log(level, "describe_regions_report: fin en %.6fs | clases=%d", perf_counter() - t0, len(grouped))
    return "\n".join(lines)


def _filter_valuable_by_plane_id(
    valuable: Dict[int, List[Dict[str, Any]]], plane_id: str
) -> Dict[int, List[Dict[str, Any]]]:
    """Filter ``valuable`` keeping only regions that reference ``plane_id``.

    The function inspects ``plane_ids`` as well as the optional ``sources``
    block, returning a structure with the same shape as the input to preserve
    ordering per dimensionality.
    """

    if not plane_id:
        return valuable

    filtered: Dict[int, List[Dict[str, Any]]] = {}
    for dim_k, regions in (valuable or {}).items():
        keep: List[Dict[str, Any]] = []
        for region in regions or []:
            planes = set(region.get("plane_ids") or [])
            sources = {src.get("plane_id") for src in region.get("sources", []) if src.get("plane_id") is not None}
            if plane_id in planes or plane_id in sources:
                keep.append(region)
        if keep:
            filtered[int(dim_k)] = keep
    return filtered


def _fmt_sel_summary(sel: Any, plane_id: Optional[str]) -> List[str]:
    """Render a compact summary of the selection structure (``sel``).

    The function is resilient to partial structures: it looks for a
    ``winning_planes`` collection first, otherwise falls back to any iterable
    provided directly. When ``plane_id`` is supplied the summary only includes
    matching planes.
    """

    if sel is None:
        return []

    planes: List[Dict[str, Any]] = []
    if isinstance(sel, dict):
        maybe_planes = sel.get("winning_planes") or sel.get("planes") or sel.get("selection")
        if maybe_planes is None:
            return []
        planes = list(maybe_planes)
    elif isinstance(sel, Iterable):  # type: ignore[unreachable]
        try:
            planes = list(sel)
        except Exception:
            return []

    lines = ["====== PLANOS SELECCIONADOS ======"]
    for plane in planes:
        pid = plane.get("plane_id", "—")
        if plane_id and pid != plane_id:
            continue
        tgt = plane.get("target_class", "—")
        dims = plane.get("dims") or plane.get("axes") or ()
        source = plane.get("source") or plane.get("family") or "?"
        score = _fmt_float(plane.get("score"))
        lines.append(
            "  id={pid} | clase={cls} | dims={dims} | score={score} | src={src}".format(
                pid=pid,
                cls=tgt,
                dims=tuple(dims) if dims else "—",
                score=score,
                src=source,
            )
        )

    if len(lines) == 1:
        return []
    return lines


def describe_regions_report_with_sel(
    valuable: Dict[int, List[Dict[str, Any]]],
    *,
    sel: Any = None,
    plane_id: Optional[str] = None,
    region_id: Optional[str] = None,
    top_per_class: int = 2,
    dataset_size: Optional[int] = None,
    max_rule_text_chars: int = 220,
    show_per_class_in_top: bool = False,
    fix_rule_text: bool = True,
    show_original_if_changed: bool = True,
    return_average_metrics: bool = False,
    verbosity: int = 0,
) -> Union[str, Dict[str, Any]]:
    """Variant of :func:`describe_regions_report` with selection context.

    Additional features:

    - ``plane_id``: filters the regions to those that reference a specific
      plane (via ``plane_ids`` or ``sources``) and surfaces the matching plane
      from ``sel`` if provided.
    - ``sel``: selection structure returned by
      :func:`prune_and_orient_planes_unified_globalmaj` (or an iterable of
      plane dicts). A compact summary of the planes is appended to the textual
      report or returned under the ``selection`` key when requesting averages.
    """

    logger = logging.getLogger(__name__)
    level = verbosity_to_level(max(1, verbosity))
    logger.log(level, "describe_regions_report_with_sel: inicio | plane_id=%s", plane_id)

    if valuable is None:
        raise ValueError("'valuable' es obligatorio en describe_regions_report_with_sel")

    filtered = _filter_valuable_by_plane_id(valuable, plane_id) if plane_id else valuable

    base = describe_regions_report(
        filtered,
        region_id=region_id,
        top_per_class=top_per_class,
        dataset_size=dataset_size,
        max_rule_text_chars=max_rule_text_chars,
        show_per_class_in_top=show_per_class_in_top,
        fix_rule_text=fix_rule_text,
        show_original_if_changed=show_original_if_changed,
        return_average_metrics=return_average_metrics,
        verbosity=verbosity,
    )

    sel_lines = _fmt_sel_summary(sel, plane_id)
    if return_average_metrics:
        result: Dict[str, Any] = {"metrics": base} if isinstance(base, dict) else {"metrics": base}
        if sel_lines:
            result["selection"] = "\n".join(sel_lines)
        logger.log(level, "describe_regions_report_with_sel: fin (promedios)")
        return result

    if sel_lines and isinstance(base, str):
        report = f"{base}\n\n" + "\n".join(sel_lines)
    else:
        report = base

    logger.log(level, "describe_regions_report_with_sel: fin")
    return report


def describe_regions_metrics(
    valuable: Dict[int, List[Dict[str, Any]]],
    *,
    region_id: Optional[str] = None,
    top_per_class: int = 2,
    dataset_size: Optional[int] = None,
    verbosity: int = 0,
) -> List[Dict[str, Any]]:
    """Return structured metrics (F1 y Lift) por clase.

    La entrada y el criterio de ranking replican ``describe_regions_report`` pero
    en lugar de un texto legible produce una lista de diccionarios numéricos que
    facilita el volcado a CSV o la inspección programática.  Cada entrada
    corresponde a una región, mantiene el orden Top-K por clase y expone las
    métricas principales (``f1`` y ``lift_precision``) junto con identificadores
    útiles para rastrear resultados.
    """

    def _as_float(value: Any) -> Optional[float]:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(value_f) or math.isinf(value_f):
            return None
        return value_f

    logger = logging.getLogger(__name__)
    level = verbosity_to_level(max(1, verbosity))
    t0 = perf_counter()
    logger.log(level, "describe_regions_metrics: inicio | region_id=%s", region_id)

    if region_id:
        region = _find_by_region_id(valuable, region_id)
        if region is None:
            return []
        size, frac = _region_size_and_frac(region, dataset_size)
        metrics = region.get("metrics", {}) or {}
        logger.log(level, "describe_regions_metrics: ficha generada en %.6fs", perf_counter() - t0)
        return [
            dict(
                class_id=int(region.get("target_class")),
                rank_within_class=1,
                region_id=region.get("region_id"),
                dims=tuple(region.get("dims") or ()),
                f1=_as_float(metrics.get("f1")),
                lift_precision=_as_float(metrics.get("lift_precision")),
                precision=_as_float(metrics.get("precision")),
                recall=_as_float(metrics.get("recall")),
                support=size,
                support_frac=float(frac) if isinstance(frac, float) else None,
                pareto=bool(region.get("is_pareto", False)),
                cost=_cost_rec(region),
            )
        ]

    grouped = _group_by_class(valuable)
    if not grouped:
        return []

    k_top = max(1, int(top_per_class))
    entries: List[Dict[str, Any]] = []
    for class_id in sorted(grouped.keys()):
        for idx, region in enumerate(grouped[class_id][:k_top], 1):
            metrics = region.get("metrics", {}) or {}
            size, frac = _region_size_and_frac(region, dataset_size)
            entries.append(
                dict(
                    class_id=int(class_id),
                    rank_within_class=int(idx),
                    region_id=region.get("region_id"),
                    dims=tuple(region.get("dims") or ()),
                    f1=_as_float(metrics.get("f1")),
                    lift_precision=_as_float(metrics.get("lift_precision")),
                    precision=_as_float(metrics.get("precision")),
                    recall=_as_float(metrics.get("recall")),
                    support=size,
                    support_frac=float(frac) if isinstance(frac, float) else None,
                    pareto=bool(region.get("is_pareto", False)),
                    cost=_cost_rec(region),
                )
            )

    logger.log(level, "describe_regions_metrics: fin en %.6fs | clases=%d", perf_counter() - t0, len(grouped))
    return entries


def plot_selected_regions_interactive(
    sel_aug: dict,                # salida de prune_and_orient_planes_unified_globalmaj(...)
    X: np.ndarray,
    y: np.ndarray,
    *,
    # ---- selección del usuario ----
    selected_plane_ids=None,      # e.g. ["pl0003", "pl0007"]
    selected_region_ids=None,     # e.g. ["rg_2d_c0_8ab309427e"]  (puede venir de sel_aug o de 'valuable')
    rules_to_show=None,           # lista de reglas devueltas por find_low_dim_spaces
    valuable=None,                # dict {dim_k: [reglas]} devuelto por find_low_dim_spaces
    # ---- ejes (1D/2D/3D). Si hay planos/regiones, se priorizan sus ejes “mejores”
    dims=(0,1,2),
    dims_options=None,
    feature_names=None,
    # ---- estilo / UI ----
    top_k_planes_per_dims: int = 6,        # (compatibilidad; no se usan por defecto)
    min_dim_alignment: float = 0.45,       # (compatibilidad)
    renderer: str = None,
    title: str = "Regiones y Planos — Interactivo",
    show: bool = True,
    return_fig: bool = False,
    extend: float = 0.06,
    points_opacity: float = 0.6,
    point_size: int = 3,
    scatter_sample_per_class: int = 900,
    region_opacity_3d: float = 0.18,
    region_opacity_2d: float = 0.30,       # opacidad del relleno de intersección
    region_opacity_1d: float = 0.40,       # (para líneas si algún caso cae a 1D puro)
    grid_res_2d: int = 220,
    grid_res_1d: int = 1200,
    volume_res_3d: int = 36,               # NUEVO: resolución volumétrica 3D
    color_planes: str = "#1f77b4",         # fallback si no se puede inferir clase
    color_regions: str = "#9467bd",        # fallback si no se puede inferir clase
    color_rules: str = "#2ca02c",          # fallback si no se puede inferir clase
    # ---- cortes para dims no mostradas ----
    slice_method: str = "mean",            # "mean" | "median" | "plane_mu"
    slice_values: dict | None = None,      # {int|"xj": float} para forzar cortes
    # ---- miscelánea ----
    rng_seed: int | None = 1337            # reproducibilidad del muestreo
):
    logger = logging.getLogger(__name__)
    verbosity = sel_aug.get("verbosity", 0) if isinstance(sel_aug, dict) else 0
    logger.log(
        verbosity_to_level(max(1, verbosity)),
        "plot_selected_regions_interactive: inicio | X=%s y=%s selected_planes=%s",
        getattr(X, 'shape', None), getattr(y, 'shape', None), selected_plane_ids,
    )
    """
    Cambios críticos integrados (3D fix):
      • **SIEMPRE** se pinta el lado n·x + b ≤ 0. Para ello, en 3D se
        reorienta cada semiespacio hacia la clase objetivo y se ABSORBE el signo
        en (n, b) antes de renderizar (evita pintar el lado opuesto).
      • Reglas AND 2D/3D: intersección con **max ≤ 0**.
      • Si una región trae `dims`, se respeta el bloque actual (se filtra por coincidencia).
    """
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio

    # ----------------- init -----------------
    if renderer:
        try: pio.renderers.default = renderer
        except Exception: pass

    X = np.asarray(X, float)
    y = np.asarray(y, int).ravel()
    N, D = X.shape
    eps = 1e-9  # tolerancia para máscaras

    if rng_seed is not None:
        rs = np.random.default_rng(int(rng_seed))
        def _choice(idx, size): return rs.choice(idx, size=size, replace=False)
    else:
        def _choice(idx, size): return np.random.choice(idx, size=size, replace=False)

    # ----------------- utilidades -----------------
    def _feat(i):
        if feature_names and 0 <= i < len(feature_names):
            return feature_names[i]
        return f"x{i}"

    def _fmt2(x): return f"{float(x):.2f}"

    def _ensure_rgb(c):
        c = str(c)
        if c.startswith("#") and len(c)==7:
            r=int(c[1:3],16); g=int(c[3:5],16); b=int(c[5:7],16)
            return f"rgb({r},{g},{b})"
        if c.startswith("rgba"):
            inner = c[c.find("(")+1:c.find(")")]
            parts = [t.strip() for t in inner.split(",")][0:3]
            if len(parts)>=3:
                r,g,b = parts[:3]
                return f"rgb({r},{g},{b})"
        return c

    def _colorscale(c):
        return [[0.0,_ensure_rgb(c)],[1.0,_ensure_rgb(c)]]

    # === parseo 1D desde rule_text ===
    _re_ineq = re.compile(
        r"x\s*(\d+)\s*(≥|<=|≤|>=|<|>)\s*([+\-−]?(?:\d+(?:\.\d*)?|\.\d+))",
        flags=re.IGNORECASE
    )
    def _parse_1d_rule_text(rule_text, expected_dim=None):
        if not isinstance(rule_text, str): return None
        s = rule_text.replace("−", "-")
        found = _re_ineq.findall(s)
        if not found: return None
        lows = []; ups = []; dim_seen = set()
        for dim_str, op, val_str in found:
            d = int(dim_str); v = float(val_str.replace("−","-"))
            dim_seen.add(d)
            if op in ("≥", ">=", ">"): lows.append(v)
            elif op in ("≤", "<=", "<"): ups.append(v)
        if expected_dim is not None and len(dim_seen) > 0 and expected_dim not in dim_seen:
            pass
        low = max(lows) if lows else None
        up  = min(ups)  if ups  else None
        if (low is not None) and (up is not None) and (up < low):
            low, up = up, low
        return dict(dim=list(dim_seen)[0] if dim_seen else expected_dim, low=low, up=up)

    # índices por id (planos y regiones globales del pipeline)
    by_pair = sel_aug.get("by_pair_augmented", {}) or {}
    plane_index = {}
    for (a,b), entry in by_pair.items():
        for p in (entry.get("winning_planes", []) or []):
            if p.get("plane_id") is not None:
                plane_index[p["plane_id"]] = dict(p, origin_pair=(int(a), int(b)))
        for p in (entry.get("other_planes", []) or []):
            if p.get("plane_id") is not None:
                plane_index[p["plane_id"]] = dict(p, origin_pair=(int(a), int(b)))

    region_index = {}
    for R in sel_aug.get("regions_global", {}).get("per_plane", []) or []:
        rid = R.get("region_id")
        if rid: region_index[rid] = dict(R)

    # ======= Índice de regiones provenientes de 'valuable' (reglas) =======
    valuable_region_index = {}
    if isinstance(valuable, dict):
        for _k, lst in valuable.items():
            for rr in lst or []:
                rid = rr.get("region_id")
                if rid:
                    valuable_region_index[rid] = rr

    def _normalize_plane_id(pid):
        if isinstance(pid, (list, tuple)) and pid:
            pid = pid[0]
        if isinstance(pid, dict):
            pid = pid.get("plane_id") or pid.get("id")
        return pid

    selected_plane_ids_raw = selected_plane_ids or []
    selected_plane_ids = []
    for pid in selected_plane_ids_raw:
        pid_norm = _normalize_plane_id(pid)
        if pid_norm is not None:
            selected_plane_ids.append(pid_norm)
    selected_region_ids = list(selected_region_ids or [])
    rules_to_show = list(rules_to_show or [])

    # --- construir mapa plane_id → target_class a partir de 'valuable' (si viene)
    plane_cls_from_valuable = {}
    if isinstance(valuable, dict):
        for dim_k, lst in valuable.items():
            for rr in lst or []:
                cls = int(rr.get("target_class", -1))
                for src in rr.get("sources", []):
                    pid = src.get("plane_id")
                    if pid is not None:
                        plane_cls_from_valuable.setdefault(pid, cls)

    # signos / global badge
    def _sign_plane(p):
        ineq = (p.get("inequality", {}) or {}).get("general", "")
        s = str(ineq).replace(" ", "")
        if ("<=0" in s) or ("≤0" in s): return +1     # n·x+b ≤ 0
        if (">=0" in s) or ("≥0" in s): return -1     # n·x+b ≥ 0
        return +1 if int(p.get("side", +1)) >= 0 else -1

    def _global_cls(p):
        g = p.get("global_for", None)
        if g is None:
            opp = p.get("opposite_side_eval", {})
            if isinstance(opp, dict) and opp.get("meets_majority_rule", False):
                g = opp.get("majority_class", None)
        return None if g is None else int(g)

    def _plane_target_class(pid, p):
        if pid in plane_cls_from_valuable: return int(plane_cls_from_valuable[pid])
        if "label" in p and p["label"] is not None:
            try: return int(p["label"])
            except Exception: pass
        g = _global_cls(p)
        if g is not None: return int(g)
        return None

    # restringir plano a dims seleccionadas (b_eff ajusta otras dims al corte)
    def _restrict(n_full, b0, mu_plane, dims_sel):
        dims_sel = tuple(int(i) for i in dims_sel)
        # base de cortes por datos
        if slice_method == "median":
            mu = np.nanmedian(X, axis=0)
        elif slice_method == "plane_mu" and mu_plane is not None:
            mu = np.asarray(mu_plane, float)
        else:
            mu = np.nanmean(X, axis=0)
        mu = np.asarray(mu, float).reshape(-1)
        # overrides explícitos
        if isinstance(slice_values, dict):
            for k,v in slice_values.items():
                if isinstance(k, str) and k.startswith('x'):
                    try: j = int(k[1:])
                    except Exception:
                        continue
                else:
                    j = int(k)
                if j not in dims_sel and 0 <= j < D:
                    mu[j] = float(v)
        other = [j for j in range(D) if j not in dims_sel]
        n_full = np.asarray(n_full, float).reshape(-1)
        b_eff = float(b0) + (float(n_full[other] @ mu[other]) if other else 0.0)
        n_eff = np.zeros(D); n_eff[list(dims_sel)] = n_full[list(dims_sel)]
        return n_eff, b_eff

    def _top_dims_from_n(n, k):
        n = np.asarray(n, float).ravel()
        idx = np.argsort(-np.abs(n))[:max(1, min(k, len(n)))]
        return tuple(int(i) for i in np.sort(idx))

    def _dims_from_rule_sources(rr, k=3):
        """Top-k dims por importancia agregada |n| de los planos fuente de la regla."""
        acc = np.zeros(D, float)
        for src in rr.get("sources", []) or []:
            pid = src.get("plane_id")
            p = plane_index.get(pid)
            if p is None:
                continue
            acc += np.abs(np.asarray(p["n"], float).ravel())
        if not np.any(acc):  # fallback si no hay fuentes o no se encontraron
            dims_rr = tuple(rr.get("dims", ()))
            return tuple(int(i) for i in sorted(dims_rr[:k]))
        idx = np.argsort(-acc)[:max(1, min(k, D))]
        return tuple(int(i) for i in np.sort(idx))



    # ---- orientación robusta hacia la clase objetivo con condición "≤ 0" ----
    def _orient_sign_to_class(n_eff, b_eff, dims_sel, tgt_cls):
        idx = np.flatnonzero(y == int(tgt_cls))
        if idx.size == 0:
            return +1
        P = X[idx][:, list(dims_sel)]
        val = b_eff + np.sum(np.array([n_eff[d]*P[:,i] for i,d in enumerate(dims_sel)]), axis=0)
        good_pos = np.count_nonzero(+val <= 0)
        good_neg = np.count_nonzero(-val <= 0)  # equivalente a val >= 0
        return +1 if good_pos >= good_neg else -1

    def _absorb_sign(n_eff, b_eff, s):
        n2 = np.array(n_eff, float)
        n2 *= float(s)
        b2 = float(b_eff) * float(s)
        return n2, b2

    # ---- helpers para 1D→2D ----
    def _restrict_1d(n_full, b0, mu, primary_dim):
        n_full = np.asarray(n_full, float).reshape(-1)
        if slice_method == "median":
            mu_base = np.nanmedian(X, axis=0)
        elif slice_method == "plane_mu" and mu is not None:
            mu_base = np.asarray(mu, float)
        else:
            mu_base = np.nanmean(X, axis=0)
        others = [j for j in range(D) if j != int(primary_dim)]
        b_eff = float(b0) + (float(n_full[others] @ mu_base[others]) if others else 0.0)
        n_eff = np.zeros(D, float)
        n_eff[int(primary_dim)] = float(n_full[int(primary_dim)])
        return n_eff, b_eff

    def _force_1d_on_primary(n_eff, dims_sel, primary_dim):
        n2 = np.array(n_eff, float)
        for j in dims_sel:
            if int(j) != int(primary_dim):
                n2[int(j)] = 0.0
        return n2

    # elegir dimensión de cobertura para 1D
    def _pick_filler_dim(primary):
        others = [j for j in range(D) if j != primary]
        if not others:
            return primary
        rng = [(j, float(np.nanmax(X[:, j]) - np.nanmin(X[:, j]))) for j in others]
        j_star = max(rng, key=lambda t: t[1])[0]
        return int(j_star)

    # paletas y puntos (colores por CLASE)
    pal = px.colors.qualitative.Plotly
    classes_all = sorted(np.unique(y))
    class_colors = {int(c): pal[i % len(pal)] for i,c in enumerate(classes_all)}

    # ----------------- elegir bloques de ejes -----------------
    user_dims_given = dims is not None
    base_k = len(dims) if user_dims_given else min(3, D)
    blocks = []

    def _maybe_extend_1d_to_2d(dims_tuple):
        if len(dims_tuple) == 1:
            prim = int(dims_tuple[0])
            fill = _pick_filler_dim(prim)
            if fill == prim and D >= 2:
                fill = (prim + 1) % D
            return (prim, fill)
        return tuple(dims_tuple)

    # Prioridad 1: regiones seleccionadas
    if selected_region_ids:
        for rid in selected_region_ids:
            if rid in region_index:
                R = region_index[rid]
                n = (R.get("geometry", {}) or {}).get("n")
                if n is None: continue
                dims_r = _top_dims_from_n(n, base_k)
            elif rid in valuable_region_index:
                # rr = valuable_region_index[rid]
                # dims_r = tuple(int(i) for i in rr.get("dims", ()))
                # if not dims_r:
                #     continue

                rr = valuable_region_index[rid]
                dims_r = tuple(int(i) for i in rr.get("dims", ()))
                if not dims_r:
                    continue
                # --- FIX: si la regla viene con 4+ dims, reducimos a las 3 más relevantes ---
                if len(dims_r) > 3:
                    dims_r = _dims_from_rule_sources(rr, k=3)

            else:
                continue
            dims_r = _maybe_extend_1d_to_2d(dims_r)
            if dims_r not in blocks:
                blocks.append(dims_r)
        if not blocks:
            base = tuple(int(i) for i in dims) if user_dims_given else tuple(range(min(3,D)))
            blocks = [_maybe_extend_1d_to_2d(base)]
    # Prioridad 2: planos seleccionados
    elif selected_plane_ids:
        for pid in selected_plane_ids:
            p = plane_index.get(pid)
            if not p: continue
            dims_for_p = _maybe_extend_1d_to_2d(_top_dims_from_n(p["n"], base_k))
            if dims_for_p not in blocks:
                blocks.append(dims_for_p)
        if dims_options:
            for opt in dims_options:
                opt = _maybe_extend_1d_to_2d(tuple(int(i) for i in opt))
                if opt not in blocks: blocks.append(opt)
    else:
        if dims_options:
            for opt in dims_options:
                opt = _maybe_extend_1d_to_2d(tuple(int(i) for i in opt))
                if opt not in blocks: blocks.append(opt)
        elif user_dims_given:
            blocks = [_maybe_extend_1d_to_2d(tuple(int(i) for i in dims))]
        else:
            blocks = [_maybe_extend_1d_to_2d(tuple(range(min(3, D))))]

    # -------------- fallback reglas desde 'valuable' por bloque/dims --------------
    def _rules_for_dims_from_valuable(dims_sel, top=3):
        if not isinstance(valuable, dict):
            return []
        dim_k = len(dims_sel)
        pool = []
        if dim_k == 2:
            prim = dims_sel[0]
            for rr in valuable.get(2, []) or []:
                if tuple(rr.get("dims", ())) == tuple(dims_sel):
                    pool.append(rr)
            for rr in valuable.get(1, []) or []:
                drr = tuple(rr.get("dims", ()))
                if len(drr) == 1 and int(drr[0]) == int(prim):
                    pool.append(rr)
        else:
            for rr in valuable.get(dim_k, []) or []:
                if tuple(rr.get("dims", ())) == tuple(dims_sel):
                    pool.append(rr)
        if not pool:
            return []
        pool.sort(key=lambda r: (r.get("metrics", {}).get("f1", 0.0),
                                 r.get("metrics", {}).get("precision", 0.0),
                                 r.get("metrics", {}).get("size", 0)), reverse=True)
        return pool[:top]

    # ----------------- helpers de render -----------------
    _grid_cache = {}
    def _grid_xy(lo, hi, rr):
        key=('xy', float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), int(rr))
        if key not in _grid_cache:
            xs=np.linspace(lo[0],hi[0],rr); ys=np.linspace(lo[1],hi[1],rr)
            Xg,Yg=np.meshgrid(xs,ys, indexing="xy")
            _grid_cache[key]=(xs,ys,Xg,Yg)
        return _grid_cache[key]

    def _grid_xyz(lo, hi, rr):
        key=('xyz', float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]), int(rr))
        if key not in _grid_cache:
            xs=np.linspace(lo[0],hi[0],rr); ys=np.linspace(lo[1],hi[1],rr); zs=np.linspace(lo[2],hi[2],rr)
            Xg,Yg,Zg=np.meshgrid(xs,ys,zs, indexing="ij")
            _grid_cache[key]=(xs,ys,zs,Xg,Yg,Zg)
        return _grid_cache[key]

    def _bounds(P):
        lo = np.nanmin(P, axis=0); hi = np.nanmax(P, axis=0)
        span = np.maximum(hi-lo, 1e-12)
        pad = float(extend)*span
        return lo-pad, hi+pad

    def _clip_line_to_rect_2d(a, b, c, lo, hi):
        epsl = 1e-12; pts = []
        if abs(b) > epsl:
            y0 = -(a*lo[0] + c)/b
            if lo[1]-1e-9 <= y0 <= hi[1]+1e-9: pts.append((lo[0], y0))
            y1 = -(a*hi[0] + c)/b
            if lo[1]-1e-9 <= y1 <= hi[1]+1e-9: pts.append((hi[0], y1))
        if abs(a) > epsl:
            x0 = -(b*lo[1] + c)/a
            if lo[0]-1e-9 <= x0 <= hi[0]+1e-9: pts.append((x0, lo[1]))
            x1 = -(b*hi[1] + c)/a
            if lo[0]-1e-9 <= x1 <= hi[0]+1e-9: pts.append((x1, hi[1]))
        uniq = []
        for p in pts:
            if all((abs(p[0]-q[0])>1e-7 or abs(p[1]-q[1])>1e-7) for q in uniq):
                uniq.append(p)
        if len(uniq) < 2:
            return None
        import itertools as _it
        best = (None, None, -1.0)
        for u,v in _it.combinations(uniq,2):
            d = (u[0]-v[0])**2 + (u[1]-v[1])**2
            if d > best[2]: best = (u,v,d)
        return best[0], best[1]

    def _add_mask_heatmap_2d(data, xs, ys, mask_bool, color, name):
        z = np.where(mask_bool, 1.0, np.nan)
        data.append(go.Heatmap(
            x=xs, y=ys, z=z,
            colorscale=_colorscale(color),
            opacity=float(region_opacity_2d),
            showscale=False,
            hoverinfo="skip",
            zmin=0.0, zmax=1.0,
            name=name, showlegend=True,
            zsmooth="best"
        ))

    # ------ RENDER HALFSPACES: condición "≤ 0" unificada ------
    def _render_halfspace_3d(data, dims_sel, n_eff, b_eff, sign, color, name, lo, hi):
        # cara del plano en la caja
        def _box_vertices(lo3, hi3):
            x0,y0,z0 = lo3; x1,y1,z1 = hi3
            return np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                             [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]], float)
        EDGE = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        V=_box_vertices(lo,hi); pts=[]; n3=np.array([n_eff[dims_sel[0]],n_eff[dims_sel[1]],n_eff[dims_sel[2]]],float)
        for i0,i1 in EDGE:
            p0,p1=V[i0],V[i1]; d0=float(n3@p0+b_eff); d1=float(n3@p1+b_eff); den=d1-d0
            if abs(den)<1e-12:
                if abs(d0)<1e-12 and abs(d1)<1e-12: pts+=[p0,p1]
                continue
            t=-d0/den
            if -1e-9<=t<=1+1e-9:
                t=min(max(t,0.0),1.0); pts.append(p0+t*(p1-p0))
        uniq=[];
        for p in pts:
            if all(np.linalg.norm(p-q)>1e-7 for q in uniq): uniq.append(p)
        P=np.array(uniq,float)
        if P.shape[0]>=3:
            i=[];j=[];k=[]
            for t in range(1,P.shape[0]-1): i.append(0);j.append(t);k.append(t+1)
            data.append(go.Mesh3d(x=P[:,0],y=P[:,1],z=P[:,2],i=i,j=j,k=k,
                                  color=_ensure_rgb(color),opacity=0.35,showlegend=False,name=name,
                                  hoverinfo="skip"))

        # volumen del semiespacio
        xs, ys, zs, Xg, Yg, Zg = _grid_xyz(lo, hi, int(volume_res_3d))
        field=sign*(n_eff[dims_sel[0]]*Xg + n_eff[dims_sel[1]]*Yg + n_eff[dims_sel[2]]*Zg + b_eff)
        mask=(field<=eps).astype(np.uint8)
        data.append(go.Volume(x=Xg.flatten(),y=Yg.flatten(),z=Zg.flatten(),
                              value=mask.flatten(),isomin=0.5,isomax=1.0,surface_count=1,
                              opacity=float(region_opacity_3d),colorscale=_colorscale(color),
                              showscale=False,name=name,showlegend=True,
                              hovertemplate=f"{_fmt2(n_eff[dims_sel[0]])}·{_feat(dims_sel[0])} + "
                                            f"{_fmt2(n_eff[dims_sel[1]])}·{_feat(dims_sel[1])} + "
                                            f"{_fmt2(n_eff[dims_sel[2]])}·{_feat(dims_sel[2])} + "
                                            f"{_fmt2(b_eff)} ≤ 0<extra></extra>"))

    def _render_halfspace_2d(data, dims_sel, n_eff, b_eff, sign, color, name, lo, hi, draw_boundary=True):
        xs, ys, Xg, Yg = _grid_xy(lo, hi, int(grid_res_2d))
        field=sign*(n_eff[dims_sel[0]]*Xg + n_eff[dims_sel[1]]*Yg + b_eff)
        mask=(field<=eps)
        _add_mask_heatmap_2d(data, xs, ys, mask, color, name)
        if draw_boundary:
            a=float(n_eff[dims_sel[0]]); b=float(n_eff[dims_sel[1]]); c=float(b_eff)
            seg = _clip_line_to_rect_2d(a,b,c, lo, hi)
            if seg is not None:
                (x0,y0),(x1,y1) = seg
                data.append(go.Scatter(x=[x0,x1],y=[y0,y1],mode="lines",
                                       line=dict(width=2,color=_ensure_rgb(color)),
                                       name=name,showlegend=False,
                                       hovertemplate=f"{_fmt2(a)}·{_feat(dims_sel[0])} + "
                                                     f"{_fmt2(b)}·{_feat(dims_sel[1])} + "
                                                     f"{_fmt2(c)} ≤ 0<extra></extra>"))

    def _render_rule_mask_AND_2d(data, dims_sel, planes, color, name, lo, hi):
        xs, ys, Xg, Yg = _grid_xy(lo, hi, int(grid_res_2d))
        field_max=None
        for n_eff,b_eff,s in planes:
            f=s*(n_eff[dims_sel[0]]*Xg + n_eff[dims_sel[1]]*Yg + b_eff)
            field_max = f if field_max is None else np.maximum(field_max,f)
        mask=(field_max<=eps)  # intersección correcta: max ≤ 0
        _add_mask_heatmap_2d(data, xs, ys, mask, color, name)
        for n_eff,b_eff,s in planes:
            a=float(n_eff[dims_sel[0]]); b=float(n_eff[dims_sel[1]]); c=float(b_eff)
            seg = _clip_line_to_rect_2d(a,b,c, lo, hi)
            if seg is not None:
                (x0,y0),(x1,y1) = seg
                data.append(go.Scatter(x=[x0,x1],y=[y0,y1],mode="lines",
                                       line=dict(width=2,color=_ensure_rgb(color)),
                                       name=f"{name}-edge",showlegend=False))

    # === 1D: renderer por parsing de rule_text ===
    def _render_rule_1d_band_from_text(data, dims_sel, rule_text, color, name, lo, hi):
        primary = int(dims_sel[0])
        parsed = _parse_1d_rule_text(rule_text, expected_dim=primary)
        if not parsed:
            return False
        low = parsed.get("low", None); up  = parsed.get("up", None)
        xs, ys, Xg, Yg = _grid_xy(lo, hi, int(grid_res_2d))
        mask = np.ones_like(Xg, dtype=bool)
        if low is not None: mask &= (Xg >= low - eps)
        if up  is not None: mask &= (Xg <= up  + eps)
        _add_mask_heatmap_2d(data, xs, ys, mask, color, name)
        if low is not None:
            data.append(go.Scatter(x=[low, low], y=[lo[1], hi[1]], mode="lines",
                                   line=dict(width=2, color=_ensure_rgb(color)),
                                   name=f"{name} · x{primary}={_fmt2(low)}", showlegend=False))
        if up is not None:
            data.append(go.Scatter(x=[up, up], y=[lo[1], hi[1]], mode="lines",
                                   line=dict(width=2, color=_ensure_rgb(color)),
                                   name=f"{name} · x{primary}={_fmt2(up)}", showlegend=False))
        return True

    # ----------------- construir figura -----------------
    fig = go.Figure()
    all_traces = []
    masks_by_block = []

    for dims_sel in blocks:
        kdim = len(dims_sel)

        # puntos -> ESCALA por estos puntos
        Pc = X[:, dims_sel]
        lo,hi = _bounds(Pc)

        block_traces=[]
        for c in classes_all:
            idx=np.flatnonzero(y==int(c))
            if idx.size>scatter_sample_per_class:
                idx=_choice(idx,size=scatter_sample_per_class)
            P = X[idx][:,dims_sel] if kdim>=2 else X[idx][:,[dims_sel[0]]]
            if kdim==3:
                block_traces.append(go.Scatter3d(x=P[:,0],y=P[:,1],z=P[:,2],mode="markers",
                                                 name=f"Clase {c}", legendgroup=f"class-{c}", showlegend=True,
                                                 marker=dict(size=int(point_size),opacity=float(points_opacity),
                                                             color=class_colors[int(c)],line=dict(width=0))))
            elif kdim==2:
                block_traces.append(go.Scatter(x=P[:,0],y=P[:,1],mode="markers",
                                               name=f"Clase {c}", legendgroup=f"class-{c}", showlegend=True,
                                               marker=dict(size=int(point_size),opacity=float(points_opacity),
                                                           color=class_colors[int(c)])))
            else:
                block_traces.append(go.Scatter(x=P[:,0],y=np.zeros_like(P[:,0]),mode="markers",
                                               name=f"Clase {c}", legendgroup=f"class-{c}", showlegend=True,
                                               marker=dict(size=int(point_size),opacity=float(points_opacity),
                                                           color=class_colors[int(c)])))

        # ======== DIBUJO DE PLANOS =========
        if selected_plane_ids:
            for pid in selected_plane_ids:
                p = plane_index.get(pid)
                if not p: continue
                dims_p = _maybe_extend_1d_to_2d(_top_dims_from_n(p["n"], min(3, D)))
                if dims_sel != dims_p:
                    continue
                n_eff, b_eff = _restrict(p["n"], p["b"], p.get("mu", np.nanmean(X,axis=0)), dims_sel)
                # orientamos SIEMPRE hacia la clase (si se conoce) y absorbemos el signo
                s = _sign_plane(p)
                tgt_cls = _plane_target_class(pid, p)
                if tgt_cls is not None:
                    s = _orient_sign_to_class(n_eff, b_eff, dims_sel, tgt_cls)
                n_draw, b_draw = _absorb_sign(n_eff, b_eff, s)
                ccol = class_colors.get(int(tgt_cls) if tgt_cls is not None else -999, color_planes)
                g = _global_cls(p)
                global_tag = (f" · GLOBAL y={int(g)}" if g is not None else "")
                name = (f"[{pid}] "
                        f"{_fmt2(n_draw[dims_sel[0]])}·{_feat(dims_sel[0])} + "
                        f"{_fmt2(n_draw[dims_sel[1]])}·{_feat(dims_sel[1])} "
                        f"{(' + ' + _fmt2(n_draw[dims_sel[2]]) + '·' + _feat(dims_sel[2])) if kdim==3 else ''}"
                        f" + {_fmt2(b_draw)}  (lado pintado: ≤ 0){global_tag}"
                       )
                if kdim==3:
                    _render_halfspace_3d(block_traces, dims_sel, n_draw, b_draw, +1, ccol, name, lo, hi)
                else:
                    _render_halfspace_2d(block_traces, dims_sel, n_draw, b_draw, +1, ccol, name, lo, hi)

        # ======== REGIONES SELECCIONADAS =========
        if selected_region_ids:
            for rid in selected_region_ids:
                # (a) región global en sel_aug
                if rid in region_index:
                    R = region_index[rid]
                    geom = R.get("geometry",{})
                    n,b0,sg = geom.get("n"), geom.get("b"), int(geom.get("side",+1))
                    if n is None: continue
                    dims_r = _maybe_extend_1d_to_2d(_top_dims_from_n(n, min(3, D)))
                    if dims_sel != dims_r:
                        continue
                    n_eff, b_eff = _restrict(n, b0, geom.get("mu", np.nanmean(X,axis=0)), dims_sel)
                    cls_r = R.get("class_id", None)
                    # orientamos a la clase si está disponible, y absorbemos signo
                    if cls_r is not None and kdim == 3:
                        s_or = _orient_sign_to_class(n_eff, b_eff, dims_sel, int(cls_r))
                    else:
                        s_or = sg
                    n_draw, b_draw = _absorb_sign(n_eff, b_eff, s_or)
                    ccol = class_colors.get(int(cls_r), color_regions)
                    name=f"[{rid}] región ★ y={int(cls_r) if cls_r is not None else '?'} (lado pintado: ≤ 0)"
                    if kdim==3:
                        _render_halfspace_3d(block_traces, dims_sel, n_draw, b_draw, +1, ccol, name, lo, hi)
                    else:
                        _render_halfspace_2d(block_traces, dims_sel, n_draw, b_draw, +1, ccol, name, lo, hi)
                # (b) región/REGLA en 'valuable'
                elif rid in valuable_region_index:
                    rr = valuable_region_index[rid]
                    # dims_rr = tuple(rr.get("dims", ()))
                    # if len(dims_rr)==1 and len(dims_sel)==2:
                    #     if int(dims_rr[0]) != int(dims_sel[0]):
                    #         continue
                    # elif tuple(dims_sel) != tuple(dims_rr):
                    #     continue

                    dims_rr = tuple(rr.get("dims", ()))
                    if len(dims_rr)==1 and len(dims_sel)==2:
                        # caso especial 1D→2D: exigimos que la primera dim del bloque sea la primaria
                        if int(dims_rr[0]) != int(dims_sel[0]):
                            continue
                    else:
                        # --- FIX: permitir que el bloque sea cualquier sub-conjunto 2D/3D de la regla 4D ---
                        if not set(dims_sel).issubset(set(dims_rr)):
                            continue

                    tgt_cls = int(rr.get("target_class", -1))
                    ccol = class_colors.get(tgt_cls, color_rules)
                    if len(dims_rr)==1 and len(dims_sel)==2 and rr.get("rule_text"):
                        name=f"[{rid}] rule y={tgt_cls}"
                        ok = _render_rule_1d_band_from_text(block_traces, dims_sel, rr["rule_text"], ccol, name, lo, hi)
                        if ok:
                            continue
                    planes=[]
                    for src in rr.get("sources",[]):
                        pid=src.get("plane_id"); p=plane_index.get(pid)
                        if not p: continue
                        if len(dims_rr) == 1 and len(dims_sel) == 2:
                            primary = int(dims_rr[0])
                            n1,b1 = _restrict_1d(p["n"], p["b"], p.get("mu", np.nanmean(X,axis=0)), primary)
                            s = _orient_sign_to_class(n1, b1, (primary,), tgt_cls)
                            n1 = _force_1d_on_primary(n1, dims_sel, primary)
                            planes.append((n1, b1, s))
                        else:
                            n_eff,b_eff = _restrict(p["n"], p["b"], p.get("mu", np.nanmean(X,axis=0)), dims_sel)
                            # *** FIX 3D: orientar y ABSORBER signo en 3D ***
                            s = _orient_sign_to_class(n_eff, b_eff, dims_sel, tgt_cls)
                            n_eff, b_eff = _absorb_sign(n_eff, b_eff, s)
                            planes.append((n_eff,b_eff,+1))
                    if not planes:
                        continue
                    if kdim==3:
                        xs, ys, zs, Xg, Yg, Zg = _grid_xyz(lo, hi, int(volume_res_3d))
                        field_max=None
                        for n_eff,b_eff,s in planes:
                            f=(n_eff[dims_sel[0]]*Xg + n_eff[dims_sel[1]]*Yg + n_eff[dims_sel[2]]*Zg + b_eff)
                            field_max = f if field_max is None else np.maximum(field_max,f)
                        mask=(field_max<=eps).astype(np.uint8)  # AND = max ≤ 0
                        name=f"[{rid}] rule y={tgt_cls}"
                        block_traces.append(go.Volume(x=Xg.flatten(),y=Yg.flatten(),z=Zg.flatten(),
                                                      value=mask.flatten(),isomin=0.5,isomax=1.0,surface_count=1,
                                                      opacity=float(region_opacity_3d),colorscale=_colorscale(ccol),
                                                      showscale=False,name=name,showlegend=True))
                    else:
                        name=f"[{rid}] rule y={tgt_cls}"
                        _render_rule_mask_AND_2d(block_traces, dims_sel, planes, ccol, name, lo, hi)

        # ======== REGLAS (rules_to_show o fallback) =========
        if not selected_region_ids:
            rules_dim = rules_to_show if rules_to_show else _rules_for_dims_from_valuable(dims_sel)
            for rr in rules_dim:
                # dims_rr = tuple(rr.get("dims", ()))
                # if len(dims_rr)==1 and len(dims_sel)==2:
                #     if int(dims_rr[0]) != int(dims_sel[0]):
                #         continue
                # elif tuple(dims_sel) != tuple(dims_rr):
                #     continue

                dims_rr = tuple(rr.get("dims", ()))
                if len(dims_rr)==1 and len(dims_sel)==2:
                    # caso especial 1D→2D: exigimos que la primera dim del bloque sea la primaria
                    if int(dims_rr[0]) != int(dims_sel[0]):
                        continue
                else:
                    # --- FIX: permitir que el bloque sea cualquier sub-conjunto 2D/3D de la regla 4D ---
                    if not set(dims_sel).issubset(set(dims_rr)):
                        continue

                tgt_cls = int(rr.get("target_class", -1))
                ccol = class_colors.get(tgt_cls, color_rules)

                if len(dims_rr)==1 and len(dims_sel)==2 and rr.get("rule_text"):
                    m=rr.get("metrics",{}); tag=f"F1={m.get('f1',0):.2f}, P={m.get('precision',0):.2f}, R={m.get('recall',0):.2f}, n={m.get('size',0)}"
                    name=f"[rule y={tgt_cls}] · {tag}"
                    ok = _render_rule_1d_band_from_text(block_traces, dims_sel, rr["rule_text"], ccol, name, lo, hi)
                    if ok:
                        continue

                planes=[]
                for src in rr.get("sources",[]):
                    pid=src.get("plane_id"); p=plane_index.get(pid)
                    if not p: continue
                    if len(dims_rr) == 1 and len(dims_sel) == 2:
                        primary = int(dims_rr[0])
                        n1,b1 = _restrict_1d(p["n"], p["b"], p.get("mu", np.nanmean(X,axis=0)), primary)
                        s = _orient_sign_to_class(n1, b1, (primary,), tgt_cls)
                        n1 = _force_1d_on_primary(n1, dims_sel, primary)
                        planes.append((n1, b1, s))
                    else:
                        n_eff,b_eff = _restrict(p["n"], p["b"], p.get("mu", np.nanmean(X,axis=0)), dims_sel)
                        # *** FIX 3D también aquí ***
                        s = _orient_sign_to_class(n_eff, b_eff, dims_sel, tgt_cls)
                        n_eff, b_eff = _absorb_sign(n_eff, b_eff, s)
                        planes.append((n_eff,b_eff,+1))
                if not planes: continue

                if kdim==3:
                    xs, ys, zs, Xg, Yg, Zg = _grid_xyz(lo, hi, int(volume_res_3d))
                    field_max=None
                    for n_eff,b_eff,s in planes:
                        f=(n_eff[dims_sel[0]]*Xg + n_eff[dims_sel[1]]*Yg + n_eff[dims_sel[2]]*Zg + b_eff)
                        field_max = f if field_max is None else np.maximum(field_max,f)
                    mask=(field_max<=eps).astype(np.uint8)
                    m=rr.get("metrics",{}); tag=f"F1={m.get('f1',0):.2f}, P={m.get('precision',0):.2f}, R={m.get('recall',0):.2f}, n={m.get('size',0)}"
                    name=f"[rule y={tgt_cls}] AND({', '.join(str(s['plane_id']) for s in rr.get('sources',[]))}) · {tag}"
                    block_traces.append(go.Volume(x=Xg.flatten(),y=Yg.flatten(),z=Zg.flatten(),
                                                  value=mask.flatten(),isomin=0.5,isomax=1.0,surface_count=1,
                                                  opacity=float(region_opacity_3d),colorscale=_colorscale(ccol),
                                                  showscale=False,name=name,showlegend=True))
                else:
                    m=rr.get("metrics",{}); tag=f"F1={m.get('f1',0):.2f}, P={m.get('precision',0):.2f}, R={m.get('recall',0):.2f}, n={m.get('size',0)}"
                    name=f"[rule y={tgt_cls}] AND({', '.join(str(s['plane_id']) for s in rr.get('sources',[]))}) · {tag}"
                    _render_rule_mask_AND_2d(block_traces, dims_sel, planes, ccol, name, lo, hi)

        # acumula
        all_traces += block_traces
        masks_by_block.append([True]*len(block_traces))

    # volcar traces y toggle por bloque
    for tr in all_traces:
        fig.add_trace(tr)

    # Guardas si no hay bloques o trazas
    if not blocks or not masks_by_block or len(fig.data)==0:
        fig.update_layout(title=title,
                          legend=dict(itemsizing="constant", groupclick="togglegroup"))
        if show:
            fig.show(config={"scrollZoom": True, "responsive": True})
            return None if not return_fig else fig
        else:
            return fig

    # visibilidad inicial = primer bloque
    n0 = len(masks_by_block[0])
    for i,tr in enumerate(fig.data):
        tr.visible = (i < n0)

    # botones por bloque de ejes
    starts=[0]
    for m in masks_by_block[:-1]:
        starts.append(starts[-1] + len(m))
    buttons=[]
    for bi, dims_sel in enumerate(blocks):
        vis=[False]*len(fig.data)
        s=starts[bi]; e=s+len(masks_by_block[bi])
        for i in range(s,e): vis[i]=True
        labels=[_feat(j) for j in dims_sel]
        if len(dims_sel)==3:
            scene_args={"scene": dict(xaxis_title=labels[0], yaxis_title=labels[1], zaxis_title=labels[2], aspectmode="cube")}
        elif len(dims_sel)==2:
            scene_args={"xaxis": dict(title=labels[0]), "yaxis": dict(title=labels[1])}
        else:
            scene_args={"xaxis": dict(title=labels[0]), "yaxis": dict(visible=False)}
        buttons.append(dict(label=f"ejes {tuple(dims_sel)}", method="update",
                            args=[{"visible": vis}, scene_args]))
    fig.update_layout(updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.12, buttons=buttons)],
                      title=title,
                      legend=dict(itemsizing="constant", groupclick="togglegroup"))

    # títulos del primer bloque
    if blocks:
        labels=[_feat(j) for j in blocks[0]]
        if len(blocks[0])==3:
            fig.update_scenes(xaxis_title=labels[0], yaxis_title=labels[1], zaxis_title=labels[2], aspectmode="cube")
        elif len(blocks[0])==2:
            fig.update_layout(xaxis_title=labels[0], yaxis_title=labels[1])
        else:
            fig.update_layout(xaxis_title=labels[0], yaxis=dict(visible=False))

    if show:
        fig.show(config={"scrollZoom": True, "responsive": True})
        return None if not return_fig else fig
    else:
        return fig

