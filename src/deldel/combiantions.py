from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# =========================
# Bitset helpers (rápidos)
# =========================

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _packbits(mask: np.ndarray) -> np.ndarray:
    """Pack boolean mask into uint8 bitset using big-endian bit order."""
    return np.packbits(mask.astype(np.uint8), bitorder="big")


def _countbits(packed: np.ndarray) -> int:
    """Popcount of packed uint8 bitset."""
    return int(_POPCOUNT_LUT[packed].sum())


def _and_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_and(a, b)


def _or_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_or(a, b)


def _invert_bits(a: np.ndarray, nbits: int) -> np.ndarray:
    """Invert packed bits but keep padding bits (beyond nbits) as 0."""
    inv = np.bitwise_not(a)
    r = nbits % 8
    if r != 0:
        # packbits(big): valid bits are the top r bits in the last byte
        last_mask = (0xFF << (8 - r)) & 0xFF
        inv = inv.copy()
        inv[-1] = inv[-1] & last_mask
    return inv


def _md5_short(data: bytes, n: int = 12) -> str:
    return hashlib.md5(data).hexdigest()[:n]


# =========================
# Plane + Rule structures
# =========================


@dataclass(frozen=True)
class Plane:
    oriented_plane_id: str
    plane_id: str
    origin_pair: Tuple[int, int]
    side: int
    dims: Tuple[int, ...]
    n_norm: np.ndarray  # shape (len(dims),)
    b_norm: float
    inequality_general: str
    family_id: Any
    metrics_by_class: Dict[int, Dict[str, float]]  # e.g. {0:{precision:.., lift:..}, ...}

    def sign(self) -> str:
        # oriented_plane_id like "pl0000:≤" or "pl0000:≥"
        if self.oriented_plane_id.endswith("≤"):
            return "≤"
        if self.oriented_plane_id.endswith("≥"):
            return "≥"
        # fallback: infer from side (not ideal)
        return "≤" if self.side < 0 else "≥"

    def mask_on_X(self, X: np.ndarray, atol: float = 1e-12) -> np.ndarray:
        """Evaluate halfspace membership."""
        # Use only dims
        Xd = X[:, self.dims]
        expr = Xd @ self.n_norm + float(self.b_norm)
        s = self.sign()
        if s == "≤":
            return expr <= atol
        else:
            return expr >= -atol


@dataclass
class RuleCandidate:
    target_class: int
    plane_indices: Tuple[int, ...]  # indices into planes list
    dims: Tuple[int, ...]  # union dims across planes
    mask_bits: np.ndarray  # packed bitset
    size: int
    tp: int
    metrics: Dict[str, float]  # includes precision/recall/f1/... + baseline + lift_precision + size + region_frac


@dataclass
class DnfCandidate:
    target_class: int
    clause_indices: Tuple[int, ...]  # indices into AND rules list
    dims: Tuple[int, ...]
    mask_bits: np.ndarray
    size: int
    tp: int
    metrics: Dict[str, float]


# =========================
# Metrics
# =========================


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2.0 * p * r, (p + r))


def _balacc(tpr: float, tnr: float) -> float:
    return 0.5 * (tpr + tnr)


def _compute_region_metrics(
    mask_bits: np.ndarray,
    y: np.ndarray,
    packed_class_masks: Dict[int, np.ndarray],
    target_class: int,
    N: int,
) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]], Dict[str, Any]]:
    """
    Compute:
      - metrics (target OVR): precision/recall/f1/acc/balacc/size/region_frac/baseline/lift_precision
      - metrics_per_class (OVR for each class)
      - region_summary (tp/fp/fn/tn/etc)
    """
    size = _countbits(mask_bits)
    region_frac = _safe_div(size, N)

    # target counts
    tmask = packed_class_masks[target_class]
    tp = _countbits(_and_bits(mask_bits, tmask))
    fp = size - tp

    total_pos = _countbits(tmask)
    fn = total_pos - tp
    tn = N - tp - fp - fn

    precision = _safe_div(tp, size)
    recall = _safe_div(tp, total_pos)

    # baseline prevalence
    baseline = _safe_div(total_pos, N)
    lift_precision = _safe_div(precision, baseline) if baseline > 0 else 0.0

    # tnr for balacc
    total_neg = N - total_pos
    tnr = _safe_div(tn, total_neg) if total_neg > 0 else 0.0

    acc = _safe_div(tp + tn, N)
    f1 = _f1(precision, recall)
    balacc = _balacc(recall, tnr)

    # "coverage" en tu sel suele alinearse con recall OVR (cuánto de la clase captura la región).
    metrics = {
        "size": float(size),
        "region_frac": float(region_frac),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "acc": float(acc),
        "balacc": float(balacc),
        "baseline": float(baseline),
        "lift_precision": float(lift_precision),
        # aliases compatibles con tu nomenclatura:
        "coverage": float(recall),
        "lift": float(lift_precision),
        "purity": float(precision),
    }

    # per-class OVR metrics inside the same region (útil para auditoría)
    metrics_per_class: Dict[int, Dict[str, float]] = {}
    for c, cmask in packed_class_masks.items():
        c_in_region = _countbits(_and_bits(mask_bits, cmask))
        c_total = _countbits(cmask)

        c_prec = _safe_div(c_in_region, size)
        c_rec = _safe_div(c_in_region, c_total)
        c_f1 = _f1(c_prec, c_rec)

        metrics_per_class[c] = {
            "precision": float(c_prec),
            "recall": float(c_rec),
            "f1": float(c_f1),
            "size": float(size),
            "region_frac": float(region_frac),
        }

    region_summary = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "N": int(N),
        "accuracy": float(acc),
        "size": int(size),
        "region_frac": float(region_frac),
    }
    return metrics, metrics_per_class, region_summary


def _compute_target_metrics_from_counts(
    size: int,
    tp: int,
    total_pos: int,
    N: int,
) -> Dict[str, float]:
    region_frac = _safe_div(size, N)
    fp = size - tp
    fn = total_pos - tp
    tn = N - tp - fp - fn

    precision = _safe_div(tp, size)
    recall = _safe_div(tp, total_pos)

    baseline = _safe_div(total_pos, N)
    lift_precision = _safe_div(precision, baseline) if baseline > 0 else 0.0

    total_neg = N - total_pos
    tnr = _safe_div(tn, total_neg) if total_neg > 0 else 0.0

    acc = _safe_div(tp + tn, N)
    f1 = _f1(precision, recall)
    balacc = _balacc(recall, tnr)

    return {
        "size": float(size),
        "region_frac": float(region_frac),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "acc": float(acc),
        "balacc": float(balacc),
        "baseline": float(baseline),
        "lift_precision": float(lift_precision),
        "coverage": float(recall),
        "lift": float(lift_precision),
        "purity": float(precision),
    }


def _compute_per_class_metrics(
    mask_bits: np.ndarray,
    packed_class_masks: Dict[int, np.ndarray],
    class_sizes: Dict[int, int],
    size: int,
    region_frac: float,
) -> Dict[int, Dict[str, float]]:
    metrics_per_class: Dict[int, Dict[str, float]] = {}
    for c, cmask in packed_class_masks.items():
        c_in_region = _countbits(_and_bits(mask_bits, cmask))
        c_total = class_sizes[c]

        c_prec = _safe_div(c_in_region, size)
        c_rec = _safe_div(c_in_region, c_total)
        c_f1 = _f1(c_prec, c_rec)

        metrics_per_class[c] = {
            "precision": float(c_prec),
            "recall": float(c_rec),
            "f1": float(c_f1),
            "size": float(size),
            "region_frac": float(region_frac),
        }
    return metrics_per_class


def _compute_region_summary_from_counts(
    size: int,
    tp: int,
    total_pos: int,
    N: int,
    acc: float,
    region_frac: float,
) -> Dict[str, Any]:
    fp = size - tp
    fn = total_pos - tp
    tn = N - tp - fp - fn
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "N": int(N),
        "accuracy": float(acc),
        "size": int(size),
        "region_frac": float(region_frac),
    }


# =========================
# Ranking (balance)
# =========================


def _wracc(metrics: Dict[str, float]) -> float:
    # WRAcc = p(cond) * (p(pos|cond) - p(pos))
    return metrics["region_frac"] * (metrics["precision"] - metrics["baseline"])


def _primary_value(metrics: Dict[str, float], metric: str) -> float:
    # metric must exist in metrics dict computed by _compute_region_metrics
    if metric not in metrics:
        raise ValueError(
            f"metric='{metric}' no está en metrics disponibles. Disponibles: {sorted(metrics.keys())}"
        )
    return float(metrics[metric])


def _rank_tuple(metrics: Dict[str, float], metric: str) -> Tuple[float, float, float, float]:
    """
    Ranking lexicográfico:
      1) métrica primaria (lo que el usuario pidió)
      2) WRAcc (balance precisión+cobertura)
      3) region_frac (tamaño)
      4) lift_precision (enriquecimiento)
    """
    return (
        _primary_value(metrics, metric),
        float(_wracc(metrics)),
        float(metrics["region_frac"]),
        float(metrics["lift_precision"]),
    )


def _rank_rule_dict(rule: Dict[str, Any], metric: str) -> Tuple[float, float, float, float]:
    metrics = dict(rule.get("metrics") or {})
    region_summary = rule.get("region_summary") or {}
    if "region_frac" not in metrics:
        metrics["region_frac"] = float(region_summary.get("region_frac", 0.0))
    if "lift_precision" not in metrics and "lift" in metrics:
        metrics["lift_precision"] = float(metrics.get("lift", 0.0))
    return _rank_tuple(metrics, metric)


def _rank_tuple_hessian(
    metrics: Dict[str, float],
    metric: str,
    hessian_score: float,
    *,
    weight: float,
) -> Tuple[float, float, float, float]:
    primary = _primary_value(metrics, metric) + weight * float(hessian_score)
    return (
        primary,
        float(_wracc(metrics)),
        float(metrics["region_frac"]),
        float(metrics["lift_precision"]),
    )


# =========================
# Core: Beam search (AND rules)
# =========================


def _beam_search_and_rules(
    planes: List[Plane],
    plane_bits: List[np.ndarray],
    y: np.ndarray,
    classes: List[int],
    target_class: int,
    packed_class_masks: Dict[int, np.ndarray],
    class_sizes: Dict[int, int],
    *,
    metric: str = "precision",
    lift_min: float = 1.0,
    beam_width: int = 16,
    max_planes: int = 7,
    min_size: int = 5,
    max_candidates: int = 150,
) -> List[RuleCandidate]:
    """
    Devuelve una lista de RuleCandidate (reglas AND) encontradas vía beam search.
    - Usa ranking lexicográfico: primary metric -> WRAcc -> size -> lift
    - Prioriza lift_min: filtra candidatos que no cumplen (si hay suficientes que sí cumplen).
    """
    N = int(y.shape[0])
    tmask = packed_class_masks[target_class]
    total_pos = class_sizes[target_class]

    # ---- Selección inicial de planos candidatos para esta clase (reduce el search space)
    # Ordenamos planos por métrica primaria individual y lift individual.
    scored_planes = []
    for i, pl in enumerate(planes):
        mbc = pl.metrics_by_class.get(target_class, {})
        # si el plano no tiene métricas para target, lo ignoramos
        if not mbc:
            continue
        # usamos lo que ya está en sel para ranking inicial (rápido)
        pl_primary = float(mbc.get(metric, mbc.get("precision", 0.0)))
        pl_lift = float(mbc.get("lift", mbc.get("lift_precision", 0.0)))
        pl_frac = float(mbc.get("region_frac", mbc.get("region_frac_eval", 0.0)))
        scored_planes.append((pl_primary, pl_lift, pl_frac, i))

    scored_planes.sort(reverse=True)
    cand_plane_indices = [i for *_rest, i in scored_planes[:max_candidates]]
    if not cand_plane_indices:
        return []

    # ---- Beam init: single-plane rules
    single_rules: List[RuleCandidate] = []
    for idx in cand_plane_indices:
        bits = plane_bits[idx]
        size = _countbits(bits)
        if size < min_size:
            continue

        dims = tuple(sorted(set(planes[idx].dims)))
        tp = _countbits(_and_bits(bits, tmask))
        metrics_t = _compute_target_metrics_from_counts(size, tp, total_pos, N)
        single_rules.append(
            RuleCandidate(
                target_class=target_class,
                plane_indices=(idx,),
                dims=dims,
                mask_bits=bits,
                size=size,
                tp=tp,
                metrics=metrics_t,
            )
        )

    if not single_rules:
        return []

    # If we have any that satisfy lift_min, keep only those for initial beam; else keep best anyway.
    good = [r for r in single_rules if r.metrics["lift_precision"] > lift_min]
    seed_pool = good if good else single_rules

    seed_pool.sort(key=lambda r: _rank_tuple(r.metrics, metric), reverse=True)
    beam = seed_pool[:beam_width]

    # Keep all visited best rules
    all_rules: Dict[Tuple[int, ...], RuleCandidate] = {r.plane_indices: r for r in beam}

    # ---- Expand
    # To avoid permutations, we only add planes with index > last added in cand list order
    # We'll map global plane idx to its position in cand list
    pos_in_cand = {pidx: j for j, pidx in enumerate(cand_plane_indices)}

    for depth in range(2, max_planes + 1):
        expansions: Dict[Tuple[int, ...], RuleCandidate] = {}

        for r in beam:
            last_pos = pos_in_cand.get(r.plane_indices[-1], -1)
            if last_pos < 0:
                continue

            for next_pos in range(last_pos + 1, len(cand_plane_indices)):
                nxt = cand_plane_indices[next_pos]
                if nxt in r.plane_indices:
                    continue

                new_planes = r.plane_indices + (nxt,)
                # AND mask
                new_bits = _and_bits(r.mask_bits, plane_bits[nxt])
                size = _countbits(new_bits)
                if size < min_size:
                    continue

                new_dims = tuple(sorted(set(r.dims).union(planes[nxt].dims)))

                tp = _countbits(_and_bits(new_bits, tmask))
                metrics_t = _compute_target_metrics_from_counts(size, tp, total_pos, N)
                cand = RuleCandidate(
                    target_class=target_class,
                    plane_indices=new_planes,
                    dims=new_dims,
                    mask_bits=new_bits,
                    size=size,
                    tp=tp,
                    metrics=metrics_t,
                )

                expansions[new_planes] = cand

        if not expansions:
            break

        # Lift filtering preference
        exp_list = list(expansions.values())
        exp_good = [x for x in exp_list if x.metrics["lift_precision"] > lift_min]
        exp_pool = exp_good if exp_good else exp_list

        exp_pool.sort(key=lambda r: _rank_tuple(r.metrics, metric), reverse=True)
        beam = exp_pool[:beam_width]

        for r in beam:
            prev = all_rules.get(r.plane_indices)
            if prev is None or _rank_tuple(r.metrics, metric) > _rank_tuple(
                prev.metrics, metric
            ):
                all_rules[r.plane_indices] = r

    # Return all rules found, sorted
    out = list(all_rules.values())
    out.sort(key=lambda r: _rank_tuple(r.metrics, metric), reverse=True)
    return out


def _beam_search_or_rules(
    planes: List[Plane],
    plane_bits: List[np.ndarray],
    y: np.ndarray,
    classes: List[int],
    target_class: int,
    packed_class_masks: Dict[int, np.ndarray],
    class_sizes: Dict[int, int],
    *,
    metric: str = "precision",
    lift_min: float = 1.0,
    beam_width: int = 16,
    max_planes: int = 7,
    min_size: int = 5,
    max_candidates: int = 150,
) -> List[RuleCandidate]:
    """
    Devuelve una lista de RuleCandidate (reglas OR) encontradas vía beam search.
    - Usa ranking lexicográfico: primary metric -> WRAcc -> size -> lift
    - Prioriza lift_min: filtra candidatos que no cumplen (si hay suficientes que sí cumplen).
    """
    N = int(y.shape[0])
    tmask = packed_class_masks[target_class]
    total_pos = class_sizes[target_class]

    scored_planes = []
    for i, pl in enumerate(planes):
        mbc = pl.metrics_by_class.get(target_class, {})
        if not mbc:
            continue
        pl_primary = float(mbc.get(metric, mbc.get("precision", 0.0)))
        pl_lift = float(mbc.get("lift", mbc.get("lift_precision", 0.0)))
        pl_frac = float(mbc.get("region_frac", mbc.get("region_frac_eval", 0.0)))
        scored_planes.append((pl_primary, pl_lift, pl_frac, i))

    scored_planes.sort(reverse=True)
    cand_plane_indices = [i for *_rest, i in scored_planes[:max_candidates]]
    if not cand_plane_indices:
        return []

    single_rules: List[RuleCandidate] = []
    for idx in cand_plane_indices:
        bits = plane_bits[idx]
        size = _countbits(bits)
        if size < min_size:
            continue

        dims = tuple(sorted(set(planes[idx].dims)))
        tp = _countbits(_and_bits(bits, tmask))
        metrics_t = _compute_target_metrics_from_counts(size, tp, total_pos, N)
        single_rules.append(
            RuleCandidate(
                target_class=target_class,
                plane_indices=(idx,),
                dims=dims,
                mask_bits=bits,
                size=size,
                tp=tp,
                metrics=metrics_t,
            )
        )

    if not single_rules:
        return []

    good = [r for r in single_rules if r.metrics["lift_precision"] > lift_min]
    seed_pool = good if good else single_rules

    seed_pool.sort(key=lambda r: _rank_tuple(r.metrics, metric), reverse=True)
    beam = seed_pool[:beam_width]

    all_rules: Dict[Tuple[int, ...], RuleCandidate] = {r.plane_indices: r for r in beam}
    pos_in_cand = {pidx: j for j, pidx in enumerate(cand_plane_indices)}

    for depth in range(2, max_planes + 1):
        expansions: Dict[Tuple[int, ...], RuleCandidate] = {}

        for r in beam:
            last_pos = pos_in_cand.get(r.plane_indices[-1], -1)
            if last_pos < 0:
                continue

            for next_pos in range(last_pos + 1, len(cand_plane_indices)):
                nxt = cand_plane_indices[next_pos]
                if nxt in r.plane_indices:
                    continue

                new_planes = r.plane_indices + (nxt,)
                new_bits = _or_bits(r.mask_bits, plane_bits[nxt])
                size = _countbits(new_bits)
                if size < min_size:
                    continue

                new_dims = tuple(sorted(set(r.dims).union(planes[nxt].dims)))

                tp = _countbits(_and_bits(new_bits, tmask))
                metrics_t = _compute_target_metrics_from_counts(size, tp, total_pos, N)
                cand = RuleCandidate(
                    target_class=target_class,
                    plane_indices=new_planes,
                    dims=new_dims,
                    mask_bits=new_bits,
                    size=size,
                    tp=tp,
                    metrics=metrics_t,
                )

                expansions[new_planes] = cand

        if not expansions:
            break

        exp_list = list(expansions.values())
        exp_good = [x for x in exp_list if x.metrics["lift_precision"] > lift_min]
        exp_pool = exp_good if exp_good else exp_list

        exp_pool.sort(key=lambda r: _rank_tuple(r.metrics, metric), reverse=True)
        beam = exp_pool[:beam_width]

        for r in beam:
            prev = all_rules.get(r.plane_indices)
            if prev is None or _rank_tuple(r.metrics, metric) > _rank_tuple(
                prev.metrics, metric
            ):
                all_rules[r.plane_indices] = r

    out = list(all_rules.values())
    out.sort(key=lambda r: _rank_tuple(r.metrics, metric), reverse=True)
    return out


def _greedy_dnf_rules(
    rules: List[Dict[str, Any]],
    packed_class_masks: Dict[int, np.ndarray],
    class_sizes: Dict[int, int],
    *,
    N: int,
    target_class: int,
    metric: str = "precision",
    max_clauses: int = 4,
    max_dnf_rules: int = 5,
) -> List[DnfCandidate]:
    tmask = packed_class_masks[target_class]
    total_pos = class_sizes[target_class]

    if not rules:
        return []

    ordered = sorted(rules, key=lambda r: _rank_rule_dict(r, metric), reverse=True)
    max_dnf_rules = max(1, int(max_dnf_rules))

    def _build_greedy(start_idx: Optional[int]) -> Optional[DnfCandidate]:
        chosen: List[int] = []
        current_mask: Optional[np.ndarray] = None
        current_metrics: Optional[Dict[str, float]] = None
        current_size = 0
        current_tp = 0

        if start_idx is not None:
            r_mask = ordered[start_idx].get("_mask_bits")
            if r_mask is None:
                return None
            current_mask = r_mask
            current_size = _countbits(current_mask)
            current_tp = _countbits(_and_bits(current_mask, tmask))
            current_metrics = _compute_target_metrics_from_counts(
                current_size, current_tp, total_pos, N
            )
            chosen.append(start_idx)

        for idx, r in enumerate(ordered):
            if len(chosen) >= max_clauses:
                break
            if idx in chosen:
                continue
            r_mask = r.get("_mask_bits")
            if r_mask is None:
                continue
            candidate_mask = r_mask if current_mask is None else _or_bits(current_mask, r_mask)
            size = _countbits(candidate_mask)
            tp = _countbits(_and_bits(candidate_mask, tmask))
            metrics_t = _compute_target_metrics_from_counts(size, tp, total_pos, N)
            if current_metrics is None or _rank_tuple(metrics_t, metric) >= _rank_tuple(
                current_metrics, metric
            ):
                chosen.append(idx)
                current_mask = candidate_mask
                current_metrics = metrics_t
                current_size = size
                current_tp = tp

        if not chosen or current_mask is None or current_metrics is None:
            return None

        dims = tuple(
            sorted({int(d) for i in chosen for d in (ordered[i].get("dims") or [])})
        )
        return DnfCandidate(
            target_class=target_class,
            clause_indices=tuple(chosen),
            dims=dims,
            mask_bits=current_mask,
            size=int(current_size),
            tp=int(current_tp),
            metrics=current_metrics,
        )

    starters = [None] + list(range(min(len(ordered), max(3, max_dnf_rules * 2))))
    candidates: Dict[Tuple[int, ...], DnfCandidate] = {}
    for start_idx in starters:
        cand = _build_greedy(start_idx)
        if cand is None:
            continue
        prev = candidates.get(cand.clause_indices)
        if prev is None or _rank_tuple(cand.metrics, metric) > _rank_tuple(prev.metrics, metric):
            candidates[cand.clause_indices] = cand

    out = list(candidates.values())
    out.sort(key=lambda r: _rank_tuple(r.metrics, metric), reverse=True)
    return out[:max_dnf_rules]


def _beam_search_dnf_rules(
    rules: List[Dict[str, Any]],
    packed_class_masks: Dict[int, np.ndarray],
    class_sizes: Dict[int, int],
    *,
    N: int,
    target_class: int,
    metric: str = "precision",
    max_clauses: int = 4,
    beam_width: int = 16,
) -> List[DnfCandidate]:
    tmask = packed_class_masks[target_class]
    total_pos = class_sizes[target_class]

    if not rules:
        return []

    ordered = sorted(rules, key=lambda r: _rank_rule_dict(r, metric), reverse=True)
    masks = [r.get("_mask_bits") for r in ordered]
    valid_indices = [i for i, m in enumerate(masks) if m is not None]
    if not valid_indices:
        return []

    singletons: List[DnfCandidate] = []
    for i in valid_indices:
        mask = masks[i]
        if mask is None:
            continue
        size = _countbits(mask)
        tp = _countbits(_and_bits(mask, tmask))
        metrics_t = _compute_target_metrics_from_counts(size, tp, total_pos, N)
        dims = tuple(sorted({int(d) for d in (ordered[i].get("dims") or [])}))
        singletons.append(
            DnfCandidate(
                target_class=target_class,
                clause_indices=(i,),
                dims=dims,
                mask_bits=mask,
                size=size,
                tp=tp,
                metrics=metrics_t,
            )
        )

    if not singletons:
        return []

    singletons.sort(key=lambda r: _rank_tuple(r.metrics, metric), reverse=True)
    beam = singletons[:beam_width]
    all_rules: Dict[Tuple[int, ...], DnfCandidate] = {r.clause_indices: r for r in beam}

    for depth in range(2, max_clauses + 1):
        expansions: Dict[Tuple[int, ...], DnfCandidate] = {}
        for r in beam:
            last_idx = r.clause_indices[-1]
            for nxt in range(last_idx + 1, len(ordered)):
                if nxt in r.clause_indices:
                    continue
                mask = masks[nxt]
                if mask is None:
                    continue
                new_indices = r.clause_indices + (nxt,)
                new_mask = _or_bits(r.mask_bits, mask)
                size = _countbits(new_mask)
                tp = _countbits(_and_bits(new_mask, tmask))
                metrics_t = _compute_target_metrics_from_counts(size, tp, total_pos, N)
                dims = tuple(sorted({int(d) for d in r.dims}.union(ordered[nxt].get("dims") or [])))
                expansions[new_indices] = DnfCandidate(
                    target_class=target_class,
                    clause_indices=new_indices,
                    dims=dims,
                    mask_bits=new_mask,
                    size=size,
                    tp=tp,
                    metrics=metrics_t,
                )

        if not expansions:
            break
        exp_list = list(expansions.values())
        exp_list.sort(key=lambda r: _rank_tuple(r.metrics, metric), reverse=True)
        beam = exp_list[:beam_width]
        for r in beam:
            prev = all_rules.get(r.clause_indices)
            if prev is None or _rank_tuple(r.metrics, metric) > _rank_tuple(
                prev.metrics, metric
            ):
                all_rules[r.clause_indices] = r

    out = list(all_rules.values())
    out.sort(key=lambda r: _rank_tuple(r.metrics, metric), reverse=True)
    return out


def _random_search_dnf_rules(
    rules: List[Dict[str, Any]],
    packed_class_masks: Dict[int, np.ndarray],
    class_sizes: Dict[int, int],
    *,
    N: int,
    target_class: int,
    metric: str = "precision",
    max_clauses: int = 4,
    iterations: int = 120,
    seed: int = 0,
    max_dnf_rules: int = 5,
) -> List[DnfCandidate]:
    tmask = packed_class_masks[target_class]
    total_pos = class_sizes[target_class]

    if not rules:
        return []

    ordered = sorted(rules, key=lambda r: _rank_rule_dict(r, metric), reverse=True)
    masks = [r.get("_mask_bits") for r in ordered]
    valid_indices = [i for i, m in enumerate(masks) if m is not None]
    if not valid_indices:
        return []

    rng = np.random.default_rng(seed)
    max_dnf_rules = max(1, int(max_dnf_rules))
    best_by_clauses: Dict[Tuple[int, ...], DnfCandidate] = {}

    for _ in range(max(1, int(iterations))):
        k = int(rng.integers(1, max_clauses + 1))
        sample = rng.choice(valid_indices, size=min(k, len(valid_indices)), replace=False)
        sample = tuple(sorted(int(x) for x in sample))
        mask = None
        dims = set()
        for idx in sample:
            dims.update(ordered[idx].get("dims") or [])
            mask = masks[idx] if mask is None else _or_bits(mask, masks[idx])
        if mask is None:
            continue
        size = _countbits(mask)
        tp = _countbits(_and_bits(mask, tmask))
        metrics_t = _compute_target_metrics_from_counts(size, tp, total_pos, N)
        cand = DnfCandidate(
            target_class=target_class,
            clause_indices=sample,
            dims=tuple(sorted(int(d) for d in dims)),
            mask_bits=mask,
            size=size,
            tp=tp,
            metrics=metrics_t,
        )
        prev = best_by_clauses.get(sample)
        if prev is None or _rank_tuple(cand.metrics, metric) > _rank_tuple(prev.metrics, metric):
            best_by_clauses[sample] = cand

    if not best_by_clauses:
        return []

    out = list(best_by_clauses.values())
    out.sort(key=lambda r: _rank_tuple(r.metrics, metric), reverse=True)
    return out[:max_dnf_rules]


def _diverse_topk_dnf_rules(
    rules: List[Dict[str, Any]],
    packed_class_masks: Dict[int, np.ndarray],
    class_sizes: Dict[int, int],
    *,
    N: int,
    target_class: int,
    metric: str = "precision",
    max_clauses: int = 3,
    candidates_per_class: int = 30,
    overlap_max: float = 0.7,
    max_dnf_rules: int = 5,
) -> List[DnfCandidate]:
    tmask = packed_class_masks[target_class]
    total_pos = class_sizes[target_class]

    if not rules:
        return []

    ordered = sorted(rules, key=lambda r: _rank_rule_dict(r, metric), reverse=True)
    ordered = ordered[: max(1, int(candidates_per_class))]
    masks = [r.get("_mask_bits") for r in ordered]
    valid_indices = [i for i, m in enumerate(masks) if m is not None]
    if not valid_indices:
        return []

    selected: List[int] = []
    current_mask: Optional[np.ndarray] = None

    for idx in valid_indices:
        mask = masks[idx]
        if mask is None:
            continue
        if current_mask is None:
            selected.append(idx)
            current_mask = mask
        else:
            overlap = _countbits(_and_bits(current_mask, mask))
            denom = max(1, _countbits(mask))
            if (overlap / denom) <= overlap_max:
                selected.append(idx)
                current_mask = _or_bits(current_mask, mask)
        if len(selected) >= max_clauses:
            break

    if not selected or current_mask is None:
        return []

    candidates: List[DnfCandidate] = []
    running_mask: Optional[np.ndarray] = None
    for i in range(1, min(len(selected), max_clauses) + 1):
        subset = selected[:i]
        running_mask = masks[subset[0]] if running_mask is None else running_mask
        if i > 1:
            running_mask = _or_bits(running_mask, masks[subset[-1]])
        if running_mask is None:
            continue
        size = _countbits(running_mask)
        tp = _countbits(_and_bits(running_mask, tmask))
        metrics_t = _compute_target_metrics_from_counts(size, tp, total_pos, N)
        dims = tuple(sorted({int(d) for idx in subset for d in (ordered[idx].get("dims") or [])}))
        candidates.append(
            DnfCandidate(
                target_class=target_class,
                clause_indices=tuple(subset),
                dims=dims,
                mask_bits=running_mask,
                size=size,
                tp=tp,
                metrics=metrics_t,
            )
        )

    if not candidates:
        return []

    candidates.sort(key=lambda r: _rank_tuple(r.metrics, metric), reverse=True)
    max_dnf_rules = max(1, int(max_dnf_rules))
    return candidates[:max_dnf_rules]


def _beam_search_and_rules_hessian(
    planes: List[Plane],
    plane_bits: List[np.ndarray],
    plane_scores: List[float],
    y: np.ndarray,
    classes: List[int],
    target_class: int,
    packed_class_masks: Dict[int, np.ndarray],
    class_sizes: Dict[int, int],
    *,
    metric: str = "precision",
    lift_min: float = 1.0,
    beam_width: int = 16,
    max_planes: int = 7,
    min_size: int = 5,
    max_candidates: int = 150,
    hessian_weight: float = 0.2,
) -> List[RuleCandidate]:
    N = int(y.shape[0])
    tmask = packed_class_masks[target_class]
    total_pos = class_sizes[target_class]

    scored_planes = []
    for i, pl in enumerate(planes):
        mbc = pl.metrics_by_class.get(target_class, {})
        if not mbc:
            continue
        pl_primary = float(mbc.get(metric, mbc.get("precision", 0.0)))
        pl_lift = float(mbc.get("lift", mbc.get("lift_precision", 0.0)))
        pl_frac = float(mbc.get("region_frac", mbc.get("region_frac_eval", 0.0)))
        scored_planes.append((pl_primary, pl_lift, pl_frac, i))

    scored_planes.sort(reverse=True)
    cand_plane_indices = [i for *_rest, i in scored_planes[:max_candidates]]
    if not cand_plane_indices:
        return []

    def _candidate_hessian(indices: Tuple[int, ...]) -> float:
        scores = [plane_scores[i] for i in indices if i < len(plane_scores)]
        return float(sum(scores) / len(scores)) if scores else 0.0

    single_rules: List[RuleCandidate] = []
    for idx in cand_plane_indices:
        bits = plane_bits[idx]
        size = _countbits(bits)
        if size < min_size:
            continue

        dims = tuple(sorted(set(planes[idx].dims)))
        tp = _countbits(_and_bits(bits, tmask))
        metrics_t = _compute_target_metrics_from_counts(size, tp, total_pos, N)
        single_rules.append(
            RuleCandidate(
                target_class=target_class,
                plane_indices=(idx,),
                dims=dims,
                mask_bits=bits,
                size=size,
                tp=tp,
                metrics=metrics_t,
            )
        )

    if not single_rules:
        return []

    good = [r for r in single_rules if r.metrics["lift_precision"] > lift_min]
    seed_pool = good if good else single_rules

    seed_pool.sort(
        key=lambda r: _rank_tuple_hessian(
            r.metrics,
            metric,
            _candidate_hessian(r.plane_indices),
            weight=hessian_weight,
        ),
        reverse=True,
    )
    beam = seed_pool[:beam_width]

    all_rules: Dict[Tuple[int, ...], RuleCandidate] = {r.plane_indices: r for r in beam}
    pos_in_cand = {pidx: j for j, pidx in enumerate(cand_plane_indices)}

    for depth in range(2, max_planes + 1):
        expansions: Dict[Tuple[int, ...], RuleCandidate] = {}

        for r in beam:
            last_pos = pos_in_cand.get(r.plane_indices[-1], -1)
            if last_pos < 0:
                continue

            for next_pos in range(last_pos + 1, len(cand_plane_indices)):
                nxt = cand_plane_indices[next_pos]
                if nxt in r.plane_indices:
                    continue

                new_planes = r.plane_indices + (nxt,)
                new_bits = _and_bits(r.mask_bits, plane_bits[nxt])
                size = _countbits(new_bits)
                if size < min_size:
                    continue

                new_dims = tuple(sorted(set(r.dims).union(planes[nxt].dims)))

                tp = _countbits(_and_bits(new_bits, tmask))
                metrics_t = _compute_target_metrics_from_counts(size, tp, total_pos, N)
                cand = RuleCandidate(
                    target_class=target_class,
                    plane_indices=new_planes,
                    dims=new_dims,
                    mask_bits=new_bits,
                    size=size,
                    tp=tp,
                    metrics=metrics_t,
                )

                expansions[new_planes] = cand

        if not expansions:
            break

        exp_list = list(expansions.values())
        exp_good = [x for x in exp_list if x.metrics["lift_precision"] > lift_min]
        exp_pool = exp_good if exp_good else exp_list

        exp_pool.sort(
            key=lambda r: _rank_tuple_hessian(
                r.metrics,
                metric,
                _candidate_hessian(r.plane_indices),
                weight=hessian_weight,
            ),
            reverse=True,
        )
        beam = exp_pool[:beam_width]

        for r in beam:
            prev = all_rules.get(r.plane_indices)
            if prev is None:
                all_rules[r.plane_indices] = r
                continue
            if _rank_tuple_hessian(
                r.metrics,
                metric,
                _candidate_hessian(r.plane_indices),
                weight=hessian_weight,
            ) > _rank_tuple_hessian(
                prev.metrics,
                metric,
                _candidate_hessian(prev.plane_indices),
                weight=hessian_weight,
            ):
                all_rules[r.plane_indices] = r

    out = list(all_rules.values())
    out.sort(
        key=lambda r: _rank_tuple_hessian(
            r.metrics,
            metric,
            _candidate_hessian(r.plane_indices),
            weight=hessian_weight,
        ),
        reverse=True,
    )
    return out


# =========================
# Pareto front (precision/recall/size)
# =========================


def _pareto_front(cands: List[RuleCandidate]) -> List[bool]:
    """
    Non-dominated across (precision, recall, size). A dominates B if
    precision>=, recall>=, size>= and at least one strictly >.
    """
    n = len(cands)
    if n == 0:
        return []

    points = np.array(
        [[c.metrics["precision"], c.metrics["recall"], c.metrics["size"]] for c in cands],
        dtype=float,
    )
    order = np.lexsort((-points[:, 2], -points[:, 1], -points[:, 0]))

    frontier_recalls: List[float] = []
    frontier_sizes: List[float] = []
    is_pareto = [False] * n

    for idx in order:
        recall = points[idx, 1]
        size = points[idx, 2]
        pos = bisect_left([-r for r in frontier_recalls], -recall)
        if pos > 0 and frontier_sizes[pos - 1] >= size:
            continue

        is_pareto[idx] = True
        frontier_recalls.insert(pos, recall)
        frontier_sizes.insert(pos, size)

        prune_pos = pos + 1
        while prune_pos < len(frontier_sizes):
            if frontier_sizes[prune_pos] <= size:
                del frontier_recalls[prune_pos]
                del frontier_sizes[prune_pos]
            else:
                break

    return is_pareto


# =========================
# Build planes from sel
# =========================


def _extract_planes_from_sel(sel: Dict[str, Any], d: int) -> List[Plane]:
    """
    Recolecta planos desde la estructura de `sel` (como find_low_dim_spaces):
      - by_pair_augmented[*].winning_planes (pares A/B)
      - regions_global.per_plane (globales)
      - winning_planes (fallback)

    Deduplica por oriented_plane_id y normaliza campos clave.
    """
    seen = set()
    planes: List[Plane] = []

    def _normalize_dims(n_norm: np.ndarray, dims_raw: Iterable[int]) -> Tuple[Tuple[int, ...], np.ndarray]:
        dims = tuple(int(dd) for dd in dims_raw)
        if not dims:
            if n_norm.size == d:
                dims = tuple(range(d))
            else:
                dims = tuple(int(i) for i in np.flatnonzero(n_norm))
        if not dims:
            dims = tuple(range(min(d, n_norm.size)))
        if n_norm.size == d and len(dims) <= d:
            return dims, np.asarray(n_norm, float)[list(dims)]
        if n_norm.size == len(dims):
            return dims, np.asarray(n_norm, float)
        return dims, np.asarray(n_norm, float)[list(dims)]

    def _origin_pair_from(entry: Dict[str, Any]) -> Tuple[int, int]:
        origin_pair = entry.get("origin_pair")
        if isinstance(origin_pair, (list, tuple)) and len(origin_pair) == 2:
            return (int(origin_pair[0]), int(origin_pair[1]))
        return (
            int(entry.get("origin_pair_a", 0)),
            int(entry.get("origin_pair_b", 0)),
        )

    def _metrics_by_class_from(entry: Dict[str, Any]) -> Dict[int, Dict[str, float]]:
        mbc_raw = entry.get("metrics_by_class", {})
        mbc: Dict[int, Dict[str, float]] = {}
        for k, v in mbc_raw.items():
            kk = int(k)
            mbc[kk] = {str(mn): float(mv) for mn, mv in v.items()}
        return mbc

    def _plane_from_entry(entry: Dict[str, Any]) -> Optional[Plane]:
        geom = entry.get("geometry", {})
        if geom is None:
            geom = {}
        n_raw = entry.get("n_norm")
        if n_raw is None:
            n_raw = entry.get("n")
        if n_raw is None:
            n_raw = geom.get("n")
        b_raw = entry.get("b_norm")
        if b_raw is None:
            b_raw = entry.get("b")
        if b_raw is None:
            b_raw = geom.get("b")
        if n_raw is None or b_raw is None:
            return None

        n_norm = np.asarray(n_raw, dtype=float).reshape(-1)
        b_arr = np.asarray(b_raw, dtype=float).reshape(-1)
        if b_arr.size == 0:
            return None
        b_norm = float(b_arr[0])
        dims_raw = entry.get("dims", [])
        dims, n_norm = _normalize_dims(n_norm, dims_raw)

        side = int(entry.get("side", geom.get("side", 1)))
        plane_id = entry.get("plane_id")
        opid = entry.get("oriented_plane_id")
        if opid is None or (isinstance(opid, str) and opid == ""):
            if plane_id is not None:
                opid = f"{plane_id}:{'≤' if side >= 0 else '≥'}"
            else:
                payload = np.hstack([n_norm.astype(float), np.array([b_norm, side], float)])
                opid = _md5_short(payload.tobytes())

        if opid in seen:
            return None
        seen.add(opid)

        ineq = entry.get("inequality", {})
        if ineq is None:
            ineq = {}
        ineq_general = str(ineq.get("general", opid))
        metrics_src = entry.get("stats", {}).get("metrics_by_class", {})
        mbc = _metrics_by_class_from(entry) or _metrics_by_class_from({"metrics_by_class": metrics_src})

        return Plane(
            oriented_plane_id=str(opid),
            plane_id=str(plane_id or str(opid).split(":")[0]),
            origin_pair=_origin_pair_from(entry),
            side=side,
            dims=dims,
            n_norm=n_norm,
            b_norm=b_norm,
            inequality_general=ineq_general,
            family_id=entry.get("family_id", None),
            metrics_by_class=mbc,
        )

    by_pair = sel.get("by_pair_augmented", {}) or {}
    for _, payload in by_pair.items():
        for entry in (payload.get("winning_planes", []) or []):
            plane = _plane_from_entry(entry)
            if plane is not None:
                planes.append(plane)

    regs = sel.get("regions_global", {}).get("per_plane", []) or []
    for entry in regs:
        plane = _plane_from_entry(entry)
        if plane is not None:
            planes.append(plane)

    for entry in (sel.get("winning_planes", []) or []):
        plane = _plane_from_entry(entry)
        if plane is not None:
            planes.append(plane)

    return planes


def _compute_plane_metrics_by_class(mask: np.ndarray, y: np.ndarray) -> Dict[int, Dict[str, float]]:
    classes = sorted(int(c) for c in np.unique(y).tolist())
    size = int(mask.sum())
    metrics_by_class: Dict[int, Dict[str, float]] = {}
    for c in classes:
        c_mask = y == c
        c_in = int(np.logical_and(mask, c_mask).sum())
        total_c = int(c_mask.sum())
        prec = (c_in / size) if size > 0 else 0.0
        rec = (c_in / total_c) if total_c > 0 else 0.0
        f1 = _f1(prec, rec) if (prec + rec) > 0 else 0.0
        baseline = (total_c / len(y)) if len(y) > 0 else 0.0
        lift = (prec / baseline) if baseline > 0 else 0.0
        metrics_by_class[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "lift": lift,
            "lift_precision": lift,
            "region_frac": (size / len(y)) if len(y) > 0 else 0.0,
        }
    return metrics_by_class


def _build_axis_plane(
    X: np.ndarray,
    y: np.ndarray,
    *,
    dim: int,
    threshold: float,
    plane_id: str,
    family_id: str,
) -> Plane:
    mask = X[:, dim] <= threshold
    metrics_by_class = _compute_plane_metrics_by_class(mask, y)
    return Plane(
        oriented_plane_id=f"{plane_id}:≤",
        plane_id=plane_id,
        origin_pair=(-1, -1),
        side=-1,
        dims=(int(dim),),
        n_norm=np.array([1.0]),
        b_norm=-float(threshold),
        inequality_general=f"x{dim} ≤ {threshold:.4f}",
        family_id=family_id,
        metrics_by_class=metrics_by_class,
    )


def _compute_grad_hessian_matrices(X: np.ndarray, y: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if X.size == 0:
        return None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        penalty="l2",
        C=0.8,
        solver="lbfgs",
        max_iter=200,
    )
    try:
        y_arr = np.asarray(y)
        clf.fit(Xs, y_arr)
    except Exception:
        return None
    probs = clf.predict_proba(Xs)
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(y_onehot.shape[0]), y_arr] = 1.0
    grad = probs - y_onehot
    grad_weights = np.linalg.norm(grad, axis=1)
    WX = Xs * grad_weights[:, None]
    G = Xs.T @ WX
    h_weights = np.sum(probs * (1 - probs), axis=1)
    HX = Xs * h_weights[:, None]
    H = Xs.T @ HX
    return G, H


def _plane_hessian_score(
    dims: Tuple[int, ...],
    G: np.ndarray,
    H: np.ndarray,
) -> float:
    if not dims:
        return 0.0
    if len(dims) == 1:
        idx = int(dims[0])
        return float(abs(H[idx, idx]))
    best = 0.0
    for i, di in enumerate(dims):
        for dj in dims[i + 1 :]:
            score = abs(G[di, dj]) + 0.35 * abs(H[di, dj])
            if score > best:
                best = score
    return float(best)


def _find_comb_dim_spaces_from_planes(
    planes: List[Plane],
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_planes: int = 7,
    metric: str = "precision",
    lift_min: float = 1.0,
    beam_width: int = 16,
    min_size: int = 5,
    max_candidates_per_class: int = 150,
    max_rules_per_class: int = 60,
    top_k_floor_per_dim: int = 12,
    include_masks: bool = False,
    projection_ref: str = "model_space",
    include_planes_used: bool = False,
) -> Dict[int, List[Dict[str, Any]]]:
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X debe ser 2D: (N, d)")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y debe ser 1D y del mismo N que X")

    N, d = X.shape
    classes = sorted(int(c) for c in np.unique(y).tolist())

    if not planes:
        return {}

    plane_bits: List[np.ndarray] = []
    dims_cache: Dict[Tuple[int, ...], np.ndarray] = {}
    for pl in planes:
        dims = pl.dims
        if dims not in dims_cache:
            dims_cache[dims] = X[:, dims]
        Xd = dims_cache[dims]
        expr = Xd @ pl.n_norm + float(pl.b_norm)
        s = pl.sign()
        if s == "≤":
            m = expr <= 1e-12
        else:
            m = expr >= -1e-12
        plane_bits.append(_packbits(m))

    packed_class_masks = {c: _packbits(y == c) for c in classes}
    class_sizes = {c: _countbits(mask) for c, mask in packed_class_masks.items()}

    all_rule_dicts: List[Dict[str, Any]] = []

    for target_class in classes:
        cands = _beam_search_and_rules(
            planes=planes,
            plane_bits=plane_bits,
            y=y,
            classes=classes,
            target_class=target_class,
            packed_class_masks=packed_class_masks,
            class_sizes=class_sizes,
            metric=metric,
            lift_min=lift_min,
            beam_width=beam_width,
            max_planes=max_planes,
            min_size=min_size,
            max_candidates=max_candidates_per_class,
        )

        cands = cands[:max_rules_per_class]
        if not cands:
            continue

        pareto_flags = _pareto_front(cands)

        region_ids: List[str] = []
        mask_sigs: List[str] = []
        for rc in cands:
            opids = tuple(planes[i].oriented_plane_id for i in rc.plane_indices)
            sig_str = f"c={target_class}|dims={rc.dims}|planes={','.join(opids)}|seed=beam_and"
            rid = "rg" + hashlib.md5(sig_str.encode("utf-8")).hexdigest()[:10]
            region_ids.append(rid)

            msig = _md5_short(
                rc.mask_bits.tobytes() + f"|c={target_class}".encode("utf-8"), n=12
            )
            mask_sigs.append(msig)

        best_by_sig: Dict[str, int] = {}
        for i, rc in enumerate(cands):
            s = mask_sigs[i]
            if s not in best_by_sig:
                best_by_sig[s] = i
            else:
                j = best_by_sig[s]
                if _rank_tuple(rc.metrics, metric) > _rank_tuple(cands[j].metrics, metric):
                    best_by_sig[s] = i

        keep_indices = sorted(
            best_by_sig.values(), key=lambda i: _rank_tuple(cands[i].metrics, metric), reverse=True
        )
        cands = [cands[i] for i in keep_indices]
        pareto_flags = [pareto_flags[i] for i in keep_indices]
        region_ids = [region_ids[i] for i in keep_indices]
        mask_sigs = [mask_sigs[i] for i in keep_indices]

        plane_sets = [set(rc.plane_indices) for rc in cands]
        generalizes = [[] for _ in cands]
        specializes = [[] for _ in cands]

        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j:
                    continue
                if plane_sets[i].issubset(plane_sets[j]) and len(plane_sets[i]) < len(
                    plane_sets[j]
                ):
                    generalizes[i].append(region_ids[j])
                    specializes[j].append(region_ids[i])

        parent_id: List[Optional[str]] = [None] * len(cands)
        deltas_to_parent: List[Optional[Dict[str, float]]] = [None] * len(cands)
        for j in range(len(cands)):
            parents = [
                i
                for i in range(len(cands))
                if plane_sets[i].issubset(plane_sets[j]) and len(plane_sets[i]) < len(plane_sets[j])
            ]
            if not parents:
                continue
            parents.sort(key=lambda i: (len(plane_sets[i]), _rank_tuple(cands[i].metrics, metric)), reverse=True)
            i = parents[0]
            parent_id[j] = region_ids[i]
            deltas_to_parent[j] = {
                "dF1": float(cands[j].metrics["f1"] - cands[i].metrics["f1"]),
                "dPrecision": float(cands[j].metrics["precision"] - cands[i].metrics["precision"]),
                "dRecall": float(cands[j].metrics["recall"] - cands[i].metrics["recall"]),
            }

        for idx, rc in enumerate(cands):
            metrics_t = rc.metrics
            region_frac = float(metrics_t["region_frac"])
            metrics_per_class = _compute_per_class_metrics(
                rc.mask_bits, packed_class_masks, class_sizes, rc.size, region_frac
            )
            region_summary = _compute_region_summary_from_counts(
                rc.size,
                rc.tp,
                class_sizes[target_class],
                N,
                float(metrics_t["acc"]),
                region_frac,
            )

            opids = tuple(planes[i].oriented_plane_id for i in rc.plane_indices)
            pieces = [planes[i].inequality_general for i in rc.plane_indices]
            rule_text = " AND ".join(pieces)

            sources = []
            fams = []
            for pi in rc.plane_indices:
                pl = planes[pi]
                fams.append(pl.family_id)
                sources.append(
                    {
                        "oriented_plane_id": pl.oriented_plane_id,
                        "plane_id": pl.plane_id,
                        "origin_pair": tuple(pl.origin_pair),
                        "family_id": pl.family_id,
                        "side": int(pl.side),
                        "dims": tuple(pl.dims),
                    }
                )

            fam_unique = sorted({str(f) for f in fams if f is not None})
            if len(fam_unique) == 1:
                family_id_val: Any = fam_unique[0]
            elif len(fam_unique) == 0:
                family_id_val = None
            else:
                family_id_val = ",".join(fam_unique)

            num_dims = int(len(rc.dims))
            num_planes = int(len(rc.plane_indices))

            rule_dict: Dict[str, Any] = {
                "region_id": region_ids[idx],
                "target_class": int(target_class),
                "dims": tuple(int(x) for x in rc.dims),
                "plane_ids": tuple(opids),
                "sources": sources,
                "rule_text": rule_text,
                "rule_pieces": pieces,
                "metrics": {
                    "size": int(metrics_t["size"]),
                    "precision": float(metrics_t["precision"]),
                    "recall": float(metrics_t["recall"]),
                    "f1": float(metrics_t["f1"]),
                    "baseline": float(metrics_t["baseline"]),
                    "lift_precision": float(metrics_t["lift_precision"]),
                },
                "metrics_per_class": metrics_per_class,
                "region_summary": region_summary,
                "projection_ref": str(projection_ref),
                "complexity": {
                    "num_dims": num_dims,
                    "num_planes": num_planes,
                },
                "is_floor": False,
                "generalizes": generalizes[idx],
                "specializes": specializes[idx],
                "is_pareto": bool(pareto_flags[idx]),
                "family_id": family_id_val,
                "parent_id": parent_id[idx],
                "deltas_to_parent": deltas_to_parent[idx],
                "planes_used": (sources if include_planes_used else []),
                "seed_type": "beam_and",
                "mask_signature": mask_sigs[idx],
                "_mask_bits": rc.mask_bits,
            }

            if include_masks:
                expanded = np.unpackbits(rc.mask_bits, bitorder="big")[:N].astype(bool)
                rule_dict["_mask"] = expanded

            all_rule_dicts.append(rule_dict)

    valuable: Dict[int, List[Dict[str, Any]]] = {}
    for rd in all_rule_dicts:
        k = int(rd["complexity"]["num_dims"])
        valuable.setdefault(k, []).append(rd)

    for k, rules in valuable.items():
        by_class: Dict[int, List[Dict[str, Any]]] = {}
        for r in rules:
            by_class.setdefault(int(r["target_class"]), []).append(r)

        for rr in by_class.values():
            rr.sort(
                key=lambda r: (
                    float(r["metrics"].get(metric, r["metrics"]["precision"])),
                    float(r["metrics"]["lift_precision"]),
                    float(r["metrics"]["size"]),
                ),
                reverse=True,
            )
            for r in rr[:top_k_floor_per_dim]:
                r["is_floor"] = True

    for k in list(valuable.keys()):
        valuable[k].sort(
            key=lambda r: (
                float(r["metrics"].get(metric, r["metrics"]["precision"])),
                float(r["metrics"]["lift_precision"]),
                float(r["metrics"]["size"]),
            ),
            reverse=True,
        )

    if not include_masks:
        for rules in valuable.values():
            for r in rules:
                if "_mask" in r:
                    del r["_mask"]
                if "_mask_bits" in r:
                    del r["_mask_bits"]

    return valuable


def _find_comb_dim_spaces_or_from_planes(
    planes: List[Plane],
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_planes: int = 7,
    metric: str = "precision",
    lift_min: float = 1.0,
    beam_width: int = 16,
    min_size: int = 5,
    max_candidates_per_class: int = 150,
    max_rules_per_class: int = 60,
    top_k_floor_per_dim: int = 12,
    include_masks: bool = False,
    projection_ref: str = "model_space",
    include_planes_used: bool = False,
) -> Dict[int, List[Dict[str, Any]]]:
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X debe ser 2D: (N, d)")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y debe ser 1D y del mismo N que X")

    N, d = X.shape
    classes = sorted(int(c) for c in np.unique(y).tolist())

    if not planes:
        return {}

    plane_bits: List[np.ndarray] = []
    dims_cache: Dict[Tuple[int, ...], np.ndarray] = {}
    for pl in planes:
        dims = pl.dims
        if dims not in dims_cache:
            dims_cache[dims] = X[:, dims]
        Xd = dims_cache[dims]
        expr = Xd @ pl.n_norm + float(pl.b_norm)
        s = pl.sign()
        if s == "≤":
            m = expr <= 1e-12
        else:
            m = expr >= -1e-12
        plane_bits.append(_packbits(m))

    packed_class_masks = {c: _packbits(y == c) for c in classes}
    class_sizes = {c: _countbits(mask) for c, mask in packed_class_masks.items()}

    all_rule_dicts: List[Dict[str, Any]] = []

    for target_class in classes:
        cands = _beam_search_or_rules(
            planes=planes,
            plane_bits=plane_bits,
            y=y,
            classes=classes,
            target_class=target_class,
            packed_class_masks=packed_class_masks,
            class_sizes=class_sizes,
            metric=metric,
            lift_min=lift_min,
            beam_width=beam_width,
            max_planes=max_planes,
            min_size=min_size,
            max_candidates=max_candidates_per_class,
        )

        cands = cands[:max_rules_per_class]
        if not cands:
            continue

        pareto_flags = _pareto_front(cands)

        region_ids: List[str] = []
        mask_sigs: List[str] = []
        for rc in cands:
            opids = tuple(planes[i].oriented_plane_id for i in rc.plane_indices)
            sig_str = f"c={target_class}|dims={rc.dims}|planes={','.join(opids)}|seed=beam_or"
            rid = "rg" + hashlib.md5(sig_str.encode("utf-8")).hexdigest()[:10]
            region_ids.append(rid)

            msig = _md5_short(
                rc.mask_bits.tobytes() + f"|c={target_class}".encode("utf-8"), n=12
            )
            mask_sigs.append(msig)

        best_by_sig: Dict[str, int] = {}
        for i, rc in enumerate(cands):
            s = mask_sigs[i]
            if s not in best_by_sig:
                best_by_sig[s] = i
            else:
                j = best_by_sig[s]
                if _rank_tuple(rc.metrics, metric) > _rank_tuple(cands[j].metrics, metric):
                    best_by_sig[s] = i

        keep_indices = sorted(
            best_by_sig.values(),
            key=lambda i: _rank_tuple(cands[i].metrics, metric),
            reverse=True,
        )
        cands = [cands[i] for i in keep_indices]
        pareto_flags = [pareto_flags[i] for i in keep_indices]
        region_ids = [region_ids[i] for i in keep_indices]
        mask_sigs = [mask_sigs[i] for i in keep_indices]

        plane_sets = [set(rc.plane_indices) for rc in cands]
        generalizes = [[] for _ in cands]
        specializes = [[] for _ in cands]

        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j:
                    continue
                if plane_sets[i].issubset(plane_sets[j]) and len(plane_sets[i]) < len(
                    plane_sets[j]
                ):
                    generalizes[j].append(region_ids[i])
                    specializes[i].append(region_ids[j])

        parent_id: List[Optional[str]] = [None] * len(cands)
        deltas_to_parent: List[Optional[Dict[str, float]]] = [None] * len(cands)
        for j in range(len(cands)):
            parents = [
                i
                for i in range(len(cands))
                if plane_sets[j].issubset(plane_sets[i]) and len(plane_sets[j]) < len(
                    plane_sets[i]
                )
            ]
            if not parents:
                continue
            min_len = min(len(plane_sets[i]) for i in parents)
            parents_min = [i for i in parents if len(plane_sets[i]) == min_len]
            parents_min.sort(key=lambda i: _rank_tuple(cands[i].metrics, metric), reverse=True)
            i = parents_min[0]
            parent_id[j] = region_ids[i]
            deltas_to_parent[j] = {
                "dF1": float(cands[j].metrics["f1"] - cands[i].metrics["f1"]),
                "dPrecision": float(cands[j].metrics["precision"] - cands[i].metrics["precision"]),
                "dRecall": float(cands[j].metrics["recall"] - cands[i].metrics["recall"]),
            }

        for idx, rc in enumerate(cands):
            metrics_t = rc.metrics
            region_frac = float(metrics_t["region_frac"])
            metrics_per_class = _compute_per_class_metrics(
                rc.mask_bits, packed_class_masks, class_sizes, rc.size, region_frac
            )
            region_summary = _compute_region_summary_from_counts(
                rc.size,
                rc.tp,
                class_sizes[target_class],
                N,
                float(metrics_t["acc"]),
                region_frac,
            )

            opids = tuple(planes[i].oriented_plane_id for i in rc.plane_indices)
            pieces = [planes[i].inequality_general for i in rc.plane_indices]
            rule_text = " OR ".join(pieces)

            sources = []
            fams = []
            for pi in rc.plane_indices:
                pl = planes[pi]
                fams.append(pl.family_id)
                sources.append(
                    {
                        "oriented_plane_id": pl.oriented_plane_id,
                        "plane_id": pl.plane_id,
                        "origin_pair": tuple(pl.origin_pair),
                        "family_id": pl.family_id,
                        "side": int(pl.side),
                        "dims": tuple(pl.dims),
                    }
                )

            fam_unique = sorted({str(f) for f in fams if f is not None})
            if len(fam_unique) == 1:
                family_id_val: Any = fam_unique[0]
            elif len(fam_unique) == 0:
                family_id_val = None
            else:
                family_id_val = ",".join(fam_unique)

            num_dims = int(len(rc.dims))
            num_planes = int(len(rc.plane_indices))

            rule_dict: Dict[str, Any] = {
                "region_id": region_ids[idx],
                "target_class": int(target_class),
                "dims": tuple(int(x) for x in rc.dims),
                "plane_ids": tuple(opids),
                "sources": sources,
                "rule_text": rule_text,
                "rule_pieces": pieces,
                "metrics": {
                    "size": int(metrics_t["size"]),
                    "precision": float(metrics_t["precision"]),
                    "recall": float(metrics_t["recall"]),
                    "f1": float(metrics_t["f1"]),
                    "baseline": float(metrics_t["baseline"]),
                    "lift_precision": float(metrics_t["lift_precision"]),
                },
                "metrics_per_class": metrics_per_class,
                "region_summary": region_summary,
                "projection_ref": str(projection_ref),
                "complexity": {
                    "num_dims": num_dims,
                    "num_planes": num_planes,
                },
                "is_floor": False,
                "generalizes": generalizes[idx],
                "specializes": specializes[idx],
                "is_pareto": bool(pareto_flags[idx]),
                "family_id": family_id_val,
                "parent_id": parent_id[idx],
                "deltas_to_parent": deltas_to_parent[idx],
                "planes_used": (sources if include_planes_used else []),
                "seed_type": "beam_or",
                "mask_signature": mask_sigs[idx],
                "_mask_bits": rc.mask_bits,
            }

            if include_masks:
                expanded = np.unpackbits(rc.mask_bits, bitorder="big")[:N].astype(bool)
                rule_dict["_mask"] = expanded

            all_rule_dicts.append(rule_dict)

    valuable: Dict[int, List[Dict[str, Any]]] = {}
    for rd in all_rule_dicts:
        k = int(rd["complexity"]["num_dims"])
        valuable.setdefault(k, []).append(rd)

    for k, rules in valuable.items():
        by_class: Dict[int, List[Dict[str, Any]]] = {}
        for r in rules:
            by_class.setdefault(int(r["target_class"]), []).append(r)

        for rr in by_class.values():
            rr.sort(
                key=lambda r: (
                    float(r["metrics"].get(metric, r["metrics"]["precision"])),
                    float(r["metrics"]["lift_precision"]),
                    float(r["metrics"]["size"]),
                ),
                reverse=True,
            )
            for r in rr[:top_k_floor_per_dim]:
                r["is_floor"] = True

    for k in list(valuable.keys()):
        valuable[k].sort(
            key=lambda r: (
                float(r["metrics"].get(metric, r["metrics"]["precision"])),
                float(r["metrics"]["lift_precision"]),
                float(r["metrics"]["size"]),
            ),
            reverse=True,
        )

    if not include_masks:
        for rules in valuable.values():
            for r in rules:
                if "_mask" in r:
                    del r["_mask"]
                if "_mask_bits" in r:
                    del r["_mask_bits"]

    return valuable


def _find_comb_dim_spaces_and_or_from_planes(
    planes: List[Plane],
    X: np.ndarray,
    y: np.ndarray,
    *,
    mode: str,
    max_planes: int = 7,
    metric: str = "precision",
    lift_min: float = 1.0,
    beam_width: int = 16,
    min_size: int = 5,
    max_candidates_per_class: int = 150,
    max_rules_per_class: int = 60,
    max_clause_candidates: int = 60,
    max_clauses: int = 4,
    clause_beam_width: int = 12,
    clause_iterations: int = 120,
    clause_diverse_topk: int = 40,
    clause_overlap_max: float = 0.8,
    max_dnf_rules_per_class: int = 5,
    top_k_floor_per_dim: int = 12,
    include_masks: bool = False,
    projection_ref: str = "model_space",
    include_planes_used: bool = False,
) -> Dict[int, List[Dict[str, Any]]]:
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X debe ser 2D: (N, d)")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y debe ser 1D y del mismo N que X")

    mode_normalized = (mode or "and_or_beam").strip().lower()
    if mode_normalized == "dnf":
        mode_normalized = "and_or_beam"

    N, d = X.shape
    classes = sorted(int(c) for c in np.unique(y).tolist())

    if not planes:
        return {}

    base_and = _find_comb_dim_spaces_from_planes(
        planes,
        X,
        y,
        max_planes=max_planes,
        metric=metric,
        lift_min=lift_min,
        beam_width=beam_width,
        min_size=min_size,
        max_candidates_per_class=max_candidates_per_class,
        max_rules_per_class=max_rules_per_class,
        top_k_floor_per_dim=top_k_floor_per_dim,
        include_masks=True,
        projection_ref=projection_ref,
        include_planes_used=True,
    )

    packed_class_masks = {c: _packbits(y == c) for c in classes}
    class_sizes = {c: _countbits(mask) for c, mask in packed_class_masks.items()}

    all_rule_dicts: List[Dict[str, Any]] = []

    for target_class in classes:
        class_rules = [r for rr in base_and.values() for r in rr if int(r["target_class"]) == target_class]
        if not class_rules:
            continue

        class_rules.sort(key=lambda r: _rank_rule_dict(r, metric), reverse=True)
        class_rules = class_rules[: max(1, int(max_clause_candidates))]

        if mode_normalized == "and_or_greedy":
            dnf_rules = _greedy_dnf_rules(
                class_rules,
                packed_class_masks,
                class_sizes,
                N=N,
                target_class=target_class,
                metric=metric,
                max_clauses=max_clauses,
                max_dnf_rules=max_dnf_rules_per_class,
            )
        elif mode_normalized == "and_or_beam":
            dnf_rules = _beam_search_dnf_rules(
                class_rules,
                packed_class_masks,
                class_sizes,
                N=N,
                target_class=target_class,
                metric=metric,
                max_clauses=max_clauses,
                beam_width=clause_beam_width,
            )
            if dnf_rules:
                dnf_rules.sort(key=lambda r: _rank_tuple(r.metrics, metric), reverse=True)
                dnf_rules = dnf_rules[: max(1, int(max_dnf_rules_per_class))]
        elif mode_normalized == "and_or_random":
            dnf_rules = _random_search_dnf_rules(
                class_rules,
                packed_class_masks,
                class_sizes,
                N=N,
                target_class=target_class,
                metric=metric,
                max_clauses=max_clauses,
                iterations=clause_iterations,
                seed=int(target_class),
                max_dnf_rules=max_dnf_rules_per_class,
            )
        elif mode_normalized == "and_or_diverse":
            dnf_rules = _diverse_topk_dnf_rules(
                class_rules,
                packed_class_masks,
                class_sizes,
                N=N,
                target_class=target_class,
                metric=metric,
                max_clauses=max_clauses,
                candidates_per_class=clause_diverse_topk,
                overlap_max=clause_overlap_max,
                max_dnf_rules=max_dnf_rules_per_class,
            )
        else:
            raise ValueError(f"Modo AND/OR no soportado: {mode}")

        if not dnf_rules:
            continue

        region_ids: List[str] = []
        mask_sigs: List[str] = []
        clause_id_sets: List[Tuple[str, ...]] = []
        for rc in dnf_rules:
            clause_rules = [class_rules[i] for i in rc.clause_indices]
            clause_ids = tuple(str(r["region_id"]) for r in clause_rules)
            clause_id_sets.append(clause_ids)
            sig_str = (
                f"c={target_class}|dims={rc.dims}|clauses={','.join(clause_ids)}|seed={mode_normalized}"
            )
            rid = "rg" + hashlib.md5(sig_str.encode("utf-8")).hexdigest()[:10]
            region_ids.append(rid)

            msig = _md5_short(
                rc.mask_bits.tobytes() + f"|c={target_class}".encode("utf-8"), n=12
            )
            mask_sigs.append(msig)

        plane_sets = [set(rc.clause_indices) for rc in dnf_rules]
        generalizes = [[] for _ in dnf_rules]
        specializes = [[] for _ in dnf_rules]

        for i in range(len(dnf_rules)):
            for j in range(len(dnf_rules)):
                if i == j:
                    continue
                if plane_sets[i].issubset(plane_sets[j]) and len(plane_sets[i]) < len(
                    plane_sets[j]
                ):
                    generalizes[j].append(region_ids[i])
                    specializes[i].append(region_ids[j])

        parent_id: List[Optional[str]] = [None] * len(dnf_rules)
        deltas_to_parent: List[Optional[Dict[str, float]]] = [None] * len(dnf_rules)
        for j in range(len(dnf_rules)):
            parents = [
                i
                for i in range(len(dnf_rules))
                if plane_sets[j].issubset(plane_sets[i]) and len(plane_sets[j]) < len(
                    plane_sets[i]
                )
            ]
            if not parents:
                continue
            min_len = min(len(plane_sets[i]) for i in parents)
            parents_min = [i for i in parents if len(plane_sets[i]) == min_len]
            parents_min.sort(key=lambda i: _rank_tuple(dnf_rules[i].metrics, metric), reverse=True)
            i = parents_min[0]
            parent_id[j] = region_ids[i]
            deltas_to_parent[j] = {
                "dF1": float(dnf_rules[j].metrics["f1"] - dnf_rules[i].metrics["f1"]),
                "dPrecision": float(
                    dnf_rules[j].metrics["precision"] - dnf_rules[i].metrics["precision"]
                ),
                "dRecall": float(dnf_rules[j].metrics["recall"] - dnf_rules[i].metrics["recall"]),
            }

        for idx, rc in enumerate(dnf_rules):
            metrics_t = rc.metrics
            region_frac = float(metrics_t["region_frac"])
            metrics_per_class = _compute_per_class_metrics(
                rc.mask_bits, packed_class_masks, class_sizes, rc.size, region_frac
            )
            region_summary = _compute_region_summary_from_counts(
                rc.size,
                rc.tp,
                class_sizes[target_class],
                N,
                float(metrics_t["acc"]),
                region_frac,
            )

            clause_rules = [class_rules[i] for i in rc.clause_indices]
            clause_texts = [str(r["rule_text"]) for r in clause_rules]
            clause_texts_wrapped = [
                f"({txt})" if " AND " in txt else txt for txt in clause_texts
            ]
            rule_text = " OR ".join(clause_texts_wrapped)

            sources = []
            for r in clause_rules:
                sources.extend(r.get("sources") or [])

            plane_ids = tuple(r.get("plane_ids") for r in clause_rules)
            num_planes = int(sum(int(r["complexity"]["num_planes"]) for r in clause_rules))
            num_dims = int(len(rc.dims))

            rule_dict: Dict[str, Any] = {
                "region_id": region_ids[idx],
                "target_class": int(target_class),
                "dims": tuple(int(x) for x in rc.dims),
                "plane_ids": plane_ids,
                "sources": sources,
                "rule_text": rule_text,
                "rule_pieces": clause_texts,
                "clauses": clause_id_sets[idx],
                "metrics": {
                    "size": int(metrics_t["size"]),
                    "precision": float(metrics_t["precision"]),
                    "recall": float(metrics_t["recall"]),
                    "f1": float(metrics_t["f1"]),
                    "baseline": float(metrics_t["baseline"]),
                    "lift_precision": float(metrics_t["lift_precision"]),
                },
                "metrics_per_class": metrics_per_class,
                "region_summary": region_summary,
                "projection_ref": str(projection_ref),
                "complexity": {
                    "num_dims": num_dims,
                    "num_planes": num_planes,
                    "num_clauses": int(len(rc.clause_indices)),
                },
                "is_floor": False,
                "generalizes": generalizes[idx],
                "specializes": specializes[idx],
                "is_pareto": False,
                "family_id": None,
                "parent_id": parent_id[idx],
                "deltas_to_parent": deltas_to_parent[idx],
                "planes_used": (sources if include_planes_used else []),
                "seed_type": mode_normalized,
                "mask_signature": mask_sigs[idx],
                "_mask_bits": rc.mask_bits,
            }

            if include_masks:
                expanded = np.unpackbits(rc.mask_bits, bitorder="big")[:N].astype(bool)
                rule_dict["_mask"] = expanded

            all_rule_dicts.append(rule_dict)

    valuable: Dict[int, List[Dict[str, Any]]] = {}
    for rd in all_rule_dicts:
        k = int(rd["complexity"]["num_dims"])
        valuable.setdefault(k, []).append(rd)

    for k, rules in valuable.items():
        by_class: Dict[int, List[Dict[str, Any]]] = {}
        for r in rules:
            by_class.setdefault(int(r["target_class"]), []).append(r)

        for rr in by_class.values():
            rr.sort(
                key=lambda r: (
                    float(r["metrics"].get(metric, r["metrics"]["precision"])),
                    float(r["metrics"]["lift_precision"]),
                    float(r["metrics"]["size"]),
                ),
                reverse=True,
            )
            for r in rr[:top_k_floor_per_dim]:
                r["is_floor"] = True

    for k in list(valuable.keys()):
        valuable[k].sort(
            key=lambda r: (
                float(r["metrics"].get(metric, r["metrics"]["precision"])),
                float(r["metrics"]["lift_precision"]),
                float(r["metrics"]["size"]),
            ),
            reverse=True,
        )

    if not include_masks:
        for rules in valuable.values():
            for r in rules:
                if "_mask" in r:
                    del r["_mask"]
                if "_mask_bits" in r:
                    del r["_mask_bits"]

    return valuable


# =========================
# Public API: find_comb_dim_spaces
# =========================


def find_comb_dim_spaces(
    sel: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    mode: str = "base",
    max_planes: int = 12,
    metric: str = "f1",
    lift_min: float = 1.0,
    beam_width: int = 36,
    min_size: int = 5,
    max_candidates_per_class: int = 150,
    max_rules_per_class: int = 480,
    top_k_floor_per_dim: int = 12,
    include_masks: bool = False,
    projection_ref: str = "model_space",
    include_planes_used: bool = False,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Construye `valuable` agrupado por num_dims (k):
      valuable[k] = [rule_dict, ...]
    Cada rule_dict sigue tu estructura (campos que no aplican se dejan None o vacíos).

    OR se maneja como "múltiples reglas AND": la lista de reglas por clase actúa como OR implícito.
    Modos soportados: "base", "hessian_rank", "hessian_filter".
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X debe ser 2D: (N, d)")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y debe ser 1D y del mismo N que X")

    mode_normalized = (mode or "base").strip().lower()
    if mode_normalized == "hessian_rank":
        return find_comb_dim_spaces_hessian_rank(
            sel,
            X,
            y,
            max_planes=max_planes,
            metric=metric,
            lift_min=lift_min,
            beam_width=beam_width,
            min_size=min_size,
            max_candidates_per_class=max_candidates_per_class,
            max_rules_per_class=max_rules_per_class,
            top_k_floor_per_dim=top_k_floor_per_dim,
            include_masks=include_masks,
            projection_ref=projection_ref,
            include_planes_used=include_planes_used,
        )
    if mode_normalized == "hessian_filter":
        return find_comb_dim_spaces_hessian_filter(
            sel,
            X,
            y,
            max_planes=max_planes,
            metric=metric,
            lift_min=lift_min,
            beam_width=beam_width,
            min_size=min_size,
            max_candidates_per_class=max_candidates_per_class,
            max_rules_per_class=max_rules_per_class,
            top_k_floor_per_dim=top_k_floor_per_dim,
            include_masks=include_masks,
            projection_ref=projection_ref,
            include_planes_used=include_planes_used,
        )
    if mode_normalized not in {"base", "default"}:
        raise ValueError(
            "mode debe ser 'base', 'hessian_rank' o 'hessian_filter'. "
            f"Recibido: {mode}"
        )

    d = X.shape[1]
    planes = _extract_planes_from_sel(sel, d)
    return _find_comb_dim_spaces_from_planes(
        planes,
        X,
        y,
        max_planes=max_planes,
        metric=metric,
        lift_min=lift_min,
        beam_width=beam_width,
        min_size=min_size,
        max_candidates_per_class=max_candidates_per_class,
        max_rules_per_class=max_rules_per_class,
        top_k_floor_per_dim=top_k_floor_per_dim,
        include_masks=include_masks,
        projection_ref=projection_ref,
        include_planes_used=include_planes_used,
    )


def find_comb_dim_spaces_full(
    sel: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    mode: str = "base",
    max_planes: int = 12,
    metric: str = "f1",
    lift_min: float = 1.0,
    beam_width: int = 36,
    min_size: int = 5,
    max_candidates_per_class: int = 150,
    max_rules_per_class: int = 480,
    max_clause_candidates: int = 60,
    max_clauses: int = 4,
    clause_beam_width: int = 12,
    clause_iterations: int = 120,
    clause_diverse_topk: int = 40,
    clause_overlap_max: float = 0.8,
    max_dnf_rules_per_class: int = 5,
    top_k_floor_per_dim: int = 12,
    include_masks: bool = False,
    projection_ref: str = "model_space",
    include_planes_used: bool = False,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Punto de entrada único con todos los modos disponibles:

    - "base", "default", "hessian_rank", "hessian_filter": variantes AND clásicas.
    - "and": alias AND clásico (equivalente a "base").
    - "or": reglas OR (unión de planos).
    - "dnf": alias de "and_or_beam" (OR de cláusulas AND vía beam search).
    - "and_or_greedy": OR de cláusulas AND con selección greedy.
    - "and_or_beam": OR de cláusulas AND con búsqueda beam.
    - "and_or_random": OR de cláusulas AND con muestreo aleatorio.
    - "and_or_diverse": OR de cláusulas AND con selección diversa.
    """
    mode_normalized = (mode or "base").strip().lower()

    if mode_normalized in {"base", "default", "hessian_rank", "hessian_filter"}:
        return find_comb_dim_spaces(
            sel,
            X,
            y,
            mode=mode_normalized,
            max_planes=max_planes,
            metric=metric,
            lift_min=lift_min,
            beam_width=beam_width,
            min_size=min_size,
            max_candidates_per_class=max_candidates_per_class,
            max_rules_per_class=max_rules_per_class,
            top_k_floor_per_dim=top_k_floor_per_dim,
            include_masks=include_masks,
            projection_ref=projection_ref,
            include_planes_used=include_planes_used,
        )

    d = X.shape[1]
    planes = _extract_planes_from_sel(sel, d)

    if mode_normalized == "or":
        return _find_comb_dim_spaces_or_from_planes(
            planes,
            X,
            y,
            max_planes=max_planes,
            metric=metric,
            lift_min=lift_min,
            beam_width=beam_width,
            min_size=min_size,
            max_candidates_per_class=max_candidates_per_class,
            max_rules_per_class=max_rules_per_class,
            top_k_floor_per_dim=top_k_floor_per_dim,
            include_masks=include_masks,
            projection_ref=projection_ref,
            include_planes_used=include_planes_used,
        )

    if mode_normalized == "and":
        return _find_comb_dim_spaces_from_planes(
            planes,
            X,
            y,
            max_planes=max_planes,
            metric=metric,
            lift_min=lift_min,
            beam_width=beam_width,
            min_size=min_size,
            max_candidates_per_class=max_candidates_per_class,
            max_rules_per_class=max_rules_per_class,
            top_k_floor_per_dim=top_k_floor_per_dim,
            include_masks=include_masks,
            projection_ref=projection_ref,
            include_planes_used=include_planes_used,
        )

    if mode_normalized in {
        "dnf",
        "and_or_greedy",
        "and_or_beam",
        "and_or_random",
        "and_or_diverse",
    }:
        return _find_comb_dim_spaces_and_or_from_planes(
            planes,
            X,
            y,
            mode=mode_normalized,
            max_planes=max_planes,
            metric=metric,
            lift_min=lift_min,
            beam_width=beam_width,
            min_size=min_size,
            max_candidates_per_class=max_candidates_per_class,
            max_rules_per_class=max_rules_per_class,
            max_clause_candidates=max_clause_candidates,
            max_clauses=max_clauses,
            clause_beam_width=clause_beam_width,
            clause_iterations=clause_iterations,
            clause_diverse_topk=clause_diverse_topk,
            clause_overlap_max=clause_overlap_max,
            max_dnf_rules_per_class=max_dnf_rules_per_class,
            top_k_floor_per_dim=top_k_floor_per_dim,
            include_masks=include_masks,
            projection_ref=projection_ref,
            include_planes_used=include_planes_used,
        )

    raise ValueError(
        "mode debe ser 'base', 'default', 'hessian_rank', 'hessian_filter', 'and', "
        "'or', 'dnf', 'and_or_greedy', 'and_or_beam', 'and_or_random' o 'and_or_diverse'. "
        f"Recibido: {mode}"
    )


def find_comb_dim_spaces_hessian_seed(
    sel: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_planes: int = 7,
    metric: str = "precision",
    lift_min: float = 1.0,
    beam_width: int = 16,
    min_size: int = 5,
    max_candidates_per_class: int = 150,
    max_rules_per_class: int = 60,
    top_k_floor_per_dim: int = 12,
    include_masks: bool = False,
    projection_ref: str = "model_space",
    include_planes_used: bool = False,
    top_pairs: int = 12,
    seed_quantiles: Tuple[float, ...] = (0.25, 0.5, 0.75),
) -> Dict[int, List[Dict[str, Any]]]:
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X debe ser 2D: (N, d)")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y debe ser 1D y del mismo N que X")

    d = X.shape[1]
    planes = _extract_planes_from_sel(sel, d)
    gh = _compute_grad_hessian_matrices(X, y)
    if gh is None:
        return _find_comb_dim_spaces_from_planes(
            planes,
            X,
            y,
            max_planes=max_planes,
            metric=metric,
            lift_min=lift_min,
            beam_width=beam_width,
            min_size=min_size,
            max_candidates_per_class=max_candidates_per_class,
            max_rules_per_class=max_rules_per_class,
            top_k_floor_per_dim=top_k_floor_per_dim,
            include_masks=include_masks,
            projection_ref=projection_ref,
            include_planes_used=include_planes_used,
        )
    G, H = gh
    pair_scores: List[Tuple[float, Tuple[int, int]]] = []
    for i in range(d):
        for j in range(i + 1, d):
            score = abs(G[i, j]) + 0.35 * abs(H[i, j])
            pair_scores.append((score, (i, j)))
    pair_scores.sort(reverse=True)
    top_pairs = max(1, int(top_pairs))
    selected_dims = sorted({idx for _, pair in pair_scores[:top_pairs] for idx in pair})
    extra_planes: List[Plane] = []
    for dim in selected_dims:
        values = X[:, dim]
        for q in seed_quantiles:
            threshold = float(np.quantile(values, q))
            plane_id = f"hess_seed_f{dim}_q{int(q * 100):02d}"
            extra_planes.append(
                _build_axis_plane(
                    X,
                    y,
                    dim=dim,
                    threshold=threshold,
                    plane_id=plane_id,
                    family_id="hessian_seed",
                )
            )
    planes = planes + extra_planes
    return _find_comb_dim_spaces_from_planes(
        planes,
        X,
        y,
        max_planes=max_planes,
        metric=metric,
        lift_min=lift_min,
        beam_width=beam_width,
        min_size=min_size,
        max_candidates_per_class=max_candidates_per_class,
        max_rules_per_class=max_rules_per_class,
        top_k_floor_per_dim=top_k_floor_per_dim,
        include_masks=include_masks,
        projection_ref=projection_ref,
        include_planes_used=include_planes_used,
    )


def find_comb_dim_spaces_hessian_rank(
    sel: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_planes: int = 7,
    metric: str = "precision",
    lift_min: float = 1.0,
    beam_width: int = 16,
    min_size: int = 5,
    max_candidates_per_class: int = 150,
    max_rules_per_class: int = 60,
    top_k_floor_per_dim: int = 12,
    include_masks: bool = False,
    projection_ref: str = "model_space",
    include_planes_used: bool = False,
    hessian_weight: float = 0.2,
) -> Dict[int, List[Dict[str, Any]]]:
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X debe ser 2D: (N, d)")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y debe ser 1D y del mismo N que X")

    d = X.shape[1]
    planes = _extract_planes_from_sel(sel, d)
    if not planes:
        return {}
    gh = _compute_grad_hessian_matrices(X, y)
    if gh is None:
        return _find_comb_dim_spaces_from_planes(
            planes,
            X,
            y,
            max_planes=max_planes,
            metric=metric,
            lift_min=lift_min,
            beam_width=beam_width,
            min_size=min_size,
            max_candidates_per_class=max_candidates_per_class,
            max_rules_per_class=max_rules_per_class,
            top_k_floor_per_dim=top_k_floor_per_dim,
            include_masks=include_masks,
            projection_ref=projection_ref,
            include_planes_used=include_planes_used,
        )
    G, H = gh
    plane_scores = [_plane_hessian_score(pl.dims, G, H) for pl in planes]
    max_score = max(plane_scores) if plane_scores else 0.0
    if max_score > 0:
        plane_scores = [score / max_score for score in plane_scores]
    plane_bits: List[np.ndarray] = []
    dims_cache: Dict[Tuple[int, ...], np.ndarray] = {}
    for pl in planes:
        dims = pl.dims
        if dims not in dims_cache:
            dims_cache[dims] = X[:, dims]
        Xd = dims_cache[dims]
        expr = Xd @ pl.n_norm + float(pl.b_norm)
        s = pl.sign()
        if s == "≤":
            m = expr <= 1e-12
        else:
            m = expr >= -1e-12
        plane_bits.append(_packbits(m))

    classes = sorted(int(c) for c in np.unique(y).tolist())
    packed_class_masks = {c: _packbits(y == c) for c in classes}
    class_sizes = {c: _countbits(mask) for c, mask in packed_class_masks.items()}

    all_rule_dicts: List[Dict[str, Any]] = []
    for target_class in classes:
        cands = _beam_search_and_rules_hessian(
            planes=planes,
            plane_bits=plane_bits,
            plane_scores=plane_scores,
            y=y,
            classes=classes,
            target_class=target_class,
            packed_class_masks=packed_class_masks,
            class_sizes=class_sizes,
            metric=metric,
            lift_min=lift_min,
            beam_width=beam_width,
            max_planes=max_planes,
            min_size=min_size,
            max_candidates=max_candidates_per_class,
            hessian_weight=hessian_weight,
        )

        cands = cands[:max_rules_per_class]
        if not cands:
            continue

        pareto_flags = _pareto_front(cands)

        region_ids: List[str] = []
        mask_sigs: List[str] = []
        for rc in cands:
            opids = tuple(planes[i].oriented_plane_id for i in rc.plane_indices)
            sig_str = f"c={target_class}|dims={rc.dims}|planes={','.join(opids)}|seed=beam_and_hessian"
            rid = "rg" + hashlib.md5(sig_str.encode("utf-8")).hexdigest()[:10]
            region_ids.append(rid)

            msig = _md5_short(
                rc.mask_bits.tobytes() + f"|c={target_class}".encode("utf-8"), n=12
            )
            mask_sigs.append(msig)

        best_by_sig: Dict[str, int] = {}
        for i, rc in enumerate(cands):
            s = mask_sigs[i]
            if s not in best_by_sig:
                best_by_sig[s] = i
            else:
                j = best_by_sig[s]
                if _rank_tuple_hessian(
                    rc.metrics,
                    metric,
                    float(sum(plane_scores[idx] for idx in rc.plane_indices)) / len(rc.plane_indices),
                    weight=hessian_weight,
                ) > _rank_tuple_hessian(
                    cands[j].metrics,
                    metric,
                    float(sum(plane_scores[idx] for idx in cands[j].plane_indices))
                    / len(cands[j].plane_indices),
                    weight=hessian_weight,
                ):
                    best_by_sig[s] = i

        keep_indices = sorted(
            best_by_sig.values(),
            key=lambda i: _rank_tuple_hessian(
                cands[i].metrics,
                metric,
                float(sum(plane_scores[idx] for idx in cands[i].plane_indices)) / len(cands[i].plane_indices),
                weight=hessian_weight,
            ),
            reverse=True,
        )
        cands = [cands[i] for i in keep_indices]
        pareto_flags = [pareto_flags[i] for i in keep_indices]
        region_ids = [region_ids[i] for i in keep_indices]
        mask_sigs = [mask_sigs[i] for i in keep_indices]

        plane_sets = [set(rc.plane_indices) for rc in cands]
        generalizes = [[] for _ in cands]
        specializes = [[] for _ in cands]

        for i in range(len(cands)):
            for j in range(len(cands)):
                if i == j:
                    continue
                if plane_sets[i].issubset(plane_sets[j]) and len(plane_sets[i]) < len(
                    plane_sets[j]
                ):
                    generalizes[i].append(region_ids[j])
                    specializes[j].append(region_ids[i])

        parent_id: List[Optional[str]] = [None] * len(cands)
        deltas_to_parent: List[Optional[Dict[str, float]]] = [None] * len(cands)
        for j in range(len(cands)):
            parents = [
                i
                for i in range(len(cands))
                if plane_sets[i].issubset(plane_sets[j]) and len(plane_sets[i]) < len(plane_sets[j])
            ]
            if not parents:
                continue
            parents.sort(
                key=lambda i: (
                    len(plane_sets[i]),
                    _rank_tuple_hessian(
                        cands[i].metrics,
                        metric,
                        float(sum(plane_scores[idx] for idx in cands[i].plane_indices))
                        / len(cands[i].plane_indices),
                        weight=hessian_weight,
                    ),
                ),
                reverse=True,
            )
            i = parents[0]
            parent_id[j] = region_ids[i]
            deltas_to_parent[j] = {
                "dF1": float(cands[j].metrics["f1"] - cands[i].metrics["f1"]),
                "dPrecision": float(cands[j].metrics["precision"] - cands[i].metrics["precision"]),
                "dRecall": float(cands[j].metrics["recall"] - cands[i].metrics["recall"]),
            }

        for idx, rc in enumerate(cands):
            metrics_t = rc.metrics
            region_frac = float(metrics_t["region_frac"])
            metrics_per_class = _compute_per_class_metrics(
                rc.mask_bits, packed_class_masks, class_sizes, rc.size, region_frac
            )
            region_summary = _compute_region_summary_from_counts(
                rc.size,
                rc.tp,
                class_sizes[target_class],
                X.shape[0],
                float(metrics_t["acc"]),
                region_frac,
            )

            opids = tuple(planes[i].oriented_plane_id for i in rc.plane_indices)
            pieces = [planes[i].inequality_general for i in rc.plane_indices]
            rule_text = " AND ".join(pieces)

            sources = []
            fams = []
            for pi in rc.plane_indices:
                pl = planes[pi]
                fams.append(pl.family_id)
                sources.append(
                    {
                        "oriented_plane_id": pl.oriented_plane_id,
                        "plane_id": pl.plane_id,
                        "origin_pair": tuple(pl.origin_pair),
                        "family_id": pl.family_id,
                        "side": int(pl.side),
                        "dims": tuple(pl.dims),
                    }
                )

            fam_unique = sorted({str(f) for f in fams if f is not None})
            if len(fam_unique) == 1:
                family_id_val: Any = fam_unique[0]
            elif len(fam_unique) == 0:
                family_id_val = None
            else:
                family_id_val = ",".join(fam_unique)

            num_dims = int(len(rc.dims))
            num_planes = int(len(rc.plane_indices))

            rule_dict: Dict[str, Any] = {
                "region_id": region_ids[idx],
                "target_class": int(target_class),
                "dims": tuple(int(x) for x in rc.dims),
                "plane_ids": tuple(opids),
                "sources": sources,
                "rule_text": rule_text,
                "rule_pieces": pieces,
                "metrics": {
                    "size": int(metrics_t["size"]),
                    "precision": float(metrics_t["precision"]),
                    "recall": float(metrics_t["recall"]),
                    "f1": float(metrics_t["f1"]),
                    "baseline": float(metrics_t["baseline"]),
                    "lift_precision": float(metrics_t["lift_precision"]),
                },
                "metrics_per_class": metrics_per_class,
                "region_summary": region_summary,
                "projection_ref": str(projection_ref),
                "complexity": {
                    "num_dims": num_dims,
                    "num_planes": num_planes,
                },
                "is_floor": False,
                "generalizes": generalizes[idx],
                "specializes": specializes[idx],
                "is_pareto": bool(pareto_flags[idx]),
                "family_id": family_id_val,
                "parent_id": parent_id[idx],
                "deltas_to_parent": deltas_to_parent[idx],
                "planes_used": (sources if include_planes_used else []),
                "seed_type": "beam_and_hessian",
                "mask_signature": mask_sigs[idx],
                "_mask_bits": rc.mask_bits,
            }

            if include_masks:
                expanded = np.unpackbits(rc.mask_bits, bitorder="big")[: X.shape[0]].astype(bool)
                rule_dict["_mask"] = expanded

            all_rule_dicts.append(rule_dict)

    valuable: Dict[int, List[Dict[str, Any]]] = {}
    for rd in all_rule_dicts:
        k = int(rd["complexity"]["num_dims"])
        valuable.setdefault(k, []).append(rd)

    for k, rules in valuable.items():
        by_class: Dict[int, List[Dict[str, Any]]] = {}
        for r in rules:
            by_class.setdefault(int(r["target_class"]), []).append(r)

        for rr in by_class.values():
            rr.sort(
                key=lambda r: (
                    float(r["metrics"].get(metric, r["metrics"]["precision"])),
                    float(r["metrics"]["lift_precision"]),
                    float(r["metrics"]["size"]),
                ),
                reverse=True,
            )
            for r in rr[:top_k_floor_per_dim]:
                r["is_floor"] = True

    for k in list(valuable.keys()):
        valuable[k].sort(
            key=lambda r: (
                float(r["metrics"].get(metric, r["metrics"]["precision"])),
                float(r["metrics"]["lift_precision"]),
                float(r["metrics"]["size"]),
            ),
            reverse=True,
        )

    if not include_masks:
        for rules in valuable.values():
            for r in rules:
                if "_mask" in r:
                    del r["_mask"]
                if "_mask_bits" in r:
                    del r["_mask_bits"]

    return valuable


def find_comb_dim_spaces_hessian_filter(
    sel: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_planes: int = 7,
    metric: str = "precision",
    lift_min: float = 1.0,
    beam_width: int = 16,
    min_size: int = 5,
    max_candidates_per_class: int = 150,
    max_rules_per_class: int = 60,
    top_k_floor_per_dim: int = 12,
    include_masks: bool = False,
    projection_ref: str = "model_space",
    include_planes_used: bool = False,
    keep_frac: float = 0.7,
    min_keep: int = 30,
) -> Dict[int, List[Dict[str, Any]]]:
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError("X debe ser 2D: (N, d)")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y debe ser 1D y del mismo N que X")

    d = X.shape[1]
    planes = _extract_planes_from_sel(sel, d)
    if not planes:
        return {}
    gh = _compute_grad_hessian_matrices(X, y)
    if gh is None:
        return _find_comb_dim_spaces_from_planes(
            planes,
            X,
            y,
            max_planes=max_planes,
            metric=metric,
            lift_min=lift_min,
            beam_width=beam_width,
            min_size=min_size,
            max_candidates_per_class=max_candidates_per_class,
            max_rules_per_class=max_rules_per_class,
            top_k_floor_per_dim=top_k_floor_per_dim,
            include_masks=include_masks,
            projection_ref=projection_ref,
            include_planes_used=include_planes_used,
        )
    G, H = gh
    plane_scores = [_plane_hessian_score(pl.dims, G, H) for pl in planes]
    if not plane_scores:
        return _find_comb_dim_spaces_from_planes(
            planes,
            X,
            y,
            max_planes=max_planes,
            metric=metric,
            lift_min=lift_min,
            beam_width=beam_width,
            min_size=min_size,
            max_candidates_per_class=max_candidates_per_class,
            max_rules_per_class=max_rules_per_class,
            top_k_floor_per_dim=top_k_floor_per_dim,
            include_masks=include_masks,
            projection_ref=projection_ref,
            include_planes_used=include_planes_used,
        )
    scores = np.array(plane_scores, dtype=float)
    threshold = float(np.quantile(scores, max(0.0, min(1.0, 1.0 - keep_frac))))
    keep_indices = [i for i, s in enumerate(scores) if s >= threshold]
    if len(keep_indices) < min_keep:
        keep_indices = list(np.argsort(-scores)[: min_keep])
    filtered_planes = [planes[i] for i in keep_indices]
    return _find_comb_dim_spaces_from_planes(
        filtered_planes,
        X,
        y,
        max_planes=max_planes,
        metric=metric,
        lift_min=lift_min,
        beam_width=beam_width,
        min_size=min_size,
        max_candidates_per_class=max_candidates_per_class,
        max_rules_per_class=max_rules_per_class,
        top_k_floor_per_dim=top_k_floor_per_dim,
        include_masks=include_masks,
        projection_ref=projection_ref,
        include_planes_used=include_planes_used,
    )


# =========================
# (Opcional) helper para OR explícito como "ruleset"
# =========================


def select_ruleset_or_greedy(
    rules: List[Dict[str, Any]],
    *,
    metric: str = "f1",
    max_total_planes: int = 7,
) -> List[str]:
    """
    Selecciona un subconjunto de reglas AND (cláusulas) que actuarán como OR implícito.
    Devuelve region_ids elegidos. Usa greedy sobre la métrica (sin recomputar masks aquí).
    Recomendación: úsalo como "post-proceso" para quedarte con pocas reglas que funcionen en conjunto.
    """
    # Nota: aquí NO recomputamos uniones reales (TP/FP) porque necesitaríamos _mask_bits y y.
    # Si quieres OR real con unión de máscaras, lo hacemos, pero mejor integrarlo con X,y y bitsets.
    # Por ahora: greedy simple en score individual + penalización por complejidad.
    chosen: List[str] = []
    used_planes = 0

    rr = sorted(
        rules,
        key=lambda r: (
            float(r["metrics"].get(metric, r["metrics"]["f1"])),
            float(r["metrics"]["lift_precision"]),
            float(r["metrics"]["size"]),
        ),
        reverse=True,
    )

    for r in rr:
        npl = int(r["complexity"]["num_planes"])
        if used_planes + npl > max_total_planes:
            continue
        chosen.append(str(r["region_id"]))
        used_planes += npl
        if used_planes >= max_total_planes:
            break

    return chosen


# =========================
# Visualización
# =========================


def plot_rule_metrics(
    valuable: Dict[int, List[Dict[str, Any]]],
    *,
    target_class: Optional[int] = None,
    metric_x: str = "recall",
    metric_y: str = "precision",
    size_metric: str = "size",
    cmap: str = "viridis",
    figsize: Tuple[float, float] = (8.0, 6.0),
    highlight_pareto: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Grafica reglas encontradas (por ejemplo con :func:`find_comb_dim_spaces`) en un scatter
    de ``metric_x`` vs ``metric_y``. El color representa el número de dimensiones y el
    tamaño el ``size_metric`` (p. ej. tamaño de la región).

    Parameters
    ----------
    valuable:
        Salida de :func:`find_comb_dim_spaces` (dict con reglas agrupadas por número de dimensiones).
    target_class:
        Si se indica, filtra las reglas a esa clase. Por defecto se muestran todas.
    metric_x, metric_y:
        Métricas a mostrar en los ejes. Deben existir en el diccionario ``metrics`` de cada regla.
    size_metric:
        Métrica que controla el tamaño del marker. No se normaliza; usa valores directos.
    cmap:
        Paleta de color para codificar ``num_dims``.
    figsize:
        Tamaño de la figura cuando no se proporciona ``ax``.
    highlight_pareto:
        Si es ``True``, marca reglas con ``is_pareto`` en una capa superior.
    ax:
        Eje de Matplotlib existente. Si es ``None``, se crea uno nuevo.

    Returns
    -------
    matplotlib.axes.Axes
        Eje con el scatter plot.
    """

    import matplotlib.pyplot as plt

    rules: List[Dict[str, Any]] = []
    for _, bucket in valuable.items():
        for rule in bucket:
            if target_class is None or int(rule.get("target_class", -1)) == int(target_class):
                rules.append(rule)

    if not rules:
        raise ValueError("No hay reglas para graficar con los filtros proporcionados.")

    xs = [float(r["metrics"].get(metric_x, 0.0)) for r in rules]
    ys = [float(r["metrics"].get(metric_y, 0.0)) for r in rules]
    sizes = [float(r["metrics"].get(size_metric, 1.0)) for r in rules]
    dims = [int(r.get("complexity", {}).get("num_dims", 0)) for r in rules]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    scatter = ax.scatter(
        xs,
        ys,
        c=dims,
        s=sizes,
        cmap=cmap,
        alpha=0.75,
        edgecolors="k",
        linewidths=0.5,
    )

    if highlight_pareto:
        pareto_x = []
        pareto_y = []
        pareto_sizes = []
        for r, x, y, s in zip(rules, xs, ys, sizes):
            if r.get("is_pareto"):
                pareto_x.append(x)
                pareto_y.append(y)
                pareto_sizes.append(max(s, 40.0))
        if pareto_x:
            ax.scatter(
                pareto_x,
                pareto_y,
                s=pareto_sizes,
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
                marker="o",
                label="Pareto",
            )
            ax.legend()

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Número de dimensiones")

    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    ax.set_title("Reglas descubiertas")
    ax.grid(True, linestyle="--", alpha=0.3)

    return ax
