from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import hashlib
import numpy as np


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


def _extract_planes_from_sel(sel: Dict[str, Any]) -> List[Plane]:
    """
    Usa sel['winning_planes'] (global) y deduplica por oriented_plane_id.
    Espera que cada entry tenga: oriented_plane_id, plane_id, origin_pair, side, dims, n_norm, b_norm, inequality, metrics_by_class, family_id.
    """
    raw = sel.get("winning_planes", [])
    seen = set()
    planes: List[Plane] = []

    for pl in raw:
        opid = pl["oriented_plane_id"]
        if opid in seen:
            continue
        seen.add(opid)

        dims = tuple(int(d) for d in pl.get("dims", []))
        n_norm = np.asarray(pl["n_norm"], dtype=float)
        b_norm = float(pl["b_norm"])
        origin_pair = pl.get("origin_pair")
        if isinstance(origin_pair, (list, tuple)) and len(origin_pair) == 2:
            origin_pair = (int(origin_pair[0]), int(origin_pair[1]))
        else:
            origin_pair = (
                int(pl.get("origin_pair_a", 0)),
                int(pl.get("origin_pair_b", 0)),
            )

        # metrics_by_class keys can be strings
        mbc_raw = pl.get("metrics_by_class", {})
        mbc: Dict[int, Dict[str, float]] = {}
        for k, v in mbc_raw.items():
            kk = int(k)
            mbc[kk] = {str(mn): float(mv) for mn, mv in v.items()}

        ineq = pl.get("inequality", {}) or {}
        ineq_general = str(ineq.get("general", opid))

        planes.append(
            Plane(
                oriented_plane_id=str(opid),
                plane_id=str(pl.get("plane_id", opid.split(":")[0])),
                origin_pair=origin_pair,
                side=int(pl.get("side", 1)),
                dims=dims,
                n_norm=n_norm,
                b_norm=b_norm,
                inequality_general=ineq_general,
                family_id=pl.get("family_id", None),
                metrics_by_class=mbc,
            )
        )

    return planes


# =========================
# Public API: find_comb_dim_spaces
# =========================


def find_comb_dim_spaces(
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
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Construye `valuable` agrupado por num_dims (k):
      valuable[k] = [rule_dict, ...]
    Cada rule_dict sigue tu estructura (campos que no aplican se dejan None o vacíos).

    OR se maneja como "múltiples reglas AND": la lista de reglas por clase actúa como OR implícito.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X debe ser 2D: (N, d)")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y debe ser 1D y del mismo N que X")

    N, d = X.shape
    classes = sorted(int(c) for c in np.unique(y).tolist())

    planes = _extract_planes_from_sel(sel)
    if not planes:
        return {}

    # Precompute plane masks (packed)
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

    # Precompute class packed masks once
    packed_class_masks = {c: _packbits(y == c) for c in classes}
    class_sizes = {c: _countbits(mask) for c, mask in packed_class_masks.items()}

    # Collect candidates per class via beam search
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

        # Keep top rules per class
        cands = cands[:max_rules_per_class]
        if not cands:
            continue

        # Pareto labels
        pareto_flags = _pareto_front(cands)

        # Build region_ids first (stable)
        region_ids: List[str] = []
        mask_sigs: List[str] = []
        for rc in cands:
            # signature based on target + oriented_plane_ids + dims
            opids = tuple(planes[i].oriented_plane_id for i in rc.plane_indices)
            sig_str = f"c={target_class}|dims={rc.dims}|planes={','.join(opids)}|seed=beam_and"
            rid = "rg" + hashlib.md5(sig_str.encode("utf-8")).hexdigest()[:10]
            region_ids.append(rid)

            msig = _md5_short(
                rc.mask_bits.tobytes() + f"|c={target_class}".encode("utf-8"), n=12
            )
            mask_sigs.append(msig)

        # Dedup by mask_signature (keep best ranked)
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

        # Rebuild inclusion relations based on plane-set inclusion (aprox jerárquico)
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

        # Choose parent_id as the "closest" generalizing rule with max |planes|
        parent_id: List[Optional[str]] = [None] * len(cands)
        deltas_to_parent: List[Optional[Dict[str, float]]] = [None] * len(cands)
        for j in range(len(cands)):
            # candidates i that generalize j => plane_sets[i] subset of plane_sets[j]
            parents = [
                i
                for i in range(len(cands))
                if plane_sets[i].issubset(plane_sets[j]) and len(plane_sets[i]) < len(plane_sets[j])
            ]
            if not parents:
                continue
            # pick the largest subset (closest parent), tie-break by rank
            parents.sort(key=lambda i: (len(plane_sets[i]), _rank_tuple(cands[i].metrics, metric)), reverse=True)
            i = parents[0]
            parent_id[j] = region_ids[i]
            deltas_to_parent[j] = {
                "delta_precision": float(cands[j].metrics["precision"] - cands[i].metrics["precision"]),
                "delta_recall": float(cands[j].metrics["recall"] - cands[i].metrics["recall"]),
                "delta_f1": float(cands[j].metrics["f1"] - cands[i].metrics["f1"]),
                "delta_size": float(cands[j].metrics["size"] - cands[i].metrics["size"]),
                "delta_lift_precision": float(
                    cands[j].metrics["lift_precision"] - cands[i].metrics["lift_precision"]
                ),
            }

        # Mark floors: top-k per (class, num_dims)
        # We'll do later globally; but we can compute here as well.
        # For now, we store and finalize floors after grouping.
        for idx, rc in enumerate(cands):
            # compute full metrics and summaries now (auditable)
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

            # sources: references to original planes
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
                family_id_val = tuple(fam_unique)

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
                "is_floor": False,  # se setea al final por (class, dim)
                "generalizes": generalizes[idx],
                "specializes": specializes[idx],
                "is_pareto": bool(pareto_flags[idx]),
                "family_id": family_id_val,
                "parent_id": parent_id[idx],
                "deltas_to_parent": deltas_to_parent[idx],
                "planes_used": ([sources] if include_planes_used else []),  # opcional
                "seed_type": "beam_and",
                "mask_signature": mask_sigs[idx],
                "_mask_bits": rc.mask_bits,
            }

            if include_masks:
                # expand boolean mask
                # unpackbits returns multiple of 8; truncate to N
                expanded = np.unpackbits(rc.mask_bits, bitorder="big")[:N].astype(bool)
                rule_dict["_mask"] = expanded

            all_rule_dicts.append(rule_dict)

    # ---- Group into valuable[k]
    valuable: Dict[int, List[Dict[str, Any]]] = {}
    for rd in all_rule_dicts:
        k = int(rd["complexity"]["num_dims"])
        valuable.setdefault(k, []).append(rd)

    # ---- Mark floors: top-k per (target_class, k)
    for k, rules in valuable.items():
        by_class: Dict[int, List[Dict[str, Any]]] = {}
        for r in rules:
            by_class.setdefault(int(r["target_class"]), []).append(r)

        for c, rr in by_class.items():
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

    # Optional: sort each k bucket for readability
    for k in list(valuable.keys()):
        valuable[k].sort(
            key=lambda r: (
                float(r["metrics"].get(metric, r["metrics"]["precision"])),
                float(r["metrics"]["lift_precision"]),
                float(r["metrics"]["size"]),
            ),
            reverse=True,
        )

    return valuable


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
