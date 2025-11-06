"""Rutinas avanzadas para la poda y orientación de planos DelDel."""

# -*- coding: utf-8 -*-
"""
Unified-first plane evaluation & selection for interpretable half-space rules.

Autor: JC + ChatGPT (GPT-5 Thinking)
Fecha: 2025-10-18

Resumen:
- Normaliza todos los planos (n,b) -> (û, b̂) con ||û||=1
- Genera medio-espacios (dos lados por plano) con oriented_plane_id
- Calcula métricas OVR por clase (precision/recall/F1/balacc, lift, coverage, purity, entropy)
- Colapsa duplicados en "familias" de medio-espacios (paralelos ~idénticos, distancia normalizada, soporte-coef. similar)
- Filtro global (min_region_size, min_abs_diff, min_rel_lift, min_purity) y Top-K por clase
- Opcional two-stage (sketch→refine) para performance
- Construye regiones globales y luego hace selección greedy por par (A,B) con penalización por familia
- Mantiene salida compatible y enriquece con nuevos campos

Notas:
- Este módulo NO depende de librerías externas (solo numpy).
- Está pensado para N grande (1e6 x 30) con batching; ajusta batch_size según memoria.
"""

from typing import Dict, Tuple, Any, List, Optional, Iterable
import math
import numpy as np


# ============================================================
# Utilidades numéricas y de texto
# ============================================================

_EPS = 1e-12


def _safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return int(x.item()) if hasattr(x, "item") else int(x)


def _normed(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).reshape(-1)
    n = float(np.linalg.norm(v)) + _EPS
    return v / n


def _normalize_plane(n: np.ndarray, b: float) -> Tuple[np.ndarray, float]:
    u = _normed(n)
    nb = float(b) / (float(np.linalg.norm(n)) + _EPS)
    return u, nb


def _cosabs(n1: np.ndarray, n2: np.ndarray) -> float:
    return float(abs(_normed(n1) @ _normed(n2)))


def _parallel_plane_distance_normed(u1: np.ndarray, b1n: float,
                                    u2: np.ndarray, b2n: float) -> float:
    """Distancia entre hiperplanos con normales unitarias (signo alineado)."""
    s = 1.0 if (u1 @ u2) >= 0 else -1.0
    return abs(b1n - s * b2n)


def _coef_support(n: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.asarray(n, float).reshape(-1)
    return (np.abs(n) >= eps)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return float(inter) / float(max(1, union))


def _cosine_on_support(n1: np.ndarray, n2: np.ndarray, mask: np.ndarray) -> float:
    v1 = np.asarray(n1, float).reshape(-1)[mask]
    v2 = np.asarray(n2, float).reshape(-1)[mask]
    if v1.size == 0 or v2.size == 0:
        return 0.0
    a = float(np.dot(v1, v2))
    b = float(np.linalg.norm(v1)) * float(np.linalg.norm(v2)) + _EPS
    return a / b


def _ineq_text_general(n: np.ndarray, b: float, side: int,
                       feat_names: Optional[List[str]] = None, tol: float = 1e-12) -> str:
    s = "≤" if side >= 0 else "≥"
    terms = []
    for j, coef in enumerate(np.asarray(n, float)):
        if abs(coef) < tol:
            continue
        name = feat_names[j] if (feat_names and j < len(feat_names)) else f"x{j}"
        if coef == 1.0:
            terms.append(f"{name}")
        elif coef == -1.0:
            terms.append(f"-{name}")
        else:
            terms.append(f"{coef:.6g}·{name}")
    left = " + ".join(terms) if terms else "0"
    if abs(b) > tol:
        left = f"{left} + {float(b):.6g}"
    return f"{left} {s} 0"


def _ineq_text_2d(n: np.ndarray, b: float, side: int,
                  dims: Tuple[int, int] = (0, 1),
                  feat_names: Optional[List[str]] = None,
                  tol_other: float = 1e-6) -> str:
    d = n.size
    i, j = int(dims[0]), int(dims[1])
    other = [k for k in range(d) if k not in (i, j)]
    if len(other) > 0 and np.sum(np.abs(n[other])) > tol_other:
        return _ineq_text_general(n, b, side, feat_names)
    xi = feat_names[i] if (feat_names and i < len(feat_names)) else f"x{i}"
    yj = feat_names[j] if (feat_names and j < len(feat_names)) else f"x{j}"
    if abs(n[j]) > 1e-12:
        m = -float(n[i] / n[j])
        c = -float(b / n[j])
        leq = (side >= 0 and n[j] > 0) or (side < 0 and n[j] < 0)
        return f"{yj} {'≤' if leq else '≥'} {m:.6g}·{xi} + {c:.6g}"
    else:
        if abs(n[i]) < 1e-12:
            return _ineq_text_general(n, b, side, feat_names)
        xcut = -float(b / n[i])
        leq = (side >= 0 and n[i] > 0) or (side < 0 and n[i] < 0)
        return f"{xi} {'≤' if leq else '≥'} {xcut:.6g}"


# ============================================================
# Métricas por clase (OVR) y auxiliares
# ============================================================


def _baseline_props(y: np.ndarray) -> Dict[int, float]:
    y = np.asarray(y, int).reshape(-1)
    labels, counts = np.unique(y, return_counts=True)
    N = float(y.size)
    return {int(c): float(n) / max(1.0, N) for c, n in zip(labels, counts)}


def _region_props(y_region: np.ndarray) -> Dict[int, float]:
    if y_region.size == 0:
        return {}
    labels, counts = np.unique(y_region, return_counts=True)
    N = float(y_region.size)
    return {int(c): float(n) / N for c, n in zip(labels, counts)}


def _entropy(p: Dict[int, float]) -> float:
    s = 0.0
    for v in p.values():
        if v > 0:
            s -= v * math.log(v + _EPS)
    return float(s)


def _metrics_counts(pred: np.ndarray, y: np.ndarray, pos_label: int) -> Dict[str, int]:
    y = np.asarray(y, int).reshape(-1)
    pred = np.asarray(pred, bool).reshape(-1)
    ya = (y == int(pos_label))
    tp = int((pred & ya).sum())
    fn = int((~pred & ya).sum())
    fp = int((pred & ~ya).sum())
    tn = int((~pred & ~ya).sum())
    return dict(tp=tp, fn=fn, fp=fp, tn=tn)


def _metrics_binary_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    N = float(tp + fp + fn + tn)
    acc = float(tp + tn) / max(1.0, N)
    prec = float(tp) / max(1.0, tp + fp)
    rec = float(tp) / max(1.0, tp + fn)
    f1 = (2 * prec * rec) / max(_EPS, (prec + rec)) if (prec + rec) > 0 else 0.0
    # balanced accuracy
    tpr = float(tp) / max(1.0, tp + fn)
    tnr = float(tn) / max(1.0, tn + fp)
    bal = 0.5 * (tpr + tnr)
    return dict(acc=acc, precision=prec, recall=rec, f1=f1, balacc=bal)


def _ovr_metrics_for_side(y: np.ndarray,
                          mask_region: np.ndarray,
                          labels_all: List[int],
                          baseline: Dict[int, float]) -> Dict[int, Dict[str, float]]:
    out = {}
    for c in labels_all:
        cnts = _metrics_counts(mask_region, y, pos_label=int(c))
        m = _metrics_binary_from_counts(**cnts)
        # coverage & purity
        N = float(y.size)
        size = float(mask_region.sum())
        prop_c_region = float(((y == int(c)) & mask_region).sum()) / max(1.0, size)
        coverage_c = float(((y == int(c)) & mask_region).sum()) / max(1.0, float((y == int(c)).sum()))
        purity = prop_c_region
        base_c = float(baseline.get(int(c), 0.0))
        lift = (prop_c_region / base_c) if base_c > 0 else (float("inf") if prop_c_region > 0 else 0.0)
        out[int(c)] = dict(
            **m,
            coverage=coverage_c,
            purity=purity,
            lift=lift,
            region_size=int(size),
            region_frac=(size / max(1.0, N))
        )
    return out


def _score_by_class(metrics: Dict[int, Dict[str, float]],
                    w1: float = 0.5, w2: float = 0.4, w3: float = 0.1, alpha: float = 0.6) -> Tuple[int, float, Dict[int, float]]:
    """Devuelve (best_class, best_score, scores_dict_por_clase)."""
    scores = {}
    for c, m in metrics.items():
        s = (w1 * m["balacc"]) + (w2 * (m["lift"] * (m["coverage"] ** alpha))) + (w3 * m["purity"])
        scores[int(c)] = float(s)
    best_c = int(max(scores, key=lambda k: scores[k])) if scores else None
    best_s = float(scores.get(best_c, 0.0)) if best_c is not None else 0.0
    return best_c, best_s, scores


# ============================================================
# Batch masks para muchos medio-espacios
# ============================================================


def _halfspace_masks_from_H(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Dado H = X@U.T + b (shape NxM), devuelve (mask_leq, mask_geq) booleanas."""
    return (H <= 0.0), (H >= 0.0)


def _batched_project(X: np.ndarray, U: np.ndarray, bvec: np.ndarray, batch: int) -> Iterable[np.ndarray]:
    """
    Itera proyectando X contra U en bloques (columnas).
    U: (d, M_total), bvec: (M_total,)
    Yields H_block: (N, m) para m<=batch
    """
    M = U.shape[1]
    off = 0
    while off < M:
        k = min(batch, M - off)
        Ub = U[:, off:off + k]          # (d, k)
        bb = bvec[off:off + k].reshape(1, -1)  # (1, k)
        H = X @ Ub + bb               # (N, k)
        yield off, off + k, H
        off += k


# ============================================================
# Recolector de planos desde `res` (compute_frontier_planes_all_modes)
# ============================================================


def _collect_planes_for_pair(res_pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    plist = []
    bylab = res_pair.get("planes_by_label", {}) or {}
    pid = 0
    for lab, lst in bylab.items():
        for p in lst:
            q = dict(p)
            q["label"] = int(lab)
            q["_pid_local"] = pid
            plist.append(q)
            pid += 1
    return plist


def _collect_all_planes(res: Dict[Tuple[int, int], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Devuelve lista de todos los planos con (n,b,origin_pair,meta)."""
    allp = []
    for pair, payload in res.items():
        a, b = int(pair[0]), int(pair[1])
        plist = _collect_planes_for_pair(payload)
        for p in plist:
            r = dict(p)
            r["origin_pair"] = (a, b)
            allp.append(r)
    return allp


# ============================================================
# Familias de medio-espacios (colapso muy estricto)
# ============================================================


def _same_family(
    n1: np.ndarray, b1: float, n2: np.ndarray, b2: float,
    *,
    cos_parallel: float = 0.999,
    tau_mult: float = 0.5,
    coef_eps: float = 1e-8,
    jaccard_support: float = 0.80,
    coef_cos_min: float = 0.995
) -> bool:
    """
    Decide si (n1,b1) y (n2,b2) (normales y b ya normalizados) pertenecen a la misma familia.
    Requiere superar tres checks:
      1) Paralelismo extremo: |cos| >= cos_parallel
      2) Distancia normalizada pequeña: |b1 - s*b2| <= tau, con s = sign(u1·u2)
         donde tau = tau_mult * tau_ref. Usamos tau_ref=1e-2 por defecto (fijo y estricto).
      3) Similitud de soporte y coseno en soporte: Jaccard >= jaccard_support y cos >= coef_cos_min
    """
    u1, b1n = _normalize_plane(n1, b1)
    u2, b2n = _normalize_plane(n2, b2)

    cos = abs(u1 @ u2)
    if cos < cos_parallel:
        return False

    # Distancia de hiperplanos normalizados
    dist = _parallel_plane_distance_normed(u1, b1n, u2, b2n)
    tau_ref = 1e-2  # muy estricto, ajustable si lo deseas
    if dist > (tau_mult * tau_ref):
        return False

    # Soportes y coseno en soporte
    s1 = _coef_support(n1, eps=coef_eps)
    s2 = _coef_support(n2, eps=coef_eps)
    if _jaccard(s1, s2) < jaccard_support:
        return False
    mask = np.logical_or(s1, s2)
    if _cosine_on_support(n1, n2, mask) < coef_cos_min:
        return False

    return True


# ============================================================
# Selección greedy por par (A,B) con penalización por familia
# ============================================================


def _pred_halfspace(X: np.ndarray, n: np.ndarray, b: float, side: int) -> np.ndarray:
    # side = +1  ->  n·x + b ≤ 0  (clase A dentro)
    # side = -1  ->  n·x + b ≥ 0
    h = X @ n.reshape(-1) + float(b)
    return (h <= 0.0) if side >= 0 else (h >= 0.0)


def _balanced_accuracy_pair(predA: np.ndarray, y_ab: np.ndarray, a: int, b: int) -> float:
    ya = (y_ab == int(a))
    yb = (y_ab == int(b))
    if ya.sum() == 0 or yb.sum() == 0:
        return float((predA == (y_ab == a)).mean()) if y_ab.size > 0 else 0.0
    tpr = float((predA & ya).sum()) / float(ya.sum() + _EPS)
    tnr = float((~predA & yb).sum()) / float(yb.sum() + _EPS)
    return 0.5 * (tpr + tnr)


def _metrics_pair_all(predA: np.ndarray, y_ab: np.ndarray, a: int, b: int) -> Dict[str, float]:
    ya = (y_ab == int(a))
    yb = (y_ab == int(b))
    tp = int((predA & ya).sum())
    tn = int((~predA & yb).sum())
    fp = int((predA & yb).sum())
    fn = int((~predA & ya).sum())
    acc = float((tp + tn) / max(1, y_ab.size))
    prec = float(tp / max(1, tp + fp))
    rec = float(tp / max(1, tp + fn))
    f1 = float((2 * prec * rec) / max(_EPS, prec + rec)) if (prec + rec) > 0 else 0.0
    bal = _balanced_accuracy_pair(predA, y_ab, a, b)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc, precision=prec, recall=rec, f1=f1, balacc=bal)


def _diversity_bonus_for_candidate(p: Dict[str, Any],
                                   selected: List[Dict[str, Any]],
                                   *,
                                   w_ortho: float = 0.03,
                                   w_parallel_sep: float = 0.02,
                                   ortho_cos_thresh: float = 0.25,
                                   parallel_cos_thresh: float = 0.98,
                                   parallel_sep_tau: float = 1.25) -> float:
    if not selected:
        return 0.0

    n = np.asarray(p["n_norm"], float)  # usar normalizada aquí
    b = float(p["b_norm"])
    # ortogonalidad
    cos_list = [abs(n @ np.asarray(s["n_norm"], float)) for s in selected]
    cos_max = max(cos_list) if cos_list else 0.0
    ortho_term = max(0.0, 1.0 - cos_max)

    # paralelos separados
    par_terms = []
    for s in selected:
        cos = abs(n @ np.asarray(s["n_norm"], float))
        if cos >= parallel_cos_thresh:
            dist = _parallel_plane_distance_normed(n, b, np.asarray(s["n_norm"], float), float(s["b_norm"]))
            if dist >= parallel_sep_tau * 1.0:  # 1.0 es umbral base en espacio normalizado
                par_terms.append(min(1.5, dist / (parallel_sep_tau * 1.0)))
    par_term = max(par_terms) if par_terms else 0.0

    if cos_max > (1.0 - ortho_cos_thresh):
        ortho_term *= 0.5

    return w_ortho * ortho_term + w_parallel_sep * par_term


def _greedy_and_selection_by_pair(
    planes_for_pair: List[Dict[str, Any]],
    Xab: np.ndarray, yab: np.ndarray, a: int, b: int,
    *,
    max_k: int = 5,
    min_improve: float = 1e-3,
    min_recall: float = 0.85,
    min_region_frac: float = 0.20,
    lambda_complexity: float = 0.0,
    diversity_additions: bool = True,
    max_diverse_add: int = 1,
    ortho_add_cos_thresh: float = 0.2,
    family_penalty: float = 0.02,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Selección greedy (AND) condicionado con bonus de diversidad y penalización por familia repetida.
    Trabaja sobre medio-espacios ya orientados y normalizados (campos: n_norm, b_norm, side, family_id).
    """
    # 1) Mejor individual
    best_item = None
    best_score = -1.0
    for p in planes_for_pair:
        pred = _pred_halfspace(Xab, p["n_norm"], p["b_norm"], side=int(p["side"]))
        sc = _balanced_accuracy_pair(pred, yab, a, b)
        if sc > best_score:
            best_score = sc
            best_item = (p, pred, sc)

    if best_item is None:
        return [], dict(balacc=0.0)

    selected = []
    used_ids = set()
    used_families = set()
    cur_pred = best_item[1].copy()
    cur_score = float(best_item[2])

    p0 = dict(best_item[0])
    p0["gain"] = float(cur_score)
    p0["metrics_single"] = _metrics_pair_all(best_item[1], yab, a, b)
    selected.append(p0)
    used_ids.add(p0["oriented_plane_id"])
    if p0.get("family_id") is not None:
        used_families.add(p0["family_id"])

    # 2) Iteraciones greedy
    while len(selected) < max_k:
        best_next = None
        best_gain_adj = 0.0
        for p in planes_for_pair:
            if p["oriented_plane_id"] in used_ids:
                continue
            pred_cand = cur_pred & _pred_halfspace(Xab, p["n_norm"], p["b_norm"], side=int(p["side"]))
            metrics = _metrics_pair_all(pred_cand, yab, a, b)
            if metrics["recall"] < min_recall:
                continue
            if pred_cand.sum() < min_region_frac * max(1, cur_pred.sum()):
                continue
            cand_score = _balanced_accuracy_pair(pred_cand, yab, a, b)
            div_bonus = _diversity_bonus_for_candidate(p, selected)

            fam_pen = 0.0
            if (p.get("family_id") is not None) and (p["family_id"] in used_families):
                fam_pen = family_penalty

            gain_adj = (cand_score - cur_score) + div_bonus - fam_pen - float(lambda_complexity)

            if gain_adj > best_gain_adj + 1e-12:
                best_gain_adj = gain_adj
                best_next = (p, pred_cand, cand_score, metrics, div_bonus, fam_pen)

        if best_next is None or (best_gain_adj < float(min_improve)):
            break

        p, cur_pred, cur_score, metrics, div_bonus, fam_pen = best_next
        q = dict(p)
        q["gain"] = float(best_gain_adj)
        q["metrics_cumulative"] = metrics
        selected.append(q)
        used_ids.add(p["oriented_plane_id"])
        if p.get("family_id") is not None:
            used_families.add(p["family_id"])

    overall = _metrics_pair_all(cur_pred, yab, a, b)

    # 3) Adiciones por diversidad (opcionales)
    if diversity_additions and (len(selected) < max_k):
        normals_sel = [np.asarray(s["n_norm"], float) for s in selected]

        def cos_to_sel(n):
            if not normals_sel:
                return 0.0
            return min([abs(np.asarray(n, float) @ u) for u in normals_sel])

        pool = [p for p in planes_for_pair if p["oriented_plane_id"] not in used_ids]
        pool.sort(key=lambda p: cos_to_sel(p["n_norm"]))  # más ortogonal primero
        added = 0
        for p in pool:
            if added >= int(max_diverse_add):
                break
            if cos_to_sel(p["n_norm"]) > ortho_add_cos_thresh:
                continue
            pred_cand = cur_pred & _pred_halfspace(Xab, p["n_norm"], p["b_norm"], side=int(p["side"]))
            metrics = _metrics_pair_all(pred_cand, yab, a, b)
            cand_score = _balanced_accuracy_pair(pred_cand, yab, a, b)
            if (cand_score + 1e-12) < cur_score:
                continue
            if metrics["recall"] < min_recall or pred_cand.sum() < min_region_frac * max(1, cur_pred.sum()):
                continue
            cur_pred = pred_cand
            cur_score = cand_score
            q = dict(p)
            q["gain"] = float(0.0)
            q["metrics_cumulative"] = metrics
            selected.append(q)
            used_ids.add(p["oriented_plane_id"])
            added += 1

        overall = _metrics_pair_all(cur_pred, yab, a, b)

    return selected, overall


# ============================================================
# API principal: Unified-first + selección por par
# ============================================================


def prune_and_orient_planes_unified_globalmaj(
    res: Dict[Tuple[int, int], Dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    *,
    # ---------- Unified-first params ----------
    feature_names: Optional[List[str]] = None,
    dims_for_text: Tuple[int, int] = (0, 1),
    global_sides: str = "opposite",          # {"opposite","both","good"}
    sketch_frac: float = 1.0,                # <1.0 activa two-stage
    refine_topK_per_class: int = 200,
    batch_size: int = 64,                    # columnas (medio-espacios) por batch para X@U
    # Familia (colapso muy estricto)
    cos_parallel: float = 0.999,
    tau_mult: float = 0.5,
    coef_eps: float = 1e-8,
    jaccard_support: float = 0.80,
    coef_cos_min: float = 0.995,
    # Guardarraíles globales
    min_region_size: int = 30,
    min_abs_diff: float = 0.05,
    min_rel_lift: float = 0.25,
    min_purity: float = 0.0,
    # Scoring
    w1_balacc: float = 0.5, w2_lift_cov: float = 0.4, w3_purity: float = 0.1, alpha_cov: float = 0.6,
    # ---------- Selección por par ----------
    max_k: int = 5,
    min_improve: float = 1e-3,
    min_recall: float = 0.85,
    min_region_frac: float = 0.20,
    lambda_complexity: float = 0.0,
    diversity_additions: bool = True,
    max_diverse_add: int = 1,
    ortho_add_cos_thresh: float = 0.2,
    family_penalty: float = 0.02,
    pair_filter: Optional[List[Tuple[int, int]]] = None,
    # ---------- IDs ----------
    assign_plane_ids: bool = True,
    plane_id_prefix: str = "pl",
    region_id_prefix: str = "rg",
) -> Dict[str, Any]:
    """
    Salida:
      {
        'by_pair_augmented': { (a,b): {... selection ...} },
        'winning_planes': [ ... planos (medio-espacios) seleccionados en algún par ... ],
        'regions_global': { 'per_plane': [...], 'per_class': {c: [region_id...] } },
        'candidates_global': [... tabla auditable de candidatos y scores ...],
        'meta': {...}
      }
    """
    # ------------------ Preparación ------------------
    X = np.asarray(X, float)
    y = np.asarray(y, int).reshape(-1)
    N, d = X.shape
    baseline = _baseline_props(y)
    labels_all = sorted(baseline.keys())

    # Enumerar TODOS los planos
    all_planes_raw = _collect_all_planes(res)
    if not all_planes_raw:
        return dict(by_pair_augmented={}, winning_planes=[], regions_global=dict(per_plane=[], per_class={c: [] for c in labels_all}),
                    candidates_global=[], meta=dict(msg="no_planes", baseline=baseline))

    # Asignar plane_id únicos (hiperplanos)
    plane_counter = 0

    def _new_plane_id():
        nonlocal plane_counter
        pid = f"{plane_id_prefix}{plane_counter:04d}"
        plane_counter += 1
        return pid

    for r in all_planes_raw:
        r["plane_id"] = _new_plane_id() if assign_plane_ids else None
        # Normalización geométrica
        n = np.asarray(r["n"], float).reshape(-1)
        b0 = float(r["b"])
        u, bn = _normalize_plane(n, b0)
        r["n_norm"] = u
        r["b_norm"] = bn

    # Construir medio-espacios orientados (dos lados por plano)
    oriented_pool = []
    for r in all_planes_raw:
        for side in (+1, -1):
            rr = dict(r)
            rr["side"] = int(side)
            rr["oriented_plane_id"] = (f"{rr['plane_id']}:{'≤' if side >= 0 else '≥'}") if rr.get("plane_id") else None
            oriented_pool.append(rr)

    # Two-stage: submuestreo
    if 0.0 < sketch_frac < 1.0:
        M = int(max(1, round(N * sketch_frac)))
        rng = np.random.default_rng(12345)
        idx = rng.choice(N, size=M, replace=False)
        X_eval = X[idx]
        y_eval = y[idx]
    else:
        X_eval = X
        y_eval = y

    # ------------------ Métricas OVR & Score global por clase ------------------
    # Proyección batched: H = X_eval @ U.T + b
    U = np.stack([r["n_norm"] for r in oriented_pool], axis=1)  # (d, M_oriented)
    bvec = np.array([r["b_norm"] for r in oriented_pool], float)  # (M_oriented,)

    # Calculamos métricas por bloques para ahorrar memoria
    all_metrics_by_class: List[Dict[int, Dict[str, float]]] = [None] * U.shape[1]
    best_class_list: List[Optional[int]] = [None] * U.shape[1]
    best_score_list: List[float] = [0.0] * U.shape[1]
    scores_per_class_list: List[Dict[int, float]] = [None] * U.shape[1]
    region_size_list: List[int] = [0] * U.shape[1]
    region_frac_list: List[float] = [0.0] * U.shape[1]
    entropy_list: List[float] = [0.0] * U.shape[1]

    for lo, hi, H in _batched_project(X_eval, U, bvec, batch=batch_size):
        mask_leq, mask_geq = _halfspace_masks_from_H(H)  # (Neval, k)
        k = H.shape[1]
        for j in range(k):
            idx_pool = lo + j
            side = oriented_pool[idx_pool]["side"]
            mask = mask_leq[:, j] if side >= 0 else mask_geq[:, j]
            y_region = y_eval[mask]
            props = _region_props(y_region)
            ent = _entropy(props)
            region_size_list[idx_pool] = int(mask.sum())
            region_frac_list[idx_pool] = float(mask.sum()) / max(1.0, float(y_eval.size))
            entropy_list[idx_pool] = float(ent)
            mbc = _ovr_metrics_for_side(y_eval, mask, labels_all, baseline)
            all_metrics_by_class[idx_pool] = mbc
            bc, bs, sc = _score_by_class(mbc, w1=w1_balacc, w2=w2_lift_cov, w3=w3_purity, alpha=alpha_cov)
            best_class_list[idx_pool] = bc
            best_score_list[idx_pool] = bs
            scores_per_class_list[idx_pool] = sc

    # Adjuntar métricas al pool
    for i, r in enumerate(oriented_pool):
        r["metrics_by_class"] = all_metrics_by_class[i]
        r["best_class"] = best_class_list[i]
        r["score_global"] = best_score_list[i]
        r["score_by_class"] = scores_per_class_list[i]
        r["region_size_eval"] = region_size_list[i]
        r["region_frac_eval"] = region_frac_list[i]
        r["entropy_eval"] = entropy_list[i]

    # ------------------ Familias (colapso extremadamente estricto) ------------------
    family_id = 0
    families: List[Dict[str, Any]] = []  # cada elemento: dict(rep=oriented_pool[i], members=[indices])
    for i, r in enumerate(oriented_pool):
        assigned = False
        for fam in families:
            rp = fam["rep"]
            if _same_family(r["n"], r["b"], rp["n"], rp["b"],
                            cos_parallel=cos_parallel, tau_mult=tau_mult,
                            coef_eps=coef_eps, jaccard_support=jaccard_support,
                            coef_cos_min=coef_cos_min):
                fam["members"].append(i)
                assigned = True
                break
        if not assigned:
            families.append(dict(rep=r, members=[i], fid=family_id))
            family_id += 1

    # Elegir representante por familia (mejor score_global)
    # Nota: marcamos family_id en todos los miembros; usamos el mejor en reportes
    for fam in families:
        best_idx = max(fam["members"], key=lambda idx: oriented_pool[idx]["score_global"])
        fam["rep"] = oriented_pool[best_idx]
        for idx in fam["members"]:
            oriented_pool[idx]["family_id"] = fam["fid"]

    # ------------------ Filtro global (guardarraíles + Top-K por clase) ------------------
    survivors_idx: List[int] = []
    # Guardarraíles por best_class del medio-espacio
    for i, r in enumerate(oriented_pool):
        bc = r["best_class"]
        if bc is None:
            continue
        mbc = r["metrics_by_class"][bc]
        size_ok = (mbc["region_size"] >= int(min_region_size))
        lift_ok = (mbc["lift"] >= (1.0 + float(min_rel_lift))) or \
                  ((mbc["purity"] - float(baseline.get(int(bc), 0.0))) >= float(min_abs_diff))
        purity_ok = (mbc["purity"] >= float(min_purity))
        if size_ok and lift_ok and purity_ok:
            survivors_idx.append(i)

    # Top-K por clase, no más de 1 por familia en esta etapa
    per_class_buckets: Dict[int, List[int]] = {c: [] for c in labels_all}
    fam_used_per_class: Dict[int, set] = {c: set() for c in labels_all}

    # ordenamos por score_global desc
    survivors_idx.sort(key=lambda i: oriented_pool[i]["score_global"], reverse=True)
    for i in survivors_idx:
        r = oriented_pool[i]
        c = r["best_class"]
        if c is None:
            continue
        if len(per_class_buckets[c]) >= int(refine_topK_per_class):
            continue
        fid = r.get("family_id")
        if (fid is not None) and (fid in fam_used_per_class[c]):
            continue
        per_class_buckets[c].append(i)
        if fid is not None:
            fam_used_per_class[c].add(fid)

    # Union de índices seleccionados
    selected_global_idx = sorted(set(sum(per_class_buckets.values(), [])))
    selected_global = [oriented_pool[i] for i in selected_global_idx]

    # Si usamos sketch, refinamos métricas en full X solo para survivors
    if 0.0 < sketch_frac < 1.0 and len(selected_global) > 0:
        U2 = np.stack([r["n_norm"] for r in selected_global], axis=1)
        b2 = np.array([r["b_norm"] for r in selected_global], float)
        metrics_full = []
        for lo, hi, H in _batched_project(X, U2, b2, batch=batch_size):
            mse_leq, mse_geq = _halfspace_masks_from_H(H)
            for j in range(H.shape[1]):
                idx_pool = lo + j
                r = selected_global[idx_pool]
                side = r["side"]
                mask = mse_leq[:, j] if side >= 0 else mse_geq[:, j]
                mbc = _ovr_metrics_for_side(y, mask, labels_all, baseline)
                r["metrics_by_class"] = mbc
                bc, bs, sc = _score_by_class(mbc, w1=w1_balacc, w2=w2_lift_cov, w3=w3_purity, alpha=alpha_cov)
                r["best_class"] = bc
                r["score_global"] = bs
                r["score_by_class"] = sc
                r["region_size_eval"] = int(mask.sum())
                r["region_frac_eval"] = float(mask.sum()) / max(1.0, float(y.size))
                r["entropy_eval"] = _entropy(_region_props(y[mask]))

        # Reordenar buckets por score recalculado
        per_class_buckets = {c: [] for c in labels_all}
        fam_used_per_class = {c: set() for c in labels_all}
        selected_global.sort(key=lambda r: r["score_global"], reverse=True)
        for idx, r in enumerate(selected_global):
            c = r["best_class"]
            if c is None:
                continue
            if len(per_class_buckets[c]) >= int(refine_topK_per_class):
                continue
            fid = r.get("family_id")
            if (fid is not None) and (fid in fam_used_per_class[c]):
                continue
            per_class_buckets[c].append(idx)
            if fid is not None:
                fam_used_per_class[c].add(fid)

    # ------------------ Construcción de regiones globales ------------------
    region_counter = 0

    def _new_region_id():
        nonlocal region_counter
        rid = f"{region_id_prefix}{region_counter:05d}"
        region_counter += 1
        return rid

    regions_per_plane: List[Dict[str, Any]] = []
    regions_per_class: Dict[int, List[str]] = {int(c): [] for c in labels_all}

    def _emit_region(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rid = _new_region_id()
        side = int(r["side"])
        n = np.asarray(r["n_norm"], float)
        b0 = float(r["b_norm"])
        bc = int(r["best_class"]) if r["best_class"] is not None else None
        if bc is None:
            return None
        reg = dict(
            region_id=rid,
            plane_id=r.get("plane_id"),
            oriented_plane_id=r.get("oriented_plane_id"),
            class_id=int(bc),
            origin_pair=tuple(map(int, r.get("origin_pair", (-1, -1)))),
            side=side,
            inequality=dict(
                general=_ineq_text_general(n, b0, side, feature_names),
                pretty2D=_ineq_text_2d(n, b0, side, dims=dims_for_text, feat_names=feature_names)
            ),
            geometry=dict(type="halfspace", n=n, b=b0, side=side),
            stats=dict(
                metrics_by_class=r["metrics_by_class"],
                score_global=r["score_global"],
                score_by_class=r["score_by_class"],
                region_size=r["region_size_eval"],
                region_frac=r["region_frac_eval"],
                entropy=r["entropy_eval"]
            ),
            labels=dict(
                meets_majority_rule=True,  # ya pasó guardarraíles
                best_class=int(bc)
            )
        )
        return reg

    # Qué lados emitir como "globales"
    def _sides_to_emit_for(r: Dict[str, Any]) -> List[int]:
        if global_sides == "both":
            return [+1, -1]
        elif global_sides == "good":
            # si este oriented es el bueno, emitir solo su lado
            return [int(r["side"])]
        else:  # "opposite"
            return [-int(r["side"])]

    # Emitir regiones (puede crear nuevas copias del lado contrario si aplica)
    for r in selected_global:
        for sd in _sides_to_emit_for(r):
            if sd == int(r["side"]):
                rr = r
            else:
                # crear un duplicado con el lado opuesto para emitir
                rr = dict(r)
                rr["side"] = int(sd)
                # recalcular best_class y métricas para ese lado
                h = X @ rr["n_norm"] + rr["b_norm"]
                mask = (h <= 0.0) if sd >= 0 else (h >= 0.0)
                mbc = _ovr_metrics_for_side(y, mask, labels_all, baseline)
                bc, bs, sc = _score_by_class(mbc, w1=w1_balacc, w2=w2_lift_cov, w3=w3_purity, alpha=alpha_cov)
                rr["metrics_by_class"] = mbc
                rr["best_class"] = bc
                rr["score_global"] = bs
                rr["score_by_class"] = sc
                rr["region_size_eval"] = int(mask.sum())
                rr["region_frac_eval"] = float(mask.sum()) / max(1.0, float(y.size))
                rr["entropy_eval"] = _entropy(_region_props(y[mask]))

            region_payload = _emit_region(rr)
            if region_payload is not None:
                regions_per_plane.append(region_payload)
                regions_per_class[int(region_payload["class_id"])].append(region_payload["region_id"])

    # ------------------ Selección por par (A,B) sobre el pool global ------------------
    # Para la selección, usamos SOLO los candidatos que sobrevivieron (selected_global),
    # pero filtrados por par si así se desea (origer pair) o abiertos con penalización por familia.
    # Aquí usamos el pool completo superviviente (mejor recall), pero con penalización por familia.

    # Índice rápido por par
    pair_keys = sorted(res.keys()) if pair_filter is None else [tuple(p) for p in pair_filter]
    by_pair_augmented: Dict[Tuple[int, int], Dict[str, Any]] = {}

    # Prepara un diccionario de candidatos por par (permitiendo todos, pero guardamos el origin para trazabilidad).
    for pair in pair_keys:
        a, b = int(pair[0]), int(pair[1])
        mask_ab = (y == a) | (y == b)
        Xab = X[mask_ab]
        yab = y[mask_ab]

        # Puedes elegir: restringir a los originados en (a,b) o permitir todos
        # Aquí permitimos todos pero guardamos el origin para trazabilidad.
        planes_for_pair = []
        for r in selected_global:
            rr = dict(r)
            # Añade inequations bonitas respecto a dims_for_text
            n = rr["n_norm"]
            b0 = rr["b_norm"]
            side = int(rr["side"])
            rr["inequality"] = dict(
                general=_ineq_text_general(n, b0, side, feature_names),
                pretty2D=_ineq_text_2d(n, b0, side, dims=dims_for_text, feat_names=feature_names)
            )
            planes_for_pair.append(rr)

        # Selección greedy
        selected, overall = _greedy_and_selection_by_pair(
            planes_for_pair, Xab, yab, a, b,
            max_k=max_k, min_improve=min_improve,
            min_recall=min_recall, min_region_frac=min_region_frac,
            lambda_complexity=lambda_complexity,
            diversity_additions=diversity_additions,
            max_diverse_add=max_diverse_add,
            ortho_add_cos_thresh=ortho_add_cos_thresh,
            family_penalty=family_penalty
        )

        # Otros planos = pool superviviente no elegido (para auditoría)
        used_ids = set([p["oriented_plane_id"] for p in selected])
        others = [p for p in planes_for_pair if p["oriented_plane_id"] not in used_ids]

        # Regla compuesta (AND) en forma de desigualdades
        ineqs = []
        for p in selected:
            ineqs.append(dict(
                oriented_plane_id=p["oriented_plane_id"],
                side="≤" if int(p["side"]) >= 0 else "≥",
                n=p["n_norm"], b=p["b_norm"],
                text=p["inequality"]["general"]
            ))

        by_pair_augmented[(a, b)] = dict(
            winning_planes=selected,
            other_planes=others,
            region_rule=dict(logic="AND", inequalities=ineqs, dims=dims_for_text),
            metrics_overall=overall,
            meta=dict(num_candidates=len(planes_for_pair))
        )

    # ------------------ Candidatos globales (tabla auditable) ------------------
    candidates_global = []
    for r in selected_global:
        candidates_global.append(dict(
            plane_id=r.get("plane_id"),
            oriented_plane_id=r.get("oriented_plane_id"),
            family_id=r.get("family_id"),
            best_class=r.get("best_class"),
            score_global=float(r.get("score_global", 0.0)),
            region_size=int(r.get("region_size_eval", 0)),
            region_frac=float(r.get("region_frac_eval", 0.0)),
            entropy=float(r.get("entropy_eval", 0.0)),
            origin_pair=tuple(map(int, r.get("origin_pair", (-1, -1))))
        ))

    # ------------------ Resumen meta ------------------
    # winning_planes: los medio-espacios seleccionados en algún par
    winning_planes_unified = []
    for k, payload in by_pair_augmented.items():
        for p in payload.get("winning_planes", []):
            winning_planes_unified.append(dict(p))  # copia

    out = dict(
        by_pair_augmented=by_pair_augmented,
        winning_planes=winning_planes_unified,
        regions_global=dict(
            per_plane=regions_per_plane,
            per_class=regions_per_class
        ),
        candidates_global=candidates_global,
        meta=dict(
            baseline=baseline,
            labels_all=labels_all,
            N=int(N), d=int(d),
            total_pairs=len(pair_keys),
            total_input_planes=len(all_planes_raw),
            total_oriented=len(oriented_pool),
            total_families=len(families),
            total_candidates_global=len(selected_global),
            total_regions=len(regions_per_plane),
            params=dict(
                global_sides=global_sides,
                sketch_frac=sketch_frac,
                refine_topK_per_class=refine_topK_per_class,
                batch_size=batch_size,
                family=dict(cos_parallel=cos_parallel, tau_mult=tau_mult,
                            coef_eps=coef_eps, jaccard_support=jaccard_support, coef_cos_min=coef_cos_min),
                guards=dict(min_region_size=min_region_size, min_abs_diff=min_abs_diff,
                            min_rel_lift=min_rel_lift, min_purity=min_purity),
                scoring=dict(w1_balacc=w1_balacc, w2_lift_cov=w2_lift_cov, w3_purity=w3_purity, alpha_cov=alpha_cov),
                selection_by_pair=dict(
                    max_k=max_k, min_improve=min_improve, min_recall=min_recall, min_region_frac=min_region_frac,
                    lambda_complexity=lambda_complexity, diversity_additions=diversity_additions,
                    max_diverse_add=max_diverse_add, ortho_add_cos_thresh=ortho_add_cos_thresh,
                    family_penalty=family_penalty
                ),
                ids=dict(plane_id_prefix=plane_id_prefix, region_id_prefix=region_id_prefix)
            )
        )
    )
    return out

