# -*- coding: utf-8 -*-
"""
find_low_dim_spaces_fast.py

Exploración masiva y eficiente de regiones tipo regla geométrica:
- Evaluación vectorizada de planos por (clase, dims).
- AND/OR de máscaras con bitsets (np.packbits) + popcount LUT (uint8).
- Uniones (OR) entre regiones con bajo empalme (IoU) para subir recall sin matar precisión.
- Paralelización opcional (joblib, backend=threading).
- Dedupl. global por máscara (elige representante más simple / mejor).
- Piso por clase (floor) por dimensión.
- Relaciones (generaliza/especializa) y frontera de Pareto (opcional).

Salida compatible: {k_dim: [registros]} donde cada registro incluye:
    region_id, target_class, dims, plane_ids, sources, rule_text, rule_pieces,
    metrics {size, precision, recall, f1, baseline, lift_precision},
    metrics_per_class, region_summary, projection_ref, complexity {num_dims, num_planes},
    is_floor, generalizes, specializes, is_pareto, family_id, parent_id, deltas_to_parent,
    seed_type ("pair", "global_seed", "global_heuristic", "union"), mask_signature
    (+ "_mask" si include_masks=True)

Autor: JC + ChatGPT (GPT-5 Thinking)
Fecha: 2025-11-07
"""

from typing import Dict, Tuple, Any, List, Optional, Union, Iterable
import logging
from time import perf_counter
import numpy as np, itertools, hashlib, time, math, random

from ._logging_utils import verbosity_to_level

# ========================= Aceleradores bitset / popcount =========================

# Tabla de popcount para uint8 (0..255)
_POP_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def _pack_bits_cols(B: np.ndarray) -> np.ndarray:
    """
    Empaqueta matriz booleana (N, K) -> (NB, K) uint8, con packbits(axis=0).
    NB = ceil(N/8). Columna j es la máscara del plano j empacada por bloques de 8.
    """
    B = np.asarray(B, dtype=bool)
    return np.packbits(B, axis=0)  # NB x K (uint8)

def _unpack_bits_col_to_bool(col_bits: np.ndarray, N: int) -> np.ndarray:
    """
    Desempaqueta una columna (NB,) uint8 -> (N,) bool.
    """
    raw = np.unpackbits(col_bits, axis=0)  # (NB*8,)
    if raw.size >= N:
        return raw[:N].astype(bool)
    out = np.zeros(N, dtype=bool)
    out[:raw.size] = raw.astype(bool)
    return out

def _bitwise_and_reduce_cols(packed: np.ndarray, idxs: Tuple[int, ...]) -> np.ndarray:
    """
    AND sobre columnas seleccionadas: (NB, K) &...& -> (NB,)
    """
    res = packed[:, idxs[0]].copy()
    for j in idxs[1:]:
        np.bitwise_and(res, packed[:, j], out=res)
    return res

def _bitwise_or_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = a.copy()
    np.bitwise_or(out, b, out=out)
    return out

def _bitwise_and_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = a.copy()
    np.bitwise_and(out, b, out=out)
    return out

def _popcount_bits(col_bits: np.ndarray) -> int:
    """
    Cuenta bits de (NB,) uint8 usando LUT.
    """
    return int(_POP_LUT[col_bits].sum())

def _iou_packed(a_bits: np.ndarray, b_bits: np.ndarray) -> float:
    inter = _popcount_bits(_bitwise_and_cols(a_bits, b_bits))
    union = _popcount_bits(_bitwise_or_cols(a_bits, b_bits))
    return float(inter / max(1, union))

def _is_subset_packed(a_bits: np.ndarray, b_bits: np.ndarray) -> bool:
    """
    ¿a ⊇ b? (b está contenido en a) => (b & ~a) == 0
    """
    inv_a = np.bitwise_not(a_bits)
    return not np.bitwise_and(b_bits, inv_a).any()

# ========================= Helpers numéricos / texto =========================

def _class_baseline(y: np.ndarray) -> Dict[int, float]:
    y = np.asarray(y, int).ravel()
    labels, counts = np.unique(y, return_counts=True)
    N = float(y.size)
    return {int(c): float(n)/max(N, 1.0) for c, n in zip(labels, counts)}

def _feat_label(i: int, feature_names: Optional[List[str]]):
    if feature_names and 0 <= i < len(feature_names):
        return feature_names[i]
    return f"x{i}"

def _fmt_affine(m: float, c: float, xname: str) -> str:
    if c >= 0:
        return f"{m:.2f}·{xname} + {c:.2f}"
    else:
        return f"{m:.2f}·{xname} {c:.2f}"

def _ineq_text_for_dims(n: np.ndarray, b: float, side: int,
                        dims: Tuple[int, ...],
                        feat_names: Optional[List[str]] = None,
                        tol: float = 1e-12) -> str:
    s = "≤" if side >= 0 else "≥"
    d = len(dims)
    n = np.asarray(n, float).reshape(-1)

    if d == 1:
        i = dims[0]
        coef = float(n[i])
        name = _feat_label(i, feat_names)
        if abs(coef) < tol:
            return f"{float(b):.2f} {s} 0"
        thr = -float(b)/coef
        leq = (side >= 0 and coef > 0) or (side < 0 and coef < 0)
        sign = "≤" if leq else "≥"
        return f"{name} {sign} {thr:.2f}"

    if d == 2:
        i, j = dims
        ni, nj = float(n[i]), float(n[j])
        xi, yj = _feat_label(i, feat_names), _feat_label(j, feat_names)
        tolz = tol
        if abs(nj) > tolz:
            m = -ni/nj
            c = -b/nj
            leq = (side >= 0 and nj > 0) or (side < 0 and nj < 0)
            sign = "≤" if leq else "≥"
            return f"{yj} {sign} {_fmt_affine(m, c, xi)}"
        elif abs(ni) > tolz:
            xcut = -b/ni
            leq = (side >= 0 and ni > 0) or (side < 0 and ni < 0)
            sign = "≤" if leq else "≥"
            return f"{xi} {sign} {xcut:.2f}"
        else:
            return f"{float(b):.2f} {s} 0"

    terms = []
    for i in dims:
        coef = float(n[i])
        if abs(coef) < tol:
            continue
        name = _feat_label(i, feat_names)
        terms.append(f"{coef:.2f}·{name}")
    left = " + ".join(terms) if terms else "0"
    if abs(b) > tol:
        left = f"{left} {float(b):.2f}"
    return f"{left} {s} 0"

def _metrics_region_multiclass_maskbits(mask_bits: np.ndarray,
                                        packed_Y_by_class: Dict[int, np.ndarray],
                                        labels: List[int],
                                        baseline: Dict[int, float],
                                        N: int) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    sel = _popcount_bits(mask_bits)
    summary = dict(size=int(sel), frac=float(sel/max(1, N)))
    per_class: Dict[int, Dict[str, float]] = {}
    for c in labels:
        y_bits = packed_Y_by_class[int(c)]
        pos = _popcount_bits(_bitwise_and_cols(mask_bits, y_bits))
        total_pos = int(_popcount_bits(y_bits))
        prec = float(pos/sel) if sel>0 else 0.0
        rec  = float(pos/max(1,total_pos))
        f1 = (2*prec*rec)/(prec+rec+1e-12)
        base = float(baseline.get(int(c), 0.0))
        lift = (prec/base) if base>0 else (math.inf if prec>0 else 0.0)
        per_class[int(c)] = dict(
            pos=int(pos), total_pos=int(total_pos),
            precision=prec, recall=rec, f1=f1,
            baseline=base, lift_precision=lift
        )
    return per_class, summary

def _project_plane_to_dims(n: np.ndarray, b: float, side: int,
                           dims: Tuple[int, ...],
                           mu_fix: np.ndarray):
    n = np.asarray(n, float).reshape(-1)
    mu = np.asarray(mu_fix, float).reshape(-1)
    D = n.size
    other = [j for j in range(D) if j not in dims]
    b_eff = float(b) + float(n[other] @ mu[other]) if other else float(b)
    n_eff = np.zeros_like(n)
    n_eff[list(dims)] = n[list(dims)]
    return n_eff, b_eff, int(side)

def _canonize_rule_text_from_kept(kept_planes: List[Tuple[np.ndarray,float,int]],
                                  dims: Tuple[int,...],
                                  feat_names: Optional[List[str]]=None,
                                  tol: float = 1e-12) -> Tuple[str, List[str]]:
    Dk = len(dims)
    if Dk == 1:
        i = dims[0]
        name = _feat_label(i, feat_names)
        low, up = -np.inf, np.inf
        saw = False
        for n_eff, b_eff, sd_eff in kept_planes:
            coef = float(n_eff[i])
            if abs(coef) < tol:
                continue
            thr = -float(b_eff)/coef
            leq = (sd_eff >= 0 and coef > 0) or (sd_eff < 0 and coef < 0)
            if leq:
                up = min(up, thr)
            else:
                low = max(low, thr)
            saw = True
        pieces = []
        if not saw:
            return "", pieces
        if np.isfinite(low) and np.isfinite(up):
            text = f"{low:.2f} ≤ {name} ≤ {up:.2f}"
            pieces = [text]
            return text, pieces
        if np.isfinite(up):
            text = f"{name} ≤ {up:.2f}"
            pieces = [text]
        else:
            text = f"{name} ≥ {low:.2f}"
            pieces = [text]
        return text, pieces

    raw = []
    for n_eff, b_eff, sd_eff in kept_planes:
        raw.append(_ineq_text_for_dims(n_eff, b_eff, sd_eff, dims, feat_names))
    uniq = sorted(set(raw))
    text = "  AND  ".join(uniq)
    return text, uniq

def _onehot_y(y: np.ndarray, labels: List[int]) -> np.ndarray:
    y = np.asarray(y, int).ravel()
    idx = {c:i for i,c in enumerate(labels)}
    Y = np.zeros((y.size, len(labels)), dtype=bool)
    for i,yy in enumerate(y):
        j = idx.get(int(yy), None)
        if j is not None:
            Y[i, j] = True
    return Y

# ========= Recolector de planos por clase =========

def _gather_candidate_planes_for_class(sel: Dict[str,Any], cls: int) -> List[Dict[str,Any]]:
    out = []
    by_pair = sel.get("by_pair_augmented", {}) or {}
    for (a,b), payload in by_pair.items():
        a,b = int(a), int(b)
        win = payload.get("winning_planes", []) or []
        for p in win:
            n = np.asarray(p["n"], float)
            b0 = float(p["b"])
            side_ab = int(p.get("side", +1))
            if cls == a:
                side_for = side_ab
            elif cls == b:
                side_for = -side_ab
            else:
                continue
            out.append(dict(
                plane_id=p.get("plane_id", None),
                n=n, b=b0, side_for_cls=int(side_for),
                origin_pair=(a,b),
                source="pair"
            ))
    regs = sel.get("regions_global", {}).get("per_plane", []) or []
    for R in regs:
        if int(R.get("class_id", -999)) != int(cls):
            continue
        geom = R.get("geometry", {}) or {}
        n = np.asarray(geom.get("n"), float)
        b0 = float(geom.get("b"))
        side = int(geom.get("side"))
        out.append(dict(
            plane_id=R.get("plane_id", None),
            n=n, b=b0, side_for_cls=int(side),
            origin_pair=tuple(R.get("origin_pair", (None,None))),
            source="global"
        ))
    return out

# ========================= Núcleo: combos AND masivos con bitsets =========================

def _build_plane_bank_for_dims(
    X: np.ndarray, dims: Tuple[int,...],
    planes_cls: List[Dict[str,Any]],
    mu: np.ndarray,
    min_norm_in_dims: float
):
    if len(planes_cls) == 0:
        return None, None, None

    Xsub = X[:, dims]
    mu_d = mu[list(dims)]

    ns_sub, bs_eff, sides, plane_ids, plane_refs, ns_full, bs_full = [], [], [], [], [], [], []
    for p in planes_cls:
        n_full = np.asarray(p["n"], float).reshape(-1)
        b0     = float(p["b"])
        sd     = int(p["side_for_cls"])
        n_sub  = n_full[list(dims)]
        if np.linalg.norm(n_sub) < float(min_norm_in_dims):
            continue
        b_eff = b0 + float(n_full @ mu) - float(n_sub @ mu_d)
        ns_sub.append(n_sub)
        bs_eff.append(b_eff)
        sides.append(sd)
        plane_ids.append(p.get("plane_id"))
        plane_refs.append(dict(plane_id=p.get("plane_id"),
                               origin_pair=tuple(p.get("origin_pair",(None,None))),
                               source=p.get("source","pair")))
        ns_full.append(n_full); bs_full.append(b0)

    if not ns_sub:
        return None, None, None

    Nsub = np.stack(ns_sub, axis=1)               # (d, K)
    H = Xsub @ Nsub + np.asarray(bs_eff)[None,:]  # (N, K)
    sides_arr = np.asarray(sides, int)
    M = np.empty_like(H, dtype=bool)              # (N, K)
    pos = sides_arr >= 0
    if pos.any():
        M[:, pos]  = H[:, pos] <= 0.0
    if (~pos).any():
        M[:, ~pos] = H[:, ~pos] >= 0.0

    packed_M = _pack_bits_cols(M)                 # (NB, K)

    plane_meta = dict(
        plane_ids = tuple(plane_ids),
        plane_refs= tuple(plane_refs),
        sides     = tuple(int(s) for s in sides),
        ns_full   = tuple(np.asarray(n) for n in ns_full),
        bs_full   = tuple(float(b) for b in bs_full)
    )
    return plane_meta, M, packed_M

def _eval_combos_fast_on_dims_bitset(
    X: np.ndarray, y: np.ndarray,
    labels: List[int],
    baseline: Dict[int,float],
    planes_cls: List[Dict[str,Any]],
    cls: int, dims: Tuple[int,...], mu: np.ndarray,
    *,
    min_support: int,
    min_rel_gain_f1: float, min_abs_gain_f1: float,
    min_lift_prec: float,  min_abs_gain_prec: float,
    min_pos_in_region: int,
    max_planes_in_rule: int,
    sample_limit_per_r: int,
    per_class_floor_topk: int,
    pieces_fn,                        # callback: (ns, bs, sides_sel, dims, mu)->(rule_text, pieces_txt)
    make_record_fn,                   # callback -> dict registro
    packed_Y_by_class: Dict[int, np.ndarray],
    include_masks_internal: bool
):
    """
    Usa bitsets empaquetados para evaluar combos AND de columnas de M.
    Devuelve (aceptadas, pool_para_floor, all_records_for_floor)
    """
    meta, M_bool, M_packed = _build_plane_bank_for_dims(X, dims, planes_cls, mu, min_norm_in_dims=1e-12)
    if M_packed is None:
        return [], [], []

    N = X.shape[0]
    K = M_packed.shape[1]
    accepted: List[Dict[str,Any]] = []
    pool: List[Tuple[float,float,int,Dict[str,Any]]] = []
    all_for_floor: List[Dict[str,Any]] = []

    base = float(baseline[int(cls)])
    cls_idx = labels.index(int(cls))
    total_pos_c = int(_popcount_bits(packed_Y_by_class[int(cls)]))

    rng_local = random.Random(0xC0FFEE ^ (hash((cls, dims)) & 0xFFFF))
    all_idxs = list(range(K))
    rng_local.shuffle(all_idxs)

    def _maybe_push_pool(rec, f1, lift, size):
        pool.append((float(f1), float(lift), int(size), rec))
        if len(pool) > int(per_class_floor_topk * 3 + 32):
            pool.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
            del pool[int(per_class_floor_topk * 3 + 32):]

    for r in range(1, int(max_planes_in_rule)+1):
        combos = itertools.combinations(all_idxs, r)
        combos = itertools.islice(combos, int(sample_limit_per_r))

        for idxs in combos:
            mask_bits = M_packed[:, idxs[0]].copy()
            for j in idxs[1:]:
                np.bitwise_and(mask_bits, M_packed[:, j], out=mask_bits)
                if not mask_bits.any():
                    break
            if not mask_bits.any():
                continue

            sel = _popcount_bits(mask_bits)
            if sel < int(min_support):
                continue

            # conteos por clase (bitsets)
            pos_c = _popcount_bits(_bitwise_and_cols(mask_bits, packed_Y_by_class[int(cls)]))
            prec = float(pos_c/sel) if sel>0 else 0.0
            rec  = float(pos_c/max(1,total_pos_c))
            f1   = (2*prec*rec)/(prec+rec+1e-12)
            lift = (prec/base) if base>0 else (math.inf if prec>0 else 0.0)

            ok_f1   = (f1 >= max(base*(1.0+float(min_rel_gain_f1)), base + float(min_abs_gain_f1)))
            ok_prec = (prec >= max(base*float(min_lift_prec),   base + float(min_abs_gain_prec)))
            if min_pos_in_region>0 and pos_c < int(min_pos_in_region):
                pass_ok = False
            else:
                pass_ok = (ok_f1 or ok_prec)

            plane_ids_sel  = tuple(meta["plane_ids"][j]  for j in idxs)
            plane_refs_sel = tuple(meta["plane_refs"][j] for j in idxs)
            ns_sel   = [meta["ns_full"][j]  for j in idxs]
            bs_sel   = [meta["bs_full"][j]  for j in idxs]
            sides_sel= [meta["sides"][j]    for j in idxs]

            rule_text, pieces_txt = pieces_fn(ns_sel, bs_sel, sides_sel, dims, mu)

            # métricas multiclass completas para ranking/compat.
            per_class, summary = _metrics_region_multiclass_maskbits(mask_bits, packed_Y_by_class, labels, baseline, N)
            M_target = per_class[int(cls)]
            recd = make_record_fn(
                cls=int(cls), dims=dims,
                plane_ids=plane_ids_sel, plane_refs=plane_refs_sel,
                rule_text=rule_text, pieces_txt=pieces_txt,
                M_target=M_target, per_class=per_class, summary=summary,
                mask_bits=mask_bits  # <-- sólo una vez
            )
            all_for_floor.append(recd)
            if pass_ok:
                accepted.append(recd)
            else:
                _maybe_push_pool(recd, f1, lift, sel)

    if pool:
        pool.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        top_pool = [rec for (_f1,_lift,_sz,rec) in pool[:int(per_class_floor_topk)]]
    else:
        top_pool = []
    return accepted, top_pool, all_for_floor

# ========================= Función principal =========================

def find_low_dim_spaces(
    X: np.ndarray,
    y: np.ndarray,
    sel: Dict[str,Any],
    *,
    feature_names: Optional[List[str]] = None,
    max_planes_in_rule: int = 3,
    max_planes_per_pair: int = 4,
    max_rules_per_dim: int = 50,
    min_support: int = 8,
    # ---- criterios adaptativos (rel + abs) ----
    min_rel_gain_f1: float = 0.01,
    min_abs_gain_f1: float = 0.05,
    min_lift_prec: float = 1.05,
    min_abs_gain_prec: float = 0.05,
    # ---- robustez geométrica ----
    min_norm_in_dims: float = 1e-8,
    drop_vacuous_in_legend: bool = True,
    # ---- cobertura por clase ----
    per_class_floor_topk: int = 3,
    consider_dims_up_to: Optional[int] = None,
    rng_seed: int = 0,
    # ---- referencia de proyección ----
    projection_ref: Any = "class_mean",  # "class_mean" | "global_mean" | "zero" | np.ndarray
    # ---- logging ----
    enable_logs: bool = True,
    max_log_records: int = 10000,
    return_logs: bool = False,
    verbosity: int = 0,
    # ---- extras ----
    include_masks: bool = False,        # si True, devuelve _mask por regla
    compute_relations: bool = True,     # generaliza/especializa + Pareto (usa bitsets internos)
    # ---- mínimo de positivos por región ----
    min_pos_in_region: int = 0,         # 0 = desactivado; si >0 exige al menos tantos positivos
    # ---- Uniones (OR) de bajo empalme ----
    enable_unions: bool = True,
    union_same_dims_only: bool = True,
    union_max_iou: float = 0.25,               # IoU máximo (poco empalme)
    union_topk_per_bucket: int = 10,           # top-K por (cls,dims) para formar pares
    union_min_gain_f1_vs_best: float = 0.01,   # mejora mínima vs mejor hijo
    union_min_gain_rec_vs_best: float = 0.03,  # alternativa por recall si lift_precision suficiente
    union_min_lift_precision: float = 1.00,    # lift mínimo para la condición de recall
    union_min_abs_prec: float = 0.00,          # alternativa por ganancia absoluta de precisión
    union_min_support: Optional[int] = None,   # si None => usa min_support
    union_max_pairs_per_bucket: int = 2000,    # tope de pares OR evaluados
    # ---- muestreo de combinaciones ----
    sample_limit_per_r: int = 5000,            # EXPLORACIÓN masiva por tamaño de regla
    # ---- límite candidato por clase ----
    max_planes_per_class_cap: int = 120,       # recorte aleatorio si supera
) -> Union[Dict[int, List[Dict[str,Any]]], Tuple[Dict[int, List[Dict[str,Any]]], List[Dict[str,Any]]]]:

    t0 = time.time()
    logs: List[Dict[str,Any]] = []
    def _log(event: str, **kw):
        if enable_logs and len(logs) < int(max_log_records):
            logs.append({"t": round(time.time()-t0, 6), "event": event, **kw})

    # Necesitamos máscaras internas si haremos relaciones o uniones
    _include_masks_internal = bool(include_masks or compute_relations or enable_unions)

    rng = np.random.RandomState(int(rng_seed))
    random.seed(int(rng_seed))
    X = np.asarray(X, float); y = np.asarray(y, int).ravel()
    N, D = X.shape
    if consider_dims_up_to is None: consider_dims_up_to = D
    consider_dims_up_to = int(min(max(1, consider_dims_up_to), D))

    logger = logging.getLogger(__name__)
    level = verbosity_to_level(verbosity)
    logger.setLevel(level)
    start_total = perf_counter()
    logger.log(
        level,
        "find_low_dim_spaces inicio | N=%d D=%d max_planes_in_rule=%d min_support=%d", 
        N,
        D,
        max_planes_in_rule,
        min_support,
    )

    # Medias para proyección
    mu_by_class = {int(c): X[y==int(c)].mean(axis=0) if (y==int(c)).any() else X.mean(axis=0)
                   for c in np.unique(y)}
    mu_global = X.mean(axis=0)
    mu_zero = np.zeros(D)

    def _mu_lookup(cls: int) -> np.ndarray:
        if isinstance(projection_ref, np.ndarray):
            return np.asarray(projection_ref, float).reshape(-1)
        if projection_ref == "global_mean": return mu_global
        if projection_ref == "zero": return mu_zero
        return mu_by_class[int(cls)]

    baseline = _class_baseline(y)
    labels = sorted(baseline.keys())

    # One-hot y bitsets por clase
    Y_onehot = _onehot_y(y, labels)
    packed_Y_by_class: Dict[int, np.ndarray] = {}
    for j, c in enumerate(labels):
        packed_Y_by_class[int(c)] = _pack_bits_cols(Y_onehot[:, [j]])[:, 0]  # (NB,)

    _log("init", N=N, D=D, labels=list(map(int, labels)), consider_dims_up_to=int(consider_dims_up_to),
         max_planes_in_rule=int(max_planes_in_rule), sample_limit_per_r=int(sample_limit_per_r))

    # Acumuladores
    valuable: Dict[int, List[Dict[str,Any]]] = {k: [] for k in range(1, consider_dims_up_to+1)}
    all_candidates_by_dim_cls: Dict[Tuple[int,int], List[Dict[str,Any]]] = {}

    # Helpers texto y registros
    def _pieces_for_rule(ns, bs, sides, dims, mu, check_vacuous=True):
        kept = []
        for (n, b0, sd) in zip(ns, bs, sides):
            n_eff, b_eff, sd_eff = _project_plane_to_dims(np.asarray(n, float), float(b0), int(sd), dims, mu)
            if np.linalg.norm(n_eff[list(dims)]) < float(min_norm_in_dims):
                continue
            if check_vacuous and drop_vacuous_in_legend:
                h = X @ n_eff + float(b_eff)
                mask_plane = (h <= 0.0) if sd_eff >= 0 else (h >= 0.0)
                if mask_plane.all():
                    continue
            kept.append((n_eff, b_eff, sd_eff))
        rule_text, pieces_txt = _canonize_rule_text_from_kept(kept, dims, feature_names)
        return rule_text, pieces_txt, kept

    def _region_id(cls: int, dims: Tuple[int, ...], plane_ids: Tuple[Any, ...], rule_text: str) -> str:
        payload = repr((int(cls), tuple(map(int, dims)), tuple(map(str, plane_ids)), str(rule_text))).encode("utf-8")
        h = hashlib.md5(payload).hexdigest()[:10]
        return f"rg_{len(dims)}d_c{int(cls)}_{h}"

    def _make_record(cls:int, dims:Tuple[int,...], plane_ids:Tuple[Any,...], plane_refs:Tuple[Dict[str,Any],...],
                     rule_text:str, pieces_txt:List[str],
                     M_target:Dict[str,float], per_class:Dict[int,Dict[str,float]], summary:Dict[str,float],
                     mask_bits:np.ndarray,
                     projection_ref=projection_ref, planes_used_payload:Optional[List[Dict[str,Any]]]=None,
                     seed_type:str="pair") -> Dict[str,Any]:
        rid = _region_id(int(cls), tuple(int(d) for d in dims), plane_ids, rule_text)
        rec = dict(
            region_id=rid,
            target_class=int(cls),
            dims=tuple(int(d) for d in dims),
            plane_ids=tuple(plane_ids),
            sources=plane_refs,
            rule_text=rule_text,
            rule_pieces=pieces_txt,
            metrics=dict(
                size=int(summary["size"]),
                precision=float(M_target["precision"]),
                recall=float(M_target["recall"]),
                f1=float(M_target["f1"]),
                baseline=float(M_target["baseline"]),
                lift_precision=float(M_target["lift_precision"]),
            ),
            metrics_per_class=per_class,
            region_summary=summary,
            projection_ref=str(projection_ref),
            complexity=dict(
                num_dims=len(dims),
                num_planes=len(plane_ids),
            ),
            is_floor=False,
            generalizes=[],
            specializes=[],
            is_pareto=False,
            family_id=None,
            parent_id=None,
            deltas_to_parent=None,
            planes_used=planes_used_payload if planes_used_payload is not None else [],
            seed_type=seed_type
        )
        sig = hashlib.md5(mask_bits.tobytes()).hexdigest()[:12]
        rec["mask_signature"] = sig
        rec["_mask_bits"] = mask_bits.copy()  # interno para uniones/dedup/relaciones
        if _include_masks_internal:
            Nloc = X.shape[0]
            rec["_mask"] = _unpack_bits_col_to_bool(mask_bits, Nloc)
        return rec

    def _accept_record(rec:Dict[str,Any], dim_k:int, cls:int):
        valuable.setdefault(dim_k, []).append(rec)
        _log("accept_rule",
             cls=int(cls), dims=tuple(int(d) for d in rec["dims"]),
             region_id=rec["region_id"], planes_in_rule=len(rec["plane_ids"]),
             size=int(rec["metrics"]["size"]), precision=float(rec["metrics"]["precision"]),
             recall=float(rec["metrics"]["recall"]), f1=float(rec["metrics"]["f1"]))

    # ---------- bucle principal: candidatos estándar vectorizados/bitset ----------
    for cls in labels:
        base_cls = float(baseline[int(cls)])
        _log("class_start", cls=int(cls), baseline=base_cls)

        cand_all = _gather_candidate_planes_for_class(sel, int(cls))
        _log("candidates_collected", cls=int(cls), total=len(cand_all))

        # limitar por par
        group: Dict[Tuple[int,int], List[Dict[str,Any]]] = {}
        for p in cand_all:
            group.setdefault(tuple(p["origin_pair"]), []).append(p)
        cand_limited: List[Dict[str,Any]] = []
        for k, lst in group.items():
            lst_sorted = lst
            cand_limited += lst_sorted[:int(max_planes_per_pair)]

        # recorte global por clase para controlar memoria/tiempo
        if len(cand_limited) > int(max_planes_per_class_cap):
            cand_limited = random.sample(cand_limited, int(max_planes_per_class_cap))
        _log("candidates_limited", cls=int(cls), after_limit=len(cand_limited))

        def _work_on_dims(dims:Tuple[int,...]):
            mu = _mu_lookup(int(cls))
            def _pieces_fn(ns, bs, sides, dims_, mu_):
                return _pieces_for_rule(ns, bs, sides, dims_, mu_, check_vacuous=True)[:2]
            def _mkrec(**kw):
                return _make_record(**kw, projection_ref=projection_ref)
            accepted, pool, all_for_floor = _eval_combos_fast_on_dims_bitset(
                X, y, labels, baseline,
                cand_limited, cls, dims, mu,
                min_support=min_support,
                min_rel_gain_f1=min_rel_gain_f1, min_abs_gain_f1=min_abs_gain_f1,
                min_lift_prec=min_lift_prec,   min_abs_gain_prec=min_abs_gain_prec,
                min_pos_in_region=min_pos_in_region,
                max_planes_in_rule=max_planes_in_rule,
                sample_limit_per_r=sample_limit_per_r,
                per_class_floor_topk=per_class_floor_topk,
                pieces_fn=_pieces_fn,
                make_record_fn=lambda cls,dims,plane_ids,plane_refs,rule_text,pieces_txt,M_target,per_class,summary,mask_bits: _mkrec(
                    cls=cls, dims=dims, plane_ids=plane_ids, plane_refs=plane_refs,
                    rule_text=rule_text, pieces_txt=pieces_txt,
                    M_target=M_target, per_class=per_class, summary=summary,
                    mask_bits=mask_bits,
                    planes_used_payload=None, seed_type="pair"
                ),
                packed_Y_by_class=packed_Y_by_class,
                include_masks_internal=_include_masks_internal
            )
            return accepted, pool, all_for_floor

        dims_jobs = []
        for dim_k in range(1, consider_dims_up_to+1):
            dims_jobs.extend([(dim_k, dims) for dims in itertools.combinations(range(D), dim_k)])

        results = []
        try:
            from joblib import Parallel, delayed  # opcional
            results = Parallel(n_jobs=-1, backend="threading", require="sharedmem")(
                delayed(_work_on_dims)(dims) for (_k, dims) in dims_jobs
            )
        except Exception:
            results = [ _work_on_dims(dims) for (_k, dims) in dims_jobs ]

        # integrar resultados
        for (accepted, pool, all_for_floor), (_k, dims) in zip(results, dims_jobs):
            dim_k = len(dims)
            for rec in accepted:
                all_candidates_by_dim_cls.setdefault((dim_k, int(cls)), []).append(rec)
            for rec in pool:
                all_candidates_by_dim_cls.setdefault((dim_k, int(cls)), []).append(rec)
            for rec in accepted:
                _accept_record(rec, dim_k, int(cls))
            for rec in all_for_floor:
                all_candidates_by_dim_cls.setdefault((dim_k, int(cls)), []).append(rec)

    # ------- cobertura por clase (floor) ------
    for dim_k in range(1, consider_dims_up_to+1):
        present_classes = {r["target_class"] for r in valuable.get(dim_k, [])}
        for cls in labels:
            if int(cls) in present_classes:
                continue
            pool = all_candidates_by_dim_cls.get((dim_k, int(cls)), [])
            if not pool:
                continue
            pool_sorted = sorted(pool, key=lambda r: (r["metrics"]["f1"], r["metrics"]["lift_precision"], r["metrics"]["size"]), reverse=True)
            take = pool_sorted[:int(per_class_floor_topk)]
            for r in take:
                r["is_floor"] = True
            if take:
                valuable.setdefault(dim_k, []).extend(take)

    # ------- NUEVO: uniones OR de bajo empalme -------
    if enable_unions:
        _added_unions = 0
        UN_MIN_SUP = int(union_min_support or min_support)
        for dim_k, L in list(valuable.items()):
            if not L:
                continue
            # buckets por (cls,dims)
            buckets: Dict[Tuple[int, Tuple[int,...]], List[Dict[str,Any]]] = {}
            for r in L:
                key = (int(r["target_class"]), tuple(int(d) for d in r["dims"]))
                buckets.setdefault(key, []).append(r)

            for (cls, dims) in buckets.keys():
                bucket = buckets[(cls, dims)]
                # top-K para pares
                bucket_sorted = sorted(
                    bucket,
                    key=lambda r: (r["metrics"]["f1"], r["metrics"]["lift_precision"], r["metrics"]["size"]),
                    reverse=True
                )[:int(max(2, union_topk_per_bucket))]

                idx_pairs = list(itertools.combinations(range(len(bucket_sorted)), 2))
                if len(idx_pairs) > int(union_max_pairs_per_bucket):
                    idx_pairs = idx_pairs[:int(union_max_pairs_per_bucket)]

                base = float(baseline[int(cls)])

                for i, j in idx_pairs:
                    A, B = bucket_sorted[i], bucket_sorted[j]
                    if union_same_dims_only and tuple(A["dims"]) != tuple(B["dims"]):
                        continue
                    dims_use = tuple(A["dims"]) if union_same_dims_only else tuple(sorted(set(A["dims"]) | set(B["dims"])) )

                    ma_bits = A.get("_mask_bits"); mb_bits = B.get("_mask_bits")
                    if ma_bits is None or mb_bits is None:
                        continue

                    iou = _iou_packed(ma_bits, mb_bits)
                    if iou > float(union_max_iou):
                        continue

                    mu_bits = _bitwise_or_cols(ma_bits, mb_bits)
                    if not mu_bits.any():
                        continue

                    per_cls, summary = _metrics_region_multiclass_maskbits(mu_bits, packed_Y_by_class, labels, baseline, N)
                    Mu = per_cls[int(cls)]
                    size_ok = summary["size"] >= UN_MIN_SUP
                    pos_ok  = True if int(min_pos_in_region) <= 0 else (int(Mu["pos"]) >= int(min_pos_in_region))
                    if not (size_ok and pos_ok):
                        continue

                    # mejora vs mejor hijo
                    best_f1  = max(A["metrics"]["f1"],        B["metrics"]["f1"])
                    best_rec = max(A["metrics"]["recall"],    B["metrics"]["recall"])
                    lift_ok = (Mu["precision"] >= max(base*float(union_min_lift_precision), base+float(union_min_abs_prec)))
                    improves = (Mu["f1"] >= best_f1 + float(union_min_gain_f1_vs_best)) or \
                               ((Mu["recall"] >= best_rec + float(union_min_gain_rec_vs_best)) and lift_ok)
                    if not improves:
                        continue

                    rule_text = f"({A['rule_text']})  OR  ({B['rule_text']})"
                    pieces_txt = [
                        f"({ ' AND '.join(A.get('rule_pieces', [])) })",
                        "OR",
                        f"({ ' AND '.join(B.get('rule_pieces', [])) })"
                    ]
                    plane_ids = tuple([*(A.get("plane_ids", ())), *(B.get("plane_ids", ()))])
                    sources   = tuple([*(A.get("sources", ())), *(B.get("sources", ()))])
                    unique_plane_count = len(set([pid for pid in plane_ids if pid is not None]))
                    rid_payload = repr(("U", int(cls), dims_use, A["region_id"], B["region_id"], rule_text)).encode("utf-8")
                    rid = f"rg_{len(dims_use)}d_c{int(cls)}_U{hashlib.md5(rid_payload).hexdigest()[:8]}"
                    lift_prec = (Mu["precision"]/base) if base>0 else (math.inf if Mu["precision"]>0 else 0.0)

                    recU = dict(
                        region_id=rid,
                        target_class=int(cls),
                        dims=tuple(int(d) for d in dims_use),
                        plane_ids=plane_ids,
                        sources=sources,
                        rule_text=rule_text,
                        rule_pieces=pieces_txt,
                        metrics=dict(
                            size=int(summary["size"]),
                            precision=float(Mu["precision"]),
                            recall=float(Mu["recall"]),
                            f1=float(Mu["f1"]),
                            baseline=float(base),
                            lift_precision=float(lift_prec)
                        ),
                        metrics_per_class=per_cls,
                        region_summary=summary,
                        projection_ref=str(projection_ref),
                        complexity=dict(
                            num_dims=len(dims_use),
                            num_planes=int(unique_plane_count),
                        ),
                        is_floor=False,
                        generalizes=[], specializes=[],
                        is_pareto=False,
                        family_id=None, parent_id=None, deltas_to_parent=None,
                        planes_used=(A.get("planes_used") or []) + (B.get("planes_used") or []),
                        seed_type="union",
                        is_union=True,
                        union_of=(A["region_id"], B["region_id"]),
                        iou=float(iou)
                    )
                    sig = hashlib.md5(mu_bits.tobytes()).hexdigest()[:12]
                    recU["mask_signature"] = sig
                    recU["_mask_bits"] = mu_bits.copy()
                    if _include_masks_internal:
                        recU["_mask"] = _unpack_bits_col_to_bool(mu_bits, N)

                    # Colocar en el bucket de su dimensionalidad real
                    valuable.setdefault(len(dims_use), []).append(recU)
                    _added_unions += 1
        _log("unions_done", added=int(_added_unions))

    # ------- DEDUP GLOBAL por región idéntica (misma máscara), conservando la más simple -------
    def _cost_rec(r: Dict[str,Any]) -> float:
        return r["complexity"]["num_dims"] + 0.5*r["complexity"]["num_planes"]
    def _better_rec(a: Dict[str,Any], b: Dict[str,Any]) -> bool:
        ca, cb = _cost_rec(a), _cost_rec(b)
        if ca != cb: return ca < cb
        ma, mb = a["metrics"], b["metrics"]
        key_a = (ma["f1"], ma["lift_precision"], ma["size"], -len(a.get("rule_pieces", [])))
        key_b = (mb["f1"], mb["lift_precision"], mb["size"], -len(b.get("rule_pieces", [])))
        return key_a > key_b

    global_best: Dict[Tuple[int, str], Dict[str,Any]] = {}
    for dim_k, L in list(valuable.items()):
        for r in L:
            key = (r["target_class"], r["mask_signature"])
            best = global_best.get(key)
            if best is None or _better_rec(r, best):
                global_best[key] = r

    selected_ids = {r["region_id"] for r in global_best.values()}
    new_valuable: Dict[int, List[Dict[str,Any]]] = {k: [] for k in valuable.keys()}
    for dim_k, L in valuable.items():
        for r in L:
            if r["region_id"] in selected_ids:
                new_valuable[dim_k].append(r)
    valuable = new_valuable

    # ------- ordenar / truncar por dim -------
    def _cost(r): return r["complexity"]["num_dims"] + 0.5*r["complexity"]["num_planes"]
    for dim_k in list(valuable.keys()):
        L = valuable[dim_k]
        if not L:
            continue
        L.sort(key=lambda r: (-r["metrics"]["f1"], -r["metrics"]["lift_precision"], -r["metrics"]["size"], _cost(r)))
        if len(L) > int(max_rules_per_dim):
            L = L[:int(max_rules_per_dim)]
        valuable[dim_k] = L

    # ------- relaciones y familias (por clase y mismas dims) + Pareto -------
    if compute_relations:
        for dim_k, L in valuable.items():
            if not L:
                continue
            by_cls_dims: Dict[Tuple[int,Tuple[int,...]], List[int]] = {}
            for idx, r in enumerate(L):
                by_cls_dims.setdefault((r["target_class"], tuple(r["dims"])), []).append(idx)

            for key, idxs in by_cls_dims.items():
                def cost_i(i):
                    rr = L[i]
                    return rr["complexity"]["num_dims"] + 0.5*rr["complexity"]["num_planes"]

                # Pareto
                dominated = set()
                for i in idxs:
                    ri = L[i]["metrics"]; ci = cost_i(i)
                    for j in idxs:
                        if i == j: continue
                        rj = L[j]["metrics"]; cj = cost_i(j)
                        if (cj <= ci and
                            rj["f1"] >= ri["f1"] and
                            rj["precision"] >= ri["precision"] and
                            rj["recall"] >= ri["recall"] and
                            ((cj < ci) or (rj["f1"] > ri["f1"]) or (rj["precision"] > ri["precision"]) or (rj["recall"] > ri["recall"]))):
                            dominated.add(i); break
                for i in idxs:
                    L[i]["is_pareto"] = (i not in dominated)

                # relaciones por inclusión (¡corregidas!)
                id2rec = {r["region_id"]: r for r in L}
                for a,b in itertools.combinations(idxs, 2):
                    Ra, Rb = L[a], L[b]
                    ma_bits, mb_bits = Ra.get("_mask_bits"), Rb.get("_mask_bits")
                    if ma_bits is None or mb_bits is None:
                        continue
                    if _is_subset_packed(ma_bits, mb_bits):   # mb ⊆ ma => a generaliza b
                        Ra["generalizes"].append(Rb["region_id"])
                        Rb["specializes"].append(Ra["region_id"])
                    elif _is_subset_packed(mb_bits, ma_bits): # ma ⊆ mb => b generaliza a
                        Rb["generalizes"].append(Ra["region_id"])
                        Ra["specializes"].append(Rb["region_id"])

                # familias/parent
                for i in idxs:
                    if L[i]["generalizes"]:
                        cand = [id2rec[rid] for rid in L[i]["generalizes"] if rid in id2rec]
                        if cand:
                            parent = sorted(cand, key=lambda r: (r["complexity"]["num_dims"] + 0.5*r["complexity"]["num_planes"]))[0]
                            L[i]["parent_id"] = parent["region_id"]
                            L[i]["family_id"] = parent.get("family_id") or parent["region_id"]
                            mi, mp = L[i]["metrics"], parent["metrics"]
                            L[i]["deltas_to_parent"] = dict(
                                dF1=mi["f1"]-mp["f1"],
                                dPrecision=mi["precision"]-mp["precision"],
                                dRecall=mi["recall"]-mp["recall"],
                            )
                    if L[i]["family_id"] is None:
                        L[i]["family_id"] = L[i]["region_id"]

            def _cost_r(r):
                return r["complexity"]["num_dims"] + 0.5*r["complexity"]["num_planes"]
            L.sort(key=lambda r: (not r["is_pareto"], -r["metrics"]["f1"], -r["metrics"]["lift_precision"],
                                  -r["metrics"]["size"], _cost_r(r)))

    # Limpieza de campos internos si no se solicitaron máscaras completas
    if not include_masks:
        for L in valuable.values():
            for r in L:
                if "_mask" in r:
                    del r["_mask"]
                if "_mask_bits" in r:
                    del r["_mask_bits"]

    total_runtime = perf_counter() - start_total
    logger.log(
        level,
        "find_low_dim_spaces completado | dimensiones=%s tiempo=%.6f s", 
        {k: len(v) for k, v in valuable.items()},
        total_runtime,
    )
    _log("done", total_time=round(time.time()-t0, 6),
         totals_by_dim={k: len(v) for k,v in valuable.items()})

    if return_logs:
        return valuable, logs
    return valuable


def _merge_kwargs(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(overrides)
    return merged


def find_low_dim_spaces_deterministic(
    X: np.ndarray,
    y: np.ndarray,
    sel: Dict[str, Any],
    **kwargs: Any,
):
    merged = _merge_kwargs(
        dict(
            rng_seed=0,
            enable_unions=False,
            compute_relations=False,
        ),
        kwargs,
    )
    return find_low_dim_spaces(X, y, sel, **merged)


def find_low_dim_spaces_support_first(
    X: np.ndarray,
    y: np.ndarray,
    sel: Dict[str, Any],
    **kwargs: Any,
):
    merged = _merge_kwargs(
        dict(
            max_planes_in_rule=2,
            min_support=max(20, int(kwargs.get("min_support", 30))),
            min_rel_gain_f1=0.10,
            min_abs_gain_prec=0.02,
            sample_limit_per_r=2000,
        ),
        kwargs,
    )
    return find_low_dim_spaces(X, y, sel, **merged)


def find_low_dim_spaces_precision_boost(
    X: np.ndarray,
    y: np.ndarray,
    sel: Dict[str, Any],
    **kwargs: Any,
):
    merged = _merge_kwargs(
        dict(
            min_abs_gain_prec=0.10,
            min_lift_prec=1.50,
            min_rel_gain_f1=0.10,
            enable_unions=False,
        ),
        kwargs,
    )
    return find_low_dim_spaces(X, y, sel, **merged)


__all__ = [
    "find_low_dim_spaces",
    "find_low_dim_spaces_deterministic",
    "find_low_dim_spaces_support_first",
    "find_low_dim_spaces_precision_boost",
]
