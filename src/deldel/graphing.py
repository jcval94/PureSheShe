"""Funciones de graficación y ajuste de fronteras ponderadas."""
import itertools
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, Tuple, Optional, List, Union, Sequence

import logging
from time import perf_counter

import numpy as np

from ._logging_utils import verbosity_to_level
from ._numeric_utils import (
    destandardize_quadratic,
    standardize_matrix,
    unpack_quadratic_parameters,
)
from .engine import (
    build_weighted_frontier as _core_build_weighted_frontier,
    compute_frontier_planes_weighted as _core_compute_frontier_planes_weighted,
    fit_quadrics_from_records_weighted as _core_fit_quadrics_from_records_weighted,
    fit_tls_plane_weighted as _core_fit_tls_plane_weighted,
)


# --- 1) Construcción de puntos frontera + pesos desde records (wrapper) ---
def build_weighted_frontier(
    records: Iterable,
    prefer_cp: bool = True,
    success_only: bool = True,
    weight_map: str = "power",    # "power" | "sigmoid" | "softmax"
    gamma: float = 2.0,           # para power
    temp: float = 0.15,           # para softmax (por par) o sigmoide
    sigmoid_center: Optional[float] = None,  # si None -> mediana por par
    density_k: Optional[int] = 8, # None para desactivar corrección densidad
    *,
    verbosity: int = 0,
) -> Tuple[Dict[Tuple[int,int], np.ndarray], Dict[Tuple[int,int], np.ndarray], Dict[Tuple[int,int], np.ndarray]]:
    logger = logging.getLogger(__name__)
    level = verbosity_to_level(verbosity)
    start = perf_counter()
    logger.log(
        level,
        "build_weighted_frontier: inicio | prefer_cp=%s success_only=%s map=%s density_k=%s",
        prefer_cp,
        success_only,
        weight_map,
        density_k,
    )

    F_by, B_by, W_by = _core_build_weighted_frontier(
        records,
        prefer_cp=prefer_cp,
        success_only=success_only,
        weight_map=weight_map,
        gamma=gamma,
        temp=temp,
        sigmoid_center=sigmoid_center,
        density_k=density_k,
    )

    logger.log(
        level,
        "build_weighted_frontier: fin en %.4fs | pares=%s",
        perf_counter() - start,
        list(F_by.keys()),
    )

    return F_by, B_by, W_by

# --- 2) TLS (plano) ponderado: PCA ponderado ---
def fit_tls_plane_weighted(F: np.ndarray, w: np.ndarray, *, verbosity: int = 0):
    logger = logging.getLogger(__name__)
    level = verbosity_to_level(verbosity)
    start = perf_counter()
    F = np.asarray(F, float)
    w = np.asarray(w, float).reshape(-1)
    logger.log(level, "fit_tls_plane_weighted: F_shape=%s w_size=%d", F.shape, w.size)

    n, b, mu = _core_fit_tls_plane_weighted(F, w)

    logger.log(level, "fit_tls_plane_weighted: fin en %.6fs", perf_counter() - start)
    return n.astype(float), float(b), mu.astype(float)

# --- 3) Cuádrica (SVD algebraico) ponderada ---
def fit_quadric_svd_weighted(F: np.ndarray, poly2_features_fn):
    Z, mu, sd = standardize_matrix(F)
    Phi, idx = poly2_features_fn(Z)
    raise RuntimeError("Usa fit_quadrics_from_records_weighted(...) que provee 'w'.")

# --- 4) Logística con sample_weight ---
def logistic_fit_with_weights(Phi: np.ndarray, y: np.ndarray, w: np.ndarray, C: float = 10.0):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, penalty="l2", max_iter=800, class_weight=None)
    clf.fit(Phi, y, sample_weight=w)
    return clf

# --- 5) Wrappers listos para tus records ---
def compute_frontier_planes_weighted(
    records: Iterable,
    prefer_cp: bool = True,
    success_only: bool = True,
    weight_map: str = "power",
    gamma: float = 2.0,
    temp: float = 0.15,
    density_k: Optional[int] = 8,
    orient_with_bases: bool = True,
    eps_orient: float = 1e-3,
    *,
    verbosity: int = 0,
):    
    logger = logging.getLogger(__name__)
    level = verbosity_to_level(verbosity)
    t0 = perf_counter()
    logger.log(level, "compute_frontier_planes_weighted: inicio | orient=%s", orient_with_bases)

    planes = _core_compute_frontier_planes_weighted(
        records,
        prefer_cp=prefer_cp,
        success_only=success_only,
        weight_map=weight_map,
        gamma=gamma,
        temp=temp,
        density_k=density_k,
        orient_with_bases=orient_with_bases,
        eps_orient=eps_orient,
    )

    logger.log(level, "compute_frontier_planes_weighted: fin en %.4fs | planos=%d", perf_counter() - t0, len(planes))
    return planes

def fit_quadrics_from_records_weighted(
    records: Iterable,
    mode: str = "svd",           # 'svd' | 'logistic'
    prefer_cp: bool = True,
    success_only: bool = True,
    weight_map: str = "power",
    gamma: float = 2.0,
    temp: float = 0.15,
    density_k: Optional[int] = 8,
    eps: float = 1e-3,
    C: float = 10.0,
    n_jobs: Optional[int] = None,
    *,
    verbosity: int = 0,
):
    logger = logging.getLogger(__name__)
    level = verbosity_to_level(verbosity)
    t0 = perf_counter()
    logger.log(level, "fit_quadrics_from_records_weighted: inicio | mode=%s", mode)
    models = _core_fit_quadrics_from_records_weighted(
        records,
        mode=mode,
        prefer_cp=prefer_cp,
        success_only=success_only,
        weight_map=weight_map,
        gamma=gamma,
        temp=temp,
        density_k=density_k,
        eps=eps,
        C=C,
        n_jobs=n_jobs,
    )

    logger.log(level, "fit_quadrics_from_records_weighted: fin en %.4fs | modelos=%d", perf_counter() - t0, len(models))
    return models


def plot_frontiers_implicit_interactive_v2(
    records: Iterable,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    planes: Optional[Dict[Tuple[int,int], Dict[str, np.ndarray]]] = None,
    planes_multi: Optional[Dict[Tuple[int,int], List[Dict[str, np.ndarray]]]] = None,
    quadrics: Optional[Dict[Tuple[int,int], Dict[str, np.ndarray]]] = None,
    cubic_models: Optional[Dict[Tuple[int,int], Dict[str, np.ndarray]]] = None,

    dims: Tuple[int, ...] = (0,1,2),
    dims_options: Optional[List[Tuple[int, ...]]] = None,
    feature_names: Optional[List[str]] = None,
    pair_filter: Optional[Iterable[Tuple[int,int]]] = None,

    prefer_cp: bool = True,
    success_only: bool = True,

    show_X: Optional[bool] = None,
    show_frontier: Optional[bool] = None,
    show_planes: Optional[bool] = None,
    show_quadrics: Optional[bool] = None,
    show_cubics: Optional[bool] = None,

    # Direcciones x0 → frontera
    show_directions: str = "lines",   # "none"|"points"|"lines"|"both"
    arrows_per_pair: Optional[int] = None,
    arrow_scale: float = 1.0,
    direction_opacity: float = 0.5,   # <- NUEVO: opacidad parametrizable para líneas y puntos
    plane_opacity: float = 0.08,

    # Detalle
    detail: str = "auto",
    decimate_X: Optional[int] = None,
    decimate_frontier: Optional[int] = None,
    grid_res_2d: Optional[int] = None,
    grid_res_3d: Optional[int] = None,
    quadric_alpha: float = 0.28,
    iso_level: float = 0.0,
    isosurface_epsilon: float = 1e-9,

    # Extensión y slice
    extend: Union[float, Tuple[float, ...]] = 1.0,
    clamp_extend_to_X: bool = True,
    plane_mode: str = "fit_dims",
    slice_filter_tol: Optional[float] = None,
    slice_filter_norm: str = "l1",     # "l1"|"l2"|"linf"

    # Recursos, RNG y salida
    max_voxels: int = 700_000,
    random_state: Optional[int] = 0,
    renderer: Optional[str] = None,
    show: bool = True,
    return_fig: bool = False,
    save_html: Optional[str] = None,

    title: str = "DelDel — Interiores, Fronteras y Superficies (v3_2)"
):
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio

    # ---------- Render target ----------
    if renderer:
        try:
            pio.renderers.default = renderer
        except Exception:
            pass
    else:
        try:
            import google.colab  # noqa
            pio.renderers.default = "colab"
        except Exception:
            pass

    # ---------- Helpers ----------
    def _axis_label(idx: int) -> str:
        if feature_names and 0 <= idx < len(feature_names):
            return feature_names[idx]
        return f"x{idx}"

    def _pair_means(full_F: np.ndarray) -> np.ndarray:
        return full_F.mean(axis=0)

    def _unique_rows_tol(a: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        if a.size == 0:
            return a.reshape(0, a.shape[-1] if a.ndim==2 else 0)
        q = np.round(a / max(tol, 1e-12), 0).astype(np.int64)
        _, idx = np.unique(q, axis=0, return_index=True)
        return a[np.sort(idx)]

    def _g_quadric_eval(P: np.ndarray, Q: np.ndarray, r: np.ndarray, c: float) -> np.ndarray:
        return np.einsum("ni,ij,nj->n", P, Q, P) + P @ r + c

    def _plane_opacity_for_idx(idx_pl: Optional[int] = None) -> float:
        base = float(plane_opacity)
        if idx_pl is None:
            return base
        return base * (0.22 + 0.06 * ((idx_pl - 1) % 3)) / 0.25

    def _g_cubic_eval(P: np.ndarray, model: dict) -> np.ndarray:
        Z = (P - model["mu"]) / model["sd"]
        n, d = Z.shape
        cols = []
        for i in range(d): cols.append(Z[:,i])
        for i in range(d): cols.append(Z[:,i]**2)
        for i in range(d):
            for j in range(i+1, d):
                cols.append(Z[:,i]*Z[:,j])
        for i in range(d): cols.append(Z[:,i]**3)
        for i in range(d):
            for j in range(d):
                if i==j: continue
                cols.append((Z[:,i]**2)*Z[:,j])
        for i in range(d):
            for j in range(i+1, d):
                for k in range(j+1, d):
                    cols.append(Z[:,i]*Z[:,j]*Z[:,k])
        cols.append(np.ones(n))
        Phi = np.column_stack(cols)
        w = model["w"]
        if w.shape[0] == Phi.shape[1]-1:
            w = np.r_[w, 0.0]
        return Phi @ w

    def _box_from_points(P: np.ndarray, pad_ratio: float = 0.07):
        lo = P.min(axis=0); hi = P.max(axis=0)
        span = hi - lo
        span = np.where(span < 1e-12, 1e-12, span)
        pad = pad_ratio * (span + 1e-12)
        return lo - pad, hi + pad

    def _apply_extend(lo: np.ndarray, hi: np.ndarray, dims_len: int, loX: np.ndarray, hiX: np.ndarray):
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        half = np.where(half < 5e-10, 5e-10, half)
        if isinstance(extend, (tuple, list, np.ndarray)):
            fac = np.asarray(extend, float)
            if fac.size not in (1, dims_len):
                raise ValueError(f"'extend' debe ser escalar o de longitud {dims_len}")
            if fac.size == 1:
                fac = np.repeat(fac, dims_len)
        else:
            fac = np.repeat(float(extend), dims_len)
        new_lo = mid - half * fac
        new_hi = mid + half * fac
        if clamp_extend_to_X:
            new_lo = np.maximum(new_lo, loX)
            new_hi = np.minimum(new_hi, hiX)
            mask = new_lo > new_hi
            if np.any(mask):
                fix = 0.5 * (new_lo[mask] + new_hi[mask])
                new_lo[mask] = fix
                new_hi[mask] = fix
        return new_lo, new_hi

    # --- Cache de grillas ---
    grid2d_cache: Dict[Tuple[Tuple[float,float], Tuple[float,float], int], Tuple[np.ndarray,np.ndarray]] = {}
    grid3d_cache: Dict[Tuple[Tuple[float,float,float], Tuple[float,float,float], Tuple[int,int,int]], Tuple[np.ndarray,np.ndarray,np.ndarray]] = {}

    def _grid_points_2d(lo2: np.ndarray, hi2: np.ndarray, res: int):
        key = (tuple(np.round(lo2,12)), tuple(np.round(hi2,12)), int(res))
        if key in grid2d_cache:
            return grid2d_cache[key]
        xs = np.linspace(lo2[0], hi2[0], res)
        ys = np.linspace(lo2[1], hi2[1], res)
        Xg, Yg = np.meshgrid(xs, ys)
        grid2d_cache[key] = (Xg, Yg)
        return Xg, Yg

    def _grid_points_3d(lo3, hi3, base,
                        *, min_res=8, max_res=96,
                        max_voxels=max_voxels, anisotropy_cap=6.0, eps_span=1e-6):
        lo3 = np.asarray(lo3, float); hi3 = np.asarray(hi3, float)
        span = np.maximum(hi3 - lo3, 0.0)
        deg = span <= eps_span
        nondeg_span = np.where(deg, np.nan, span)
        m = np.nanmean(nondeg_span)
        if not np.isfinite(m) or m <= 0:
            ratios = np.ones(3)
        else:
            ratios = span / m
            ratios = np.clip(ratios, 1.0/anisotropy_cap, anisotropy_cap)
        res = np.rint(base * ratios).astype(int)
        res = np.clip(res, min_res, max_res)
        res[deg] = 2

        def _downscale_to_limit(r, limit):
            r = r.astype(int)
            while int(r[0]) * int(r[1]) * int(r[2]) > limit:
                r = np.maximum(2, np.floor(r * 0.9)).astype(int)
            return r
        res = _downscale_to_limit(res, max_voxels)

        key = (tuple(np.round(lo3,12)), tuple(np.round(hi3,12)), tuple(map(int, res.tolist())))
        if key in grid3d_cache:
            return grid3d_cache[key]

        xs = np.linspace(lo3[0], hi3[0], int(res[0]), dtype=np.float32)
        ys = np.linspace(lo3[1], hi3[1], int(res[1]), dtype=np.float32)
        zs = np.linspace(lo3[2], hi3[2], int(res[2]), dtype=np.float32)
        out = np.meshgrid(xs, ys, zs, indexing="xy")
        grid3d_cache[key] = out
        return out

    def _make_full_points_from_2d_grid(Xg, Yg, template: np.ndarray, dims_opt: Tuple[int,...]):
        n_tot = Xg.size
        P = np.repeat(template.reshape(1,-1), n_tot, axis=0)
        P[:, dims_opt[0]] = Xg.reshape(-1)
        P[:, dims_opt[1]] = Yg.reshape(-1)
        return P

    def _make_full_points_from_3d_grid(Xg, Yg, Zg, template: np.ndarray, dims_opt: Tuple[int,...]):
        n_tot = Xg.size
        P = np.repeat(template.reshape(1,-1), n_tot, axis=0)
        P[:, dims_opt[0]] = Xg.reshape(-1)
        P[:, dims_opt[1]] = Yg.reshape(-1)
        P[:, dims_opt[2]] = Zg.reshape(-1)
        return P

    # === records → fronteras y bases ===
    def _stack_frontier_sets_from_records(records, prefer_cp=True, success_only=True):
        from collections import defaultdict
        F_by, B_by = defaultdict(list), defaultdict(list)
        d = None
        for r in records:
            a, b = int(r.y0), int(r.y1)
            if a == b:
                continue
            if success_only and not bool(getattr(r, "success", True)):
                continue
            x0 = np.asarray(r.x0, float).reshape(1, -1)
            if prefer_cp and getattr(r, "cp_count", 0) and np.asarray(getattr(r, "cp_x")).size > 0:
                F = np.asarray(r.cp_x, float)
            else:
                F = np.asarray(r.x1, float).reshape(1, -1)
            if d is None: d = F.shape[1]
            m = F.shape[0]
            F_by[(a,b)].append(F)
            B_by[(a,b)].append(np.repeat(x0, m, axis=0))
        F_by = {k: _unique_rows_tol(np.vstack(v)) for k, v in F_by.items()}
        B_by = {k: np.vstack(v) for k, v in B_by.items()}
        return F_by, B_by, (0 if d is None else d)

    def _frontier_points_and_weights_from_records(records, prefer_cp=True, success_only=True):
        from collections import defaultdict
        Ftmp, Stmp = defaultdict(list), defaultdict(list)
        for r in records:
            if success_only and not bool(getattr(r, "success", True)):
                continue
            a, b = int(r.y0), int(r.y1)
            if a == b:
                continue
            score = float(getattr(r, "final_score", 1.0))
            if prefer_cp and getattr(r, "cp_count", 0) and np.asarray(getattr(r, "cp_x")).size>0:
                F = np.asarray(r.cp_x, float)
            else:
                F = np.asarray(r.x1, float).reshape(1,-1)
            m = F.shape[0]
            Ftmp[(a,b)].append(F)
            Stmp[(a,b)].append(np.full(m, score, float))
        F_by = {k: np.vstack(v) for k,v in Ftmp.items()}
        W_by = {}
        for k in F_by.keys():
            s = np.concatenate(Stmp[k]).astype(float)
            w = np.clip(s, 0.0, 1.0)**2
            W_by[k] = w / (w.sum() + 1e-12)
        return F_by, W_by

    def _fit_tls_plane_in_dims(F_full: np.ndarray, w: np.ndarray, dims_sel: Tuple[int,...]):
        P = F_full[:, dims_sel]
        w = (w / (w.sum()+1e-12)).reshape(-1)
        mu = (w[:,None] * P).sum(axis=0)
        Z  = P - mu
        Sw = Z.T @ (w[:,None] * Z)
        evals, evecs = np.linalg.eigh(Sw)
        n_sub = evecs[:, np.argmin(evals)]
        b_eff = -float(n_sub @ mu)
        return n_sub.astype(float), float(b_eff)

    def _restrict_plane_to_dims(n: np.ndarray, b: float, mu: np.ndarray, dims_sel: Tuple[int, ...]):
        d = n.size
        dims_sel = tuple(int(i) for i in dims_sel)
        other = [j for j in range(d) if j not in dims_sel]
        b_eff = b + float(np.dot(n[other], mu[other])) if other else b
        n_sub = n[list(dims_sel)]
        return n_sub.astype(float), float(b_eff)

    def _plane_surface_mesh_3d(n_sub: np.ndarray, b_eff: float, lo: np.ndarray, hi: np.ndarray, res: int = 14):
        n_sub = np.asarray(n_sub, float).reshape(3)
        k = int(np.argmax(np.abs(n_sub)))
        free = [i for i in (0,1,2) if i != k]
        ax0, ax1 = free[0], free[1]
        g0 = np.linspace(lo[ax0], hi[ax0], res)
        g1 = np.linspace(lo[ax1], hi[ax1], res)
        G0, G1 = np.meshgrid(g0, g1)
        denom = n_sub[k] if abs(n_sub[k]) > 1e-12 else 1e-12
        Gk = -(n_sub[ax0]*G0 + n_sub[ax1]*G1 + b_eff) / denom
        grids = {ax0: G0, ax1: G1, k: Gk}
        Xp, Yp, Zp = grids[0], grids[1], grids[2]
        return Xp, Yp, Zp

    # ---------- Validaciones dims ----------
    X = np.asarray(X, float)
    d_total = X.shape[1]
    assert len(dims) in (2,3), "dims debe tener longitud 2 o 3"
    dims = tuple(int(i) for i in dims)
    for i in dims:
        assert 0 <= i < d_total, f"Índice dims fuera de rango: {i}"

    # ---------- Fronteras desde records ----------
    frontier_by_pair, bases_by_pair, d_detect = _stack_frontier_sets_from_records(records, prefer_cp, success_only)
    if d_detect == 0:
        raise ValueError("No hay puntos de frontera para graficar.")
    pair_keys = sorted(frontier_by_pair.keys())

    if pair_filter:
        pf = set(map(tuple, pair_filter))
        pair_keys = [p for p in pair_keys if p in pf]
        if not pair_keys:
            raise ValueError("pair_filter dejó 0 pares para graficar.")

    if plane_mode == "fit_dims":
        F_for_fit, W_for_fit = _frontier_points_and_weights_from_records(records, prefer_cp, success_only)
    else:
        F_for_fit, W_for_fit = {}, {}

    def _coerce_planes_multi(planes, planes_multi):
        if planes_multi is not None:
            return {k: list(v) for k, v in planes_multi.items() if isinstance(v, (list, tuple))}
        out = {}
        if planes is None:
            return out
        for k, meta in planes.items():
            out[k] = list(meta) if isinstance(meta, (list, tuple)) else [meta]
        return out

    planes_list_by_pair = _coerce_planes_multi(planes, planes_multi)
    if (not planes_list_by_pair) and (planes is None):
        try:
            base_multi = compute_frontier_planes_weighted_autoK(
                records,
                prefer_cp=prefer_cp, success_only=success_only,
                weight_map="power", gamma=2.0, temp=0.15, density_k=8,
                max_planes_per_pair=5, plane_p90_tol="auto",
                split_gain_min=0.25, min_points_per_plane=6,
                sample_cap=2000, random_state=0
            )
            planes_list_by_pair = {
                k: [ {"n":m["n"], "b":m["b"], "mu":m["mu"]} for m in lst ]
                for k, lst in base_multi.items()
            }
        except Exception:
            planes_list_by_pair = {}

    pair_templates = {p: _pair_means(frontier_by_pair[p]) for p in pair_keys}

    # ---------- Colores por clase ----------
    import plotly.express as px
    class_colors = {}
    if (y is not None) and (len(y) == X.shape[0]):
        classes = np.unique(y)
        pal = px.colors.qualitative.Plotly
        for k, c in enumerate(classes):
            class_colors[int(c)] = pal[k % len(pal)]
    else:
        classes = np.array([0])
        class_colors[0] = "rgba(130,130,130,0.70)"
        y = np.zeros(X.shape[0], dtype=int)

    # ---------- dims_options ----------
    if dims_options is None:
        dims_options = [dims]
    else:
        dims_options = [tuple(map(int, opt)) for opt in dims_options if len(opt) in (2,3)]
    seen = set(); _tmp = []
    for op in dims_options:
        if op not in seen:
            _tmp.append(op); seen.add(op)
    dims_options = _tmp

    # ---------- presets de 'detail' ----------
    def _resolve_detail(detail: str, nX: int, n_pairs: int, n_frontier_tot: int):
        if detail not in {"auto","fast","balanced","high"}:
            detail = "balanced"
        if detail == "auto":
            load = nX + n_frontier_tot + 2000*n_pairs
            detail = "fast" if load > 50_000 else "balanced"
        if detail == "fast":
            return dict(grid2d=120, grid3d=16, decX=4, decF=4, arrows=20)
        if detail == "balanced":
            return dict(grid2d=200, grid3d=28, decX=3, decF=3, arrows=40)
        return dict(grid2d=300, grid3d=40, decX=2, decF=2, arrows=60)

    n_frontier_tot = sum(frontier_by_pair[p].shape[0] for p in pair_keys)
    preset = _resolve_detail(detail, X.shape[0], len(pair_keys), n_frontier_tot)

    if grid_res_2d is None: grid_res_2d = preset["grid2d"]
    if grid_res_3d is None: grid_res_3d = preset["grid3d"]
    if decimate_X is None: decimate_X = preset["decX"]
    if decimate_frontier is None: decimate_frontier = preset["decF"]
    if arrows_per_pair is None: arrows_per_pair = preset["arrows"]

    if show_X is None: show_X = True
    if show_frontier is None: show_frontier = True
    if show_planes is None:
        show_planes = bool(planes_list_by_pair) or bool(planes) or (plane_mode == "fit_dims")
    if show_quadrics is None: show_quadrics = bool(quadrics)
    if show_cubics is None: show_cubics = bool(cubic_models)

    show_dirs_mode = str(show_directions).lower()
    show_dirs_points = show_dirs_mode in ("points","both")
    show_dirs_lines  = show_dirs_mode in ("lines","both")

    # Validadores de modelos
    def _parse_quadric(mdl):
        try:
            Q = np.asarray(mdl["Q"], float); r = np.asarray(mdl["r"], float); c = float(mdl["c"])
            if Q.shape != (d_total, d_total) or r.shape != (d_total,):
                print(f"[WARN] Cuádrica omitida por forma incompatible: Q{Q.shape}, r{r.shape}, d={d_total}")
                return None
            return Q, r, c
        except Exception as e:
            print(f"[WARN] Cuádrica inválida: {e}")
            return None

    def _parse_cubic(mdl):
        try:
            mu = np.asarray(mdl["mu"], float); sd = np.asarray(mdl["sd"], float); w = np.asarray(mdl["w"], float)
            if mu.shape != (d_total,) or sd.shape != (d_total,):
                print(f"[WARN] Cúbica omitida por forma incompatible: mu{mu.shape}, sd{sd.shape}, d={d_total}")
                return None
            return {"mu":mu, "sd":sd, "w":w}
        except Exception as e:
            print(f"[WARN] Cúbica inválida: {e}")
            return None

    # ---------- Construcción de trazas ----------
    all_traces = []
    all_vis_masks = []

    def _stratified_indices(y, step):
        if step is None or step < 2:
            return np.arange(y.size)
        sel = []
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            sel.append(idx[::step])
        return np.concatenate(sel) if sel else np.arange(y.size)

    rng = np.random.RandomState(None if random_state is None else int(random_state))

    # Flags para mostrar un único ítem de leyenda por grupo
    legend_item_lines_done = False
    legend_item_points_done = False

    for opt_i, dims_opt in enumerate(dims_options):
        is_3d = (len(dims_opt) == 3)
        vis_here = []
        legend_planes_done = set()
        legend_planes_done = set()

        Xopt = X[:, dims_opt]
        loX = Xopt.min(axis=0)
        hiX = Xopt.max(axis=0)

        # ---- X por clase ----
        if show_X:
            idx_all = _stratified_indices(y, decimate_X)
            for c in np.unique(y[idx_all]):
                sel = idx_all[y[idx_all] == c]
                if sel.size == 0: continue
                P = X[sel][:, dims_opt]
                if is_3d:
                    tr = go.Scatter3d(
                        x=P[:,0], y=P[:,1], z=P[:,2], mode="markers",
                        name=f"Clase {c} (X)", legendgroup=f"class-{c}",
                        marker=dict(size=3, opacity=0.55, color=class_colors[int(c)]),
                        hovertemplate=f"Clase {c}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}<br>z:%{{z:.3f}}"
                    )
                else:
                    tr = go.Scatter(
                        x=P[:,0], y=P[:,1], mode="markers",
                        name=f"Clase {c} (X)", legendgroup=f"class-{c}",
                        marker=dict(size=5, opacity=0.55, color=class_colors[int(c)]),
                        hovertemplate=f"Clase {c}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}"
                    )
                all_traces.append(tr); vis_here.append(True)

        # ---- Fronteras (puntos) ----
        if show_frontier:
            for p in pair_keys:
                F_full = frontier_by_pair[p]
                F_use = F_full
                if plane_mode == "slice" and slice_filter_tol is not None and (planes_list_by_pair.get(p) or (planes and p in planes)):
                    metas = planes_list_by_pair.get(p, [planes[p]] if (planes and p in planes) else [])
                    other = [j for j in range(d_total) if j not in dims_opt]
                    if other and metas:
                        mu_other = np.mean([np.asarray(m["mu"], float)[other] for m in metas], axis=0)
                        diff = np.abs(F_full[:, other] - mu_other)
                        if slice_filter_norm == "l2":
                            dist = np.sqrt((diff**2).sum(axis=1)) / (diff.shape[1]**0.5 if diff.shape[1] else 1.0)
                        elif slice_filter_norm == "linf":
                            dist = diff.max(axis=1)
                        else:  # "l1"
                            dist = diff.mean(axis=1)
                        mask = (dist <= float(slice_filter_tol))
                        if np.any(mask): F_use = F_full[mask]
                if F_use.size == 0: continue
                if isinstance(decimate_frontier, int) and decimate_frontier >= 2:
                    F_use = F_use[::decimate_frontier]
                Qp = F_use[:, dims_opt]
                col = "rgba(30,30,30,0.85)"; outline = "rgba(0,0,0,0.6)"
                if is_3d:
                    tr = go.Scatter3d(
                        x=Qp[:,0], y=Qp[:,1], z=Qp[:,2], mode="markers",
                        name=f"Frontera {p}", legendgroup=f"pair-{p}",
                        marker=dict(size=4, color=col, line=dict(width=1, color=outline)),
                        hovertemplate=f"{p}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}<br>z:%{{z:.3f}}"
                    )
                else:
                    tr = go.Scatter(
                        x=Qp[:,0], y=Qp[:,1], mode="markers",
                        name=f"Frontera {p}", legendgroup=f"pair-{p}",
                        marker=dict(size=6, color=col, line=dict(width=1, color=outline)),
                        hovertemplate=f"{p}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}"
                    )
                all_traces.append(tr); vis_here.append(True)

        # ---- Planos ----
        if show_planes:
            for p in pair_keys:
                Fp = frontier_by_pair[p][:, dims_opt]
                if Fp.size == 0:
                    continue
                lo, hi = _box_from_points(Fp, pad_ratio=0.05)
                lo, hi = _apply_extend(lo, hi, dims_len=len(dims_opt), loX=loX, hiX=hiX)

                planes_for_p = planes_list_by_pair.get(p, None)
                if planes_for_p:
                    for idx_pl, meta in enumerate(planes_for_p, start=1):
                        n = np.asarray(meta["n"], float); b0 = float(meta["b"]); mu = np.asarray(meta["mu"], float)
                        n_sub, b_eff = _restrict_plane_to_dims(n, b0, mu, dims_opt)
                        if is_3d:
                            Xp_s, Yp_s, Zp_s = _plane_surface_mesh_3d(n_sub, b_eff, lo, hi, res=14)
                            op = _plane_opacity_for_idx(idx_pl)
                            col = "rgba(55,55,55,0.95)"
                            tr = go.Surface(
                                x=Xp_s, y=Yp_s, z=Zp_s, name=f"Plano {p} #{idx_pl}",
                                legendgroup=f"pair-{p}", showscale=False, opacity=op,
                                colorscale=[[0, col],[1, col]]
                            )
                        else:
                            xs = np.linspace(lo[0], hi[0], 300)
                            op = _plane_opacity_for_idx(idx_pl)
                            col = f"rgba(50,50,50,{op})"
                            if abs(n_sub[1]) > 1e-12:
                                ys = -(n_sub[0]*xs + b_eff) / n_sub[1]
                                tr = go.Scatter(x=xs, y=ys, mode="lines",
                                                name=f"Plano {p} #{idx_pl}", legendgroup=f"pair-{p}",
                                                line=dict(width=2, color=col))
                            else:
                                x0p = -b_eff / (n_sub[0] if abs(n_sub[0])>1e-12 else 1e-12)
                                tr = go.Scatter(x=[x0p,x0p], y=[lo[1],hi[1]], mode="lines",
                                                name=f"Plano {p} #{idx_pl}", legendgroup=f"pair-{p}",
                                                line=dict(width=2, color=col))
                        all_traces.append(tr); vis_here.append(True)
                else:
                    if plane_mode == "fit_dims":
                        F_full = F_for_fit.get(p, frontier_by_pair[p])
                        w = W_for_fit.get(p, np.ones(F_full.shape[0], float)/max(1, F_full.shape[0]))
                        n_sub, b_eff = _fit_tls_plane_in_dims(F_full, w, dims_opt)
                    else:
                        if not planes or p not in planes:
                            continue
                        n = np.asarray(planes[p]["n"], float)
                        b0 = float(plane_meta["b"])
                        mu = np.asarray(plane_meta["mu"], float)
                        n_sub, b_eff = _restrict_plane_to_dims(n, b0, mu, dims_opt)

                    if is_3d:
                        Xp_s, Yp_s, Zp_s = _plane_surface_mesh_3d(n_sub, b_eff, lo, hi, res=14)
                        tr = go.Surface(
                            x=Xp_s, y=Yp_s, z=Zp_s, name=f"Plano {p}", legendgroup=f"pair-{p}",
                            showscale=False, opacity=_plane_opacity_for_idx(),
                            colorscale=[[0, "rgba(50,50,50,0.9)"], [1, "rgba(50,50,50,0.9)"]]
                        )
                    else:
                        xs = np.linspace(lo[0], hi[0], 300)
                        col = f"rgba(50,50,50,{_plane_opacity_for_idx()})"
                        if abs(n_sub[1]) > 1e-12:
                            ys = -(n_sub[0]*xs + b_eff) / n_sub[1]
                            tr = go.Scatter(x=xs, y=ys, mode="lines",
                                            name=f"Plano {p}", legendgroup=f"pair-{p}",
                                            line=dict(width=2, color=col))
                        else:
                            x0p = -b_eff / (n_sub[0] if abs(n_sub[0])>1e-12 else 1e-12)
                            tr = go.Scatter(x=[x0p,x0p], y=[lo[1],hi[1]], mode="lines",
                                            name=f"Plano {p}", legendgroup=f"pair-{p}",
                                            line=dict(width=2, color=col))
                    all_traces.append(tr); vis_here.append(True)

        # ---- Cuádricas ----
        if show_quadrics and quadrics:
            for p in pair_keys:
                mdl = quadrics.get(p)
                if mdl is None: continue
                parsed = _parse_quadric(mdl)
                if parsed is None:
                    continue
                Q, r, cst = parsed
                Fp = frontier_by_pair[p][:, dims_opt]
                lo, hi = _box_from_points(Fp, pad_ratio=0.06)
                lo, hi = _apply_extend(lo, hi, dims_len=len(dims_opt), loX=loX, hiX=hiX)
                template = pair_templates[p]
                if is_3d:
                    Xg, Yg, Zg = _grid_points_3d(lo, hi, grid_res_3d)
                    Pfull = _make_full_points_from_3d_grid(Xg, Yg, Zg, template, dims_opt)
                    G = _g_quadric_eval(Pfull, Q, r, cst).reshape(Xg.shape)
                    tr = go.Isosurface(
                        x=Xg.flatten(), y=Yg.flatten(), z=Zg.flatten(),
                        value=G.flatten(),
                        isomin=iso_level - isosurface_epsilon, isomax=iso_level + isosurface_epsilon, surface_count=1,
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        showscale=False, opacity=quadric_alpha,
                        name=f"Cuádrica {p}", legendgroup=f"pair-{p}",
                        colorscale=[[0,"#444"],[1,"#444"]]
                    )
                else:
                    Xg, Yg = _grid_points_2d(lo, hi, grid_res_2d)
                    Pfull = _make_full_points_from_2d_grid(Xg, Yg, template, dims_opt)
                    G = _g_quadric_eval(Pfull, Q, r, cst).reshape(Xg.shape)
                    tr = go.Contour(
                        x=Xg[0,:], y=Yg[:,0], z=G,
                        contours=dict(start=iso_level, end=iso_level, size=1.0),
                        showscale=False, name=f"Cuádrica {p}", legendgroup=f"pair-{p}",
                        line=dict(width=3, color="rgba(20,20,20,0.95)")
                    )
                all_traces.append(tr); vis_here.append(True)

        # ---- Cúbicas ----
        if show_cubics and cubic_models:
            for p in pair_keys:
                mdl = cubic_models.get(p)
                if mdl is None: continue
                mdlp = _parse_cubic(mdl)
                if mdlp is None:
                    continue
                Fp = frontier_by_pair[p][:, dims_opt]
                lo, hi = _box_from_points(Fp, pad_ratio=0.06)
                lo, hi = _apply_extend(lo, hi, dims_len=len(dims_opt), loX=loX, hiX=hiX)
                template = pair_templates[p]
                if is_3d:
                    Xg, Yg, Zg = _grid_points_3d(lo, hi, max(16, grid_res_3d//2))
                    Pfull = _make_full_points_from_3d_grid(Xg, Yg, Zg, template, dims_opt)
                    G = _g_cubic_eval(Pfull, mdlp).reshape(Xg.shape)
                    tr = go.Isosurface(
                        x=Xg.flatten(), y=Yg.flatten(), z=Zg.flatten(),
                        value=G.flatten(),
                        isomin=iso_level - isosurface_epsilon, isomax=iso_level + isosurface_epsilon, surface_count=1,
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        showscale=False, opacity=quadric_alpha,
                        name=f"Cúbica {p}", legendgroup=f"pair-{p}",
                        colorscale=[[0,"#777"],[1,"#777"]]
                    )
                else:
                    Xg, Yg = _grid_points_2d(lo, hi, max(100, grid_res_2d//2))
                    Pfull = _make_full_points_from_2d_grid(Xg, Yg, template, dims_opt)
                    G = _g_cubic_eval(Pfull, mdlp).reshape(Xg.shape)
                    tr = go.Contour(
                        x=Xg[0,:], y=Yg[:,0], z=G,
                        contours=dict(start=iso_level, end=iso_level, size=1.0),
                        showscale=False, name=f"Cúbica {p}", legendgroup=f"pair-{p}",
                        line=dict(width=2, color="rgba(80,80,80,0.95)")
                    )
                all_traces.append(tr); vis_here.append(True)

        # ---- Direcciones (con opacidad parametrizable + leyenda agrupada) ----
        if (show_dirs_points or show_dirs_lines):
            for p in pair_keys:
                F = frontier_by_pair[p]; B = bases_by_pair.get(p)
                if F.size == 0 or B is None or B.size == 0:
                    continue
                m = F.shape[0]; k = min(int(arrows_per_pair), m)
                if k <= 0:
                    continue
                idx = rng.choice(m, size=k, replace=False)
                Fp = F[idx][:, dims_opt]; Bp = B[idx][:, dims_opt]
                U  = (Fp - Bp) * float(arrow_scale)

                # Líneas
                if show_dirs_lines:
                    if is_3d:
                        xs, ys, zs = [], [], []
                        for i in range(k):
                            xs += [Bp[i,0], Bp[i,0]+U[i,0], None]
                            ys += [Bp[i,1], Bp[i,1]+U[i,1], None]
                            zs += [Bp[i,2], Bp[i,2]+U[i,2], None]
                        tr = go.Scatter3d(
                            x=xs, y=ys, z=zs, mode="lines",
                            name=("Direcciones (líneas)" if not legend_item_lines_done else None),
                            legendgroup="dirs-lines",
                            showlegend=not legend_item_lines_done,
                            line=dict(width=2, color=f"rgba(30,30,30,{float(direction_opacity)})")
                        )
                    else:
                        xs, ys = [], []
                        for i in range(k):
                            xs += [Bp[i,0], Bp[i,0]+U[i,0], None]
                            ys += [Bp[i,1], Bp[i,1]+U[i,1], None]
                        tr = go.Scatter(
                            x=xs, y=ys, mode="lines",
                            name=("Direcciones (líneas)" if not legend_item_lines_done else None),
                            legendgroup="dirs-lines",
                            showlegend=not legend_item_lines_done,
                            line=dict(width=2, color=f"rgba(30,30,30,{float(direction_opacity)})")
                        )
                    all_traces.append(tr); vis_here.append(True)
                    legend_item_lines_done = True

                # Puntos base
                if show_dirs_points:
                    if is_3d:
                        trp = go.Scatter3d(
                            x=Bp[:,0], y=Bp[:,1], z=Bp[:,2], mode="markers",
                            name=("Direcciones (puntos)" if not legend_item_points_done else None),
                            legendgroup="dirs-points",
                            showlegend=not legend_item_points_done,
                            marker=dict(size=3, opacity=float(direction_opacity),
                                        color=f"rgba(10,10,10,{float(direction_opacity)})"),
                            hovertemplate=f"Base {p}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}<br>z:%{{z:.3f}}"
                        )
                    else:
                        trp = go.Scatter(
                            x=Bp[:,0], y=Bp[:,1], mode="markers",
                            name=("Direcciones (puntos)" if not legend_item_points_done else None),
                            legendgroup="dirs-points",
                            showlegend=not legend_item_points_done,
                            marker=dict(size=4, opacity=float(direction_opacity),
                                        color=f"rgba(10,10,10,{float(direction_opacity)})"),
                            hovertemplate=f"Base {p}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}"
                        )
                    all_traces.append(trp); vis_here.append(True)
                    legend_item_points_done = True

        all_vis_masks.append(vis_here)

    # ---------- Layout + menú ----------
    fig = go.Figure(data=all_traces)

    # Todo invisible, luego activar primer bloque
    for tr in fig.data:
        tr.visible = False
    cnt = 0
    for v in all_vis_masks[0]:
        fig.data[cnt].visible = bool(v)
        cnt += 1

    ax_titles = [ _axis_label(i) for i in dims_options[0] ]
    if len(dims_options[0]) == 3:
        fig.update_layout(scene=dict(
            xaxis_title=ax_titles[0],
            yaxis_title=ax_titles[1],
            zaxis_title=ax_titles[2],
            aspectmode="cube"
        ))
    else:
        fig.update_xaxes(title_text=ax_titles[0])
        fig.update_yaxes(title_text=ax_titles[1])

    # Importante: groupclick="togglegroup" para poder ocultar/mostrar TODOS los puntos/líneas de direcciones
    fig.update_layout(title=title, legend=dict(itemsizing="constant", groupclick="togglegroup"))

    if len(dims_options) > 1:
        block_sizes = [len(m) for m in all_vis_masks]
        starts = np.cumsum([0] + block_sizes)
        total = sum(block_sizes)

        def build_mask(i):
            mask = [False]*int(total)
            s, e = int(starts[i]), int(starts[i+1])
            mask[s:e] = all_vis_masks[i]
            return mask

        buttons = []
        for i, dims_opt in enumerate(dims_options):
            labels = [ _axis_label(j) for j in dims_opt ]
            vis_mask = build_mask(i)
            scene_args = ({"scene": dict(
                xaxis_title=labels[0], yaxis_title=labels[1], zaxis_title=labels[2], aspectmode="cube"
            )} if len(dims_opt)==3 else {"xaxis": {"title": labels[0]}, "yaxis": {"title": labels[1]}})
            buttons.append(dict(label=f"ejes {tuple(dims_opt)}", method="update",
                                args=[{"visible": vis_mask}, scene_args]))
        fig.update_layout(updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.12, buttons=buttons)])

    if save_html:
        try:
            fig.write_html(save_html, include_plotlyjs="cdn")
        except Exception as e:
            print(f"[WARN] No se pudo guardar HTML: {e}")

    if show:
        fig.show(config={"scrollZoom": True, "responsive": True})
        return None if not return_fig else fig
    else:
        return fig


def plot_frontiers_implicit_interactive_v3(
    sel: Iterable,
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    planes: Optional[Dict[Tuple[int,int], Dict[str, np.ndarray]]] = None,
    planes_multi: Optional[Dict[Tuple[int,int], List[Dict[str, np.ndarray]]]] = None,
    quadrics: Optional[Dict[Tuple[int,int], Dict[str, np.ndarray]]] = None,
    cubic_models: Optional[Dict[Tuple[int,int], Dict[str, np.ndarray]]] = None,

    dims: Tuple[int, ...] = (0,1,2),
    dims_options: Optional[List[Tuple[int, ...]]] = None,
    feature_names: Optional[List[str]] = None,
    pair_filter: Optional[Iterable[Tuple[int,int]]] = None,

    prefer_cp: bool = True,
    success_only: bool = True,

    show_X: Optional[bool] = None,
    show_frontier: Optional[bool] = None,
    show_planes: Optional[bool] = None,
    show_quadrics: Optional[bool] = None,
    show_cubics: Optional[bool] = None,
    selected_plane_ids: Optional[Iterable[str]] = None,
    plane_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,

    # Direcciones x0 → frontera
    show_directions: str = "lines",   # "none"|"points"|"lines"|"both"
    arrows_per_pair: Optional[int] = None,
    arrow_scale: float = 1.0,
    direction_opacity: float = 0.5,   # <- NUEVO: opacidad parametrizable para líneas y puntos
    plane_opacity: float = 0.08,

    # Detalle
    detail: str = "auto",
    decimate_X: Optional[int] = None,
    decimate_frontier: Optional[int] = None,
    grid_res_2d: Optional[int] = None,
    grid_res_3d: Optional[int] = None,
    quadric_alpha: float = 0.28,
    iso_level: float = 0.0,
    isosurface_epsilon: float = 1e-9,

    # Extensión y slice
    extend: Union[float, Tuple[float, ...]] = 1.0,
    clamp_extend_to_X: bool = True,
    plane_mode: str = "fit_dims",
    slice_filter_tol: Optional[float] = None,
    slice_filter_norm: str = "l1",     # "l1"|"l2"|"linf"

    # Recursos, RNG y salida
    max_voxels: int = 700_000,
    random_state: Optional[int] = 0,
    renderer: Optional[str] = None,
    show: bool = True,
    return_fig: bool = False,
    save_html: Optional[str] = None,
    fig_width: Optional[int] = 1300,
    fig_height: Optional[int] = 900,

    title: str = "DelDel — Interiores, Fronteras y Superficies (v3)"
):
    """
    Versión v3 de plot_frontiers_implicit_interactive_v2.

    Parameters
    ----------
    sel
        Salida de :func:`prune_and_orient_planes_unified_globalmaj(...)`. Puede
        incluir matrices de frontera por par (por ejemplo,
        ``frontier_by_pair``/``frontier_points``); en caso contrario se usa
        ``X``/``y`` como fallback para construir puntos y bases de dirección.
    fig_width
        Ancho (en px) del lienzo. Si es ``None`` no se aplica un ancho fijo.
    fig_height
        Alto (en px) del lienzo. Si es ``None`` no se aplica un alto fijo.

    Notes
    -----
    ``plane_opacity`` controla la transparencia base de los planos. Cuando hay
    múltiples planos por par se mantiene el leve escalonado por índice
    (0.22 + 0.06 * ...), escalando ese patrón sobre ``plane_opacity``.

    ``fig_width`` y ``fig_height`` controlan el tamaño del lienzo (en px). Si
    son ``None``, se dejan sin tocar.
    """
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio

    # ---------- Render target ----------
    if renderer:
        try:
            pio.renderers.default = renderer
        except Exception:
            pass
    else:
        try:
            import google.colab  # noqa
            pio.renderers.default = "colab"
        except Exception:
            pass

    # ---------- Helpers ----------
    def _axis_label(idx: int) -> str:
        if feature_names and 0 <= idx < len(feature_names):
            return feature_names[idx]
        return f"x{idx}"

    def _normalize_pair(pair) -> Optional[Tuple[int, int]]:
        try:
            return (int(pair[0]), int(pair[1]))
        except Exception:
            return None

    def _normalize_plane_meta(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(meta, dict):
            return None
        n = meta.get("n", None) if meta.get("n", None) is not None else meta.get("n_norm")
        b = meta["b"] if "b" in meta else meta.get("b_norm")
        if n is None or b is None:
            return None
        mu = meta.get("mu", None)
        if mu is None:
            mu = meta.get("mu_global", None)
        n_arr = np.asarray(n, float)
        mu_arr = np.asarray(mu, float) if mu is not None else np.zeros_like(n_arr)
        out = dict(meta)
        out["n"] = n_arr
        out["b"] = float(b)
        out["mu"] = mu_arr
        return out

    def _extract_planes_from_sel(sel_value: Any, pair_keys: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
        planes_by_pair: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        if not isinstance(sel_value, dict):
            return planes_by_pair

        def _add(pair: Optional[Tuple[int, int]], meta: Optional[Dict[str, Any]]):
            if pair is None or meta is None:
                return
            planes_by_pair.setdefault(pair, []).append(meta)

        for pair, payload in (sel_value.get("by_pair_augmented") or {}).items():
            norm_pair = _normalize_pair(pair)
            if norm_pair is None:
                continue
            for bucket in ("winning_planes", "other_planes"):
                for plane in (payload.get(bucket) or []):
                    meta = _normalize_plane_meta(plane)
                    if meta is not None:
                        meta.setdefault("origin_pair", norm_pair)
                    _add(norm_pair, meta)

        for key in ("planes_oriented", "planes", "winning_planes", "selection"):
            for plane in (sel_value.get(key) or []):
                meta = _normalize_plane_meta(plane)
                if meta is None:
                    continue
                pair = _normalize_pair(meta.get("origin_pair") or meta.get("pair") or ())
                if pair is None and len(pair_keys) == 1:
                    pair = pair_keys[0]
                    meta.setdefault("origin_pair", pair)
                _add(pair, meta)

        return planes_by_pair


    def _pair_means(full_F: np.ndarray) -> np.ndarray:
        return full_F.mean(axis=0)

    def _g_quadric_eval(P: np.ndarray, Q: np.ndarray, r: np.ndarray, c: float) -> np.ndarray:
        return np.einsum("ni,ij,nj->n", P, Q, P) + P @ r + c

    def _g_cubic_eval(P: np.ndarray, model: dict) -> np.ndarray:
        Z = (P - model["mu"]) / model["sd"]
        n, d = Z.shape
        cols = []
        for i in range(d): cols.append(Z[:,i])
        for i in range(d): cols.append(Z[:,i]**2)
        for i in range(d):
            for j in range(i+1, d):
                cols.append(Z[:,i]*Z[:,j])
        for i in range(d): cols.append(Z[:,i]**3)
        for i in range(d):
            for j in range(d):
                if i==j: continue
                cols.append((Z[:,i]**2)*Z[:,j])
        for i in range(d):
            for j in range(i+1, d):
                for k in range(j+1, d):
                    cols.append(Z[:,i]*Z[:,j]*Z[:,k])
        cols.append(np.ones(n))
        Phi = np.column_stack(cols)
        w = model["w"]
        if w.shape[0] == Phi.shape[1]-1:
            w = np.r_[w, 0.0]
        return Phi @ w

    def _box_from_points(P: np.ndarray, pad_ratio: float = 0.07):
        lo = P.min(axis=0); hi = P.max(axis=0)
        span = hi - lo
        span = np.where(span < 1e-12, 1e-12, span)
        pad = pad_ratio * (span + 1e-12)
        return lo - pad, hi + pad

    def _apply_extend(lo: np.ndarray, hi: np.ndarray, dims_len: int, loX: np.ndarray, hiX: np.ndarray):
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        half = np.where(half < 5e-10, 5e-10, half)
        if isinstance(extend, (tuple, list, np.ndarray)):
            fac = np.asarray(extend, float)
            if fac.size not in (1, dims_len):
                raise ValueError(f"'extend' debe ser escalar o de longitud {dims_len}")
            if fac.size == 1:
                fac = np.repeat(fac, dims_len)
        else:
            fac = np.repeat(float(extend), dims_len)
        new_lo = mid - half * fac
        new_hi = mid + half * fac
        if clamp_extend_to_X:
            new_lo = np.maximum(new_lo, loX)
            new_hi = np.minimum(new_hi, hiX)
            mask = new_lo > new_hi
            if np.any(mask):
                fix = 0.5 * (new_lo[mask] + new_hi[mask])
                new_lo[mask] = fix
                new_hi[mask] = fix
        return new_lo, new_hi

    # --- Cache de grillas ---
    grid2d_cache: Dict[Tuple[Tuple[float,float], Tuple[float,float], int], Tuple[np.ndarray,np.ndarray]] = {}
    grid3d_cache: Dict[Tuple[Tuple[float,float,float], Tuple[float,float,float], Tuple[int,int,int]], Tuple[np.ndarray,np.ndarray,np.ndarray]] = {}

    def _grid_points_2d(lo2: np.ndarray, hi2: np.ndarray, res: int):
        key = (tuple(np.round(lo2,12)), tuple(np.round(hi2,12)), int(res))
        if key in grid2d_cache:
            return grid2d_cache[key]
        xs = np.linspace(lo2[0], hi2[0], res)
        ys = np.linspace(lo2[1], hi2[1], res)
        Xg, Yg = np.meshgrid(xs, ys)
        grid2d_cache[key] = (Xg, Yg)
        return Xg, Yg

    def _grid_points_3d(lo3, hi3, base,
                        *, min_res=8, max_res=96,
                        max_voxels=max_voxels, anisotropy_cap=6.0, eps_span=1e-6):
        lo3 = np.asarray(lo3, float); hi3 = np.asarray(hi3, float)
        span = np.maximum(hi3 - lo3, 0.0)
        deg = span <= eps_span
        nondeg_span = np.where(deg, np.nan, span)
        m = np.nanmean(nondeg_span)
        if not np.isfinite(m) or m <= 0:
            ratios = np.ones(3)
        else:
            ratios = span / m
            ratios = np.clip(ratios, 1.0/anisotropy_cap, anisotropy_cap)
        res = np.rint(base * ratios).astype(int)
        res = np.clip(res, min_res, max_res)
        res[deg] = 2

        def _downscale_to_limit(r, limit):
            r = r.astype(int)
            while int(r[0]) * int(r[1]) * int(r[2]) > limit:
                r = np.maximum(2, np.floor(r * 0.9)).astype(int)
            return r
        res = _downscale_to_limit(res, max_voxels)

        key = (tuple(np.round(lo3,12)), tuple(np.round(hi3,12)), tuple(map(int, res.tolist())))
        if key in grid3d_cache:
            return grid3d_cache[key]

        xs = np.linspace(lo3[0], hi3[0], int(res[0]), dtype=np.float32)
        ys = np.linspace(lo3[1], hi3[1], int(res[1]), dtype=np.float32)
        zs = np.linspace(lo3[2], hi3[2], int(res[2]), dtype=np.float32)
        out = np.meshgrid(xs, ys, zs, indexing="xy")
        grid3d_cache[key] = out
        return out

    def _make_full_points_from_2d_grid(Xg, Yg, template: np.ndarray, dims_opt: Tuple[int,...]):
        n_tot = Xg.size
        P = np.repeat(template.reshape(1,-1), n_tot, axis=0)
        P[:, dims_opt[0]] = Xg.reshape(-1)
        P[:, dims_opt[1]] = Yg.reshape(-1)
        return P

    def _make_full_points_from_3d_grid(Xg, Yg, Zg, template: np.ndarray, dims_opt: Tuple[int,...]):
        n_tot = Xg.size
        P = np.repeat(template.reshape(1,-1), n_tot, axis=0)
        P[:, dims_opt[0]] = Xg.reshape(-1)
        P[:, dims_opt[1]] = Yg.reshape(-1)
        P[:, dims_opt[2]] = Zg.reshape(-1)
        return P

    def _frontier_weights_uniform(frontier_by_pair: Dict[Tuple[int, int], np.ndarray]) -> Dict[Tuple[int, int], np.ndarray]:
        weights: Dict[Tuple[int, int], np.ndarray] = {}
        for pair, pts in frontier_by_pair.items():
            n = int(pts.shape[0])
            weights[pair] = np.ones(n, float) / max(1, n)
        return weights

    def _pairs_from_sel(sel_value: Any) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        if isinstance(sel_value, dict):
            for pair in (sel_value.get("by_pair_augmented") or {}).keys():
                norm_pair = _normalize_pair(pair)
                if norm_pair is not None:
                    pairs.append(norm_pair)
            for key in ("winning_planes", "planes", "selection"):
                for plane in (sel_value.get(key) or []):
                    if not isinstance(plane, dict):
                        continue
                    norm_pair = _normalize_pair(plane.get("origin_pair") or plane.get("pair") or ())
                    if norm_pair is not None:
                        pairs.append(norm_pair)
            for key in ("frontier_by_pair", "frontiers_by_pair", "frontier_points_by_pair", "frontier_points", "frontier"):
                block = sel_value.get(key)
                if isinstance(block, dict):
                    for pair in block.keys():
                        norm_pair = _normalize_pair(pair)
                        if norm_pair is not None:
                            pairs.append(norm_pair)
        elif isinstance(sel_value, Iterable):
            for plane in sel_value:
                if isinstance(plane, dict):
                    norm_pair = _normalize_pair(plane.get("origin_pair") or plane.get("pair") or ())
                    if norm_pair is not None:
                        pairs.append(norm_pair)
        out = []
        seen = set()
        for pair in pairs:
            if pair not in seen:
                out.append(pair)
                seen.add(pair)
        return out

    def _extract_frontier_from_sel(sel_value: Any) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], np.ndarray]]:
        frontier_by_pair: Dict[Tuple[int, int], np.ndarray] = {}
        bases_by_pair: Dict[Tuple[int, int], np.ndarray] = {}
        if not isinstance(sel_value, dict):
            return frontier_by_pair, bases_by_pair

        for key in ("frontier_by_pair", "frontiers_by_pair", "frontier_points_by_pair", "frontier_points", "frontier"):
            block = sel_value.get(key)
            if not isinstance(block, dict):
                continue
            for pair, payload in block.items():
                norm_pair = _normalize_pair(pair)
                if norm_pair is None:
                    continue
                arr = np.asarray(payload, float)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.size == 0:
                    continue
                frontier_by_pair[norm_pair] = arr

        for pair, payload in (sel_value.get("by_pair_augmented") or {}).items():
            if not isinstance(payload, dict):
                continue
            norm_pair = _normalize_pair(pair)
            if norm_pair is None:
                continue
            for key in ("frontier_points", "frontier", "frontier_by_pair"):
                block = payload.get(key)
                if block is None:
                    continue
                arr = np.asarray(block, float)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.size == 0:
                    continue
                frontier_by_pair[norm_pair] = arr

        for key in ("bases_by_pair", "base_points_by_pair", "bases", "base_points"):
            block = sel_value.get(key)
            if not isinstance(block, dict):
                continue
            for pair, payload in block.items():
                norm_pair = _normalize_pair(pair)
                if norm_pair is None:
                    continue
                arr = np.asarray(payload, float)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.size == 0:
                    continue
                bases_by_pair[norm_pair] = arr

        return frontier_by_pair, bases_by_pair

    def _fallback_frontier_from_xy(pair_keys: Sequence[Tuple[int, int]]) -> Dict[Tuple[int, int], np.ndarray]:
        fallback: Dict[Tuple[int, int], np.ndarray] = {}
        for a, b in pair_keys:
            Xa = X[y == int(a)]
            Xb = X[y == int(b)]
            if Xa.size == 0 or Xb.size == 0:
                continue
            fallback[(int(a), int(b))] = np.vstack([Xa, Xb])
        return fallback

    def _bases_from_xy_for_frontier(
        frontier_by_pair: Dict[Tuple[int, int], np.ndarray],
        pair_keys: Sequence[Tuple[int, int]],
        rng_local: np.random.RandomState
    ) -> Dict[Tuple[int, int], np.ndarray]:
        bases: Dict[Tuple[int, int], np.ndarray] = {}
        for a, b in pair_keys:
            pair = (int(a), int(b))
            F = frontier_by_pair.get(pair)
            if F is None or F.size == 0:
                continue
            Xa = X[y == int(a)]
            if Xa.size == 0:
                continue
            m = int(F.shape[0])
            replace = Xa.shape[0] < m
            idx = rng_local.choice(Xa.shape[0], size=m, replace=replace)
            bases[pair] = Xa[idx]
        return bases

    def _fit_tls_plane_in_dims(F_full: np.ndarray, w: np.ndarray, dims_sel: Tuple[int,...]):
        P = F_full[:, dims_sel]
        w = (w / (w.sum()+1e-12)).reshape(-1)
        mu = (w[:,None] * P).sum(axis=0)
        Z  = P - mu
        Sw = Z.T @ (w[:,None] * Z)
        evals, evecs = np.linalg.eigh(Sw)
        n_sub = evecs[:, np.argmin(evals)]
        b_eff = -float(n_sub @ mu)
        return n_sub.astype(float), float(b_eff)

    def _restrict_plane_to_dims(n: np.ndarray, b: float, mu: np.ndarray, dims_sel: Tuple[int, ...]):
        d = n.size
        dims_sel = tuple(int(i) for i in dims_sel)
        other = [j for j in range(d) if j not in dims_sel]
        b_eff = b + float(np.dot(n[other], mu[other])) if other else b
        n_sub = n[list(dims_sel)]
        return n_sub.astype(float), float(b_eff)

    def _plane_surface_mesh_3d(n_sub: np.ndarray, b_eff: float, lo: np.ndarray, hi: np.ndarray, res: int = 14):
        n_sub = np.asarray(n_sub, float).reshape(3)
        k = int(np.argmax(np.abs(n_sub)))
        free = [i for i in (0,1,2) if i != k]
        ax0, ax1 = free[0], free[1]
        g0 = np.linspace(lo[ax0], hi[ax0], res)
        g1 = np.linspace(lo[ax1], hi[ax1], res)
        G0, G1 = np.meshgrid(g0, g1)
        denom = n_sub[k] if abs(n_sub[k]) > 1e-12 else 1e-12
        Gk = -(n_sub[ax0]*G0 + n_sub[ax1]*G1 + b_eff) / denom
        grids = {ax0: G0, ax1: G1, k: Gk}
        Xp, Yp, Zp = grids[0], grids[1], grids[2]
        return Xp, Yp, Zp

    # ---------- Validaciones dims ----------
    X = np.asarray(X, float)
    y = np.asarray(y, int).reshape(-1) if y is not None else None
    d_total = X.shape[1]
    assert len(dims) in (2,3), "dims debe tener longitud 2 o 3"
    dims = tuple(int(i) for i in dims)
    for i in dims:
        assert 0 <= i < d_total, f"Índice dims fuera de rango: {i}"

    rng = np.random.RandomState(None if random_state is None else int(random_state))

    # ---------- Fronteras desde sel (o fallback X/y) ----------
    pair_keys = _pairs_from_sel(sel)
    if (not pair_keys) and (y is not None):
        classes = sorted(np.unique(y))
        pair_keys = [(int(a), int(b)) for a, b in itertools.combinations(classes, 2)]
    if pair_filter:
        pf = set(map(tuple, pair_filter))
        pair_keys = [p for p in pair_keys if p in pf]
        if not pair_keys:
            raise ValueError("pair_filter dejó 0 pares para graficar.")

    frontier_by_pair, bases_by_pair = _extract_frontier_from_sel(sel)
    if frontier_by_pair:
        missing_pairs = [p for p in pair_keys if p not in frontier_by_pair]
        if missing_pairs and y is not None:
            frontier_by_pair.update(_fallback_frontier_from_xy(missing_pairs))
    elif y is not None:
        frontier_by_pair = _fallback_frontier_from_xy(pair_keys)

    if not frontier_by_pair:
        raise ValueError("No hay puntos de frontera para graficar.")

    pair_keys = sorted(frontier_by_pair.keys()) if frontier_by_pair else pair_keys
    if pair_filter:
        pf = set(map(tuple, pair_filter))
        pair_keys = [p for p in pair_keys if p in pf]

    if not bases_by_pair and y is not None:
        bases_by_pair = _bases_from_xy_for_frontier(frontier_by_pair, pair_keys, rng)

    if plane_mode == "fit_dims":
        F_for_fit = {pair: frontier_by_pair[pair] for pair in pair_keys if pair in frontier_by_pair}
        W_for_fit = _frontier_weights_uniform(F_for_fit)
    else:
        F_for_fit, W_for_fit = {}, {}

    def _coerce_planes_multi(planes, planes_multi):
        if planes_multi is not None:
            return {k: list(v) for k, v in planes_multi.items() if isinstance(v, (list, tuple))}
        out = {}
        if planes is None:
            return out
        for k, meta in planes.items():
            out[k] = list(meta) if isinstance(meta, (list, tuple)) else [meta]
        return out

    planes_list_by_pair = _coerce_planes_multi(planes, planes_multi)
    planes_from_sel = _extract_planes_from_sel(sel, pair_keys)
    if planes_from_sel:
        if not planes_list_by_pair:
            planes_list_by_pair = planes_from_sel
        else:
            for pair, plist in planes_from_sel.items():
                planes_list_by_pair.setdefault(pair, []).extend(plist)


    pair_templates = {p: _pair_means(frontier_by_pair[p]) for p in pair_keys}

    # ---------- Colores por clase ----------
    import plotly.express as px
    class_colors = {}
    if (y is not None) and (len(y) == X.shape[0]):
        classes = np.unique(y)
        pal = px.colors.qualitative.Plotly
        for k, c in enumerate(classes):
            class_colors[int(c)] = pal[k % len(pal)]
    else:
        classes = np.array([0])
        class_colors[0] = "rgba(130,130,130,0.70)"
        y = np.zeros(X.shape[0], dtype=int)

    # ---------- dims_options ----------
    if dims_options is None:
        dims_options = [dims]
    else:
        dims_options = [tuple(map(int, opt)) for opt in dims_options if len(opt) in (2,3)]
    seen = set(); _tmp = []
    for op in dims_options:
        if op not in seen:
            _tmp.append(op); seen.add(op)
    dims_options = _tmp

    # ---------- presets de 'detail' ----------
    def _resolve_detail(detail: str, nX: int, n_pairs: int, n_frontier_tot: int):
        if detail not in {"auto","fast","balanced","high"}:
            detail = "balanced"
        if detail == "auto":
            load = nX + n_frontier_tot + 2000*n_pairs
            detail = "fast" if load > 50_000 else "balanced"
        if detail == "fast":
            return dict(grid2d=120, grid3d=16, decX=4, decF=4, arrows=20)
        if detail == "balanced":
            return dict(grid2d=200, grid3d=28, decX=3, decF=3, arrows=40)
        return dict(grid2d=300, grid3d=40, decX=2, decF=2, arrows=60)

    n_frontier_tot = sum(frontier_by_pair[p].shape[0] for p in pair_keys)
    preset = _resolve_detail(detail, X.shape[0], len(pair_keys), n_frontier_tot)

    if grid_res_2d is None: grid_res_2d = preset["grid2d"]
    if grid_res_3d is None: grid_res_3d = preset["grid3d"]
    if decimate_X is None: decimate_X = preset["decX"]
    if decimate_frontier is None: decimate_frontier = preset["decF"]
    if arrows_per_pair is None: arrows_per_pair = preset["arrows"]

    if show_X is None: show_X = True
    if show_frontier is None: show_frontier = True
    if show_planes is None:
        show_planes = True
    if show_quadrics is None: show_quadrics = bool(quadrics)
    if show_cubics is None: show_cubics = bool(cubic_models)

    show_dirs_mode = str(show_directions).lower()
    show_dirs_points = show_dirs_mode in ("points","both")
    show_dirs_lines  = show_dirs_mode in ("lines","both")

    selected_plane_ids_set = set(selected_plane_ids) if selected_plane_ids else None

    legend_pair_titles_done: set = set()

    def _legend_pair_title(pair: Tuple[int, int]) -> Optional[str]:
        if pair in legend_pair_titles_done:
            return None
        legend_pair_titles_done.add(pair)
        return f"Par ({pair[0]}, {pair[1]})"

    def _plane_visible(meta: Dict[str, Any]) -> bool:
        visible = True
        if selected_plane_ids_set is not None:
            pid = meta.get("plane_id")
            oid = meta.get("oriented_plane_id")
            visible = (pid in selected_plane_ids_set) or (oid in selected_plane_ids_set)
        if plane_filter is not None:
            try:
                visible = visible and bool(plane_filter(meta))
            except Exception:
                visible = False
        return visible

    # Validadores de modelos
    def _parse_quadric(mdl):
        try:
            Q = np.asarray(mdl["Q"], float); r = np.asarray(mdl["r"], float); c = float(mdl["c"])
            if Q.shape != (d_total, d_total) or r.shape != (d_total,):
                print(f"[WARN] Cuádrica omitida por forma incompatible: Q{Q.shape}, r{r.shape}, d={d_total}")
                return None
            return Q, r, c
        except Exception as e:
            print(f"[WARN] Cuádrica inválida: {e}")
            return None

    def _parse_cubic(mdl):
        try:
            mu = np.asarray(mdl["mu"], float); sd = np.asarray(mdl["sd"], float); w = np.asarray(mdl["w"], float)
            if mu.shape != (d_total,) or sd.shape != (d_total,):
                print(f"[WARN] Cúbica omitida por forma incompatible: mu{mu.shape}, sd{sd.shape}, d={d_total}")
                return None
            return {"mu":mu, "sd":sd, "w":w}
        except Exception as e:
            print(f"[WARN] Cúbica inválida: {e}")
            return None

    # ---------- Construcción de trazas ----------
    all_traces = []
    all_vis_masks = []

    def _stratified_indices(y, step):
        if step is None or step < 2:
            return np.arange(y.size)
        sel = []
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            sel.append(idx[::step])
        return np.concatenate(sel) if sel else np.arange(y.size)

    # Flags para mostrar un único ítem de leyenda por grupo
    legend_item_lines_done = False
    legend_item_points_done = False

    for opt_i, dims_opt in enumerate(dims_options):
        is_3d = (len(dims_opt) == 3)
        vis_here = []
        legend_planes_done = set()

        Xopt = X[:, dims_opt]
        loX = Xopt.min(axis=0)
        hiX = Xopt.max(axis=0)

        # ---- X por clase ----
        if show_X:
            idx_all = _stratified_indices(y, decimate_X)
            for c in np.unique(y[idx_all]):
                sel = idx_all[y[idx_all] == c]
                if sel.size == 0: continue
                P = X[sel][:, dims_opt]
                if is_3d:
                    tr = go.Scatter3d(
                        x=P[:,0], y=P[:,1], z=P[:,2], mode="markers",
                        name=f"Clase {c} (X)", legendgroup=f"class-{c}",
                        marker=dict(size=3, opacity=0.55, color=class_colors[int(c)]),
                        hovertemplate=f"Clase {c}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}<br>z:%{{z:.3f}}"
                    )
                else:
                    tr = go.Scatter(
                        x=P[:,0], y=P[:,1], mode="markers",
                        name=f"Clase {c} (X)", legendgroup=f"class-{c}",
                        marker=dict(size=5, opacity=0.55, color=class_colors[int(c)]),
                        hovertemplate=f"Clase {c}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}"
                    )
                all_traces.append(tr); vis_here.append(True)

        # ---- Fronteras (puntos) ----
        if show_frontier:
            for p in pair_keys:
                F_full = frontier_by_pair[p]
                F_use = F_full
                if plane_mode == "slice" and slice_filter_tol is not None and (planes_list_by_pair.get(p) or (planes and p in planes)):
                    metas = planes_list_by_pair.get(p, [planes[p]] if (planes and p in planes) else [])
                    other = [j for j in range(d_total) if j not in dims_opt]
                    if other and metas:
                        mu_other = np.mean([np.asarray(m["mu"], float)[other] for m in metas], axis=0)
                        diff = np.abs(F_full[:, other] - mu_other)
                        if slice_filter_norm == "l2":
                            dist = np.sqrt((diff**2).sum(axis=1)) / (diff.shape[1]**0.5 if diff.shape[1] else 1.0)
                        elif slice_filter_norm == "linf":
                            dist = diff.max(axis=1)
                        else:  # "l1"
                            dist = diff.mean(axis=1)
                        mask = (dist <= float(slice_filter_tol))
                        if np.any(mask): F_use = F_full[mask]
                if F_use.size == 0: continue
                if isinstance(decimate_frontier, int) and decimate_frontier >= 2:
                    F_use = F_use[::decimate_frontier]
                Qp = F_use[:, dims_opt]
                col = "rgba(30,30,30,0.85)"; outline = "rgba(0,0,0,0.6)"
                legend_title = _legend_pair_title(p)
                if is_3d:
                    tr = go.Scatter3d(
                        x=Qp[:,0], y=Qp[:,1], z=Qp[:,2], mode="markers",
                        name=f"Frontera ({p[0]}, {p[1]})", legendgroup=f"pair-{p}",
                        legendgrouptitle_text=legend_title,
                        marker=dict(size=4, color=col, line=dict(width=1, color=outline)),
                        hovertemplate=f"{p}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}<br>z:%{{z:.3f}}"
                    )
                else:
                    tr = go.Scatter(
                        x=Qp[:,0], y=Qp[:,1], mode="markers",
                        name=f"Frontera ({p[0]}, {p[1]})", legendgroup=f"pair-{p}",
                        legendgrouptitle_text=legend_title,
                        marker=dict(size=6, color=col, line=dict(width=1, color=outline)),
                        hovertemplate=f"{p}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}"
                    )
                all_traces.append(tr); vis_here.append(True)

        # ---- Planos ----
        if show_planes:
            for p in pair_keys:
                Fp = frontier_by_pair[p][:, dims_opt]
                if Fp.size == 0:
                    continue
                lo, hi = _box_from_points(Fp, pad_ratio=0.05)
                lo, hi = _apply_extend(lo, hi, dims_len=len(dims_opt), loX=loX, hiX=hiX)

                planes_for_p = planes_list_by_pair.get(p, None)
                if planes_for_p:
                    for idx_pl, meta in enumerate(planes_for_p, start=1):
                        n = np.asarray(meta["n"], float); b0 = float(meta["b"]); mu = np.asarray(meta["mu"], float)
                        n_sub, b_eff = _restrict_plane_to_dims(n, b0, mu, dims_opt)
                        plane_visible = _plane_visible(meta)
                        visible_value = True if plane_visible else "legendonly"
                        plane_group = f"plane-{p}"
                        show_legend = p not in legend_planes_done
                        legend_name = (f"Planos ({p[0]}, {p[1]})" if show_legend
                                       else f"Plano ({p[0]}, {p[1]}) #{idx_pl}")
                        if is_3d:
                            Xp_s, Yp_s, Zp_s = _plane_surface_mesh_3d(n_sub, b_eff, lo, hi, res=14)
                            op = 0.22 + 0.06*((idx_pl-1) % 3)
                            col = "rgba(55,55,55,0.95)"
                            tr = go.Surface(
                                x=Xp_s, y=Yp_s, z=Zp_s, name=legend_name,
                                legendgroup=plane_group, showlegend=show_legend,
                                showscale=False, opacity=op, visible=visible_value,
                                colorscale=[[0, col],[1, col]]
                            )
                        else:
                            xs = np.linspace(lo[0], hi[0], 300)
                            if abs(n_sub[1]) > 1e-12:
                                ys = -(n_sub[0]*xs + b_eff) / n_sub[1]
                                tr = go.Scatter(x=xs, y=ys, mode="lines",
                                                name=legend_name, legendgroup=plane_group, showlegend=show_legend,
                                                line=dict(width=2, color="rgba(50,50,50,0.9)"), visible=visible_value)
                            else:
                                x0p = -b_eff / (n_sub[0] if abs(n_sub[0])>1e-12 else 1e-12)
                                tr = go.Scatter(x=[x0p,x0p], y=[lo[1],hi[1]], mode="lines",
                                                name=legend_name, legendgroup=plane_group, showlegend=show_legend,
                                                line=dict(width=2, color="rgba(50,50,50,0.9)"), visible=visible_value)
                        all_traces.append(tr); vis_here.append(visible_value)
                        legend_planes_done.add(p)
                else:
                    plane_meta = None
                    idx_pl = 1
                    if plane_mode == "fit_dims":
                        F_full = F_for_fit.get(p, frontier_by_pair[p])
                        w = W_for_fit.get(p, np.ones(F_full.shape[0], float)/max(1, F_full.shape[0]))
                        n_sub, b_eff = _fit_tls_plane_in_dims(F_full, w, dims_opt)
                    else:
                        if not planes or p not in planes:
                            continue
                        plane_meta = planes[p]
                        n = np.asarray(plane_meta["n"], float)
                        b0 = float(plane_meta["b"])
                        mu = np.asarray(plane_meta["mu"], float)
                        n_sub, b_eff = _restrict_plane_to_dims(n, b0, mu, dims_opt)

                    plane_visible = (_plane_visible(plane_meta) if plane_meta is not None else (selected_plane_ids_set is None and plane_filter is None))
                    visible_value = True if plane_visible else "legendonly"
                    plane_group = f"plane-{p}"
                    show_legend = p not in legend_planes_done
                    legend_name = (f"Planos ({p[0]}, {p[1]})" if show_legend
                                   else f"Plano ({p[0]}, {p[1]}) #{idx_pl}")
                    if is_3d:
                        Xp_s, Yp_s, Zp_s = _plane_surface_mesh_3d(n_sub, b_eff, lo, hi, res=14)
                        tr = go.Surface(
                            x=Xp_s, y=Yp_s, z=Zp_s, name=legend_name,
                            legendgroup=plane_group, showlegend=show_legend,
                            showscale=False, opacity=0.25, visible=visible_value,
                            colorscale=[[0, "rgba(50,50,50,0.9)"], [1, "rgba(50,50,50,0.9)"]]
                        )
                    else:
                        xs = np.linspace(lo[0], hi[0], 300)
                        if abs(n_sub[1]) > 1e-12:
                            ys = -(n_sub[0]*xs + b_eff) / n_sub[1]
                            tr = go.Scatter(x=xs, y=ys, mode="lines",
                                            name=legend_name, legendgroup=plane_group, showlegend=show_legend,
                                            line=dict(width=2, color="rgba(50,50,50,0.9)"), visible=visible_value)
                        else:
                            x0p = -b_eff / (n_sub[0] if abs(n_sub[0])>1e-12 else 1e-12)
                            tr = go.Scatter(x=[x0p,x0p], y=[lo[1],hi[1]], mode="lines",
                                            name=legend_name, legendgroup=plane_group, showlegend=show_legend,
                                            line=dict(width=2, color="rgba(50,50,50,0.9)"), visible=visible_value)
                    all_traces.append(tr); vis_here.append(visible_value)
                    legend_planes_done.add(p)

        # ---- Cuádricas ----
        if show_quadrics and quadrics:
            for p in pair_keys:
                mdl = quadrics.get(p)
                if mdl is None: continue
                parsed = _parse_quadric(mdl)
                if parsed is None:
                    continue
                Q, r, cst = parsed
                Fp = frontier_by_pair[p][:, dims_opt]
                lo, hi = _box_from_points(Fp, pad_ratio=0.06)
                lo, hi = _apply_extend(lo, hi, dims_len=len(dims_opt), loX=loX, hiX=hiX)
                template = pair_templates[p]
                legend_title = _legend_pair_title(p)
                if is_3d:
                    Xg, Yg, Zg = _grid_points_3d(lo, hi, grid_res_3d)
                    Pfull = _make_full_points_from_3d_grid(Xg, Yg, Zg, template, dims_opt)
                    G = _g_quadric_eval(Pfull, Q, r, cst).reshape(Xg.shape)
                    tr = go.Isosurface(
                        x=Xg.flatten(), y=Yg.flatten(), z=Zg.flatten(),
                        value=G.flatten(),
                        isomin=iso_level - isosurface_epsilon, isomax=iso_level + isosurface_epsilon, surface_count=1,
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        showscale=False, opacity=quadric_alpha,
                        name=f"Cuádrica {p}", legendgroup=f"pair-{p}",
                        legendgrouptitle_text=legend_title,
                        colorscale=[[0,"#444"],[1,"#444"]]
                    )
                else:
                    Xg, Yg = _grid_points_2d(lo, hi, grid_res_2d)
                    Pfull = _make_full_points_from_2d_grid(Xg, Yg, template, dims_opt)
                    G = _g_quadric_eval(Pfull, Q, r, cst).reshape(Xg.shape)
                    tr = go.Contour(
                        x=Xg[0,:], y=Yg[:,0], z=G,
                        contours=dict(start=iso_level, end=iso_level, size=1.0),
                        showscale=False, name=f"Cuádrica {p}", legendgroup=f"pair-{p}",
                        legendgrouptitle_text=legend_title,
                        line=dict(width=3, color="rgba(20,20,20,0.95)")
                    )
                all_traces.append(tr); vis_here.append(True)

        # ---- Cúbicas ----
        if show_cubics and cubic_models:
            for p in pair_keys:
                mdl = cubic_models.get(p)
                if mdl is None: continue
                mdlp = _parse_cubic(mdl)
                if mdlp is None:
                    continue
                Fp = frontier_by_pair[p][:, dims_opt]
                lo, hi = _box_from_points(Fp, pad_ratio=0.06)
                lo, hi = _apply_extend(lo, hi, dims_len=len(dims_opt), loX=loX, hiX=hiX)
                template = pair_templates[p]
                legend_title = _legend_pair_title(p)
                if is_3d:
                    Xg, Yg, Zg = _grid_points_3d(lo, hi, max(16, grid_res_3d//2))
                    Pfull = _make_full_points_from_3d_grid(Xg, Yg, Zg, template, dims_opt)
                    G = _g_cubic_eval(Pfull, mdlp).reshape(Xg.shape)
                    tr = go.Isosurface(
                        x=Xg.flatten(), y=Yg.flatten(), z=Zg.flatten(),
                        value=G.flatten(),
                        isomin=iso_level - isosurface_epsilon, isomax=iso_level + isosurface_epsilon, surface_count=1,
                        caps=dict(x_show=False, y_show=False, z_show=False),
                        showscale=False, opacity=quadric_alpha,
                        name=f"Cúbica {p}", legendgroup=f"pair-{p}",
                        legendgrouptitle_text=legend_title,
                        colorscale=[[0,"#777"],[1,"#777"]]
                    )
                else:
                    Xg, Yg = _grid_points_2d(lo, hi, max(100, grid_res_2d//2))
                    Pfull = _make_full_points_from_2d_grid(Xg, Yg, template, dims_opt)
                    G = _g_cubic_eval(Pfull, mdlp).reshape(Xg.shape)
                    tr = go.Contour(
                        x=Xg[0,:], y=Yg[:,0], z=G,
                        contours=dict(start=iso_level, end=iso_level, size=1.0),
                        showscale=False, name=f"Cúbica {p}", legendgroup=f"pair-{p}",
                        legendgrouptitle_text=legend_title,
                        line=dict(width=2, color="rgba(80,80,80,0.95)")
                    )
                all_traces.append(tr); vis_here.append(True)

        # ---- Direcciones (con opacidad parametrizable + leyenda agrupada) ----
        if (show_dirs_points or show_dirs_lines):
            for p in pair_keys:
                F = frontier_by_pair[p]; B = bases_by_pair.get(p)
                if F.size == 0 or B is None or B.size == 0:
                    continue
                m = F.shape[0]; k = min(int(arrows_per_pair), m)
                if k <= 0:
                    continue
                idx = rng.choice(m, size=k, replace=False)
                Fp = F[idx][:, dims_opt]; Bp = B[idx][:, dims_opt]
                U  = (Fp - Bp) * float(arrow_scale)

                # Líneas
                if show_dirs_lines:
                    if is_3d:
                        xs, ys, zs = [], [], []
                        for i in range(k):
                            xs += [Bp[i,0], Bp[i,0]+U[i,0], None]
                            ys += [Bp[i,1], Bp[i,1]+U[i,1], None]
                            zs += [Bp[i,2], Bp[i,2]+U[i,2], None]
                        tr = go.Scatter3d(
                            x=xs, y=ys, z=zs, mode="lines",
                            name=("Direcciones (líneas)" if not legend_item_lines_done else None),
                            legendgroup="dirs-lines",
                            showlegend=not legend_item_lines_done,
                            line=dict(width=2, color=f"rgba(30,30,30,{float(direction_opacity)})")
                        )
                    else:
                        xs, ys = [], []
                        for i in range(k):
                            xs += [Bp[i,0], Bp[i,0]+U[i,0], None]
                            ys += [Bp[i,1], Bp[i,1]+U[i,1], None]
                        tr = go.Scatter(
                            x=xs, y=ys, mode="lines",
                            name=("Direcciones (líneas)" if not legend_item_lines_done else None),
                            legendgroup="dirs-lines",
                            showlegend=not legend_item_lines_done,
                            line=dict(width=2, color=f"rgba(30,30,30,{float(direction_opacity)})")
                        )
                    all_traces.append(tr); vis_here.append(True)
                    legend_item_lines_done = True

                # Puntos base
                if show_dirs_points:
                    if is_3d:
                        trp = go.Scatter3d(
                            x=Bp[:,0], y=Bp[:,1], z=Bp[:,2], mode="markers",
                            name=("Direcciones (puntos)" if not legend_item_points_done else None),
                            legendgroup="dirs-points",
                            showlegend=not legend_item_points_done,
                            marker=dict(size=3, opacity=float(direction_opacity),
                                        color=f"rgba(10,10,10,{float(direction_opacity)})"),
                            hovertemplate=f"Base {p}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}<br>z:%{{z:.3f}}"
                        )
                    else:
                        trp = go.Scatter(
                            x=Bp[:,0], y=Bp[:,1], mode="markers",
                            name=("Direcciones (puntos)" if not legend_item_points_done else None),
                            legendgroup="dirs-points",
                            showlegend=not legend_item_points_done,
                            marker=dict(size=4, opacity=float(direction_opacity),
                                        color=f"rgba(10,10,10,{float(direction_opacity)})"),
                            hovertemplate=f"Base {p}<br>x:%{{x:.3f}}<br>y:%{{y:.3f}}"
                        )
                    all_traces.append(trp); vis_here.append(True)
                    legend_item_points_done = True

        all_vis_masks.append(vis_here)

    # ---------- Layout + menú ----------
    fig = go.Figure(data=all_traces)

    # Todo invisible, luego activar primer bloque
    for tr in fig.data:
        tr.visible = False
    cnt = 0
    for v in all_vis_masks[0]:
        fig.data[cnt].visible = v
        cnt += 1

    ax_titles = [ _axis_label(i) for i in dims_options[0] ]
    if len(dims_options[0]) == 3:
        fig.update_layout(scene=dict(
            xaxis_title=ax_titles[0],
            yaxis_title=ax_titles[1],
            zaxis_title=ax_titles[2],
            aspectmode="cube"
        ))
    else:
        fig.update_xaxes(title_text=ax_titles[0])
        fig.update_yaxes(title_text=ax_titles[1])

    # Importante: groupclick="togglegroup" para poder ocultar/mostrar TODOS los puntos/líneas de direcciones
    fig.update_layout(title=title, legend=dict(itemsizing="constant", groupclick="togglegroup"))
    if fig_width is not None or fig_height is not None:
        fig.update_layout(width=fig_width, height=fig_height)

    if len(dims_options) > 1:
        block_sizes = [len(m) for m in all_vis_masks]
        starts = np.cumsum([0] + block_sizes)
        total = sum(block_sizes)

        def build_mask(i):
            mask = [False]*int(total)
            s, e = int(starts[i]), int(starts[i+1])
            mask[s:e] = all_vis_masks[i]
            return mask

        buttons = []
        for i, dims_opt in enumerate(dims_options):
            labels = [ _axis_label(j) for j in dims_opt ]
            vis_mask = build_mask(i)
            scene_args = ({"scene": dict(
                xaxis_title=labels[0], yaxis_title=labels[1], zaxis_title=labels[2], aspectmode="cube"
            )} if len(dims_opt)==3 else {"xaxis": {"title": labels[0]}, "yaxis": {"title": labels[1]}})
            buttons.append(dict(label=f"ejes {tuple(dims_opt)}", method="update",
                                args=[{"visible": vis_mask}, scene_args]))
        fig.update_layout(updatemenus=[dict(type="buttons", direction="right", x=0.0, y=1.12, buttons=buttons)])

    if save_html:
        try:
            fig.write_html(save_html, include_plotlyjs="cdn")
        except Exception as e:
            print(f"[WARN] No se pudo guardar HTML: {e}")

    if show:
        fig.show(config={"scrollZoom": True, "responsive": True})
        return None if not return_fig else fig
    else:
        return fig
