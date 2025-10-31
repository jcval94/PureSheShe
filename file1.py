from __future__ import annotations

# === Instrumentation utils: model call logging ===========================================
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple
import time
from contextlib import contextmanager
import inspect
import numpy as _np
import copy

_DELD_STAGE = ContextVar("DELD_STAGE", default=None)
_DELD_CALL_LOGGER = ContextVar("DELD_CALL_LOGGER", default=None)

@dataclass
class ModelCall:
    ts: float
    stage: Optional[str]
    source: str
    fn: str
    batch: int
    n_features: int
    cache_enabled: bool
    cache_hits: int
    cache_misses: int
    duration_ms: float
    extra: Dict[str, Any] = field(default_factory=dict)
    input: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)

def _log_model_call(entry: Dict[str, Any]) -> None:
    cb = _DELD_CALL_LOGGER.get()
    if cb is not None:
        cb(entry)

def _infer_stage() -> str:
    try:
        for fr in inspect.stack():
            func = fr.function
            if func in {"scores", "_scores_raw", "_predict_labels"}:
                continue
            loc = fr.frame.f_locals
            if "self" in loc:
                try: clsname = loc["self"].__class__.__name__
                except Exception: clsname = None
                if clsname == "DelDel":
                    return f"{clsname}.{func}"
        top = inspect.stack()[1]
        return f"{top.frame.f_globals.get('__name__','__main__')}.{top.function}"
    except Exception:
        return "unknown"

@contextmanager
def _collect_calls(target_list: List[Dict[str, Any]]):
    def _append_call(e: Dict[str, Any]): target_list.append(e)
    _tok = _DELD_CALL_LOGGER.set(_append_call)
    try:
        yield
    finally:
        _DELD_CALL_LOGGER.reset(_tok)

def _snapshot_array(arr, max_rows: int = 2, max_cols: int = 8, max_elems: int = 2000):
    try:
        a = _np.asarray(arr)
        shape = tuple(a.shape)
        if a.size > max_elems: return {"shape": shape, "note": "omitted_large"}
        dtype = str(a.dtype)
        a2 = a.reshape(1, -1) if a.ndim == 1 else a
        r = min(max_rows, a2.shape[0] if a2.ndim >= 1 else 1)
        c = min(max_cols, a2.shape[1] if a2.ndim >= 2 else (a2.size if a2.ndim==1 else 1))
        sample = a2[:r, :c] if a2.ndim >= 2 else a2[:r]
        return {"shape": shape, "dtype": dtype, "sample": sample.tolist()}
    except Exception as e:
        return {"error": str(e)}
# =========================================================================================
import logging, time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Iterable, NamedTuple
import numpy as np
from collections import defaultdict

# -------------------------
# Logging
# -------------------------
def _get_logger(name: str = "deldel", level: int = logging.INFO) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        lg.addHandler(h)
    lg.setLevel(level)
    return lg

# -------------------------
# Utils
# -------------------------
_EPS = 1e-12
def _softmax(z, axis=1):
    z = np.asarray(z, float); z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z); return ez / (np.sum(ez, axis=axis, keepdims=True) + _EPS)
def _sigmoid(z): z = np.asarray(z, float); return 1.0 / (1.0 + np.exp(-z))

def _jsd(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence en [0,1] usando log base 2."""
    P = np.clip(np.asarray(P, float), eps, 1.0); P /= P.sum()
    Q = np.clip(np.asarray(Q, float), eps, 1.0); Q /= Q.sum()
    M = 0.5*(P+Q)
    def _kl(A,B): return float(np.sum(A * (np.log2(A) - np.log2(B))))
    return 0.5*_kl(P, M) + 0.5*_kl(Q, M)

def _midpoint_flip_gate_simple(model, x0, x1, y0, y1, iters=3):
    """
    Devuelve (ok, t*) si encuentra flip con bisecciones en el segmento [0,1],
    usando SOLO midpoints (3-5 predicciones típicamente).
    """
    tL, tR = 0.0, 1.0
    yL = _predict_labels(model, x0[None,:])[0]
    yR = _predict_labels(model, x1[None,:])[0]
    for _ in range(max(1, int(iters))):
        tm = 0.5*(tL+tR)
        xm = (1.0-tm)*x0 + tm*x1
        ym = _predict_labels(model, xm[None,:])[0]
        if ym != yL:
            tR = tm; yR = ym
        else:
            tL = tm; yL = ym
    if yL != yR:
        return True, 0.5*(tL+tR)
    tm = 0.5*(tL+tR)
    ym = _predict_labels(model, ((1.0-tm)*x0 + tm*x1)[None,:])[0]
    return (ym != y0), tm if (ym != y0) else (False, None)

# -------------------------
# Adaptador de probas
# -------------------------
class ScoreAdaptor:
    def __init__(self, model, mode: str = "auto",
                 cache_enabled: bool = True, cache_decimals: int = 6):
        self.model = model
        self.mode = mode
        self.cache_enabled = bool(cache_enabled)
        self.cache_decimals = int(cache_decimals)
        self._cache: Dict[bytes, np.ndarray] = {}
        self._pd_cols = None  # cachea nombres de columnas si hay

    def _maybe_frame(self, X: np.ndarray):
        # convierte a DataFrame una sola vez por batch si el modelo lo requiere
        try:
            import pandas as pd
        except Exception:
            pd = None
        if pd is not None and hasattr(self.model, "feature_names_in_"):
            if self._pd_cols is None:
                self._pd_cols = list(getattr(self.model, "feature_names_in_"))
            if X.shape[1] == len(self._pd_cols):
                return pd.DataFrame(X, columns=self._pd_cols)
        return X
    def scores_dedup(self, X: np.ndarray) -> np.ndarray:
        import numpy as np
        X = np.asarray(X, float)
        if X.ndim == 1:
            return self.scores(X)
        Q = np.round(X, getattr(self, "cache_decimals", 6))
        uniq, idx, inv = np.unique(Q, axis=0, return_index=True, return_inverse=True)
        Suniq = self._scores_raw(uniq)
        keys = [u.tobytes() for u in uniq]
        for k, val in zip(keys, Suniq): self._cache[k] = val
        return Suniq[inv]

    def _scores_raw(self, X: np.ndarray) -> np.ndarray:
        m = self.model; mode = self.mode
        if mode == "auto":
            if hasattr(m, "predict_proba"): mode = "proba"
            elif hasattr(m, "decision_function"): mode = "decision"
            elif callable(m): mode = "callable"
            else: raise ValueError("Modelo no soportado.")
        X_in = self._maybe_frame(X)
        if mode == "proba":
            return np.asarray(m.predict_proba(X_in), float)
        if mode == "decision":
            D = np.asarray(m.decision_function(X_in), float)
            if D.ndim == 1:
                P1 = _sigmoid(D).reshape(-1, 1); return np.c_[1.0 - P1, P1]
            return _softmax(D, axis=1)
        if mode == "callable":
            S = np.asarray(m(X_in), float)
            if S.ndim == 1: S = np.c_[1.0 - S, S.reshape(-1,1)]
            return S
        raise ValueError("Modo desconocido.")

    def scores(self, X: np.ndarray) -> np.ndarray:
        from time import perf_counter
        import numpy as np
        X = np.asarray(X, float)
        batch = 1 if X.ndim == 1 else X.shape[0]
        nfeat = X.size if X.ndim == 1 else X.shape[1]
        cache_hits = 0
        cache_misses = 0
        cache_enabled = bool(getattr(self, "cache_enabled", False)) and (X.ndim != 1)

        # Decide effective mode
        mode_used = getattr(self, "mode", "auto")
        if mode_used == "auto":
            m = self.model
            if hasattr(m, "predict_proba"): mode_used = "proba"
            elif hasattr(m, "decision_function"): mode_used = "decision"
            elif callable(m): mode_used = "callable"
            else: mode_used = "unknown"

        t0 = perf_counter()
        if not cache_enabled:
            out = self._scores_raw(X.reshape(1,-1) if X.ndim==1 else X)
            cache_misses = batch
        else:
            Q = np.round(X, getattr(self, "cache_decimals", 6))
            keys = [Q[i].tobytes() for i in range(Q.shape[0])]
            miss_idx = [i for i,k in enumerate(keys) if k not in self._cache]
            cache_misses = len(miss_idx); cache_hits = batch - cache_misses
            if miss_idx:
                S_new = self._scores_raw(Q[miss_idx])
                for j, i in enumerate(miss_idx):
                    self._cache[keys[i]] = S_new[j]
            out = np.vstack([self._cache[k] for k in keys])
        dt_ms = (perf_counter() - t0) * 1000.0

        _inp_snap = _snapshot_array(X); _out_snap = _snapshot_array(out)
        stage_val = _DELD_STAGE.get() or _infer_stage()
        _log_model_call(ModelCall(
            ts=time.time(), stage=stage_val, source="ScoreAdaptor.scores",
            fn={"proba":"predict_proba","decision":"decision_function","callable":"callable"}.get(mode_used, str(mode_used)),
            batch=batch, n_features=nfeat, cache_enabled=cache_enabled,
            cache_hits=cache_hits, cache_misses=cache_misses, duration_ms=dt_ms,
            extra={"model_class": type(self.model).__name__}, input=_inp_snap, output=_out_snap
        ).__dict__)

        return out

# -------------------------
# Secante + bisección (vectorizado) para flips
# -------------------------
def batch_false_position_flip(
    adaptor: ScoreAdaptor,
    A: np.ndarray, B: np.ndarray,
    yA: np.ndarray, yB: np.ndarray,
    iters: int = 2, final_bisect: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, d = A.shape
    SA = adaptor.scores(A); SB = adaptor.scores(B)
    hA = SA[np.arange(m), yA] - SA[np.arange(m), yB]
    hB = SB[np.arange(m), yA] - SB[np.arange(m), yB]

    denom = (hA - hB)
    t = np.clip(hA / (denom + 1e-12), 1e-4, 1-1e-4)
    X = (1.0 - t)[:, None]*A + t[:, None]*B
    S = adaptor.scores(X); y = np.argmax(S, axis=1)

    for _ in range(max(0, iters-1)):
        hX = S[np.arange(m), yA] - S[np.arange(m), yB]
        maskA = (y == yA) | ((hX > 0) & (y != yA) & (y != yB))
        maskB = (y == yB) | ((hX <= 0) & (y != yA) & (y != yB))
        A = np.where(maskA[:, None], X, A); B = np.where(maskB[:, None], X, B)
        SA = np.where(maskA[:, None], S, SA); SB = np.where(maskB[:, None], S, SB)
        hA = SA[np.arange(m), yA] - SA[np.arange(m), yB]
        hB = SB[np.arange(m), yA] - SB[np.arange(m), yB]
        t = np.clip(hA / ((hA - hB) + 1e-12), 1e-4, 1-1e-4)
        X = (1.0 - t)[:, None]*A + t[:, None]*B
        S = adaptor.scores(X); y = np.argmax(S, axis=1)

    lo = np.zeros(m); hi = np.ones(m); XA, XB = A.copy(), B.copy()
    for _ in range(final_bisect):
        mid = 0.5*(lo+hi)
        XM = (1.0 - mid)[:, None]*XA + mid[:, None]*XB
        SM = adaptor.scores(XM); yM = np.argmax(SM, axis=1)
        goA = (yM == yA)
        lo = np.where(goA, mid, lo); hi = np.where(goA, hi, mid)
        XA = np.where(goA[:,None], XM, XA); XB = np.where(~goA[:,None], XM, XB)
    Xstar = (1.0 - hi)[:, None]*A + hi[:, None]*B
    Sstar = adaptor.scores(Xstar); ystar = np.argmax(Sstar, axis=1)
    return Xstar, ystar, Sstar

def batch_false_position_flip(
    adaptor: ScoreAdaptor,
    A: np.ndarray, B: np.ndarray,
    yA: np.ndarray, yB: np.ndarray,
    iters: int = 2, final_bisect: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Misma lógica y MISMOS puntos que la versión original.
    Cambios:
      - Evalúa A y B en UNA sola llamada (apilando).
      - Usa scores_dedup si existe para reducir invocaciones del modelo
        cuando haya filas duplicadas dentro de una misma ronda.
    """
    m, d = A.shape
    eval_scores = adaptor.scores_dedup if hasattr(adaptor, "scores_dedup") else adaptor.scores

    # (1) A y B juntos en una sola llamada ──> mismos puntos, -1 llamada
    SAB = eval_scores(np.vstack([A, B]))
    SA, SB = SAB[:m], SAB[m:]

    # Paso de falsa posición inicial
    hA = SA[np.arange(m), yA] - SA[np.arange(m), yB]
    hB = SB[np.arange(m), yA] - SB[np.arange(m), yB]
    denom = (hA - hB)
    t = np.clip(hA / (denom + 1e-12), 1e-4, 1 - 1e-4)
    X = (1.0 - t)[:, None] * A + t[:, None] * B

    S = eval_scores(X)               # mismos X que antes
    y = np.argmax(S, axis=1)

    # Iteraciones de falsa posición (cada ronda requiere una evaluación nueva)
    for _ in range(max(0, iters - 1)):
        hX = S[np.arange(m), yA] - S[np.arange(m), yB]
        maskA = (y == yA) | ((hX > 0) & (y != yA) & (y != yB))
        maskB = (y == yB) | ((hX <= 0) & (y != yA) & (y != yB))

        A = np.where(maskA[:, None], X, A)
        B = np.where(maskB[:, None], X, B)
        SA = np.where(maskA[:, None], S, SA)
        SB = np.where(maskB[:, None], S, SB)

        hA = SA[np.arange(m), yA] - SA[np.arange(m), yB]
        hB = SB[np.arange(m), yA] - SB[np.arange(m), yB]
        t = np.clip(hA / ((hA - hB) + 1e-12), 1e-4, 1 - 1e-4)
        X = (1.0 - t)[:, None] * A + t[:, None] * B

        S = eval_scores(X)           # mismos X que antes
        y = np.argmax(S, axis=1)

    # Bisección final (cada ronda depende de la anterior; 1 llamada por ronda)
    lo = np.zeros(m); hi = np.ones(m); XA, XB = A.copy(), B.copy()
    for _ in range(final_bisect):
        mid = 0.5 * (lo + hi)
        XM = (1.0 - mid)[:, None] * XA + mid[:, None] * XB

        SM = eval_scores(XM)         # mismos XM que antes
        yM = np.argmax(SM, axis=1)

        goA = (yM == yA)
        lo = np.where(goA, mid, lo)
        hi = np.where(goA, hi, mid)
        XA = np.where(goA[:, None], XM, XA)
        XB = np.where(~goA[:, None], XM, XB)

    # Punto final en la frontera para cada par
    Xstar = (1.0 - hi)[:, None] * A + hi[:, None] * B
    Sstar = eval_scores(Xstar)       # mismos Xstar que antes
    ystar = np.argmax(Sstar, axis=1)
    return Xstar, ystar, Sstar



# -------------------------
# Records & Configs
# -------------------------
@dataclass
class DeltaRecord:
    index_a: int
    index_b: int
    method: str
    success: bool
    y0: int
    y1: int
    delta_norm_l2: float
    delta_norm_linf: float
    score_change: float
    distance_term: float
    change_term: float
    final_score: float
    time_ms: float
    x0: np.ndarray
    x1: np.ndarray
    delta: np.ndarray
    S0: np.ndarray
    S1: np.ndarray
    # --- NUEVOS: trazabilidad del cambio de probabilidad ---
    prob_swing: float = 0.0
    margin_gain: float = 0.0
    jsd_change: float = 0.0
    # --- Puntos de cambio a lo largo del segmento ---
    cp_t: np.ndarray = field(default_factory=lambda: np.empty(0, float))
    cp_x: np.ndarray = field(default_factory=lambda: np.empty((0,0), float))  # (m,d)
    cp_y_left: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    cp_y_right: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    cp_count: int = 0

@dataclass
class DelDelConfig:
    # I/O y scoring base
    mode: str = "auto"
    log_level: int = logging.INFO
    secant_iters: int = 2
    final_bisect: int = 8
    distance_metric: str = "l2"
    alpha_change: float = 0.6
    min_pair_margin_end: float = 0.02
    min_logit_gain: float = 0.25
    random_state: int = 0
    # mezcla para 'score_change'
    prob_swing_weight: float = 0.7
    use_jsd: bool = False
    jsd_weight: float = 0.15
    # ----- “1 solo knob” + política de candidatos -----
    segments_target: int = 120        # objetivo global de segmentos
    near_frac: float = 0.7            # proporción NEAR vs FAR
    k_near_base: int = 5              # k-NN por i en NEAR
    k_far_per_i: int = 2              # muestras por i en FAR
    q_near: float = 0.35              # cuantil distancia NEAR
    q_far: float = 0.80               # cuantil distancia FAR
    margin_quantile: float = 0.25     # seeds cerca del margen
    pregate_tau: float = 0.08         # |m_ab| umbral de empate
    # heurística barata de priorización
    w_marg: float = 0.45
    w_ent: float = 0.35
    w_dist: float = 0.20
    # límite global (si None ⇒ 3*segments_target)
    hard_cap: Optional[int] = None


    per_point_max_use: int = 1          # sin reutilizar; sube a 2 solo si hace falta

    # semillas / candidatos
    seed_overdraw: float = 3.0          # semillas ≈ seed_overdraw * presupuesto de clase
    # (q_near, q_far ya no son necesarios estrictamente; los mantengo por compatibilidad)

    # scoring barato (ponderaciones)
    w_div: float  = 0.30                # diversidad angular


    def __post_init__(self):
        if self.hard_cap is None:
            self.hard_cap = 3 * int(self.segments_target)

@dataclass
class ChangePointConfig:
    enabled: bool = True
    mode: str = "auto"           # 'auto' | 'treefast' | 'generic'
    # límites y presupuesto
    per_record_max_points: int = 32
    only_success: bool = True
    topk_records: Optional[int] = None
    # treefast
    max_candidates: int = 512
    max_bisect_iters: int = 12
    # generic
    base_samples: int = 128

    # ---- NUEVOS (adelgazado y caps) ----
    unique_tol: float = 1e-6     # tolerancia para unificar t ~duplicados
    mids_cap: int = 2048         # máximo de midpoints a evaluar por segmento

# -------------------------
# Predicción etiquetas (para CP)
# -------------------------
def _predict_labels(model, X: np.ndarray, *, adaptor: Optional[ScoreAdaptor] = None, verbose: bool = False) -> np.ndarray:
    X = np.asarray(X, float)
    if adaptor is not None:
        S = adaptor.scores_dedup(X) if hasattr(adaptor, 'scores_dedup') else adaptor.scores(X)
        return np.argmax(S, axis=1)
    # (solo si no hay adaptor, opcionalmente imprime)
    if verbose:
        print('Usando modelo', X.shape)
    if hasattr(model, "predict"):
        y = model.predict(X); return np.asarray(y)
    if hasattr(model, "predict_proba"):
        P = model.predict_proba(X); return np.argmax(P, axis=1)
    if hasattr(model, "decision_function"):
        D = model.decision_function(X)
        if D.ndim == 1: return (D > 0).astype(int)
        return np.argmax(D, axis=1)
    raise ValueError("El modelo no soporta predict/predict_proba/decision_function")


def _points_on_segment(x0: np.ndarray, x1: np.ndarray, t: np.ndarray) -> np.ndarray:
    return x0[None, :] + t[:, None] * (x1 - x0)[None, :]

def _unique_sorted(arr: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    if arr.size == 0:
        return arr
    arr = np.sort(arr)
    keep = [0]
    for i in range(1, arr.size):
        if abs(arr[i] - arr[keep[-1]]) > tol:
            keep.append(i)
    return arr[keep]

# -------------------------
# TREEFAST helpers
# -------------------------
def _iter_sklearn_trees(model):
    """Rinde cada árbol (obj con .tree_) en RF/GBDT de sklearn."""
    if not hasattr(model, "estimators_"):
        return
    ests = model.estimators_
    try:
        arr = np.array(ests, dtype=object).flatten()
    except Exception:
        arr = ests
    for e in arr:
        if hasattr(e, "tree_"):
            yield e.tree_

def _is_tree_ensemble(model) -> bool:
    try:
        for _ in _iter_sklearn_trees(model):
            return True
        return False
    except Exception:
        return False

def _cross_ts_for_tree(tree, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    f = tree.feature
    thr = tree.threshold
    mask = f >= 0
    if not np.any(mask):
        return np.empty(0, float)
    f = f[mask]; thr = thr[mask]
    v0 = x0[f]; v1 = x1[f]
    den = v1 - v0
    mask_nz = np.abs(den) > 1e-12
    if not np.any(mask_nz):
        return np.empty(0, float)
    f = f[mask_nz]; v0 = v0[mask_nz]; v1 = v1[mask_nz]; thr = thr[mask_nz]; den = den[mask_nz]
    sign0 = v0 - thr; sign1 = v1 - thr
    mask_cross = (sign0 * sign1) < 0.0
    if not np.any(mask_cross):
        return np.empty(0, float)
    t = (thr[mask_cross] - v0[mask_cross]) / den[mask_cross]
    mask_01 = (t > 0.0) & (t < 1.0)
    return t[mask_01]

def _treefast_candidates(model, x0: np.ndarray, x1: np.ndarray,
                         max_candidates: Optional[int] = None,
                         unique_tol: float = 1e-6) -> np.ndarray:
    ts = []
    for tree in _iter_sklearn_trees(model):
        ts.append(_cross_ts_for_tree(tree, x0, x1))
    if len(ts) == 0:
        return np.empty(0, float)
    t_all = np.concatenate(ts) if len(ts) > 1 else ts[0]
    # unificar con tolerancia más laxa (evita miles de t casi repetidos)
    t_all = _unique_sorted(t_all, tol=unique_tol)
    if max_candidates is not None and t_all.size > max_candidates:
        # muestreo equiespaciado alrededor del intervalo (más robusto que centrar en 0.5)
        idx = np.linspace(0, t_all.size - 1, max_candidates).round().astype(int)
        t_all = t_all[np.unique(idx)]
    return t_all


def _bisection_labels_on_segment(model, x0: np.ndarray, x1: np.ndarray,
                                 tL: np.ndarray, tR: np.ndarray,
                                 max_iters: int = 22, eps: float = 1e-9,
                                 adaptor: Optional[ScoreAdaptor] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tL = np.asarray(tL, float).copy()
    tR = np.asarray(tR, float).copy()
    YL = _predict_labels(model, _points_on_segment(x0, x1, tL + eps), adaptor=adaptor)
    YR = _predict_labels(model, _points_on_segment(x0, x1, tR - eps), adaptor=adaptor)
    for _ in range(max_iters):
        tm = 0.5*(tL + tR)
        Ym = _predict_labels(model, _points_on_segment(x0, x1, tm), adaptor=adaptor)
        go_left = (Ym == YL)
        tL = np.where(go_left, tm, tL); YL = np.where(go_left, Ym, YL)
        tR = np.where(~go_left, tm, tR); YR = np.where(~go_left, Ym, YR)
    t_star = 0.5*(tL + tR)
    return t_star, YL, YR


def _find_change_points_along_segment_treefast(
    model, x0: np.ndarray, x1: np.ndarray,
    max_candidates: int = 4096, max_bisect_iters: int = 22,
    *, adaptor: Optional[ScoreAdaptor] = None,
    unique_tol: float = 1e-6, mids_cap: int = 2048,
    limit_points: Optional[int] = None
) -> List[Dict[str, Any]]:
    x0 = np.asarray(x0, float); x1 = np.asarray(x1, float)
    if np.allclose(x0, x1): return []
    # candidatos t con unificación más laxa y cap temprano
    t_cand = _treefast_candidates(model, x0, x1,
                                  max_candidates=max_candidates,
                                  unique_tol=unique_tol)
    if t_cand.size == 0:
        return []
    # bordes + midpoints
    tB = _unique_sorted(np.concatenate([[0.0], t_cand, [1.0]]), tol=unique_tol)
    mids = 0.5*(tB[:-1] + tB[1:])
    # cap a mids (si hay demasiados, subsample equiespaciado)
    if mids.size > mids_cap:
        idx = np.linspace(0, mids.size - 1, mids_cap).round().astype(int)
        mids = mids[np.unique(idx)]

    # etiquetas en mids (1 solo batch; con adaptor usa cache)
    y_mid = _predict_labels(model, _points_on_segment(x0, x1, mids), adaptor=adaptor)
    change_idx = np.where(y_mid[1:] != y_mid[:-1])[0]
    if change_idx.size == 0:
        return []

    # ---- EARLY LIMIT por presupuesto de puntos por record ----
    if limit_points is not None and change_idx.size > limit_points:
        keep = np.linspace(0, change_idx.size - 1, limit_points).round().astype(int)
        change_idx = change_idx[np.unique(keep)]

    tL = tB[change_idx]; tR = tB[change_idx + 1]
    t_star, yL, yR = _bisection_labels_on_segment(
        model, x0, x1, tL, tR, max_iters=max_bisect_iters, adaptor=adaptor
    )
    pts = _points_on_segment(x0, x1, t_star)
    out = []
    for ts, p, yl, yr in zip(t_star, pts, yL, yR):
        out.append({"t": float(ts), "x": p.astype(float), "y_left": int(yl), "y_right": int(yr)})
    return out


# -------------------------
# GENÉRICO (barrido adaptativo)
# -------------------------
def _find_change_points_along_segment_generic(
    model, x0: np.ndarray, x1: np.ndarray,
    base_samples: int = 64, max_bisect_iters: int = 22,
    *, adaptor: Optional[ScoreAdaptor] = None,
    limit_points: Optional[int] = None
) -> List[Dict[str, Any]]:
    x0 = np.asarray(x0, float); x1 = np.asarray(x1, float)
    if np.allclose(x0, x1): return []
    t_grid = np.linspace(0.0, 1.0, base_samples + 1)
    mids = 0.5*(t_grid[:-1] + t_grid[1:])
    y_mid = _predict_labels(model, _points_on_segment(x0, x1, mids), adaptor=adaptor)
    change_idx = np.where(y_mid[1:] != y_mid[:-1])[0]
    if change_idx.size == 0:
        return []
    # EARLY LIMIT
    if limit_points is not None and change_idx.size > limit_points:
        keep = np.linspace(0, change_idx.size - 1, limit_points).round().astype(int)
        change_idx = change_idx[np.unique(keep)]
    tL = t_grid[change_idx]; tR = t_grid[change_idx + 1]
    t_star, yL, yR = _bisection_labels_on_segment(
        model, x0, x1, tL, tR, max_iters=max_bisect_iters, adaptor=adaptor
    )
    pts = _points_on_segment(x0, x1, t_star)
    out = []
    for ts, p, yl, yr in zip(t_star, pts, yL, yR):
        out.append({"t": float(ts), "x": p.astype(float), "y_left": int(yl), "y_right": int(yr)})
    return out


def _find_change_points_along_segment(
    model, x0: np.ndarray, x1: np.ndarray, cp: ChangePointConfig,
    *, adaptor: Optional[ScoreAdaptor] = None
) -> List[Dict[str, Any]]:
    mode = cp.mode
    if mode == "auto":
        mode = "treefast" if _is_tree_ensemble(model) else "generic"
    if mode == "treefast":
        return _find_change_points_along_segment_treefast(
            model, x0, x1,
            max_candidates=cp.max_candidates,
            max_bisect_iters=cp.max_bisect_iters,
            adaptor=adaptor,
            unique_tol=cp.unique_tol,
            mids_cap=cp.mids_cap,
            limit_points=cp.per_record_max_points
        )
    if mode == "generic":
        return _find_change_points_along_segment_generic(
            model, x0, x1,
            base_samples=cp.base_samples,
            max_bisect_iters=cp.max_bisect_iters,
            adaptor=adaptor,
            limit_points=cp.per_record_max_points
        )
    raise ValueError("ChangePointConfig.mode debe ser 'auto' | 'treefast' | 'generic'")



# === Batch pool for CP stage ==================================================
class _ReqCP(NamedTuple):
    rid: int
    kind: str
    idx: int
    x: np.ndarray

class _BatchPoolCP:
    def __init__(self, adaptor: ScoreAdaptor, cache_decimals: int):
        self.adaptor = adaptor
        self._bak = adaptor.cache_decimals
        self.adaptor.cache_decimals = cache_decimals
        self.reqs: List[_ReqCP] = []
        self.out: Dict[Tuple[int,str,int], np.ndarray] = {}

    def push(self, rid: int, kind: str, idx: int, x1d: np.ndarray):
        self.reqs.append(_ReqCP(rid, kind, idx, np.asarray(x1d, float)))

    def flush(self):
        if not self.reqs:
            return
        X = np.vstack([r.x for r in self.reqs])
        S = self.adaptor.scores_dedup(X) if hasattr(self.adaptor, "scores_dedup") else self.adaptor.scores(X)
        for r, s in zip(self.reqs, S):
            self.out[(r.rid, r.kind, r.idx)] = s
        self.reqs.clear()

    def get(self, rid: int, kind: str, idx: int) -> np.ndarray:
        return self.out[(rid, kind, idx)]

    def close(self):
        self.adaptor.cache_decimals = self._bak
# =============================================================================

# -------------------------
# Núcleo Balanced (pares) + CP integrado
# -------------------------
class DelDel:
    def __init__(self, config: DelDelConfig, cp_config: Optional[ChangePointConfig] = None):
        self.cfg = config
        self.cp_cfg = cp_config or ChangePointConfig(enabled=False)
        self.logger = _get_logger("DelDel", config.log_level)
        self.records_: List[DeltaRecord] = []
        self._adaptor: Optional[ScoreAdaptor] = None
        self._X: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self.calls_: List[Dict[str, Any]] = []

    def _pair_candidates_round_robin(self, labels):
        """
        Selección de segmentos con:
          - Objetivo global exacto: segments_target (mezclado por round-robin al final).
          - Balanceo por par NO ordenado con cuotas exactas según pair_mix_target (o uniforme).
          - Candidatos NEAR/FAR sin llamadas al modelo (distancias + P ya computadas).
          - Unicidad por par (origen no se repite en el par) y presupuesto global de reuso de origen.
          - Cap de reutilización de destino por par.
          - Expansión adaptativa para pares deficitarios (sin modelo).
          - Fases de relleno (relajar presupuestos) para alcanzar cuotas/per-par y total.

        Devuelve: [A], [B], yA_list, yB_list
        """
        import numpy as np
        rng = np.random.RandomState(getattr(self.cfg, "random_state", 0))

        # =================== Parámetros (con defaults) ===================
        segments_target = int(getattr(self.cfg, "segments_target", 120))

        # Mezcla explotación/exploración (cerca/lejos)
        near_frac   = float(getattr(self.cfg, "near_frac", 0.7))
        k_near_base = int(getattr(self.cfg, "k_near_base", 5))
        k_far_per_i = int(getattr(self.cfg, "k_far_per_i", 2))
        q_near      = float(getattr(self.cfg, "q_near", 0.35))
        q_far       = float(getattr(self.cfg, "q_far", 0.80))
        margin_q    = float(getattr(self.cfg, "margin_quantile", 0.25))
        pregate_tau = float(getattr(self.cfg, "pregate_tau", 0.08))

        # Heurística de score 100% barata
        w_marg = float(getattr(self.cfg, "w_marg", 0.45))
        w_ent  = float(getattr(self.cfg, "w_ent", 0.35))
        w_dist = float(getattr(self.cfg, "w_dist", 0.20))

        # Unicidad / caps / adaptación
        origin_global_budget   = int(getattr(self.cfg, "origin_global_budget", 2))   # veces máx que un origen puede aparecer en TOTAL
        max_dest_reuse_per_pair= int(getattr(self.cfg, "max_dest_reuse_per_pair", 2))# veces máx que un mismo destino (jj) se use en un par
        adaptive_rounds        = int(getattr(self.cfg, "adaptive_rounds", 2))
        adaptive_margin_q_step = float(getattr(self.cfg, "adaptive_margin_q_step", 0.10))
        adaptive_k_near_scale  = float(getattr(self.cfg, "adaptive_k_near_scale", 1.5))
        adaptive_q_near_delta  = float(getattr(self.cfg, "adaptive_q_near_delta", -0.05)) # q_near ↓ ⇒ más "cerca"
        adaptive_q_far_delta   = float(getattr(self.cfg, "adaptive_q_far_delta",  +0.05)) # q_far ↑  ⇒ más "lejos"

        print("\n=== [_pair_candidates_round_robin] ===")
        print(f"segments_target={segments_target} | near_frac={near_frac} | "
              f"k_near_base={k_near_base} | k_far_per_i={k_far_per_i} | "
              f"q_near={q_near} | q_far={q_far} | margin_q={margin_q} | pregate_tau={pregate_tau}")
        print(f"[Uniqueness] origin_global_budget={origin_global_budget} | "
              f"max_dest_reuse_per_pair={max_dest_reuse_per_pair} | adaptive_rounds={adaptive_rounds}")

        # =================== Datos base ===================
        X, P, y = self._X, self._P, self._y
        if X is None or P is None or y is None:
            print(" [ERROR] X/P/y no inicializados. Regresando vacío.")
            return [], [], [], []
        n, d = X.shape
        C = P.shape[1]
        dist_norm = np.sqrt(float(d)) + 1e-9

        # ============ Pares NO ORDENADOS y distribución objetivo ============
        # # Construye lista de pares (a<b) presentes
        # uniq_labels = sorted(np.unique(y).tolist())
        # pairs_all = [(a,b) for i,a in enumerate(uniq_labels) for b in uniq_labels[i+1:]]

        # # Distribución objetivo: cfg.pair_mix_target (fracciones) o uniforme
        # user_mix = getattr(self.cfg, "pair_mix_target", None)
        # if user_mix:
        #     # normalizar por si no suma 1
        #     keys_ok = [(min(a,b), max(a,b)) for (a,b) in user_mix.keys()]
        #     pair_mix = {}
        #     s = 0.0
        #     for k in pairs_all:
        #         if k in keys_ok:
        #             pair_mix[k] = float(user_mix[k])
        #             s += pair_mix[k]
        #         else:
        #             pair_mix[k] = 0.0
        #     if s <= 0:
        #         pair_mix = {k: 1.0/len(pairs_all) for k in pairs_all}
        #     else:
        #         pair_mix = {k: v/s for k,v in pair_mix.items()}
        # else:
        #     pair_mix = {k: 1.0/len(pairs_all) for k in pairs_all} if pairs_all else {}

        # # Cuotas por par (sum exactly segments_target vía largest remainder)
        # quotas = {}
        # remainders = []
        # total_floor = 0
        # for k in pairs_all:
        #     q = pair_mix.get(k, 0.0) * segments_target
        #     quotas[k] = int(np.floor(q))
        #     remainders.append((q - quotas[k], k))
        #     total_floor += quotas[k]
        # # distribuye lo que falta
        # missing = segments_target - total_floor
        # remainders.sort(reverse=True)
        # for i in range(missing):
        #     quotas[remainders[i][1]] += 1
        # ============ Pares NO ORDENADOS y distribución objetivo (proporcional a clases) ============
        # Proporciones por clase estimadas con el modelo (sin usar y):
        P_all = self._P  # proba ya calculada en 01_scores_global
        C = P_all.shape[1]
        p_hat = P_all.mean(axis=0)  # proporción por clase (suma 1 aprox.)
        # Clases "presentes": aquellas con masa no despreciable
        present = [c for c in range(C) if p_hat[c] > 1e-12]

        # Construir lista de pares no ordenados sólo con clases presentes
        pairs_all = [(a, b) for i, a in enumerate(present) for b in present[i+1:]]

        # Mezcla objetivo por par:
        # - Si el usuario provee pair_mix_target, se respeta (normalizando).
        # - En caso contrario, se usa producto de marginales: w_ab ∝ p_a * p_b.
        user_mix = getattr(self.cfg, "pair_mix_target", None)
        if user_mix:
            # Normalizar y limpiar claves a no ordenadas (a<b)
            mix_tmp = {}
            s = 0.0
            for (ua, ub), v in user_mix.items():
                k = (min(int(ua), int(ub)), max(int(ua), int(ub)))
                if k in pairs_all:
                    mix_tmp[k] = float(v)
                    s += float(v)
            if s <= 0.0:
                # fallback a producto de marginales si el usuario pasó ceros
                mix_tmp = {k: float(p_hat[k[0]] * p_hat[k[1]]) for k in pairs_all}
                s = sum(mix_tmp.values()) + 1e-12
            pair_mix = {k: mix_tmp[k] / s for k in pairs_all}
        else:
            weights = {k: float(p_hat[k[0]] * p_hat[k[1]]) for k in pairs_all}
            s = sum(weights.values()) + 1e-12
            pair_mix = {k: (weights[k] / s) for k in pairs_all}

        # Cuotas por par (sumar EXACTAMENTE segments_target) con método de residuos máximos
        quotas = {}
        remainders = []
        total_floor = 0
        for k in pairs_all:
            q = pair_mix.get(k, 0.0) * segments_target
            quotas[k] = int(np.floor(q))
            remainders.append((q - quotas[k], k))
            total_floor += quotas[k]
        missing = segments_target - total_floor
        remainders.sort(reverse=True)
        for i in range(missing):
            quotas[remainders[i][1]] += 1

        print("Pares (no ordenados) y cuotas objetivo (producto de marginales salvo override):", quotas)
        print("Proporciones por clase (p_hat):", {i: round(float(p_hat[i]), 4) for i in present})

        print("Pares (no ordenados) y cuotas objetivo:", quotas)

        if not pairs_all:
            print("No hay pares disponibles con labels presentes.")
            return [], [], [], []

        # =================== Utils locales ===================
        def _entropy_rows(Q):
            return -np.sum(Q * np.log(Q + 1e-12), axis=1)

        def _build_bag_for_pair(a, b, *,
                                margin_q_local,
                                q_near_local, q_far_local,
                                k_near_local, k_far_local) -> dict:
            """Devuelve dict con 'cand' lista (score, ii, jj, dist), stats y umbrales usados."""
            out = {
                "cand": [],
                "stats": {},
                "thresholds": {},
            }
            m_ab = P[:, a] - P[:, b]
            Ia_all = np.where(y == a)[0]
            Ib_all = np.where(y == b)[0]
            if Ia_all.size == 0 or Ib_all.size == 0:
                out["stats"]["empty_classes"] = True
                return out

            # Seeds cerca del margen (por lado, usando |m_ab|)
            thr_a = np.quantile(np.abs(m_ab[Ia_all]), margin_q_local) if Ia_all.size>0 else 0.0
            thr_b = np.quantile(np.abs(m_ab[Ib_all]), margin_q_local) if Ib_all.size>0 else 0.0
            base_seed = max(8, int(1.5 * np.sqrt(max(1, quotas.get((a,b), k_near_local)))))
            Ia = Ia_all[np.argsort(np.abs(m_ab[Ia_all]))[:base_seed*6]]
            Ia = Ia[np.abs(m_ab[Ia]) <= thr_a + 1e-12]
            Ib = Ib_all[np.argsort(np.abs(m_ab[Ib_all]))[:base_seed*6]]
            Ib = Ib[np.abs(m_ab[Ib]) <= thr_b + 1e-12]

            if Ia.size == 0 or Ib.size == 0:
                out["stats"]["no_seeds"] = True
                out["thresholds"] = {"thr_a": thr_a, "thr_b": thr_b}
                return out

            # Ajuste k_near por disponibilidad
            k_near_use = int(max(k_near_base, np.ceil(quotas.get((a,b), k_near_base) / max(1, Ia.size))))
            k_near_use = min(k_near_use, max(1, Ib.size))

            X_a, X_b = X[Ia], X[Ib]
            P_a, P_b = P[Ia], P[Ib]
            H_a, H_b = _entropy_rows(P_a), _entropy_rows(P_b)

            # Distancias ALL vs ALL (típicamente chico; si crece, cambiar a ANN)
            D = np.linalg.norm(X_a[:, None, :] - X_b[None, :, :], axis=2)
            qn = np.quantile(D, q_near_local) if D.size else 0.0
            qf = np.quantile(D, q_far_local) if D.size else 0.0

            cand = []
            seen_pairs = set()
            # --- NEAR ---
            near_count_rej = 0
            for i_idx, ii in enumerate(Ia):
                drow = D[i_idx]
                if Ib.size > k_near_use:
                    J = np.argpartition(drow, kth=k_near_use)[:k_near_use]
                else:
                    J = np.arange(Ib.size)
                for j_idx in J:
                    jj = Ib[j_idx]
                    dij = float(drow[j_idx])
                    if dij > qn:
                        continue
                    # pre-gate sin modelo
                    sgn_i = np.sign(m_ab[ii]); sgn_j = np.sign(m_ab[jj])
                    near_i = np.abs(m_ab[ii]) <= pregate_tau
                    near_j = np.abs(m_ab[jj]) <= pregate_tau
                    passes = (sgn_i * sgn_j <= 0) or (near_i or near_j)
                    if not passes:
                        near_count_rej += 1
                        continue
                    key = (int(ii), int(jj))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    s_m = (1.0 - abs(m_ab[ii]) + 1.0 - abs(m_ab[jj])) / 2.0
                    s_e = 0.5 * (H_a[i_idx] + H_b[j_idx])
                    s_d = 1.0 - (dij / dist_norm)
                    score = w_marg * s_m + w_ent * s_e + w_dist * s_d
                    cand.append((float(score), int(ii), int(jj), dij))
            # --- FAR ---
            far_count_rej = 0
            for i_idx, ii in enumerate(Ia):
                drow = D[i_idx]
                J_far = np.where(drow >= qf)[0]
                if J_far.size == 0:
                    continue
                if J_far.size > k_far_local:
                    J_far_sel = rng.choice(J_far, size=k_far_local, replace=False)
                else:
                    J_far_sel = J_far
                for j_idx in J_far_sel:
                    jj = Ib[j_idx]
                    dij = float(drow[j_idx])
                    sgn_i = np.sign(m_ab[ii]); sgn_j = np.sign(m_ab[jj])
                    near_i = np.abs(m_ab[ii]) <= pregate_tau
                    near_j = np.abs(m_ab[jj]) <= pregate_tau
                    passes = (sgn_i * sgn_j <= 0) or (near_i or near_j)
                    if not passes:
                        far_count_rej += 1
                        continue
                    key = (int(ii), int(jj))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    s_m = (1.0 - abs(m_ab[ii]) + 1.0 - abs(m_ab[jj])) / 2.0
                    s_e = 0.5 * (H_a[i_idx] + H_b[j_idx])
                    s_d = 1.0 - (min(dij, dist_norm) / dist_norm)
                    score = w_marg * s_m + w_ent * s_e + w_dist * s_d
                    cand.append((float(score), int(ii), int(jj), dij))

            cand.sort(reverse=True)
            uniq_origins = len({ii for _, ii, _, _ in cand})
            out["cand"] = cand
            out["thresholds"] = {"thr_a": float(thr_a), "thr_b": float(thr_b),
                                "q_near_val": float(qn), "q_far_val": float(qf)}
            out["stats"] = {
                "n_candidates": len(cand),
                "uniq_origins": uniq_origins,
                "near_rej": int(near_count_rej),
                "far_rej": int(far_count_rej),
                "Ia": int(Ia.size), "Ib": int(Ib.size)
            }
            # top10 orígenes pesados (sólo para diagnóstico)
            from collections import Counter
            c = Counter(ii for _,ii,_,_ in cand)
            top10 = ", ".join(f"{k}:{c[k]}" for k in [p for p,_ in c.most_common(10)])
            out["stats"]["heavy_origins_top10"] = top10
            return out

        # =================== Construcción de bolsas por par (ronda 0) ===================
        bags = {}
        for (a,b) in pairs_all:
            bag = _build_bag_for_pair(a, b,
                                      margin_q_local=margin_q,
                                      q_near_local=q_near, q_far_local=q_far,
                                      k_near_local=k_near_base, k_far_local=k_far_per_i)
            bags[(a,b)] = {
                "cand": bag["cand"],
                "params": {"margin_q": margin_q, "q_near": q_near, "q_far": q_far,
                          "k_near": k_near_base, "k_far": k_far_per_i},
                "stats": bag["stats"],
                "thresholds": bag["thresholds"],
                "rounds_built": 1
            }
            s = bag["stats"]
            thr = bag["thresholds"]
            print(f"[{(a,b)}] bag: candidatos={s.get('n_candidates',0)} | orígenes_únicos={s.get('uniq_origins',0)} "
                  f"| thr_a={thr.get('thr_a',0):.4f}, thr_b={thr.get('thr_b',0):.4f} | "
                  f"qn={thr.get('q_near_val',0):.3f}, qf={thr.get('q_far_val',0):.3f}")
            if s.get("heavy_origins_top10"):
                print(f"    Heavy-origins top10: {s['heavy_origins_top10']}")

        # Orden de servicio: pares con MENOR capacidad primero (evita cuellos)
        order_pairs = sorted(pairs_all, key=lambda k: bags[k]["stats"].get("uniq_origins", 0))
        # Estado global de uso de orígenes
        origin_used_global = {}  # idx -> count

        per_pair_lists = {k: [] for k in pairs_all}  # seleccionados (ii,jj)
        per_pair_reasons = {k: {"origin_used_pair":0, "origin_budget_global":0,
                                "dest_reuse_cap":0, "duplicate":0} for k in pairs_all}

        # =================== Selector por par con adaptaciones ===================
        def _pick_for_pair(key, target, allow_pair_origin_reuse=False, pair_origin_cap=1,
                          local_budget_mult=1, dest_reuse_cap=None):
            """Consume candidatos de bags[key] y llena per_pair_lists[key] hasta target."""
            (a,b) = key
            out = per_pair_lists[key]
            reasons = per_pair_reasons[key]
            cand = bags[key]["cand"]
            params = bags[key]["params"]

            # Estructuras locales de uso en el par
            origin_count_pair = {}  # ii -> times used en este par
            dest_count_pair   = {}  # jj -> times used en este par
            # Si ya había picks previos (rellenos), respétalos
            for (ii,jj) in out:
                origin_count_pair[ii] = origin_count_pair.get(ii,0)+1
                dest_count_pair[jj]   = dest_count_pair.get(jj,0)+1

            cap_dest = max_dest_reuse_per_pair if dest_reuse_cap is None else int(dest_reuse_cap)
            # Utilidad interna para intentar picks de una lista de candidatos
            def _scan_and_pick(candidates):
                picked_now = 0
                seen_dup = set((ii,jj) for (ii,jj) in out)
                for sc, ii, jj, _ in candidates:
                    keyp = (ii,jj)
                    if keyp in seen_dup:
                        reasons["duplicate"] += 1
                        continue
                    # Budget global
                    gcount = origin_used_global.get(ii, 0)
                    if gcount >= int(origin_global_budget*local_budget_mult):
                        reasons["origin_budget_global"] += 1
                        continue
                    # Cap destino por par
                    if dest_count_pair.get(jj, 0) >= cap_dest:
                        reasons["dest_reuse_cap"] += 1
                        continue
                    # Unicidad / cap por par en origen
                    if not allow_pair_origin_reuse:
                        if origin_count_pair.get(ii, 0) >= 1:
                            reasons["origin_used_pair"] += 1
                            continue
                    else:
                        if origin_count_pair.get(ii, 0) >= int(pair_origin_cap):
                            reasons["origin_used_pair"] += 1
                            continue
                    # OK, tomar
                    out.append((ii,jj))
                    seen_dup.add(keyp)
                    origin_count_pair[ii] = origin_count_pair.get(ii, 0) + 1
                    dest_count_pair[jj]   = dest_count_pair.get(jj, 0) + 1
                    origin_used_global[ii]= gcount + 1
                    picked_now += 1
                    if len(out) >= target:
                        break
                return picked_now

            # 1) Intento principal sobre bolsa actual
            _scan_and_pick(cand)
            if len(out) >= target:
                return

            # 2) Adaptaciones (construir más candidatos sin modelo)
            rounds_left = max(0, adaptive_rounds - bags[key]["rounds_built"])
            while len(out) < target and rounds_left > 0:
                # relajar parámetros localmente
                params["margin_q"] = min(0.99, params["margin_q"] + adaptive_margin_q_step)
                params["k_near"]   = int(np.ceil(params["k_near"] * adaptive_k_near_scale))
                params["q_near"]   = max(0.01, params["q_near"] + adaptive_q_near_delta)
                params["q_far"]    = min(0.99, params["q_far"]  + adaptive_q_far_delta)

                bag_new = _build_bag_for_pair(a, b,
                                              margin_q_local=params["margin_q"],
                                              q_near_local=params["q_near"], q_far_local=params["q_far"],
                                              k_near_local=params["k_near"], k_far_local=params["k_far"])
                bags[key]["rounds_built"] += 1
                rounds_left -= 1

                # merge de candidatos evitando duplicados exactos
                seen = set((ii,jj) for _,ii,jj,_ in cand)
                added = 0
                for sc,ii,jj,dd in bag_new["cand"]:
                    if (ii,jj) in seen:
                        continue
                    cand.append((sc,ii,jj,dd))
                    seen.add((ii,jj))
                    added += 1
                cand.sort(reverse=True)
                bags[key]["cand"] = cand  # actualizar referencia

                # intentar de nuevo
                _scan_and_pick(cand)

            # 3) Fases de relleno local si aún falta
            if len(out) < target:
                # 3A) Aumentar presupuesto global y cap de destino (manteniendo unicidad en par)
                _scan_and_pick(cand)  # primer barrido (por si cambió estado global)
                if len(out) < target:
                    _scan_and_pick(cand)  # otro pase
                if len(out) < target:
                    _scan_and_pick(cand)  # y otro

            if len(out) < target:
                # 3B) Permitir reutilizar origen en el par con cap pequeño
                _scan_and_pick(cand)  # por si hay margen
                allow_pair_origin_reuse_local = True
                cap_local_pair = 2  # hasta 2 veces el mismo origen en el par, sólo para relleno
                # Re-ejecutar con flags relajados
                prev_len = -1
                while len(out) < target and len(out) != prev_len:
                    prev_len = len(out)
                    # re-scan con límites relajados
                    # usar multiplicador de presupuesto global 1.5
                    _picked = 0
                    # re-scan manual para respetar los nuevos límites:
                    seen_dup = set((ii,jj) for (ii,jj) in out)
                    for sc, ii, jj, _ in cand:
                        if (ii,jj) in seen_dup:
                            continue
                        gcount = origin_used_global.get(ii, 0)
                        if gcount >= int(origin_global_budget*1.5):
                            continue
                        if (per_pair_reasons[key]["dest_reuse_cap"] is not None and
                            per_pair_reasons[key]["dest_reuse_cap"] >= 10**9):
                            pass
                        # dest cap más laxo
                        if dest_count_pair.get(jj, 0) >= (cap_dest+2):
                            continue
                        if origin_count_pair.get(ii, 0) >= cap_local_pair:
                            continue
                        out.append((ii,jj))
                        seen_dup.add((ii,jj))
                        origin_count_pair[ii] = origin_count_pair.get(ii, 0) + 1
                        dest_count_pair[jj]   = dest_count_pair.get(jj, 0) + 1
                        origin_used_global[ii]= gcount + 1
                        _picked += 1
                        if len(out) >= target:
                            break
                    if _picked == 0:
                        break
            return

        # =================== Recorrido por pares en orden de escasez ===================
        for key in order_pairs:
            tgt = quotas[key]
            _pick_for_pair(key, tgt)

        # Resumen por par
        total_sel = sum(len(per_pair_lists[k]) for k in pairs_all)
        print("[Resumen por par] segmentos:",
              {k: len(per_pair_lists[k]) for k in pairs_all},
              "| objetivo:", quotas)
        for k in pairs_all:
            reasons = per_pair_reasons[k]
            print(f"  {k}: picked={len(per_pair_lists[k])} | target={quotas[k]} | "
                  f"reasons={reasons} | uniq_origins_in_bag={bags[k]['stats'].get('uniq_origins',0)}")

        # =================== Garantizar EXACTAMENTE segments_target ===================
        if total_sel < segments_target:
            # Fase de relleno global: tratar de completar dentro de cada par primero;
            # si aún falta, redistribuir flexibilizando per-pair (pero sin cambiar las clases del par).
            missing_total = segments_target - total_sel
            print(f"[Relleno global] faltan={missing_total}. Se relajan límites y se reintenta en todos los pares.")
            # Vuelta 1: repetir _pick_for_pair con límites más laxos
            for key in order_pairs:
                if len(per_pair_lists[key]) < quotas[key]:
                    _pick_for_pair(key, quotas[key],
                                  allow_pair_origin_reuse=True, pair_origin_cap=3,
                                  local_budget_mult=2.0, dest_reuse_cap=max_dest_reuse_per_pair+3)
            total_sel = sum(len(per_pair_lists[k]) for k in pairs_all)

        # Recalcular (por si aún no se llegó, forzaremos redistribución final a pares con más stock)
        total_sel = sum(len(per_pair_lists[k]) for k in pairs_all)
        if total_sel < segments_target:
            print(f"[Aviso] No fue posible alcanzar segments_target={segments_target} con las restricciones. "
                  f"Seleccionados={total_sel}. Se hará un último intento redistributivo dentro de los mismos pares.")
            # Último intento: tomar del resto de candidatos aunque rompa más las reglas (pero sin cambiar el par)
            for key in order_pairs:
                need = quotas[key] - len(per_pair_lists[key])
                if need <= 0:
                    continue
                cand = bags[key]["cand"]
                seen = set(per_pair_lists[key])
                for sc,ii,jj,_ in cand:
                    if (ii,jj) in seen:
                        continue
                    per_pair_lists[key].append((ii,jj))
                    seen.add((ii,jj))
                    total_sel += 1
                    if total_sel >= segments_target or len(per_pair_lists[key]) >= quotas[key]:
                        break
                if total_sel >= segments_target:
                    break

        # =================== Ensamble final: round-robin por par ===================
        A_list, B_list, yA_list, yB_list = [], [], [], []
        keys = [k for k in pairs_all if len(per_pair_lists[k]) > 0]
        ptr = {k:0 for k in keys}
        exhausted = set()
        print("\n[Round-robin final]")
        print("Resumen elegidos por par:", {k: len(per_pair_lists[k]) for k in keys})
        while len(A_list) < segments_target and len(exhausted) < len(keys):
            for k in keys:
                if k in exhausted:
                    continue
                p = ptr[k]
                lst = per_pair_lists[k]
                if p >= len(lst):
                    exhausted.add(k)
                    continue
                ia, ib = lst[p]
                ptr[k] = p + 1
                A_list.append(X[ia]); B_list.append(X[ib])
                yA_list.append(k[0]);  yB_list.append(k[1])
                if len(A_list) >= segments_target:
                    break

        print(f"TOTAL seleccionados: {len(A_list)} segmentos (objetivo={segments_target})")

        if len(A_list) == 0:
            return [], [], [], []
        return [np.vstack(A_list)], [np.vstack(B_list)], yA_list, yB_list



    # ---------- API principal ----------
    def fit(self, X: np.ndarray, model: Any) -> "DelDel":
        with _collect_calls(self.calls_):

            self.calls_.clear()
            from time import perf_counter
            from contextlib import contextmanager

            @contextmanager
            def _tick(name: str):
                t = perf_counter()
                _tok_stage = _DELD_STAGE.set(name)
                try:
                    yield
                finally:
                    _DELD_STAGE.reset(_tok_stage)
                    dt = (perf_counter() - t) * 1000.0
                    timings[name] = timings.get(name, 0.0) + dt
                    events.append((name, dt))

            timings: Dict[str, float] = {}
            events: List[Tuple[str, float]] = []
            counts: Dict[str, Any] = {}

            t_total0 = perf_counter()

            with _tick("00_setup"):
                self._X = np.asarray(X, float)
                self._adaptor = ScoreAdaptor(model, mode=self.cfg.mode)
                self.records_.clear()
                counts["n_samples"], counts["n_features"] = self._X.shape

            with _tick("01_scores_global"):
                self._P = self._adaptor.scores(self._X)
                self._y = np.argmax(self._P, axis=1)
                labels = sorted(np.unique(self._y).tolist())
                counts["n_labels"] = len(labels)
                print('labels', labels)

            with _tick("02_pair_candidates_round_robin"):
                A_batches, B_batches, yA_list, yB_list = self._pair_candidates_round_robin(labels)

            if not A_batches:
                _get_logger("DelDel", logging.WARNING).warning(
                    "No se pudieron generar pares (verifica configuración y datos)."
                )
                self.time_stats_ = {
                    "timings_ms": timings,
                    "counts": counts,
                    "per_batch": [],
                    "per_pair": {},
                    "notes": "Sin pares candidatos"
                }
                return self

            all_records: List[DeltaRecord] = []
            per_batch_stats: List[Dict[str, Any]] = []
            counts["n_pair_candidates"] = len(yA_list)

            for b_idx, (A, B) in enumerate(zip(A_batches, B_batches)):
                batch_stat = {"batch_index": b_idx}
                yA = np.asarray(yA_list, int)
                yB = np.asarray(yB_list, int)

                with _tick("03_batch_false_position_flip"):
                    t_fp0 = perf_counter()
                    Xstar, ystar, Sstar = batch_false_position_flip(
                        self._adaptor, A, B, yA, yB,
                        iters=self.cfg.secant_iters, final_bisect=self.cfg.final_bisect
                    )
                    batch_stat["flip_ms"] = (perf_counter() - t_fp0) * 1000.0

                with _tick("04_scores_A"):
                    t_sa0 = perf_counter()
                    SA = self._adaptor.scores(A)
                    batch_stat["scoresA_ms"] = (perf_counter() - t_sa0) * 1000.0

                cfg = self.cfg
                w_swing = float(cfg.prob_swing_weight)
                use_jsd = bool(cfg.use_jsd)
                w_jsd   = float(cfg.jsd_weight) if use_jsd else 0.0

                t_build0 = perf_counter()
                with _tick("05_build_records"):
                    for k in range(len(A)):
                        t_rec0 = perf_counter()

                        x0 = A[k]; x1 = Xstar[k]
                        S0 = SA[k]; S1 = Sstar[k]
                        y0 = int(np.argmax(S0)); y1 = int(np.argmax(S1))
                        dvec = x1 - x0
                        dn2 = float(np.linalg.norm(dvec, 2))
                        dni = float(np.linalg.norm(dvec, np.inf))

                        m1 = float(S1[y0] - S1[y1])
                        logit_gain = float(
                            np.log((S1[y1] + 1e-12) / (S1[y0] + 1e-12))
                            - np.log((S0[y1] + 1e-12) / (S0[y0] + 1e-12))
                        )
                        robust = (y1 != y0) and (m1 <= -self.cfg.min_pair_margin_end) and (logit_gain >= self.cfg.min_logit_gain)

                        drop_a     = max(0.0, float(S0[y0] - S1[y0]))
                        gain_b     = max(0.0, float(S1[y1] - S0[y1]))
                        prob_swing = 0.5 * (drop_a + gain_b)

                        m0 = float(S0[y0] - S0[y1])
                        margin_gain = max(0.0, m0 - m1)
                        jsd_val = _jsd(S0, S1) if use_jsd else 0.0
                        strength_final_margin = max(0.0, -m1)
                        change_mix = (w_swing * prob_swing) + ((1.0 - w_swing) * strength_final_margin) + (w_jsd * jsd_val)

                        rec_time_ms = (perf_counter() - t_rec0) * 1000.0

                        rec = DeltaRecord(
                            index_a=-1, index_b=-1, method="pair_rr",
                            success=bool(robust), y0=y0, y1=y1,
                            delta_norm_l2=dn2, delta_norm_linf=dni,
                            score_change=float(change_mix), distance_term=0.0, change_term=0.0, final_score=0.0,
                            time_ms=float(rec_time_ms),
                            x0=x0, x1=x1, delta=dvec, S0=S0, S1=S1,
                            prob_swing=float(prob_swing), margin_gain=float(margin_gain), jsd_change=float(jsd_val)
                        )
                        all_records.append(rec)

                batch_stat["build_records_ms"] = (perf_counter() - t_build0) * 1000.0
                batch_stat["n_records_in_batch"] = len(A)
                per_batch_stats.append(batch_stat)

            counts["n_records_total"] = len(all_records)
            counts["n_success_total"] = int(sum(1 for r in all_records if r.success))

            with _tick("06_group_by_pair"):
                by_pair: Dict[Tuple[int, int], List[int]] = {}
                for idx, r in enumerate(all_records):
                    if r.success:
                        by_pair.setdefault((r.y0, r.y1), []).append(idx)

            with _tick("07_pairwise_norm_and_score"):
                for key, idxs in by_pair.items():
                    dists = np.array([
                        all_records[i].delta_norm_l2 if self.cfg.distance_metric == "l2"
                        else all_records[i].delta_norm_linf
                        for i in idxs
                    ], float)
                    changes = np.array([all_records[i].score_change for i in idxs], float)
                    d95 = np.percentile(dists, 95) + 1e-12
                    c95 = np.percentile(changes, 95) + 1e-12
                    d_term = np.clip(dists / d95, 0, 1)
                    c_term = np.clip(changes / c95, 0, 1)
                    alpha = float(self.cfg.alpha_change)
                    final = alpha * c_term + (1 - alpha) * (1 - d_term)
                    for ii, dt, ct, fs in zip(idxs, d_term, c_term, final):
                        all_records[ii].distance_term = float(dt)
                        all_records[ii].change_term = float(ct)
                        all_records[ii].final_score = float(fs)

            with _tick("08_sort_records"):
                self.records_ = sorted(all_records, key=lambda r: r.final_score, reverse=True)

            if self.cp_cfg.enabled:
                with _tick("09_compute_change_points_for_records"):
                    self._compute_change_points_for_records(model)
            else:
                timings["09_compute_change_points_for_records"] = 0.0

            with _tick("10_logging"):
                self.logger.info(
                    "DelDel listo. Pares=%d | tiempo_total=%.1f ms",
                    len(self.records_), (perf_counter() - t_total0) * 1000.0
                )

            timings["total_ms"] = (perf_counter() - t_total0) * 1000.0
            self.time_stats_ = {
                "timings_ms": timings,
                "events_ms": events,
                "counts": counts,
                "per_batch": per_batch_stats
            }
            return self


    def fit(self, X: np.ndarray, model: Any) -> "DelDel":
        from time import perf_counter
        from contextlib import contextmanager
        from collections import Counter  # <-- añadido para los conteos

        @contextmanager
        def _tick(name: str):
            t = perf_counter()
            try:
                yield
            finally:
                dt = (perf_counter() - t) * 1000.0
                timings[name] = timings.get(name, 0.0) + dt
                events.append((name, dt))

        timings: Dict[str, float] = {}
        events: List[Tuple[str, float]] = []
        counts: Dict[str, Any] = {}

        t_total0 = perf_counter()

        with _tick("00_setup"):
            self._X = np.asarray(X, float)
            self._adaptor = ScoreAdaptor(model, mode=self.cfg.mode)
            self.records_.clear()
            counts["n_samples"], counts["n_features"] = self._X.shape

        with _tick("01_scores_global"):
            self._P = self._adaptor.scores(self._X)
            self._y = np.argmax(self._P, axis=1)
            labels = sorted(np.unique(self._y).tolist())
            counts["n_labels"] = len(labels)
            print('labels', labels)

        with _tick("02_pair_candidates_round_robin"):
            A_batches, B_batches, yA_list, yB_list = self._pair_candidates_round_robin(labels)
            # print("pair_candidates_round_robin", A_batches)
            # print("pair_candidates_round_robin", B_batches)
            # print("pair_candidates_round_robin", yA_list)
            # print("pair_candidates_round_robin", yB_list)

        if not A_batches:
            _get_logger("DelDel", logging.WARNING).warning(
                "No se pudieron generar pares (verifica configuración y datos)."
            )
            self.time_stats_ = {
                "timings_ms": timings,
                "counts": counts,
                "per_batch": [],
                "per_pair": {},
                "notes": "Sin pares candidatos"
            }
            return self

        all_records: List[DeltaRecord] = []
        per_batch_stats: List[Dict[str, Any]] = []
        counts["n_pair_candidates"] = len(yA_list)
        #--------------------------------------------------------------------------------------------------------------------------------------------
        #--------------------------------------------------------------------------------------------------------------------------------------------


        for b_idx, (A, B) in enumerate(zip(A_batches, B_batches)):
            if b_idx==0:
                print("Ver A: ", A)
                print("Ver B: ", B)
                print("Ver yA: ", yA_list)
                print("Ver yB: ", yB_list)
            print(b_idx)
            batch_stat = {"batch_index": b_idx}
            yA = np.asarray(yA_list, int)
            yB = np.asarray(yB_list, int)

            with _tick("03_batch_false_position_flip"):
                t_fp0 = perf_counter()
                Xstar, ystar, Sstar = batch_false_position_flip(
                    self._adaptor, A, B, yA, yB,
                    iters=self.cfg.secant_iters, final_bisect=self.cfg.final_bisect
                )
                batch_stat["flip_ms"] = (perf_counter() - t_fp0) * 1000.0
                print("Ver Xstar: ", Xstar.shape, type(Xstar), Xstar)
                print("Ver ystar: ", ystar.shape, type(ystar), ystar)
                print("Ver Sstar: ", Sstar.shape, type(Sstar), Sstar)

            with _tick("04_scores_A"):
                t_sa0 = perf_counter()
                SA = self._adaptor.scores(A)
                batch_stat["scoresA_ms"] = (perf_counter() - t_sa0) * 1000.0

            cfg = self.cfg
            w_swing = float(cfg.prob_swing_weight)
            use_jsd = bool(cfg.use_jsd)
            w_jsd   = float(cfg.jsd_weight) if use_jsd else 0.0

            t_build0 = perf_counter()
            with _tick("05_build_records"):
                for k in range(len(A)):
                    t_rec0 = perf_counter()

                    x0 = A[k]; x1 = Xstar[k]
                    S0 = SA[k]; S1 = Sstar[k]
                    y0 = int(np.argmax(S0)); y1 = int(np.argmax(S1))
                    dvec = x1 - x0
                    dn2 = float(np.linalg.norm(dvec, 2))
                    dni = float(np.linalg.norm(dvec, np.inf))

                    m1 = float(S1[y0] - S1[y1])
                    logit_gain = float(
                        np.log((S1[y1] + 1e-12) / (S1[y0] + 1e-12))
                        - np.log((S0[y1] + 1e-12) / (S0[y0] + 1e-12))
                    )
                    robust = (y1 != y0) and (m1 <= -self.cfg.min_pair_margin_end) and (logit_gain >= self.cfg.min_logit_gain)

                    drop_a     = max(0.0, float(S0[y0] - S1[y0]))
                    gain_b     = max(0.0, float(S1[y1] - S0[y1]))
                    prob_swing = 0.5 * (drop_a + gain_b)

                    m0 = float(S0[y0] - S0[y1])
                    margin_gain = max(0.0, m0 - m1)
                    jsd_val = _jsd(S0, S1) if use_jsd else 0.0
                    strength_final_margin = max(0.0, -m1)
                    change_mix = (w_swing * prob_swing) + ((1.0 - w_swing) * strength_final_margin) + (w_jsd * jsd_val)

                    rec_time_ms = (perf_counter() - t_rec0) * 1000.0

                    rec = DeltaRecord(
                        index_a=-1, index_b=-1, method="pair_rr",
                        success=bool(robust), y0=y0, y1=y1,
                        delta_norm_l2=dn2, delta_norm_linf=dni,
                        score_change=float(change_mix), distance_term=0.0, change_term=0.0, final_score=0.0,
                        time_ms=float(rec_time_ms),
                        x0=x0, x1=x1, delta=dvec, S0=S0, S1=S1,
                        prob_swing=float(prob_swing), margin_gain=float(margin_gain), jsd_change=float(jsd_val)
                    )
                    all_records.append(rec)

            batch_stat["build_records_ms"] = (perf_counter() - t_build0) * 1000.0
            batch_stat["n_records_in_batch"] = len(A)
            per_batch_stats.append(batch_stat)

        # === LOG de pares tras 05_build_records (todos los records construidos) ===
        pairs_after_05 = dict(Counter((r.y0, r.y1) for r in all_records))
        counts["pair_counts_after_05_build_records"] = pairs_after_05
        print("pair_counts_after_05_build_records", pairs_after_05)

        counts["n_records_total"] = len(all_records)
        counts["n_success_total"] = int(sum(1 for r in all_records if r.success))

        with _tick("06_group_by_pair"):
            by_pair: Dict[Tuple[int, int], List[int]] = {}
            for idx, r in enumerate(all_records):
                if r.success:
                    by_pair.setdefault((r.y0, r.y1), []).append(idx)

        # === LOG de pares tras 06_group_by_pair (sólo exitosos) ===
        pairs_after_06 = {k: len(v) for k, v in by_pair.items()}
        counts["pair_counts_after_06_group_by_pair_success"] = pairs_after_06
        print("pair_counts_after_06_group_by_pair_success", pairs_after_06)

        with _tick("07_pairwise_norm_and_score"):
            for key, idxs in by_pair.items():
                dists = np.array([
                    all_records[i].delta_norm_l2 if self.cfg.distance_metric == "l2"
                    else all_records[i].delta_norm_linf
                    for i in idxs
                ], float)
                changes = np.array([all_records[i].score_change for i in idxs], float)
                d95 = np.percentile(dists, 95) + 1e-12
                c95 = np.percentile(changes, 95) + 1e-12
                d_term = np.clip(dists / d95, 0, 1)
                c_term = np.clip(changes / c95, 0, 1)
                alpha = float(self.cfg.alpha_change)
                final = alpha * c_term + (1 - alpha) * (1 - d_term)
                for ii, dt, ct, fs in zip(idxs, d_term, c_term, final):
                    all_records[ii].distance_term = float(dt)
                    all_records[ii].change_term = float(ct)
                    all_records[ii].final_score = float(fs)

        with _tick("08_sort_records"):
            self.records_ = sorted(all_records, key=lambda r: r.final_score, reverse=True)

        # === LOG de pares tras 08_sort_records (self.records_ ordenados) ===
        pairs_after_08 = dict(Counter((r.y0, r.y1) for r in self.records_))
        counts["pair_counts_after_08_sort_records"] = pairs_after_08
        print("pair_counts_after_08_sort_records", pairs_after_08)

        if self.cp_cfg.enabled:
            with _tick("09_compute_change_points_for_records"):
                self._compute_change_points_for_records(model)
        else:
            timings["09_compute_change_points_for_records"] = 0.0

        with _tick("10_logging"):
            self.logger.info(
                "DelDel listo. Pares=%d | tiempo_total=%.1f ms",
                len(self.records_), (perf_counter() - t_total0) * 1000.0
            )

        timings["total_ms"] = (perf_counter() - t_total0) * 1000.0
        self.time_stats_ = {
            "timings_ms": timings,
            "events_ms": events,
            "counts": counts,
            "per_batch": per_batch_stats
        }
        return self

    def _compute_change_points_for_records(self, model: Any) -> None:
        cp = self.cp_cfg
        recs = self.records_
        if cp.only_success:
            recs = [r for r in recs if r.success]
        if cp.topk_records is not None and cp.topk_records > 0:
            recs = recs[:cp.topk_records]

        adaptor = self._adaptor
        # Ronda 0: mids de la malla base para todos los records (generic)
        base_samples = int(getattr(cp, "base_samples", 64))
        max_bisect_iters = int(getattr(cp, "max_bisect_iters", 22))
        limit_points = getattr(cp, "per_record_max_points", None)

        pool = _BatchPoolCP(adaptor, getattr(self.cfg, "cache_decimals_stage09", 4))
        try:
            # Prepara mallas y empuja mids
            rec_info = []
            for rid, r in enumerate(recs):
                x0 = np.asarray(r.x0, float); x1 = np.asarray(r.x1, float)
                if np.allclose(x0, x1):
                    rec_info.append((rid, r, None, None, None)); continue
                t_grid = np.linspace(0.0, 1.0, base_samples + 1)
                mids = 0.5*(t_grid[:-1] + t_grid[1:])
                Xm = _points_on_segment(x0, x1, mids)
                for j, row in enumerate(Xm): pool.push(rid, "MID0", j, row)
                rec_info.append((rid, r, t_grid, mids, (x0, x1)))

            pool.flush()

            # Determina cambios y define intervalos [tL,tR] por record
            actives = {}
            for rid, r, t_grid, mids, ends in rec_info:
                if t_grid is None:
                    r.cp_t = np.empty(0, float); r.cp_x = np.empty((0, r.x0.size), float)
                    r.cp_y_left = np.empty(0, int); r.cp_y_right = np.empty(0, int); r.cp_count = 0
                    continue
                y_mid = np.argmax(np.vstack([pool.get(rid, "MID0", j) for j in range(mids.size)]), axis=1)
                change_idx = np.where(y_mid[1:] != y_mid[:-1])[0]
                if change_idx.size == 0:
                    r.cp_t = np.empty(0, float); r.cp_x = np.empty((0, r.x0.size), float)
                    r.cp_y_left = np.empty(0, int); r.cp_y_right = np.empty(0, int); r.cp_count = 0
                    continue
                if limit_points is not None and change_idx.size > limit_points:
                    keep = np.linspace(0, change_idx.size - 1, limit_points).round().astype(int)
                    change_idx = change_idx[np.unique(keep)]
                tB = t_grid
                tL = tB[change_idx]; tR = tB[change_idx + 1]
                yL = y_mid[change_idx]; yR = y_mid[change_idx + 1]
                actives[rid] = {"r": r, "ends": ends, "tL": tL, "tR": tR, "yL": yL, "yR": yR}

            # Rondas de bisección batched
            for _ in range(max_bisect_iters):
                push_count = 0
                for rid, info in actives.items():
                    x0, x1 = info["ends"]
                    tL, tR = info["tL"], info["tR"]
                    mids = 0.5*(tL + tR)
                    Xm = _points_on_segment(x0, x1, mids)
                    for j, row in enumerate(Xm):
                        pool.push(rid, "BIS", j, row)
                        push_count += 1
                if push_count == 0: break
                pool.flush()
                next_actives = {}
                for rid, info in actives.items():
                    x0, x1 = info["ends"]
                    tL, tR, yL, yR = info["tL"], info["tR"], info["yL"], info["yR"]
                    m = tL.size
                    yM = np.argmax(np.vstack([pool.get(rid, "BIS", j) for j in range(m)]), axis=1)
                    # Actualiza intervalos (bisección clásica por etiquetas)
                    keepL = (yM == yL)
                    keepR = ~keepL
                    tL2 = np.where(keepL, 0.5*(tL + tR), tL)
                    tR2 = np.where(keepR, 0.5*(tL + tR), tR)
                    yL2 = np.where(keepL, yM, yL)
                    yR2 = np.where(keepR, yM, yR)
                    # Criterio de parada: cuando ya no cambian (longitud mínima)
                    done = np.all(np.abs(tR2 - tL2) < 1e-6)
                    if not done:
                        next_actives[rid] = {"r": info["r"], "ends": (x0, x1), "tL": tL2, "tR": tR2, "yL": yL2, "yR": yR2}
                    else:
                        # guarda resultado aproximado
                        t_star = 0.5*(tL2 + tR2)
                        info["r"].cp_t = t_star.astype(float)
                        info["r"].cp_x = _points_on_segment(x0, x1, t_star).astype(float)
                        info["r"].cp_y_left = yL2.astype(int)
                        info["r"].cp_y_right = yR2.astype(int)
                        info["r"].cp_count = int(t_star.size)
                actives = next_actives

            # Finaliza los que queden activos (máx iters alcanzado)
            for rid, info in actives.items():
                x0, x1 = info["ends"]
                t_star = 0.5*(info["tL"] + info["tR"])
                info["r"].cp_t = t_star.astype(float)
                info["r"].cp_x = _points_on_segment(x0, x1, t_star).astype(float)
                info["r"].cp_y_left = info["yL"].astype(int)
                info["r"].cp_y_right = info["yR"].astype(int)
                info["r"].cp_count = int(t_star.size)
        finally:
            pool.close()


    # getters
    def results(self) -> List[DeltaRecord]: return self.records_
    def topk(self, k: int = 10) -> List[DeltaRecord]: return self.records_[:max(0,int(k))]
    def to_dicts(self) -> List[Dict[str, Any]]: return [asdict(r) for r in self.records_]
    @property
    def X_(self): return self._X
    @property
    def model_(self): return None if self._adaptor is None else self._adaptor.model

# =============================================================================
# frontier_from_deltas.py
# =============================================================================

@dataclass
class DeltaRecordLite:
    y0: int
    y1: int
    x0: np.ndarray
    x1: np.ndarray
    cp_x: np.ndarray  # (m,d) o (0,0)
    cp_count: int

def _unique_rows(a: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Deja filas únicas con tolerancia (euclídea)."""
    if a.size == 0:
        return a.reshape(0, a.shape[-1] if a.ndim==2 else 0)
    r = np.round(a / max(tol, 1e-12), 0).astype(np.int64)
    _, idx = np.unique(r, axis=0, return_index=True)
    return a[np.sort(idx)]

def _stack_or_empty(lst: List[np.ndarray]) -> np.ndarray:
    return np.vstack(lst) if len(lst) else np.empty((0, 0), float)

# ---- PCA 3D vía SVD ----
class PCA3D:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None  # (3, d)

    def fit(self, X: np.ndarray):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        Z = X - self.mean_
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        self.components_ = Vt[:3, :]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = X - self.mean_
        return Z @ self.components_.T

def build_sets_from_records(
    records: Iterable
) -> Tuple[Dict[int, np.ndarray], Dict[Tuple[int,int], np.ndarray], Dict[Tuple[int,int], np.ndarray]]:
    interior = defaultdict(list)
    frontier = defaultdict(list)
    directions = defaultdict(list)

    for r in records:
        y0, y1 = int(r.y0), int(r.y1)
        x0 = np.asarray(r.x0, float).reshape(-1)
        x1 = np.asarray(r.x1, float).reshape(-1)

        if y0 == y1:
            interior[y0].append(x0)
            continue

        if getattr(r, "cp_count", 0) and np.asarray(getattr(r, "cp_x")).size > 0:
            F = np.asarray(r.cp_x, float)  # (m,d)
        else:
            F = x1.reshape(1, -1)

        frontier[(y0, y1)].append(F)
        directions[(y0, y1)].append(F - x0.reshape(1, -1))

    interior_by_class = {c: _unique_rows(np.vstack(v)) for c, v in interior.items()}
    frontier_by_pair = {k: _unique_rows(np.vstack(v)) for k, v in frontier.items()}
    dir_by_pair = {k: np.vstack(v) for k, v in directions.items()}
    return interior_by_class, frontier_by_pair, dir_by_pair

def frontier_by_class(frontier_by_pair: Dict[Tuple[int,int], np.ndarray]) -> Dict[int, np.ndarray]:
    by_class = defaultdict(list)
    for (a, b), P in frontier_by_pair.items():
        by_class[a].append(P)
        by_class[b].append(P)
    return {c: _unique_rows(np.vstack(v)) for c, v in by_class.items() if len(v)}

def choose_dims_topvar(frontier_by_pair: Dict[Tuple[int,int], np.ndarray], k: int = 3) -> Tuple[List[int], np.ndarray]:
    if not frontier_by_pair:
        return list(range(k)), np.eye(k, dtype=float)
    all_frontier = np.vstack([P for P in frontier_by_pair.values() if P.size > 0])
    var = np.var(all_frontier, axis=0)
    dims = np.argsort(var)[-k:]
    return dims.tolist(), var

def project_all_to_3d(
    interior_by_class: Dict[int, np.ndarray],
    frontier_by_pair: Dict[Tuple[int,int], np.ndarray],
    method: str = "pca",
    dims: Optional[Tuple[int,int,int]] = None
):
    stacks = [v for v in interior_by_class.values()] + [v for v in frontier_by_pair.values()]
    Xall = np.vstack([v for v in stacks if v.size > 0]) if stacks else np.empty((0, 0))

    if method == "pca":
        pca = PCA3D().fit(Xall)
        proj = lambda A: pca.transform(A)
        info = {"method": "pca", "mean": pca.mean_, "components": pca.components_}
    elif method == "topvar":
        dims_auto, _ = choose_dims_topvar(frontier_by_pair, k=3)
        dims_use = tuple(dims_auto)
        proj = lambda A: A[:, dims_use]
        info = {"method": "topvar", "dims": dims_use}
    elif method == "dims":
        assert dims is not None and len(dims) == 3, "Para method='dims' debes pasar dims=(i,j,k)."
        dims_use = tuple(int(i) for i in dims)
        proj = lambda A: A[:, dims_use]
        info = {"method": "dims", "dims": dims_use}
    else:
        raise ValueError("method debe ser 'pca' | 'topvar' | 'dims'")

    interior3d = {c: proj(A) for c, A in interior_by_class.items()}
    frontier3d = {k: proj(A) for k, A in frontier_by_pair.items()}
    by_class3d = frontier_by_class(frontier3d)

    return {
        "interior3d": interior3d,
        "frontier3d_by_pair": frontier3d,
        "frontier3d_by_class": by_class3d,
        "projection_info": info,
    }

def convex_hull_faces(points3d: np.ndarray):
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        return None
    if points3d.shape[0] < 4:
        return None
    hull = ConvexHull(points3d)
    return hull.simplices  # (n_faces, 3)

# =============================================================================
# weights_and_weighted_fits.py
# =============================================================================

# --- 1) Construcción de puntos frontera + pesos desde records ---
def build_weighted_frontier(
    records: Iterable,
    prefer_cp: bool = True,
    success_only: bool = True,
    weight_map: str = "power",    # "power" | "sigmoid" | "softmax"
    gamma: float = 2.0,           # para power
    temp: float = 0.15,           # para softmax o sigmoide
    sigmoid_center: Optional[float] = None,  # si None -> mediana por par
    density_k: Optional[int] = 8, # None para desactivar corrección densidad
) -> Tuple[Dict[Tuple[int,int], np.ndarray], Dict[Tuple[int,int], np.ndarray], Dict[Tuple[int,int], np.ndarray]]:
    Ftmp, Btmp, Stmp = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        if success_only and not bool(getattr(r, "success", True)):
            continue
        a, b = int(r.y0), int(r.y1)
        if a == b:
            continue
        score = float(getattr(r, "final_score", 1.0))
        x0 = np.asarray(r.x0, float).reshape(1,-1)
        if prefer_cp and getattr(r, "cp_count", 0) and np.asarray(getattr(r, "cp_x")).size>0:
            F = np.asarray(r.cp_x, float)            # (m,d)
        else:
            F = np.asarray(r.x1, float).reshape(1,-1)# (1,d)
        m = F.shape[0]
        Ftmp[(a,b)].append(F)
        Btmp[(a,b)].append(np.repeat(x0, m, axis=0))
        Stmp[(a,b)].append(np.full(m, score, float))

    F_by = {k: np.vstack(v) for k,v in Ftmp.items()}
    B_by = {k: np.vstack(v) for k,v in Btmp.items()}
    S_by = {k: np.concatenate(v) for k,v in Stmp.items()}

    W_by = {}
    for k in F_by.keys():
        s = S_by[k].copy()
        if s.size == 0:
            W_by[k] = s; continue

        if weight_map == "power":
            w = np.clip(s, 0.0, 1.0)**float(gamma)
        elif weight_map == "sigmoid":
            cen = float(np.median(s)) if sigmoid_center is None else float(sigmoid_center)
            w = 1.0 / (1.0 + np.exp(-(s - cen)/(temp if temp>1e-9 else 1e-9)))
        elif weight_map == "softmax":
            t = max(temp, 1e-9)
            z = (s - s.max())/t
            w = np.exp(z)
        else:
            w = s  # identidad

        if density_k is not None and density_k > 0 and F_by[k].shape[0] > density_k:
            P = F_by[k]
            D = np.linalg.norm(P[:,None,:] - P[None,:,:], axis=2)
            idx = np.argpartition(D, kth=density_k, axis=1)[:, :density_k+1]
            dens = D[np.arange(D.shape[0])[:,None], idx].mean(axis=1) + 1e-9
            w = w / dens

        W_by[k] = w / (w.sum() + 1e-12)

    return F_by, B_by, W_by

# --- 2) TLS (plano) ponderado: PCA ponderado ---
def fit_tls_plane_weighted(F: np.ndarray, w: np.ndarray):
    F = np.asarray(F, float); w = np.asarray(w, float).reshape(-1)
    assert F.shape[0] >= 3 and F.shape[0] == w.size
    w = w / (w.sum() + 1e-12)
    mu = (w[:,None] * F).sum(axis=0)
    Z  = F - mu
    Sw = Z.T @ (w[:,None] * Z)
    evals, evecs = np.linalg.eigh(Sw)
    n = evecs[:, np.argmin(evals)]
    b = -float(n @ mu)
    return n.astype(float), float(b), mu.astype(float)

# --- 3) Cuádrica/cúbica ponderada ---
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
    C: float = 10.0
):
    def _standardize(X):
        mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd<1e-12]=1.0
        return (X-mu)/sd, mu, sd
    def _destandardize(Qz, rz, cz, mu, sd):
        D = np.diag(1.0/sd); Qx = D @ Qz @ D; r0 = D @ rz
        rx = r0 - 2.0*(Qx @ mu); cx = float(mu.T@Qx@mu - r0.T@mu + cz)
        return Qx, rx, cx
    def _unpack(theta, idx, d):
        Qz = np.zeros((d,d))
        for i in range(d): Qz[i,i] = theta[idx["diag"][i]]
        for k,(i,j) in enumerate(idx["pairs"]): Qz[i,j]=Qz[j,i]=0.5*theta[idx["off"][k]]
        rz = theta[idx["lin"][0]: idx["lin"][0]+d]; cz = theta[idx["c"]]
        return Qz, rz, float(cz)
    def _poly2(Z):
        n,d = Z.shape
        diag=[Z[:,i]**2 for i in range(d)]
        off=[]; pairs=[]
        for i in range(d):
            for j in range(i+1,d):
                off.append(2.0*Z[:,i]*Z[:,j]); pairs.append((i,j))
        lin=[Z[:,i] for i in range(d)]
        Phi = np.column_stack(diag+off+lin+[np.ones(n)])
        idx={"diag":list(range(d)),
             "off":list(range(d, d+len(off))),
             "lin":list(range(d+len(off), d+len(off)+d)),
             "c":d+len(off)+d, "pairs":pairs}
        return Phi, idx

    F_by, B_by, W_by = build_weighted_frontier(records, prefer_cp, success_only,
                                               weight_map, gamma, temp, None, density_k)
    models = {}
    for key, F in F_by.items():
        w = W_by[key]
        if F.shape[0] < 3:
            continue

        if mode == "svd":
            Z, mu, sd = _standardize(F)
            Phi, idx = _poly2(Z)
            sw = np.sqrt(w + 1e-12)
            Phi_w = Phi * sw[:,None]
            U,S,Vt = np.linalg.svd(Phi_w, full_matrices=False)
            theta = Vt[-1,:] / (np.linalg.norm(Vt[-1,:]) + 1e-12)
            d = F.shape[1]
            Qz, rz, cz = _unpack(theta, idx, d)
            Qx, rx, cx = _destandardize(Qz, rz, cz, mu, sd)
            models[key] = {"Q":Qx, "r":rx, "c":cx, "mode":"svd_w", "cond": S[-1]/(S[0]+1e-12), "weights": w}

        elif mode == "logistic":
            B = B_by[key]
            Udir = F - B
            Uu = Udir / (np.linalg.norm(Udir, axis=1, keepdims=True) + 1e-12)
            eps_r = eps * (0.5 + 0.5*(1.0 - (w / (w.max()+1e-12))))
            Xa = F + (eps_r[:,None]*Uu); Xb = F - (eps_r[:,None]*Uu)
            X = np.vstack([Xa, Xb])
            ybin = np.hstack([np.ones(Xa.shape[0], int), -np.ones(Xb.shape[0], int)])
            w_lr = np.hstack([w, w])

            Z, mu, sd = _standardize(X)
            Phi, idx = _poly2(Z)
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=C, penalty="l2", max_iter=800)
            clf.fit(Phi, ybin, sample_weight=w_lr)
            wcoef = clf.coef_.reshape(-1); b0 = clf.intercept_[0]
            theta = np.r_[wcoef, b0]
            d = X.shape[1]
            Qz, rz, cz = _unpack(theta, idx, d)
            Qx, rx, cx = _destandardize(Qz, rz, cz, mu, sd)
            models[key] = {"Q":Qx, "r":rx, "c":cx, "mode":"logistic_w", "weights": w}
        else:
            raise ValueError("mode debe ser 'svd' o 'logistic'")
    return models

def fit_cubic_from_records_weighted(
    records,
    *,
    prefer_cp: bool = True,
    success_only: bool = True,
    mode: str = "svd",          # 'svd' | 'logistic'
    weight_map: str = "power",  # 'power' | 'sigmoid' | 'softmax'
    gamma: float = 2.0,
    temp: float = 0.15,
    sigmoid_center: float = None,
    density_k: int = 8,
    eps: float = 1e-3,
    C: float = 5.0
):
    import numpy as np
    from collections import defaultdict

    def _unique_rows_tol(a: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        if a.size == 0:
            return a.reshape(0, a.shape[-1] if a.ndim==2 else 0)
        q = np.round(a / max(tol, 1e-12), 0).astype(np.int64)
        _, idx = np.unique(q, axis=0, return_index=True)
        return a[np.sort(idx)]

    def _stack_frontier(records, prefer_cp=True, success_only=True):
        F_by, B_by, S_by = defaultdict(list), defaultdict(list), defaultdict(list)
        d = None
        for r in records:
            if success_only and not bool(getattr(r, "success", True)):
                continue
            a, b = int(r.y0), int(r.y1)
            if a == b:
                continue
            score = float(getattr(r, "final_score", 1.0))
            x0 = np.asarray(r.x0, float).reshape(1,-1)
            if prefer_cp and getattr(r, "cp_count", 0) and np.asarray(getattr(r, "cp_x")).size>0:
                F = np.asarray(r.cp_x, float)
            else:
                F = np.asarray(r.x1, float).reshape(1,-1)
            if d is None: d = F.shape[1]
            m = F.shape[0]
            F_by[(a,b)].append(F)
            B_by[(a,b)].append(np.repeat(x0, m, axis=0))
            S_by[(a,b)].append(np.full(m, score, float))
        F_by = {k: _unique_rows_tol(np.vstack(v)) for k,v in F_by.items()}
        B_by = {k: np.vstack(v) for k,v in B_by.items()}
        S_by = {k: np.concatenate(v) for k,v in S_by.items()}
        return F_by, B_by, S_by, (0 if d is None else d)

    def _weights_from_scores_per_pair(scores: np.ndarray, P: np.ndarray):
        s = scores.copy().astype(float)
        if weight_map == "power":
            w = np.clip(s, 0.0, 1.0)**float(gamma)
        elif weight_map == "sigmoid":
            cen = float(np.median(s)) if sigmoid_center is None else float(sigmoid_center)
            w = 1.0 / (1.0 + np.exp(-(s - cen)/(temp if temp>1e-9 else 1e-9)))
        elif weight_map == "softmax":
            t = max(temp, 1e-9); z = (s - s.max())/t; w = np.exp(z)
        else:
            w = s
        if density_k and density_k > 0 and P.shape[0] > density_k:
            D = np.linalg.norm(P[:,None,:] - P[None,:,:], axis=2)
            idx = np.argpartition(D, kth=density_k, axis=1)[:, :density_k+1]
            dens = D[np.arange(D.shape[0])[:,None], idx].mean(axis=1) + 1e-9
            w = w / dens
        return w / (w.sum() + 1e-12)

    def _standardize(X: np.ndarray):
        mu = X.mean(axis=0)
        sd = X.std(axis=0); sd[sd<1e-12] = 1.0
        Z = (X - mu) / sd
        return Z, mu, sd

    def _poly3_features(Z: np.ndarray):
        Z = np.asarray(Z, float)
        n, d = Z.shape
        cols = []
        catalog = {"lin":[], "quad_diag":[], "quad_off":[], "cubic_diag":[], "cubic_mixed2":[], "cubic_tri":[]}
        for i in range(d):
            cols.append(Z[:,i]); catalog["lin"].append(("x", (i,)))
        for i in range(d):
            cols.append(Z[:,i]**2); catalog["quad_diag"].append(("x2", (i,)))
        for i in range(d):
            for j in range(i+1, d):
                cols.append(Z[:,i]*Z[:,j]); catalog["quad_off"].append(("xixj", (i,j)))
        for i in range(d):
            cols.append(Z[:,i]**3); catalog["cubic_diag"].append(("x3", (i,)))
        for i in range(d):
            for j in range(d):
                if i == j: continue
                cols.append((Z[:,i]**2)*Z[:,j]); catalog["cubic_mixed2"].append(("xi2xj", (i,j)))
        for i in range(d):
            for j in range(i+1, d):
                for k in range(j+1, d):
                    cols.append(Z[:,i]*Z[:,j]*Z[:,k]); catalog["cubic_tri"].append(("xixjxk", (i,j,k)))
        cols.append(np.ones(n))
        Phi = np.column_stack(cols)
        return Phi, catalog

    F_by, B_by, S_by, d = _stack_frontier(records, prefer_cp, success_only)
    out = {}
    if d == 0:
        return out

    for key, F in F_by.items():
        if F.shape[0] < 5:
            continue
        B = B_by[key]
        scores = S_by[key]
        w = _weights_from_scores_per_pair(scores, F)

        if mode == "svd":
            Z, mu, sd = _standardize(F)
            Phi, catalog = _poly3_features(Z)
            sw = np.sqrt(w + 1e-12)
            Phi_w = Phi * sw[:, None]
            U,S,Vt = np.linalg.svd(Phi_w, full_matrices=False)
            wvec = Vt[-1, :]
            wvec = wvec / (np.linalg.norm(wvec) + 1e-12)
            out[key] = {
                "w": wvec, "mu": mu, "sd": sd, "catalog": catalog,
                "mode": "svd_w", "cond": S[-1]/(S[0]+1e-12), "weights": w
            }

        elif mode == "logistic":
            Udir = F - B
            Uu = Udir / (np.linalg.norm(Udir, axis=1, keepdims=True) + 1e-12)
            eps_r = eps * (0.5 + 0.5*(1.0 - (w / (w.max()+1e-12))))
            Xa = F + (eps_r[:,None]*Uu); Xb = F - (eps_r[:,None]*Uu)
            X = np.vstack([Xa, Xb])
            ybin = np.hstack([np.ones(Xa.shape[0], dtype=int), np.zeros(Xb.shape[0], dtype=int)])
            w_lr = np.hstack([w, w])

            Z, mu, sd = _standardize(X)
            Phi, catalog = _poly3_features(Z)

            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=C, penalty="l2", fit_intercept=False, max_iter=1000)
            clf.fit(Phi, ybin, sample_weight=w_lr)
            wcoef = clf.coef_.reshape(-1)
            out[key] = {
                "w": wcoef, "mu": mu, "sd": sd, "catalog": catalog,
                "mode": "logistic_w", "weights": w
            }

        else:
            raise ValueError("mode debe ser 'svd' o 'logistic'")

    return out

# === 6) Utilidades para cúbicas: evaluación y cruces sobre segmento ===
def _poly3_features_eval(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, float)
    n, d = Z.shape
    cols = []
    for i in range(d): cols.append(Z[:, i])                 # lin
    for i in range(d): cols.append(Z[:, i] ** 2)            # quad diag
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(Z[:, i] * Z[:, j])                  # quad off
    for i in range(d): cols.append(Z[:, i] ** 3)            # cubic diag
    for i in range(d):
        for j in range(d):
            if i == j: continue
            cols.append((Z[:, i] ** 2) * Z[:, j])           # cubic mixed^2
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                cols.append(Z[:, i] * Z[:, j] * Z[:, k])    # cubic tri
    cols.append(np.ones(n))
    return np.column_stack(cols) if cols else np.ones((n, 1))

def _cubic_g_from_model(model: Dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    mu = np.asarray(model["mu"], float)
    sd = np.asarray(model["sd"], float)
    Z = (X - mu) / sd
    Phi = _poly3_features_eval(Z)
    w = np.asarray(model["w"], float).reshape(-1)
    return Phi @ w

def eval_cubic_models(
    models: Dict[Tuple[int, int], Dict],
    X: np.ndarray,
    keys: Optional[Iterable[Tuple[int, int]]] = None,
    return_proba_when_logistic: bool = True,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    X = np.asarray(X, float)
    pairs = list(models.keys()) if keys is None else list(keys)
    out = {}
    for key in pairs:
        m = models[key]
        g = _cubic_g_from_model(m, X)
        p = None
        if return_proba_when_logistic and m.get("mode") == "logistic_w":
            p = 1.0 / (1.0 + np.exp(-g))
        out[key] = {"g": g, "p": p, "mode": m.get("mode", "")}
    return out

def find_segment_crossings_cubic(
    models: Dict[Tuple[int, int], Dict],
    key: Tuple[int, int],
    x0: np.ndarray,
    x1: np.ndarray,
    *,
    n_sub: int = 128,
    near_zero: float = 1e-6,
    tol: float = 1e-7,
    max_iter: int = 50
) -> List[Dict]:
    assert key in models, f"Par {key} no encontrado en models."
    m = models[key]
    x0 = np.asarray(x0, float).reshape(-1)
    x1 = np.asarray(x1, float).reshape(-1)
    d = x0.size
    assert x1.size == d

    def gx_of_t(t: np.ndarray) -> np.ndarray:
        X = x0[None, :] + t[:, None] * (x1 - x0)[None, :]
        return _cubic_g_from_model(m, X)

    ts = np.linspace(0.0, 1.0, n_sub + 1)
    gs = gx_of_t(ts)

    brackets = []
    nz_hits = [float(t) for t, g in zip(ts, gs) if abs(g) <= near_zero]
    for i in range(n_sub):
        gL, gR = gs[i], gs[i + 1]
        if gL == 0.0 or gR == 0.0:
            continue
        if np.sign(gL) != np.sign(gR):
            brackets.append((ts[i], ts[i + 1]))

    roots_t: List[float] = []

    def _append_unique(tcand: float, bag: List[float], atol: float = 1e-6):
        for ti in bag:
            if abs(ti - tcand) <= atol:
                return
        bag.append(float(tcand))

    for t_hit in nz_hits:
        _append_unique(t_hit, roots_t)

    for (tL, tR) in brackets:
        gL, gR = gx_of_t(np.array([tL, tR]))
        if np.sign(gL) == np.sign(gR):
            continue
        a, b = float(tL), float(tR)
        fa, fb = float(gL), float(gR)
        for _ in range(max_iter):
            c = 0.5 * (a + b)
            fc = float(gx_of_t(np.array([c]))[0])
            if abs(fc) <= tol or (b - a) <= tol:
                _append_unique(c, roots_t)
                break
            if np.sign(fa) * np.sign(fc) <= 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        else:
            ca, cb = abs(fa), abs(fb)
            c = a if ca < cb else b
            _append_unique(c, roots_t)

    roots_t.sort()
    out = []
    for t in roots_t:
        x = x0 + t * (x1 - x0)
        g = float(_cubic_g_from_model(m, x.reshape(1, -1))[0])
        out.append({"t": t, "x": x, "g": g})
    return out



# --- 1) Construcción de puntos frontera + pesos desde records ---
def build_weighted_frontier(
    records: Iterable,
    prefer_cp: bool = True,
    success_only: bool = True,
    weight_map: str = "power",    # "power" | "sigmoid" | "softmax"
    gamma: float = 2.0,           # para power
    temp: float = 0.15,           # para softmax (por par) o sigmoide
    sigmoid_center: Optional[float] = None,  # si None -> mediana por par
    density_k: Optional[int] = 8, # None para desactivar corrección densidad
) -> Tuple[Dict[Tuple[int,int], np.ndarray], Dict[Tuple[int,int], np.ndarray], Dict[Tuple[int,int], np.ndarray]]:
    from collections import defaultdict

    Ftmp, Btmp, Stmp = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        if success_only and not bool(getattr(r, "success", True)):
            continue
        a, b = int(r.y0), int(r.y1)
        if a == b:
            continue
        score = float(getattr(r, "final_score", 1.0))
        x0 = np.asarray(r.x0, float).reshape(1,-1)
        if prefer_cp and getattr(r, "cp_count", 0) and np.asarray(getattr(r, "cp_x")).size>0:
            F = np.asarray(r.cp_x, float)            # (m,d)
        else:
            F = np.asarray(r.x1, float).reshape(1,-1)# (1,d)
        m = F.shape[0]
        Ftmp[(a,b)].append(F)
        Btmp[(a,b)].append(np.repeat(x0, m, axis=0))
        Stmp[(a,b)].append(np.full(m, score, float))

    F_by = {k: np.vstack(v) for k,v in Ftmp.items()}
    B_by = {k: np.vstack(v) for k,v in Btmp.items()}
    S_by = {k: np.concatenate(v) for k,v in Stmp.items()}

    W_by = {}
    for k in F_by.keys():
        s = S_by[k].copy()
        if s.size == 0:
            W_by[k] = s; continue

        if weight_map == "power":
            w = np.clip(s, 0.0, 1.0)**float(gamma)
        elif weight_map == "sigmoid":
            cen = float(np.median(s)) if sigmoid_center is None else float(sigmoid_center)
            w = 1.0 / (1.0 + np.exp(-(s - cen)/(temp if temp>1e-9 else 1e-9)))
        elif weight_map == "softmax":
            t = max(temp, 1e-9)
            z = (s - s.max())/t
            w = np.exp(z)
        else:
            w = s  # identidad

        if density_k is not None and density_k > 0 and F_by[k].shape[0] > density_k:
            P = F_by[k]
            D = np.linalg.norm(P[:,None,:] - P[None,:,:], axis=2)
            idx = np.argpartition(D, kth=density_k, axis=1)[:, :density_k+1]
            dens = D[np.arange(D.shape[0])[:,None], idx].mean(axis=1) + 1e-9
            w = w / dens

        W_by[k] = w / (w.sum() + 1e-12)

    return F_by, B_by, W_by

# --- 2) TLS (plano) ponderado: PCA ponderado ---
def fit_tls_plane_weighted(F: np.ndarray, w: np.ndarray):
    F = np.asarray(F, float); w = np.asarray(w, float).reshape(-1)
    assert F.shape[0] >= 3 and F.shape[0] == w.size
    w = w / (w.sum() + 1e-12)
    mu = (w[:,None] * F).sum(axis=0)
    Z  = F - mu
    Sw = Z.T @ (w[:,None] * Z)
    evals, evecs = np.linalg.eigh(Sw)
    n = evecs[:, np.argmin(evals)]
    b = -float(n @ mu)
    return n.astype(float), float(b), mu.astype(float)

# --- 3) Cuádrica (SVD algebraico) ponderada ---
def fit_quadric_svd_weighted(F: np.ndarray, poly2_features_fn):
    def _standardize(X):
        mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd<1e-12]=1.0
        Z = (X-mu)/sd
        return Z, mu, sd
    def _destandardize(Qz, rz, cz, mu, sd):
        D = np.diag(1.0/sd)
        Qx = D @ Qz @ D
        r0 = D @ rz
        rx = r0 - 2.0*(Qx @ mu)
        cx = float(mu.T @ Qx @ mu - r0.T @ mu + cz)
        return Qx, rx, cx
    def _unpack(theta, idx, d):
        Qz = np.zeros((d,d));
        for i in range(d): Qz[i,i] = theta[idx["diag"][i]]
        for k,(i,j) in enumerate(idx["pairs"]):
            Qz[i,j]=Qz[j,i]=0.5*theta[idx["off"][k]]
        rz = theta[idx["lin"][0]: idx["lin"][0]+d]
        cz = theta[idx["c"]]
        return Qz, rz, float(cz)

    Z, mu, sd = _standardize(F)
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
    eps_orient: float = 1e-3
):
    F_by, B_by, W_by = build_weighted_frontier(records, prefer_cp, success_only,
                                               weight_map, gamma, temp, None, density_k)
    planes = {}
    for key, F in F_by.items():
        w = W_by[key]
        if F.shape[0] < 3:
            continue
        n, b, mu = fit_tls_plane_weighted(F, w)
        if orient_with_bases and key in B_by:
            B = B_by[key]
            U = F - B
            U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
            side_plus  = (F + eps_orient*U) @ n + b
            side_minus = (F - eps_orient*U) @ n + b
            if np.average(side_plus, weights=w) < np.average(side_minus, weights=w):
                n, b = -n, -b
            b = -float(n @ mu)
        planes[key] = {"n": n, "b": b, "mu": mu, "count": int(F.shape[0]), "points": F, "weights": w}
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
    C: float = 10.0
):
    def _standardize(X):
        mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd<1e-12]=1.0
        return (X-mu)/sd, mu, sd
    def _destandardize(Qz, rz, cz, mu, sd):
        D = np.diag(1.0/sd); Qx = D @ Qz @ D; r0 = D @ rz
        rx = r0 - 2.0*(Qx @ mu); cx = float(mu.T@Qx@mu - r0.T@mu + cz)
        return Qx, rx, cx
    def _unpack(theta, idx, d):
        Qz = np.zeros((d,d))
        for i in range(d): Qz[i,i] = theta[idx["diag"][i]]
        for k,(i,j) in enumerate(idx["pairs"]): Qz[i,j]=Qz[j,i]=0.5*theta[idx["off"][k]]
        rz = theta[idx["lin"][0]: idx["lin"][0]+d]; cz = theta[idx["c"]]
        return Qz, rz, float(cz)
    def _poly2(Z):
        n,d = Z.shape
        diag=[Z[:,i]**2 for i in range(d)]
        off=[]; pairs=[]
        for i in range(d):
            for j in range(i+1,d):
                off.append(2.0*Z[:,i]*Z[:,j]); pairs.append((i,j))
        lin=[Z[:,i] for i in range(d)]
        Phi = np.column_stack(diag+off+lin+[np.ones(n)])
        idx={"diag":list(range(d)),
             "off":list(range(d, d+len(off))),
             "lin":list(range(d+len(off), d+len(off)+d)),
             "c":d+len(off)+d, "pairs":pairs}
        return Phi, idx

    F_by, B_by, W_by = build_weighted_frontier(records, prefer_cp, success_only,
                                               weight_map, gamma, temp, None, density_k)
    models = {}
    for key, F in F_by.items():
        w = W_by[key];
        if F.shape[0] < 3:
            continue

        if mode == "svd":
            Z, mu, sd = _standardize(F)
            Phi, idx = _poly2(Z)
            sw = np.sqrt(w + 1e-12)
            Phi_w = Phi * sw[:,None]
            U,S,Vt = np.linalg.svd(Phi_w, full_matrices=False)
            theta = Vt[-1,:] / (np.linalg.norm(Vt[-1,:]) + 1e-12)
            d = F.shape[1]
            Qz, rz, cz = _unpack(theta, idx, d)
            Qx, rx, cx = _destandardize(Qz, rz, cz, mu, sd)
            models[key] = {"Q":Qx, "r":rx, "c":cx, "mode":"svd_w", "cond": S[-1]/(S[0]+1e-12), "weights": w}

        elif mode == "logistic":
            B = B_by[key]
            Udir = F - B
            Uu = Udir / (np.linalg.norm(Udir, axis=1, keepdims=True) + 1e-12)
            eps_r = eps * (0.5 + 0.5*(1.0 - (w / (w.max()+1e-12))))  # en [0.5*eps, eps]
            Xa = F + (eps_r[:,None]*Uu); Xb = F - (eps_r[:,None]*Uu)
            X = np.vstack([Xa, Xb])
            ybin = np.hstack([np.ones(Xa.shape[0], int), -np.ones(Xb.shape[0], int)])
            w_lr = np.hstack([w, w])
            Z, mu, sd = _standardize(X)
            Phi, idx = _poly2(Z)
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=C, penalty="l2", max_iter=800)
            clf.fit(Phi, ybin, sample_weight=w_lr)
            wcoef = clf.coef_.reshape(-1); b0 = clf.intercept_[0]
            theta = np.r_[wcoef, b0]
            d = X.shape[1]
            Qz, rz, cz = _unpack(theta, idx, d)
            Qx, rx, cx = _destandardize(Qz, rz, cz, mu, sd)
            models[key] = {"Q":Qx, "r":rx, "c":cx, "mode":"logistic_w", "weights": w}
        else:
            raise ValueError("mode debe ser 'svd' o 'logistic'")
    return models


def fit_cubic_from_records_weighted(
    records,
    *,
    prefer_cp: bool = True,
    success_only: bool = True,
    mode: str = "svd",          # 'svd' | 'logistic'
    weight_map: str = "power",  # 'power' | 'sigmoid' | 'softmax'
    gamma: float = 2.0,
    temp: float = 0.15,
    sigmoid_center: float = None,
    density_k: int = 8,
    eps: float = 1e-3,
    C: float = 5.0
):
    import numpy as np
    from collections import defaultdict

    def _unique_rows_tol(a: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        if a.size == 0:
            return a.reshape(0, a.shape[-1] if a.ndim==2 else 0)
        q = np.round(a / max(tol, 1e-12), 0).astype(np.int64)
        _, idx = np.unique(q, axis=0, return_index=True)
        return a[np.sort(idx)]

    def _stack_frontier(records, prefer_cp=True, success_only=True):
        F_by, B_by, S_by = defaultdict(list), defaultdict(list), defaultdict(list)
        d = None
        for r in records:
            if success_only and not bool(getattr(r, "success", True)):
                continue
            a, b = int(r.y0), int(r.y1)
            if a == b:
                continue
            score = float(getattr(r, "final_score", 1.0))
            x0 = np.asarray(r.x0, float).reshape(1,-1)
            if prefer_cp and getattr(r, "cp_count", 0) and np.asarray(getattr(r, "cp_x")).size>0:
                F = np.asarray(r.cp_x, float)            # (m,d)
            else:
                F = np.asarray(r.x1, float).reshape(1,-1) # (1,d)
            if d is None: d = F.shape[1]
            m = F.shape[0]
            F_by[(a,b)].append(F)
            B_by[(a,b)].append(np.repeat(x0, m, axis=0))
            S_by[(a,b)].append(np.full(m, score, float))
        F_by = {k: _unique_rows_tol(np.vstack(v)) for k,v in F_by.items()}
        B_by = {k: np.vstack(v) for k,v in B_by.items()}
        S_by = {k: np.concatenate(v) for k,v in S_by.items()}
        return F_by, B_by, S_by, (0 if d is None else d)

    def _weights_from_scores_per_pair(scores: np.ndarray, P: np.ndarray):
        s = scores.copy().astype(float)
        if weight_map == "power":
            w = np.clip(s, 0.0, 1.0)**float(gamma)
        elif weight_map == "sigmoid":
            cen = float(np.median(s)) if sigmoid_center is None else float(sigmoid_center)
            w = 1.0 / (1.0 + np.exp(-(s - cen)/(temp if temp>1e-9 else 1e-9)))
        elif weight_map == "softmax":
            t = max(temp, 1e-9); z = (s - s.max())/t; w = np.exp(z)
        else:
            w = s
        if density_k and density_k > 0 and P.shape[0] > density_k:
            D = np.linalg.norm(P[:,None,:] - P[None,:,:], axis=2)
            idx = np.argpartition(D, kth=density_k, axis=1)[:, :density_k+1]
            dens = D[np.arange(D.shape[0])[:,None], idx].mean(axis=1) + 1e-9
            w = w / dens
        return w / (w.sum() + 1e-12)

    def _standardize(X: np.ndarray):
        mu = X.mean(axis=0)
        sd = X.std(axis=0); sd[sd<1e-12] = 1.0
        Z = (X - mu) / sd
        return Z, mu, sd

    def _poly3_features(Z: np.ndarray):
        Z = np.asarray(Z, float)
        n, d = Z.shape
        cols = []
        catalog = {"lin":[], "quad_diag":[], "quad_off":[], "cubic_diag":[], "cubic_mixed2":[], "cubic_tri":[]}
        for i in range(d):
            cols.append(Z[:,i]); catalog["lin"].append(("x", (i,)))
        for i in range(d):
            cols.append(Z[:,i]**2); catalog["quad_diag"].append(("x2", (i,)))
        for i in range(d):
            for j in range(i+1, d):
                cols.append(Z[:,i]*Z[:,j]); catalog["quad_off"].append(("xixj", (i,j)))
        for i in range(d):
            cols.append(Z[:,i]**3); catalog["cubic_diag"].append(("x3", (i,)))
        for i in range(d):
            for j in range(d):
                if i == j: continue
                cols.append((Z[:,i]**2)*Z[:,j]); catalog["cubic_mixed2"].append(("xi2xj", (i,j)))
        for i in range(d):
            for j in range(i+1, d):
                for k in range(j+1, d):
                    cols.append(Z[:,i]*Z[:,j]*Z[:,k]); catalog["cubic_tri"].append(("xixjxk", (i,j,k)))
        cols.append(np.ones(n))
        Phi = np.column_stack(cols)
        return Phi, catalog

    F_by, B_by, S_by, d = _stack_frontier(records, prefer_cp, success_only)
    out = {}
    if d == 0:
        return out

    for key, F in F_by.items():
        if F.shape[0] < 5:
            continue
        B = B_by[key]
        scores = S_by[key]
        w = _weights_from_scores_per_pair(scores, F)

        if mode == "svd":
            Z, mu, sd = _standardize(F)
            Phi, catalog = _poly3_features(Z)
            sw = np.sqrt(w + 1e-12)
            Phi_w = Phi * sw[:, None]
            U,S,Vt = np.linalg.svd(Phi_w, full_matrices=False)
            wvec = Vt[-1, :]
            wvec = wvec / (np.linalg.norm(wvec) + 1e-12)
            out[key] = {
                "w": wvec, "mu": mu, "sd": sd, "catalog": catalog,
                "mode": "svd_w", "cond": S[-1]/(S[0]+1e-12), "weights": w
            }

        elif mode == "logistic":
            Udir = F - B
            Uu = Udir / (np.linalg.norm(Udir, axis=1, keepdims=True) + 1e-12)
            eps_r = eps * (0.5 + 0.5*(1.0 - (w / (w.max()+1e-12))))
            Xa = F + (eps_r[:,None]*Uu); Xb = F - (eps_r[:,None]*Uu)
            X = np.vstack([Xa, Xb])
            ybin = np.hstack([np.ones(Xa.shape[0], dtype=int), np.zeros(Xb.shape[0], dtype=int)])
            w_lr = np.hstack([w, w])

            Z, mu, sd = _standardize(X)
            Phi, catalog = _poly3_features(Z)

            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(
                C=C, penalty="l2", fit_intercept=False, max_iter=1000
            )
            clf.fit(Phi, ybin, sample_weight=w_lr)
            wcoef = clf.coef_.reshape(-1)
            out[key] = {
                "w": wcoef, "mu": mu, "sd": sd, "catalog": catalog,
                "mode": "logistic_w", "weights": w
            }

        else:
            raise ValueError("mode debe ser 'svd' o 'logistic'")

    return out


import copy
import numpy as np
from typing import Iterable, List, Optional, Union

def normalize_pair_order(
    records: Iterable,
    *,
    make_copy: bool = True,
    class_: Optional[Union[int, List[int], tuple, set]] = None,
    limit_: bool = False,
):
    """
    Normaliza y asegura y0 <= y1 en cada DeltaRecord, con filtrado por clases y
    (opcionalmente) colapso de clases fuera del conjunto indicado.

    Novedades
    ---------
    class : int | list[int] | tuple[int] | set[int] | None
        Si se provee, devolverá únicamente los records donde **alguna** de estas
        clases aparezca en y0 o y1. Acepta escalar o colección.
    limit : bool
        Si True y 'class' fue provisto, entonces *toda* clase que NO esté en el
        conjunto 'class' será mapeada a -999 (incluye y0/y1 y también
        cp_y_left/cp_y_right si existen). Si 'class' es None, 'limit' se ignora.

    Comportamiento base
    -------------------
    - Si y0 > y1: intercambia y0<->y1 y x0<->x1, y actualiza:
        * delta := x1 - x0
        * S0 <-> S1 (si existen)
        * cp_y_left <-> cp_y_right (si existen)
      (cp_x y cp_t permanecen tal cual)

    Parámetros
    ----------
    records : Iterable[DeltaRecord]
    make_copy : bool
        Si True (por defecto), devuelve copias; si False, modifica in-place.

    Devuelve
    --------
    list[DeltaRecord]
    """

    # --- Normaliza la entrada 'class' a un conjunto de enteros (o None)
    if class_ is None:
        class_set = None
    else:
        if isinstance(class_, (list, tuple, set)):
            class_set = {int(c) for c in class_}
        else:
            class_set = {int(class_)}

    out = []

    for r in records:
        rr = copy.deepcopy(r) if make_copy else r

        # Etiquetas robustas a np.int_
        y0_orig = int(rr.y0)
        y1_orig = int(rr.y1)

        # --- Filtro por clases: conserva si hay intersección con {y0,y1}
        if class_set is not None:
            if (y0_orig not in class_set) and (y1_orig not in class_set):
                # No coincide con ninguna clase objetivo; descartar
                continue

        # --- Si limit=True y class_set existe: colapsa todo lo fuera a -999
        if limit_ and class_set is not None:
            y0_new = y0_orig if y0_orig in class_set else -999
            y1_new = y1_orig if y1_orig in class_set else -999
            rr.y0, rr.y1 = int(y0_new), int(y1_new)

            # También mapea cp_y_left / cp_y_right si existen
            if hasattr(rr, "cp_y_left") and rr.cp_y_left is not None:
                cpl = np.asarray(rr.cp_y_left)
                # tolera dtype mixtos; fuerza a int donde aplica
                rr.cp_y_left = np.where(np.isin(cpl, list(class_set)), cpl, -999).astype(int)

            if hasattr(rr, "cp_y_right") and rr.cp_y_right is not None:
                cpr = np.asarray(rr.cp_y_right)
                rr.cp_y_right = np.where(np.isin(cpr, list(class_set)), cpr, -999).astype(int)
        else:
            # Sin colapso: conserva y0,y1 originales
            rr.y0, rr.y1 = y0_orig, y1_orig

        # --- Asegura orden (y0 <= y1) con swaps coherentes
        y0 = int(rr.y0); y1 = int(rr.y1)
        if y0 > y1:
            # 1) Swap etiquetas
            rr.y0, rr.y1 = y1, y0

            # 2) Swap bases y puntos destino
            x0_old = np.asarray(rr.x0, float).copy()
            x1_old = np.asarray(rr.x1, float).copy()
            rr.x0, rr.x1 = x1_old, x0_old

            # 3) Delta coherente
            try:
                rr.delta = rr.x1 - rr.x0
            except Exception:
                pass

            # 4) Probabilidades/scores en cada extremo (si existen)
            if hasattr(rr, "S0") and hasattr(rr, "S1"):
                S0_old = np.asarray(rr.S0).copy()
                S1_old = np.asarray(rr.S1).copy()
                rr.S0, rr.S1 = S1_old, S0_old

            # 5) Puntos de cambio: solo se intercambian los lados (izq/der)
            if hasattr(rr, "cp_y_left") and hasattr(rr, "cp_y_right"):
                yl = np.asarray(rr.cp_y_left).copy()
                yr = np.asarray(rr.cp_y_right).copy()
                rr.cp_y_left, rr.cp_y_right = yr, yl
            # (cp_x y cp_t permanecen tal cual)

        out.append(rr)

    return out

