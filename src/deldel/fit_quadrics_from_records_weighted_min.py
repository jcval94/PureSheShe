# fit_quadrics_from_records_weighted_min.py
# Módulo mínimo para ajustar cuadráticas ponderadas a partir de "records".
# Solo contiene lo necesario para la definición de
# `fit_quadrics_from_records_weighted` y su helper `build_weighted_frontier`.

from typing import Iterable, Optional, Tuple, Dict
from collections import defaultdict
import logging
from time import perf_counter

import numpy as np

from deldel.utils import _destandardize, _standardize, _unpack


logger = logging.getLogger(__name__)


def _vlog(verbosity: int, threshold: int, message: str, **kwargs) -> None:
    if verbosity >= threshold:
        logger.info(message + (" | " + ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""))


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
    verbosity: int = 0,
) -> Tuple[Dict[Tuple[int, int], np.ndarray],
           Dict[Tuple[int, int], np.ndarray],
           Dict[Tuple[int, int], np.ndarray]]:
    """
    A partir de una colección de `records` construye:
    - F_by[(y0,y1)]: puntos frontera (F)
    - B_by[(y0,y1)]: puntos base (x0) alineados con F
    - W_by[(y0,y1)]: pesos normalizados por punto

    Cada `record` se asume con atributos:
    - y0, y1: clases (enteros)
    - success (bool, opcional)
    - final_score (float, opcional)
    - x0, x1: np.ndarray de dimensión d
    - cp_count, cp_x (opcional) para puntos de cambio precomputados.
    """

    t0 = perf_counter()
    _vlog(
        verbosity,
        1,
        "build_weighted_frontier:start",
        prefer_cp=prefer_cp,
        success_only=success_only,
        weight_map=weight_map,
    )

    Ftmp, Btmp, Stmp = defaultdict(list), defaultdict(list), defaultdict(list)

    for idx, r in enumerate(records):
        # Filtrar por éxito si se solicita
        if success_only and not bool(getattr(r, "success", True)):
            _vlog(verbosity, 3, "build_weighted_frontier:skip_failure", index=idx)
            continue

        a, b = int(r.y0), int(r.y1)
        if a == b:
            _vlog(verbosity, 3, "build_weighted_frontier:skip_same_class", index=idx, cls=a)
            continue

        score = float(getattr(r, "final_score", 1.0))
        x0 = np.asarray(r.x0, float).reshape(1, -1)

        try:
            if prefer_cp and getattr(r, "cp_count", 0) and np.asarray(getattr(r, "cp_x")).size > 0:
                F = np.asarray(r.cp_x, float)             # (m, d)
            else:
                F = np.asarray(r.x1, float).reshape(1, -1)  # (1, d)
        except Exception as exc:  # pragma: no cover - logging path
            _vlog(verbosity, 1, "build_weighted_frontier:error_extract", index=idx, err=exc)
            continue

        m = F.shape[0]
        Ftmp[(a, b)].append(F)
        Btmp[(a, b)].append(np.repeat(x0, m, axis=0))
        Stmp[(a, b)].append(np.full(m, score, float))

    # Apilar por par de clases
    F_by = {k: np.vstack(v) for k, v in Ftmp.items()}
    B_by = {k: np.vstack(v) for k, v in Btmp.items()}
    S_by = {k: np.concatenate(v) for k, v in Stmp.items()}

    # Construir pesos normalizados
    W_by: Dict[Tuple[int, int], np.ndarray] = {}
    for k in F_by.keys():
        s = S_by[k].copy().astype(float)
        if s.size == 0:
            W_by[k] = s
            continue

        if weight_map == "power":
            w = np.clip(s, 0.0, 1.0) ** float(gamma)
        elif weight_map == "sigmoid":
            cen = float(np.median(s)) if sigmoid_center is None else float(sigmoid_center)
            denom = temp if temp > 1e-9 else 1e-9
            w = 1.0 / (1.0 + np.exp(-(s - cen) / denom))
        elif weight_map == "softmax":
            t = max(temp, 1e-9)
            z = (s - s.max()) / t
            w = np.exp(z)
        else:
            # identidad
            w = s

        # Corrección simple por densidad local (k-NN en el propio par)
        if density_k is not None and density_k > 0 and F_by[k].shape[0] > density_k:
            P = F_by[k]
            D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
            idx = np.argpartition(D, kth=density_k, axis=1)[:, : density_k + 1]
            dens = D[np.arange(D.shape[0])[:, None], idx].mean(axis=1) + 1e-9
            w = w / dens

        W_by[k] = w / (w.sum() + 1e-12)

    _vlog(
        verbosity,
        1,
        "build_weighted_frontier:done",
        n_pairs=len(F_by),
        total_points=sum(v.shape[0] for v in F_by.values()) if F_by else 0,
        elapsed_s=round(perf_counter() - t0, 4),
    )
    return F_by, B_by, W_by


# --- 2) Ajuste de cuádricas ponderadas a partir de records ---
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
    verbosity: int = 0,
) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """
    Ajusta superficies cuadráticas ponderadas por cada par de clases (y0, y1).

    Devuelve un dict:
        models[(y0,y1)] = {
            "Q":  (d,d) np.ndarray simétrica,
            "r":  (d,)   np.ndarray,
            "c":  float,
            "mode": "svd_w" | "logistic_w",
            "cond": float (solo en modo 'svd'),
            "weights": w (pesos usados)
        }
    """

    def _poly2(Z: np.ndarray):
        n, d = Z.shape
        diag = [Z[:, i] ** 2 for i in range(d)]
        off = []
        pairs = []
        for i in range(d):
            for j in range(i + 1, d):
                off.append(2.0 * Z[:, i] * Z[:, j])
                pairs.append((i, j))
        lin = [Z[:, i] for i in range(d)]
        Phi = np.column_stack(diag + off + lin + [np.ones(n)])
        idx = {
            "diag": list(range(d)),
            "off": list(range(d, d + len(off))),
            "lin": list(range(d + len(off), d + len(off) + d)),
            "c": d + len(off) + d,
            "pairs": pairs,
        }
        return Phi, idx

    t0_global = perf_counter()
    _vlog(verbosity, 1, "fit_quadrics_from_records_weighted:start", mode=mode, weight_map=weight_map)

    # Construir frontera y pesos
    F_by, B_by, W_by = build_weighted_frontier(
        records,
        prefer_cp=prefer_cp,
        success_only=success_only,
        weight_map=weight_map,
        gamma=gamma,
        temp=temp,
        sigmoid_center=None,
        density_k=density_k,
        verbosity=verbosity,
    )

    def _fit_single(item):
        key, F = item
        pair_start = perf_counter()
        w = W_by[key]
        if F.shape[0] < 3:
            _vlog(verbosity, 2, "fit_quadrics:skip_small", pair=key, points=F.shape[0])
            return key, None

        cond_val = float("nan")

        if mode == "svd":
            # Ajuste algebraico mínimo (TLS) ponderado vía SVD
            Z, mu, sd = _standardize(F)
            Phi, idx = _poly2(Z)
            sw = np.sqrt(w + 1e-12)
            Phi_w = Phi * sw[:, None]

            U, S, Vt = np.linalg.svd(Phi_w, full_matrices=False)
            theta = Vt[-1, :] / (np.linalg.norm(Vt[-1, :]) + 1e-12)

            d = F.shape[1]
            Qz, rz, cz = _unpack(theta, idx, d)
            Qx, rx, cx = _destandardize(Qz, rz, cz, mu, sd)
            cond_val = float(S[-1] / (S[0] + 1e-12))

            model = {
                "Q": Qx,
                "r": rx,
                "c": cx,
                "mode": "svd_w",
                "cond": cond_val,
                "weights": w,
            }

        elif mode == "logistic":
            # Clasificador cuadrático (regresión logística) sobre muestras +/- epsilon
            from sklearn.linear_model import LogisticRegression

            B = B_by[key]
            Udir = F - B
            Uu = Udir / (np.linalg.norm(Udir, axis=1, keepdims=True) + 1e-12)

            # Radios un poco más pequeños para puntos muy pesados
            eps_r = eps * (0.5 + 0.5 * (1.0 - (w / (w.max() + 1e-12))))  # en [0.5*eps, eps]

            Xa = F + (eps_r[:, None] * Uu)
            Xb = F - (eps_r[:, None] * Uu)
            X = np.vstack([Xa, Xb])
            ybin = np.hstack(
                [np.ones(Xa.shape[0], dtype=int), -np.ones(Xb.shape[0], dtype=int)]
            )
            w_lr = np.hstack([w, w])

            Z, mu, sd = _standardize(X)
            Phi, idx = _poly2(Z)

            clf = LogisticRegression(C=C, penalty="l2", max_iter=800)
            clf.fit(Phi, ybin, sample_weight=w_lr)

            wcoef = clf.coef_.reshape(-1)
            b0 = clf.intercept_[0]
            theta = np.r_[wcoef, b0]

            d = X.shape[1]
            Qz, rz, cz = _unpack(theta, idx, d)
            Qx, rx, cx = _destandardize(Qz, rz, cz, mu, sd)

            model = {
                "Q": Qx,
                "r": rx,
                "c": cx,
                "mode": "logistic_w",
                "cond": cond_val,
                "weights": w,
            }
        else:
            _vlog(verbosity, 0, "fit_quadrics:error_mode", mode=mode)
            raise ValueError("mode debe ser 'svd' o 'logistic'")

        _vlog(
            verbosity,
            1,
            "fit_quadrics:pair_done",
            pair=key,
            mode=mode,
            cond=cond_val,
            elapsed_s=round(perf_counter() - pair_start, 6),
        )
        return key, model

    items = list(F_by.items())
    if n_jobs is not None and len(items) > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_single)(item) for item in items
        )
    else:
        results = [_fit_single(item) for item in items]

    models: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {k: v for k, v in results if v is not None}

    _vlog(
        verbosity,
        1,
        "fit_quadrics_from_records_weighted:done",
        total_pairs=len(models),
        elapsed_s=round(perf_counter() - t0_global, 4),
    )
    return models
