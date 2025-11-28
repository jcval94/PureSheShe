# fit_quadrics_from_records_weighted_min.py
# Módulo mínimo para ajustar cuadráticas ponderadas a partir de "records".
# Ahora delega en las implementaciones centrales de engine.py para evitar
# duplicación de lógica.

from typing import Iterable, Optional, Tuple, Dict

import logging
from time import perf_counter
import numpy as np

from .engine import (
    build_weighted_frontier as _core_build_weighted_frontier,
    fit_quadrics_from_records_weighted as _core_fit_quadrics_from_records_weighted,
)

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
    t0 = perf_counter()
    _vlog(
        verbosity,
        1,
        "build_weighted_frontier:start",
        prefer_cp=prefer_cp,
        success_only=success_only,
        weight_map=weight_map,
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
    t0_global = perf_counter()
    _vlog(verbosity, 1, "fit_quadrics_from_records_weighted:start", mode=mode, weight_map=weight_map)

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

    _vlog(
        verbosity,
        1,
        "fit_quadrics_from_records_weighted:done",
        total_pairs=len(models),
        elapsed_s=round(perf_counter() - t0_global, 4),
    )
    return models
