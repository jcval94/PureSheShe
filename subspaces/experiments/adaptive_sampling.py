"""Funciones y estructuras para el muestreo adaptativo del bundle."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from deldel.engine import DeltaRecord
else:
    DeltaRecord = Any

__all__ = [
    "AdaptiveSamplingInfo",
    "maybe_adaptive_sample",
    "adaptive_sample_size",
    "infer_sample_dimensions",
    "take_rows",
]


@dataclass(frozen=True)
class AdaptiveSamplingInfo:
    """Resumen de la decisión de muestreo adaptativo."""

    sampling_enabled: bool
    sampled_size: int
    original_size: int
    total_size: int
    n_features: int

    @property
    def sample_fraction(self) -> float:
        """Fracción del total utilizada tras el muestreo."""

        if self.original_size <= 0:
            return 0.0
        return self.sampled_size / float(self.original_size)


def take_rows(data, indices: np.ndarray):
    """Selecciona filas de ``data`` de acuerdo con ``indices``."""

    if data is None:
        return None
    if hasattr(data, "iloc"):
        return data.iloc[indices]
    if isinstance(data, np.ndarray):
        return data[indices]
    if isinstance(data, list):
        return [data[int(i)] for i in indices]
    if isinstance(data, tuple):
        return tuple(data[int(i)] for i in indices)
    try:
        return data[indices]  # type: ignore[index]
    except (TypeError, KeyError):
        return [data[int(i)] for i in indices]


def infer_sample_dimensions(X, n_samples: int) -> Tuple[int, int, int]:
    """Obtiene tamaño total, número de características y filas×columnas."""

    shape = getattr(X, "shape", None)
    if shape is not None:
        try:
            dims = []
            for dim in shape:
                if dim is None:
                    raise ValueError
                dims.append(int(dim))
            if dims:
                total_size = int(np.prod(dims))
                if len(dims) > 1:
                    n_features = int(np.prod(dims[1:]))
                else:
                    n_features = 1
                row_col_product = int(dims[0] * n_features)
                return total_size, n_features, row_col_product
        except (TypeError, ValueError, OverflowError):
            pass

    first_row: Optional[object] = None
    if hasattr(X, "iloc"):
        try:
            first_row = X.iloc[0]
        except Exception:
            first_row = None
    if first_row is None and hasattr(X, "__getitem__"):
        try:
            first_row = X[0]
        except Exception:
            first_row = None

    if first_row is not None:
        try:
            n_features = int(len(first_row))
        except TypeError:
            n_features = 1
    else:
        n_features = 1

    adjusted = max(1, n_features)
    total_size = int(n_samples * adjusted)
    row_col_product = int(n_samples * adjusted)
    return total_size, adjusted, row_col_product


def adaptive_sample_size(n_samples: int, n_features: int, total_size: int) -> int:
    """Calcula el tamaño de muestra siguiendo la heurística adaptativa."""

    adjusted_features = max(1, n_features)
    feature_factor = (adjusted_features / 8.0) ** 0.3
    raw_size = (n_samples ** 0.6) * feature_factor
    sample_size = max(500, min(n_samples, int(math.ceil(raw_size))))

    scaled_sample = sample_size * 3
    lower_bound = total_size * 0.001
    upper_bound = total_size * 0.05

    if lower_bound <= scaled_sample <= upper_bound:
        sample_size = int(min(n_samples, max(sample_size, math.ceil(scaled_sample))))
    return int(sample_size)


def maybe_adaptive_sample(
    X,
    y,
    records: Sequence["DeltaRecord"],
    *,
    adaptive_sampling: bool,
    random_state: Optional[int],
) -> Tuple[object, object, Sequence["DeltaRecord"], AdaptiveSamplingInfo]:
    """Aplica la heurística de muestreo adaptativo según corresponda."""

    n_samples = len(y)
    total_size, n_features, row_col_product = infer_sample_dimensions(X, n_samples)

    if (
        not adaptive_sampling
        or n_samples <= 100
        or row_col_product <= 500
    ):
        return (
            X,
            y,
            records,
            AdaptiveSamplingInfo(
                sampling_enabled=False,
                sampled_size=n_samples,
                original_size=n_samples,
                total_size=total_size,
                n_features=n_features,
            ),
        )

    sample_size = adaptive_sample_size(n_samples, n_features, total_size)
    if sample_size >= n_samples:
        return (
            X,
            y,
            records,
            AdaptiveSamplingInfo(
                sampling_enabled=False,
                sampled_size=n_samples,
                original_size=n_samples,
                total_size=total_size,
                n_features=n_features,
            ),
        )

    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_samples, size=sample_size, replace=False)
    indices.sort()

    sampled_X = take_rows(X, indices)
    sampled_y = take_rows(y, indices)

    if records and len(records) == n_samples:
        sampled_records = take_rows(records, indices)
    else:
        sampled_records = records

    info = AdaptiveSamplingInfo(
        sampling_enabled=True,
        sampled_size=int(sample_size),
        original_size=n_samples,
        total_size=total_size,
        n_features=n_features,
    )
    return sampled_X, sampled_y, sampled_records, info
