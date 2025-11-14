"""Utilidades para ejecutar el bundle de métodos destacados."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from deldel.engine import DeltaRecord
from deldel.subspace_change_detector import MultiClassSubspaceExplorer, SubspaceReport


CORE_METHOD_KEYS: Tuple[str, ...] = (
    "method_8_extratrees",  # ExtraTrees shallow routes
    "method_10b_gradient_hessian",  # Gradient-Hessian synergy (mejorado)
    "method_1b_topk_guided",  # Top-k guided combinations (mejorado)
    "method_10_gradient_synergy",  # Gradient synergy matrix (baseline)
    "method_6b_sparse_proj_guided",  # Sparse projections Fisher-guided (mejorado)
)


@dataclass
class CoreBundleResult:
    """Resultado compacto del bundle de métodos."""

    explorer: MultiClassSubspaceExplorer
    reports: Sequence[SubspaceReport]


def _create_bundle_explorer(
    *,
    max_sets: int,
    combo_sizes: Sequence[int],
    random_state: Optional[int],
    cv_splits: int,
    enabled_methods: Sequence[str],
) -> MultiClassSubspaceExplorer:
    return MultiClassSubspaceExplorer(
        max_sets=max_sets,
        combo_sizes=tuple(combo_sizes),
        random_state=random_state,
        cv_splits=cv_splits,
        filter_top_k=24,
        chi2_pool=28,
        random_samples=90,
        corr_threshold=0.55,
        corr_max_features=320,
        mi_max_features=320,
        rf_estimators=100,
        rf_max_depth=6,
        enabled_methods=enabled_methods,
        max_candidates_per_method=120,
    )


def _adaptive_sample_size(
    n_samples: int,
    n_features: int,
    *,
    total_size: Optional[int] = None,
    row_col_product: Optional[int] = None,
) -> int:
    """Return an adaptive sample size informed by rows, columns and volume."""

    if n_samples <= 100:
        return n_samples

    adjusted_features = max(1, n_features)

    if row_col_product is not None and row_col_product <= 1000:
        return n_samples

    feature_factor = (adjusted_features / 8.0) ** 0.3
    raw_size = (n_samples ** 0.6) * feature_factor
    sample_size = int(max(100, min(n_samples, raw_size)))

    if total_size is None:
        total_size = n_samples * adjusted_features

    if total_size <= 0:
        return n_samples

    scaled_sample = sample_size * 3
    lower_bound = total_size * 0.001
    upper_bound = total_size * 0.05

    if lower_bound <= scaled_sample <= upper_bound:
        sample_size = int(min(n_samples, max(scaled_sample, 1)))

    return sample_size


def _take_rows(data, indices: np.ndarray):
    if hasattr(data, "iloc"):
        return data.iloc[indices]

    if hasattr(data, "take"):
        try:
            return data.take(indices, axis=0)
        except TypeError:
            pass

    arr = np.asarray(data)
    return arr[indices]


def _maybe_adaptive_sample(
    X,
    y,
    records: Iterable[DeltaRecord],
    *,
    enabled: bool,
    random_state: Optional[int],
) -> Tuple[object, object, Iterable[DeltaRecord], int]:
    y_array = np.asarray(y)
    n_samples = len(y_array)

    if not enabled:
        records_list = list(records)
        return X, y, records_list, n_samples

    total_size: Optional[int] = None
    row_col_product: Optional[int] = None
    n_features = None

    shape = getattr(X, "shape", None)
    if shape is not None and hasattr(shape, "__len__") and len(shape) >= 1:
        dims: List[int] = []
        for dim in shape:
            if dim is None:
                dims = []
                break
            try:
                dims.append(int(dim))
            except (TypeError, ValueError):
                dims = []
                break
        if dims:
            total_size = int(np.prod(dims))
            if len(dims) == 1:
                n_features = 1
            else:
                feature_extent = int(np.prod(dims[1:]))
                n_features = max(1, feature_extent)
            row_col_product = dims[0] * (n_features or 1)

    if n_features is None:
        try:
            first_row = X[0]
        except (TypeError, IndexError):
            n_features = 1
        else:
            n_features = len(first_row) if hasattr(first_row, "__len__") else 1
        if total_size is None:
            total_size = n_samples * int(n_features)
        if row_col_product is None:
            row_col_product = n_samples * int(n_features)

    target = _adaptive_sample_size(
        n_samples,
        int(n_features or 1),
        total_size=total_size,
        row_col_product=row_col_product,
    )
    if target >= n_samples:
        records_list = list(records)
        return X, y, records_list, n_samples

    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(n_samples, size=target, replace=False))

    X_sampled = _take_rows(X, indices)

    if hasattr(y, "iloc"):
        y_sampled = y.iloc[indices]
    elif hasattr(y, "take"):
        try:
            y_sampled = y.take(indices, axis=0)
        except TypeError:
            y_sampled = y_array[indices]
    elif isinstance(y, list):
        y_sampled = [y[i] for i in indices]
    elif isinstance(y, tuple):
        y_sampled = tuple(y[i] for i in indices)
    else:
        y_sampled = y_array[indices]

    records_list: List[DeltaRecord] = list(records)
    if records_list:
        records_sampled = [records_list[i] for i in indices if i < len(records_list)]
    else:
        records_sampled = records_list

    return X_sampled, y_sampled, records_sampled, target


def run_core_method_bundle(
    X,
    y,
    records: Iterable[DeltaRecord],
    *,
    max_sets: int = 30,
    combo_sizes: Sequence[int] = (2, 3),
    random_state: Optional[int] = None,
    cv_splits: int = 3,
    adaptive_sampling: bool = True,
) -> CoreBundleResult:
    """Ejecuta ``MultiClassSubspaceExplorer`` limitado a los cinco métodos ganadores.

    Solo expone los parámetros imprescindibles para controlar la cantidad de
    combinaciones exploradas y la reproducibilidad; el resto usa heurísticas
    balanceadas para los datasets grandes utilizados en las evaluaciones.
    """

    explorer = _create_bundle_explorer(
        max_sets=max_sets,
        combo_sizes=combo_sizes,
        random_state=random_state,
        cv_splits=cv_splits,
        enabled_methods=CORE_METHOD_KEYS,
    )
    X_sampled, y_sampled, records_sampled, _ = _maybe_adaptive_sample(
        X,
        y,
        records,
        enabled=adaptive_sampling,
        random_state=random_state,
    )
    explorer.fit(X_sampled, y_sampled, records_sampled)
    reports = explorer.get_report()
    return CoreBundleResult(explorer=explorer, reports=reports)


def run_single_method(
    X,
    y,
    records: Iterable[DeltaRecord],
    method_key: str,
    *,
    max_sets: int = 30,
    combo_sizes: Sequence[int] = (2, 3),
    random_state: Optional[int] = None,
    cv_splits: int = 3,
    adaptive_sampling: bool = True,
) -> MultiClassSubspaceExplorer:
    """Ejecuta el explorador sólo con un método del bundle para validación."""

    explorer = _create_bundle_explorer(
        max_sets=max_sets,
        combo_sizes=combo_sizes,
        random_state=random_state,
        cv_splits=cv_splits,
        enabled_methods=(method_key,),
    )
    X_sampled, y_sampled, records_sampled, _ = _maybe_adaptive_sample(
        X,
        y,
        records,
        enabled=adaptive_sampling,
        random_state=random_state,
    )
    explorer.fit(X_sampled, y_sampled, records_sampled)
    return explorer

