"""Utilidades para ejecutar el bundle de métodos destacados."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class CoreMethodSummary:
    """Resumen ejecutivo por método dentro del bundle núcleo."""

    method_key: str
    method_name: str
    candidate_sets: int
    selected_reports: int
    elapsed_seconds: float


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


def run_core_method_bundle(
    X,
    y,
    records: Iterable[DeltaRecord],
    *,
    max_sets: int = 30,
    combo_sizes: Sequence[int] = (2, 3),
    random_state: Optional[int] = None,
    cv_splits: int = 3,
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
    explorer.fit(X, y, records)
    reports = explorer.get_report()
    return CoreBundleResult(explorer=explorer, reports=reports)


def summarize_core_bundle(result: CoreBundleResult) -> List[CoreMethodSummary]:
    """Genera métricas compactas por método tras ejecutar el bundle.

    La función no altera el resultado original; únicamente consolida información
    de ``MultiClassSubspaceExplorer`` en una estructura fácil de consumir para
    logging o visualizaciones.  ``run_core_method_bundle`` se mantiene
    independiente y puede usarse sin invocar este resumen auxiliar.
    """

    explorer = result.explorer
    method_keys = list(explorer.method_name_map_.keys())
    selected_counts = {key: 0 for key in method_keys}
    for report in result.reports:
        combo = tuple(sorted(report.features))
        for method_key in explorer.candidate_sources_.get(combo, set()):
            if method_key in selected_counts:
                selected_counts[method_key] += 1

    summaries: List[CoreMethodSummary] = []
    for method_key in method_keys:
        friendly = explorer.method_name_map_[method_key]
        candidates = explorer.method_candidate_sets_.get(method_key, set())
        elapsed = explorer.method_timings_.get(friendly, 0.0)
        summaries.append(
            CoreMethodSummary(
                method_key=method_key,
                method_name=friendly,
                candidate_sets=len(candidates),
                selected_reports=selected_counts.get(method_key, 0),
                elapsed_seconds=float(elapsed),
            )
        )
    return summaries


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
) -> MultiClassSubspaceExplorer:
    """Ejecuta el explorador sólo con un método del bundle para validación."""

    explorer = _create_bundle_explorer(
        max_sets=max_sets,
        combo_sizes=combo_sizes,
        random_state=random_state,
        cv_splits=cv_splits,
        enabled_methods=(method_key,),
    )
    explorer.fit(X, y, records)
    return explorer

