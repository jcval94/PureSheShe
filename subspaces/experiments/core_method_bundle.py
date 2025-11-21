"""Utilidades para ejecutar el bundle de métodos destacados."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from deldel.engine import DeltaRecord
from deldel.subspace_change_detector import MultiClassSubspaceExplorer, SubspaceReport

from .adaptive_sampling import AdaptiveSamplingInfo, maybe_adaptive_sample


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
    sampling_info: AdaptiveSamplingInfo


@dataclass(frozen=True)
class CoreMethodSummary:
    """Resumen ejecutivo por método dentro del bundle núcleo."""

    method_key: str
    method_name: str
    candidate_sets: int
    selected_reports: int
    elapsed_seconds: float


_maybe_adaptive_sample = maybe_adaptive_sample


def _create_bundle_explorer(
    *,
    max_sets: Optional[int],
    combo_sizes: Sequence[int],
    random_state: Optional[int],
    cv_splits: int,
    enabled_methods: Sequence[str],
) -> MultiClassSubspaceExplorer:
    """Crea un ``MultiClassSubspaceExplorer`` listo para el bundle.

    Parámetros
    ----------
    max_sets:
        Cantidad máxima de combinaciones finales que el explorador intentará
        devolver.  Usa ``None`` (valor por defecto) para devolver todas las
        combinaciones generadas, ordenadas de mejor a peor.
    combo_sizes:
        Tamaños de las combinaciones de columnas que se van a evaluar.
        Cada número indica cuántas columnas forman un subespacio: ``(2, 3)``
        explora pares y tríos de columnas, mientras que ``(4,)`` probaría sólo
        combinaciones de cuatro columnas.  Valores mayores cubren interacciones
        más complejas, pero también multiplican el número de candidatos y el
        tiempo de búsqueda.
    random_state:
        Semilla para reproducir los muestreos aleatorios; usa ``None`` para
        comportamientos no deterministas.
    cv_splits:
        Número de particiones de validación cruzada para estimar F1 en cada
        conjunto candidato.
    enabled_methods:
        Claves internas de los métodos que el bundle debe activar.
    """
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
    max_sets: Optional[int] = None,
    combo_sizes: Sequence[int] = (2, 3),
    random_state: Optional[int] = None,
    cv_splits: int = 3,
    adaptive_sampling: bool = True,
) -> CoreBundleResult:
    """Ejecuta ``MultiClassSubspaceExplorer`` limitado a los cinco métodos ganadores.

    Solo expone los parámetros imprescindibles para controlar la cantidad de
    combinaciones exploradas y la reproducibilidad; el resto usa heurísticas
    balanceadas para los datasets grandes utilizados en las evaluaciones.

    Parámetros
    ----------
    X:
        Matriz de características original (puede ser NumPy, pandas o similar).
    y:
        Etiquetas de clase asociadas a cada fila de ``X``.
    records:
        Secuencia de ``DeltaRecord`` generados previamente por DelDel.
    max_sets:
        Límite opcional de conjuntos de columnas que se conservarán al final;
        usa ``None`` (por defecto) para conservar todos los subespacios
        evaluados.
    combo_sizes:
        Tamaños de las combinaciones a explorar.  Cada valor representa cuántas
        columnas se agrupan en un subespacio candidato: ``(2, 3)`` evalúa pares
        y tríos, ``(1,)`` probaría columnas individuales y ``(2, 3, 4)``
        cubriría interacciones desde pares hasta cuartetos.  Incluir tamaños
        mayores aumenta el coste computacional porque crece el número de
        combinaciones posibles.
    random_state:
        Semilla opcional para reproducir los resultados; usa ``None`` para un
        muestreo distinto en cada ejecución.
    cv_splits:
        Número de particiones en la validación cruzada que estima el F1
        macro de cada subespacio.
    adaptive_sampling:
        Si es ``True``, aplica un muestreo adaptativo para acelerar datasets
        grandes; si es ``False``, trabaja con todos los registros.
    """

    records_list = list(records)
    sampled_X, sampled_y, sampled_records, sampling_info = _maybe_adaptive_sample(
        X,
        y,
        records_list,
        adaptive_sampling=adaptive_sampling,
        random_state=random_state,
    )

    explorer = _create_bundle_explorer(
        max_sets=max_sets,
        combo_sizes=combo_sizes,
        random_state=random_state,
        cv_splits=cv_splits,
        enabled_methods=CORE_METHOD_KEYS,
    )
    explorer.fit(sampled_X, sampled_y, sampled_records)
    reports = explorer.get_report()
    return CoreBundleResult(
        explorer=explorer,
        reports=reports,
        sampling_info=sampling_info,
    )


def summarize_core_bundle(result: CoreBundleResult) -> List[CoreMethodSummary]:
    """Genera métricas compactas por método tras ejecutar el bundle.

    La función no altera el resultado original; únicamente consolida información
    de ``MultiClassSubspaceExplorer`` en una estructura fácil de consumir para
    logging o visualizaciones.  ``run_core_method_bundle`` se mantiene
    independiente y puede usarse sin invocar este resumen auxiliar.

    Parámetros
    ----------
    result:
        Objeto devuelto por ``run_core_method_bundle`` con el explorador, los
        reportes seleccionados y el detalle del muestreo.
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
    max_sets: Optional[int] = None,
    combo_sizes: Sequence[int] = (2, 3),
    random_state: Optional[int] = None,
    cv_splits: int = 3,
    adaptive_sampling: bool = True,
) -> MultiClassSubspaceExplorer:
    """Ejecuta el explorador sólo con un método del bundle para validación.

    Parámetros
    ----------
    X:
        Conjunto de características de entrada (array, DataFrame, etc.).
    y:
        Etiquetas de clase correspondientes a ``X``.
    records:
        Lista de ``DeltaRecord`` generados por DelDel que contienen la
        información de planos.
    method_key:
        Clave interna del método que se desea probar en solitario.
    max_sets:
        Número máximo de subespacios que se devolverán; usa ``None`` para
        conservarlos todos en orden descendente de F1.
    combo_sizes:
        Tamaños de las combinaciones de columnas a evaluar.  ``(2, 3)`` recorre
        pares y tríos, útil para capturar interacciones simples; ``(1,)`` se
        limita a columnas individuales y ``(4,)`` buscaría sólo combinaciones de
        cuatro columnas.  Añadir tamaños más grandes implica explorar muchas más
        combinaciones y puede ralentizar la ejecución.
    random_state:
        Semilla opcional para repetir la misma selección aleatoria.
    cv_splits:
        Cantidad de particiones de validación cruzada utilizadas para puntuar
        cada candidato.
    adaptive_sampling:
        Controla si se activa el muestreo adaptativo para agilizar datasets
        grandes.
    """

    records_list = list(records)
    sampled_X, sampled_y, sampled_records, _ = _maybe_adaptive_sample(
        X,
        y,
        records_list,
        adaptive_sampling=adaptive_sampling,
        random_state=random_state,
    )

    explorer = _create_bundle_explorer(
        max_sets=max_sets,
        combo_sizes=combo_sizes,
        random_state=random_state,
        cv_splits=cv_splits,
        enabled_methods=(method_key,),
    )
    explorer.fit(sampled_X, sampled_y, sampled_records)
    return explorer

