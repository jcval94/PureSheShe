"""Experimentos para presets intermedios entre ``fast`` y ``ultra_fast``.

Ejecuta tres aproximaciones basadas en las ideas solicitadas:

1. Presupuesto CV mínimo (``mini_fast_cv``): igual al preset ``fast`` pero con
   un presupuesto de CV diminuto y 100% explotación (sin muestreo aleatorio).
2. Proxy MI + micro-CV (``proxy_plus_microcv``): rankea con el proxy de
   ``ultra_fast`` y reevalúa con una CV de 2 folds los mejores k candidatos.
3. Proxy MI calibrado (``calibrated_proxy``): reescala el proxy MI con la F1 de
   baseline y la cantidad de clases antes de seleccionar los top k a validar.

El script genera dataframes sintéticos con >=10 variables, recorre cada método
clave del bundle y escribe un CSV en ``experiments_outputs`` con métricas de
velocidad y precisión (F1 macro) para cada combinación. Por defecto solo se
ejecuta la pareja recomendada ``proxy_plus_microcv`` + ``method_8_extratrees``;
para reproducir el barrido completo añade la opción ``--all-options``.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for candidate in (PROJECT_ROOT / "src", PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from deldel.subspace_change_detector import MultiClassSubspaceExplorer
from subspaces.experiments.core_method_bundle import CORE_METHOD_KEYS

DEFAULT_OPTION = "proxy_plus_microcv"
DEFAULT_METHOD_KEY = "method_8_extratrees"

DatasetDef = tuple[str, pd.DataFrame, np.ndarray]
CandidateMap = Dict[str, List[Tuple[str, ...]]]


def _make_datasets() -> List[DatasetDef]:
    """Genera tres datasets sintéticos con >=10 variables y 4 clases."""

    configs = [
        {"n_samples": 180, "n_features": 12, "n_informative": 6, "random_state": 11},
        {"n_samples": 220, "n_features": 14, "n_informative": 7, "random_state": 13},
        {"n_samples": 260, "n_features": 16, "n_informative": 8, "random_state": 17},
    ]

    datasets: List[DatasetDef] = []
    for idx, cfg in enumerate(configs, start=1):
        X, y = make_classification(
            n_samples=cfg["n_samples"],
            n_features=cfg["n_features"],
            n_informative=cfg["n_informative"],
            n_redundant=0,
            n_repeated=0,
            n_classes=4,
            class_sep=1.1,
            flip_y=0.02,
            random_state=cfg["random_state"],
        )
        columns = [f"v{j}" for j in range(cfg["n_features"])]
        datasets.append((f"df{idx}", pd.DataFrame(X, columns=columns), y))
    return datasets


def _base_explorer(
    *,
    random_state: int,
    cv_splits: int,
    fast_eval_budget: int = 20,
    fast_eval_top_frac: float = 0.8,
) -> MultiClassSubspaceExplorer:
    """Crea un explorador compacto para los experimentos."""

    return MultiClassSubspaceExplorer(
        max_sets=None,
        combo_sizes=(2, 3),
        filter_top_k=10,
        chi2_pool=10,
        random_samples=18,
        corr_threshold=0.45,
        corr_max_features=48,
        mi_max_features=48,
        rf_estimators=24,
        rf_max_depth=4,
        cv_splits=cv_splits,
        random_state=random_state,
        enabled_methods=CORE_METHOD_KEYS,
        max_candidates_per_method=30,
        max_total_candidates=90,
        fast_eval_budget=fast_eval_budget,
        fast_eval_top_frac=float(fast_eval_top_frac),
        fast_compute_secondary_metrics=False,
    )


def _load_precomputed(explorer: MultiClassSubspaceExplorer, precomputed: dict) -> None:
    """Carga en el explorador el estado precalculado (mismo flujo que ``fit``)."""

    table = precomputed["table"]
    explorer.feature_names_ = list(table.columns)
    explorer.classes_ = precomputed["classes"]
    explorer.baseline_f1_ = precomputed["baseline_f1"]
    explorer.mi_scores_ = dict(precomputed["mi_scores"])
    explorer.stump_scores_ = dict(precomputed["stump_scores"])
    explorer.chi2_scores_ = dict(precomputed["chi2_all"])
    explorer.feature_ranking_ = list(precomputed["ranked_features"])


def _build_candidates(
    explorer: MultiClassSubspaceExplorer, precomputed: dict
) -> CandidateMap:
    """Construye y devuelve los candidatos por método usando el análisis compartido."""

    table = precomputed["table"]
    y_arr = precomputed["y_arr"]
    ranked_features = precomputed["ranked_features"]
    mi_scores = precomputed["mi_scores"]
    numeric_cols = precomputed["numeric_cols"]
    categorical_cols = precomputed["categorical_cols"]
    chi2_all_dict = precomputed["chi2_all"]

    _load_precomputed(explorer, precomputed)
    explorer._current_preset = "high_quality"
    explorer._build_candidate_sets(
        table,
        y_arr,
        ranked_features,
        mi_scores,
        numeric_cols,
        categorical_cols,
        chi2_all_dict,
        method_key=None,
    )
    return {
        key: sorted(value)
        for key, value in explorer.method_candidate_sets_.items()
        if value
    }


def _prepare_candidate_sources(
    method_key: str, candidates: Sequence[Tuple[str, ...]]
) -> Dict[Tuple[str, ...], set[str]]:
    """Genera el mapa de procedencia mínimo para el método indicado."""

    return {tuple(sorted(combo)): {method_key} for combo in candidates}


def _score_with_candidates(
    *,
    explorer: MultiClassSubspaceExplorer,
    precomputed: dict,
    candidates: Sequence[Tuple[str, ...]],
    method_key: str,
    preset: str,
) -> tuple[List, float]:
    """Evalúa los candidatos indicados con el preset deseado y mide el tiempo."""

    _load_precomputed(explorer, precomputed)
    explorer._current_preset = preset
    explorer.method_candidate_sets_ = {method_key: set(map(tuple, candidates))}
    explorer.candidate_sources_ = _prepare_candidate_sources(method_key, candidates)
    explorer.method_timings_ = {explorer.method_name_map_[method_key]: 0.0}
    explorer.all_candidate_sets_ = list(set(tuple(sorted(c)) for c in candidates))

    start = time.perf_counter()
    reports = explorer._score_candidates(precomputed["table"], precomputed["y_arr"], list(candidates))
    elapsed = time.perf_counter() - start
    annotated = explorer._annotate_reports(reports)
    return annotated, elapsed


def _mini_fast_cv(
    *,
    precomputed: dict,
    candidates: Sequence[Tuple[str, ...]],
    method_key: str,
    random_state: int,
) -> dict:
    """Preset fast con presupuesto de CV diminuto y sin exploración aleatoria."""

    explorer = _base_explorer(
        random_state=random_state,
        cv_splits=3,
        fast_eval_budget=4,
        fast_eval_top_frac=1.0,
    )
    reports, elapsed = _score_with_candidates(
        explorer=explorer,
        precomputed=precomputed,
        candidates=candidates,
        method_key=method_key,
        preset="fast",
    )
    return _summarize_result(
        label="mini_fast_cv",
        method_key=method_key,
        reports=reports,
        elapsed_proxy=0.0,
        elapsed_cv=elapsed,
        candidate_pool=len(candidates),
        evaluated=len(reports),
        baseline=float(explorer.baseline_f1_),
    )


def _proxy_plus_microcv(
    *,
    precomputed: dict,
    candidates: Sequence[Tuple[str, ...]],
    method_key: str,
    random_state: int,
    top_k: int = 8,
) -> dict:
    """Ranking ultra-fast y reordenamiento con micro-CV (2 folds)."""

    proxy_explorer = _base_explorer(random_state=random_state, cv_splits=2)
    proxy_reports, proxy_elapsed = _score_with_candidates(
        explorer=proxy_explorer,
        precomputed=precomputed,
        candidates=candidates,
        method_key=method_key,
        preset="ultra_fast",
    )

    top_candidates = [r.features for r in sorted(proxy_reports, key=lambda r: r.mean_macro_f1, reverse=True)[:top_k]]

    micro_cv_explorer = _base_explorer(
        random_state=random_state,
        cv_splits=2,
        fast_eval_budget=max(top_k, 4),
        fast_eval_top_frac=1.0,
    )
    cv_reports, cv_elapsed = _score_with_candidates(
        explorer=micro_cv_explorer,
        precomputed=precomputed,
        candidates=top_candidates,
        method_key=method_key,
        preset="fast",
    )

    return _summarize_result(
        label="proxy_plus_microcv",
        method_key=method_key,
        reports=cv_reports,
        elapsed_proxy=proxy_elapsed,
        elapsed_cv=cv_elapsed,
        candidate_pool=len(candidates),
        evaluated=len(top_candidates),
        baseline=float(micro_cv_explorer.baseline_f1_),
    )


def _calibrated_proxy(
    *,
    precomputed: dict,
    candidates: Sequence[Tuple[str, ...]],
    method_key: str,
    random_state: int,
    top_k: int = 8,
) -> dict:
    """Ranking por proxy MI reescalado con baseline F1 y cardinalidad de clases."""

    mi_scores = precomputed["mi_scores"]
    baseline_f1 = float(precomputed["baseline_f1"])
    n_classes = max(1, len(precomputed["classes"]))

    start_proxy = time.perf_counter()
    scored_candidates: List[Tuple[float, Tuple[str, ...]]] = []
    for combo in candidates:
        if not combo:
            continue
        raw_mi = sum(float(mi_scores.get(f, 0.0)) for f in combo)
        normalized = raw_mi / max(len(combo), 1)
        calibrated = normalized * (1.0 + baseline_f1) * math.log1p(n_classes)
        scored_candidates.append((calibrated, combo))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    proxy_elapsed = time.perf_counter() - start_proxy

    top_candidates = [combo for _, combo in scored_candidates[:top_k]]

    cv_explorer = _base_explorer(
        random_state=random_state,
        cv_splits=3,
        fast_eval_budget=max(top_k, 4),
        fast_eval_top_frac=1.0,
    )
    cv_reports, cv_elapsed = _score_with_candidates(
        explorer=cv_explorer,
        precomputed=precomputed,
        candidates=top_candidates,
        method_key=method_key,
        preset="fast",
    )

    return _summarize_result(
        label="calibrated_proxy",
        method_key=method_key,
        reports=cv_reports,
        elapsed_proxy=proxy_elapsed,
        elapsed_cv=cv_elapsed,
        candidate_pool=len(candidates),
        evaluated=len(top_candidates),
        baseline=float(cv_explorer.baseline_f1_),
    )


def _summarize_result(
    *,
    label: str,
    method_key: str,
    reports,
    elapsed_proxy: float,
    elapsed_cv: float,
    candidate_pool: int,
    evaluated: int,
    baseline: float,
) -> dict:
    best_f1 = max((r.mean_macro_f1 for r in reports), default=float("nan"))
    mean_f1 = float(np.mean([r.mean_macro_f1 for r in reports])) if reports else float("nan")
    return {
        "option": label,
        "method_key": method_key,
        "candidate_pool": candidate_pool,
        "evaluated_candidates": evaluated,
        "best_mean_macro_f1": best_f1,
        "avg_mean_macro_f1": mean_f1,
        "baseline_f1": baseline,
        "elapsed_proxy_s": elapsed_proxy,
        "elapsed_cv_s": elapsed_cv,
        "elapsed_total_s": elapsed_proxy + elapsed_cv,
    }


def _run_all_options(
    *,
    dataset_id: str,
    df: pd.DataFrame,
    y: np.ndarray,
    random_state: int,
    only_default: bool,
) -> List[dict]:
    base_explorer = _base_explorer(random_state=random_state, cv_splits=3)
    precomputed = base_explorer.precompute_analysis(df, y, skip_feature_stats=False)
    candidates_by_method = _build_candidates(base_explorer, precomputed)

    rows: List[dict] = []
    for method_key, candidates in candidates_by_method.items():
        if only_default and method_key != DEFAULT_METHOD_KEY:
            continue

        option_fns = [
            _mini_fast_cv,
            _proxy_plus_microcv,
            _calibrated_proxy,
        ]
        if only_default:
            option_fns = []
            if DEFAULT_OPTION == "proxy_plus_microcv":
                option_fns.append(_proxy_plus_microcv)

        for fn in option_fns:
            rows.append(
                {
                    "dataset": dataset_id,
                    "n_samples": df.shape[0],
                    "n_features": df.shape[1],
                    **fn(
                        precomputed=precomputed,
                        candidates=candidates,
                        method_key=method_key,
                        random_state=random_state,
                    ),
                }
            )
    return rows


def run_experiments(*, only_default: bool = True) -> List[dict]:
    rng = np.random.RandomState(123)
    rows: List[dict] = []
    for dataset_id, df, y in _make_datasets():
        rows.extend(
            _run_all_options(
                dataset_id=dataset_id,
                df=df,
                y=y,
                random_state=int(rng.randint(0, 10000)),
                only_default=only_default,
            )
        )
    return rows


def write_csv(output_path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "experiments_outputs" / "mid_preset_options.csv"

    # Por defecto solo se ejecuta la combinación recomendada
    # ``proxy_plus_microcv`` + ``method_8_extratrees``. Para replicar todos los
    # experimentos originales, pasar ``--all-options`` como argumento.
    only_default = "--all-options" not in sys.argv

    write_csv(output_path, run_experiments(only_default=only_default))


if __name__ == "__main__":
    main()
