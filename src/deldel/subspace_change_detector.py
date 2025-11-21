"""Pipeline ligero para detectar subespacios discriminativos a partir de ``d.records_``.

El módulo implementa un flujo de 4 etapas (Filtro ➜ Candidatos ➜ Ranking ➜ records)
para problemas multiclase con variables mixtas.  La idea es reducir el espacio de
búsqueda, generar combinaciones prometedoras y finalmente ajustar planos/half-spaces
por subespacio usando los ``DeltaRecord`` disponibles.

Catálogo de ideas utilizadas y en qué etapa aportan más valor:

* Top-k + combinaciones aleatorias dentro del top-k — Generación de candidatos.
* Pre-ranking individual por χ² y luego combinar top individuales — Filtro/Candidatos.
* Conjuntos por correlación absoluta interna — Candidatos/Ranking.
* Ramas repetidas en RandomForest profundo — Candidatos.
* Árboles de decisión profundidad-1 para caída de entropía — Filtro.
* Agrupamiento por vecinos de información mutua compartida — Filtro.
* Ratio varianza intra/inter clase barato — Ranking.
* Importancias por regresión logística con L1 suave — Ranking.
* Selección estable por bootstrap ligero — Ranking.
* Binning adaptativo para continuas antes de χ² — Filtro.
* Warm-start de planos usando centroides de records — records.

Mantiene dependencias sólo en NumPy y scikit-learn; las entradas tipo DataFrame se
manejan sin depender de pandas para permitir uso en entornos ligeros.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Set

from time import perf_counter

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.extmath import randomized_svd

from .engine import DeltaRecord


@dataclass
class SubspacePlane:
    """Representa un plano ajustado sobre un par de clases en un subespacio."""

    classes: Tuple[int, int]
    normal: np.ndarray
    bias: float
    rmse: float
    support: int
    weight: float


@dataclass
class SubspaceReport:
    """Resumen completo para un conjunto de columnas."""

    features: Tuple[str, ...]
    mean_macro_f1: float
    std_macro_f1: float
    lift_vs_majority: float
    coverage_ratio: float
    per_class_f1: Dict[int, float]
    support: int
    variance_ratio: float
    l1_importance: float
    method_keys: Tuple[str, ...] = ()
    method_key: Optional[str] = None
    global_yes: int = 0
    top50_yes: int = 0
    planes: List[SubspacePlane] = field(default_factory=list)


def _top_hits_by_method(
    reports: Sequence[SubspaceReport],
    candidate_sources: Dict[Tuple[str, ...], Set[str]],
    method_keys: Sequence[str],
    *,
    top_k: int = 50,
) -> Dict[str, int]:
    """Cuenta cuántos reportes top-k provienen de cada método."""

    sorted_reports = sorted(reports, key=lambda r: r.mean_macro_f1, reverse=True)[:top_k]
    counts: Dict[str, int] = {key: 0 for key in method_keys}
    for report in sorted_reports:
        combo = tuple(sorted(report.features))
        for method_key in candidate_sources.get(combo, set()):
            if method_key in counts:
                counts[method_key] += 1
    return counts


@dataclass
class _Table:
    columns: List[str]
    arrays: List[np.ndarray]

    def __len__(self) -> int:
        return int(self.arrays[0].shape[0]) if self.arrays else 0

    def column_index(self, name: str) -> int:
        return self.columns.index(name)

    def column_data(self, name: str) -> np.ndarray:
        return np.asarray(self.arrays[self.column_index(name)]).reshape(-1)

    def numeric_columns(self) -> List[str]:
        return [name for name, arr in zip(self.columns, self.arrays) if _is_numeric(arr) and not _is_bool(arr)]

    def categorical_columns(self) -> List[str]:
        return [name for name, arr in zip(self.columns, self.arrays) if not (_is_numeric(arr) and not _is_bool(arr))]


class MultiClassSubspaceExplorer:
    """Explora subespacios de baja dimensión y ajusta planos usando ``d.records_``."""

    def __init__(
        self,
        *,
        max_sets: int = 35,
        combo_sizes: Sequence[int] = (2, 3),
        filter_top_k: int = 20,
        chi2_pool: int = 20,
        random_samples: int = 120,
        corr_threshold: float = 0.55,
        corr_max_features: Optional[int] = None,
        rf_estimators: int = 80,
        rf_max_depth: Optional[int] = 6,
        cv_splits: int = 3,
        mi_max_features: Optional[int] = None,
        max_total_candidates: Optional[int] = None,
        max_candidates_per_method: Optional[int] = None,
        random_state: Optional[int] = None,
        enabled_methods: Optional[Sequence[str]] = None,
    ) -> None:
        self.max_sets = int(max_sets)
        self.combo_sizes = tuple(int(s) for s in combo_sizes if int(s) >= 1)
        self.filter_top_k = int(filter_top_k)
        self.chi2_pool = int(chi2_pool)
        self.random_samples = int(random_samples)
        self.corr_threshold = float(corr_threshold)
        self.corr_max_features = None if corr_max_features is None else int(corr_max_features)
        self.rf_estimators = int(rf_estimators)
        self.rf_max_depth = None if rf_max_depth is None else int(rf_max_depth)
        self.cv_splits = int(cv_splits)
        self.mi_max_features = None if mi_max_features is None else int(mi_max_features)
        self.max_total_candidates = (
            None if max_total_candidates is None else int(max_total_candidates)
        )
        self.max_candidates_per_method = (
            None if max_candidates_per_method is None else int(max_candidates_per_method)
        )
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
        self.feature_names_: Optional[List[str]] = None
        self.classes_: Optional[np.ndarray] = None
        self.baseline_f1_: float = float("nan")
        self.report_: List[SubspaceReport] = []
        self.feature_ranking_: List[str] = []
        self.mi_scores_: Dict[str, float] = {}
        self.chi2_scores_: Dict[str, float] = {}
        self.stump_scores_: Dict[str, float] = {}
        self.method_timings_: Dict[str, float] = {}
        self.method_candidate_sets_: Dict[str, Set[Tuple[str, ...]]] = {}
        self.candidate_sources_: Dict[Tuple[str, ...], Set[str]] = {}
        self.all_candidate_sets_: List[Tuple[str, ...]] = []
        base_method_map: Dict[str, str] = {
            "method_1_topk_random": "Top-k random combinations",
            "method_1b_topk_guided": "Top-k guided combinations (mejorado)",
            "method_2_chi2": "Chi2 top combinations",
            "method_2b_chi2_guided": "Chi2 targeted combinations (mejorado)",
            "method_3_corr_groups": "High correlation groups",
            "method_3b_corr_guided": "Correlation clusters refined (mejorado)",
            "method_4_rf_paths": "Random forest pair frequency",
            "method_4b_rf_weighted": "Random forest weighted paths (mejorado)",
            "method_5_leverage": "Leverage score filter",
            "method_5b_leverage_class": "Leverage class-focused (mejorado)",
            "method_6_sparse_proj": "Sparse random projections",
            "method_6b_sparse_proj_guided": "Sparse projections Fisher-guided (mejorado)",
            "method_7_lazy_greedy": "Lazy greedy dispersion",
            "method_7b_lazy_greedy_refined": "Lazy greedy with variance lookahead (mejorado)",
            "method_8_extratrees": "ExtraTrees shallow routes",
            "method_8b_extratrees_refined": "ExtraTrees purity-weighted (mejorado)",
            "method_9_countsketch": "CountSketch heavy hitters",
            "method_9b_countsketch_refined": "CountSketch contrastive (mejorado)",
            "method_10_gradient_synergy": "Gradient synergy matrix",
            "method_10b_gradient_hessian": "Gradient-Hessian synergy (mejorado)",
            "method_11_minhash_lsh": "MinHash class co-occurrence",
            "method_11b_minhash_refined": "MinHash contrastive bands (mejorado)",
        }
        self._full_method_name_map = dict(base_method_map)
        if enabled_methods is None:
            self.method_name_map_ = dict(base_method_map)
        else:
            normalized_keys: Set[str] = set()
            friendly_lookup = {name.lower(): key for key, name in base_method_map.items()}
            for entry in enabled_methods:
                if entry in base_method_map:
                    normalized_keys.add(entry)
                else:
                    key = friendly_lookup.get(str(entry).lower())
                    if key is None:
                        raise ValueError(
                            f"Unknown method '{entry}'. Expected one of: {sorted(base_method_map)}"
                        )
                    normalized_keys.add(key)
            if not normalized_keys:
                raise ValueError("enabled_methods no puede estar vacío")
            ordered_keys = [key for key in base_method_map if key in normalized_keys]
            self.method_name_map_ = {key: base_method_map[key] for key in ordered_keys}
        self._enabled_method_keys: Set[str] = set(self.method_name_map_.keys())
        self.candidate_reports_: Dict[Tuple[str, ...], SubspaceReport] = {}
        self.evaluated_reports_: List[SubspaceReport] = []

    def _method_enabled(self, method_key: str) -> bool:
        return method_key in self._enabled_method_keys

    def fit(
        self,
        X: Any,
        y: Sequence[int],
        records: Iterable[DeltaRecord],
    ) -> "MultiClassSubspaceExplorer":
        """Ejecuta el flujo completo y guarda el reporte en ``self.report_``."""

        table = _ensure_table(X)
        self.feature_names_ = list(table.columns)
        y_arr = np.asarray(y)
        classes, y_codes = np.unique(y_arr, return_inverse=True)
        self.classes_ = classes
        self.baseline_f1_ = _majority_macro_f1(y_arr, classes)

        numeric_cols = table.numeric_columns()
        categorical_cols = table.categorical_columns()

        if self.mi_max_features is not None and len(table.columns) > self.mi_max_features:
            subset_idx = np.sort(self._rng.choice(len(table.columns), size=self.mi_max_features, replace=False))
            mi_subset = [table.columns[i] for i in subset_idx]
        else:
            mi_subset = list(table.columns)
        mi_partial = _mutual_info_scores(table, y_arr, columns_subset=mi_subset)
        mi_scores = {name: mi_partial.get(name, 0.0) for name in table.columns}
        stump_scores = _stump_entropy_drop(table, y_arr)
        chi2_cat = _chi2_scores({name: table.column_data(name) for name in categorical_cols}, y_codes, classes) if categorical_cols else {}
        chi2_num = _chi2_scores(_quantile_binning(table, numeric_cols), y_codes, classes) if numeric_cols else {}

        self.mi_scores_ = dict(mi_scores)
        self.stump_scores_ = dict(stump_scores)
        chi2_all_dict = {**chi2_cat, **{name: chi2_num.get(name, 0.0) for name in numeric_cols}}
        self.chi2_scores_ = dict(chi2_all_dict)

        ranked_features = _rank_features(
            mi_scores,
            chi2_cat,
            chi2_num,
            stump_scores,
            numeric_cols,
            categorical_cols,
            self.filter_top_k,
        )

        self.feature_ranking_ = list(ranked_features)

        candidate_sets = self._build_candidate_sets(
            table,
            y_arr,
            ranked_features,
            mi_scores,
            numeric_cols,
            categorical_cols,
            chi2_all_dict,
        )

        scored = self._score_candidates(table, y_arr, candidate_sets)
        self.report_ = self._attach_planes(scored, records)
        return self

    def get_report(self) -> List[SubspaceReport]:
        """Devuelve los mejores subespacios encontrados."""

        return self._annotate_reports(list(self.report_))

    def _annotate_reports(
        self, reports: Sequence[SubspaceReport]
    ) -> List[SubspaceReport]:
        """Completa los reportes con metadatos de origen por método."""

        if not reports:
            return []

        method_keys = list(self.method_name_map_.keys())
        top_hits = _top_hits_by_method(
            reports,
            self.candidate_sources_,
            method_keys,
            top_k=50,
        )
        global_counts = {
            key: len(self.method_candidate_sets_.get(key, set())) for key in method_keys
        }

        annotated: List[SubspaceReport] = []
        for report in reports:
            combo = tuple(sorted(report.features))
            origins = tuple(sorted(self.candidate_sources_.get(combo, set())))
            report.method_keys = origins
            report.method_key = origins[0] if origins else None
            report.global_yes = sum(global_counts.get(key, 0) for key in origins)
            report.top50_yes = sum(top_hits.get(key, 0) for key in origins)
            annotated.append(report)

        return annotated

    def _build_candidate_sets(
        self,
        table: _Table,
        y: np.ndarray,
        ranked_features: List[str],
        mi_scores: Dict[str, float],
        numeric_cols: List[str],
        categorical_cols: List[str],
        chi2_scores: Dict[str, float],
    ) -> List[Tuple[str, ...]]:
        top_pool = ranked_features[: max(self.filter_top_k, len(self.combo_sizes))]
        feature_to_idx = {name: idx for idx, name in enumerate(self.feature_names_ or [])}
        pool_idx = np.array([feature_to_idx[f] for f in top_pool], int) if top_pool else np.empty(0, int)

        method_sets: Dict[str, Set[Tuple[str, ...]]] = {key: set() for key in self.method_name_map_}
        candidate_sources: Dict[Tuple[str, ...], Set[str]] = {}
        timings: Dict[str, float] = {key: 0.0 for key in self.method_name_map_}

        def _method_rng(method_key: str) -> np.random.RandomState:
            if self.random_state is None:
                return self._rng
            return np.random.RandomState(self.random_state)

        def _add_candidate(combo: Tuple[str, ...], method_key: str) -> None:
            if not self._method_enabled(method_key):
                return
            combo_sorted = tuple(sorted(combo))
            method_sets[method_key].add(combo_sorted)
            candidate_sources.setdefault(combo_sorted, set()).add(method_key)

        def _time_block(method_key: str, func: Callable[[], None]) -> None:
            if not self._method_enabled(method_key):
                return
            start = perf_counter()
            func()
            timings[method_key] += perf_counter() - start

        def _run_method_2_chi2() -> None:
            if not chi2_scores:
                return
            sorted_by_chi2 = sorted(chi2_scores.items(), key=lambda kv: kv[1], reverse=True)
            chi2_top = [name for name, _ in sorted_by_chi2[: self.chi2_pool]]
            rng = _method_rng("method_2_chi2")
            for size in self.combo_sizes:
                if len(chi2_top) < size:
                    continue
                n_possible = math.comb(len(chi2_top), size)
                if n_possible <= self.random_samples:
                    iterator = combinations(chi2_top, size)
                else:
                    seen_idx: Set[Tuple[int, ...]] = set()
                    iterator = []
                    while len(seen_idx) < self.random_samples:
                        idx = tuple(sorted(rng.choice(len(chi2_top), size=size, replace=False)))
                        if idx in seen_idx:
                            continue
                        seen_idx.add(idx)
                        iterator.append(tuple(chi2_top[i] for i in idx))
                    iterator = iter(iterator)
                for combo in iterator:
                    _add_candidate(combo, "method_2_chi2")

        _time_block("method_2_chi2", _run_method_2_chi2)

        def _run_method_2b() -> None:
            guided_chi2 = _chi2_guided_candidates(
                table,
                chi2_scores,
                mi_scores,
                self.combo_sizes,
            )
            for combo in guided_chi2:
                _add_candidate(combo, "method_2b_chi2_guided")

        _time_block("method_2b_chi2_guided", _run_method_2b)

        def _run_method_1() -> None:
            rng = _method_rng("method_1_topk_random")
            for size in self.combo_sizes:
                if pool_idx.size < size:
                    continue
                n_possible = math.comb(pool_idx.size, size)
                n_draws = min(self.random_samples, n_possible)
                seen: Set[Tuple[int, ...]] = set()
                while len(seen) < n_draws:
                    idx = tuple(sorted(rng.choice(pool_idx, size=size, replace=False)))
                    if idx in seen:
                        continue
                    seen.add(idx)
                    combo = tuple(self.feature_names_[i] for i in idx)
                    _add_candidate(combo, "method_1_topk_random")
                    if sum(len(s) for s in method_sets.values()) > self.random_samples * len(self.combo_sizes) * 2:
                        break
            mi_neighbor = _mi_neighbor_combos(ranked_features, mi_scores, self.combo_sizes)
            for combo in mi_neighbor:
                _add_candidate(combo, "method_1_topk_random")

        _time_block("method_1_topk_random", _run_method_1)

        def _run_method_1b() -> None:
            rng = _method_rng("method_1b_topk_guided")
            guided_topk = _topk_guided_candidates(
                table,
                ranked_features,
                mi_scores,
                self.combo_sizes,
                rng,
            )
            for combo in guided_topk:
                _add_candidate(combo, "method_1b_topk_guided")

        _time_block("method_1b_topk_guided", _run_method_1b)

        def _run_method_3() -> None:
            rng = _method_rng("method_3_corr_groups")
            corr_cols = list(numeric_cols)
            if self.corr_max_features is not None and len(corr_cols) > self.corr_max_features:
                idx = np.sort(rng.choice(len(corr_cols), size=self.corr_max_features, replace=False))
                corr_cols = [corr_cols[i] for i in idx]
            corr_pairs = _high_corr_groups(table, corr_cols, threshold=self.corr_threshold)
            for combo in corr_pairs:
                if len(combo) in self.combo_sizes:
                    _add_candidate(tuple(combo), "method_3_corr_groups")
                elif len(combo) > 1:
                    for size in self.combo_sizes:
                        if size < len(combo):
                            for sub in combinations(combo, size):
                                _add_candidate(tuple(sub), "method_3_corr_groups")

        _time_block("method_3_corr_groups", _run_method_3)

        def _run_method_3b() -> None:
            rng = _method_rng("method_3b_corr_guided")
            corr_guided = _correlation_guided_candidates(
                table,
                numeric_cols,
                mi_scores,
                self.combo_sizes,
                self.corr_threshold,
                rng,
            )
            for combo in corr_guided:
                _add_candidate(combo, "method_3b_corr_guided")

        _time_block("method_3b_corr_guided", _run_method_3b)

        def _run_method_4() -> None:
            if len(top_pool) < 2:
                return
            rf_pairs = _rf_pairs(
                table,
                top_pool,
                y,
                n_estimators=self.rf_estimators,
                max_depth=self.rf_max_depth,
                random_state=self.random_state,
            )
            for pair in rf_pairs:
                _add_candidate(tuple(pair), "method_4_rf_paths")

        _time_block("method_4_rf_paths", _run_method_4)

        def _run_method_4b() -> None:
            if len(top_pool) < 2:
                return
            rf_weighted = _rf_weighted_candidates(
                table,
                top_pool,
                y,
                self.combo_sizes,
                n_estimators=self.rf_estimators,
                max_depth=self.rf_max_depth,
                random_state=self.random_state,
            )
            for combo in rf_weighted:
                _add_candidate(combo, "method_4b_rf_weighted")

        _time_block("method_4b_rf_weighted", _run_method_4b)

        def _run_method_5() -> None:
            if not self.feature_names_:
                return
            rng = _method_rng("method_5_leverage")
            leverage_sets = _leverage_candidates(
                table,
                self.feature_names_,
                self.combo_sizes,
                self.filter_top_k,
                rng,
            )
            for combo in leverage_sets:
                _add_candidate(combo, "method_5_leverage")

        _time_block("method_5_leverage", _run_method_5)

        def _run_method_5b() -> None:
            if not self.feature_names_:
                return
            rng = _method_rng("method_5b_leverage_class")
            leverage_class_sets = _leverage_class_candidates(
                table,
                y,
                self.feature_names_,
                self.combo_sizes,
                self.filter_top_k,
                rng,
            )
            for combo in leverage_class_sets:
                _add_candidate(combo, "method_5b_leverage_class")

        _time_block("method_5b_leverage_class", _run_method_5b)

        def _run_method_6() -> None:
            rng = _method_rng("method_6_sparse_proj")
            sparse_proj_sets = _sparse_projection_candidates(
                table,
                y,
                ranked_features,
                self.combo_sizes,
                rng,
            )
            for combo in sparse_proj_sets:
                _add_candidate(combo, "method_6_sparse_proj")

        _time_block("method_6_sparse_proj", _run_method_6)

        def _run_method_6b() -> None:
            rng = _method_rng("method_6b_sparse_proj_guided")
            sparse_guided_sets = _sparse_projection_guided_candidates(
                table,
                y,
                ranked_features,
                self.combo_sizes,
                rng,
            )
            for combo in sparse_guided_sets:
                _add_candidate(combo, "method_6b_sparse_proj_guided")

        _time_block("method_6b_sparse_proj_guided", _run_method_6b)

        def _run_method_7() -> None:
            rng = _method_rng("method_7_lazy_greedy")
            lazy_sets = _lazy_greedy_candidates(
                table,
                ranked_features,
                self.combo_sizes,
                mi_scores,
                rng,
            )
            for combo in lazy_sets:
                _add_candidate(combo, "method_7_lazy_greedy")

        _time_block("method_7_lazy_greedy", _run_method_7)

        def _run_method_7b() -> None:
            rng = _method_rng("method_7b_lazy_greedy_refined")
            lazy_refined = _lazy_greedy_refined(
                table,
                ranked_features,
                self.combo_sizes,
                mi_scores,
                rng,
            )
            for combo in lazy_refined:
                _add_candidate(combo, "method_7b_lazy_greedy_refined")

        _time_block("method_7b_lazy_greedy_refined", _run_method_7b)

        def _run_method_8() -> None:
            if len(top_pool) < 2:
                return
            rng = _method_rng("method_8_extratrees")
            extra_sets = _extra_trees_routes(
                table,
                top_pool,
                y,
                self.combo_sizes,
                self.rf_estimators,
                max(2, min(self.rf_max_depth or 3, 3)),
                rng,
            )
            for combo in extra_sets:
                _add_candidate(combo, "method_8_extratrees")

        _time_block("method_8_extratrees", _run_method_8)

        def _run_method_8b() -> None:
            if len(top_pool) < 2:
                return
            rng = _method_rng("method_8b_extratrees_refined")
            extra_refined = _extra_trees_refined(
                table,
                top_pool,
                y,
                self.combo_sizes,
                self.rf_estimators,
                max(2, min(self.rf_max_depth or 3, 3)),
                rng,
            )
            for combo in extra_refined:
                _add_candidate(combo, "method_8b_extratrees_refined")

        _time_block("method_8b_extratrees_refined", _run_method_8b)

        def _run_method_9() -> None:
            rng = _method_rng("method_9_countsketch")
            sketch_sets = _countsketch_candidates(
                table,
                y,
                ranked_features,
                self.combo_sizes,
                rng,
                max_features=self.filter_top_k,
            )
            for combo in sketch_sets:
                _add_candidate(combo, "method_9_countsketch")

        _time_block("method_9_countsketch", _run_method_9)

        def _run_method_9b() -> None:
            rng = _method_rng("method_9b_countsketch_refined")
            sketch_refined = _countsketch_refined_candidates(
                table,
                y,
                ranked_features,
                self.combo_sizes,
                rng,
            )
            for combo in sketch_refined:
                _add_candidate(combo, "method_9b_countsketch_refined")

        _time_block("method_9b_countsketch_refined", _run_method_9b)

        def _run_method_10() -> None:
            rng = _method_rng("method_10_gradient_synergy")
            grad_sets = _gradient_synergy_candidates(
                table,
                y,
                ranked_features,
                self.combo_sizes,
                rng,
            )
            for combo in grad_sets:
                _add_candidate(combo, "method_10_gradient_synergy")

        _time_block("method_10_gradient_synergy", _run_method_10)

        def _run_method_10b() -> None:
            rng = _method_rng("method_10b_gradient_hessian")
            grad_hessian_sets = _gradient_hessian_candidates(
                table,
                y,
                ranked_features,
                self.combo_sizes,
                rng,
            )
            for combo in grad_hessian_sets:
                _add_candidate(combo, "method_10b_gradient_hessian")

        _time_block("method_10b_gradient_hessian", _run_method_10b)

        def _run_method_11() -> None:
            rng = _method_rng("method_11_minhash_lsh")
            lsh_sets = _minhash_lsh_candidates(
                table,
                y,
                ranked_features,
                self.combo_sizes,
                rng,
            )
            for combo in lsh_sets:
                _add_candidate(combo, "method_11_minhash_lsh")

        _time_block("method_11_minhash_lsh", _run_method_11)

        def _run_method_11b() -> None:
            rng = _method_rng("method_11b_minhash_refined")
            lsh_refined_sets = _minhash_refined_candidates(
                table,
                y,
                ranked_features,
                self.combo_sizes,
                rng,
            )
            for combo in lsh_refined_sets:
                _add_candidate(combo, "method_11b_minhash_refined")

        _time_block("method_11b_minhash_refined", _run_method_11b)

        candidate_sets: Set[Tuple[str, ...]] = set()
        for combos in method_sets.values():
            candidate_sets.update(combos)

        if self.max_candidates_per_method is not None:
            limit = int(self.max_candidates_per_method)
            for method_key, combos in list(method_sets.items()):
                if len(combos) <= limit:
                    continue
                sorted_combos = sorted(combos)
                keep = set(sorted_combos[:limit])
                removed = set(sorted_combos[limit:])
                method_sets[method_key] = keep
                for combo in removed:
                    methods = candidate_sources.get(combo)
                    if not methods:
                        continue
                    methods.discard(method_key)
                    if not methods:
                        candidate_sources.pop(combo, None)
            candidate_sets = set()
            for combos in method_sets.values():
                candidate_sets.update(combos)

        candidate_list = sorted(candidate_sets)
        if self.max_total_candidates is not None and len(candidate_list) > self.max_total_candidates:
            candidate_list = candidate_list[: self.max_total_candidates]
        kept = set(candidate_list)
        self.method_candidate_sets_ = {k: {combo for combo in v if combo in kept} for k, v in method_sets.items()}
        self.candidate_sources_ = {combo: set(methods) for combo, methods in candidate_sources.items() if combo in kept}
        self.method_timings_ = {self.method_name_map_[k]: timings[k] for k in self.method_name_map_}
        self.all_candidate_sets_ = list(candidate_list)
        return candidate_list

    def _score_candidates(
        self,
        table: _Table,
        y: np.ndarray,
        candidates: Sequence[Tuple[str, ...]],
    ) -> List[SubspaceReport]:
        if not candidates:
            self.evaluated_reports_ = []
            self.candidate_reports_ = {}
            return []

        classes = self.classes_
        assert classes is not None
        baseline = max(self.baseline_f1_, 1e-9)
        results: List[SubspaceReport] = []
        counts = np.unique(y, return_counts=True)[1]
        n_splits = int(min(self.cv_splits, counts.min())) if counts.size else 0
        if n_splits < 2:
            self.evaluated_reports_ = []
            self.candidate_reports_ = {}
            return []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        for features in candidates:
            X_sub = _encode_features(table, features)
            if X_sub.size == 0:
                continue
            try:
                metrics = _evaluate_subspace(X_sub, y, skf, classes)
            except ValueError:
                continue
            var_ratio = _variance_ratio(X_sub, y)
            l1_imp = _logistic_l1_importance(X_sub, y, self.random_state)
            report = SubspaceReport(
                features=tuple(features),
                mean_macro_f1=metrics["macro_f1_mean"],
                std_macro_f1=metrics["macro_f1_std"],
                lift_vs_majority=metrics["macro_f1_mean"] / baseline,
                coverage_ratio=metrics["coverage_mean"],
                per_class_f1=metrics["per_class_f1"],
                support=int(metrics["support"]),
                variance_ratio=float(var_ratio),
                l1_importance=float(l1_imp),
            )
            results.append(report)

        results.sort(key=lambda r: r.mean_macro_f1, reverse=True)
        self.evaluated_reports_ = list(results)
        self.candidate_reports_ = {report.features: report for report in results}
        return self._pick_diverse_by_size(results)

    def _pick_diverse_by_size(
        self, ordered_reports: Sequence[SubspaceReport]
    ) -> List[SubspaceReport]:
        """Selecciona los mejores subespacios privilegiando variedad en tamaño."""

        if not ordered_reports:
            return []

        grouped: Dict[int, List[SubspaceReport]] = {}
        for report in ordered_reports:
            grouped.setdefault(len(report.features), []).append(report)

        ranked_sizes = sorted(
            grouped.keys(),
            key=lambda size: (
                -grouped[size][0].mean_macro_f1 if grouped[size] else float("inf"),
                size,
            ),
        )

        selection: List[SubspaceReport] = []
        while len(selection) < self.max_sets:
            added = False
            for size in ranked_sizes:
                group = grouped.get(size)
                if not group:
                    continue
                selection.append(group.pop(0))
                added = True
                if len(selection) >= self.max_sets:
                    break
            if not added:
                break
        return selection

    def _attach_planes(
        self,
        reports: List[SubspaceReport],
        records: Iterable[DeltaRecord],
    ) -> List[SubspaceReport]:
        records_list = list(records)
        if not records_list:
            return reports

        for report in reports:
            assert self.feature_names_ is not None
            indices = [self.feature_names_.index(f) for f in report.features]
            planes = _planes_from_records(records_list, indices)
            report.planes.extend(planes)
        return reports


# =============================================================
# Funciones auxiliares de datos
# =============================================================

def _ensure_table(X: Any) -> _Table:
    if hasattr(X, "columns") and hasattr(X, "__getitem__"):
        columns = list(getattr(X, "columns"))
        arrays = [np.asarray(X[col]) for col in columns]
    else:
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError("X debe ser 2D")
        columns = [f"x{i}" for i in range(arr.shape[1])]
        arrays = [arr[:, i] for i in range(arr.shape[1])]
    arrays = [np.asarray(a).reshape(-1) for a in arrays]
    return _Table(columns=columns, arrays=arrays)


def _majority_macro_f1(y: np.ndarray, classes: np.ndarray) -> float:
    counts = np.array([(y == c).sum() for c in classes], float)
    majority = classes[int(np.argmax(counts))]
    y_pred = np.full_like(y, majority)
    return float(f1_score(y, y_pred, average="macro"))


def _mutual_info_scores(table: _Table, y: np.ndarray, columns_subset: Optional[Sequence[str]] = None) -> Dict[str, float]:
    if columns_subset is None:
        columns = table.columns
    else:
        columns = [col for col in columns_subset if col in table.columns]
    parts = []
    discrete = []
    for name in columns:
        arr = table.column_data(name)
        if _is_numeric(arr) and not _is_bool(arr):
            values = _fill_numeric(arr.astype(float))
            parts.append(values.reshape(-1, 1))
            discrete.append(False)
        else:
            codes, _ = _factorize(arr)
            parts.append(codes.astype(float).reshape(-1, 1))
            discrete.append(True)
    if not parts:
        return {name: 0.0 for name in columns}
    X = np.hstack(parts)
    mi = mutual_info_classif(X, y, discrete_features=np.array(discrete))
    return {col: float(score) for col, score in zip(columns, mi)}


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts / total
    mask = probs > 0
    return float(-np.sum(probs[mask] * np.log2(probs[mask])))


def _stump_entropy_drop(table: _Table, y: np.ndarray) -> Dict[str, float]:
    if y.size == 0:
        return {name: 0.0 for name in table.columns}
    classes, y_codes = np.unique(y, return_inverse=True)
    base_entropy = _entropy_from_counts(np.bincount(y_codes, minlength=classes.size))
    scores: Dict[str, float] = {}
    for name in table.columns:
        values = table.column_data(name)
        if values.size == 0:
            scores[name] = 0.0
            continue
        if _is_numeric(values) and not _is_bool(values):
            arr = _fill_numeric(np.asarray(values, float))
            mask = ~np.isnan(arr)
            if mask.sum() < 2:
                scores[name] = 0.0
                continue
            thr = float(np.median(arr[mask]))
            left_mask = mask & (arr <= thr)
            right_mask = mask & (arr > thr)
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                scores[name] = 0.0
                continue
            left_counts = np.bincount(y_codes[left_mask], minlength=classes.size)
            right_counts = np.bincount(y_codes[right_mask], minlength=classes.size)
            weight_left = left_counts.sum()
            weight_right = right_counts.sum()
            weighted_entropy = (
                weight_left * _entropy_from_counts(left_counts)
                + weight_right * _entropy_from_counts(right_counts)
            ) / max(weight_left + weight_right, 1e-12)
            scores[name] = float(max(0.0, base_entropy - weighted_entropy))
        else:
            codes, _ = _factorize(values)
            mask = codes >= 0
            if mask.sum() < 2:
                scores[name] = 0.0
                continue
            best_drop = 0.0
            for level in range(int(codes[mask].max()) + 1):
                level_mask = mask & (codes == level)
                if level_mask.sum() == 0 or (~level_mask & mask).sum() == 0:
                    continue
                left_counts = np.bincount(y_codes[level_mask], minlength=classes.size)
                right_counts = np.bincount(y_codes[mask & (codes != level)], minlength=classes.size)
                weight_left = left_counts.sum()
                weight_right = right_counts.sum()
                weighted_entropy = (
                    weight_left * _entropy_from_counts(left_counts)
                    + weight_right * _entropy_from_counts(right_counts)
                ) / max(weight_left + weight_right, 1e-12)
                best_drop = max(best_drop, base_entropy - weighted_entropy)
            scores[name] = float(max(0.0, best_drop))
    return scores


def _chi2_scores(columns: Dict[str, np.ndarray], y_codes: np.ndarray, classes: np.ndarray) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for name, values in columns.items():
        if values.size == 0:
            scores[name] = 0.0
            continue
        codes, _ = _factorize(values)
        mask = codes >= 0
        if not mask.any():
            scores[name] = 0.0
            continue
        codes = codes[mask]
        y_col = y_codes[mask]
        n_levels = int(codes.max()) + 1
        observed = np.zeros((n_levels, classes.size), float)
        for class_idx in range(classes.size):
            cls_mask = y_col == class_idx
            if not np.any(cls_mask):
                continue
            observed[:, class_idx] = np.bincount(codes[cls_mask], minlength=n_levels)
        row_sum = observed.sum(axis=1, keepdims=True)
        col_sum = observed.sum(axis=0, keepdims=True)
        total = float(observed.sum())
        expected = (row_sum @ col_sum) / max(total, 1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            contrib = (observed - expected) ** 2 / (expected + 1e-12)
        scores[name] = float(np.nan_to_num(contrib).sum())
    return scores


def _rank_features(
    mi_scores: Dict[str, float],
    chi2_cat: Dict[str, float],
    chi2_num: Dict[str, float],
    stump_scores: Dict[str, float],
    numeric_cols: List[str],
    categorical_cols: List[str],
    top_k: int,
) -> List[str]:
    combined: Dict[str, float] = {}
    max_mi = max(mi_scores.values()) if mi_scores else 1.0
    max_chi_cat = max(chi2_cat.values()) if chi2_cat else 1.0
    max_chi_num = max(chi2_num.values()) if chi2_num else 1.0
    max_stump = max(stump_scores.values()) if stump_scores else 1.0

    for col in numeric_cols:
        mi_part = mi_scores.get(col, 0.0) / (max_mi + 1e-12)
        chi_part = chi2_num.get(col, 0.0) / (max_chi_num + 1e-12)
        stump_part = stump_scores.get(col, 0.0) / (max_stump + 1e-12)
        combined[col] = 0.5 * mi_part + 0.2 * chi_part + 0.3 * stump_part

    for col in categorical_cols:
        mi_part = mi_scores.get(col, 0.0) / (max_mi + 1e-12)
        chi_part = chi2_cat.get(col, 0.0) / (max_chi_cat + 1e-12)
        stump_part = stump_scores.get(col, 0.0) / (max_stump + 1e-12)
        combined[col] = 0.5 * mi_part + 0.4 * chi_part + 0.1 * stump_part

    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:top_k]]


def _quantile_binning(table: _Table, numeric_cols: Sequence[str], max_bins: int = 8) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    if not numeric_cols:
        return result
    q = max(2, min(max_bins, int(round(math.sqrt(len(table))) or 2)))
    for name in numeric_cols:
        values = np.asarray(table.column_data(name), float)
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            continue
        nunique = np.unique(values[mask]).size
        if nunique < 2:
            continue
        bins = min(q, nunique)
        edges = np.quantile(values[mask], np.linspace(0.0, 1.0, bins + 1))
        edges = np.unique(edges)
        if edges.size <= 1:
            continue
        codes = np.full(values.shape, -1, int)
        codes[mask] = np.digitize(values[mask], edges[1:-1], right=True)
        result[name] = codes
    return result


def _mi_neighbor_combos(ranked: Sequence[str], mi_scores: Dict[str, float], combo_sizes: Sequence[int]) -> Set[Tuple[str, ...]]:
    combos: Set[Tuple[str, ...]] = set()
    ordered = [f for f in ranked if f in mi_scores]
    for size in combo_sizes:
        if size <= 1 or len(ordered) < size:
            continue
        for i in range(len(ordered) - size + 1):
            window = ordered[i : i + size]
            vals = [mi_scores.get(f, 0.0) for f in window]
            vmax = max(vals)
            vmin = min(vals)
            if vmax <= 0:
                combos.add(tuple(sorted(window)))
            else:
                if (vmax - vmin) / (vmax + 1e-12) <= 0.15:
                    combos.add(tuple(sorted(window)))
    return combos


def _high_corr_groups(table: _Table, numeric_cols: Sequence[str], threshold: float = 0.55) -> List[Tuple[str, ...]]:
    if len(numeric_cols) < 2:
        return []
    matrix = np.column_stack([_fill_numeric(table.column_data(name).astype(float)) for name in numeric_cols])
    if matrix.shape[1] < 2:
        return []
    corr = np.corrcoef(matrix, rowvar=False)
    groups: List[Tuple[str, ...]] = []
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr[i, j]) >= threshold:
                groups.append((numeric_cols[i], numeric_cols[j]))
    return groups


def _rf_pairs(
    table: _Table,
    feature_names: Sequence[str],
    y: np.ndarray,
    *,
    n_estimators: int,
    max_depth: Optional[int],
    random_state: Optional[int],
) -> List[Tuple[str, str]]:
    if len(feature_names) < 2:
        return []
    arr = []
    for name in feature_names:
        values = table.column_data(name)
        if _is_numeric(values) and not _is_bool(values):
            arr.append(_fill_numeric(values.astype(float)))
        else:
            codes, _ = _factorize(values)
            arr.append(codes.astype(float))
    X = np.column_stack(arr)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X, y)
    counts: Dict[Tuple[int, int], int] = {}
    for tree in rf.estimators_:
        used = tree.tree_.feature
        used = used[used >= 0]
        if used.size < 2:
            continue
        uniq = np.unique(used)
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                pair = (int(uniq[i]), int(uniq[j]))
                counts[pair] = counts.get(pair, 0) + 1
    sorted_pairs = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    name_pairs = [(feature_names[i], feature_names[j]) for (i, j), _ in sorted_pairs]
    return name_pairs[: max(10, len(name_pairs))]


def _table_to_matrix(table: _Table, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.empty((len(table), 0), float)
    parts: List[np.ndarray] = []
    for name in columns:
        values = table.column_data(name)
        if _is_numeric(values) and not _is_bool(values):
            parts.append(_fill_numeric(values.astype(float)))
        else:
            codes, _ = _factorize(values)
            parts.append(codes.astype(float))
    return np.column_stack(parts) if parts else np.empty((len(table), 0), float)


def _sample_combinations(
    elements: Sequence[str], size: int, rng: np.random.RandomState, max_draws: int
) -> Set[Tuple[str, ...]]:
    combos: Set[Tuple[str, ...]] = set()
    if size <= 0 or len(elements) < size:
        return combos
    n_possible = math.comb(len(elements), size)
    if n_possible <= max_draws:
        combos.update(tuple(sorted(combo)) for combo in combinations(elements, size))
        return combos
    seen: Set[Tuple[int, ...]] = set()
    while len(combos) < max_draws:
        idx = tuple(sorted(rng.choice(len(elements), size=size, replace=False)))
        if idx in seen:
            continue
        seen.add(idx)
        combos.add(tuple(sorted(elements[i] for i in idx)))
    return combos


def _topk_guided_candidates(
    table: _Table,
    ranked_features: Sequence[str],
    mi_scores: Dict[str, float],
    combo_sizes: Sequence[int],
    rng: np.random.RandomState,
    *,
    max_pool: int = 32,
    max_draws: int = 420,
) -> Set[Tuple[str, ...]]:
    pool = ranked_features[: min(len(ranked_features), max_pool)]
    if len(pool) < 2:
        return set()
    X = _table_to_matrix(table, pool)
    if X.size == 0:
        return set()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    corr = np.corrcoef(Xs, rowvar=False)
    if np.isnan(corr).any():
        corr = np.nan_to_num(corr, nan=0.0)
    combos: Set[Tuple[str, ...]] = set()
    for size in combo_sizes:
        if len(pool) < size or size <= 1:
            continue
        if math.comb(len(pool), size) <= max_draws:
            candidate_indices = [tuple(idx) for idx in combinations(range(len(pool)), size)]
        else:
            candidate_indices = list(
                _sample_combinations(tuple(range(len(pool))), size, rng, max_draws)
            )
        scored: List[Tuple[float, Tuple[str, ...]]] = []
        for idxs in candidate_indices:
            names = tuple(sorted(pool[int(i)] for i in idxs))
            mi_sum = sum(mi_scores.get(name, 0.0) for name in names)
            if mi_sum <= 0:
                continue
            penalty = 0.0
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    penalty += 0.22 * abs(corr[int(idxs[i]), int(idxs[j])])
            diversity = min(mi_scores.get(name, 0.0) for name in names)
            score = mi_sum - penalty + 0.18 * diversity
            scored.append((score, names))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        for _, combo in scored[: max(20, min(45, len(scored)) )]:
            combos.add(combo)
    return combos


def _chi2_guided_candidates(
    table: _Table,
    chi2_scores: Dict[str, float],
    mi_scores: Dict[str, float],
    combo_sizes: Sequence[int],
    *,
    max_pool: int = 28,
    max_results: int = 40,
) -> Set[Tuple[str, ...]]:
    if not chi2_scores:
        return set()
    ordered = sorted(chi2_scores.items(), key=lambda kv: kv[1], reverse=True)
    pool = [name for name, _ in ordered[: min(len(ordered), max_pool)]]
    combos: List[Tuple[float, Tuple[str, ...]]] = []
    uniqueness_cache: Dict[str, float] = {}
    for size in combo_sizes:
        if size <= 1 or len(pool) < size:
            continue
        for combo in combinations(pool, size):
            chi2_sum = sum(chi2_scores.get(name, 0.0) for name in combo)
            mi_sum = sum(mi_scores.get(name, 0.0) for name in combo)
            diversity = 0.0
            for name in combo:
                if name not in uniqueness_cache:
                    values = table.column_data(name)
                    uniqueness_cache[name] = math.log1p(len(np.unique(values)))
                diversity += uniqueness_cache[name]
            score = chi2_sum + 0.45 * mi_sum + 0.12 * diversity
            combos.append((score, tuple(sorted(combo))))
    combos.sort(key=lambda kv: kv[0], reverse=True)
    top = combos[: max_results]
    return {combo for _, combo in top}


def _correlation_guided_candidates(
    table: _Table,
    numeric_cols: Sequence[str],
    mi_scores: Dict[str, float],
    combo_sizes: Sequence[int],
    threshold: float,
    rng: np.random.RandomState,
    *,
    max_pool: int = 36,
) -> Set[Tuple[str, ...]]:
    pool = [col for col in numeric_cols if col in table.columns][: min(max_pool, len(numeric_cols))]
    if len(pool) < 2:
        return set()
    X = _table_to_matrix(table, pool)
    if X.size == 0:
        return set()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    corr = np.corrcoef(Xs, rowvar=False)
    if np.isnan(corr).any():
        corr = np.nan_to_num(corr, nan=0.0)
    combos: List[Tuple[float, Tuple[str, ...]]] = []
    for size in combo_sizes:
        if len(pool) < size or size <= 1:
            continue
        total = math.comb(len(pool), size)
        if total <= 320:
            indices_iter = combinations(range(len(pool)), size)
        else:
            indices_iter = _sample_combinations(tuple(range(len(pool))), size, rng, 320)
        for idxs in indices_iter:
            idxs = tuple(int(i) for i in idxs)
            names = tuple(sorted(pool[i] for i in idxs))
            corr_vals = []
            valid = True
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    value = abs(corr[idxs[i], idxs[j]])
                    if value < threshold * 0.85:
                        valid = False
                        break
                    corr_vals.append(value)
                if not valid:
                    break
            if not valid or not corr_vals:
                continue
            corr_score = sum(corr_vals) / len(corr_vals)
            mi_sum = sum(mi_scores.get(name, 0.0) for name in names)
            score = corr_score * (1.0 + 0.35 * mi_sum)
            combos.append((score, names))
    combos.sort(key=lambda kv: kv[0], reverse=True)
    return {combo for _, combo in combos[: min(60, len(combos))]}


def _rf_weighted_candidates(
    table: _Table,
    feature_names: Sequence[str],
    y: np.ndarray,
    combo_sizes: Sequence[int],
    *,
    n_estimators: int,
    max_depth: Optional[int],
    random_state: Optional[int],
) -> Set[Tuple[str, ...]]:
    if len(feature_names) < 2:
        return set()
    X = _table_to_matrix(table, feature_names)
    if X.size == 0:
        return set()
    rf = RandomForestClassifier(
        n_estimators=max(n_estimators, 60),
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
        bootstrap=False,
    )
    try:
        rf.fit(X, y)
    except Exception:
        return set()
    weights: Dict[Tuple[int, ...], float] = defaultdict(float)
    for tree in rf.estimators_:
        tree_ = tree.tree_
        stack: List[Tuple[int, Tuple[int, ...], float]] = [(0, tuple(), 0.0)]
        while stack:
            node, path, gain_sum = stack.pop()
            feature = tree_.feature[node]
            left = tree_.children_left[node]
            right = tree_.children_right[node]
            if left == -1 or right == -1:
                if not path:
                    continue
                unique = tuple(sorted(set(path)))
                for size in combo_sizes:
                    if len(unique) < size or size <= 1:
                        continue
                    for combo in combinations(unique, size):
                        weights[tuple(sorted(combo))] = max(weights[tuple(sorted(combo))], gain_sum)
                continue
            samples = float(tree_.weighted_n_node_samples[node])
            if samples <= 0:
                gain = 0.0
            else:
                left_weight = float(tree_.weighted_n_node_samples[left])
                right_weight = float(tree_.weighted_n_node_samples[right])
                weighted_children = (
                    left_weight * tree_.impurity[left] + right_weight * tree_.impurity[right]
                ) / max(samples, 1e-9)
                gain = max(tree_.impurity[node] - weighted_children, 0.0)
            new_path = path + ((feature,) if feature >= 0 else tuple())
            stack.append((left, new_path, gain_sum + gain))
            stack.append((right, new_path, gain_sum + gain))
    scored: List[Tuple[float, Tuple[str, ...]]] = []
    for combo_idx, score in weights.items():
        names = tuple(sorted(feature_names[i] for i in combo_idx))
        scored.append((score, names))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    return {combo for _, combo in scored[: min(70, len(scored))]}


def _leverage_class_candidates(
    table: _Table,
    y: np.ndarray,
    feature_names: Sequence[str],
    combo_sizes: Sequence[int],
    top_k: int,
    rng: np.random.RandomState,
) -> Set[Tuple[str, ...]]:
    if not feature_names:
        return set()
    X = _table_to_matrix(table, feature_names)
    if X.size == 0:
        return set()
    X = X.astype(float, copy=False)
    X = np.nan_to_num(X)
    X_centered = X - X.mean(axis=0, keepdims=True)
    rank = max(1, min(X_centered.shape[1], 6))
    try:
        _, _, Vt = randomized_svd(X_centered, n_components=rank, random_state=rng)
    except Exception:
        return set()
    leverage_global = np.sum(Vt**2, axis=0)
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    class_leverage = np.zeros_like(leverage_global)
    class_means = []
    for cls in classes:
        mask = y_arr == cls
        if np.sum(mask) < 2:
            continue
        Xc = X[mask]
        class_means.append(Xc.mean(axis=0))
        Xc = Xc - Xc.mean(axis=0, keepdims=True)
        try:
            _, _, Vc = randomized_svd(
                Xc,
                n_components=min(rank, max(1, min(Xc.shape[1], 4))),
                random_state=rng,
            )
            class_leverage += np.sum(Vc**2, axis=0)
        except Exception:
            continue
    if class_means:
        between = np.var(np.vstack(class_means), axis=0)
    else:
        between = np.zeros_like(leverage_global)
    scores = leverage_global + 0.65 * class_leverage + 1.1 * between
    order = np.argsort(-scores)
    top = [feature_names[i] for i in order[: min(top_k, len(feature_names))]]
    score_map = {feature_names[i]: float(scores[i]) for i in range(len(feature_names))}
    combos: List[Tuple[float, Tuple[str, ...]]] = []
    for size in combo_sizes:
        if len(top) < size or size <= 1:
            continue
        for combo in combinations(top, size):
            base = sum(score_map[name] for name in combo)
            penalty = 0.0
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    penalty += 0.1 * abs(score_map[combo[i]] - score_map[combo[j]])
            combos.append((base - penalty, tuple(sorted(combo))))
    combos.sort(key=lambda kv: kv[0], reverse=True)
    return {combo for _, combo in combos[: min(60, len(combos))]}


def _sparse_projection_guided_candidates(
    table: _Table,
    y: np.ndarray,
    ranked_features: Sequence[str],
    combo_sizes: Sequence[int],
    rng: np.random.RandomState,
    *,
    n_projections: int = 30,
    density: float = 0.12,
) -> Set[Tuple[str, ...]]:
    pool = ranked_features[: min(len(ranked_features), 45)]
    if len(pool) < 2:
        return set()
    X = _table_to_matrix(table, pool)
    if X.size == 0:
        return set()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_features = Xs.shape[1]
    classes = np.unique(y)
    feature_weights: Dict[int, float] = defaultdict(float)
    projection_sets: List[Tuple[float, List[int]]] = []
    non_zero = max(1, int(math.ceil(density * n_features)))
    for _ in range(n_projections):
        idx = rng.choice(n_features, size=min(non_zero, n_features), replace=False)
        vec = np.zeros(n_features)
        vec[idx] = rng.choice([-1.0, 1.0], size=idx.size)
        proj = Xs @ vec
        mu = proj.mean()
        between = 0.0
        within = 0.0
        for cls in classes:
            mask = y == cls
            if not np.any(mask):
                continue
            cls_proj = proj[mask]
            cls_mu = cls_proj.mean()
            between += mask.sum() * (cls_mu - mu) ** 2
            within += float(np.sum((cls_proj - cls_mu) ** 2))
        fisher = between / (within + 1e-9)
        if fisher <= 0:
            continue
        order = np.argsort(-np.abs(vec))[: max(6, max(combo_sizes) + 2)]
        top_idx = [int(i) for i in order if abs(vec[i]) > 0]
        if not top_idx:
            continue
        for i in top_idx:
            feature_weights[i] += float(fisher * abs(vec[i]))
        projection_sets.append((float(fisher), top_idx))
    if not projection_sets:
        return set()
    combos: Set[Tuple[str, ...]] = set()
    weight_items = sorted(feature_weights.items(), key=lambda kv: kv[1], reverse=True)
    ranked_idx = [idx for idx, _ in weight_items[: max(18, max(combo_sizes) * 5)]]
    for size in combo_sizes:
        if size <= 1 or not ranked_idx or len(ranked_idx) < size:
            continue
        scored: List[Tuple[float, Tuple[str, ...]]] = []
        for combo in combinations(ranked_idx, size):
            score = sum(feature_weights[i] for i in combo)
            scored.append((score, tuple(sorted(pool[i] for i in combo))))
        scored.sort(key=lambda kv: kv[0], reverse=True)
        for _, combo in scored[: min(35, len(scored))]:
            combos.add(combo)
    for fisher, idxs in projection_sets[: min(10, len(projection_sets))]:
        if len(idxs) < 2:
            continue
        names = [pool[i] for i in idxs]
        for size in combo_sizes:
            if len(names) < size or size <= 1:
                continue
            combos.update(_sample_combinations(names, size, rng, max_draws=6))
    return combos


def _lazy_greedy_refined(
    table: _Table,
    ranked_features: Sequence[str],
    combo_sizes: Sequence[int],
    base_scores: Dict[str, float],
    rng: np.random.RandomState,
    *,
    max_pool: int = 35,
) -> Set[Tuple[str, ...]]:
    pool = ranked_features[: min(len(ranked_features), max_pool)]
    if len(pool) < 2:
        return set()
    X = _table_to_matrix(table, pool)
    if X.size == 0:
        return set()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cov = np.cov(Xs, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    if np.isnan(cov).any():
        cov = np.nan_to_num(cov, nan=0.0)
    base = np.array([base_scores.get(name, 0.0) for name in pool])
    var = np.diag(cov)
    combos: Set[Tuple[str, ...]] = set()

    def combo_score(idxs: Tuple[int, ...]) -> float:
        idxs = tuple(sorted(idxs))
        weight = float(base[list(idxs)].sum() + 0.15 * var[list(idxs)].sum())
        penalty = 0.0
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                penalty += 0.18 * abs(cov[idxs[i], idxs[j]])
        sub = Xs[:, list(idxs)]
        try:
            cov_sub = np.cov(sub, rowvar=False)
            if cov_sub.ndim == 0:
                cov_sub = np.array([[float(cov_sub)]])
            sign, logdet = np.linalg.slogdet(cov_sub + 1e-6 * np.eye(cov_sub.shape[0]))
            scatter = float(logdet) if sign > 0 else 0.0
        except Exception:
            scatter = 0.0
        return weight - penalty + 0.25 * scatter

    seeds = np.argsort(-base)[: max(6, min(len(pool), 10))]
    for size in sorted(set(combo_sizes)):
        if size <= 1 or len(pool) < size:
            continue
        for seed in seeds:
            current = (int(seed),)
            available = [i for i in range(len(pool)) if i != seed]
            while len(current) < size and available:
                scores = [combo_score(tuple(sorted(current + (candidate,)))) for candidate in available]
                best_idx = int(np.argmax(scores))
                candidate = available.pop(best_idx)
                current = tuple(sorted(current + (candidate,)))
            if len(current) == size:
                combos.add(tuple(sorted(pool[i] for i in current)))
    for _ in range(max(3, len(seeds) // 2)):
        chosen = rng.choice(len(pool), size=max(combo_sizes), replace=False)
        for size in combo_sizes:
            if size <= 1:
                continue
            subset = tuple(sorted(pool[i] for i in chosen[:size]))
            combos.add(subset)
    return combos


def _extra_trees_refined(
    table: _Table,
    feature_names: Sequence[str],
    y: np.ndarray,
    combo_sizes: Sequence[int],
    n_estimators: int,
    max_depth: int,
    rng: np.random.RandomState,
) -> Set[Tuple[str, ...]]:
    if len(feature_names) < 2:
        return set()
    X = _table_to_matrix(table, feature_names)
    if X.size == 0:
        return set()
    clf = ExtraTreesClassifier(
        n_estimators=max(60, n_estimators),
        max_depth=max_depth,
        n_jobs=-1,
        random_state=int(rng.randint(0, 2**32)),
        bootstrap=False,
    )
    try:
        clf.fit(X, y)
    except Exception:
        return set()
    weights: Dict[Tuple[int, ...], float] = defaultdict(float)
    for tree in clf.estimators_:
        tree_ = tree.tree_
        stack: List[Tuple[int, Tuple[int, ...]]] = [(0, tuple())]
        while stack:
            node, path = stack.pop()
            feature = tree_.feature[node]
            left = tree_.children_left[node]
            right = tree_.children_right[node]
            if left == -1 or right == -1:
                if not path:
                    continue
                counts = tree_.value[node][0]
                total = np.sum(counts)
                if total <= 0:
                    continue
                purity = float(np.max(counts) / total)
                unique = tuple(sorted(set(path)))
                depth_penalty = 1.0 / (1.0 + len(unique))
                score = purity / depth_penalty
                for size in combo_sizes:
                    if len(unique) < size or size <= 1:
                        continue
                    for combo in combinations(unique, size):
                        weights[tuple(sorted(combo))] = max(weights[tuple(sorted(combo))], score)
                continue
            new_path = path + ((feature,) if feature >= 0 else tuple())
            stack.append((left, new_path))
            stack.append((right, new_path))
    scored: List[Tuple[float, Tuple[str, ...]]] = []
    for combo_idx, score in weights.items():
        names = tuple(sorted(feature_names[i] for i in combo_idx))
        scored.append((score, names))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    return {combo for _, combo in scored[: min(70, len(scored))]}


def _countsketch_refined_candidates(
    table: _Table,
    y: np.ndarray,
    ranked_features: Sequence[str],
    combo_sizes: Sequence[int],
    rng: np.random.RandomState,
    *,
    max_features: int = 30,
    buckets: int = 257,
) -> Set[Tuple[str, ...]]:
    features = ranked_features[: min(len(ranked_features), max_features)]
    if len(features) < 2:
        return set()
    encoded: List[np.ndarray] = []
    for name in features:
        values = table.column_data(name)
        if _is_numeric(values) and not _is_bool(values):
            quantiles = np.quantile(_fill_numeric(values.astype(float)), [0.2, 0.4, 0.6, 0.8])
            codes = np.digitize(values, quantiles, right=False)
        else:
            codes, _ = _factorize(values)
        encoded.append(codes.astype(np.int64))
    data = np.column_stack(encoded)
    primes = np.array([3, 5, 7, 11, 13, 17, 19, 23, 29, 31], dtype=np.int64)
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    scores: Dict[Tuple[str, ...], float] = {}
    for size in combo_sizes:
        if size <= 1 or data.shape[1] < size:
            continue
        max_eval = min(220, math.comb(data.shape[1], size))
        iterator = _sample_combinations(tuple(range(data.shape[1])), size, rng, max_eval)
        for combo in iterator:
            idxs = [int(c) for c in combo]
            hashed = np.zeros(data.shape[0], dtype=np.int64)
            for pos, idx in enumerate(idxs):
                hashed ^= ((data[:, idx].astype(np.int64) + 1) * primes[pos % primes.size]) % buckets
            class_scores: List[float] = []
            for cls in classes:
                mask = y_arr == cls
                if not np.any(mask):
                    continue
                counts = np.bincount(hashed[mask], minlength=buckets)
                if counts.size == 0:
                    continue
                ratio = counts.max() / max(1, mask.sum())
                class_scores.append(ratio)
            if not class_scores:
                continue
            best = max(class_scores)
            if len(class_scores) > 1:
                sorted_scores = sorted(class_scores, reverse=True)
                contrast = sorted_scores[0] - sorted_scores[1]
            else:
                contrast = class_scores[0]
            score = best + 0.6 * contrast
            names = tuple(sorted(features[i] for i in idxs))
            if score > scores.get(names, 0.0):
                scores[names] = score
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {combo for combo, _ in ranked[: min(80, len(ranked))]}


def _gradient_hessian_candidates(
    table: _Table,
    y: np.ndarray,
    ranked_features: Sequence[str],
    combo_sizes: Sequence[int],
    rng: np.random.RandomState,
    *,
    max_features: int = 40,
    top_pairs: int = 90,
) -> Set[Tuple[str, ...]]:
    features = ranked_features[: min(len(ranked_features), max_features)]
    if len(features) < 2:
        return set()
    X = _table_to_matrix(table, features)
    if X.size == 0:
        return set()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        penalty="l2",
        C=0.8,
        solver="lbfgs",
        max_iter=200,
    )
    try:
        y_arr = np.asarray(y)
        clf.fit(Xs, y_arr)
    except Exception:
        return set()
    probs = clf.predict_proba(Xs)
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(y_onehot.shape[0]), y_arr] = 1.0
    grad = probs - y_onehot
    grad_weights = np.linalg.norm(grad, axis=1)
    WX = Xs * grad_weights[:, None]
    G = Xs.T @ WX
    h_weights = np.sum(probs * (1 - probs), axis=1)
    HX = Xs * h_weights[:, None]
    H = Xs.T @ HX
    pair_scores: List[Tuple[float, Tuple[int, int]]] = []
    for i in range(Xs.shape[1]):
        for j in range(i + 1, Xs.shape[1]):
            score = abs(G[i, j]) + 0.35 * abs(H[i, j])
            if score <= 0:
                continue
            pair_scores.append((score, (i, j)))
    pair_scores.sort(reverse=True)
    combos: Set[Tuple[str, ...]] = set()
    for score, (i, j) in pair_scores[:top_pairs]:
        names = tuple(sorted((features[i], features[j])))
        combos.add(names)
    for size in combo_sizes:
        if size <= 2 or len(features) < size:
            continue
        for _, (i, j) in pair_scores[:top_pairs]:
            selected = {i, j}
            order = np.argsort(-np.abs(G[i] + G[j]))
            for idx in order:
                if idx in selected:
                    continue
                selected.add(int(idx))
                if len(selected) == size:
                    break
            if len(selected) == size:
                combos.add(tuple(sorted(features[k] for k in selected)))
    for _ in range(min(12, len(pair_scores))):
        chosen = set()
        while len(chosen) < max(combo_sizes):
            chosen.add(int(rng.choice(len(features))))
        for size in combo_sizes:
            if size <= 1:
                continue
            subset = tuple(sorted(features[idx] for idx in list(chosen)[:size]))
            combos.add(subset)
    return combos


def _minhash_refined_candidates(
    table: _Table,
    y: np.ndarray,
    ranked_features: Sequence[str],
    combo_sizes: Sequence[int],
    rng: np.random.RandomState,
    *,
    num_perm: int = 24,
    rows_per_band: int = 4,
) -> Set[Tuple[str, ...]]:
    features = ranked_features[: min(len(ranked_features), 45)]
    if len(features) < 2:
        return set()
    bool_matrix: List[np.ndarray] = []
    for name in features:
        values = table.column_data(name)
        if _is_numeric(values) and not _is_bool(values):
            thresh_low, thresh_high = np.quantile(_fill_numeric(values.astype(float)), [0.4, 0.6])
            bool_matrix.append((values > thresh_high).astype(bool))
            bool_matrix.append((values < thresh_low).astype(bool))
        else:
            codes, _ = _factorize(values)
            modes = np.bincount(codes)
            majority = np.argmax(modes) if modes.size else 0
            bool_matrix.append((codes == majority))
    if not bool_matrix:
        return set()
    bool_arr = np.vstack(bool_matrix)
    hashes = rng.randint(0, 2**32 - 1, size=(num_perm, bool_arr.shape[1]), dtype=np.uint64)
    signatures = np.empty((bool_arr.shape[0], num_perm), dtype=np.uint64)
    for idx, row in enumerate(bool_arr):
        mask = row.astype(bool)
        if not np.any(mask):
            signatures[idx] = np.uint64(2**32 - 1)
            continue
        hashed = hashes[:, mask]
        signatures[idx] = hashed.min(axis=1)
    num_bands = max(1, num_perm // rows_per_band)
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
    for feature_idx in range(signatures.shape[0]):
        for band in range(num_bands):
            start = band * rows_per_band
            end = min(start + rows_per_band, num_perm)
            key = tuple(int(x) for x in signatures[feature_idx, start:end])
            if not key:
                continue
            buckets.setdefault((band, key), []).append(feature_idx)
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    scores: Dict[Tuple[str, ...], float] = {}
    for candidates in buckets.values():
        if len(candidates) < 2:
            continue
        names = [features[idx % len(features)] for idx in candidates]
        for size in combo_sizes:
            if size <= 1 or len(names) < size:
                continue
            for combo in combinations(names, size):
                idxs = [features.index(name) for name in combo]
                mask = np.ones(bool_arr.shape[1], dtype=bool)
                for idx in idxs:
                    mask &= bool_arr[idx]
                if not np.any(mask):
                    continue
                class_ratios = []
                for cls in classes:
                    cls_mask = y_arr == cls
                    if not np.any(cls_mask):
                        continue
                    overlap = np.sum(mask & cls_mask)
                    if overlap == 0:
                        continue
                    ratio = overlap / max(1, np.sum(cls_mask))
                    class_ratios.append(ratio)
                if not class_ratios:
                    continue
                best = max(class_ratios)
                sorted_ratios = sorted(class_ratios, reverse=True)
                contrast = sorted_ratios[0] - (sorted_ratios[1] if len(sorted_ratios) > 1 else 0.0)
                score = best + 0.5 * contrast
                combo_sorted = tuple(sorted(combo))
                if score > scores.get(combo_sorted, 0.0):
                    scores[combo_sorted] = score
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {combo for combo, _ in ranked[: min(80, len(ranked))]}


def _leverage_candidates(
    table: _Table,
    feature_names: Sequence[str],
    combo_sizes: Sequence[int],
    top_k: int,
    rng: np.random.RandomState,
) -> Set[Tuple[str, ...]]:
    if not feature_names:
        return set()
    X = _table_to_matrix(table, feature_names)
    if X.size == 0:
        return set()
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    rank = max(1, min(X_centered.shape[1] - 1, 8)) if X_centered.shape[1] > 1 else 1
    try:
        _, _, Vt = randomized_svd(X_centered, n_components=rank, random_state=rng)
    except Exception:
        return set()
    leverage = np.sum(Vt**2, axis=0)
    order = np.argsort(-leverage)
    top = [feature_names[i] for i in order[: min(top_k, len(feature_names))]]
    combos: Set[Tuple[str, ...]] = set()
    max_draws = min(18, max(8, top_k))
    for size in combo_sizes:
        combos.update(_sample_combinations(top, size, rng, max_draws))
    return combos


def _sparse_projection_candidates(
    table: _Table,
    y: np.ndarray,
    ranked_features: Sequence[str],
    combo_sizes: Sequence[int],
    rng: np.random.RandomState,
    *,
    n_projections: int = 24,
    density: float = 0.15,
) -> Set[Tuple[str, ...]]:
    pool = ranked_features[: max(10, min(len(ranked_features), 40))]
    if not pool or len(pool) < 2:
        return set()
    X = _table_to_matrix(table, pool)
    if X.size == 0:
        return set()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_features = Xs.shape[1]
    classes = np.unique(y)
    combos: Set[Tuple[str, ...]] = set()
    max_size = max(combo_sizes) if combo_sizes else 0
    top_per_projection = max(max_size + 1, 6)
    non_zero = max(1, int(math.ceil(density * n_features)))
    for _ in range(n_projections):
        idx = rng.choice(n_features, size=min(non_zero, n_features), replace=False)
        signs = rng.choice([-1.0, 1.0], size=idx.size)
        vec = np.zeros(n_features)
        vec[idx] = signs
        proj = Xs @ vec
        mu = proj.mean()
        between = 0.0
        within = 0.0
        for cls in classes:
            mask = y == cls
            if not np.any(mask):
                continue
            cls_proj = proj[mask]
            cls_mu = cls_proj.mean()
            between += mask.sum() * (cls_mu - mu) ** 2
            within += float(np.sum((cls_proj - cls_mu) ** 2))
        fisher = between / (within + 1e-9)
        if fisher <= 0:
            continue
        order = np.argsort(-np.abs(vec))[:top_per_projection]
        selected = [pool[i] for i in order if abs(vec[i]) > 0]
        for size in combo_sizes:
            if len(selected) < size:
                continue
            draw_cap = min(12, max(6, int(fisher * 3)))
            combos.update(_sample_combinations(selected, size, rng, draw_cap))
    return combos


def _lazy_greedy_candidates(
    table: _Table,
    ranked_features: Sequence[str],
    combo_sizes: Sequence[int],
    base_scores: Dict[str, float],
    rng: np.random.RandomState,
    *,
    max_seeds: int = 8,
) -> Set[Tuple[str, ...]]:
    pool = ranked_features[: max(10, min(len(ranked_features), 40))]
    if len(pool) < 2:
        return set()
    X = _table_to_matrix(table, pool)
    if X.size == 0:
        return set()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    corr = np.corrcoef(Xs, rowvar=False)
    if np.isnan(corr).any():
        corr = np.nan_to_num(corr, nan=0.0)
    base = np.array([base_scores.get(name, 0.0) for name in pool])
    order = np.argsort(-base)
    combos: Set[Tuple[str, ...]] = set()

    def combo_score(idxs: Tuple[int, ...]) -> float:
        score = float(base[list(idxs)].sum())
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                score -= 0.18 * abs(corr[idxs[i], idxs[j]])
        return score

    seeds = [idx for idx in order[: min(max_seeds, len(pool))] if base[idx] > -np.inf]
    for size in sorted(set(combo_sizes)):
        if size <= 1 or len(pool) < size:
            continue
        for seed in seeds:
            current = (seed,)
            available = [i for i in range(len(pool)) if i != seed]
            while len(current) < size and available:
                best_idx = None
                best_score = -float("inf")
                for candidate in available:
                    combo = tuple(sorted(current + (candidate,)))
                    score = combo_score(combo)
                    if score > best_score:
                        best_score = score
                        best_idx = candidate
                if best_idx is None:
                    break
                current = tuple(sorted(current + (best_idx,)))
                available.remove(best_idx)
            if len(current) == size:
                combos.add(tuple(sorted(pool[i] for i in current)))
    # random restarts for diversity
    for _ in range(max(1, len(seeds) // 2)):
        rng.shuffle(order)
        seed = order[0]
        for size in sorted(set(combo_sizes)):
            if size <= 1 or len(pool) < size:
                continue
            chosen = {seed}
            while len(chosen) < size:
                candidate = int(rng.choice(len(pool)))
                chosen.add(candidate)
            combo = tuple(sorted(pool[i] for i in chosen))
            combos.add(combo)
    return combos


def _extra_trees_routes(
    table: _Table,
    feature_names: Sequence[str],
    y: np.ndarray,
    combo_sizes: Sequence[int],
    n_estimators: int,
    max_depth: int,
    rng: np.random.RandomState,
) -> Set[Tuple[str, ...]]:
    if len(feature_names) < 2:
        return set()
    X = _table_to_matrix(table, feature_names)
    if X.size == 0:
        return set()
    clf = ExtraTreesClassifier(
        n_estimators=max(30, n_estimators // 2),
        max_depth=max_depth,
        n_jobs=-1,
        random_state=int(rng.randint(0, 2**32)),
        bootstrap=False,
    )
    try:
        clf.fit(X, y)
    except Exception:
        return set()
    combos: Set[Tuple[str, ...]] = set()
    for tree in clf.estimators_:
        used = tree.tree_.feature
        used = used[used >= 0]
        if used.size < 2:
            continue
        uniq = np.unique(used)
        names = [feature_names[i] for i in uniq]
        local_rng = np.random.RandomState(int(rng.randint(0, 2**32)))
        for size in combo_sizes:
            combos.update(_sample_combinations(names, size, local_rng, 3))
    return combos


def _countsketch_candidates(
    table: _Table,
    y: np.ndarray,
    ranked_features: Sequence[str],
    combo_sizes: Sequence[int],
    rng: np.random.RandomState,
    *,
    max_features: int = 25,
    buckets: int = 257,
) -> Set[Tuple[str, ...]]:
    features = ranked_features[: min(len(ranked_features), max_features)]
    if len(features) < 2:
        return set()
    encoded: List[np.ndarray] = []
    for name in features:
        values = table.column_data(name)
        if _is_numeric(values) and not _is_bool(values):
            quantiles = np.quantile(_fill_numeric(values.astype(float)), [0.25, 0.5, 0.75])
            codes = np.digitize(values, quantiles, right=False)
        else:
            codes, _ = _factorize(values)
        encoded.append(codes.astype(int))
    data = np.column_stack(encoded)
    combos: Set[Tuple[str, ...]] = set()
    primes = np.array([811, 1_201, 1_703, 1_999, 2_189], dtype=np.int64)
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    for size in combo_sizes:
        if size <= 1 or data.shape[1] < size:
            continue
        max_eval = min(180, math.comb(data.shape[1], size))
        iterator = _sample_combinations(list(range(data.shape[1])), size, rng, max_eval)
        for combo in iterator:
            idxs = [int(c) for c in combo]
            hashed = np.zeros(data.shape[0], dtype=np.int64)
            for pos, idx in enumerate(idxs):
                hashed ^= ((data[:, idx].astype(np.int64) + 1) * primes[pos % primes.size]) % buckets
            purity = 0.0
            for cls in classes:
                mask = y_arr == cls
                if not np.any(mask):
                    continue
                counts = np.bincount(hashed[mask], minlength=buckets)
                if counts.size == 0:
                    continue
                cls_ratio = counts.max() / max(1, mask.sum())
                purity = max(purity, cls_ratio)
            if purity >= 0.55:
                combos.add(tuple(sorted(features[i] for i in idxs)))
    return combos


def _gradient_synergy_candidates(
    table: _Table,
    y: np.ndarray,
    ranked_features: Sequence[str],
    combo_sizes: Sequence[int],
    rng: np.random.RandomState,
    *,
    max_features: int = 35,
    top_pairs: int = 70,
) -> Set[Tuple[str, ...]]:
    features = ranked_features[: min(len(ranked_features), max_features)]
    if len(features) < 2:
        return set()
    X = _table_to_matrix(table, features)
    if X.size == 0:
        return set()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=200,
    )
    try:
        y_arr = np.asarray(y)
        clf.fit(Xs, y_arr)
    except Exception:
        return set()
    probs = clf.predict_proba(Xs)
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(y_onehot.shape[0]), y_arr] = 1.0
    grad = probs - y_onehot
    weights = np.linalg.norm(grad, axis=1)
    WX = Xs * weights[:, None]
    G = Xs.T @ WX
    pair_scores: List[Tuple[float, Tuple[int, int]]] = []
    for i in range(Xs.shape[1]):
        for j in range(i + 1, Xs.shape[1]):
            score = abs(G[i, j])
            pair_scores.append((score, (i, j)))
    pair_scores.sort(reverse=True)
    combos: Set[Tuple[str, ...]] = set()
    for score, (i, j) in pair_scores[:top_pairs]:
        if score <= 0:
            continue
        combo = tuple(sorted((features[i], features[j])))
        combos.add(combo)
    for size in combo_sizes:
        if size <= 2 or len(features) < size:
            continue
        base_pairs = pair_scores[: min(top_pairs, len(pair_scores))]
        for _, (i, j) in base_pairs:
            selected = {i, j}
            order = np.argsort(-np.abs(G[i]))
            for idx in order:
                if idx in selected:
                    continue
                selected.add(int(idx))
                if len(selected) == size:
                    break
            if len(selected) == size:
                combos.add(tuple(sorted(features[k] for k in selected)))
    # random augmentations for diversity
    for _ in range(min(10, len(pair_scores))):
        if len(features) < max(combo_sizes):
            break
        chosen = set()
        while len(chosen) < max(combo_sizes):
            chosen.add(int(rng.choice(len(features))))
        for size in combo_sizes:
            if size <= 1:
                continue
            subset = tuple(sorted(features[idx] for idx in list(chosen)[:size]))
            combos.add(subset)
    return combos


def _minhash_lsh_candidates(
    table: _Table,
    y: np.ndarray,
    ranked_features: Sequence[str],
    combo_sizes: Sequence[int],
    rng: np.random.RandomState,
    *,
    num_perm: int = 20,
    rows_per_band: int = 4,
) -> Set[Tuple[str, ...]]:
    features = ranked_features[: min(len(ranked_features), 40)]
    if len(features) < 2:
        return set()
    bool_matrix: List[np.ndarray] = []
    for name in features:
        values = table.column_data(name)
        if _is_numeric(values) and not _is_bool(values):
            thresh = np.nanmedian(_fill_numeric(values.astype(float)))
            mask = _fill_numeric(values.astype(float)) >= thresh
        else:
            codes, _ = _factorize(values)
            if np.all(codes < 0):
                mask = np.zeros(len(values), bool)
            else:
                mode = np.bincount(codes[codes >= 0]).argmax()
                mask = codes == mode
        bool_matrix.append(mask.astype(bool))
    bool_arr = np.vstack(bool_matrix)
    if bool_arr.size == 0:
        return set()
    num_bands = max(1, num_perm // rows_per_band)
    hashes = rng.randint(1, 2_147_483_647, size=(num_perm, bool_arr.shape[1])).astype(np.int64)
    signatures = np.full((bool_arr.shape[0], num_perm), np.iinfo(np.int64).max, dtype=np.int64)
    for idx, mask in enumerate(bool_arr):
        if not np.any(mask):
            continue
        hashed = hashes[:, mask]
        signatures[idx] = hashed.min(axis=1)
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
    for feature_idx in range(signatures.shape[0]):
        for band in range(num_bands):
            start = band * rows_per_band
            end = min(start + rows_per_band, num_perm)
            key = tuple(signatures[feature_idx, start:end])
            if not key:
                continue
            buckets.setdefault((band, key), []).append(feature_idx)
    combos: Set[Tuple[str, ...]] = set()
    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    for candidates in buckets.values():
        if len(candidates) < 2:
            continue
        names = [features[idx] for idx in candidates]
        for size in combo_sizes:
            if size <= 1 or len(names) < size:
                continue
            for combo in combinations(names, size):
                idxs = [features.index(name) for name in combo]
                mask = np.all(bool_arr[idxs], axis=0)
                if not np.any(mask):
                    continue
                best = 0.0
                for cls in classes:
                    cls_mask = y_arr == cls
                    if not np.any(cls_mask):
                        continue
                    in_class = np.sum(mask & cls_mask)
                    if in_class == 0:
                        continue
                    ratio = in_class / max(1, np.sum(cls_mask))
                    contrast = in_class / max(1, np.sum(mask))
                    best = max(best, ratio * contrast)
                if best >= 0.05:
                    combos.add(tuple(sorted(combo)))
    return combos


def _encode_features(table: _Table, features: Sequence[str]) -> np.ndarray:
    cols = []
    for name in features:
        values = table.column_data(name)
        if _is_numeric(values) and not _is_bool(values):
            cols.append(_fill_numeric(values.astype(float)))
        else:
            codes, _ = _factorize(values)
            cols.append(codes.astype(float))
    return np.column_stack(cols) if cols else np.empty((len(table), 0), float)


def _evaluate_subspace(
    X: np.ndarray,
    y: np.ndarray,
    skf: StratifiedKFold,
    classes: np.ndarray,
) -> Dict[str, Any]:
    scaler = StandardScaler()
    macro_scores: List[float] = []
    coverage: List[float] = []
    per_class_accum = np.zeros(classes.size, float)
    support = 0

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=200,
            solver="lbfgs",
        )
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        macro_scores.append(float(f1_score(y_test, y_pred, average="macro")))
        per_class = f1_score(y_test, y_pred, labels=classes, average=None, zero_division=0)
        per_class_accum += per_class
        coverage.append(len(np.unique(y_pred)) / classes.size)
        support += len(test_idx)

    per_class_avg = {int(cls): float(score / len(macro_scores)) for cls, score in zip(classes, per_class_accum)}
    return dict(
        macro_f1_mean=float(np.mean(macro_scores)),
        macro_f1_std=float(np.std(macro_scores)),
        coverage_mean=float(np.mean(coverage)),
        per_class_f1=per_class_avg,
        support=int(support),
    )


def _variance_ratio(X: np.ndarray, y: np.ndarray) -> float:
    X = np.asarray(X, float)
    if X.size == 0:
        return 0.0
    y = np.asarray(y)
    mu = X.mean(axis=0)
    between = 0.0
    within = 0.0
    for cls in np.unique(y):
        mask = y == cls
        if not np.any(mask):
            continue
        Xc = X[mask]
        muc = Xc.mean(axis=0)
        between += mask.sum() * float(np.sum((muc - mu) ** 2))
        within += float(np.sum((Xc - muc) ** 2))
    return float(between / (within + 1e-9))


def _logistic_l1_importance(X: np.ndarray, y: np.ndarray, random_state: Optional[int]) -> float:
    if X.size == 0:
        return 0.0
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    rng = np.random.RandomState(random_state)
    if Xs.shape[0] > 2500:
        idx = rng.choice(Xs.shape[0], size=2500, replace=False)
        Xs = Xs[idx]
        y = np.asarray(y)[idx]
    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.8,
        max_iter=120,
        n_jobs=1,
        random_state=random_state,
    )
    try:
        clf.fit(Xs, y)
    except Exception:
        return 0.0
    coef = np.asarray(clf.coef_, float)
    return float(np.mean(np.abs(coef)))


def _planes_from_records(records: Sequence[DeltaRecord], indices: Sequence[int]) -> List[SubspacePlane]:
    idx = list(indices)
    points_by_pair: Dict[Tuple[int, int], List[np.ndarray]] = {}
    weights_by_pair: Dict[Tuple[int, int], List[float]] = {}

    for rec in records:
        if not getattr(rec, "success", True):
            continue
        a, b = int(rec.y0), int(rec.y1)
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        w = float(getattr(rec, "final_score", 1.0))
        x0 = np.asarray(rec.x0, float)[idx]
        x1 = np.asarray(rec.x1, float)[idx]
        pts = [x0, x1]
        if getattr(rec, "cp_count", 0) and np.asarray(getattr(rec, "cp_x", np.empty((0, 0)))).size:
            cp = np.asarray(rec.cp_x, float)
            if cp.ndim == 1:
                cp = cp.reshape(1, -1)
            pts.extend(cp[:, idx])
        else:
            pts.append(0.5 * (x0 + x1))
        arr = np.vstack(pts)
        points_by_pair.setdefault(key, []).append(arr)
        weights_by_pair.setdefault(key, []).append(np.full(arr.shape[0], w, float))

    planes: List[SubspacePlane] = []
    for key, chunks in points_by_pair.items():
        P = np.vstack(chunks)
        W = np.concatenate(weights_by_pair[key]) if key in weights_by_pair else np.ones(P.shape[0])
        if P.shape[0] <= len(indices):
            continue
        normal, bias, rmse = _fit_plane(P, W)
        planes.append(SubspacePlane(classes=key, normal=normal, bias=bias, rmse=rmse, support=int(P.shape[0]), weight=float(W.sum())))
    return planes


def _fit_plane(P: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, float, float]:
    P = np.asarray(P, float)
    weights = np.asarray(weights, float).reshape(-1)
    weights = np.clip(weights, 1e-6, None)
    sw = float(weights.sum())
    mu = (weights[:, None] * P).sum(axis=0) / sw
    Z = P - mu[None, :]
    Z_w = Z * np.sqrt(weights[:, None])
    _, _, Vt = np.linalg.svd(Z_w, full_matrices=False)
    normal = Vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    bias = -float(normal @ mu)
    residuals = P @ normal + bias
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return normal, bias, rmse


def _fill_numeric(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, float)
    if arr.size == 0:
        return arr
    if np.isnan(arr).all():
        return np.zeros_like(arr)
    return np.nan_to_num(arr, nan=float(np.nanmean(arr)))


def _factorize(values: np.ndarray) -> Tuple[np.ndarray, Dict[Any, int]]:
    vals = np.asarray(values)
    codes = np.full(vals.shape, -1, int)
    mapping: Dict[Any, int] = {}
    next_code = 0
    for idx, val in enumerate(vals):
        if _is_missing(val):
            continue
        key = val
        if key not in mapping:
            mapping[key] = next_code
            next_code += 1
        codes[idx] = mapping[key]
    return codes, mapping


def _is_numeric(arr: np.ndarray) -> bool:
    dtype = np.asarray(arr).dtype
    return dtype.kind in {"i", "u", "f", "c"}


def _is_bool(arr: np.ndarray) -> bool:
    dtype = np.asarray(arr).dtype
    return dtype == np.bool_ or dtype.kind == "b"


def _is_missing(value: Any) -> bool:
    try:
        return value is None or bool(value != value)
    except Exception:
        return False


__all__ = [
    "SubspacePlane",
    "SubspaceReport",
    "MultiClassSubspaceExplorer",
]
