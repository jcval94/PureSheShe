"""Experimentos reproducibles para evaluar MultiClassSubspaceExplorer."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
import sys
import warnings
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

warnings.filterwarnings(
    "ignore",
    message=".*'multi_class'.*deprecated.*",
    category=FutureWarning,
)

from deldel.engine import ChangePointConfig, DelDel, DelDelConfig
from deldel.subspace_change_detector import MultiClassSubspaceExplorer, SubspaceReport


class ColumnarView:
    """Vista mínima tipo DataFrame para mezclar columnas numéricas y categóricas."""

    def __init__(self, columns: Sequence[str], arrays: Sequence[np.ndarray]):
        self.columns = list(columns)
        self._mapping = {col: np.asarray(arr) for col, arr in zip(self.columns, arrays)}

    def __getitem__(self, key: str) -> np.ndarray:
        return self._mapping[key]


def _make_mixed_dataset(
    *,
    n_samples: int = 5000,
    n_features: int = 20,
    n_informative: int = 16,
    n_classes: int = 6,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, ColumnarView]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=min(25, n_features // 10),
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.8,
        random_state=random_state,
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rng = np.random.RandomState(random_state)
    bool_count = max(5, n_features // 5)
    cat_count = max(4, n_features // 6)
    bool_idx = rng.choice(n_features, size=bool_count, replace=False)
    remaining = np.setdiff1d(np.arange(n_features), bool_idx, assume_unique=True)
    cat_idx = rng.choice(remaining, size=cat_count, replace=False)

    columns: List[str] = []
    arrays: List[np.ndarray] = []
    for j in range(n_features):
        name = f"x{j}"
        col = X_scaled[:, j]
        if j in bool_idx:
            arrays.append(col > np.median(col))
        elif j in cat_idx:
            quantiles = np.quantile(col, [0.2, 0.4, 0.6, 0.8])
            categories = np.empty(col.shape[0], dtype=object)
            categories[:] = "q4"
            categories[col <= quantiles[0]] = "q0"
            categories[(col > quantiles[0]) & (col <= quantiles[1])] = "q1"
            categories[(col > quantiles[1]) & (col <= quantiles[2])] = "q2"
            categories[(col > quantiles[2]) & (col <= quantiles[3])] = "q3"
            arrays.append(categories)
        else:
            arrays.append(col)
        columns.append(name)

    table = ColumnarView(columns, arrays)
    return X_scaled, y, table


def _fit_deldel(X: np.ndarray, model: RandomForestClassifier) -> DelDel:
    cfg = DelDelConfig(log_level=logging.ERROR, segments_target=32)
    cp_cfg = ChangePointConfig(enabled=False)
    d = DelDel(cfg, cp_cfg)
    d.fit(X, model)
    return d


def _method_sort_key(name: str) -> Tuple[int, str]:
    parts = name.split("_")
    token = parts[1] if len(parts) > 1 else "999"
    numeric = "".join(ch for ch in token if ch.isdigit())
    suffix = "".join(ch for ch in token if not ch.isdigit())
    return (int(numeric) if numeric else 999, suffix or "")


def _runtime_rows(
    explorer: MultiClassSubspaceExplorer,
    reports: Sequence[SubspaceReport],
) -> List[List[object]]:
    method_keys = sorted(explorer.method_name_map_.keys(), key=_method_sort_key)
    candidate_sources = explorer.candidate_sources_
    selected_counts: Dict[str, int] = {key: 0 for key in method_keys}
    for report in reports:
        methods = candidate_sources.get(tuple(sorted(report.features)), set())
        for method in methods:
            if method in selected_counts:
                selected_counts[method] += 1

    timings_friendly = explorer.method_timings_
    rows: List[List[object]] = []
    total_elapsed = 0.0
    total_candidates = len(explorer.all_candidate_sets_)
    total_scored = len(explorer.evaluated_reports_)
    unique_selected = len({tuple(sorted(r.features)) for r in reports})
    for idx, internal in enumerate(method_keys, start=1):
        friendly = explorer.method_name_map_.get(internal, internal)
        elapsed = float(timings_friendly.get(friendly, 0.0))
        total_elapsed += elapsed
        rows.append(
            [
                f"Metodo {idx}: {friendly}",
                round(elapsed, 6),
                len(explorer.method_candidate_sets_.get(internal, set())),
                selected_counts.get(internal, 0),
                total_candidates,
                total_scored,
                len(reports),
            ]
        )

    rows.append([
        "Total",
        round(total_elapsed, 6),
        total_candidates,
        unique_selected,
        total_candidates,
        total_scored,
        len(reports),
    ])
    return rows


def _method_sets_rows(
    explorer: MultiClassSubspaceExplorer,
    _reports: Sequence[SubspaceReport],
) -> List[List[object]]:
    method_keys = sorted(explorer.method_name_map_.keys(), key=_method_sort_key)
    headers = ["Conjunto"] + [f"Metodo {idx}" for idx in range(1, len(method_keys) + 1)] + ["Score Global"]
    rows: List[List[object]] = [headers]
    sources = explorer.candidate_sources_
    scored_map = explorer.candidate_reports_
    for combo in explorer.all_candidate_sets_:
        methods = sources.get(tuple(combo), set())
        row = [f"({', '.join(combo)})"]
        for idx, method in enumerate(method_keys, start=1):
            value = "Si" if method in methods else "No"
            row.append(value)
        report = scored_map.get(tuple(combo))
        row.append(round(report.mean_macro_f1, 6) if report else "")
        rows.append(row)
    return rows


def _ranking_rows(explorer: MultiClassSubspaceExplorer, top_k: int = 200) -> List[List[object]]:
    headers = ["Feature", "Rank", "MutualInfo", "Chi2", "StumpGain"]
    rows: List[List[object]] = [headers]
    ranking = explorer.feature_ranking_[:top_k]
    for idx, feature in enumerate(ranking, start=1):
        rows.append(
            [
                feature,
                idx,
                round(explorer.mi_scores_.get(feature, 0.0), 6),
                round(explorer.chi2_scores_.get(feature, 0.0), 6),
                round(explorer.stump_scores_.get(feature, 0.0), 6),
            ]
        )
    return rows


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    outputs_dir = base_dir / "outputs"
    runtimes_path = outputs_dir / "runtimes" / "runtime_summary.csv"
    method_sets_path = outputs_dir / "method_sets" / "method_sets.csv"
    rankings_path = outputs_dir / "rankings" / "feature_ranking.csv"

    print("Generando dataset artificial...")
    X, y, table = _make_mixed_dataset()

    print("Entrenando modelo RandomForest...")
    model = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    model.fit(X, y)

    print("Obteniendo records con DelDel...")
    deldel = _fit_deldel(X, model)

    print("Explorando subespacios discriminativos...")
    explorer = MultiClassSubspaceExplorer(
        random_state=0,
        mi_max_features=400,
        corr_max_features=400,
        max_sets=15,
        combo_sizes=(2, 3, 4),
        chi2_pool=18,
        random_samples=36,
        cv_splits=2,
        max_total_candidates=620,
    )
    t0 = perf_counter()
    explorer.fit(table, y, deldel.records_)
    fit_time = perf_counter() - t0
    reports = explorer.get_report()
    print(f"Reporte generado en {fit_time:.2f} s con {len(reports)} conjuntos seleccionados")

    runtime_rows = [[
        "Metodo",
        "Tiempo (s)",
        "Candidatos metodo",
        "Seleccionados metodo",
        "Total candidatos unicos",
        "Conjuntos evaluados",
        "Reportes finales",
    ]]
    runtime_rows.extend(_runtime_rows(explorer, reports))

    method_rows = _method_sets_rows(explorer, reports)
    ranking_rows = _ranking_rows(explorer)

    for path, rows in [
        (runtimes_path, runtime_rows),
        (method_sets_path, method_rows),
        (rankings_path, ranking_rows),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerows(rows)
        print(f"Archivo guardado en {path}")


if __name__ == "__main__":
    main()
