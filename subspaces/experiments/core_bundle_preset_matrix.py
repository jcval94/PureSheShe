"""Ejecuta el bundle núcleo en varios presets/flags y guarda métricas en CSV."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from deldel.subspace_change_detector import MultiClassSubspaceExplorer
from subspaces.experiments.core_method_bundle import CORE_METHOD_KEYS


DatasetDef = tuple[str, pd.DataFrame, np.ndarray]


def _make_datasets() -> List[DatasetDef]:
    """Genera dataframes sintéticos con >=10 variables para las pruebas."""

    configs = [
        {"n_samples": 180, "n_features": 12, "n_informative": 6, "random_state": 0},
        {"n_samples": 220, "n_features": 14, "n_informative": 7, "random_state": 1},
        {"n_samples": 260, "n_features": 16, "n_informative": 8, "random_state": 2},
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


def _preset_cases() -> Sequence[tuple[str, bool, bool]]:
    """Genera combinaciones de preset + flags a evaluar."""

    return [
        ("high_quality", False, False),
        ("high_quality", True, True),
        ("fast", False, False),
        ("fast", True, True),
        ("ultra_fast", True, False),
        ("ultra_fast", True, True),
    ]


def _build_explorer(random_state: int) -> MultiClassSubspaceExplorer:
    return MultiClassSubspaceExplorer(
        max_sets=None,
        combo_sizes=(2, 3, 4),
        filter_top_k=8,
        chi2_pool=8,
        random_samples=16,
        corr_threshold=0.4,
        corr_max_features=48,
        mi_max_features=48,
        rf_estimators=12,
        rf_max_depth=3,
        cv_splits=2,
        random_state=random_state,
        enabled_methods=CORE_METHOD_KEYS,
        max_candidates_per_method=20,
        max_total_candidates=60,
        fast_eval_budget=10,
        fast_eval_top_frac=0.8,
        fast_compute_secondary_metrics=False,
    )


def _summarize_run(
    *,
    explorer: MultiClassSubspaceExplorer,
    reports,
    dataset_id: str,
    preset: str,
    skip_feature_stats: bool,
    skip_attach_planes: bool,
    elapsed: float,
    n_samples: int,
    n_features: int,
) -> dict[str, object]:
    best_f1 = max((r.mean_macro_f1 for r in reports), default=float("nan"))
    mean_f1 = float(np.mean([r.mean_macro_f1 for r in reports])) if reports else float("nan")
    return {
        "dataset": dataset_id,
        "n_samples": n_samples,
        "n_features": n_features,
        "preset": preset,
        "skip_feature_stats": skip_feature_stats,
        "skip_attach_planes": skip_attach_planes,
        "candidate_sets": len(explorer.all_candidate_sets_),
        "selected_reports": len(reports),
        "best_mean_macro_f1": best_f1,
        "avg_mean_macro_f1": mean_f1,
        "baseline_f1": explorer.baseline_f1_,
        "elapsed_seconds": elapsed,
    }


def run_matrix() -> List[dict[str, object]]:
    rows: List[dict[str, object]] = []
    for dataset_id, df, y in _make_datasets():
        for preset, skip_stats, skip_attach in _preset_cases():
            explorer = _build_explorer(random_state=42)
            start = time.perf_counter()
            explorer.fit(
                df,
                y,
                records=[],
                preset=preset,
                skip_feature_stats=skip_stats,
                skip_attach_planes=skip_attach,
                method_key=None,
            )
            elapsed = time.perf_counter() - start
            reports = explorer.get_report()
            rows.append(
                _summarize_run(
                    explorer=explorer,
                    reports=reports,
                    dataset_id=dataset_id,
                    preset=preset,
                    skip_feature_stats=skip_stats,
                    skip_attach_planes=skip_attach,
                    elapsed=elapsed,
                    n_samples=df.shape[0],
                    n_features=df.shape[1],
                )
            )
    return rows


def write_csv(output_path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "experiments_outputs" / "core_bundle_preset_matrix.csv"
    write_csv(output_path, run_matrix())


if __name__ == "__main__":
    main()
