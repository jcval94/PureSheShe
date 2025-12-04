"""Benchmark de métodos de agrupamiento de familias de planos.

Genera pools sintéticos de planos orientados con familias conocidas y compara
cuatro estrategias de colapso:
- greedy (original)
- connected (componentes conectados sobre la matriz de similitud binaria)
- kmeans (k-medias sobre la matriz de distancia precomputada)
- dbscan (densidad sobre matriz precomputada)

Resultados:
- Guarda resultados por corrida en `experiments_outputs/family_clustering_results.csv`.
- Guarda resumen agregado en `experiments_outputs/family_clustering_results_summary.csv`.
"""

import os
import sys
from time import perf_counter
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deldel.globalmaj import _cluster_families, _family_similarity_matrix


def _make_metrics(best_class: int, rng: np.random.Generator) -> Dict[int, Dict[str, float]]:
    base = rng.uniform(0.55, 0.9)
    lift = rng.uniform(1.1, 2.0)
    precision = min(0.99, base + rng.uniform(0.02, 0.08))
    recall = min(0.99, base + rng.uniform(0.01, 0.06))
    f1 = (2 * precision * recall) / max(1e-12, precision + recall)
    return {
        int(best_class): {
            "lift": float(lift),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "purity": float(precision),
            "region_size": 0,
        }
    }


def _sample_plane(d: int, base: np.ndarray, base_b: float, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    noise = rng.normal(scale=0.01, size=d)
    n = base + noise
    b = float(base_b + rng.normal(scale=0.02))
    return n, b


def build_synthetic_pool(
    *,
    n_planes: int,
    n_features: int,
    fam_size_range: Tuple[int, int],
    seed: int = 0,
) -> Tuple[List[Dict[str, float]], List[int]]:
    rng = np.random.default_rng(seed)
    oriented_pool: List[Dict[str, float]] = []
    true_labels: List[int] = []

    fam_id = 0
    while len(oriented_pool) < n_planes:
        fam_size = rng.integers(fam_size_range[0], fam_size_range[1] + 1)
        fam_size = int(min(fam_size, n_planes - len(oriented_pool)))
        base_n = rng.normal(size=n_features)
        base_b = rng.normal(scale=0.1)
        side = int(rng.choice([-1, 1]))
        best_class = int(rng.integers(0, 5))
        label_weight = float(rng.uniform(0.5, 2.0))
        for _ in range(fam_size):
            n, b = _sample_plane(n_features, base_n, base_b, rng)
            metrics = _make_metrics(best_class, rng)
            score_global = float(rng.uniform(0.5, 1.0))
            oriented_pool.append(
                dict(
                    n=n,
                    b=b,
                    side=side,
                    best_class=best_class,
                    metrics_by_class=metrics,
                    score_global=score_global,
                    label=label_weight,
                )
            )
            true_labels.append(fam_id)
        fam_id += 1

    return oriented_pool, true_labels


def run_one(
    *,
    oriented_pool: List[Dict[str, float]],
    true_labels: List[int],
    mode: str,
    seed: int,
    sim: np.ndarray | None,
) -> Dict[str, float]:
    pool_copy = [dict(p) for p in oriented_pool]
    t0 = perf_counter()
    _ = _cluster_families(
        pool_copy,
        cos_parallel=0.995,
        tau_mult=0.75,
        coef_eps=1e-8,
        jaccard_support=0.75,
        coef_cos_min=0.99,
        X_sample=None,
        jaccard_region=0.75,
        metric_dist_threshold=0.5,
        metric_weights={"lift": 1.2, "recall": 1.0, "precision": 1.0, "f1": 1.1},
        metric_norm="l2",
        mode=mode,
        dbscan_eps=0.51,
        dbscan_min_samples=1,
        precomputed_sim=sim,
    )
    elapsed = perf_counter() - t0
    pred_labels = [p.get("family_id", -1) for p in pool_copy]
    ari = float(adjusted_rand_score(true_labels, pred_labels))
    return dict(
        mode=mode,
        seed=seed,
        n_planes=len(pool_copy),
        ari=ari,
        time_sec=float(elapsed),
        n_true_families=len(set(true_labels)),
        n_pred_families=len(set(pred_labels)),
    )


if __name__ == "__main__":
    rows: List[Dict[str, float]] = []
    seeds = [11, 27, 42]
    sizes = [700]
    for seed in seeds:
        for n_planes in sizes:
            pool, truth = build_synthetic_pool(
                n_planes=n_planes,
                n_features=12,
                fam_size_range=(3, 8),
                seed=seed,
            )
            sim = _family_similarity_matrix(
                pool,
                cos_parallel=0.995,
                tau_mult=0.75,
                coef_eps=1e-8,
                jaccard_support=0.75,
                coef_cos_min=0.99,
                X_sample=None,
                jaccard_region=0.75,
                metric_dist_threshold=0.5,
                metric_weights={"lift": 1.2, "recall": 1.0, "precision": 1.0, "f1": 1.1},
                metric_norm="l2",
            )
            for mode in ["greedy", "connected", "kmeans", "dbscan"]:
                pre = None if mode == "greedy" else sim
                res = run_one(oriented_pool=pool, true_labels=truth, mode=mode, seed=seed, sim=pre)
                res["n_planes"] = n_planes
                rows.append(res)

    df = pd.DataFrame(rows)
    df.to_csv("experiments_outputs/family_clustering_results.csv", index=False)

    summary = (
        df.groupby("mode")
        .agg(
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            ari_min=("ari", "min"),
            ari_max=("ari", "max"),
            time_mean_sec=("time_sec", "mean"),
            time_std_sec=("time_sec", "std"),
            time_min_sec=("time_sec", "min"),
            time_max_sec=("time_sec", "max"),
            pred_families_mean=("n_pred_families", "mean"),
            pred_families_std=("n_pred_families", "std"),
            pred_families_min=("n_pred_families", "min"),
            pred_families_max=("n_pred_families", "max"),
            n_runs=("ari", "count"),
        )
        .reset_index()
    )
    summary.to_csv("experiments_outputs/family_clustering_results_summary.csv", index=False)
