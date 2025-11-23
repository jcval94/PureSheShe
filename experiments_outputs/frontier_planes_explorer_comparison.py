"""Comparación entre la versión base y la versión con explorer_fast.

Genera un CSV con tiempos y conteo de planos producidos al ejecutar
``compute_frontier_planes_all_modes`` con y sin subespacios provenientes de
``MultiClassSubspaceExplorer``.
"""

import csv
from time import perf_counter
from typing import Dict, Iterable, Tuple

import numpy as np

from deldel import (
    DeltaRecord,
    MultiClassSubspaceExplorer,
    compute_frontier_planes_all_modes,
    make_corner_class_dataset,
)


def _make_records(seed: int = 0) -> Iterable[DeltaRecord]:
    rng = np.random.default_rng(seed)
    dim = 4
    records = []
    for pair in ((0, 1), (1, 2)):
        a, b = pair
        for _ in range(6):
            x0 = rng.normal(size=dim)
            direction = rng.normal(size=dim) * 0.2
            x1 = x0 + direction + (b - a) * 0.1
            delta = x1 - x0
            S0 = np.full(dim, 0.05)
            S1 = np.full(dim, 0.05)
            S0[a] = 0.9
            S1[b] = 0.9
            records.append(
                DeltaRecord(
                    index_a=0,
                    index_b=1,
                    method="benchmark",
                    success=True,
                    y0=a,
                    y1=b,
                    delta_norm_l2=float(np.linalg.norm(delta, ord=2)),
                    delta_norm_linf=float(np.linalg.norm(delta, ord=np.inf)),
                    score_change=0.75,
                    distance_term=0.1,
                    change_term=0.2,
                    final_score=0.8,
                    time_ms=0.5,
                    x0=x0,
                    x1=x1,
                    delta=delta,
                    S0=S0,
                    S1=S1,
                )
            )
    return records


def _count_planes(res: Dict[Tuple[int, int], Dict[str, object]]) -> int:
    return int(
        sum(
            len(plist)
            for block in res.values()
            for plist in (block.get("planes_by_label", {}) or {}).values()
        )
    )


def _count_subspace_planes(res: Dict[Tuple[int, int], Dict[str, object]]) -> int:
    total = 0
    for block in res.values():
        sub_blocks = block.get("subspace_variants") or {}
        for sub in sub_blocks.values():
            total += _count_planes({(0, 1): sub})
    return int(total)


def main() -> None:
    records = _make_records()
    X, y, feature_names = make_corner_class_dataset(
        n_per_cluster=300,
        std_class1=0.35,
        std_other=0.7,
        a=2.6,
        random_state=0,
    )

    explorer_fast = MultiClassSubspaceExplorer(random_state=0)
    t0 = perf_counter()
    explorer_fast.fit(X, y, preset="fast")
    explorer_time = perf_counter() - t0
    reports = explorer_fast.get_report()

    start = perf_counter()
    base = compute_frontier_planes_all_modes(
        records,
        mode="C",
        max_models_per_round=6,
        seed=0,
        min_cluster_size=2,
    )
    base_time = perf_counter() - start

    start = perf_counter()
    with_explorer = compute_frontier_planes_all_modes(
        records,
        mode="C",
        max_models_per_round=6,
        seed=0,
        min_cluster_size=2,
        explorer_reports=reports,
        explorer_feature_names=feature_names,
        explorer_top_k=6,
    )
    explorer_run_time = perf_counter() - start

    rows = [
        {
            "variant": "baseline",
            "frontier_time_s": f"{base_time:.6f}",
            "planes_full": _count_planes(base),
            "planes_subspaces": 0,
            "explorer_fit_s": f"{explorer_time:.6f}",
        },
        {
            "variant": "explorer_fast",
            "frontier_time_s": f"{explorer_run_time:.6f}",
            "planes_full": _count_planes(with_explorer),
            "planes_subspaces": _count_subspace_planes(with_explorer),
            "explorer_fit_s": f"{explorer_time:.6f}",
        },
    ]

    path = "experiments_outputs/frontier_planes_explorer_comparison.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "frontier_time_s",
                "planes_full",
                "planes_subspaces",
                "explorer_fit_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV generado en {path}")


if __name__ == "__main__":
    main()
