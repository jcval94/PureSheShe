import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from deldel.engine import ScoreAdaptor
from deldel.frontier_planes_all_modes import _fit_plane_tls, _multi_ransac_lo_lts, _refine_plane_irls
from deldel.find_low_dim_spaces_fast import (
    _class_baseline,
    _metrics_region_multiclass_maskbits,
    _onehot_y,
    _pack_bits_cols,
)

OUTPUT_DIR = Path("experiments_outputs")
CSV_PATH = OUTPUT_DIR / "profiling_microbenchmarks.csv"
JSON_PATH = OUTPUT_DIR / "profiling_microbenchmarks.json"


class DummyModel:
    def __init__(self, n_features: int, n_classes: int, seed: int = 0) -> None:
        rng = np.random.RandomState(seed)
        self.W = rng.normal(size=(n_features, n_classes))
        self.b = rng.normal(size=(n_classes,))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W + self.b
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


def _bench(fn, *args, repeats: int = 5, warmup: int = 1) -> dict:
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn(*args)
        times.append(perf_counter() - t0)
    return {
        "repeats": repeats,
        "min_s": min(times),
        "mean_s": float(np.mean(times)),
        "max_s": max(times),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)

    results = []

    # _multi_ransac_lo_lts
    P = rng.normal(size=(120, 4))
    results.append(
        {"benchmark": "_multi_ransac_lo_lts", **_bench(_multi_ransac_lo_lts, P, repeats=3)}
    )

    # _refine_plane_irls
    n0, b0 = _fit_plane_tls(P)
    results.append(
        {"benchmark": "_refine_plane_irls", **_bench(_refine_plane_irls, P, n0, b0, repeats=5)}
    )

    # _metrics_region_multiclass_maskbits
    N = 256
    y = rng.randint(0, 3, size=N)
    labels = sorted(set(map(int, y)))
    baseline = _class_baseline(y)
    onehot = _onehot_y(y, labels)
    packed_Y_by_class = {
        int(c): _pack_bits_cols(onehot[:, [idx]])[:, 0] for idx, c in enumerate(labels)
    }
    mask = rng.rand(N) > 0.45
    mask_bits = _pack_bits_cols(mask.reshape(-1, 1))[:, 0]
    results.append(
        {
            "benchmark": "_metrics_region_multiclass_maskbits",
            **_bench(_metrics_region_multiclass_maskbits, mask_bits, packed_Y_by_class, labels, baseline, N),
        }
    )

    # ScoreAdaptor.scores
    X = rng.normal(size=(200, 6))
    model = DummyModel(n_features=6, n_classes=3)
    adaptor = ScoreAdaptor(model)
    results.append({"benchmark": "ScoreAdaptor.scores", **_bench(adaptor.scores, X, repeats=5)})

    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with JSON_PATH.open("w", encoding="utf-8") as handle:
        json.dump({"results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
