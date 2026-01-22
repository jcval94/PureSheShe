# A/B testing: `_midpoint_flip_gate_simple` batching

## Environment
- Host: container runtime
- Python: 3.12
- Libraries: scikit-learn 1.3+, pandas, numpy

## Methodology
- Trained a `LogisticRegression` on a synthetic dataset (`make_classification`).
- Compared **baseline (old)** vs **new (patched)** for `_midpoint_flip_gate_simple` only.
- 50 experiments (10 seeds × 5 configurations). Each configuration varies:
  - Number of features `{10, 20, 40, 60, 80}`
  - Batch size `{4, 8, 16, 32, 64}`
  - Duplicate fraction `{0.0, 0.25, 0.5, 0.75, 0.9}`
- Each experiment runs `_midpoint_flip_gate_simple` 400 times in both variants.

## Aggregate results (50 experiments)

### `_midpoint_flip_gate_simple` speedup (old/new)
- Mean: **1.012×**
- Median: **0.990×**
- P10/P90: **0.908× / 1.152×**
- Min/Max: **0.716× / 1.328×**
- Wins (new faster): **22/50**

## Per-experiment detail

| ID | Seed | Features | Batch | Dup frac | Midpoint old (s) | Midpoint new (s) | Midpoint speedup |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 01 | 0 | 10 | 4 | 0.00 | 0.00044610 | 0.00040329 | 1.106 |
| 02 | 0 | 20 | 8 | 0.25 | 0.00042736 | 0.00045369 | 0.942 |
| 03 | 0 | 40 | 16 | 0.50 | 0.00042389 | 0.00045151 | 0.939 |
| 04 | 0 | 60 | 32 | 0.75 | 0.00047410 | 0.00038999 | 1.216 |
| 05 | 0 | 80 | 64 | 0.90 | 0.00043693 | 0.00040288 | 1.085 |
| 06 | 1 | 10 | 4 | 0.00 | 0.00047127 | 0.00047814 | 0.986 |
| 07 | 1 | 20 | 8 | 0.25 | 0.00048270 | 0.00048539 | 0.994 |
| 08 | 1 | 40 | 16 | 0.50 | 0.00045779 | 0.00050469 | 0.907 |
| 09 | 1 | 60 | 32 | 0.75 | 0.00053401 | 0.00052457 | 1.018 |
| 10 | 1 | 80 | 64 | 0.90 | 0.00049475 | 0.00049064 | 1.008 |
| 11 | 2 | 10 | 4 | 0.00 | 0.00049545 | 0.00043408 | 1.141 |
| 12 | 2 | 20 | 8 | 0.25 | 0.00055161 | 0.00044520 | 1.239 |
| 13 | 2 | 40 | 16 | 0.50 | 0.00044273 | 0.00049279 | 0.898 |
| 14 | 2 | 60 | 32 | 0.75 | 0.00045428 | 0.00039439 | 1.152 |
| 15 | 2 | 80 | 64 | 0.90 | 0.00044238 | 0.00047113 | 0.939 |
| 16 | 3 | 10 | 4 | 0.00 | 0.00044457 | 0.00049416 | 0.900 |
| 17 | 3 | 20 | 8 | 0.25 | 0.00044762 | 0.00045951 | 0.974 |
| 18 | 3 | 40 | 16 | 0.50 | 0.00043041 | 0.00044035 | 0.977 |
| 19 | 3 | 60 | 32 | 0.75 | 0.00043960 | 0.00042922 | 1.024 |
| 20 | 3 | 80 | 64 | 0.90 | 0.00053193 | 0.00045940 | 1.158 |
| 21 | 4 | 10 | 4 | 0.00 | 0.00045544 | 0.00044672 | 1.020 |
| 22 | 4 | 20 | 8 | 0.25 | 0.00045909 | 0.00047289 | 0.971 |
| 23 | 4 | 40 | 16 | 0.50 | 0.00043232 | 0.00047311 | 0.914 |
| 24 | 4 | 60 | 32 | 0.75 | 0.00045341 | 0.00041471 | 1.093 |
| 25 | 4 | 80 | 64 | 0.90 | 0.00044740 | 0.00044804 | 0.999 |
| 26 | 5 | 10 | 4 | 0.00 | 0.00045111 | 0.00045311 | 0.996 |
| 27 | 5 | 20 | 8 | 0.25 | 0.00044180 | 0.00045988 | 0.961 |
| 28 | 5 | 40 | 16 | 0.50 | 0.00044039 | 0.00041024 | 1.073 |
| 29 | 5 | 60 | 32 | 0.75 | 0.00045217 | 0.00048344 | 0.935 |
| 30 | 5 | 80 | 64 | 0.90 | 0.00047501 | 0.00044197 | 1.075 |
| 31 | 6 | 10 | 4 | 0.00 | 0.00047454 | 0.00048958 | 0.969 |
| 32 | 6 | 20 | 8 | 0.25 | 0.00042613 | 0.00044770 | 0.952 |
| 33 | 6 | 40 | 16 | 0.50 | 0.00044989 | 0.00047255 | 0.952 |
| 34 | 6 | 60 | 32 | 0.75 | 0.00045576 | 0.00040481 | 1.126 |
| 35 | 6 | 80 | 64 | 0.90 | 0.00045828 | 0.00043452 | 1.055 |
| 36 | 7 | 10 | 4 | 0.00 | 0.00043662 | 0.00047757 | 0.914 |
| 37 | 7 | 20 | 8 | 0.25 | 0.00044457 | 0.00046687 | 0.952 |
| 38 | 7 | 40 | 16 | 0.50 | 0.00043593 | 0.00042823 | 1.018 |
| 39 | 7 | 60 | 32 | 0.75 | 0.00046641 | 0.00042612 | 1.095 |
| 40 | 7 | 80 | 64 | 0.90 | 0.00045832 | 0.00045202 | 1.014 |
| 41 | 8 | 10 | 4 | 0.00 | 0.00044579 | 0.00050053 | 0.891 |
| 42 | 8 | 20 | 8 | 0.25 | 0.00043388 | 0.00045443 | 0.955 |
| 43 | 8 | 40 | 16 | 0.50 | 0.00050102 | 0.00037740 | 1.328 |
| 44 | 8 | 60 | 32 | 0.75 | 0.00045078 | 0.00043743 | 1.031 |
| 45 | 8 | 80 | 64 | 0.90 | 0.00046389 | 0.00048645 | 0.954 |
| 46 | 9 | 10 | 4 | 0.00 | 0.00042627 | 0.00059562 | 0.716 |
| 47 | 9 | 20 | 8 | 0.25 | 0.00049846 | 0.00039238 | 1.270 |
| 48 | 9 | 40 | 16 | 0.50 | 0.00044980 | 0.00047822 | 0.941 |
| 49 | 9 | 60 | 32 | 0.75 | 0.00043809 | 0.00047418 | 0.924 |
| 50 | 9 | 80 | 64 | 0.90 | 0.00043477 | 0.00047859 | 0.908 |

## Conclusion
- The batching change in `_midpoint_flip_gate_simple` **does not deliver consistent speedups** across 50 experiments (median < 1.0 and only 22/50 wins).
- **Recommendation:** Do **not** roll out the change broadly as a performance optimization, since the gains are inconsistent and often negative.

## Repro command (from repo root)
```bash
python - <<'PY'
import sys
from pathlib import Path
sys.path.append(str(Path('.').resolve() / 'src'))

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from deldel.engine import _midpoint_flip_gate_simple

warnings.filterwarnings("ignore", category=UserWarning)

feature_counts = [10, 20, 40, 60, 80]
batch_sizes = [4, 8, 16, 32, 64]
duplicate_fracs = [0.0, 0.25, 0.5, 0.75, 0.9]

experiments = []
exp_id = 0

for seed in range(10):
    for idx in range(5):
        exp_id += 1
        n_features = feature_counts[idx]
        batch = batch_sizes[idx]
        dup_frac = duplicate_fracs[idx]
        X, y = make_classification(
            n_samples=2000,
            n_features=n_features,
            n_informative=max(5, n_features // 2),
            random_state=seed,
        )
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        model = LogisticRegression(max_iter=300)
        model.fit(X, y)

        X_query = X.sample(n=800, random_state=seed).to_numpy(copy=True)
        if dup_frac > 0:
            dup_count = int(len(X_query) * dup_frac)
            X_query[:dup_count] = X_query[-dup_count:]

        def old_midpoint_flip_gate_simple(model, x0, x1, y0, y1, iters=3):
            tL, tR = 0.0, 1.0
            yL = model.predict(x0[None, :])[0]
            yR = model.predict(x1[None, :])[0]
            for _ in range(max(1, int(iters))):
                tm = 0.5 * (tL + tR)
                xm = (1.0 - tm) * x0 + tm * x1
                ym = model.predict(xm[None, :])[0]
                if ym != yL:
                    tR = tm; yR = ym
                else:
                    tL = tm; yL = ym
            if yL != yR:
                return True, 0.5 * (tL + tR)
            tm = 0.5 * (tL + tR)
            ym = model.predict(((1.0 - tm) * x0 + tm * x1)[None, :])[0]
            return (ym != y0), tm if (ym != y0) else (False, None)

        n_runs = 400
        start = time.perf_counter()
        for i in range(n_runs):
            _midpoint_flip_gate_simple(
                model,
                X_query[i % len(X_query)],
                X_query[(i + 1) % len(X_query)],
                y[0],
                y[1],
            )
        new_mid = (time.perf_counter() - start) / n_runs

        start = time.perf_counter()
        for i in range(n_runs):
            old_midpoint_flip_gate_simple(
                model,
                X_query[i % len(X_query)],
                X_query[(i + 1) % len(X_query)],
                y[0],
                y[1],
            )
        old_mid = (time.perf_counter() - start) / n_runs

        experiments.append({
            "id": exp_id,
            "seed": seed,
            "features": n_features,
            "batch": batch,
            "dup_frac": dup_frac,
            "mid_old": old_mid,
            "mid_new": new_mid,
            "mid_speedup": old_mid / new_mid,
        })

print(len(experiments))
mid_speedups = np.array([e["mid_speedup"] for e in experiments])
summary = {
    "mean": float(mid_speedups.mean()),
    "median": float(np.median(mid_speedups)),
    "min": float(mid_speedups.min()),
    "max": float(mid_speedups.max()),
    "p10": float(np.percentile(mid_speedups, 10)),
    "p90": float(np.percentile(mid_speedups, 90)),
    "wins": int((mid_speedups > 1.0).sum()),
    "total": len(mid_speedups),
}
print(summary)

for e in experiments:
    print(
        f"{e['id']:02d} seed={e['seed']} features={e['features']} batch={e['batch']} dup={e['dup_frac']:.2f} "
        f"mid_old={e['mid_old']:.8f} mid_new={e['mid_new']:.8f} mid_speedup={e['mid_speedup']:.3f}"
    )
PY
```
