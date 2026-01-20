# A/B test: _multi_ransac_lo_lts optimizations

## Setup (initial quick run)
- Dataset: 3 synthetic planes (2,500 points per plane) + 20% outliers, 3D, gaussian noise (σ=0.02).
- Parameters: `max_models=4`, `max_iter=300`, `seed=0`, `dplus1=4`.
- New implementation used `use_float32=True` (float32 only for IRLS weights; residuals remain float64 for stable dot/out).
- Numba status: **not available** in this environment (`_NUMBA_AVAILABLE=False`), so the numba path was not exercised.

## Results (initial quick run)
| Metric | Baseline | New |
| --- | --- | --- |
| Mean runtime (s, n=5) | 0.2685 | 0.2693 |
| Speedup | — | 0.997× |

### Raw timings (seconds)
- Baseline: `[0.3025, 0.2639, 0.2538, 0.2638, 0.2583]`
- New: `[0.2539, 0.2587, 0.2586, 0.2563, 0.3188]`

### New profile (aggregate, seconds)
- SVD: `0.2334`
- Residuals: `0.1603`
- Weights: `0.0073`

## Extended A/B run (more robust)
- Baseline: `HEAD~1` (pre-change) `frontier_planes_all_modes.py`.
- Dataset: 4 synthetic planes (5,000 points per plane) + 25% outliers, 3D, gaussian noise (σ=0.03).
- Parameters: `max_models=4`, `max_iter=400`, `dplus1=4`.
- Seeds: 0–4 (five datasets), 6 runs per seed (total N=30 per variant).
- New implementation used `use_float32=True` for weights only.
- Numba status: **not available** in this environment, so numba path was not exercised.

### Summary statistics (seconds)
| Metric | Baseline | New | Speedup |
| --- | --- | --- | --- |
| Mean | 0.7346 | 0.7011 | 1.050× |
| Median | 0.7281 | 0.6992 | 1.047× |
| Std. dev. | 0.0458 | 0.0412 | 0.0655 |
| P90 | 0.7844 | 0.7517 | 1.109× |
| P95 | 0.8175 | 0.7693 | 1.168× |

## Notes
- The refactor reduces temporary allocations via reusable buffers (`r`, `w`, `Z`, `Zw`) and in-place ops.
- Float32 is limited to the weight vector to avoid changing the covariance/eigensolve step.
- In the robust run, the mean speedup is ~5% with variability across runs; gains are modest without numba.
