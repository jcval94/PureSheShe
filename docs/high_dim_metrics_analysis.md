# High-dimensional metrics by region: summary

This document summarizes the contents of `experiments_outputs/high_dim_metrics_by_region.csv`.

## Dataset overview

- **Rows:** 75
- **Regions:** 7 unique `region_id` values covering three classes (0–2) with repeated runs per region.
- **Metrics tracked:** `f1`, `precision`, `recall`, `lift_precision`, `support`, `support_frac`, and a boolean `pareto` flag (all entries are marked Pareto-optimal).
- **Costs and dimensions:** Two cost settings (1.5 and 3.0) and five distinct dimensional configurations: `(4,)`, `(19,)`, `(4, 5)`, `(4, 19)`, and `(5, 19)`.

## Regional performance

Across all rows, per-region metrics are constant—each region’s runs share identical scores. The best-performing region by mean F1 is **rg_1d_c0_ce88474a36** (F1 0.745, precision 0.676, recall 0.829, support fraction 0.408). The lowest F1 is **rg_2d_c1_8880740fc8** (F1 0.554, precision 0.549, recall 0.559, support fraction 0.339). Other regions cluster between F1 0.58 and 0.74 with varying precision/recall balances.

## Dimensional patterns

Averaging by `dims` reveals two distinct regimes:

- **High-precision, lower-recall:** `(4, 19)` and `(5, 19)` emphasize precision (0.841 and 0.739) with smaller support fractions (0.260 and 0.218) and reduced recall (0.658 and 0.483).
- **Higher-recall, moderate-precision:** `(4,)` and `(19,)` deliver the strongest recall (0.824 and 0.790) and larger support (≈0.50), with balanced F1 (0.668 and 0.635).
- **Mixed profile:** `(4, 5)` sits in the middle (precision 0.549, recall 0.559, support fraction 0.339, F1 0.554).

## Cost effects

- **Cost 1.5 (35 rows):** Higher recall (0.810) and support fraction (0.499), with moderate precision (0.552) and F1 0.654.
- **Cost 3.0 (40 rows):** Higher precision (0.693) but lower recall (0.555) and support (0.274), resulting in a slightly lower mean F1 of 0.611.

Overall, lower-cost configurations favor coverage/recall, while higher-cost setups trade coverage for precision.
