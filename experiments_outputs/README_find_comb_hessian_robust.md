# Robust comparison for Hessian variants

Summary statistics across dataset configurations and seeds (mean of per-run top-5 metrics).

| Variant | Runtime mean (s) | Runtime std | F1 mean | Prec mean | Recall mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| find_comb_dim_spaces | 7.860 | 5.673 | 0.650 | 0.589 | 0.766 |
| find_comb_dim_spaces_hessian_seed | 8.547 | 5.839 | 0.652 | 0.584 | 0.777 |
| find_comb_dim_spaces_hessian_rank | 7.251 | 5.591 | 0.649 | 0.586 | 0.770 |
| find_comb_dim_spaces_hessian_filter | 4.847 | 3.840 | 0.577 | 0.508 | 0.720 |

CSV generated: `experiments_outputs/find_comb_hessian_robust.csv`