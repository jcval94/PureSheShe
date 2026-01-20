# Comparativa Hessiana para find_comb_dim_spaces

Resumen de tiempos y métricas (media de top-5 por clase).

| Variante | Runtime (s) | F1 mean | Prec mean | Recall mean |
| --- | ---: | ---: | ---: | ---: |
| find_comb_dim_spaces | 10.662 | 0.728 | 0.667 | 0.805 |
| find_comb_dim_spaces_hessian_seed | 11.742 | 0.728 | 0.667 | 0.805 |
| find_comb_dim_spaces_hessian_rank | 6.575 | 0.728 | 0.667 | 0.805 |
| find_comb_dim_spaces_hessian_filter | 4.955 | 0.634 | 0.539 | 0.795 |

CSV generado: `experiments_outputs/find_comb_hessian_variants.csv`