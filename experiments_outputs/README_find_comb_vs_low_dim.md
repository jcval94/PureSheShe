# Comparativa `find_comb_dim_spaces` vs `find_low_dim_spaces`

Este experimento sigue el script `run_comb_vs_low_dim_competition.py` para generar un dataset sintético (600 muestras, 12 features, 2 clases) y comparar ambos buscadores usando una selección de planos sintética cuando el pipeline de `prune_and_orient_planes_unified_globalmaj` no devuelve candidatos.

## Métricas agregadas

| Variante | Runtime (s) | F1 global | Lift precisión global | Regiones |
| --- | --- | --- | --- | --- |
| find_low_dim_spaces | 8.936 | 0.639 | 1.635 | 248 |
| find_comb_dim_spaces | 48.750 | 0.738 | 1.329 | 1847 |

## Métricas por clase (Top-1)

- `find_low_dim_spaces`
  - Clase 0: F1=0.638, Lift=1.553, Prec=0.771, Rec=0.544.
  - Clase 1: F1=0.644, Lift=1.645, Prec=0.828, Rec=0.526.
- `find_comb_dim_spaces`
  - Clase 0: F1=0.753, Lift=1.357, Prec=0.674, Rec=0.852.
  - Clase 1: F1=0.729, Lift=1.263, Prec=0.635, Rec=0.854.

## Artefactos

- CSV resumido: `experiments_outputs/find_comb_vs_low_dim_competition.csv`.
- Reportes Top-3 por clase en `experiments_outputs/ab_test_reports/`.
- Script reproducible: `experiments_outputs/run_comb_vs_low_dim_competition.py`.
