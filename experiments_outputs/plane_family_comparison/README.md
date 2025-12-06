# Comparación de planos por familia (versión actual vs. previa)

Este experimento ejecuta `prune_and_orient_planes_unified_globalmaj` sobre tres datasets multiclase (iris, wine y un
`make_classification` con 4 clases) usando fronteras eje-alineadas con `thresholds_per_dim=2`. Se compararon:

- **Versión actual**: HEAD `$(git rev-parse --short HEAD)`.
- **Versión previa**: worktree `../PureSheShe_prev` (`$(cd ../PureSheShe_prev && git rev-parse --short HEAD)`).

Los resultados completos están en `plane_family_comparison.csv`.

## Resumen rápido

- **Planos ganadores por familia**: la versión actual mantiene muchos más ganadores por par (p. ej., iris pasa de 18 → 162 y
  wine de 18 → 411), reflejando la nueva lógica de un representante por familia más la inclusión de regiones globales.
- **Tamaño de familias**: el promedio de planos por familia crece (iris: 2.88 → 6.59), pero el número de familias por par
  baja (wine: 176 → 137; synthetic_mix: 211 → 143), indicando una agrupación más agresiva con más variantes retenidas por
  familia.
- **Other vs. winning**: en wine y synthetic_mix los `other_planes` bajan (1323 → 939 y 3540 → 2742), mientras que en iris
  suben (448 → 906) debido a la separación de ganadores vs. planes de lift ≤ 1.
- **Métricas promedio**: la versión previa no guardaba métricas por par en los planos (`winners_*` y `others_*` aparecen
  vacías). En la versión actual los ganadores mantienen `f1` medio entre 0.66 y 0.79 según dataset, y los "others" quedan
  ligeramente por debajo.
- **Tiempo de ejecución**: la versión actual es igual o ligeramente más rápida (≈0.06 s menos en iris y ~2 s menos en
  synthetic_mix) pese a manejar más ganadores.

## Métricas agregadas por dataset

| Versión | Dataset       | Runtime s (prom) | Winning planos (suma) | Other planos (suma) | Familias prom. | Planos/familia prom. | Max planos/fam | F1 ganadores prom. | F1 others prom. |
|:-------:|:--------------|-----------------:|----------------------:|--------------------:|---------------:|---------------------:|---------------:|-------------------:|----------------:|
| actual  | iris          | 1.06             | 162                   | 906                 | 54.0           | 6.59                 | 16             | 0.79               | 0.73            |
| actual  | wine          | 9.32             | 411                   | 939                 | 137.0          | 3.28                 | 14             | 0.69               | 0.69            |
| actual  | synthetic_mix | 21.99            | 858                   | 2742                | 143.0          | 4.20                 | 12             | 0.66               | 0.64            |
| previa  | iris          | 1.12             | 18                    | 448                 | 54.0           | 2.88                 | 3              | —                  | —               |
| previa  | wine          | 9.76             | 18                    | 1323                | 176.0          | 2.54                 | 3              | —                  | —               |
| previa  | synthetic_mix | 23.99            | 36                    | 3540                | 211.0          | 2.82                 | 4              | —                  | —               |

Notas: tiempos y promedios se calcularon por par de clases; los valores de F1 faltan en la versión previa por ausencia de
`metrics_pair` en los planos.
