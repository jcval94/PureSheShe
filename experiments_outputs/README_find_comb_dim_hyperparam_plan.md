# Resultados del plan de hiperparametrización

## Top configs (ordenado por combos/seg)

| rank | mode | max_planes | beam_width | max_candidates_per_class | max_rules_per_class | min_size | lift_min | elapsed_s | total_rules | combos_per_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | base | 10 | 36 | 900 | 720 | 2 | 0.50 | 2.737 | 51 | 18.63 |
| 2 | base | 10 | 48 | 900 | 720 | 2 | 0.50 | 3.635 | 66 | 18.16 |
| 3 | base | 10 | 48 | 600 | 720 | 2 | 0.50 | 3.698 | 66 | 17.85 |
| 4 | base | 10 | 36 | 600 | 720 | 2 | 0.50 | 2.883 | 51 | 17.69 |
| 5 | hessian_rank | 10 | 48 | 600 | 720 | 2 | 0.50 | 3.803 | 65 | 17.09 |
| 6 | hessian_rank | 10 | 48 | 900 | 720 | 2 | 0.50 | 3.880 | 65 | 16.75 |
| 7 | hessian_rank | 10 | 36 | 900 | 720 | 2 | 0.50 | 3.156 | 47 | 14.89 |
| 8 | base | 12 | 36 | 600 | 720 | 2 | 0.50 | 3.436 | 51 | 14.84 |
| 9 | hessian_rank | 10 | 36 | 600 | 720 | 2 | 0.50 | 3.199 | 47 | 14.69 |
| 10 | hessian_rank | 12 | 48 | 900 | 720 | 2 | 0.50 | 4.493 | 66 | 14.69 |
| 11 | base | 12 | 48 | 600 | 720 | 2 | 0.50 | 4.519 | 66 | 14.60 |
| 12 | hessian_rank | 12 | 48 | 600 | 720 | 2 | 0.50 | 4.669 | 66 | 14.14 |
| 13 | hessian_rank | 12 | 36 | 900 | 720 | 2 | 0.50 | 3.406 | 48 | 14.09 |
| 14 | base | 12 | 48 | 900 | 720 | 2 | 0.50 | 4.722 | 66 | 13.98 |
| 15 | hessian_rank | 12 | 36 | 600 | 720 | 2 | 0.50 | 3.451 | 48 | 13.91 |
| 16 | base | 12 | 36 | 900 | 720 | 2 | 0.50 | 3.733 | 51 | 13.66 |

## Recomendaciones (Parte 2)

Mejor config: **mode=base**, max_planes=10, beam_width=36.

- Tiempo alto: baja max_planes o beam_width para reducir costo.
- Pocas reglas: baja lift_min y min_size, o sube beam_width.
