# Hiperparametrización find_comb_dim_spaces_full

Dataset: make_corner_class_dataset (4D, 3 clases).

## Conclusiones (tiempo vs número de reglas)

El análisis usa la métrica **rules_per_sec = total_rules / elapsed_s**. Se reportan los parámetros con mayor lift sobre la mediana global de rules_per_sec.

### Parámetros con mayor impacto positivo en rules_per_sec

| param | value | count | avg_rules_per_sec | lift_vs_median | avg_total_rules | avg_elapsed_s |
| --- | --- | --- | --- | --- | --- | --- |
| mode | base | 3456 | 7002.48 | 2.10 | 12.00 | 0.0017 |
| clause_overlap_max | 0.8 | 3456 | 4175.87 | 1.25 | 10.22 | 0.0028 |
| max_clauses | 2 | 3456 | 4054.30 | 1.22 | 9.56 | 0.0026 |
| max_planes | 10 | 2592 | 4030.66 | 1.21 | 9.63 | 0.0027 |
| beam_width | 48 | 3456 | 4012.40 | 1.20 | 9.63 | 0.0027 |
| max_planes | 16 | 2592 | 3995.53 | 1.20 | 9.63 | 0.0027 |
| max_rules_per_class | 480 | 5184 | 3988.69 | 1.20 | 9.63 | 0.0027 |
| clause_beam_width | 12 | 5184 | 3984.99 | 1.20 | 9.63 | 0.0027 |

### Top 10 configuraciones por rules_per_sec

| rank | mode | max_planes | beam_width | max_candidates_per_class | max_rules_per_class | max_clauses | clause_beam_width | clause_iterations | clause_diverse_topk | clause_overlap_max | elapsed_s | total_rules | rules_per_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | base | 10 | 36 | 600 | 720 | 3 | 16 | 120 | 40 | 0.80 | 0.0015 | 12 | 8000.00 |
| 2 | base | 10 | 36 | 600 | 720 | 3 | 16 | 180 | 40 | 0.60 | 0.0015 | 12 | 8000.00 |
| 3 | base | 10 | 48 | 600 | 720 | 3 | 16 | 120 | 30 | 0.70 | 0.0015 | 12 | 8000.00 |
| 4 | base | 10 | 48 | 600 | 720 | 3 | 16 | 120 | 40 | 0.60 | 0.0015 | 12 | 8000.00 |
| 5 | base | 10 | 48 | 600 | 720 | 3 | 16 | 120 | 40 | 0.80 | 0.0015 | 12 | 8000.00 |
| 6 | base | 10 | 48 | 600 | 720 | 4 | 12 | 120 | 30 | 0.70 | 0.0015 | 12 | 8000.00 |
| 7 | base | 10 | 48 | 600 | 720 | 4 | 12 | 120 | 30 | 0.80 | 0.0015 | 12 | 8000.00 |
| 8 | base | 10 | 48 | 600 | 720 | 2 | 16 | 120 | 40 | 0.70 | 0.0015 | 12 | 8000.00 |
| 9 | base | 16 | 48 | 600 | 720 | 2 | 16 | 120 | 40 | 0.80 | 0.0015 | 12 | 8000.00 |
| 10 | base | 16 | 48 | 600 | 720 | 3 | 12 | 120 | 30 | 0.60 | 0.0015 | 12 | 8000.00 |

### Top 10 configuraciones por total_rules (sin considerar tiempo)

| rank | mode | max_planes | beam_width | max_candidates_per_class | max_rules_per_class | max_clauses | clause_beam_width | clause_iterations | clause_diverse_topk | clause_overlap_max | elapsed_s | total_rules | rules_per_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | and_or_beam | 10 | 24 | 300 | 480 | 2 | 12 | 120 | 30 | 0.60 | 0.0037 | 13 | 3513.51 |
| 2 | and_or_beam | 10 | 24 | 300 | 480 | 2 | 12 | 120 | 30 | 0.70 | 0.0043 | 13 | 3023.26 |
| 3 | and_or_beam | 10 | 24 | 300 | 480 | 2 | 12 | 120 | 30 | 0.80 | 0.0042 | 13 | 3095.24 |
| 4 | and_or_beam | 10 | 24 | 300 | 480 | 2 | 12 | 120 | 40 | 0.60 | 0.0038 | 13 | 3421.05 |
| 5 | and_or_beam | 10 | 24 | 300 | 480 | 2 | 12 | 120 | 40 | 0.70 | 0.0037 | 13 | 3513.51 |
| 6 | and_or_beam | 10 | 24 | 300 | 480 | 2 | 12 | 120 | 40 | 0.80 | 0.0037 | 13 | 3513.51 |
| 7 | and_or_beam | 10 | 24 | 300 | 480 | 2 | 12 | 180 | 30 | 0.60 | 0.0037 | 13 | 3513.51 |
| 8 | and_or_beam | 10 | 24 | 300 | 480 | 2 | 12 | 180 | 30 | 0.70 | 0.0036 | 13 | 3611.11 |
| 9 | and_or_beam | 10 | 24 | 300 | 480 | 2 | 12 | 180 | 30 | 0.80 | 0.0038 | 13 | 3421.05 |
| 10 | and_or_beam | 10 | 24 | 300 | 480 | 2 | 12 | 180 | 40 | 0.60 | 0.0035 | 13 | 3714.29 |

## CSVs

- Tabla completa: find_comb_dim_spaces_full_hyperparam.csv
- Resumen por parámetro: find_comb_dim_spaces_full_hyperparam_summary.csv
