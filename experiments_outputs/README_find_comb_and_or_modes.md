# Comparativa AND/OR/DNF en find_comb_dim_spaces

Resultados generados con 50 variantes de `make_corner_class_dataset` (350 ejecuciones
totales al evaluar 7 modos) y 3 combinaciones de hiperparámetros.

## Métricas promedio (promedio sobre reglas)

| modo | tiempo (s) | f1 | precision | recall | lift_precision | reglas |
|---|---:|---:|---:|---:|---:|---:|
| and | 0.268 | 0.518 | 0.659 | 0.625 | 3.952 | 12.5 |
| or | 0.309 | 0.489 | 0.509 | 0.809 | 1.605 | 7.9 |
| dnf | 0.277 | 0.542 | 0.572 | 0.741 | 3.661 | 45.1 |
| and_or_greedy | 0.271 | 0.601 | 0.883 | 0.509 | 4.842 | 3.0 |
| and_or_beam | 0.282 | 0.542 | 0.572 | 0.741 | 3.661 | 45.1 |
| and_or_random | 0.299 | 0.601 | 0.883 | 0.509 | 4.842 | 3.0 |
| and_or_diverse | 0.270 | 0.615 | 0.638 | 0.766 | 2.885 | 3.0 |

## Configuración

Se usaron 50 datasets con distintos `n_per_cluster`, dispersiones y seeds. Se probaron 3 hiperparámetros con variaciones en `max_planes`, `beam_width`, `min_size`, `max_candidates_per_class` y `max_rules_per_class`, además de `max_clauses=3` para modos DNF/AND-OR.

## Conclusiones generales

- **and** mantiene lift alto y recall consistente.
- **or** reduce reglas y puede mejorar precisión, pero con lift menor.
- **dnf** y **and_or_beam** exploran uniones de cláusulas AND para mejorar cobertura.
- **and_or_greedy** ofrece un balance rápido con pocas cláusulas.
- **and_or_random** aporta exploración estocástica adicional.
- **and_or_diverse** prioriza diversidad entre cláusulas para reducir solapamiento.
