# Comparativa AND/OR/DNF en find_comb_dim_spaces

Resultados generados con 50 variantes de `make_corner_class_dataset` (400 ejecuciones
totales al evaluar 8 modos) y combinaciones de hiperparámetros por modo.

## Métricas promedio (promedio sobre reglas)

| modo | tiempo (s) | f1 | precision | recall | lift_precision | reglas |
|---|---:|---:|---:|---:|---:|---:|
| find_comb_dim_spaces | 1.764 | 0.396 | 0.449 | 0.670 | 2.101 | 93.5 |
| and | 1.782 | 0.396 | 0.449 | 0.670 | 2.101 | 93.5 |
| or | 5.054 | 0.420 | 0.462 | 0.673 | 1.232 | 99.2 |
| dnf | 2.449 | 0.594 | 0.884 | 0.515 | 4.851 | 35.9 |
| and_or_greedy | 1.756 | 0.634 | 0.714 | 0.726 | 3.517 | 35.9 |
| and_or_beam | 4.125 | 0.590 | 0.931 | 0.478 | 5.226 | 51.0 |
| and_or_random | 1.917 | 0.542 | 0.754 | 0.561 | 3.812 | 35.9 |
| and_or_diverse | 1.761 | 0.571 | 0.595 | 0.800 | 2.698 | 26.8 |

## Métricas promedio por clase 0

| modo | f1 | precision | recall | lift_precision | reglas |
|---|---:|---:|---:|---:|---:|
| find_comb_dim_spaces | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| and | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| or | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| dnf | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| and_or_greedy | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| and_or_beam | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| and_or_random | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |
| and_or_diverse | 0.000 | 0.000 | 0.000 | 0.000 | 0.0 |

## Métricas promedio por clase 1

| modo | f1 | precision | recall | lift_precision | reglas |
|---|---:|---:|---:|---:|---:|
| find_comb_dim_spaces | 0.371 | 0.410 | 0.691 | 1.282 | 67.9 |
| and | 0.371 | 0.410 | 0.691 | 1.282 | 67.9 |
| or | 0.376 | 0.408 | 0.656 | 1.201 | 82.9 |
| dnf | 0.088 | 0.160 | 0.061 | 0.213 | 0.7 |
| and_or_greedy | 0.022 | 0.040 | 0.015 | 0.053 | 0.1 |
| and_or_beam | 0.091 | 0.160 | 0.064 | 0.213 | 0.7 |
| and_or_random | 0.510 | 0.940 | 0.353 | 1.253 | 6.3 |
| and_or_diverse | 0.573 | 0.940 | 0.413 | 1.253 | 1.0 |

## Métricas promedio por clase 2

| modo | f1 | precision | recall | lift_precision | reglas |
|---|---:|---:|---:|---:|---:|
| find_comb_dim_spaces | 0.503 | 0.394 | 0.789 | 2.943 | 16.0 |
| and | 0.503 | 0.394 | 0.789 | 2.943 | 16.0 |
| or | 0.628 | 0.749 | 0.734 | 1.400 | 12.5 |
| dnf | 0.752 | 0.917 | 0.669 | 1.471 | 10.1 |
| and_or_greedy | 0.613 | 0.755 | 0.525 | 1.007 | 5.2 |
| and_or_beam | 0.725 | 0.911 | 0.617 | 1.266 | 12.2 |
| and_or_random | 0.637 | 0.696 | 0.728 | 2.358 | 11.2 |
| and_or_diverse | 0.835 | 0.997 | 0.723 | 1.330 | 1.7 |

## Ajustes para incrementar reglas

Se incrementaron los presupuestos combinatorios para producir más reglas: mayores `max_planes`, `beam_width`, `max_candidates_per_class`, `max_rules_per_class` y `top_k_floor_per_dim`, además de relajar la poda con `min_size` y `lift_min` más bajos. Para los modos con cláusulas se elevaron `max_clause_candidates`, `clause_beam_width`, `clause_iterations`, `clause_diverse_topk`, `max_clauses` y `max_dnf_rules_per_class`.

## Configuración

Se usaron 50 datasets con distintos `n_per_cluster`, dispersiones y seeds. Se probaron configuraciones de hiperparámetros con variaciones en `max_planes`, `beam_width`, `min_size`, `max_candidates_per_class`, `max_rules_per_class`, `lift_min` y parámetros de cláusulas (`max_clause_candidates`, `clause_beam_width`, `clause_iterations`, `clause_diverse_topk`, `clause_overlap_max`, `max_clauses`). Los modos **or** y **and_or_beam** usan un bloque agresivo con valores más altos.

## Conclusiones generales

- **and** mantiene lift alto y recall consistente.
- **or** reduce reglas y puede mejorar precisión, pero con lift menor.
- **dnf** y **and_or_beam** exploran uniones de cláusulas AND para mejorar cobertura.
- **and_or_greedy** ofrece un balance rápido con pocas cláusulas.
- **and_or_random** aporta exploración estocástica adicional.
- **and_or_diverse** prioriza diversidad entre cláusulas para reducir solapamiento.
