# Barrido agresivo de `find_low_dim_spaces`

Este experimento ejecuta combinaciones de hiperparámetros en cada uno de cinco datasets ligeros (incluido el corner 4D). El pipeline sigue `EXPERIMENT_PROTOCOL`: se generan los datos, se ajusta un bosque aleatorio, se obtienen planos frontera (con *fallback* de selección demo cuando no hay planos) y se exploran reglas de baja dimensión con muestreo reducido para acelerar la búsqueda masiva.

## Datasets y configuraciones clave
- `corner`: 200 filas, 4 dimensiones, clases balanceadas por `make_corner_class_dataset`.
- `medium_classification`: 200 filas, 6 dimensiones, `make_classification` balanceado.
- `wide_classification`: 240 filas, 7 dimensiones.
- `imbalanced`: 200 filas, 6 dimensiones con pesos `[0.55, 0.3, 0.15]`.
- `high_dim`: 260 filas, 8 dimensiones.

### Grillas ejecutadas
- **Sweep aleatorio original (300 corridas/dataset):** `max_planes_in_rule`∈[1,2], `max_planes_per_pair`∈[rule,rule+1], `min_support`∈[6,15], `max_rules_per_dim`∈[6,14], `consider_dims_up_to`≤3, `sample_limit_per_r`∈{500…2000}, `enable_unions=False`, semillas aleatorias variadas.
- **Grid B solicitado (324 corridas/dataset, añadidas sobre los resultados existentes):**
  - `consider_dims_up_to`: {2, 5} (capado a la dimensión del dataset).
  - `max_planes_in_rule`: {1, 4}.
  - `max_planes_per_pair`: {2, 6, 8} (forzado a ser ≥ `max_planes_in_rule`).
  - `min_lift_prec`: {1.0, 1.2, 1.6}.
  - `min_rel_gain_f1`: {0.02, 0.07, 0.12}.
  - `min_support`: {8, 16, 32}.
  - `sample_limit_per_r`: {75}.

## Resultados agregados
- Total de corridas: 3,120 (624 por dataset), almacenadas en `aggressive_find_low_dim_sweep.csv`.
- Grid B (corridas 300–623 de cada dataset): promedio de top-3 `mean_f1` por dataset=0.656, `mean_lift_precision`=2.006, tiempo medio por corrida≈0.238 s. 【d51eec†L1-L7】
- Mejores corridas por dataset (índice de corrida y parámetros en JSON) se documentan en el CSV; destacan `mean_f1` máximos entre 0.55 (corner) y 0.75 (high_dim) dentro del Grid B. 【F:experiments_outputs/aggressive_find_low_dim_sweep.csv†L1502-L3121】

## Notas operativas
- Se silenciaron las trazas internas de `find_low_dim_spaces` con `redirect_stdout/redirect_stderr` para evitar saturar la consola durante las 3,120 ejecuciones.
- Cuando `prune_and_orient_planes_unified_globalmaj` no retornó planos, se activó `_build_demo_selection` para mantener el flujo completo.
