# Pruebas de `prune_and_orient_planes_unified_globalmaj`

Este directorio contiene un barrido de hiperparámetros sobre cinco datasets sintéticos de tamaños y dimensionalidades variadas. El objetivo fue validar la estabilidad de `prune_and_orient_planes_unified_globalmaj` y encontrar configuraciones que maximicen la `balanced accuracy` promedio a nivel de pares de clases.

## Cómo reproducir

Ejecuta el script dedicado con la ruta del repositorio como *cwd*:

```bash
PYTHONPATH=src python experiments_outputs/prune_orientation_sweep/run_prune_orientation_sweep.py
```

Genera `sweep_results.csv` con 30 ejecuciones (5 datasets × 6 combinaciones de parámetros). El script construye un conjunto ligero de planos frontera a partir de cortes por dimensión, invoca la función principal y vuelca las métricas por par en el CSV resultante.

## Datasets explorados

| Dataset        | Muestras | Dimensiones |
|----------------|----------|-------------|
| corner_small   | 1,120    | 4           |
| corner_large   | 2,080    | 4           |
| highdim_mid    | 6,000    | 18          |
| highdim_large  | 12,000   | 28          |
| mixed_easy     | 4,500    | 12          |

Todos los runs se completaron sin errores y con regiones globales emitidas para cada combinación de parámetros.

## Hiperparámetros evaluados

Se variaron tres parámetros clave:

- `family_clustering_mode ∈ {greedy, connected, dbscan}`
- `tau_mult ∈ {0.50, 0.75}`
- `min_region_size ∈ {20, 35}`
- Para los modos DBSCAN se fijó `family_dbscan_eps = 0.30`.

El resto de parámetros de selección se mantuvieron constantes (`min_abs_diff=0.01`, `min_rel_lift=0.05`, `max_k=6`, `min_recall=0.75`, `min_region_frac=0.05`).

## Resultados

El archivo [`sweep_results.csv`](./sweep_results.csv) resume las métricas por par. Al promediar la `balanced accuracy` sobre todos los datasets y pares, el mejor desempeño se obtuvo con:

- `family_clustering_mode = dbscan`
- `tau_mult = 0.50`
- `min_region_size = 20`
- `family_dbscan_eps = 0.30`

Esta configuración alcanzó una `balanced accuracy` media de **0.852** (mediana 0.847) y un F1 medio de **0.830** sobre 15 pares evaluados. Los promedios por dataset con este set fueron:

- corner_small: media 0.898, mediana 0.944
- corner_large: media 0.893, mediana 0.942
- highdim_mid: media 0.839, mediana 0.847
- highdim_large: media 0.770, mediana 0.747
- mixed_easy: media 0.861, mediana 0.863

El modo `dbscan` con `tau_mult=0.75` y `min_region_size=35` empató en media, pero la opción con umbral de región menor retuvo más candidatos globales y se prefiere por simplicidad.
