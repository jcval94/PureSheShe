# Profiling (CPU/memoria) + A/B testing

Este README resume:

- Cómo activar el profiling opcional en los puntos de entrada principales.
- Los microbenchmarks mínimos solicitados.
- El A/B testing (baseline vs profiling) ejecutado en este entorno.

## Puntos de entrada con profiling opcional

Se agregó un wrapper opcional con `ProfilingConfig` para activar:

- CPU: `cProfile` + `pstats` (CSV/JSON).
- Memoria: `tracemalloc` (CSV/JSON).

Los puntos de entrada instrumentados son:

- `DelDel.fit` (en `src/deldel/engine.py`).
- `compute_frontier_planes_all_modes` (en `src/deldel/frontier_planes_all_modes.py`).
- `find_low_dim_spaces` (en `src/deldel/find_low_dim_spaces_fast.py`).

> El profiling **solo** se activa con `ProfilingConfig(enabled=True, ...)`.
> Por defecto, **no** se perfila nada en producción.

## Microbenchmarks mínimos

Script:

```bash
PYTHONPATH=src python experiments_outputs/run_microbenchmarks_profile.py
```

Salida (generada localmente):

- `experiments_outputs/profiling_microbenchmarks.csv`
- `experiments_outputs/profiling_microbenchmarks.json`

Resumen (mean_s) de la última ejecución en este entorno:

| benchmark | mean_s |
| --- | --- |
| `_multi_ransac_lo_lts` | 0.189923 |
| `_refine_plane_irls` | 0.001307 |
| `_metrics_region_multiclass_maskbits` | 0.0000468 |
| `ScoreAdaptor.scores` | 0.000433 |

## A/B testing (baseline vs profiling)

Script:

```bash
PYTHONPATH=src python experiments_outputs/run_profiling_ab_test.py
```

Salida (generada localmente):

- `experiments_outputs/profiling_ab_test.csv`
- `experiments_outputs/profiling_ab_test.json`
- Archivos de profiling (`*.cpu.csv/json`, `*.memory.csv/json`) en `experiments_outputs/`.

Resultados (s) de la última ejecución en este entorno:

| Stage | Baseline | Profiled | Overhead |
| --- | --- | --- | --- |
| `DelDel.fit` | 0.013055 | 0.150499 | 11.53x |
| `compute_frontier_planes_all_modes` | 0.053196 | 0.144542 | 2.72x |
| `find_low_dim_spaces` | 0.050791 | 0.226881 | 4.47x |

> El profiling añade overhead significativo; por eso está **desactivado por defecto** y requiere un flag explícito.
