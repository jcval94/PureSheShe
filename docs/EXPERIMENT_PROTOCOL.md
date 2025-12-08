# Protocolo maestro de experimentación (lectura obligatoria)

Este documento debe leerse antes de cualquier cambio o ejecución sobre el repositorio. Resume el flujo que **toda persona o IA**
debe seguir para evaluar modificaciones usando las métricas prioritarias: **lift** y **F1** calculadas con
`describe_regions_report(valuable, top_per_class=3, dataset_size=X.shape[0], return_average_metrics=True)`. Con
`top_per_class=3`, promedia siempre las tres mejores reglas de **cada clase**; la optimización debe maximizar resultados de forma
separada por clase y luego consolidar los promedios globales.

## Objetivo

- Maximizar el **lift** y el **F1** obtenidos mediante `describe_regions_report`, maximizando por clase y analizando los
  promedios por clase antes del promedio global.
- Medir el impacto de cada cambio repitiendo exactamente el mismo flujo de datos, modelo y selección de planos.
- Registrar los resultados (F1, lift y tiempo de ejecución promedio) en `docs/experiments_log.csv` junto con un resumen del
  cambio realizado.

## Flujo de referencia (reproducible y obligatorio)

Usa siempre el siguiente pipeline para generar los artefactos de evaluación. No alteres semillas, tamaños ni hiperparámetros a
menos que el experimento lo requiera explícitamente y documenta cualquier variación.

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from deldel import describe_regions_report
from deldel.deldel import DelDel, DelDelConfig
from deldel.find_low_dim_spaces_fast import find_low_dim_spaces
from deldel.frontier_planes import compute_frontier_planes_all_modes
from deldel.frontier_planes.unified_family_selector import prune_and_orient_planes_unified_globalmaj
from deldel.change_point import ChangePointConfig

# Dataset y modelo de referencia
X, y = make_classification(
    n_samples=1200,
    n_features=20,
    n_informative=8,
    class_sep=1.2,
    random_state=0,
)
model = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)

# Registros generados por DelDel
cfg = DelDelConfig(segments_target=180, random_state=0)
records = DelDel(cfg, ChangePointConfig(enabled=False)).fit(X, model).records_

# Planos frontera (modo C mejorado)
res_c = compute_frontier_planes_all_modes(
    records,
    mode="C",
    min_cluster_size=12,
    max_models_per_round=6,
    seed=0,
)

# Notas rápidas sobre `records_` y diversidad de clases
res_c = compute_frontier_planes_all_modes(
    records,
    mode="C",
    min_cluster_size=12,
    max_models_per_round=6,
    seed=0,
    explorer_reports=explorer_fast.get_report(),
    explorer_feature_names=[f"f{i}" for i in range(X.shape[1])],
    explorer_top_k=5,
)

# Selección global orientada
sel = prune_and_orient_planes_unified_globalmaj(
    res_c,
    X,
    y,
    feature_names=[f"f{i}" for i in range(X.shape[1])],
    max_k=8,
    min_improve=1e-3,
    min_region_size=25,
    min_abs_diff=0.02,
    min_rel_lift=0.05,
    # family_clustering_mode="connected",  # opciones: "dbscan" (por defecto), "greedy" o "connected"
)

valuable = find_low_dim_spaces(
    X, y, sel,
    feature_names=[f"x{i}" for i in range(X.shape[1])],
    max_planes_in_rule=3,       # conjunciones hasta 3 planos
    max_planes_per_pair=4,      # como tope por par
    min_support=40,             # evita reglas con muy pocos puntos
    min_rel_gain_f1=0.05,       # pide F1 al menos 30% sobre baseline
    min_lift_prec=1.40,         # o bien precision con lift >= 1.4
    consider_dims_up_to=X.shape[1],  # busca 1..d
    rng_seed=0,
)

describe_regions_report(valuable, top_per_class=5, dataset_size=X.shape[0])
```

> **Recordatorio:** la evaluación final debe usar `describe_regions_report(valuable, top_per_class=3, dataset_size=X.shape[0],
> return_average_metrics=True)` para capturar los promedios por clase y el promedio global.
> Promedia las métricas F1 y lift de las tres primeras reglas por clase para comparar versiones.

## Pasos para evaluar cambios

1. **Preparar el entorno** con las dependencias del proyecto (`pip install -e .[dev]` o `pip install -r requirements-dev.txt`).
2. **Ejecutar el flujo de referencia** exactamente como se muestra arriba.
3. **Medir F1 y lift** usando `top_per_class=3`. Calcula los promedios sobre las tres reglas priorizadas por clase.
4. **Medir el tiempo de ejecución promedio** del pipeline completo (excluye instalaciones). Usa, por ejemplo, `time.perf_counter` y
   repite al menos 3 veces si el tiempo lo permite; reporta la media.
5. **Determinar la mejora** comparando con la última entrada registrada en `docs/experiments_log.csv`.
6. **Registrar los resultados** añadiendo una fila en `docs/experiments_log.csv` con F1, lift, tiempo promedio, resumen de
   cambios y hallazgos.
7. **Documentar variaciones**: si alteras hiperparámetros, añade una nota en la columna de resultados para asegurar la
   reproducibilidad.

## Plantilla de reporte rápido

- **Objetivo del experimento:** breve descripción (ej. "ajuste de min_rel_lift").
- **Cambio aplicado:** ramas de código afectadas, parámetros modificados.
- **Resultados:** F1 y lift medios (Top-3 por clase), tiempo promedio, observaciones sobre estabilidad.
- **Conclusión:** ¿mejora, regresión o efecto neutro? Indica evidencia cuantitativa.

Sigue este protocolo para cualquier experimento futuro. Si algún paso no puede ejecutarse (limitaciones de hardware o tiempo),
explica la razón en el CSV y documenta el impacto esperado.
