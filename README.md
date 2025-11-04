# DelDel

DelDel es una biblioteca ligera para instrumentar modelos de clasificación y analizar sus fronteras de decisión. Incluye utilidades para registrar llamadas al modelo, explorar pares de puntos, detectar cambios de etiqueta y visualizar fronteras con Plotly.

## Instalación

```bash
pip install .
```

## Importaciones principales

El paquete expone en su nivel superior las clases y funciones más utilizadas:

```python
from deldel import (
    DelDel,
    DelDelConfig,
    ChangePointConfig,
    compute_frontier_planes_all_modes,
    compute_frontier_planes_weighted,
    plot_frontiers_implicit_interactive_v2,
    plot_planes_with_point_lines,
)
```

## Acceso a utilidades internas del repositorio

Si necesitas piezas más específicas, puedes importarlas directamente desde los módulos que viven en `src/deldel/`:

```python
from deldel.engine import build_weighted_frontier, fit_tls_plane_weighted
from deldel.frontier_planes_all_modes import (
    compute_frontier_planes_all_modes,
    plot_planes_with_point_lines,
)
from deldel.datasets import make_corner_class_dataset
from deldel.experiments import run_corner_pipeline_experiments
```

Revisa estos archivos para extender el comportamiento o comprender los detalles de implementación.

## Ejemplo rápido

Entrena un modelo, calcula planos de frontera y genera una visualización interactiva:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from deldel import (
    DelDel,
    DelDelConfig,
    ChangePointConfig,
    compute_frontier_planes_all_modes,
    compute_frontier_planes_weighted,
    plot_frontiers_implicit_interactive_v2,
    plot_planes_with_point_lines,
)

# ====== Datos de ejemplo ======
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)

# ====== Configuración ======
cfg = DelDelConfig(
    segments_target=120,
    random_state=0,
)

cp_cfg = ChangePointConfig(
    enabled=False,
    mode="treefast",
    per_record_max_points=8,
    max_candidates=128,
    max_bisect_iters=8,
)

# ====== Ejecución ======
d = DelDel(cfg, cp_cfg).fit(X, model)
print("Top-5 records con cambios de etiqueta:")
for r in d.topk(5):
    print(f"y0={r.y0} → y1={r.y1} | flips={r.cp_count} | ΔL2={r.delta_norm_l2:.3f}")

records = d.records_
planes = compute_frontier_planes_weighted(records, prefer_cp=True, weight_map="softmax")

fig = plot_frontiers_implicit_interactive_v2(
    records,
    X,
    y,
    planes=planes,
    show_planes=True,
    dims=(0, 1, 3),
    detail="high",
    grid_res_3d=72,
    extend=1.3,
    clamp_extend_to_X=True,
)

pairs = sorted({tuple(sorted((r.y0, r.y1))) for r in records if r.y0 != r.y1})
resC = compute_frontier_planes_all_modes(
    records,
    pairs=pairs,
    mode="C",
    min_cluster_size=10,
    max_models_per_round=6,
    seed=0,
)

plot_planes_with_point_lines(
    resC,
    records=records,
    line_kind="segment",
    show_planes=False,
    plane_opacity=0.25,
    line_opacity=0.75,
    dims=(0, 1, 2),
    title="Segmentos por punto",
)
```

## Recursos adicionales

- `tests/`: ejemplos automatizados que ejercitan diferentes configuraciones.
- `src/deldel/engine.py`: núcleo del algoritmo y estructuras de datos.
- `src/deldel/frontier_planes_all_modes.py`: utilidades para recolectar y visualizar planos.
- `src/deldel/datasets.py`: generadores de conjuntos sintéticos.
- `src/deldel/experiments.py`: pipelines de experimentos reproducibles.

¡Explora y adapta DelDel a tus necesidades!
