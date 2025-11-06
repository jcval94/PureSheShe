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

### Exploración de subespacios valiosos

Para ir más allá de la visualización y detectar reglas en subespacios de baja dimensión, puedes apoyarte en
`find_low_dim_spaces`. El siguiente fragmento imprime las reglas halladas por dimensión y recopila los
identificadores de planos involucrados:

```python
%%time
# 9s
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


planos_ = []
# Presentación: de 1 a d dimensiones
for dim_k in sorted(valuable.keys()):
    print('---------------------------------------------------------------------------------------------------------------')
    if not valuable[dim_k]:
        continue
    print(f"\n=== Espacios valiosos en {dim_k}D ===")
    for r in valuable[dim_k]:
        print('-------------------------------------------------')
        cls = r["target_class"]; dims = r["dims"]; pid = r["plane_ids"]
        m = r["metrics"]
        print(f"Clase {cls} | dims={dims} | rule: {r['rule_text']}")
        print(f"  F1={m['f1']:.3f}  Prec={m['precision']:.3f}  Rec={m['recall']:.3f}  "
              f"size={m['size']}  lift(P)={m['lift_precision']:.2f}  baseline={m['baseline']:.3f}")
        print(r['plane_ids'])
        planos_.append(r['plane_ids'])


    print(f"Total: {len(valuable[dim_k])} reglas válidas")
```

### Experimento de rejilla para `find_low_dim_spaces`

El directorio `experiments_outputs/` incluye un script que ejecuta la tubería basada en
`make_corner_class_dataset` y explora distintas configuraciones de `find_low_dim_spaces`. El
resultado queda registrado en `experiments_outputs/finder_runs_params.csv`, con métricas agregadas por
dimensión y una columna JSON para los parámetros efectivos usados en cada corrida.

```bash
python experiments_outputs/run_find_low_dim_param_sweep.py
```

Esto generará la tabla `finder_runs_params.csv`, útil para comparar la sensibilidad del buscador ante
variaciones en soporte mínimo, ganancias requeridas y tamaño máximo de las reglas.

## Recursos adicionales

- `tests/`: ejemplos automatizados que ejercitan diferentes configuraciones.
- `src/deldel/engine.py`: núcleo del algoritmo y estructuras de datos.
- `src/deldel/frontier_planes_all_modes.py`: utilidades para recolectar y visualizar planos.
- `src/deldel/datasets.py`: generadores de conjuntos sintéticos.
- `src/deldel/experiments.py`: pipelines de experimentos reproducibles.

¡Explora y adapta DelDel a tus necesidades!
