# DelDel

DelDel es una biblioteca ligera para instrumentar clasificadores y descubrir reglas en subespacios de baja dimensión. La
configuración por defecto se centra en el `MultiClassSubspaceExplorer`, que ahora puede ejecutarse con un bundle curado de los
cinco mejores métodos derivados del análisis multi-dataset.

## Instalación

El proyecto utiliza la convención de *src layout* y, gracias a la configuración en `pyproject.toml`, tanto `deldel` como
`subspaces` quedan disponibles tras la instalación. Todo el proceso se resume en un único bloque de comandos:

```bash
git clone https://github.com/<usuario>/PureSheShe.git
cd PureSheShe
pip install .        # instalación estándar (incluye subspaces.*)
# o, para desarrollo editable con extras:
pip install -e .[dev]
```

El comando `pip install .` funciona en cualquier entorno compatible (incluido Google Colab) porque el paquete expone
explícitamente ambos módulos. Para flujos de desarrollo también se puede instalar en modo editable con extras (`-e .[dev]`).

> **¿Qué le faltaba al paquete?**<br>
> Solo era necesario decirle a `setuptools` dónde vivían `deldel` (en `src/`) y `subspaces` (en la raíz). Esto se resolvió
> declarando explícitamente ambos paquetes y sus rutas dentro de `[tool.setuptools]` en `pyproject.toml`, habilitando
> instalaciones directas tras un `git clone` sin pasos adicionales.

### Requirements y dependencias

Los requerimientos de ejecución se declaran en dos sitios sincronizados:

- `[project.dependencies]` dentro de `pyproject.toml`.
- Los archivos planos `requirements.txt` (ejecución) y `requirements-dev.txt` (desarrollo/tests).

Si solo se necesita preparar un entorno para correr scripts directamente desde `src/` (sin instalar el paquete), basta con:

```bash
pip install -r requirements.txt
```

Para ejecutar la batería de pruebas o trabajar en modo editable es conveniente instalar también las dependencias de
desarrollo:

```bash
pip install -r requirements-dev.txt
# o, si ya instalaste el paquete:
pip install -e .[dev]
```

Ambos archivos están pensados para usarse en Colab o entornos aislados donde resulta útil listar explícitamente las
dependencias antes de hacer `pip install .`. La guía `docs/COLAB.md` incluye ejemplos con cada flujo.

## Dataset sintético en esquinas 4D

`deldel.datasets.make_corner_class_dataset` replica el dataset 4D con clases en las esquinas de un hipercubo que se usa en
las pruebas y ejemplos de la librería. Devuelve una tupla `(X, y, feature_names)` lista para alimentar cualquier
clasificador de scikit-learn. La función `plot_corner_class_dataset` (opcional) genera un PCA 3D y una matriz de dispersión
para explorar la estructura del dataset; solo requiere tener instalados `matplotlib` y `pandas`.

```python
from sklearn.ensemble import RandomForestClassifier

from deldel.datasets import make_corner_class_dataset, plot_corner_class_dataset

# Dataset 4D con 3 clases (clase 1 en esquinas)
X, y, feature_names = make_corner_class_dataset(
    n_per_cluster=150,
    std_class1=0.4,
    std_other=0.7,
    a=3.0,
    random_state=42,
)

# Entrenar un clasificador rápido
clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, y)
print(f"Accuracy en el dataset sintético: {clf.score(X, y):.3f}")

# Visualización opcional (requiere matplotlib/pandas)
plot_corner_class_dataset(X, y, feature_names)
```

## Ejemplo de uso de `find_low_dim_spaces`

El siguiente ejemplo reproduce el flujo recomendado para explorar reglas en subespacios de baja dimensión a partir de una
selección de planos frontera (`sel`).

```python
from deldel.find_low_dim_spaces_fast import find_low_dim_spaces

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

### Informe textual de regiones

`describe_regions_report` genera un informe compacto en español a partir de las regiones descubiertas. Internamente:

- Ordena por Pareto → F1 → LiftPrecision, dando prioridad a las reglas más fuertes.
- Deduplica por firma estable (clase objetivo + dims + regla normalizada) y conserva la mejor variante según el ranking.
- Ofrece dos modos: ficha detallada por `region_id` o Top-K por clase (`top_per_class`).
- Calcula métricas de tamaño relativas (`dataset_size`) y normaliza las desigualdades de las reglas (`≤/≥` → `<=/>=`) para
  mostrar textos consistentes.

Ejemplo de uso rápido sobre el resultado de `find_low_dim_spaces`:

```python
from deldel import describe_regions_report

# Ficha individual (si conoces el id de la región)
print(describe_regions_report(valuable, region_id="rg_2d_c0_8ab309427e", dataset_size=X.shape[0]))

# Top-5 por clase con deduplicación
print(describe_regions_report(valuable, top_per_class=5, dataset_size=X.shape[0]))
```

## Visualización interactiva de fronteras y superficies

```python
from deldel import compute_frontier_planes_weighted, plot_frontiers_implicit_interactive_v2

planes   = compute_frontier_planes_weighted(records, prefer_cp=True, weight_map='softmax')

fig = plot_frontiers_implicit_interactive_v2(
    records, X, y,
    planes=planes, show_planes=True,
    dims=(0,1,3),
    detail="high",              # preset más denso
    grid_res_3d=72,             # o 80–96 si ves “escalones”
    extend=1.3, clamp_extend_to_X=True
)
```

```python
from deldel import fit_quadrics_from_records_weighted, plot_frontiers_implicit_interactive_v2

records = d.records_
quadrics = fit_quadrics_from_records_weighted(
    records, mode="logistic", C=8.0, density_k=8
)

X_sample = X[:1000]
y_sample = y[:1000]

fig = plot_frontiers_implicit_interactive_v2(
    records, X_sample, y_sample,
    quadrics=quadrics,
    show_planes=False,
    dims=(0,1,3),
    detail="high",              # preset más denso
    grid_res_3d=72,             # o 80–96 si ves “escalones”
    extend=4.3, clamp_extend_to_X=True
)
```

### Exploración interactiva de regiones y planos específicos

El helper `plot_selected_regions_interactive` permite abrir una figura de Plotly
con las regiones y planos devueltos por `find_low_dim_spaces` (o las regiones
globales contenidas en `sel`).  El snippet siguiente usa la versión integrada en
`deldel.reporting_plotting`, que pinta siempre el semiespacio \(n·x + b \le 0\)
orientado hacia la clase objetivo y puede consumir ids de reglas calculadas en
`valuable`.

```python
from deldel import describe_regions_report, plot_selected_regions_interactive

#   • Si es una región/regla devuelta por find_low_dim_spaces (p.ej. 'rg_2d_c0_8ab309427e'):
region_id_ = 'rg_3d_c0_648451c033'
print(describe_regions_report(
    valuable,
    region_id=region_id_,
    dataset_size=X.shape[0])
)

plot_selected_regions_interactive(sel, X, y,
    selected_region_ids=[region_id_],
    valuable=valuable)   # <— necesario para resolver ese ID
```

## Bundle de métodos núcleo

El módulo `subspaces.experiments.core_method_bundle` encapsula los cinco métodos ganadores y los expone a través de la función
`run_core_method_bundle`. El bundle incluye:

1. **ExtraTrees shallow routes** (`method_8_extratrees`)
2. **Gradient-Hessian synergy (mejorado)** (`method_10b_gradient_hessian`)
3. **Top-k guided combinations (mejorado)** (`method_1b_topk_guided`)
4. **Gradient synergy matrix** (`method_10_gradient_synergy`)
5. **Sparse projections Fisher-guided (mejorado)** (`method_6b_sparse_proj_guided`)

Los parámetros expuestos se reducen a los imprescindibles (`combo_sizes`, `random_state`, `cv_splits` y un `max_sets` opcional)
con heurísticas robustas para el resto.  Por defecto `max_sets=None`, así que el explorador conserva todos los subespacios
encontrados y los ordena de mayor a menor F1 sin recortes posteriores.

### Preset intermedio recomendado (y ahora por defecto)

Las pruebas con `subspaces/experiments/preset_middle_options.py` muestran que la
combinación `proxy_plus_microcv` + `method_8_extratrees` ofrece el mejor
compromiso velocidad/precisión entre las variantes intermedias: F1 promedio en
el rango 0.39–0.46 (lift >0.28 sobre el baseline ≈0.10) con tiempos totales de
≈0.18–0.20 s por dataset.【F:experiments_outputs/mid_preset_options.csv†L2-L5】
Por ello no solo es la opción por defecto del script, sino también del
explorador general: `MultiClassSubspaceExplorer.fit` ahora arranca en modo
`preset="proxy_microcv"` y con `method_key="method_8_extratrees"`, que rankea
candidatos por proxy MI y pasa un top reducido por micro-CV.  Para replicar el
atajo favorito basta con invocarlo directamente (sin necesidad de fijar el
método):

```python
explorer = MultiClassSubspaceExplorer(
    random_state=0,
    combo_sizes=(2, 3),
    fast_eval_budget=8,          # top del proxy que pasa a micro-CV
)
explorer.fit(X, y)  # usa method_8_extratrees por defecto
reports = explorer.get_report()
```

Si necesitas el barrido completo de opciones originales añade `--all-options`
al invocar el script de presets, usa `method_key=None` (o `"all"`) al llamar a
`fit` para evaluar todos los métodos, o cambia explícitamente
`preset="high_quality"` para volver al flujo exhaustivo previo.

### Regresión: `fit` sin depender de records

`MultiClassSubspaceExplorer.fit` ya no necesita (ni acepta) `records` para
rankear subespacios. El top-10 producido con la configuración por defecto de
los tests se mantiene estable tras la eliminación de esa dependencia: los
promedios de F1 coinciden en los diez primeros subespacios y quedan recogidos
en `experiments_outputs/multiclass_explorer_regression.csv` para trazabilidad.【F:experiments_outputs/multiclass_explorer_regression.csv†L1-L11】

> **Planes en reportes**<br>
> La lista `SubspaceReport.planes` se conserva por compatibilidad, pero ahora
> permanece vacía porque los planos no se derivan a partir de registros.

### Ejemplo rápido de uso

El siguiente fragmento ajusta un modelo base con `DelDel`, recopila los registros de cambios (`DeltaRecord`) y lanza el bundle
conservando todas las combinaciones evaluadas, ordenadas de mejor a peor. Si se quiere priorizar velocidad, el mismo flujo
 admite presets rápidos (`preset="fast"`) o ultra rápidos (`preset="ultra_fast"` con `skip_feature_stats=True`) desde el
 explorador subyacente.

> **ImportError/ModuleNotFoundError**<br>
> Si al ejecutar el ejemplo aparece `ModuleNotFoundError: No module named 'subspaces'`, significa que el paquete no ha sido
> instalado en el entorno actual. Basta con instalar el proyecto (p. ej. `pip install .` o `pip install -e .`) o, para notebooks
> rápidos, añadir `src/` al `PYTHONPATH` con `import sys; sys.path.append('/ruta/a/PureSheShe/src')` antes de los imports.

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from deldel import (
    ChangePointConfig,
    DelDel,
    DelDelConfig,
    compute_frontier_planes_all_modes,
    prune_and_orient_planes_unified_globalmaj,
)
from subspaces.experiments.core_method_bundle import run_core_method_bundle, summarize_core_bundle

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
)

# Bundle con los cinco métodos (sin recortes finales)
bundle = run_core_method_bundle(
    X,
    y,
    records,
    combo_sizes=(2, 3),
    random_state=0,
)

# Exploraciones rápidas opcionales (fuera del bundle) usando el mismo análisis
from deldel.subspace_change_detector import MultiClassSubspaceExplorer

explorer_fast = MultiClassSubspaceExplorer()
explorer_fast.fit(
    X,
    y,
    preset="fast",               # CV ligero y poda agresiva
)

explorer_ultra = MultiClassSubspaceExplorer()
explorer_ultra.fit(
    X,
    y,
    preset="ultra_fast",         # usa proxy MI sin CV
    skip_feature_stats=True,      # evita stumps/chi2
)

for report in bundle.reports:
    print(report.method_key, report.global_yes, report.top50_yes)

# Resumen ejecutivo por método
for method_summary in summarize_core_bundle(bundle):
    print(
        f"{method_summary.method_name}: {method_summary.selected_reports} seleccionados,",
        f"{method_summary.candidate_sets} candidatos,",
        f"{method_summary.elapsed_seconds:.4f}s"
    )

print(f"Selección final con {len(sel['winning_planes'])} planos ganadores")
```

### Obtener los subespacios top por método

`SubspaceReport` ahora incluye metadatos de procedencia (`method_key` y `method_keys`) que
permiten identificar rápidamente qué método generó cada subespacio. El siguiente fragmento
recorre todos los reportes y arma un diccionario con el mejor subespacio devuelto por cada
método del bundle:

```python
from subspaces.experiments.core_method_bundle import CORE_METHOD_KEYS, run_core_method_bundle

bundle = run_core_method_bundle(
    X,
    y,
    records,
    combo_sizes=(2, 3),
    random_state=0,
)

top_subspaces = {}
for key in CORE_METHOD_KEYS:
    candidatos = [r for r in bundle.reports if key in r.method_keys]
    if not candidatos:
        continue
    mejor = max(candidatos, key=lambda r: r.mean_macro_f1)
    top_subspaces[key] = {
        "features": mejor.features,  # columnas que definen el subespacio
        "f1": mejor.mean_macro_f1,
        "planes": mejor.planes,      # planos/half-spaces cuando hay registros disponibles
    }

for key, info in top_subspaces.items():
    print(f"{key}: {info['features']} (F1={info['f1']:.3f}, planos={len(info['planes'])})")
```

Si se quiere inspeccionar sólo los subespacios con planos generados, basta con filtrar
`info['planes']` o descartar los que tengan la lista vacía.

### Comparar contra ejecuciones individuales

`run_single_method` reutiliza la misma configuración y permite validar que cada técnica produce los mismos conteos cuando se
corre por separado.

```python
from subspaces.experiments.core_method_bundle import run_single_method, CORE_METHOD_KEYS

for key in CORE_METHOD_KEYS:
    explorer = run_single_method(X, y, key, max_sets=5, combo_sizes=(2, 3), random_state=0)
    report = explorer.get_report()[0]
    print(f"{key}: Sí globales={report.global_yes}, Top50={report.top50_yes}")
```

## Resultados reproducibles en datasets grandes

El script `subspaces/scripts/run_core_bundle_large_datasets.py` aplica el bundle sobre tres datasets sintéticos de mayor escala y
genera reportes en `subspaces/outputs/core_bundle/`.

```bash
PYTHONPATH=src:. python subspaces/scripts/run_core_bundle_large_datasets.py
```

### Estadísticas por dataset

| Clave | Dataset | Muestras | Columnas | Tiempo bundle (s) | Reportes evaluados |
| --- | --- | --- | --- | --- | --- |
| mix_large | Synthetic Mix Large | 1 400 | 24 | 29.3253 | 3 |
| wide_large | Synthetic Wide Large | 1 300 | 30 | 29.1262 | 3 |
| imbalanced_large | Synthetic Imbalanced Large | 1 250 | 26 | 25.8487 | 3 |

Los detalles completos están disponibles en `subspaces/outputs/core_bundle/core_bundle_dataset_stats.csv`.

### Métricas por método dentro del bundle

| Dataset | Método | Top 50 “Si” | Total “Si” | Tiempo (s) |
| --- | --- | --- | --- | --- |
| Synthetic Mix Large | ExtraTrees shallow routes | 0 | 120 | 0.200697 |
| Synthetic Mix Large | Gradient-Hessian synergy (mejorado) | 0 | 120 | 0.064258 |
| Synthetic Mix Large | Top-k guided combinations (mejorado) | 2 | 90 | 0.029602 |
| Synthetic Mix Large | Gradient synergy matrix | 0 | 120 | 0.130676 |
| Synthetic Mix Large | Sparse projections Fisher-guided (mejorado) | 3 | 97 | 0.025454 |
| Synthetic Wide Large | ExtraTrees shallow routes | 0 | 120 | 0.203004 |
| Synthetic Wide Large | Gradient-Hessian synergy (mejorado) | 0 | 120 | 0.061770 |
| Synthetic Wide Large | Top-k guided combinations (mejorado) | 3 | 90 | 0.029600 |
| Synthetic Wide Large | Gradient synergy matrix | 0 | 120 | 0.039737 |
| Synthetic Wide Large | Sparse projections Fisher-guided (mejorado) | 0 | 99 | 0.023454 |
| Synthetic Imbalanced Large | ExtraTrees shallow routes | 1 | 120 | 0.204090 |
| Synthetic Imbalanced Large | Gradient-Hessian synergy (mejorado) | 1 | 120 | 0.024008 |
| Synthetic Imbalanced Large | Top-k guided combinations (mejorado) | 2 | 90 | 0.030443 |
| Synthetic Imbalanced Large | Gradient synergy matrix | 0 | 120 | 0.030586 |
| Synthetic Imbalanced Large | Sparse projections Fisher-guided (mejorado) | 0 | 105 | 0.021795 |

Los resultados se guardan en `subspaces/outputs/core_bundle/core_bundle_summary.csv` para facilitar comparaciones futuras.

### Conclusiones rápidas sobre presets y skips

El barrido `experiments_outputs/core_bundle_preset_matrix.csv` ya no recorta los reportes: en `high_quality` y
`ultra_fast` se devuelven todos los conjuntos generados (37–46 y 20 respectivamente), preservando el ranking completo.
`fast` mantiene el mejor F1 de `high_quality` (≈0.52–0.57) con un presupuesto de 10 evaluaciones por dataset y tiempos de
~0.43–0.51 s; al activar `skip_feature_stats` se ahorran ~0.01–0.18 s sin perder el mejor F1 ni variar el número de candidatos.
`ultra_fast` sigue respondiendo en ~0.07–0.11 s con los 20 candidatos generados, pero el F1 se queda cerca del baseline, útil
solo para inspecciones relámpago.

## Recursos adicionales

- `subspaces/scripts/run_multi_dataset_method_analysis.py`: produce los rankings históricos por dataset y el top 5 global.
- `tests/test_core_method_bundle.py`: comprueba la paridad entre el bundle y las ejecuciones individuales.
- `docs/COLAB.md`: guía paso a paso para clonar, instalar y usar DelDel dentro de Google Colab.

### Uso rápido en Google Colab

1. Abra un notebook nuevo y ejecute:

   ```python
   !git clone https://github.com/<usuario>/PureSheShe.git
   %cd PureSheShe
   !pip install .
   ```

2. Importe los módulos necesarios dentro del mismo notebook:

   ```python
   import deldel
   from subspaces.experiments.core_method_bundle import run_core_method_bundle
   ```

3. Siga cualquiera de los ejemplos descritos anteriormente o revise la guía completa en `docs/COLAB.md` para más detalles (modo
editable, configuración de `PYTHONPATH`, etc.).
