# DelDel

DelDel es una biblioteca ligera para instrumentar clasificadores y descubrir reglas en subespacios de baja dimensión. La
configuración por defecto se centra en el `MultiClassSubspaceExplorer`, que ahora puede ejecutarse con un bundle curado de los
cinco mejores métodos derivados del análisis multi-dataset.

## Instalación

```bash
pip install .
```

## Bundle de métodos núcleo

El módulo `subspaces.experiments.core_method_bundle` encapsula los cinco métodos ganadores y los expone a través de la función
`run_core_method_bundle`. El bundle incluye:

1. **ExtraTrees shallow routes** (`method_8_extratrees`)
2. **Gradient-Hessian synergy (mejorado)** (`method_10b_gradient_hessian`)
3. **Top-k guided combinations (mejorado)** (`method_1b_topk_guided`)
4. **Gradient synergy matrix** (`method_10_gradient_synergy`)
5. **Sparse projections Fisher-guided (mejorado)** (`method_6b_sparse_proj_guided`)

Los parámetros expuestos se reducen a los imprescindibles (`max_sets`, `combo_sizes`, `random_state`, `cv_splits`) y se aplican
heurísticas robustas para el resto de la configuración.

### Ejemplo rápido de uso

El siguiente fragmento ajusta un modelo base con `DelDel`, recopila los registros de cambios (`DeltaRecord`) y lanza el bundle con
un máximo de cinco combinaciones evaluadas por método.

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from deldel import ChangePointConfig, DelDel, DelDelConfig
from subspaces.experiments.core_method_bundle import run_core_method_bundle

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

# Bundle con los cinco métodos
bundle = run_core_method_bundle(
    X,
    y,
    records,
    max_sets=5,
    combo_sizes=(2, 3),
    random_state=0,
)

for report in bundle.reports:
    print(report.method_key, report.global_yes, report.top50_yes)
```

### Comparar contra ejecuciones individuales

`run_single_method` reutiliza la misma configuración y permite validar que cada técnica produce los mismos conteos cuando se
corre por separado.

```python
from subspaces.experiments.core_method_bundle import run_single_method, CORE_METHOD_KEYS

for key in CORE_METHOD_KEYS:
    explorer = run_single_method(X, y, records, key, max_sets=5, combo_sizes=(2, 3), random_state=0)
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

## Recursos adicionales

- `subspaces/scripts/run_multi_dataset_method_analysis.py`: produce los rankings históricos por dataset y el top 5 global.
- `tests/test_core_method_bundle.py`: comprueba la paridad entre el bundle y las ejecuciones individuales.
