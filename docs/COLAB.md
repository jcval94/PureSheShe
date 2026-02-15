# Uso de DelDel en Google Colab

Esta guía explica los pasos para clonar el repositorio, instalar la biblioteca mediante `pip install .` y verificar que Colab reconozca el paquete `deldel`.

## Requisitos previos

- Entorno con Python 3.9 o superior (Colab estándar usa Python 3.10, por lo que no requiere cambios).
- `git` y `pip` disponibles (ambos vienen preinstalados en Colab).

## Pasos en Colab

1. **Clonar el repositorio**

   ```python
   !git clone https://github.com/<usuario>/PureSheShe.git
   %cd PureSheShe
   ```

2. **Instalar las dependencias declaradas en `requirements.txt`**

   ```python
   !pip install -r requirements.txt
   ```

   Esto instala `numpy`, `pandas`, `plotly` y `scikit-learn` sin necesidad de empaquetar todavía el proyecto.

3. **Instalar la biblioteca (modo normal o editable)**

   ```python
   !pip install .
   # para desarrollo interactivo:
   # !pip install -e .[dev]
   ```

   Gracias a la configuración de `pyproject.toml`, el comando localiza automáticamente los paquetes dentro de `src/` y, si se
   usa `.[dev]`, añade `pytest` para correr pruebas.

4. **Verificar la instalación**

   ```python
   import deldel
   print(deldel.__version__ if hasattr(deldel, "__version__") else "DelDel importado correctamente")
   ```

## Notas adicionales

- Si se necesita acceder a scripts auxiliares fuera del paquete (por ejemplo `subspaces/scripts`), defina `PYTHONPATH` en la sesión:

  ```python
  import os
  os.environ["PYTHONPATH"] = ":".join([os.environ.get("PYTHONPATH", ""), "src"])
  ```

- Para usar GPUs o TPUs solo es necesario activar el acelerador en `Entorno de ejecución > Cambiar tipo de entorno de ejecución` antes de ejecutar las celdas anteriores.

Con estos pasos, Colab queda listo para ejecutar cualquiera de los ejemplos descritos en el README.

## Ejemplo completo: comparar `model_types`, métricas Top-K y reglas

Puedes ejecutar un flujo más completo con gráficas y reglas lógicas (AND/OR/DNF) con:

```python
%%time

!python experiments_outputs/run_model_type_regions_dashboard.py
```

Si quieres correrlo inline en una celda, este es el bloque mínimo (incluyendo la llamada pedida a `describe_regions_report`):

```python
%%time

model_types = [
    "base", "default", "hessian_rank", "hessian_filter", "and", "or", "dnf",
    "and_or_beam", "and_or_random", "and_or_diverse", "and_or_greedy",
]

from deldel import find_comb_dim_spaces_full, describe_regions_report

for model_type in model_types:
    print(model_type)
    valuable = find_comb_dim_spaces_full(
        sel,
        X,
        y,
        mode=model_type,
        max_planes=12,
        metric="f1",
        beam_width=36,
        max_rules_per_class=480,
    )

    mets = describe_regions_report(
        valuable,
        top_per_class=9,
        dataset_size=X.shape[0],
        return_average_metrics=True,
    )

    print("global:", mets["global_mean"])
    print("reglas en details:", len(mets["details"]))
```

El script `experiments_outputs/run_model_type_regions_dashboard.py` además guarda:

- `summary_metrics.csv` (comparativa global y por clase)
- `top_rules_by_class.csv` (reglas top, con expresión lógica)
- `global_ranking.html`, `per_class_ranking.html`, `rules_f1_vs_lift.html` (gráficas interactivas)

### Ejecución directa en Colab (paso a paso)

Si lo quieres correr exactamente con:

```python
!python experiments_outputs/run_model_type_regions_dashboard.py
```

asegúrate antes de estar dentro de la carpeta del repo y tener dependencias instaladas:

```python
!git clone https://github.com/<usuario>/PureSheShe.git
%cd PureSheShe
!pip install -r requirements.txt
```

Ahora sí, ejecútalo:

```python
!python experiments_outputs/run_model_type_regions_dashboard.py
```

Si sale `No such file or directory`, verifica con:

```python
!pwd
!ls experiments_outputs
```

y vuelve a hacer `%cd PureSheShe`.
