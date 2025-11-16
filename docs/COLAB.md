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
