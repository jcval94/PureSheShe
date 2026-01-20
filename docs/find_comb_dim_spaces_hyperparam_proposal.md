# Propuesta de hiperparametrización agresiva para `find_comb_dim_spaces`

## Objetivo
Maximizar la **cantidad de combinaciones de espacios** devueltas en `valuable` **por unidad de tiempo**, manteniendo el coste de cómputo controlado. La idea es explotar los hiperparámetros que más influyen en el crecimiento combinatorio sin disparar el tiempo de búsqueda.

> **KPI recomendado**
>
> ```python
> total_rules = sum(len(v) for v in valuable.values())
> combos_per_sec = total_rules / elapsed_seconds
> ```
>
> Esta métrica permite comparar configuraciones que generan muchas reglas en poco tiempo.

## Parámetros con mayor impacto (y costo)
Estos parámetros aparecen en la API pública y controlan directamente el espacio de búsqueda y el número de reglas candidatas. **Son los principales “aceleradores” del número de combinaciones**, pero también son los que más incrementan el tiempo de ejecución.【F:src/deldel/combiantions.py†L1211-L1230】

1. **`max_planes`**: límite de planos por regla. Es el mayor multiplicador del espacio combinatorio (más planos ⇒ más combinaciones).【F:src/deldel/combiantions.py†L1217-L1220】
2. **`beam_width`**: ancho del beam search; más anchura explora más reglas candidatas por nivel.【F:src/deldel/combiantions.py†L1221-L1223】
3. **`max_candidates_per_class`**: tope de candidatos por clase antes de filtrar a reglas finales.【F:src/deldel/combiantions.py†L1223-L1225】
4. **`max_rules_per_class`**: tope final de reglas por clase en `valuable` (tu ejemplo ya está alto).【F:src/deldel/combiantions.py†L1224-L1226】
5. **`min_size` y `lift_min`**: umbrales de poda (más bajos ⇒ más reglas, más tiempo).【F:src/deldel/combiantions.py†L1222-L1224】
6. **`mode`**: variantes Hessianas (`hessian_rank`/`hessian_filter`) pueden acelerar o filtrar agresivamente. Para maximizar combinaciones por tiempo, **`hessian_rank`** suele ofrecer más velocidad sin filtrar tanto como `hessian_filter` (que reduce mucho reglas).【F:src/deldel/combiantions.py†L1214-L1262】

> **Nota:** `top_k_floor_per_dim` solo marca reglas como “floor”; no aumenta el número total de reglas generadas, así que no mejora la métrica combos/segundo.【F:src/deldel/combiantions.py†L1225-L1230】

---

## Estrategia propuesta (agresiva, pero eficiente)

### 1) Búsqueda en dos etapas (rápida → intensiva)

**Etapa A (exploración rápida con Hessian Rank):**
- Objetivo: medir qué regiones del espacio hiperparamétrico producen más reglas por segundo.
- Configuración base:

```python
valuable = find_comb_dim_spaces(
    sel,
    X,
    y,
    mode="hessian_rank",
    metric="f1",
    max_planes=6,
    beam_width=12,
    max_candidates_per_class=150,
    max_rules_per_class=200,
    min_size=3,
    lift_min=0.9,
)
```

**Etapa B (intensificación agresiva):**
- Se parte de las 2–3 mejores configuraciones de Etapa A (por combos/seg). Se incrementan los multiplicadores combinatorios.
- Configuración sugerida:

```python
valuable = find_comb_dim_spaces(
    sel,
    X,
    y,
    mode="hessian_rank",
    metric="f1",
    max_planes=8,
    beam_width=24,
    max_candidates_per_class=300,
    max_rules_per_class=480,
    min_size=3,
    lift_min=0.8,
)
```

Si el tiempo sube demasiado, vuelve a **`max_planes=7`** o **`beam_width=16`**, que son los controles más sensibles al tiempo.

---

## Grid agresivo recomendado (prioriza combinaciones/tiempo)
Usa un **grid mínimo pero potente**, para no invertir tiempo en combinaciones demasiado similares. Cada punto debe medirse con `combos_per_sec`:

| Parámetro | Valores sugeridos | Justificación |
|---|---|---|
| `mode` | `hessian_rank`, `base` | `hessian_rank` da mayor velocidad; `base` sirve como referencia. |
| `max_planes` | 6, 8, 10 | Principal multiplicador del número de combinaciones. |
| `beam_width` | 12, 24, 36 | Amplifica la exploración de combinaciones por nivel. |
| `max_candidates_per_class` | 150, 300, 600 | Controla el embudo de candidatos antes del filtrado final. |
| `max_rules_per_class` | 240, 480 | Asegura que la salida no sea el cuello de botella. |
| `min_size` | 3, 5 | Relaja la poda (3 es más agresivo). |
| `lift_min` | 0.8, 1.0 | Más bajo ⇒ más reglas válidas. |

**Regla práctica:**
- Si `total_rules` queda cerca de `max_rules_per_class` en muchas clases, **sube `max_rules_per_class`**.
- Si `total_rules` cae mucho al subir `max_planes`, **sube `beam_width` o baja `lift_min`**.
- Si el tiempo crece más rápido que el número de reglas, **baja `max_planes` primero**, luego `beam_width`.

---

## Heurísticas para mantener el tiempo bajo control

1. **Usar `mode="hessian_rank"` como default** para búsqueda agresiva rápida (es la variante más veloz sin filtrar tanto como `hessian_filter`).【F:src/deldel/combiantions.py†L1214-L1262】
2. **No activar `include_masks` ni `include_planes_used`** en exploración; solo en etapas de análisis, porque agregan sobrecosto de memoria y serialización.【F:src/deldel/combiantions.py†L1226-L1230】
3. **Ajuste gradual:** subir un solo multiplicador combinatorio por iteración (p. ej., subir `max_planes` primero; si responde bien, luego `beam_width`).

---

## Plantilla de experimento (comparación rápida)

```python
import time

configs = [
    dict(mode="hessian_rank", max_planes=6, beam_width=12, max_candidates_per_class=150,
         max_rules_per_class=240, min_size=3, lift_min=0.9),
    dict(mode="hessian_rank", max_planes=8, beam_width=24, max_candidates_per_class=300,
         max_rules_per_class=480, min_size=3, lift_min=0.8),
    dict(mode="base", max_planes=8, beam_width=24, max_candidates_per_class=300,
         max_rules_per_class=480, min_size=3, lift_min=0.8),
]

for cfg in configs:
    t0 = time.time()
    valuable = find_comb_dim_spaces(sel, X, y, metric="f1", **cfg)
    elapsed = time.time() - t0
    total_rules = sum(len(v) for v in valuable.values())
    print(cfg, total_rules, total_rules / elapsed)
```

---

## Parte 2: qué cambiar después de obtener resultados (y por qué)

Una vez tengas el ranking por `combos_per_sec`, aplica estos ajustes **basados en síntomas**:

### A) Muchas reglas, pero tiempo demasiado alto
**Síntoma:** `total_rules` alto, pero `combos_per_sec` cae.

**Ajustes recomendados:**
1. **Bajar `max_planes` en 1–2 niveles**: es el multiplicador combinatorio más costoso en tiempo.【F:src/deldel/combiantions.py†L1217-L1220】
2. **Reducir `beam_width` un escalón** si el tiempo sigue alto: menos expansión por nivel.【F:src/deldel/combiantions.py†L1221-L1223】
3. **Subir `lift_min` a 1.0** si aún hay demasiadas reglas con bajo valor predictivo: poda más agresiva.【F:src/deldel/combiantions.py†L1222-L1224】

**Por qué:** estos parámetros son los que más crecen exponencialmente el número de combinaciones. Reducirlos baja el coste sin eliminar toda la variedad de reglas.

### B) Pocas reglas (output “se queda corto”)
**Síntoma:** `total_rules` muy por debajo de `max_rules_per_class`.

**Ajustes recomendados:**
1. **Bajar `lift_min` (0.8 → 0.6)** para aceptar regiones con menor lift, que igual aportan combinaciones nuevas.【F:src/deldel/combiantions.py†L1222-L1224】
2. **Bajar `min_size` (5 → 3)** para permitir reglas más pequeñas.【F:src/deldel/combiantions.py†L1222-L1224】
3. **Subir `beam_width`** si el explorador parece quedarse sin candidatos temprano.【F:src/deldel/combiantions.py†L1221-L1223】

**Por qué:** la poda temprana suele eliminar demasiadas reglas; relajarla incrementa el número de combinaciones.

### C) Muchas reglas, pero solo en pocas clases
**Síntoma:** una o dos clases saturan el output, otras quedan vacías.

**Ajustes recomendados:**
1. **Subir `max_candidates_per_class`** para ampliar la exploración en clases pobres.【F:src/deldel/combiantions.py†L1223-L1225】
2. **Subir `max_rules_per_class`** si hay saturación en clases fuertes, para no “recortar” su diversidad.【F:src/deldel/combiantions.py†L1224-L1226】

**Por qué:** el embudo por clase limita la diversidad; expandirlo balancea la distribución.

### D) Buen número de reglas, pero baja calidad global
**Síntoma:** `total_rules` alto, pero métricas `f1`/`precision` bajas.

**Ajustes recomendados:**
1. **Subir `lift_min` y `min_size`** para recortar reglas débiles.【F:src/deldel/combiantions.py†L1222-L1224】
2. **Bajar `max_planes`** para evitar combinaciones complejas con sobreajuste.【F:src/deldel/combiantions.py†L1217-L1220】

**Por qué:** la agresividad combinatoria aumenta la diversidad, pero también el ruido; endurecer filtros recupera calidad.

### E) Velocidad muy buena, pero pocas combinaciones por dimensión
**Síntoma:** `combos_per_sec` alto pero `valuable` concentra casi todo en 1–2 dimensiones.

**Ajustes recomendados:**
1. **Subir `max_planes`** y mantener `beam_width` estable.
2. **Cambiar a `mode="base"`** para ver si el sesgo del `hessian_rank` está concentrando demasiado.【F:src/deldel/combiantions.py†L1214-L1262】

**Por qué:** más planos permiten reglas multi-dim; el modo base explora sin priorización Hessiana.

---

## Resultado esperado
- **Más combinaciones útiles en menos tiempo** al priorizar `max_planes`/`beam_width` y usar `hessian_rank`.
- Control de costes al medir siempre **combos/seg** y ajustar en función del cuello de botella (tiempo vs. reglas).

Si quieres, puedo convertir esta propuesta en un script automatizado para ejecutar el grid y guardar resultados en un CSV para comparar configuraciones.
