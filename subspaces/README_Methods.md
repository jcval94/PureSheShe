# Catálogo de métodos de generación de subespacios

La siguiente tabla resume las estrategias disponibles dentro de `MultiClassSubspaceExplorer`.
Cada método base cuenta con una variante **mejorada** que refina la selección sin disparar los
tiempos de ejecución.

| ID | Método base | Etapa | Idea clave | Ajuste en la variante mejorada |
|----|-------------|-------|------------|--------------------------------|
| 1 | Top-k random combinations | Candidatos | Combinaciones aleatorias sobre las columnas mejor rankeadas. | Variante guiada pondera MI y penaliza correlación para priorizar parejas complementarias. |
| 2 | Chi2 top combinations | Filtro / Candidatos | Combina variables con mayor estadístico χ² tras binning. | Re-puntuación con χ², MI y diversidad categórica antes de aceptar un conjunto. |
| 3 | High correlation groups | Candidatos | Agrupa columnas numéricas altamente correlacionadas. | Exige correlación fuerte y suma de MI elevada para retener pares y tríos. |
| 4 | Random forest pair frequency | Candidatos | Cuenta pares que aparecen con frecuencia en rutas de RandomForest. | Suma ganancias de impureza por ruta y extiende la selección a combinaciones mayores. |
| 5 | Leverage score filter | Filtro | Usa leverage scores vía SVD aleatorizado para detectar columnas influyentes. | Añade leverage por clase y dispersión entre medias para resaltar variables discriminantes. |
| 6 | Sparse random projections | Filtro / Candidatos | Proyecciones dispersas con criterio Fisher para ubicar cargas relevantes. | Acumula pesos por proyección y arma combinaciones con mayor contribución Fisher. |
| 7 | Lazy greedy dispersion | Ranking | Selección greedy con penalización por correlación entre columnas. | Incorpora varianzas y un lookahead log-det para evaluar el aporte incremental. |
| 8 | ExtraTrees shallow routes | Candidatos | Usa rutas poco profundas de ExtraTrees como fuente de combinaciones. | Pondera por pureza de hoja y profundidad efectiva antes de priorizar cada conjunto. |
| 9 | CountSketch heavy hitters | Candidatos | Sketch hashing para detectar interacciones frecuentes por clase. | Mide contraste entre clases en el sketch para filtrar falsos positivos. |
| 10 | Gradient synergy matrix | Ranking | Matriz Xᵀ diag(||g||) X derivada del gradiente de una LR. | Combina gradientes y términos Hessianos aproximados para priorizar sinergias útiles. |
| 11 | MinHash class co-occurrence | Candidatos | MinHash/LSH sobre binarizaciones para hallar columnas que co-ocurren. | Cruza firmas por banda y pondera la cobertura diferencial por clase. |

## Cómo mantener los métodos eficientes

* **Reutilizar pre-cálculos** para evitar recomputaciones pesadas.
* **Submuestreos controlados** mediante sketches, argpartition y muestreos aleatorios.
* **Vectorización completa** sobre NumPy/BLAS en los cálculos intensivos.
* **Heurísticas perezosas** en lugar de loops exhaustivos para mantener tiempos bajos.

Las once estrategias originales y sus once variantes reforzadas trabajan en conjunto para
ofrecer un catálogo amplio de subespacios candidatos manteniendo exploraciones rápidas.
