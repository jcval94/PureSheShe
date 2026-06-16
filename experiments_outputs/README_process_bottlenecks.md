# A/B test de cuellos de botella

Dataset: corner sintetico con 12,800 filas y 4 features.
Metrica: F1 y lift de precision promedio Top-3 por clase con `describe_regions_metrics`.
Objetivo: mantener la salida visible parecida a la original y reducir tiempo.

## Cuellos de botella encontrados

1. El armado de texto de reglas repetia trabajo geometrico por cada combinacion candidata.
2. El fallback de cobertura por clase guardaba todos los candidatos rechazados aunque solo usa los mejores pocos por clase/dimension.
3. Se materializaban mascaras booleanas internas para relaciones/uniones y luego se borraban de la salida publica.
4. El cap global por clase muestreaba candidatos al azar despues del ranking por prioridad, con riesgo de perder planos buenos.

## Resultado

Mejor fila implementada: `implemented_internal_fast_path` con speedup 1.44x, delta F1 Top-3 +0.000000, delta lift Top-3 +0.000000, y similitud de reglas top 1.00.

## CSV

Detalle completo: `experiments_outputs/process_bottleneck_ab_results.csv`.

| Variante | Implementado | Speedup | Similitud | Delta F1 | Delta lift | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| original_head | baseline | 1.00x | 1.00 | +0.000000 | +0.000000 | reference |
| implemented_internal_fast_path | yes | 1.44x | 1.00 | +0.000000 | +0.000000 | keep |
| config_no_relations | no | 1.19x | 1.00 | +0.000000 | +0.000000 | do not default; metadata loss |
| config_no_unions | no | 1.49x | 1.00 | +0.000000 | +0.000000 | do not default; output less complete |
| config_no_unions_no_relations | no | 2.13x | 1.00 | +0.000000 | +0.000000 | fast option only; too much metadata loss |
| support_first_existing | already available | 1.71x | 1.00 | +0.000000 | +0.000000 | keep as opt-in preset |
| deterministic_existing | already available | 2.33x | 1.00 | +0.000000 | +0.000000 | keep as opt-in preset |
