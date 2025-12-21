# Observaciones sobre `find_low_dim_spaces`

## Cobertura de planos candidatos
- Los planos que llegan a `find_low_dim_spaces` se toman de `sel['by_pair_augmented'][pair]['winning_planes']` y de `sel['regions_global']['per_plane']`, asignando una orientación (`side_for_cls`) en función del par de clases de origen.
- Ahora se pueden clonar los planos con precisión casi perfecta (`priority_perfect`) usando la desigualdad opuesta para la misma clase. Esto expone también el lado invertido cuando es prometedor, mitigando el hueco detectado en la versión anterior.

## Priorización por clase (precisión y lift)
- Cada plano conserva sus métricas **por clase** (`priority_precision` y `priority_lift`), y la priorización usa esa dupla en lugar de un simple umbral global. Se ordenan primero los planos perfectos y, dentro de cada clase, los de mayor precisión y lift antes de aplicar `max_planes_per_pair`, evitando descartar candidatos óptimos por azar o por promedios entre clases.
- La enumeración de combinaciones respeta ese orden (sin barajado), por lo que las reglas empiezan a formarse con los planos más prometedores para cada clase.

## Impacto
- Las regiones derivadas de planos con precisión 100% quedan primero en la cola de evaluación y, si procede, se prueban sus lados opuestos. En el experimento sintético de verificación, la F1 global subió de 0.57 a 0.80 cuando se activaron ambas mejoras, evidenciando su efecto en reglas simples de 1 plano.
