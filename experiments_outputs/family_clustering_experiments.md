# Comparativa de métodos de agrupamiento de familias

Se generaron pools sintéticos de medio-espacios con familias conocidas (700 planos, 12 dimensiones, familias de 3–8 miembros) y se repitió el experimento con 3 semillas. Se evaluaron cuatro estrategias:

- **greedy**: bucle original dependiente del orden.
- **connected**: componentes conectados sobre la matriz binaria de similitud.
- **kmeans**: K-medias sobre la matriz de distancia precomputada (k acotado por el estimado de componentes).
- **dbscan**: densidad sobre matriz precomputada.

## Métricas agregadas (3 corridas por método)

| Método        | ARI medio | ARI mín–máx | Desv. ARI | Tiempo medio (s) | Tiempo mín–máx (s) | Desv. tiempo (s) | Familias pred. (media ± des v.) |
|---------------|-----------|-------------|-----------|------------------|--------------------|------------------|---------------------------------|
| connected     | 0.6183    | 0.5824–0.6385 | 0.0312  | 0.0109           | 0.0101–0.0114      | 0.0007           | 356.33 ± 9.29                    |
| dbscan        | 0.6183    | 0.5824–0.6385 | 0.0312  | 0.0649           | 0.0378–0.0855      | 0.0245           | 356.33 ± 9.29                    |
| greedy (base) | 0.3839    | 0.3650–0.3942 | 0.0164  | 2.8691           | 2.7977–2.9534      | 0.0787           | 429.33 ± 4.93                    |
| kmeans        | 0.0068    | 0.0066–0.0070 | 0.0002  | 0.0980           | 0.0700–0.1286      | 0.0294           | 50.00 ± 0.00                     |

## Observaciones

- **Connected components** sigue siendo el mejor compromiso: ~260× más rápido que el greedy y con +0.23 puntos de ARI promedio, manteniendo un conteo de familias cercano al verdadero.
- **DBSCAN** iguala el ARI de connected con ~6× más tiempo que connected, pero aún ~44× más rápido que el greedy.
- **K-means** no captura la estructura de familias (ARI ~0.007) y colapsa agresivamente en 50 clusters, por lo que no es adecuado en este espacio de similitud.
- El enfoque original `greedy` sigue siendo sensible al orden y es con diferencia el más lento.
