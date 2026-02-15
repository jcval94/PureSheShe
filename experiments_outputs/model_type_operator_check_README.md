# Validación de operadores en `describe_regions_report`

- Modos evaluados: **11**.
- Modos en PASS: **11/11**.
- Modos A U B y A INT B esperados: dnf, and_or_beam, and_or_random, and_or_diverse, and_or_greedy.
- Modos A U B y A INT B que muestran ambas en una misma `Regla:`: dnf, and_or_beam, and_or_random, and_or_diverse, and_or_greedy.

## Conclusiones

- Los modos de unión pura (`or`) muestran `OR` y no muestran `AND` en las reglas del reporte.
- Los modos de intersección pura (`base`, `default`, `hessian_rank`, `hessian_filter`, `and`) muestran `AND` y no muestran `OR`.
- Los modos mixtos (`dnf`, `and_or_*`) muestran `OR` y `AND`; además se observan reglas donde ambos operadores conviven en la misma línea `Regla:`.
