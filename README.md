# DelDel

DelDel es una biblioteca ligera para instrumentar modelos de clasificación y analizar sus fronteras de decisión. Incluye utilidades para registrar llamadas, explorar pares de puntos, detectar cambios de etiqueta y visualizar fronteras con Plotly.

## Instalación

```
pip install .
```

## Uso rápido

```python
from deldel import (
    DelDel,
    DelDelConfig,
    ChangePointConfig,
    compute_frontier_planes_weighted,
    plot_frontiers_implicit_interactive_v2,
)
```

## Ejemplo completo

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from deldel import (
    DelDel,
    DelDelConfig,
    ChangePointConfig,
    compute_frontier_planes_weighted,
    plot_frontiers_implicit_interactive_v2,
)

# ====== Datos de ejemplo ======
X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)

# ====== Configuración ======
cfg = DelDelConfig(
    segments_target=120,    # total de segmentos que el usuario quiere
    random_state=0
)

# ChangePointConfig optimizado:
# - Se limita el número de candidatos
# - Se reducen iteraciones de bisección
# - Se pone un límite por record
cp_cfg = ChangePointConfig(
    enabled=False,
    mode="treefast",          # puede ser "treefast" o "generic"
    per_record_max_points=8,   # máximo 8 puntos de cambio por segmento
    max_candidates=128,        # antes era 4096
    max_bisect_iters=8         # antes era 22
)

# ====== Ejecución ======
d = DelDel(cfg, cp_cfg).fit(X, model)

print("Resumen de puntos de cambio por Top-5 records:")
for r in d.topk(5):
    print(f"y0={r.y0} → y1={r.y1} | flips={r.cp_count} | ΔL2={r.delta_norm_l2:.3f}")
    if r.cp_count > 0:
        print("   ts:", np.round(r.cp_t, 3), "| clases:", list(zip(r.cp_y_left, r.cp_y_right)))

# ====== Tiempos ======
print("\nTiempos de ejecución (ms):")
for k, v in d.time_stats_["timings_ms"].items():
    print(f"{k:35s}: {v:.1f}")

records = d.records_
print(len(records), sum(r.y0 == r.y1 for r in records))
planes = compute_frontier_planes_weighted(records, prefer_cp=True, weight_map="softmax")

fig = plot_frontiers_implicit_interactive_v2(
    records, X, y,
    planes=planes, show_planes=True,
    dims=(0, 1, 3),
    detail="high",              # preset más denso
    grid_res_3d=72,             # o 80–96 si ves “escalones”
    extend=1.3, clamp_extend_to_X=True
)

# === Ejemplo 1: Resumen ejecutivo de llamadas (stage x fn) ===
import pandas as pd
from collections import defaultdict

print(len(d.calls_))
llams = pd.DataFrame(d.calls_)

def calls_summary(calls):
    by_stage_fn = defaultdict(lambda: defaultdict(lambda: {
        "count": 0,
        "ms": 0.0,
        "hits": 0,
        "miss": 0,
        "batch": 0,
    }))
    for c in calls:
        st = c.get("stage") or "unknown"
        fn = c.get("fn") or "unknown"
        rec = by_stage_fn[st][fn]
        rec["count"] += 1
        rec["ms"] += float(c.get("duration_ms", 0.0))
        rec["hits"] += int(c.get("cache_hits", 0))
        rec["miss"] += int(c.get("cache_misses", 0))
        rec["batch"] += int(c.get("batch", 0))

    rows = []
    for stage, dfn in by_stage_fn.items():
        for fn, r in dfn.items():
            total = r["hits"] + r["miss"]
            hitrate = (r["hits"] / total * 100.0) if total > 0 else 0.0
            rows.append({
                "stage": stage,
                "fn": fn,
                "calls": r["count"],
                "avg_ms": r["ms"] / max(1, r["count"]),
                "tot_ms": r["ms"],
                "hit_rate_%": hitrate,
                "avg_batch": r["batch"] / max(1, r["count"]),
            })

    rows.sort(key=lambda z: z["tot_ms"], reverse=True)
    return rows

def print_table(rows, top=None, title="RESUMEN EJECUTIVO"):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
    hdr = f"{'stage':38}  {'fn':18}  {'calls':>6}  {'avg_ms':>8}  {'tot_ms':>9}  {'hit%':>6}  {'avg_batch':>10}"
    print(hdr)
    print("-" * len(hdr))
    k = 0
    for r in rows:
        if top and k >= top:
            break
        print(
            f"{r['stage'][:38]:38}  {r['fn'][:18]:18}  {r['calls']:6d}  {r['avg_ms']:8.2f}  {r['tot_ms']:9.2f}  {r['hit_rate_%']:6.1f}  {r['avg_batch']:10.1f}"
        )
        k += 1

rows = calls_summary(d.calls_)
print_table(rows, top=12, title="Reporte ejecutivo (para el jefe que todo lo ve)")
```
