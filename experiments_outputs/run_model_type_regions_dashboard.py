"""Dashboard comparativo de model_types usando reportes por regiones y gráficas.

Uso rápido:
    python experiments_outputs/run_model_type_regions_dashboard.py

El script:
1) Construye un dataset sintético multiclase.
2) Ejecuta `find_comb_dim_spaces_full` para varios `model_types`.
3) Calcula métricas agregadas con:
       mets = describe_regions_report(
           valuable,
           top_per_class=9,
           dataset_size=X.shape[0],
           return_average_metrics=True,
       )
4) Exporta tablas CSV y gráficas HTML (Plotly) para comparar qué tipo rinde mejor.
5) Imprime reglas top por clase y por tipo de modelo con conectores lógicos explícitos
   (AND / OR, incluyendo formato DNF cuando aplica).
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deldel import describe_regions_report, find_comb_dim_spaces_full  # noqa: E402
from deldel.datasets import make_corner_class_dataset  # noqa: E402
from experiments_outputs.run_comb_vs_low_dim_competition import (  # noqa: E402
    build_synthetic_sel,
    ensure_norm_fields,
    enrich_plane_metrics,
)

MODEL_TYPES = [
    "base",
    "default",
    "hessian_rank",
    "hessian_filter",
    "and",
    "or",
    "dnf",
    "and_or_beam",
    "and_or_random",
    "and_or_diverse",
    "and_or_greedy",
]


def _rule_to_boolean_expression(region: Dict[str, Any]) -> str:
    pieces = [str(p).strip() for p in (region.get("rule_pieces") or []) if str(p).strip()]
    rule_text = str(region.get("rule_text") or "").strip()
    seed_type = str(region.get("seed_type") or "").lower()

    if not pieces:
        return rule_text or "(sin regla)"

    # DNF o familias and_or: cada pieza puede ser una cláusula AND, unidas con OR.
    if seed_type in {"dnf", "and_or_beam", "and_or_random", "and_or_diverse", "and_or_greedy"}:
        clauses = []
        for piece in pieces:
            text = piece.strip()
            if " AND " in text and not text.startswith("("):
                text = f"({text})"
            clauses.append(text)
        return " OR ".join(clauses)

    # Modo OR (beam_or*): unión OR entre literales.
    if seed_type.startswith("beam_or") or seed_type == "or":
        return " OR ".join(pieces)

    # Por defecto: AND entre literales.
    return " AND ".join(pieces)


def _flatten_regions(valuable: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    regions: List[Dict[str, Any]] = []
    for recs in (valuable or {}).values():
        regions.extend(recs or [])
    return regions


def _top_regions_by_class(valuable: Dict[int, List[Dict[str, Any]]], top_n: int) -> Dict[int, List[Dict[str, Any]]]:
    regions = _flatten_regions(valuable)
    by_class: Dict[int, List[Dict[str, Any]]] = {}
    for region in regions:
        cls = int(region.get("target_class"))
        by_class.setdefault(cls, []).append(region)

    for cls, recs in by_class.items():
        recs.sort(
            key=lambda r: (
                float((r.get("metrics") or {}).get("f1", 0.0)),
                float((r.get("metrics") or {}).get("lift_precision", 0.0)),
            ),
            reverse=True,
        )
        by_class[cls] = recs[:top_n]
    return by_class


def run_dashboard() -> Tuple[pd.DataFrame, pd.DataFrame]:
    X, y, feature_names = make_corner_class_dataset(
        n_per_cluster=180,
        std_class1=0.35,
        std_other=0.75,
        a=3.0,
        random_state=42,
    )
    sel = build_synthetic_sel(X, y, feature_names, num_thresholds=10)
    sel = ensure_norm_fields(sel)
    sel = enrich_plane_metrics(sel, X, y)

    rows_summary: List[Dict[str, Any]] = []
    rows_rules: List[Dict[str, Any]] = []

    print("=== Comparativa de model_types ===")
    for model_type in MODEL_TYPES:
        print(f"\n>>> {model_type}")

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

        details = list(mets.get("details") or [])
        print(f"Reglas evaluadas (top acumulado): {len(details)}")

        global_mean = dict(mets.get("global_mean") or {})
        rows_summary.append(
            {
                "model_type": model_type,
                "global_mean_f1": float(global_mean.get("f1") or 0.0),
                "global_mean_lift_precision": float(global_mean.get("lift_precision") or 0.0),
                "num_top_rules": len(details),
                "num_total_regions": len(_flatten_regions(valuable)),
            }
        )

        per_class = dict(mets.get("per_class") or {})
        for class_id, class_metrics in per_class.items():
            rows_summary.append(
                {
                    "model_type": model_type,
                    "class_id": int(class_id),
                    "mean_f1": float(class_metrics.get("mean_f1") or 0.0),
                    "mean_lift_precision": float(class_metrics.get("mean_lift_precision") or 0.0),
                    "count": int(class_metrics.get("count") or 0),
                }
            )

        top_by_class = _top_regions_by_class(valuable, top_n=5)
        for class_id, regions in sorted(top_by_class.items()):
            print(f"  Clase {class_id}")
            for rank, region in enumerate(regions, 1):
                metrics = region.get("metrics") or {}
                rule_expr = _rule_to_boolean_expression(region)
                print(
                    f"    #{rank} id={region.get('region_id')} | "
                    f"F1={float(metrics.get('f1', 0.0)):.3f} | "
                    f"Lift={float(metrics.get('lift_precision', 0.0)):.3f}"
                )
                print(f"       Regla: {rule_expr}")
                rows_rules.append(
                    {
                        "model_type": model_type,
                        "class_id": int(class_id),
                        "rank": rank,
                        "region_id": region.get("region_id"),
                        "f1": float(metrics.get("f1", 0.0)),
                        "precision": float(metrics.get("precision", 0.0)),
                        "recall": float(metrics.get("recall", 0.0)),
                        "lift_precision": float(metrics.get("lift_precision", 0.0)),
                        "rule_expression": rule_expr,
                    }
                )

    summary_df = pd.DataFrame(rows_summary)
    rules_df = pd.DataFrame(rows_rules)

    out_dir = ROOT / "experiments_outputs" / "model_type_regions_dashboard"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    rules_df.to_csv(out_dir / "top_rules_by_class.csv", index=False)

    global_df = summary_df.dropna(subset=["global_mean_f1"]).drop_duplicates(subset=["model_type"])
    per_class_df = summary_df.dropna(subset=["class_id"]).copy()

    fig_global = px.bar(
        global_df.sort_values("global_mean_f1", ascending=False),
        x="model_type",
        y="global_mean_f1",
        color="global_mean_lift_precision",
        title="Ranking global por model_type (Top-9 por clase)",
        labels={"global_mean_f1": "F1 medio global", "global_mean_lift_precision": "Lift medio global"},
    )
    fig_global.write_html(out_dir / "global_ranking.html")

    fig_per_class = px.bar(
        per_class_df.sort_values(["class_id", "mean_f1"], ascending=[True, False]),
        x="model_type",
        y="mean_f1",
        color="mean_lift_precision",
        facet_row="class_id",
        title="Comparativa por clase: F1 medio vs lift medio",
        labels={"mean_f1": "F1 medio", "mean_lift_precision": "Lift medio"},
    )
    fig_per_class.write_html(out_dir / "per_class_ranking.html")

    fig_rules = px.scatter(
        rules_df,
        x="f1",
        y="lift_precision",
        color="model_type",
        facet_col="class_id",
        hover_data=["region_id", "rank", "rule_expression"],
        title="Top reglas por clase: F1 vs Lift",
    )
    fig_rules.write_html(out_dir / "rules_f1_vs_lift.html")

    print("\nArchivos generados:")
    print(f"- {out_dir / 'summary_metrics.csv'}")
    print(f"- {out_dir / 'top_rules_by_class.csv'}")
    print(f"- {out_dir / 'global_ranking.html'}")
    print(f"- {out_dir / 'per_class_ranking.html'}")
    print(f"- {out_dir / 'rules_f1_vs_lift.html'}")

    return summary_df, rules_df


if __name__ == "__main__":
    run_dashboard()
