"""Benchmark bottleneck proposals for the low-dimensional finder.

The script compares the original implementation stored in ``HEAD`` with the
current working tree and a few configuration proposals.  It writes:

* ``experiments_outputs/process_bottleneck_ab_results.csv``
* ``experiments_outputs/README_process_bottlenecks.md``
"""

from __future__ import annotations

import copy
import csv
import io
import statistics
import subprocess
import sys
import types
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deldel import describe_regions_metrics, find_low_dim_spaces, make_corner_class_dataset  # noqa: E402
from deldel.experiments import _build_demo_selection  # noqa: E402
from deldel.find_low_dim_spaces_fast import (  # noqa: E402
    find_low_dim_spaces_deterministic,
    find_low_dim_spaces_support_first,
)


CSV_PATH = ROOT / "experiments_outputs" / "process_bottleneck_ab_results.csv"
README_PATH = ROOT / "experiments_outputs" / "README_process_bottlenecks.md"


VariantFunc = Callable[..., Dict[int, List[Dict[str, Any]]]]


def _load_head_find_low_dim_spaces() -> VariantFunc:
    source = subprocess.check_output(
        ["git", "show", "HEAD:src/deldel/find_low_dim_spaces_fast.py"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    module = types.ModuleType("deldel.find_low_dim_spaces_fast_head")
    module.__package__ = "deldel"
    exec(compile(source, "deldel.find_low_dim_spaces_fast_head", "exec"), module.__dict__)
    return module.find_low_dim_spaces


def _flatten(valuable: Dict[int, List[Dict[str, Any]]]) -> Iterable[Dict[str, Any]]:
    for dim in sorted(valuable):
        for row in valuable.get(dim, []) or []:
            yield row


def _top_signatures(valuable: Dict[int, List[Dict[str, Any]]], per_class: int = 3) -> set[Tuple[Any, ...]]:
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for row in _flatten(valuable):
        buckets.setdefault(int(row.get("target_class")), []).append(row)
    signatures = set()
    for cls, rows in buckets.items():
        rows = sorted(
            rows,
            key=lambda r: (
                float(r.get("metrics", {}).get("f1", 0.0)),
                float(r.get("metrics", {}).get("lift_precision", 0.0)),
                float(r.get("metrics", {}).get("size", 0.0)),
            ),
            reverse=True,
        )[:per_class]
        for row in rows:
            signatures.add(
                (
                    cls,
                    tuple(row.get("dims", ())),
                    str(row.get("rule_text", "")).replace(" ", ""),
                )
            )
    return signatures


def _summarise(valuable: Dict[int, List[Dict[str, Any]]], dataset_size: int) -> Dict[str, Any]:
    metrics = describe_regions_metrics(valuable, top_per_class=3, dataset_size=dataset_size)
    total_regions = sum(len(rows or []) for rows in valuable.values())
    if metrics:
        mean_f1 = statistics.mean(float(row.get("f1", 0.0)) for row in metrics)
        mean_lift = statistics.mean(float(row.get("lift_precision", 0.0)) for row in metrics)
    else:
        mean_f1 = 0.0
        mean_lift = 0.0
    return {
        "total_regions": int(total_regions),
        "top3_rows": int(len(metrics)),
        "mean_top3_f1": float(mean_f1),
        "mean_top3_lift": float(mean_lift),
        "signatures": _top_signatures(valuable),
    }


def _timed_variant(
    name: str,
    func: VariantFunc,
    X,
    y,
    selection: Dict[str, Any],
    base_kwargs: Dict[str, Any],
    overrides: Dict[str, Any],
    repetitions: int,
) -> Tuple[List[float], Dict[str, Any]]:
    times: List[float] = []
    last_value: Dict[int, List[Dict[str, Any]]] = {}
    kwargs = dict(base_kwargs)
    kwargs.update(overrides)
    for _ in range(repetitions):
        payload = copy.deepcopy(selection)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            start = perf_counter()
            last_value = func(X, y, payload, **kwargs)
            times.append(perf_counter() - start)
    summary = _summarise(last_value, dataset_size=int(y.shape[0]))
    return times, summary


def _write_readme(rows: List[Dict[str, Any]]) -> None:
    best = max(
        (row for row in rows if row["implemented"] == "yes"),
        key=lambda row: (float(row["speedup_vs_original"]), float(row["output_similarity_top3"])),
    )
    lines = [
        "# A/B test de cuellos de botella",
        "",
        "Dataset: corner sintetico con 12,800 filas y 4 features.",
        "Metrica: F1 y lift de precision promedio Top-3 por clase con `describe_regions_metrics`.",
        "Objetivo: mantener la salida visible parecida a la original y reducir tiempo.",
        "",
        "## Cuellos de botella encontrados",
        "",
        "1. El armado de texto de reglas repetia trabajo geometrico por cada combinacion candidata.",
        "2. El fallback de cobertura por clase guardaba todos los candidatos rechazados aunque solo usa los mejores pocos por clase/dimension.",
        "3. Se materializaban mascaras booleanas internas para relaciones/uniones y luego se borraban de la salida publica.",
        "4. El cap global por clase muestreaba candidatos al azar despues del ranking por prioridad, con riesgo de perder planos buenos.",
        "",
        "## Resultado",
        "",
        f"Mejor fila implementada: `{best['variant']}` con speedup {float(best['speedup_vs_original']):.2f}x, "
        f"delta F1 Top-3 {float(best['delta_mean_top3_f1']):+.6f}, "
        f"delta lift Top-3 {float(best['delta_mean_top3_lift']):+.6f}, "
        f"y similitud de reglas top {float(best['output_similarity_top3']):.2f}.",
        "",
        "## CSV",
        "",
        f"Detalle completo: `{CSV_PATH.relative_to(ROOT).as_posix()}`.",
        "",
        "| Variante | Implementado | Speedup | Similitud | Delta F1 | Delta lift | Decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['implemented']} | "
            f"{float(row['speedup_vs_original']):.2f}x | "
            f"{float(row['output_similarity_top3']):.2f} | "
            f"{float(row['delta_mean_top3_f1']):+.6f} | "
            f"{float(row['delta_mean_top3_lift']):+.6f} | "
            f"{row['decision']} |"
        )
    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    X, y, feature_names = make_corner_class_dataset(
        n_per_cluster=1600,
        std_class1=0.45,
        std_other=0.85,
        a=2.8,
        random_state=0,
    )
    selection = _build_demo_selection(X, y, feature_names)
    base_kwargs = dict(
        feature_names=list(feature_names),
        max_planes_in_rule=3,
        max_planes_per_pair=4,
        min_support=40,
        min_rel_gain_f1=0.05,
        min_lift_prec=1.40,
        consider_dims_up_to=X.shape[1],
        rng_seed=0,
        verbosity=0,
    )

    original = _load_head_find_low_dim_spaces()
    variants: List[Dict[str, Any]] = [
        {
            "variant": "original_head",
            "proposal": "Original algorithm",
            "bottleneck_addressed": "baseline",
            "func": original,
            "overrides": {},
            "implemented": "baseline",
            "decision": "reference",
        },
        {
            "variant": "implemented_internal_fast_path",
            "proposal": "Precompute rule pieces, keep compact masks, top-only floor, priority cap",
            "bottleneck_addressed": "repeated rule construction; excess floor records; mask materialization; random cap",
            "func": find_low_dim_spaces,
            "overrides": {},
            "implemented": "yes",
            "decision": "keep",
        },
        {
            "variant": "config_no_relations",
            "proposal": "Skip relation/family graph after ranking",
            "bottleneck_addressed": "pairwise subset checks for final regions",
            "func": find_low_dim_spaces,
            "overrides": {"compute_relations": False},
            "implemented": "no",
            "decision": "do not default; metadata loss",
        },
        {
            "variant": "config_no_unions",
            "proposal": "Skip OR-union expansion",
            "bottleneck_addressed": "union pair evaluation",
            "func": find_low_dim_spaces,
            "overrides": {"enable_unions": False},
            "implemented": "no",
            "decision": "do not default; output less complete",
        },
        {
            "variant": "config_no_unions_no_relations",
            "proposal": "Minimal post-processing",
            "bottleneck_addressed": "union pair evaluation and relation graph",
            "func": find_low_dim_spaces,
            "overrides": {"enable_unions": False, "compute_relations": False},
            "implemented": "no",
            "decision": "fast option only; too much metadata loss",
        },
        {
            "variant": "support_first_existing",
            "proposal": "Favor high-support two-plane rules",
            "bottleneck_addressed": "combination explosion",
            "func": find_low_dim_spaces_support_first,
            "overrides": {},
            "implemented": "already available",
            "decision": "keep as opt-in preset",
        },
        {
            "variant": "deterministic_existing",
            "proposal": "Disable unions and relation graph for deterministic compact output",
            "bottleneck_addressed": "post-processing cost and output size",
            "func": find_low_dim_spaces_deterministic,
            "overrides": {},
            "implemented": "already available",
            "decision": "keep as opt-in preset",
        },
    ]

    repetitions = 5
    rows: List[Dict[str, Any]] = []
    baseline_summary: Dict[str, Any] | None = None
    baseline_mean = 0.0
    baseline_signatures: set[Tuple[Any, ...]] = set()

    for variant in variants:
        times, summary = _timed_variant(
            variant["variant"],
            variant["func"],
            X,
            y,
            selection,
            base_kwargs,
            variant["overrides"],
            repetitions,
        )
        mean_runtime = statistics.mean(times)
        median_runtime = statistics.median(times)

        if variant["variant"] == "original_head":
            baseline_summary = summary
            baseline_mean = mean_runtime
            baseline_signatures = set(summary["signatures"])

        assert baseline_summary is not None
        signatures = set(summary["signatures"])
        union = baseline_signatures | signatures
        inter = baseline_signatures & signatures
        similarity = (len(inter) / len(union)) if union else 1.0

        row = {
            "variant": variant["variant"],
            "proposal": variant["proposal"],
            "bottleneck_addressed": variant["bottleneck_addressed"],
            "implemented": variant["implemented"],
            "decision": variant["decision"],
            "repetitions": repetitions,
            "mean_runtime_s": f"{mean_runtime:.6f}",
            "median_runtime_s": f"{median_runtime:.6f}",
            "speedup_vs_original": f"{(baseline_mean / mean_runtime) if mean_runtime else 0.0:.6f}",
            "total_regions": summary["total_regions"],
            "top3_rows": summary["top3_rows"],
            "mean_top3_f1": f"{summary['mean_top3_f1']:.6f}",
            "mean_top3_lift": f"{summary['mean_top3_lift']:.6f}",
            "delta_mean_top3_f1": f"{summary['mean_top3_f1'] - baseline_summary['mean_top3_f1']:.6f}",
            "delta_mean_top3_lift": f"{summary['mean_top3_lift'] - baseline_summary['mean_top3_lift']:.6f}",
            "output_similarity_top3": f"{similarity:.6f}",
        }
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _write_readme(rows)
    print(f"Wrote {CSV_PATH.relative_to(ROOT)}")
    print(f"Wrote {README_PATH.relative_to(ROOT)}")
    for row in rows:
        print(
            f"{row['variant']}: speedup={row['speedup_vs_original']} "
            f"similarity={row['output_similarity_top3']} "
            f"f1_delta={row['delta_mean_top3_f1']}"
        )


if __name__ == "__main__":
    main()
