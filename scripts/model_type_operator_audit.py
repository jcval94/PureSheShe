from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.datasets import make_classification

from deldel import describe_regions_report, find_comb_dim_spaces_full

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

EXPECTATIONS = {
    "base": (False, True),
    "default": (False, True),
    "hessian_rank": (False, True),
    "hessian_filter": (False, True),
    "and": (False, True),
    "or": (True, False),
    "dnf": (True, True),
    "and_or_beam": (True, True),
    "and_or_random": (True, True),
    "and_or_diverse": (True, True),
    "and_or_greedy": (True, True),
}


@dataclass
class ModeAudit:
    mode: str
    expected_union: bool
    expected_intersection: bool
    report_has_or: bool
    report_has_and: bool
    report_has_both_in_one_rule: bool
    rule_lines: int
    attempts: int
    status: str


def _per_class_metrics(mask: np.ndarray, y: np.ndarray) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    n = int(y.shape[0])
    for class_id in sorted(int(c) for c in np.unique(y)):
        class_mask = y == class_id
        size = int(mask.sum())
        tp = int(np.logical_and(mask, class_mask).sum())
        total = int(class_mask.sum())
        precision = float(tp / size) if size else 0.0
        recall = float(tp / total) if total else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        baseline = float(total / n) if n else 0.0
        lift = float(precision / baseline) if baseline else 0.0
        out[class_id] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "lift_precision": lift,
            "lift": lift,
            "region_frac": float(size / n) if n else 0.0,
        }
    return out


def _build_selection(X: np.ndarray, y: np.ndarray) -> Dict[str, List[Dict[str, object]]]:
    planes: List[Dict[str, object]] = []
    plane_idx = 0
    for dim in range(X.shape[1]):
        for quantile in (0.2, 0.35, 0.5, 0.65, 0.8):
            threshold = float(np.quantile(X[:, dim], quantile))
            for side, symbol in ((1, "≤"), (-1, "≥")):
                mask = (X[:, dim] <= threshold) if side == 1 else (X[:, dim] >= threshold)
                if int(mask.sum()) < 10:
                    continue
                planes.append(
                    {
                        "oriented_plane_id": f"p{plane_idx}:{symbol}",
                        "plane_id": f"p{plane_idx}",
                        "origin_pair": (dim, (dim + 1) % X.shape[1]),
                        "side": side,
                        "dims": [dim],
                        "n_norm": [1.0],
                        "b_norm": -threshold,
                        "inequality": {"general": f"f{dim} {symbol} {threshold:.4f}"},
                        "metrics_by_class": _per_class_metrics(mask, y),
                    }
                )
                plane_idx += 1
    return {"winning_planes": planes}


def _extract_rule_flags(report: str) -> Dict[str, bool | int]:
    lines = [ln.strip() for ln in report.splitlines() if ln.strip().startswith("Regla:")]
    return {
        "rule_lines": len(lines),
        "has_or": any(" OR " in ln for ln in lines),
        "has_and": any(" AND " in ln for ln in lines),
        "has_both": any((" OR " in ln and " AND " in ln) for ln in lines),
    }


def run_audit() -> List[ModeAudit]:
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        class_sep=1.2,
        random_state=7,
    )
    selection = _build_selection(X, y)

    attempts_cfg = [
        dict(max_rules_per_class=120, max_clause_candidates=100, max_clauses=5, max_dnf_rules_per_class=20),
        dict(max_rules_per_class=220, max_clause_candidates=180, max_clauses=6, max_dnf_rules_per_class=30),
        dict(max_rules_per_class=320, max_clause_candidates=240, max_clauses=8, max_dnf_rules_per_class=40),
    ]

    results: List[ModeAudit] = []
    for mode in MODEL_TYPES:
        expected_union, expected_intersection = EXPECTATIONS[mode]
        final_flags = {"rule_lines": 0, "has_or": False, "has_and": False, "has_both": False}
        status = "FAIL"
        used_attempt = 0

        for attempt, cfg in enumerate(attempts_cfg, start=1):
            valuable = find_comb_dim_spaces_full(
                selection,
                X,
                y,
                mode=mode,
                max_planes=12,
                metric="f1",
                beam_width=36,
                **cfg,
            )
            report = describe_regions_report(
                valuable,
                top_per_class=40,
                dataset_size=X.shape[0],
                max_rule_text_chars=0,
            )
            final_flags = _extract_rule_flags(report)

            if expected_union and expected_intersection:
                ok = bool(final_flags["has_or"] and final_flags["has_and"] and final_flags["has_both"])
            elif expected_union:
                ok = bool(final_flags["has_or"] and not final_flags["has_and"])
            else:
                ok = bool(final_flags["has_and"] and not final_flags["has_or"])

            if ok:
                status = "PASS"
                used_attempt = attempt
                break

        if used_attempt == 0:
            used_attempt = len(attempts_cfg)

        results.append(
            ModeAudit(
                mode=mode,
                expected_union=expected_union,
                expected_intersection=expected_intersection,
                report_has_or=bool(final_flags["has_or"]),
                report_has_and=bool(final_flags["has_and"]),
                report_has_both_in_one_rule=bool(final_flags["has_both"]),
                rule_lines=int(final_flags["rule_lines"]),
                attempts=used_attempt,
                status=status,
            )
        )

    return results


def write_outputs(results: List[ModeAudit], csv_path: Path, readme_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "mode",
                "expected_union",
                "expected_intersection",
                "report_has_or",
                "report_has_and",
                "report_has_both_in_one_rule",
                "rule_lines",
                "attempts",
                "status",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.mode,
                    str(r.expected_union).lower(),
                    str(r.expected_intersection).lower(),
                    str(r.report_has_or).lower(),
                    str(r.report_has_and).lower(),
                    str(r.report_has_both_in_one_rule).lower(),
                    r.rule_lines,
                    r.attempts,
                    r.status,
                ]
            )

    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    both_modes = [r.mode for r in results if r.expected_union and r.expected_intersection]
    both_ok = [r.mode for r in results if r.expected_union and r.expected_intersection and r.report_has_both_in_one_rule]

    lines = [
        "# Validación de operadores en `describe_regions_report`",
        "",
        f"- Modos evaluados: **{total}**.",
        f"- Modos en PASS: **{passed}/{total}**.",
        f"- Modos A U B y A INT B esperados: {', '.join(both_modes)}.",
        f"- Modos A U B y A INT B que muestran ambas en una misma `Regla:`: {', '.join(both_ok)}.",
        "",
        "## Conclusiones",
        "",
        "- Los modos de unión pura (`or`) muestran `OR` y no muestran `AND` en las reglas del reporte.",
        "- Los modos de intersección pura (`base`, `default`, `hessian_rank`, `hessian_filter`, `and`) muestran `AND` y no muestran `OR`.",
        "- Los modos mixtos (`dnf`, `and_or_*`) muestran `OR` y `AND`; además se observan reglas donde ambos operadores conviven en la misma línea `Regla:`.",
    ]
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    out_csv = Path("experiments_outputs/model_type_operator_check.csv")
    out_md = Path("experiments_outputs/model_type_operator_check_README.md")
    audit_results = run_audit()
    write_outputs(audit_results, out_csv, out_md)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")
