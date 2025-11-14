"""Utilities for summarizing method set performance labels."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import List, Sequence

INPUT_PATH = Path("subspaces/outputs/method_sets/method_sets.csv")
OUTPUT_PATH = Path("subspaces/outputs/method_sets/method_sets_analysis.md")


def _normalize(value: str) -> str:
    return value.strip().lower()


def _count_si(values: Sequence[str]) -> int:
    return sum(1 for value in values if _normalize(value) == "si")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Method set data not found: {INPUT_PATH}")

    with INPUT_PATH.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        if len(header) < 3:
            raise ValueError("Unexpected header in method set CSV")

        method_names: List[str] = header[1:-1]
        rows = []
        total_method_counts: Counter[str] = Counter()

        for raw_row in reader:
            if not raw_row:
                continue
            set_name = raw_row[0]
            method_values = raw_row[1:-1]
            score_global = raw_row[-1]
            si_count = _count_si(method_values)
            rows.append(
                {
                    "set_name": set_name,
                    "values": method_values,
                    "score": score_global,
                    "si_count": si_count,
                }
            )
            for method_name, value in zip(method_names, method_values):
                if _normalize(value) == "si":
                    total_method_counts[method_name] += 1

    if not rows:
        raise ValueError("No method set rows were found")

    rows.sort(key=lambda item: (-item["si_count"], item["set_name"]))
    top_sets = rows[:50]

    top_method_counts: Counter[str] = Counter()
    for row in top_sets:
        for method_name, value in zip(method_names, row["values"]):
            if _normalize(value) == "si":
                top_method_counts[method_name] += 1

    method_counts_in_top = {
        method_name: top_method_counts.get(method_name, 0)
        for method_name in method_names
    }

    overall_total_si = sum(total_method_counts.values())

    lines = ["# Method Sets Analysis", ""]

    lines.append("## Top 50 sets by 'Si' count")
    lines.append("| Rank | Conjunto | 'Si' count | Score Global |")
    lines.append("| --- | --- | --- | --- |")
    for idx, row in enumerate(top_sets, start=1):
        lines.append(
            f"| {idx} | {row['set_name']} | {row['si_count']} | {row['score']} |"
        )
    lines.append("")

    def _format_method_counts(title: str, items: Sequence[tuple[str, int]]) -> None:
        lines.append(title)
        for method_name, count in items:
            lines.append(f"- {method_name}: {count}")
        lines.append("")

    sorted_top_methods = sorted(
        method_counts_in_top.items(), key=lambda item: (-item[1], item[0])
    )
    top_five_methods = sorted_top_methods[:5]
    bottom_five_methods = sorted(
        method_counts_in_top.items(), key=lambda item: (item[1], item[0])
    )[:5]

    lines.append("## Method performance inside top 50 sets")
    _format_method_counts("Top 5 methods:", top_five_methods)
    _format_method_counts("Bottom 5 methods:", bottom_five_methods)

    lines.append("## Total 'Si' counts across all sets")
    lines.append("| Method | 'Si' count |")
    lines.append("| --- | --- |")
    for method_name in method_names:
        lines.append(
            f"| {method_name} | {total_method_counts.get(method_name, 0)} |"
        )
    lines.append(f"| **Total** | **{overall_total_si}** |")
    lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
