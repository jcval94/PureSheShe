from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import model_type_operator_audit as audit  # noqa: E402


def test_model_types_report_expected_operators() -> None:
    results = audit.run_audit()
    by_mode = {row.mode: row for row in results}

    assert len(results) == len(audit.MODEL_TYPES)
    assert all(row.status == "PASS" for row in results)

    mixed_modes = ["dnf", "and_or_beam", "and_or_random", "and_or_diverse", "and_or_greedy"]
    for mode in mixed_modes:
        assert by_mode[mode].report_has_or
        assert by_mode[mode].report_has_and
        assert by_mode[mode].report_has_both_in_one_rule
