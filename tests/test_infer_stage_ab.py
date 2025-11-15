import pytest

from src.deldel import engine, stage_infer_ab


def test_infer_stage_matches_previous_behavior():
    rows = stage_infer_ab.collect_stage_results()
    assert rows, "Expected at least one scenario to be validated"
    dataframe_rows = [row for row in rows if row.get("dataframe_label")]
    assert dataframe_rows, "Expected DataFrame-based scenarios to be included"
    for row in rows:
        assert row["values_match"] == "yes"
        assert row["legacy_value"] == row["current_value"]
    for row in dataframe_rows:
        assert row["cache_expected"] == "yes"
        assert row["dataframe_rows"] >= 1
        assert row["dataframe_cols"] >= 1


def test_dataframe_scenarios_emit_deldel_stage():
    df_scenarios = [
        scenario for scenario in stage_infer_ab.SCENARIOS if scenario.metadata.get("dataframe_label")
    ]
    assert df_scenarios, "Expected dedicated DataFrame scenarios"
    for scenario in df_scenarios:
        stage = scenario.run(stage_infer_ab.infer_stage_legacy)
        assert stage.startswith("DelDel."), stage


def test_infer_stage_does_not_require_inspect_stack(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("inspect.stack should not be used")

    monkeypatch.setattr(engine.inspect, "stack", boom)
    assert engine._infer_stage() != "unknown"


@pytest.mark.parametrize("iterations", [500])
def test_infer_stage_ab_performance(iterations):
    timings = stage_infer_ab.benchmark_stage_variants(iterations=iterations, repeat=2)
    by_run = {}
    for row in timings:
        run_index = int(row["run_index"])
        by_run.setdefault(run_index, {})[row["variant"]] = row
    assert len(by_run) == 2
    for run_data in by_run.values():
        assert "legacy" in run_data and "current" in run_data
        legacy = run_data["legacy"]
        current = run_data["current"]
        assert current["total_runtime_s"] <= legacy["total_runtime_s"] * 1.10
        assert run_data["ratio_current_over_legacy"]["total_runtime_s"] <= 0.5


def test_infer_stage_ab_csv_writer(tmp_path):
    results_path = tmp_path / "results.csv"
    timings_path = tmp_path / "timings.csv"

    summary = stage_infer_ab.write_infer_stage_ab_csvs(
        iterations=[50, 100],
        repeat=2,
        results_path=str(results_path),
        timings_path=str(timings_path),
    )

    assert results_path.exists()
    assert timings_path.exists()

    rows = results_path.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    for col in ["scenario", "legacy_value", "current_value", "values_match"]:
        assert col in header
    assert all("yes" in line for line in rows[1:])
    assert "dataframe_label" in header

    timing_rows = timings_path.read_text(encoding="utf-8").strip().splitlines()
    timing_header = timing_rows[0].split(",")
    for col in ["variant", "iterations", "run_index"]:
        assert col in timing_header
    assert any("legacy" in line for line in timing_rows[1:])
    assert any("ratio_current_over_legacy" in line for line in timing_rows[1:])

    assert len(list(summary["timings"])) == len(timing_rows) - 1
