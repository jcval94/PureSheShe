from deldel.reporting_plotting import describe_regions_report


def test_describe_regions_report_returns_details_sorted_by_lift():
    valuable = {
        2: [
            {
                "region_id": "r0a",
                "target_class": 0,
                "is_pareto": False,
                "metrics": {"f1": 0.7, "lift_precision": 1.1},
            },
            {
                "region_id": "r0b",
                "target_class": 0,
                "is_pareto": False,
                "metrics": {"f1": 0.6, "lift_precision": 2.0},
            },
            {
                "region_id": "r1a",
                "target_class": 1,
                "is_pareto": False,
                "metrics": {"f1": 0.8, "lift_precision": 0.9},
            },
            {
                "region_id": "r1b",
                "target_class": 1,
                "is_pareto": False,
                "metrics": {"f1": 0.75, "lift_precision": 3.0},
            },
        ]
    }

    result = describe_regions_report(valuable, top_per_class=1, return_average_metrics=True)

    assert result["per_class"][0]["count"] == 1
    assert result["per_class"][1]["count"] == 1

    details = result["details"]
    assert [item["region_id"] for item in details] == ["r0a", "r1a"]
    lifts = [item["metrics"]["lift_precision"] for item in details]
    assert lifts == sorted(lifts, reverse=True)


def test_describe_regions_report_details_respects_top_per_class():
    valuable = {
        2: [
            {
                "region_id": "r0a",
                "target_class": 0,
                "is_pareto": False,
                "metrics": {"f1": 0.7, "lift_precision": 1.1},
            },
            {
                "region_id": "r0b",
                "target_class": 0,
                "is_pareto": False,
                "metrics": {"f1": 0.6, "lift_precision": 2.0},
            },
            {
                "region_id": "r1a",
                "target_class": 1,
                "is_pareto": False,
                "metrics": {"f1": 0.8, "lift_precision": 0.9},
            },
            {
                "region_id": "r1b",
                "target_class": 1,
                "is_pareto": False,
                "metrics": {"f1": 0.75, "lift_precision": 3.0},
            },
        ]
    }

    result = describe_regions_report(valuable, top_per_class=2, return_average_metrics=True)

    assert result["per_class"][0]["count"] == 2
    assert result["per_class"][1]["count"] == 2
    assert len(result["details"]) == 4
