import math
import pathlib
import sys
from datetime import date

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.business_impact import generate_executive_report, recommend_experiment_action
from src.roi_calculator import calculate_roi, project_revenue_series


def test_calculate_roi_returns_incremental_and_annualized_revenue():
    result = calculate_roi(
        baseline_rate=0.10,
        lift_percent=15,
        daily_traffic=1_000,
        revenue_per_conversion=50,
        experiment_days=14,
    )

    assert math.isclose(result["incremental_conversions"], 210.0, rel_tol=1e-9)
    assert math.isclose(result["incremental_revenue"], 10_500.0, rel_tol=1e-9)
    assert math.isclose(result["annualized_revenue"], 273_750.0, rel_tol=1e-9)
    assert "gross upside" in result["experiment_cost_note"]


def test_project_revenue_series_returns_baseline_and_projected_values():
    result = project_revenue_series(
        baseline_rate=0.20,
        lift_percent=10,
        daily_traffic=500,
        revenue_per_conversion=25,
        experiment_days=10,
    )

    assert math.isclose(result["baseline_revenue"], 25_000.0, rel_tol=1e-9)
    assert math.isclose(result["projected_revenue"], 27_500.0, rel_tol=1e-9)


def test_recommendation_and_report_are_copy_paste_ready():
    recommendation = recommend_experiment_action(
        significant=True,
        lift_percent=12.5,
        incremental_revenue=8_000.0,
    )
    report = generate_executive_report(
        test_name="Checkout CTA Refresh",
        start_date=date(2026, 3, 1),
        end_date=date(2026, 3, 14),
        sample_size=24_000,
        significant=True,
        p_value=0.0314,
        confidence_interval=(0.0101, 0.0420),
        incremental_revenue=8_000.0,
        lift_percent=12.5,
        recommendation=recommendation,
    )

    assert recommendation == "ship"
    assert "Checkout CTA Refresh" in report
    assert "2026-03-01 to 2026-03-14" in report
    assert "P-value: 0.0314" in report
    assert "Incremental revenue: $8,000.00" in report
    assert "- ship" in report


def test_roi_validation_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        calculate_roi(1.1, 10, 100, 25, 14)

    with pytest.raises(ValueError):
        project_revenue_series(0.1, -100, 100, 25, 14)

    with pytest.raises(ValueError):
        calculate_roi(0.1, 10, 100, 25, 0)
