"""Business impact helpers for experiment ROI projections."""

from __future__ import annotations


def calculate_roi(
    baseline_rate,
    lift_percent,
    daily_traffic,
    revenue_per_conversion,
    experiment_days,
):
    """
    Estimate incremental conversions and revenue from an experiment lift.

    `lift_percent` is interpreted as a relative lift, e.g. 10 -> +10%.
    """
    _validate_roi_inputs(
        baseline_rate=baseline_rate,
        lift_percent=lift_percent,
        daily_traffic=daily_traffic,
        revenue_per_conversion=revenue_per_conversion,
        experiment_days=experiment_days,
    )

    baseline_conversions = float(baseline_rate) * float(daily_traffic) * float(experiment_days)
    incremental_conversions = baseline_conversions * (float(lift_percent) / 100.0)
    incremental_revenue = incremental_conversions * float(revenue_per_conversion)
    annualized_revenue = 0.0 if float(experiment_days) == 0 else incremental_revenue * (365.0 / float(experiment_days))

    return {
        "incremental_conversions": float(incremental_conversions),
        "incremental_revenue": float(incremental_revenue),
        "annualized_revenue": float(annualized_revenue),
        "experiment_cost_note": (
            "This is gross upside only. Subtract engineering time, tooling, discounting, and opportunity cost separately."
        ),
    }


def project_revenue_series(
    baseline_rate,
    lift_percent,
    daily_traffic,
    revenue_per_conversion,
    experiment_days,
):
    """Return baseline and projected revenue for the experiment window."""
    _validate_roi_inputs(
        baseline_rate=baseline_rate,
        lift_percent=lift_percent,
        daily_traffic=daily_traffic,
        revenue_per_conversion=revenue_per_conversion,
        experiment_days=experiment_days,
    )

    experiment_days = float(experiment_days)
    baseline_revenue = float(baseline_rate) * float(daily_traffic) * float(revenue_per_conversion) * experiment_days
    projected_revenue = baseline_revenue * (1 + float(lift_percent) / 100.0)
    return {
        "baseline_revenue": float(baseline_revenue),
        "projected_revenue": float(projected_revenue),
    }


def _validate_roi_inputs(
    baseline_rate,
    lift_percent,
    daily_traffic,
    revenue_per_conversion,
    experiment_days,
):
    if not 0 <= float(baseline_rate) <= 1:
        raise ValueError("baseline_rate must be between 0 and 1.")
    if float(lift_percent) <= -100:
        raise ValueError("lift_percent must be greater than -100.")
    if float(daily_traffic) < 0:
        raise ValueError("daily_traffic must be non-negative.")
    if float(revenue_per_conversion) < 0:
        raise ValueError("revenue_per_conversion must be non-negative.")
    if float(experiment_days) <= 0:
        raise ValueError("experiment_days must be positive.")
