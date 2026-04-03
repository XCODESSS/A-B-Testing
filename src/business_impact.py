"""Business-impact summary helpers for the Streamlit app."""

from __future__ import annotations

from datetime import date


def recommend_experiment_action(significant: bool, lift_percent: float, incremental_revenue: float) -> str:
    """Map experiment outcome to a simple executive recommendation."""
    if significant and lift_percent > 0 and incremental_revenue > 0:
        return "ship"
    if significant and (lift_percent <= 0 or incremental_revenue <= 0):
        return "don't ship"
    return "run longer"


def generate_executive_report(
    test_name: str,
    start_date: date,
    end_date: date,
    sample_size: int,
    significant: bool,
    p_value: float,
    confidence_interval: tuple[float, float] | None,
    incremental_revenue: float,
    lift_percent: float,
    recommendation: str,
) -> str:
    """Create a copy-pasteable markdown summary for stakeholders."""
    statistical_result = "Significant" if significant else "Not significant"
    ci_text = _format_confidence_interval(confidence_interval)
    sample_size = int(sample_size)

    return "\n".join(
        [
            "## Executive Experiment Summary",
            "",
            f"**Test name:** {test_name.strip() or 'Untitled experiment'}",
            f"**Dates:** {start_date.isoformat()} to {end_date.isoformat()}",
            f"**Sample size:** {sample_size:,}",
            "",
            "### Statistical Result",
            f"- Result: {statistical_result}",
            f"- P-value: {float(p_value):.4f}",
            f"- Confidence interval: {ci_text}",
            "",
            "### Business Impact",
            f"- Estimated lift: {float(lift_percent):.2f}%",
            f"- Incremental revenue: ${float(incremental_revenue):,.2f}",
            "",
            "### Recommendation",
            f"- {recommendation}",
        ]
    )


def _format_confidence_interval(confidence_interval: tuple[float, float] | None) -> str:
    if confidence_interval is None:
        return "Not provided"

    lower, upper = confidence_interval
    return f"[{float(lower):.4f}, {float(upper):.4f}]"
