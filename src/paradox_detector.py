"""Simpson's paradox detection utilities."""

from __future__ import annotations

from collections import Counter
from numbers import Number

import numpy as np
import pandas as pd

CONTROL_ALIASES = {"0", "a", "baseline", "control", "false"}
TREATMENT_ALIASES = {"1", "b", "test", "treatment", "true", "variant"}
FAILURE_ALIASES = {"0", "fail", "failure", "false", "n", "no", "not_converted"}
SUCCESS_ALIASES = {"1", "converted", "success", "true", "win", "y", "yes"}


def detect_simpsons_paradox(df, treatment_col, outcome_col, confounder_col):
    """
    Detect whether the aggregate winner conflicts with subgroup winners.

    Returns:
        dict containing aggregate results, subgroup results, and an explanation.
    """
    _validate_inputs(df, treatment_col, outcome_col, confounder_col)

    working_df = pd.DataFrame(
        {
            "treatment": df[treatment_col].map(_normalize_value),
            "success": _coerce_success_indicator(df[outcome_col]),
            "confounder": df[confounder_col],
        }
    ).dropna(subset=["treatment", "success", "confounder"])

    control_label, treatment_label = _identify_variant_labels(working_df["treatment"])
    aggregate_result = _summarize_slice(working_df, control_label, treatment_label)
    aggregate_result["winner"] = _winner_name(aggregate_result["winner"])

    subgroup_results = []
    subgroup_winners = []
    for group_value in pd.unique(working_df["confounder"]):
        subgroup_df = working_df.loc[working_df["confounder"] == group_value]
        subgroup_summary = _summarize_slice(subgroup_df, control_label, treatment_label)
        subgroup_winner = _winner_name(subgroup_summary["winner"])
        subgroup_results.append(
            {
                "group": group_value,
                "treatment_rate": subgroup_summary["treatment_rate"],
                "control_rate": subgroup_summary["control_rate"],
                "winner": subgroup_winner,
            }
        )
        if subgroup_winner in {"treatment", "control"}:
            subgroup_winners.append(subgroup_winner)

    winner_counts = Counter(subgroup_winners)
    majority_winner = _majority_winner(winner_counts)
    paradox_detected = (
        majority_winner is not None
        and aggregate_result["winner"] in {"treatment", "control"}
        and aggregate_result["winner"] != majority_winner
    )

    explanation = _build_explanation(
        aggregate_result=aggregate_result,
        winner_counts=winner_counts,
        majority_winner=majority_winner,
        paradox_detected=paradox_detected,
    )

    return {
        "paradox_detected": paradox_detected,
        "aggregate_result": aggregate_result,
        "subgroup_results": subgroup_results,
        "reversal_explanation": explanation,
    }


def _validate_inputs(df, treatment_col, outcome_col, confounder_col):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    if missing_columns := [
        column
        for column in (treatment_col, outcome_col, confounder_col)
        if column not in df.columns
    ]:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")


def _normalize_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, Number):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return str(numeric_value)
    return str(value).strip().lower()


def _coerce_success_indicator(series: pd.Series) -> pd.Series:
    normalized = series.map(_normalize_value)
    unique_values = list(pd.unique(normalized.dropna()))
    if len(unique_values) != 2:
        raise ValueError("outcome_col must be binary and contain exactly two values.")

    success_value = _identify_success_value(unique_values)
    return normalized == success_value


def _identify_success_value(unique_values):
    if any(value in SUCCESS_ALIASES for value in unique_values):
        for value in unique_values:
            if value in SUCCESS_ALIASES:
                return value

    if any(value in FAILURE_ALIASES for value in unique_values):
        for value in unique_values:
            if value not in FAILURE_ALIASES:
                return value

    try:
        return max(unique_values, key=float)
    except ValueError:
        return unique_values[-1]


def _identify_variant_labels(series: pd.Series):
    unique_values = list(pd.unique(series.dropna()))
    if len(unique_values) != 2:
        raise ValueError("treatment_col must contain exactly two groups.")

    control_label = next((value for value in unique_values if value in CONTROL_ALIASES), None)
    treatment_label = next((value for value in unique_values if value in TREATMENT_ALIASES), None)

    if control_label is not None and treatment_label is not None and control_label != treatment_label:
        return control_label, treatment_label

    if set(unique_values) == {"0", "1"}:
        return "0", "1"

    return unique_values[0], unique_values[1]


def _summarize_slice(df: pd.DataFrame, control_label: str, treatment_label: str):
    treatment_mask = df["treatment"] == treatment_label
    control_mask = df["treatment"] == control_label

    treatment_total = int(treatment_mask.sum())
    control_total = int(control_mask.sum())

    treatment_rate = _safe_rate(df.loc[treatment_mask, "success"])
    control_rate = _safe_rate(df.loc[control_mask, "success"])

    return {
        "treatment_rate": treatment_rate,
        "control_rate": control_rate,
        "winner": _compare_rates(treatment_rate, control_rate, treatment_total, control_total),
        "treatment_size": treatment_total,
        "control_size": control_total,
    }


def _safe_rate(values: pd.Series) -> float:
    return float("nan") if values.empty else float(values.mean())


def _compare_rates(treatment_rate: float, control_rate: float, treatment_total: int, control_total: int):
    if treatment_total == 0 or control_total == 0:
        return "insufficient_data"
    if np.isclose(treatment_rate, control_rate):
        return "tie"
    return "treatment" if treatment_rate > control_rate else "control"


def _winner_name(winner: str) -> str:
    return winner


def _majority_winner(winner_counts: Counter):
    if not winner_counts:
        return None
    ordered = winner_counts.most_common()
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]


def _build_explanation(aggregate_result, winner_counts, majority_winner, paradox_detected: bool):
    treatment_rate = 100 * aggregate_result["treatment_rate"]
    control_rate = 100 * aggregate_result["control_rate"]

    if paradox_detected:
        return (
            f"Aggregate results favor {aggregate_result['winner']} "
            f"({treatment_rate:.1f}% treatment vs {control_rate:.1f}% control), "
            f"but most subgroups favor {majority_winner}. This reversal is consistent with "
            "Simpson's paradox, where a lurking variable and unequal subgroup sizes distort "
            "the overall comparison."
        )

    if majority_winner is None:
        return (
            "No clear subgroup majority winner was found, so there is no aggregate-versus-subgroup "
            "reversal to flag."
        )

    return (
        f"Aggregate results favor {aggregate_result['winner']} "
        f"({treatment_rate:.1f}% treatment vs {control_rate:.1f}% control), and the subgroup "
        f"pattern does not reverse that conclusion. Subgroup winner counts: {dict(winner_counts)}."
    )
