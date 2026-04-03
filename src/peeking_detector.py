"""Utilities for flagging optional-stopping / peeking risk."""

from __future__ import annotations

from typing import Sequence

DEFAULT_ALPHA = 0.05


def detect_peeking(pvalues_over_time: list[float] | Sequence[float], alpha=DEFAULT_ALPHA):
    """
    Assess the risk that repeated interim p-value checks inflated false-positive risk.

    Returns:
        dict with peeking_risk, first_significant_at, final_significant,
        false_positive_risk_note.
    """
    _validate_alpha(alpha)
    pvalues = _normalize_pvalues(pvalues_over_time)

    first_significant_at = next((index for index, value in enumerate(pvalues) if value <= alpha), None)
    final_significant = pvalues[-1] <= alpha
    interim_crossings = _find_crossings(pvalues, alpha, stop_before_last=True)

    if len(interim_crossings) > 1:
        peeking_risk = "high"
    elif len(interim_crossings) == 1:
        peeking_risk = "medium"
    else:
        peeking_risk = "low"

    return {
        "peeking_risk": peeking_risk,
        "first_significant_at": first_significant_at,
        "final_significant": final_significant,
        "false_positive_risk_note": _build_false_positive_note(
            pvalues=pvalues,
            alpha=alpha,
            peeking_risk=peeking_risk,
            interim_crossings=interim_crossings,
            first_significant_at=first_significant_at,
            final_significant=final_significant,
        ),
    }


def find_significance_crossings(
    pvalues_over_time: list[float] | Sequence[float], alpha=DEFAULT_ALPHA
) -> list[int]:
    """Return the indices where the series crosses into statistical significance."""
    _validate_alpha(alpha)
    pvalues = _normalize_pvalues(pvalues_over_time)
    return _find_crossings(pvalues, alpha, stop_before_last=False)


def _normalize_pvalues(pvalues_over_time: list[float] | Sequence[float]) -> list[float]:
    if not pvalues_over_time:
        raise ValueError("Provide at least one p-value.")

    normalized = []
    for index, value in enumerate(pvalues_over_time):
        numeric_value = float(value)
        if not 0 <= numeric_value <= 1:
            raise ValueError(f"p-value at index {index} must be between 0 and 1.")
        normalized.append(numeric_value)
    return normalized


def _validate_alpha(alpha):
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be between 0 and 1.")


def _find_crossings(pvalues: Sequence[float], alpha: float, stop_before_last: bool) -> list[int]:
    crossings = []
    previous_significant = False
    last_index = len(pvalues) - 1

    for index, value in enumerate(pvalues):
        if stop_before_last and index == last_index:
            break

        current_significant = value <= alpha
        if current_significant and not previous_significant:
            crossings.append(index)
        previous_significant = current_significant

    return crossings


def _build_false_positive_note(
    pvalues: Sequence[float],
    alpha: float,
    peeking_risk: str,
    interim_crossings: Sequence[int],
    first_significant_at: int | None,
    final_significant: bool,
) -> str:
    last_check = len(pvalues) - 1

    if peeking_risk == "high":
        return (
            f"The p-value crossed below alpha={alpha:.3f} multiple times before the final check. "
            "Repeated looks make lucky early wins more likely, so treat this result as high false-positive risk."
        )
    if peeking_risk == "medium":
        return (
            f"The p-value first crossed alpha={alpha:.3f} at interim check {interim_crossings[0]}. "
            "Because significance appeared before the final look, the experiment may have benefited from peeking."
        )
    if first_significant_at == last_check and final_significant:
        return (
            f"The p-value stayed above alpha={alpha:.3f} until the final check, which is the cleanest pattern here. "
            "False-positive risk from optional stopping looks low."
        )
    return (
        f"The p-value never crossed alpha={alpha:.3f}. "
        "There is no visible sign of peeking-driven significance in this sequence."
    )
