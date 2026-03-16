"""Multiple-comparison correction utilities."""

from __future__ import annotations

from typing import Sequence


def apply_bonferroni(pvalues: list[float], alpha=0.05) -> list[dict]:
    """Apply Bonferroni correction to a list of p-values."""
    validated_pvalues = _validate_inputs(pvalues, alpha)
    n_tests = len(validated_pvalues)

    return [
        {
            "original_p": p_value,
            "corrected_p": min(p_value * n_tests, 1.0),
            "significant": min(p_value * n_tests, 1.0) <= alpha,
        }
        for p_value in validated_pvalues
    ]


def apply_benjamini_hochberg(pvalues: list[float], alpha=0.05) -> list[dict]:
    """Apply Benjamini-Hochberg FDR correction to a list of p-values."""
    validated_pvalues = _validate_inputs(pvalues, alpha)
    n_tests = len(validated_pvalues)
    indexed_pvalues = sorted(enumerate(validated_pvalues), key=lambda item: item[1])

    adjusted = [0.0] * n_tests
    running_min = 1.0
    for reverse_rank, (original_index, p_value) in enumerate(reversed(indexed_pvalues), start=1):
        rank = n_tests - reverse_rank + 1
        candidate = min((p_value * n_tests) / rank, 1.0)
        running_min = min(running_min, candidate)
        adjusted[original_index] = running_min

    return [
        {
            "original_p": p_value,
            "corrected_p": adjusted[index],
            "significant": adjusted[index] <= alpha,
        }
        for index, p_value in enumerate(validated_pvalues)
    ]


def _validate_inputs(pvalues: Sequence[float], alpha: float):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if not pvalues:
        raise ValueError("pvalues must contain at least one value.")

    validated_pvalues = [float(p_value) for p_value in pvalues]
    if any(p_value < 0 or p_value > 1 for p_value in validated_pvalues):
        raise ValueError("pvalues must all be between 0 and 1.")

    return validated_pvalues
