"""Statistical tests for A/B analysis."""

from __future__ import annotations

import numpy as np
from scipy import stats

DEFAULT_ALPHA = 0.05
MIN_T_TEST_OBSERVATIONS = 2
MIN_GROUP_OBSERVATIONS = 1
MIN_CONTINGENCY_SHAPE = 2


def t_test(group_a, group_b, alpha=DEFAULT_ALPHA):
    """
    Welch's two-sample t-test.
    Returns: statistic, p_value, significant, confidence_interval
    """
    _validate_alpha(alpha)
    sample_a = _to_valid_numeric_array(
        group_a, "group_a", minimum_size=MIN_T_TEST_OBSERVATIONS
    )
    sample_b = _to_valid_numeric_array(
        group_b, "group_b", minimum_size=MIN_T_TEST_OBSERVATIONS
    )

    mean_difference = float(np.mean(sample_a) - np.mean(sample_b))
    variance_term = (
        np.var(sample_a, ddof=1) / sample_a.size
        + np.var(sample_b, ddof=1) / sample_b.size
    )
    standard_error = float(np.sqrt(variance_term))

    if standard_error == 0:
        return _zero_variance_t_test_result(mean_difference, alpha)

    degrees_of_freedom = _welch_degrees_of_freedom(sample_a, sample_b)
    statistic = float(mean_difference / standard_error)
    p_value = float(2 * stats.t.sf(np.abs(statistic), degrees_of_freedom))
    critical_value = float(stats.t.ppf(1 - alpha / 2, degrees_of_freedom))
    margin = critical_value * standard_error
    confidence_interval = (mean_difference - margin, mean_difference + margin)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < alpha,
        "confidence_interval": confidence_interval,
    }


def chi_square_test(contingency_table, alpha=DEFAULT_ALPHA):
    """
    Pearson chi-square test of independence for a contingency table.
    Returns: statistic, p_value, significant
    """
    _validate_alpha(alpha)
    observed = _to_valid_contingency_table(contingency_table)

    statistic, p_value, _, _ = stats.chi2_contingency(
        observed, correction=False
    )
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": float(p_value) < alpha,
    }


def mann_whitney_test(group_a, group_b, alpha=DEFAULT_ALPHA):
    """
    Mann-Whitney U test for two independent samples.
    Returns: statistic, p_value, significant
    """
    _validate_alpha(alpha)
    sample_a = _to_valid_numeric_array(
        group_a, "group_a", minimum_size=MIN_GROUP_OBSERVATIONS
    )
    sample_b = _to_valid_numeric_array(
        group_b, "group_b", minimum_size=MIN_GROUP_OBSERVATIONS
    )

    u_statistic = _mann_whitney_u_statistic(sample_a, sample_b)
    scipy_result = stats.mannwhitneyu(
        sample_a, sample_b, alternative="two-sided", method="auto"
    )
    p_value = float(scipy_result.pvalue)

    return {
        "statistic": float(u_statistic),
        "p_value": p_value,
        "significant": p_value < alpha,
    }


def _validate_alpha(alpha):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def _to_valid_numeric_array(group, group_name: str, minimum_size: int) -> np.ndarray:
    values = np.asarray(group, dtype=float).ravel()

    if values.size < minimum_size:
        raise ValueError(
            f"{group_name} must contain at least {minimum_size} observations."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{group_name} must contain only finite numeric values.")

    return values


def _to_valid_contingency_table(contingency_table) -> np.ndarray:
    observed = np.asarray(contingency_table, dtype=float)

    if observed.ndim != 2:
        raise ValueError("contingency_table must be a 2D array.")
    if observed.shape[0] < MIN_CONTINGENCY_SHAPE or observed.shape[1] < MIN_CONTINGENCY_SHAPE:
        raise ValueError("contingency_table must be at least 2x2.")
    if not np.isfinite(observed).all():
        raise ValueError("contingency_table must contain only finite values.")
    if (observed < 0).any():
        raise ValueError("contingency_table cannot contain negative counts.")
    if np.any(observed.sum(axis=1) == 0) or np.any(observed.sum(axis=0) == 0):
        raise ValueError(
            "contingency_table cannot have rows or columns with zero total count."
        )

    return observed


def _welch_degrees_of_freedom(group_a: np.ndarray, group_b: np.ndarray) -> float:
    size_a = group_a.size
    size_b = group_b.size
    variance_a = np.var(group_a, ddof=1)
    variance_b = np.var(group_b, ddof=1)

    term_a = variance_a / size_a
    term_b = variance_b / size_b
    numerator = (term_a + term_b) ** 2
    denominator = (term_a**2) / (size_a - 1) + (term_b**2) / (size_b - 1)

    return np.inf if denominator == 0 else numerator / denominator


def _zero_variance_t_test_result(mean_difference: float, alpha: float):
    statistic = (
        0.0
        if mean_difference == 0
        else float(np.sign(mean_difference) * np.inf)
    )
    p_value = 1.0 if mean_difference == 0 else 0.0
    confidence_interval = (mean_difference, mean_difference)

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": p_value < alpha,
        "confidence_interval": confidence_interval,
    }


def _mann_whitney_u_statistic(
    sample_a: np.ndarray, sample_b: np.ndarray
) -> float:
    combined = np.concatenate([sample_a, sample_b])
    ranks = stats.rankdata(combined, method="average")
    rank_sum_a = float(np.sum(ranks[: sample_a.size]))
    size_a = sample_a.size
    size_b = sample_b.size

    return rank_sum_a - (size_a * (size_a + 1) / 2)