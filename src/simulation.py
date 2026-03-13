"""Simulation utilities for p-value behavior."""

from __future__ import annotations

import numpy as np

from src.stat_tests import chi_square_test, mann_whitney_test, t_test

TEST_TYPE_T = "t_test"
TEST_TYPE_CHI = "chi_square"
TEST_TYPE_MW = "mann_whitney"
VALID_TEST_TYPES = {TEST_TYPE_T, TEST_TYPE_CHI, TEST_TYPE_MW}
_TEST_TYPE_ALIASES = {
    TEST_TYPE_T: TEST_TYPE_T,
    TEST_TYPE_CHI: TEST_TYPE_CHI,
    TEST_TYPE_MW: TEST_TYPE_MW,
    "T-test": TEST_TYPE_T,
    "Chi-square": TEST_TYPE_CHI,
    "Mann-Whitney": TEST_TYPE_MW,
}


def simulate_pvalue_distribution(
    n_simulations=1000,
    sample_size_per_group=100,
    effect_size=0.0,
    test_type="t_test",
    alpha=0.05,
    random_state=None,
    sample_size=None,
):
    """
    Run repeated A/B tests and collect p-values.

    effect_size=0.0 means the null hypothesis is true (no real effect).
    effect_size>0.0 means the alternative is true (a real effect exists).

    Returns:
        np.ndarray: p-values of shape (n_simulations,).
    """
    resolved_sample_size = sample_size_per_group if sample_size is None else sample_size
    normalized_test_type = _normalize_test_type(test_type)

    _validate_simulation_inputs(
        n_simulations=n_simulations,
        sample_size=resolved_sample_size,
        effect_size=effect_size,
        test_type=normalized_test_type,
        alpha=alpha,
    )

    rng = np.random.default_rng(seed=random_state)
    p_values = np.empty(int(n_simulations), dtype=float)

    for index in range(int(n_simulations)):
        if normalized_test_type == TEST_TYPE_T:
            p_values[index] = _simulate_t_test_p_value(
                rng=rng,
                sample_size=int(resolved_sample_size),
                effect_size=float(effect_size),
                alpha=float(alpha),
            )
            continue

        if normalized_test_type == TEST_TYPE_MW:
            p_values[index] = _simulate_mann_whitney_p_value(
                rng=rng,
                sample_size=int(resolved_sample_size),
                effect_size=float(effect_size),
                alpha=float(alpha),
            )
            continue

        p_values[index] = _simulate_chi_square_p_value(
            rng=rng,
            sample_size=int(resolved_sample_size),
            effect_size=float(effect_size),
            alpha=float(alpha),
        )

    return p_values


def _normalize_test_type(test_type: str) -> str:
    if test_type not in _TEST_TYPE_ALIASES:
        valid_values = ", ".join(sorted(VALID_TEST_TYPES))
        raise ValueError(f"test_type must be one of: {valid_values}.")
    return _TEST_TYPE_ALIASES[test_type]


def _validate_simulation_inputs(n_simulations, sample_size, effect_size, test_type, alpha):
    if int(n_simulations) < 1:
        raise ValueError("n_simulations must be an integer >= 1.")
    if int(sample_size) < 2:
        raise ValueError("sample_size must be an integer >= 2.")
    if effect_size < 0:
        raise ValueError("effect_size must be >= 0.")
    if test_type not in VALID_TEST_TYPES:
        raise ValueError(f"test_type must be one of {VALID_TEST_TYPES}.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def _simulate_t_test_p_value(rng, sample_size: int, effect_size: float, alpha: float) -> float:
    control = rng.normal(loc=0.0, scale=1.0, size=sample_size)
    treatment = rng.normal(loc=effect_size, scale=1.0, size=sample_size)
    return float(t_test(control, treatment, alpha=alpha)["p_value"])


def _simulate_mann_whitney_p_value(
    rng, sample_size: int, effect_size: float, alpha: float
) -> float:
    control = rng.exponential(scale=1.0, size=sample_size)
    treatment = rng.exponential(scale=1.0, size=sample_size) + effect_size
    return float(mann_whitney_test(control, treatment, alpha=alpha)["p_value"])


def _simulate_chi_square_p_value(
    rng, sample_size: int, effect_size: float, alpha: float
) -> float:
    baseline_rate = 0.5
    treatment_rate = min(0.99, baseline_rate + effect_size)

    control_successes = int(rng.binomial(n=sample_size, p=baseline_rate))
    treatment_successes = int(rng.binomial(n=sample_size, p=treatment_rate))

    contingency = np.array(
        [
            [control_successes, sample_size - control_successes],
            [treatment_successes, sample_size - treatment_successes],
        ],
        dtype=float,
    )
    return float(chi_square_test(contingency, alpha=alpha)["p_value"])
