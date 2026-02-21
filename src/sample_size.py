#IMPORTS--------------------------------------------------------------------
import scipy.stats as stats
import math
# NAMED CONSTANTS
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
MDE_ABSOLUTE  = 'absolute'
MDE_RELATIVE  = 'relative'
VALID_MDE_TYPES = [MDE_ABSOLUTE, MDE_RELATIVE]

def calculate_sample_size(
    baseline_rate,
    minimum_detectable_effect,
    alpha=DEFAULT_ALPHA,
    power=DEFAULT_POWER,
    mde_type=MDE_ABSOLUTE,
):
    """
    Calculate the required sample size for an A/B test.
    Parameters:
    - baseline_rate: The conversion rate of the control group (as a decimal).
    - mde: The minimum detectable effect (as a decimal).
    - alpha: The significance level (commonly set to 0.05).
    - power: The desired power of the test (commonly set to 0.8).
    Returns:
    - The required total sample size.
    Formula: n = (Z_α/2 + Z_β)² * (p1(1-p1) + p2(1-p2)) / (p1-p2)²
    Where:
    - Z_α/2 is the z-score corresponding to the significance level (alpha).
    - Z_β is the z-score corresponding to the power (1 - beta).
    - p1 is the baseline conversion rate (control group).
    - p2 is the expected conversion rate in the treatment group (baseline_rate + mde).
    - p1-p2 is the difference between the two conversion rates.
    - n is the required sample size per group.
    """
    _validate_inputs(baseline_rate, minimum_detectable_effect, alpha, power, mde_type)
    treatment_rate = _compute_treatment_rate(
        baseline_rate, minimum_detectable_effect, mde_type
    )
    _validate_treatment_rate(treatment_rate)

    pooled_variance = _combined_variance(baseline_rate, treatment_rate)
    effected_squared = (treatment_rate - baseline_rate) ** 2
    z_threshold = _required_z_threshold(alpha, power)

    raw_sample_size = (z_threshold ** 2 * pooled_variance) / effected_squared

    return math.ceil(raw_sample_size)

#----Input validation(fail fasst)---

def _validate_inputs(baseline_rate, minimum_detectable_effect, alpha, power, mde_type):
    """
    Required sample size per group for a two-proportion z-test.
    Returns the smallest integer n (per group) such that detecting
    `minimum_detectable_effect` is possible at the given
    alpha and power levels. Multiply by 2 for total experiment size.
    """
    if not _is_valid_probability(baseline_rate):
        raise ValueError(f"baseline_rate must be in (0, 1), got {baseline_rate}")

    if minimum_detectable_effect <= 0:
        raise ValueError(
            f"minimum_detectable_effect must be positive, got {minimum_detectable_effect}"
        )

    if not _is_valid_probability(alpha):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if not _is_valid_probability(power):
        raise ValueError(f"power must be in (0, 1), got {power}")

    if mde_type not in VALID_MDE_TYPES:
        raise ValueError(
            f"mde_type must be one of {VALID_MDE_TYPES}, got '{mde_type}'"
        )

def _validate_treatment_rate(treatment_rate):
    if not _is_valid_probability(treatment_rate):
        raise ValueError(f"Treatment rate must be in (0, 1), got {treatment_rate}"
        )

def _is_valid_probability(value):
    return 0 < value < 1

#----Core Computations (pure functions)---

def _compute_treatment_rate(baseline_rate, mde, mde_type):
    if mde_type == MDE_ABSOLUTE:
        return baseline_rate + mde
    return baseline_rate * (1 + mde)

def _combined_variance(control_rate, treatment_rate):
    """Unpooled variance: sum of individual Bernoulli variances."""
    control_variance = control_rate * (1 - control_rate)
    treatment_variance = treatment_rate * (1 - treatment_rate)
    return control_variance + treatment_variance

def _required_z_threshold(alpha, power):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return z_alpha + z_beta

print(calculate_sample_size(
    baseline_rate=0.10,
    minimum_detectable_effect=0.02,
    alpha=0.05,
    power=0.80
))