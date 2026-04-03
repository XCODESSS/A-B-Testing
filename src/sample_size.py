import math

from scipy import stats

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
MDE_ABSOLUTE = "absolute"
MDE_RELATIVE = "relative"
VALID_MDE_TYPES = {MDE_ABSOLUTE, MDE_RELATIVE}


def calculate_sample_size_proportion(
    baseline_rate,
    minimum_detectable_effect,
    alpha=DEFAULT_ALPHA,
    power=DEFAULT_POWER,
    mde_type=MDE_ABSOLUTE,
):
    """
    Sample size per group for a two-proportion z-test.

    Formula:
        n = (Z_alpha/2 + Z_beta)^2 * (p1(1-p1) + p2(1-p2)) / (p1 - p2)^2
    """
    _validate_proportion_inputs(
        baseline_rate, minimum_detectable_effect, alpha, power, mde_type
    )

    treatment_rate = _compute_treatment_value(
        baseline_rate, minimum_detectable_effect, mde_type
    )
    _validate_treatment_rate(treatment_rate)

    variance = _bernoulli_combined_variance(baseline_rate, treatment_rate)
    effect_squared = (treatment_rate - baseline_rate) ** 2
    z_threshold = _required_z_threshold(alpha, power)

    per_group = (z_threshold**2 * variance) / effect_squared
    return math.ceil(per_group)


def calculate_sample_size_continuous(
    baseline_mean,
    minimum_detectable_effect,
    std_dev,
    alpha=DEFAULT_ALPHA,
    power=DEFAULT_POWER,
    mde_type=MDE_ABSOLUTE,
):
    """
    Sample size per group for a two-sample test on continuous metrics.

    Formula:
        n = 2 * (Z_alpha/2 + Z_beta)^2 * sigma^2 / delta^2
    """
    _validate_continuous_inputs(
        std_dev, minimum_detectable_effect, alpha, power, mde_type
    )

    treatment_mean = _compute_treatment_value(
        baseline_mean, minimum_detectable_effect, mde_type
    )
    effect_squared = (treatment_mean - baseline_mean) ** 2
    z_threshold = _required_z_threshold(alpha, power)

    per_group = 2 * (z_threshold**2) * (std_dev**2) / effect_squared
    return math.ceil(per_group)


def calculate_power(
    baseline_rate,
    minimum_detectable_effect,
    sample_size_per_group,
    alpha=DEFAULT_ALPHA,
    mde_type=MDE_ABSOLUTE,
):
    """
    Power achieved at a given sample size per group for a two-proportion z-test.
    """
    _validate_power_inputs(
        baseline_rate,
        minimum_detectable_effect,
        alpha,
        sample_size_per_group,
        mde_type,
    )

    treatment_rate = _compute_treatment_value(
        baseline_rate, minimum_detectable_effect, mde_type
    )
    _validate_treatment_rate(treatment_rate)

    variance = _bernoulli_combined_variance(baseline_rate, treatment_rate)
    effect_squared = (treatment_rate - baseline_rate) ** 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_combined = math.sqrt(sample_size_per_group * effect_squared / variance)
    z_beta = z_combined - z_alpha

    return float(stats.norm.cdf(z_beta))


def _validate_proportion_inputs(baseline_rate, mde, alpha, power, mde_type):
    if not _is_valid_probability(baseline_rate):
        raise ValueError(
            f"baseline_rate must be in (0, 1), got {baseline_rate}"
        )
    _validate_common_inputs(mde, alpha, power, mde_type)


def _validate_continuous_inputs(std_dev, mde, alpha, power, mde_type):
    if std_dev <= 0:
        raise ValueError(f"std_dev must be positive, got {std_dev}")
    _validate_common_inputs(mde, alpha, power, mde_type)


def _validate_power_inputs(
    baseline_rate, mde, alpha, sample_size_per_group, mde_type
):
    if not _is_valid_probability(baseline_rate):
        raise ValueError(
            f"baseline_rate must be in (0, 1), got {baseline_rate}"
        )
    if mde <= 0:
        raise ValueError(
            f"minimum_detectable_effect must be positive, got {mde}"
        )
    if not _is_valid_probability(alpha):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not isinstance(sample_size_per_group, int) or sample_size_per_group < 2:
        raise ValueError(
            "sample_size_per_group must be an integer >= 2, "
            f"got {sample_size_per_group}"
        )
    if mde_type not in VALID_MDE_TYPES:
        raise ValueError(
            f"mde_type must be one of {VALID_MDE_TYPES}, got '{mde_type}'"
        )


def _validate_common_inputs(mde, alpha, power, mde_type):
    if mde <= 0:
        raise ValueError(
            f"minimum_detectable_effect must be positive, got {mde}"
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
        raise ValueError(
            f"Treatment rate {treatment_rate:.4f} falls outside (0, 1). "
            "Reduce your minimum detectable effect."
        )


def _is_valid_probability(value):
    return 0 < value < 1


def _compute_treatment_value(baseline, mde, mde_type):
    return baseline * (1 + mde) if mde_type == MDE_RELATIVE else baseline + mde


def _bernoulli_combined_variance(control_rate, treatment_rate):
    control_variance = control_rate * (1 - control_rate)
    treatment_variance = treatment_rate * (1 - treatment_rate)
    return control_variance + treatment_variance


def _required_z_threshold(alpha, power):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return z_alpha + z_beta
