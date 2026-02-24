import math
from scipy import stats

# --- Named Constants ---

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
MDE_ABSOLUTE = "absolute"
MDE_RELATIVE = "relative"
VALID_MDE_TYPES = {MDE_ABSOLUTE, MDE_RELATIVE}


# === Public API ===


def calculate_sample_size_proportion(
    baseline_rate,
    minimum_detectable_effect,
    alpha=DEFAULT_ALPHA,
    power=DEFAULT_POWER,
    mde_type=MDE_ABSOLUTE,
):
    """
    Sample size per group for a two-proportion z-test.

    Formula: n = (Z_α/2 + Z_β)² * (p1(1-p1) + p2(1-p2)) / (p1 - p2)²

    Multiply by 2 for total experiment size.
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
    Sample size per group for a two-sample t/z-test on continuous metrics.

    Formula: n = 2 * (Z_α/2 + Z_β)² * σ² / δ²

    Multiply by 2 for total experiment size.
    """
    _validate_continuous_inputs(
        std_dev, minimum_detectable_effect, alpha, power, mde_type
    )

    treatment_mean = _compute_treatment_value(
        baseline_mean, minimum_detectable_effect, mde_type
    )

    effect_squared = (treatment_mean - baseline_mean) ** 2
    z_threshold = _required_z_threshold(alpha, power)

    # Factor of 2: one σ² per group
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
    Exact inverse of calculate_sample_size_proportion.

    Derivation — rearranging the sample size formula:
        n    = (Z_α/2 + Z_β)² × σ² / δ²
        Z_β  = √(n × δ² / σ²) − Z_α/2
        power = Φ(Z_β)

    Due to ceil() in calculate_sample_size_proportion, passing the returned n
    back here will yield power slightly above the originally requested power.
    """
    _validate_power_inputs(
        baseline_rate, minimum_detectable_effect, alpha,
        sample_size_per_group, mde_type,
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


# === Validation — Fail Fast ===


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


# === Core Computations (pure functions) ===


def _compute_treatment_value(baseline, mde, mde_type):
    """Works for both proportions and continuous means."""
    return baseline * (1 + mde) if mde_type == MDE_RELATIVE else baseline + mde


def _bernoulli_combined_variance(control_rate, treatment_rate):
    """Unpooled variance: sum of individual Bernoulli variances."""
    control_variance = control_rate * (1 - control_rate)
    treatment_variance = treatment_rate * (1 - treatment_rate)
    return control_variance + treatment_variance


def _required_z_threshold(alpha, power):
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return z_alpha + z_beta


# === Usage ===

if __name__ == "__main__":
    proportion_n = calculate_sample_size_proportion(
        baseline_rate=0.10,
        minimum_detectable_effect=0.02,
        alpha=0.05,
        power=0.80,
    )
    print(f"Proportion test — per group: {proportion_n}")
    print(f"Proportion test — total:     {proportion_n * 2}")

    recovered_power = calculate_power(
        baseline_rate=0.10,
        minimum_detectable_effect=0.02,
        sample_size_per_group=proportion_n,
        alpha=0.05,
    )
    # Slightly above 0.80 due to ceil() in calculate_sample_size_proportion
    print(f"Recovered power at n={proportion_n}: {recovered_power:.4f}")

    continuous_n = calculate_sample_size_continuous(
        baseline_mean=50.0,
        minimum_detectable_effect=5.0,
        std_dev=20.0,
        alpha=0.05,
        power=0.80,
    )
    print(f"Continuous test — per group: {continuous_n}")
    print(f"Continuous test — total:     {continuous_n * 2}")