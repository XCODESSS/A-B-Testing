import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.sample_size import (
    DEFAULT_ALPHA,
    DEFAULT_POWER,
    calculate_power,
    calculate_sample_size_continuous,
    calculate_sample_size_proportion,
)
from src.simulation import simulate_pvalue_distribution
from src.stat_tests import chi_square_test, mann_whitney_test, t_test

st.set_page_config(
    page_title="A/B Testing Intelligence Platform",
    page_icon="📊",
    layout="centered",
)

TEST_TYPE_PROPORTION = "Conversion Rate"
TEST_TYPE_CONTINUOUS = "Continuous Metric"
MDE_TYPE_ABSOLUTE = "Absolute"
MDE_TYPE_RELATIVE = "Relative"

STAT_TEST_T = "T-test"
STAT_TEST_CHI = "Chi-square"
STAT_TEST_MW = "Mann-Whitney"
P_VALUE_TEST_OPTIONS = {
    STAT_TEST_T: "t_test",
    STAT_TEST_CHI: "chi_square",
    STAT_TEST_MW: "mann_whitney",
}

MIN_PROBABILITY = 0.001
MAX_PROBABILITY = 0.999
MIN_MDE = 0.01
MIN_STD_DEV = 0.01
MIN_POWER = 0.70

POWER_CURVE_MDE_MIN = 0.005
POWER_CURVE_MDE_MAX = 0.10
POWER_CURVE_POINTS = 50
TARGET_POWER = 0.80


def render_test_type_selector():
    return st.radio(
        "Test Type",
        options=[TEST_TYPE_PROPORTION, TEST_TYPE_CONTINUOUS],
        horizontal=True,
    )


def render_mde_type_selector():
    return st.radio(
        "MDE Type",
        options=[MDE_TYPE_ABSOLUTE, MDE_TYPE_RELATIVE],
        horizontal=True,
    )


def render_common_inputs():
    col_alpha, col_power = st.columns(2)
    with col_alpha:
        alpha = st.number_input(
            "Significance Level (alpha)",
            min_value=MIN_PROBABILITY,
            max_value=MAX_PROBABILITY,
            value=DEFAULT_ALPHA,
            step=0.001,
            format="%.3f",
        )
    with col_power:
        power = st.number_input(
            "Statistical Power (1 - beta)",
            min_value=MIN_POWER,
            max_value=MAX_PROBABILITY,
            value=DEFAULT_POWER,
            step=0.05,
            format="%.2f",
        )
    return alpha, power


def render_proportion_inputs(mde_type_label):
    is_relative = mde_type_label == MDE_TYPE_RELATIVE
    col_baseline, col_mde = st.columns(2)
    with col_baseline:
        baseline_rate = st.number_input(
            "Baseline Conversion Rate",
            min_value=MIN_PROBABILITY,
            max_value=MAX_PROBABILITY,
            value=0.10,
            step=0.01,
            format="%.3f",
        )
    with col_mde:
        mde = st.number_input(
            f"Minimum Detectable Effect ({'relative' if is_relative else 'absolute'})",
            min_value=MIN_MDE,
            max_value=0.50 if is_relative else MAX_PROBABILITY,
            value=0.05 if is_relative else 0.02,
            step=0.005,
            format="%.3f",
        )
    return baseline_rate, mde


def render_continuous_inputs():
    col_baseline, col_mde, col_std = st.columns(3)
    with col_baseline:
        baseline_mean = st.number_input("Baseline Mean", value=50.0, step=1.0, format="%.2f")
    with col_mde:
        mde = st.number_input(
            "Minimum Detectable Effect (absolute)",
            min_value=MIN_MDE,
            value=5.0,
            step=0.5,
            format="%.2f",
        )
    with col_std:
        std_dev = st.number_input(
            "Standard Deviation (sigma)",
            min_value=MIN_STD_DEV,
            value=20.0,
            step=1.0,
            format="%.2f",
        )
    return baseline_mean, mde, std_dev


def display_results(per_group, summary_text):
    total = per_group * 2
    st.divider()
    col_per, col_total = st.columns(2)
    with col_per:
        st.metric("Per Group", f"{per_group:,}")
    with col_total:
        st.metric("Total", f"{total:,}")
    st.info(summary_text)


def run_proportion_calculation(baseline_rate, mde, alpha, power, mde_type_key):
    try:
        per_group = calculate_sample_size_proportion(
            baseline_rate=baseline_rate,
            minimum_detectable_effect=mde,
            alpha=alpha,
            power=power,
            mde_type=mde_type_key,
        )
        treatment_rate = baseline_rate * (1 + mde) if mde_type_key == "relative" else baseline_rate + mde
        display_results(
            per_group,
            f"Need {per_group:,} users per group ({per_group * 2:,} total) to detect "
            f"{baseline_rate:.1%} -> {treatment_rate:.1%} at power {power:.0%}, alpha={alpha}.",
        )
    except ValueError as error:
        st.error(f"Invalid input: {error}")


def run_continuous_calculation(baseline_mean, mde, std_dev, alpha, power, mde_type_key):
    try:
        per_group = calculate_sample_size_continuous(
            baseline_mean=baseline_mean,
            minimum_detectable_effect=mde,
            std_dev=std_dev,
            alpha=alpha,
            power=power,
            mde_type=mde_type_key,
        )
        treatment_mean = baseline_mean * (1 + mde) if mde_type_key == "relative" else baseline_mean + mde
        display_results(
            per_group,
            f"Need {per_group:,} observations per group ({per_group * 2:,} total) "
            f"to detect {baseline_mean:.2f} -> {treatment_mean:.2f} at power {power:.0%}, alpha={alpha}.",
        )
    except ValueError as error:
        st.error(f"Invalid input: {error}")


def _render_power_curve(mde_range, powers):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=mde_range,
            y=powers,
            mode="lines",
            name="Achieved Power",
            line=dict(color="#4F8EF7", width=2),
        )
    )
    fig.add_hline(
        y=TARGET_POWER,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"{int(TARGET_POWER * 100)}% target",
        annotation_position="bottom right",
    )
    fig.update_layout(
        xaxis_title="Minimum Detectable Effect (absolute)",
        yaxis_title="Statistical Power",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        xaxis=dict(tickformat=".3f"),
        hovermode="x unified",
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_crossover_callout(mde_range, powers):
    crossover = next((mde for mde, pwr in zip(mde_range, powers) if pwr >= TARGET_POWER), None)
    if crossover is None:
        st.warning("Power never reaches 80% in this range. Increase sample size.")
        return
    st.info(
        f"At this sample size, you reach {int(TARGET_POWER * 100)}% power "
        f"at MDE={crossover:.3f} (absolute)."
    )


def _load_csv_input():
    upload = st.file_uploader("Upload CSV", type=["csv"])
    csv_text = st.text_area(
        "Or paste CSV data",
        height=140,
        placeholder="group,value\ncontrol,10.2\ntreatment,11.1",
    )

    if upload is not None:
        return pd.read_csv(upload)
    return pd.read_csv(io.StringIO(csv_text)) if csv_text.strip() else None


def _run_numeric_group_test(test_type, df, alpha):
    required_columns = {"group", "value"}
    if not required_columns.issubset(df.columns):
        raise ValueError("For T-test/Mann-Whitney, CSV must contain columns: group,value")

    labels = df["group"].astype(str)
    unique_groups = labels.unique()
    if len(unique_groups) != 2:
        raise ValueError("group column must contain exactly two groups.")

    first_group = df.loc[labels == unique_groups[0], "value"].to_numpy(dtype=float)
    second_group = df.loc[labels == unique_groups[1], "value"].to_numpy(dtype=float)

    if test_type == STAT_TEST_T:
        return t_test(first_group, second_group, alpha=alpha)
    return mann_whitney_test(first_group, second_group, alpha=alpha)


def _run_selected_stat_test(test_type, df, alpha):
    if test_type in {STAT_TEST_T, STAT_TEST_MW}:
        return _run_numeric_group_test(test_type, df, alpha)

    if df.shape[1] == 2 and not np.issubdtype(df.dtypes.iloc[0], np.number):
        contingency = pd.crosstab(df.iloc[:, 0], df.iloc[:, 1]).to_numpy()
    else:
        contingency = df.select_dtypes(include=[np.number]).to_numpy()
    return chi_square_test(contingency, alpha=alpha)


def _render_stat_test_result(result, alpha):
    st.subheader("Results")
    col_stat, col_p = st.columns(2)
    with col_stat:
        st.metric("Statistic", f"{result['statistic']:.6f}")
    with col_p:
        st.metric("P-value", f"{result['p_value']:.6f}")

    conclusion = "Reject H0 (statistically significant)" if result["significant"] else "Fail to reject H0"
    st.write(f"Conclusion at alpha={alpha:.3f}: **{conclusion}**")

    if "confidence_interval" in result:
        lower, upper = result["confidence_interval"]
        st.write(f"Confidence interval: [{lower:.6f}, {upper:.6f}]")


def _render_pvalue_histogram(p_values, alpha):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=p_values, nbinsx=30, name="p-values"))
    fig.add_vline(
        x=alpha,
        line_color="red",
        line_dash="dash",
        annotation_text=f"alpha={alpha:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="P-value",
        yaxis_title="Frequency",
        bargap=0.05,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _calculate_family_wise_error_rate(n_tests, alpha=DEFAULT_ALPHA):
    return 1 - (1 - alpha) ** int(n_tests)


def _build_multiple_testing_table(alpha=DEFAULT_ALPHA):
    test_counts = np.array([1, 5, 10, 20, 50], dtype=int)
    table = pd.DataFrame(
        {
            "Tests": test_counts,
            "False Positive Rate": [
                _calculate_family_wise_error_rate(n_tests=count, alpha=alpha)
                for count in test_counts
            ],
        }
    )
    table["False Positive Rate"] = (100 * table["False Positive Rate"]).map(
        lambda value: f"{value:.1f}%"
    )
    return table


def render_sample_size_calculator():
    st.title("A/B Testing Intelligence Platform")
    st.caption("Sample Size Calculator")

    test_type = render_test_type_selector()
    mde_type_label = render_mde_type_selector()
    mde_type_key = mde_type_label.lower()
    alpha, power = render_common_inputs()

    if test_type == TEST_TYPE_PROPORTION:
        baseline_rate, mde = render_proportion_inputs(mde_type_label)
        run_proportion_calculation(baseline_rate, mde, alpha, power, mde_type_key)
    else:
        baseline_mean, mde, std_dev = render_continuous_inputs()
        run_continuous_calculation(baseline_mean, mde, std_dev, alpha, power, mde_type_key)


def render_power_analysis():
    st.title("A/B Testing Intelligence Platform")
    st.caption("Power Analysis")

    baseline = st.number_input(
        "Baseline Conversion Rate",
        min_value=MIN_PROBABILITY,
        max_value=MAX_PROBABILITY,
        value=0.10,
        step=0.01,
        format="%.2f",
    )
    sample_size_per_group = st.number_input(
        "Sample Size Per Group",
        min_value=100,
        max_value=1_000_000,
        value=3839,
        step=100,
    )
    alpha = st.number_input(
        "Significance Level (alpha)",
        min_value=MIN_PROBABILITY,
        max_value=MAX_PROBABILITY,
        value=DEFAULT_ALPHA,
        step=0.001,
        format="%.3f",
    )

    mde_range = np.linspace(POWER_CURVE_MDE_MIN, POWER_CURVE_MDE_MAX, POWER_CURVE_POINTS)
    try:
        powers = [calculate_power(baseline, mde, int(sample_size_per_group), alpha) for mde in mde_range]
    except ValueError as error:
        st.error(f"Invalid input: {error}")
        return

    _render_power_curve(mde_range, powers)
    _render_crossover_callout(mde_range, powers)


def render_statistical_tests():
    st.title("A/B Testing Intelligence Platform")
    st.caption("Statistical Tests")

    test_type = st.selectbox("Choose test", [STAT_TEST_T, STAT_TEST_CHI, STAT_TEST_MW])
    alpha = st.number_input(
        "Significance Level (alpha)",
        min_value=MIN_PROBABILITY,
        max_value=MAX_PROBABILITY,
        value=DEFAULT_ALPHA,
        step=0.001,
        format="%.3f",
    )
    st.write("Input data as CSV upload or pasted CSV.")
    st.code(
        "T-test/Mann-Whitney format:\n"
        "group,value\n"
        "control,1.2\n"
        "treatment,1.4\n\n"
        "Chi-square formats:\n"
        "1) category_a,category_b\n"
        "2) contingency matrix numeric CSV",
        language="text",
    )
    df = _load_csv_input()

    if st.button("Run Test", type="primary"):
        if df is None:
            st.error("Provide CSV data before running the test.")
            return
        try:
            result = _run_selected_stat_test(test_type, df, alpha)
            _render_stat_test_result(result, alpha)
        except Exception as error:
            st.error(f"Could not run test: {error}")


def render_pvalue_distribution():
    st.title("A/B Testing Intelligence Platform")
    st.caption("P-value Distribution")
    alpha = DEFAULT_ALPHA
    st.write(f"Reference alpha is fixed at {alpha:.2f}.")

    n_simulations = st.slider("Number of simulations", min_value=100, max_value=10000, value=1000, step=100)
    sample_size = st.slider("Sample size per group", min_value=50, max_value=5000, value=500, step=50)
    effect_size = st.slider("Effect size", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
    test_type = st.selectbox("Test type", list(P_VALUE_TEST_OPTIONS.keys()))

    if st.button("Run Simulation", type="primary"):
        p_values = simulate_pvalue_distribution(
            n_simulations=n_simulations,
            sample_size_per_group=sample_size,
            effect_size=effect_size,
            test_type=P_VALUE_TEST_OPTIONS[test_type],
            alpha=alpha,
        )
        _render_pvalue_histogram(p_values, alpha)

        significant_rate = float(np.mean(p_values < alpha))
        mean_p_value = float(np.mean(p_values))
        col_sig, col_mean = st.columns(2)
        with col_sig:
            st.metric("Tests significant", f"{100 * significant_rate:.2f}%")
        with col_mean:
            st.metric("Mean p-value", f"{mean_p_value:.3f}")

        if effect_size == 0:
            st.info(
                "Interpretation: With no true effect, p-values should look close to uniform "
                "between 0 and 1. Around 5% below 0.05 is expected by chance."
            )
        else:
            st.info(
                "Interpretation: With a real effect, p-values should skew toward 0. "
                "More mass near 0 and a lower mean p-value indicate stronger signal."
            )

    st.subheader("Multiple Testing Problem")
    st.write("`P(>=1 false positive) = 1 - (1 - alpha)^n`")
    n_tests = st.number_input("Number of independent tests", min_value=1, max_value=500, value=20, step=1)
    false_positive_probability = _calculate_family_wise_error_rate(n_tests=n_tests, alpha=alpha)
    st.write(
        f"For **{int(n_tests)}** tests, `P(>=1 false positive)` is "
        f"**{100 * false_positive_probability:.1f}%**."
    )

    st.dataframe(_build_multiple_testing_table(alpha=alpha), use_container_width=True)

    corrected_alpha = alpha / int(n_tests)
    st.write(f"Bonferroni corrected alpha: `alpha / n = {alpha:.2f} / {int(n_tests)} = {corrected_alpha:.6f}`")
    st.warning("This is why you can't just run 20 A/B tests and cherry-pick winners.")


PAGE_ROUTER = {
    "Sample Size Calculator": render_sample_size_calculator,
    "Power Analysis": render_power_analysis,
    "Statistical Tests": render_statistical_tests,
    "P-value Distribution": render_pvalue_distribution,
}


def render_sidebar():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Select tool", options=list(PAGE_ROUTER.keys()), index=0)


def main():
    page = render_sidebar()
    if render_fn := PAGE_ROUTER.get(page):
        render_fn()


if __name__ == "__main__":
    main()
