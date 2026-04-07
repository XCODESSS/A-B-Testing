import io
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.business_impact import generate_executive_report, recommend_experiment_action
from src.corrections import apply_benjamini_hochberg, apply_bonferroni
from src.paradox_detector import detect_simpsons_paradox
from src.peeking_detector import detect_peeking, find_significance_crossings
from src.roi_calculator import calculate_roi, project_revenue_series
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

APP_TITLE = "A/B Testing Intelligence Platform"
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


def render_test_type_selector(key_suffix="default"):
    return st.radio(
        "Test Type",
        options=[TEST_TYPE_PROPORTION, TEST_TYPE_CONTINUOUS],
        horizontal=True,
        key=f"test_type_{key_suffix}",
    )


def render_mde_type_selector(key_suffix="default"):
    return st.radio(
        "MDE Type",
        options=[MDE_TYPE_ABSOLUTE, MDE_TYPE_RELATIVE],
        horizontal=True,
        key=f"mde_type_{key_suffix}",
    )


def render_common_inputs(key_suffix="default"):
    col_alpha, col_power = st.columns(2)
    with col_alpha:
        alpha = st.number_input(
            "Significance Level (alpha)",
            min_value=MIN_PROBABILITY,
            max_value=MAX_PROBABILITY,
            value=DEFAULT_ALPHA,
            step=0.001,
            format="%.3f",
            key=f"alpha_{key_suffix}",
        )
    with col_power:
        power = st.number_input(
            "Statistical Power (1 - beta)",
            min_value=MIN_POWER,
            max_value=MAX_PROBABILITY,
            value=DEFAULT_POWER,
            step=0.05,
            format="%.2f",
            key=f"power_{key_suffix}",
        )
    return alpha, power


def render_proportion_inputs(mde_type_label, key_suffix="default"):
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
            key=f"baseline_rate_{key_suffix}",
        )
    with col_mde:
        mde = st.number_input(
            f"Minimum Detectable Effect ({'relative' if is_relative else 'absolute'})",
            min_value=MIN_MDE,
            max_value=0.50 if is_relative else MAX_PROBABILITY,
            value=0.05 if is_relative else 0.02,
            step=0.005,
            format="%.3f",
            key=f"mde_{key_suffix}",
        )
    return baseline_rate, mde


def render_continuous_inputs(key_suffix="default"):
    col_baseline, col_mde, col_std = st.columns(3)
    with col_baseline:
        baseline_mean = st.number_input(
            "Baseline Mean",
            value=50.0,
            step=1.0,
            format="%.2f",
            key=f"baseline_mean_{key_suffix}",
        )
    with col_mde:
        mde = st.number_input(
            "Minimum Detectable Effect (absolute)",
            min_value=MIN_MDE,
            value=5.0,
            step=0.5,
            format="%.2f",
            key=f"continuous_mde_{key_suffix}",
        )
    with col_std:
        std_dev = st.number_input(
            "Standard Deviation (sigma)",
            min_value=MIN_STD_DEV,
            value=20.0,
            step=1.0,
            format="%.2f",
            key=f"std_dev_{key_suffix}",
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
    st.plotly_chart(fig, width="stretch")


def _render_crossover_callout(mde_range, powers):
    crossover = next((mde for mde, pwr in zip(mde_range, powers) if pwr >= TARGET_POWER), None)
    if crossover is None:
        st.warning("Power never reaches 80% in this range. Increase sample size.")
        return
    st.info(
        f"At this sample size, you reach {int(TARGET_POWER * 100)}% power "
        f"at MDE={crossover:.3f} (absolute)."
    )


def _load_csv_input(key_prefix="csv_input", placeholder="group,value\ncontrol,10.2\ntreatment,11.1"):
    upload = st.file_uploader("Upload CSV", type=["csv"], key=f"{key_prefix}_upload")
    csv_text = st.text_area(
        "Or paste CSV data",
        height=140,
        placeholder=placeholder,
        key=f"{key_prefix}_text",
    )

    if upload is not None:
        return pd.read_csv(upload)
    return pd.read_csv(io.StringIO(csv_text)) if csv_text.strip() else None


def _load_sequence_input(key_prefix, uploader_label, text_label, placeholder):
    upload = st.file_uploader(uploader_label, type=["txt", "csv"], key=f"{key_prefix}_upload")
    raw_text = st.text_area(
        text_label,
        height=120,
        placeholder=placeholder,
        key=f"{key_prefix}_text",
    )

    return upload.getvalue().decode("utf-8") if upload is not None else raw_text


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


def _estimate_sample_size(df, test_type):
    if test_type in {STAT_TEST_T, STAT_TEST_MW}:
        return int(df.shape[0])

    if df.shape[1] == 2 and not np.issubdtype(df.dtypes.iloc[0], np.number):
        contingency = pd.crosstab(df.iloc[:, 0], df.iloc[:, 1]).to_numpy()
    else:
        contingency = df.select_dtypes(include=[np.number]).to_numpy()
    return int(np.sum(contingency))


def _render_stat_test_result(result, alpha, sample_size):
    st.subheader("Results")
    col_stat, col_p, col_sig = st.columns(3)
    with col_stat:
        st.metric("Statistic", f"{result['statistic']:.6f}")
    with col_p:
        st.metric("P-value", f"{result['p_value']:.6f}")
    with col_sig:
        st.metric("Significant", "Yes" if result["significant"] else "No")

    conclusion = "Reject H0 (statistically significant)" if result["significant"] else "Fail to reject H0"
    st.write(f"Conclusion at alpha={alpha:.3f}: **{conclusion}**")

    confidence_interval = result.get("confidence_interval")
    if confidence_interval:
        lower, upper = confidence_interval
        st.write(f"Confidence interval: [{lower:.6f}, {upper:.6f}]")

    st.session_state["latest_stat_result"] = {
        "significant": result["significant"],
        "p_value": float(result["p_value"]),
        "confidence_interval": confidence_interval,
        "alpha": float(alpha),
        "sample_size": int(sample_size),
    }


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
    st.plotly_chart(fig, width="stretch")


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


def _build_rate_chart(labels, values, title, color):
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=color,
                text=[f"{100 * value:.1f}%" for value in values],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=title,
        yaxis_title="Conversion Rate",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        margin=dict(t=60, b=40),
    )
    return fig


def _build_subgroup_chart(subgroup_results):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(result["group"]) for result in subgroup_results],
            y=[result["control_rate"] for result in subgroup_results],
            name="Control",
            marker_color="#6C7A89",
            text=[f"{100 * result['control_rate']:.1f}%" for result in subgroup_results],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[str(result["group"]) for result in subgroup_results],
            y=[result["treatment_rate"] for result in subgroup_results],
            name="Treatment",
            marker_color="#2E86AB",
            text=[f"{100 * result['treatment_rate']:.1f}%" for result in subgroup_results],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Subgroup Breakdown",
        barmode="group",
        xaxis_title="Confounder Group",
        yaxis_title="Conversion Rate",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        margin=dict(t=60, b=40),
    )
    return fig


def _build_revenue_chart(revenue_projection):
    values = [
        revenue_projection["baseline_revenue"],
        revenue_projection["projected_revenue"],
    ]
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Baseline Revenue", "Projected Revenue"],
                y=values,
                marker_color=["#6C7A89", "#2E86AB"],
                text=[f"${value:,.0f}" for value in values],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="Experiment Window Revenue",
        yaxis_title="Revenue",
        margin=dict(t=60, b=40),
    )
    return fig


def _build_peeking_chart(pvalues, alpha):
    crossing_indices = find_significance_crossings(pvalues, alpha=alpha)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(pvalues))),
            y=pvalues,
            mode="lines+markers",
            name="P-value",
            line=dict(color="#2E86AB", width=2),
        )
    )
    if crossing_indices:
        fig.add_trace(
            go.Scatter(
                x=crossing_indices,
                y=[pvalues[index] for index in crossing_indices],
                mode="markers",
                name="Crossed alpha",
                marker=dict(color="#D64541", size=11, symbol="diamond"),
            )
        )

    fig.add_hline(
        y=alpha,
        line_color="#D64541",
        line_dash="dash",
        annotation_text=f"alpha={alpha:.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Interim Check Index",
        yaxis_title="P-value",
        yaxis=dict(range=[0, 1]),
        margin=dict(t=40, b=40),
        hovermode="x unified",
    )
    return fig


def _parse_pvalue_text(raw_text: str):
    if tokens := [
        token for token in re.split(r"[\s,]+", raw_text.strip()) if token
    ]:
        return [float(token) for token in tokens]
    else:
        raise ValueError("Enter at least one p-value.")


def _render_paradox_results(result):
    aggregate = result["aggregate_result"]
    subgroup_results = result["subgroup_results"]

    if result["paradox_detected"]:
        st.error(f"Simpson's paradox detected. {result['reversal_explanation']}")
    else:
        st.success(f"No Simpson's paradox detected. {result['reversal_explanation']}")

    col_aggregate, col_subgroups = st.columns(2)
    with col_aggregate:
        aggregate_fig = _build_rate_chart(
            labels=["Control", "Treatment"],
            values=[aggregate["control_rate"], aggregate["treatment_rate"]],
            title="Aggregate Result",
            color=["#6C7A89", "#2E86AB"],
        )
        st.plotly_chart(aggregate_fig, width="stretch")
        st.write(
            f"Winner: **{aggregate['winner']}**"
            f" | Control: **{100 * aggregate['control_rate']:.1f}%**"
            f" | Treatment: **{100 * aggregate['treatment_rate']:.1f}%**"
        )

    with col_subgroups:
        subgroup_fig = _build_subgroup_chart(subgroup_results)
        st.plotly_chart(subgroup_fig, width="stretch")
        st.dataframe(pd.DataFrame(subgroup_results), width="stretch")

    st.info(
        "Why this happens: a lurking variable can change the mix of easy and hard users across variants. "
        "If treatment gets more of the hard-to-convert traffic, unequal group sizes can make the overall "
        "winner reverse even when treatment wins inside the subgroups."
    )


def _render_corrections_table(parsed_pvalues, alpha):
    bonferroni = apply_bonferroni(parsed_pvalues, alpha=alpha)
    benjamini_hochberg = apply_benjamini_hochberg(parsed_pvalues, alpha=alpha)

    correction_table = pd.DataFrame(
        {
            "original_p": [entry["original_p"] for entry in bonferroni],
            "bonferroni_corrected_p": [entry["corrected_p"] for entry in bonferroni],
            "bonferroni_significant": [entry["significant"] for entry in bonferroni],
            "bh_corrected_p": [entry["corrected_p"] for entry in benjamini_hochberg],
            "bh_significant": [entry["significant"] for entry in benjamini_hochberg],
        }
    )
    st.dataframe(correction_table, width="stretch")


def render_sample_size_calculator(show_header=True, key_suffix="sample_size"):
    if show_header:
        st.title(APP_TITLE)
        st.caption("Sample Size Calculator")

    test_type = render_test_type_selector(key_suffix)
    mde_type_label = render_mde_type_selector(key_suffix)
    mde_type_key = mde_type_label.lower()
    alpha, power = render_common_inputs(key_suffix)

    if test_type == TEST_TYPE_PROPORTION:
        baseline_rate, mde = render_proportion_inputs(mde_type_label, key_suffix)
        run_proportion_calculation(baseline_rate, mde, alpha, power, mde_type_key)
    else:
        baseline_mean, mde, std_dev = render_continuous_inputs(key_suffix)
        run_continuous_calculation(baseline_mean, mde, std_dev, alpha, power, mde_type_key)


def render_power_analysis(show_header=True):
    if show_header:
        st.title(APP_TITLE)
        st.caption("Power Analysis")

    baseline = st.number_input(
        "Baseline Conversion Rate",
        min_value=MIN_PROBABILITY,
        max_value=MAX_PROBABILITY,
        value=0.10,
        step=0.01,
        format="%.2f",
        key="power_baseline_rate",
    )
    sample_size_per_group = st.number_input(
        "Sample Size Per Group",
        min_value=100,
        max_value=1_000_000,
        value=3839,
        step=100,
        key="power_sample_size_per_group",
    )
    alpha = st.number_input(
        "Significance Level (alpha)",
        min_value=MIN_PROBABILITY,
        max_value=MAX_PROBABILITY,
        value=DEFAULT_ALPHA,
        step=0.001,
        format="%.3f",
        key="power_analysis_alpha",
    )

    mde_range = np.linspace(POWER_CURVE_MDE_MIN, POWER_CURVE_MDE_MAX, POWER_CURVE_POINTS)
    try:
        powers = [calculate_power(baseline, mde, int(sample_size_per_group), alpha) for mde in mde_range]
    except ValueError as error:
        st.error(f"Invalid input: {error}")
        return

    _render_power_curve(mde_range, powers)
    _render_crossover_callout(mde_range, powers)


def render_statistical_tests(show_header=True):
    if show_header:
        st.title(APP_TITLE)
        st.caption("Statistical Tests")

    test_type = st.selectbox("Choose test", [STAT_TEST_T, STAT_TEST_CHI, STAT_TEST_MW])
    alpha = st.number_input(
        "Significance Level (alpha)",
        min_value=MIN_PROBABILITY,
        max_value=MAX_PROBABILITY,
        value=DEFAULT_ALPHA,
        step=0.001,
        format="%.3f",
        key="stat_test_alpha",
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
            _render_stat_test_result(result, alpha, _estimate_sample_size(df, test_type))
        except Exception as error:
            st.error(f"Could not run test: {error}")


def render_pvalue_distribution(show_header=True):
    if show_header:
        st.title(APP_TITLE)
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

    st.dataframe(_build_multiple_testing_table(alpha=alpha), width="stretch")

    corrected_alpha = alpha / int(n_tests)
    st.write(f"Bonferroni corrected alpha: `alpha / n = {alpha:.2f} / {int(n_tests)} = {corrected_alpha:.6f}`")
    st.warning("This is why you can't just run 20 A/B tests and cherry-pick winners.")


def render_pitfall_detection(show_header=True):
    if show_header:
        st.title(APP_TITLE)
        st.caption("Pitfall Detection")

    st.subheader("Simpson's Paradox Detector")

    df = _load_csv_input(
        key_prefix="pitfall_detection",
        placeholder="variant,converted,segment\ncontrol,1,desktop\ncontrol,0,mobile\ntreatment,1,desktop",
    )

    if df is not None:
        _extracted_from_render_pitfall_detection_14(df)
    alpha = _extracted_from_render_pitfall_detection_29(
        "Multiple Comparison Correction",
        "Correction alpha",
        "pitfall_correction_alpha",
    )
    raw_pvalues = st.text_area(
        "Paste p-values",
        height=120,
        placeholder="0.001, 0.02, 0.04, 0.18",
        key="pitfall_pvalues",
    )

    if st.button("Apply Corrections"):
        try:
            parsed_pvalues = _parse_pvalue_text(raw_pvalues)
            _render_corrections_table(parsed_pvalues, alpha)
        except Exception as error:
            st.error(f"Could not apply corrections: {error}")

    peeking_alpha = _extracted_from_render_pitfall_detection_29(
        "Peeking Detector", "Peeking alpha", "peeking_alpha"
    )
    peeking_raw_text = _load_sequence_input(
        key_prefix="peeking_sequence",
        uploader_label="Upload p-values sequence",
        text_label="Paste p-values (one per interim check, or comma-separated)",
        placeholder="0.42\n0.18\n0.04\n0.07\n0.03",
    )

    if st.button("Detect Peeking"):
        try:
            _extracted_from_render_pitfall_detection_74(peeking_raw_text, peeking_alpha)
        except Exception as error:
            st.error(f"Could not analyze p-values: {error}")


# TODO Rename this here and in `render_pitfall_detection`
def _extracted_from_render_pitfall_detection_74(peeking_raw_text, peeking_alpha):
    pvalues = _parse_pvalue_text(peeking_raw_text)
    result = detect_peeking(pvalues, alpha=peeking_alpha)
    st.plotly_chart(_build_peeking_chart(pvalues, peeking_alpha), width="stretch")

    col_risk, col_first, col_final = st.columns(3)
    with col_risk:
        st.metric("Peeking Risk", result["peeking_risk"].title())
    with col_first:
        first_significant_at = result["first_significant_at"]
        st.metric("First Significant At", "None" if first_significant_at is None else first_significant_at)
    with col_final:
        st.metric("Final Significant", "Yes" if result["final_significant"] else "No")

    if result["peeking_risk"] == "high":
        st.error(result["false_positive_risk_note"])
    elif result["peeking_risk"] == "medium":
        st.warning(result["false_positive_risk_note"])
    else:
        st.info(result["false_positive_risk_note"])


# TODO Rename this here and in `render_pitfall_detection`
def _extracted_from_render_pitfall_detection_14(df):
    st.write("Preview")
    st.dataframe(df.head(10), width="stretch")

    columns = list(df.columns)
    treatment_col = st.selectbox("Treatment column", columns, key="pitfall_treatment")
    outcome_col = st.selectbox("Outcome column", columns, key="pitfall_outcome")
    confounder_col = st.selectbox("Confounder column", columns, key="pitfall_confounder")

    if st.button("Detect Simpson's Paradox", type="primary"):
        try:
            result = detect_simpsons_paradox(df, treatment_col, outcome_col, confounder_col)
            _render_paradox_results(result)
        except Exception as error:
            st.error(f"Could not analyze dataset: {error}")


# TODO Rename this here and in `render_pitfall_detection`
def _extracted_from_render_pitfall_detection_29(arg0, arg1, key):
    st.divider()
    st.subheader(arg0)
    return st.number_input(
        arg1,
        min_value=MIN_PROBABILITY,
        max_value=MAX_PROBABILITY,
        value=DEFAULT_ALPHA,
        step=0.001,
        format="%.3f",
        key=key,
    )


def render_experiment_planning():
    st.title(APP_TITLE)
    st.caption("Experiment Planning")
    st.subheader("Sample Size Calculator")
    render_sample_size_calculator(show_header=False, key_suffix="planning")
    st.divider()
    st.subheader("Power Analysis")
    render_power_analysis(show_header=False)


def render_statistical_analysis():
    st.title(APP_TITLE)
    st.caption("Statistical Analysis")
    st.subheader("Statistical Tests")
    render_statistical_tests(show_header=False)
    st.divider()
    st.subheader("P-value Distribution")
    render_pvalue_distribution(show_header=False)


def render_business_impact():
    st.title(APP_TITLE)
    st.caption("Business Impact")
    st.subheader("ROI Calculator")

    col_left, col_right = st.columns(2)
    with col_left:
        baseline_rate = st.slider(
            "Baseline conversion rate",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            format="%.2f",
        )
        lift_percent = st.slider(
            "Expected lift (%)",
            min_value=-50.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            format="%.1f",
        )
        experiment_days = st.slider(
            "Experiment length (days)",
            min_value=1,
            max_value=90,
            value=14,
            step=1,
        )
    with col_right:
        daily_traffic = st.number_input("Daily traffic", min_value=0, value=5000, step=100)
        revenue_per_conversion = st.number_input(
            "Revenue per conversion",
            min_value=0.0,
            value=50.0,
            step=5.0,
            format="%.2f",
        )

    try:
        roi_result = calculate_roi(
            baseline_rate=baseline_rate,
            lift_percent=lift_percent,
            daily_traffic=daily_traffic,
            revenue_per_conversion=revenue_per_conversion,
            experiment_days=experiment_days,
        )
        revenue_projection = project_revenue_series(
            baseline_rate=baseline_rate,
            lift_percent=lift_percent,
            daily_traffic=daily_traffic,
            revenue_per_conversion=revenue_per_conversion,
            experiment_days=experiment_days,
        )
    except ValueError as error:
        st.error(f"Invalid input: {error}")
        return

    col_conv, col_revenue, col_annualized = st.columns(3)
    with col_conv:
        st.metric("Incremental Conversions", f"{roi_result['incremental_conversions']:.1f}")
    with col_revenue:
        st.metric("Incremental Revenue", f"${roi_result['incremental_revenue']:,.2f}")
    with col_annualized:
        st.metric("Annualized Revenue", f"${roi_result['annualized_revenue']:,.2f}")

    st.plotly_chart(_build_revenue_chart(revenue_projection), width="stretch")
    st.info(roi_result["experiment_cost_note"])

    st.divider()
    st.subheader("Executive Report Generator")
    latest_stat_result = st.session_state.get("latest_stat_result", {})
    default_start = date.today() - timedelta(days=int(experiment_days))
    default_end = date.today()
    confidence_interval = latest_stat_result.get("confidence_interval")

    report_test_name = st.text_input("Test name", value="Homepage CTA Test")
    col_dates, col_sample = st.columns(2)
    with col_dates:
        report_start_date = st.date_input("Start date", value=default_start)
        report_end_date = st.date_input("End date", value=default_end)
    with col_sample:
        report_sample_size = st.number_input(
            "Sample size",
            min_value=1,
            value=max(1, int(latest_stat_result.get("sample_size", daily_traffic * experiment_days))),
            step=100,
        )
        report_alpha = st.number_input(
            "Report alpha",
            min_value=MIN_PROBABILITY,
            max_value=MAX_PROBABILITY,
            value=float(latest_stat_result.get("alpha", DEFAULT_ALPHA)),
            step=0.001,
            format="%.3f",
        )

    col_stats, col_ci = st.columns(2)
    with col_stats:
        report_p_value = st.number_input(
            "P-value",
            min_value=0.0,
            max_value=1.0,
            value=float(latest_stat_result.get("p_value", DEFAULT_ALPHA)),
            step=0.001,
            format="%.4f",
        )
        report_significant = report_p_value < report_alpha
        st.metric("Statistical result", "Significant" if report_significant else "Not significant")
    with col_ci:
        report_ci_lower = st.number_input(
            "Confidence interval lower",
            value=float(confidence_interval[0]) if confidence_interval else 0.0,
            step=0.001,
            format="%.4f",
        )
        report_ci_upper = st.number_input(
            "Confidence interval upper",
            value=float(confidence_interval[1]) if confidence_interval else 0.0,
            step=0.001,
            format="%.4f",
        )

    recommendation = recommend_experiment_action(
        significant=report_significant,
        lift_percent=lift_percent,
        incremental_revenue=roi_result["incremental_revenue"],
    )
    st.write(f"Recommended action based on the current inputs: **{recommendation}**")

    if report_end_date < report_start_date:
        st.error("End date must be on or after the start date.")
    elif st.button("Generate Report", type="primary"):
        st.session_state["executive_report_markdown"] = generate_executive_report(
            test_name=report_test_name,
            start_date=report_start_date,
            end_date=report_end_date,
            sample_size=int(report_sample_size),
            significant=report_significant,
            p_value=report_p_value,
            confidence_interval=(report_ci_lower, report_ci_upper),
            incremental_revenue=roi_result["incremental_revenue"],
            lift_percent=lift_percent,
            recommendation=recommendation,
        )

    if report_markdown := st.session_state.get("executive_report_markdown"):
        st.markdown(report_markdown)
        st.code(report_markdown, language="markdown")


PAGE_ROUTER = {
    "Experiment Planning": render_experiment_planning,
    "Statistical Analysis": render_statistical_analysis,
    "Pitfall Detection": render_pitfall_detection,
    "Business Impact": render_business_impact,
}


def render_sidebar():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Select page", options=list(PAGE_ROUTER.keys()), index=0)


def main():
    page = render_sidebar()
    if render_fn := PAGE_ROUTER.get(page):
        render_fn()


if __name__ == "__main__":
    main()
