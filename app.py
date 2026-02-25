import numpy as np
import plotly.graph_objects as go
import streamlit as st
from src.sample_size import (
    calculate_sample_size_proportion,
    calculate_sample_size_continuous,
    calculate_power,
    DEFAULT_ALPHA,
    DEFAULT_POWER,
)

# ---- Page Config ----

st.set_page_config(
    page_title="A/B Test Sample Size Calculator",
    page_icon="📊",
    layout="centered",
)

# ---- UI Constants ----

TEST_TYPE_PROPORTION = "Conversion Rate"
TEST_TYPE_CONTINUOUS = "Continuous Metric"
MDE_TYPE_ABSOLUTE = "Absolute"
MDE_TYPE_RELATIVE = "Relative"

MIN_PROBABILITY = 0.001
MAX_PROBABILITY = 0.999
MIN_MDE = 0.01
MIN_STD_DEV = 0.01
MIN_POWER = 0.70

POWER_CURVE_MDE_MIN = 0.005
POWER_CURVE_MDE_MAX = 0.10
POWER_CURVE_POINTS = 50
TARGET_POWER = 0.80


# ==== Input Widgets ====


def render_test_type_selector():
    return st.radio(
        "Test Type",
        options=[TEST_TYPE_PROPORTION, TEST_TYPE_CONTINUOUS],
        horizontal=True,
        help=(
            "Conversion Rate: binary outcomes (click / no-click). "
            "Continuous Metric: numeric outcomes (revenue, time spent, etc.)."
        ),
    )


def render_mde_type_selector():
    return st.radio(
        "MDE Type",
        options=[MDE_TYPE_ABSOLUTE, MDE_TYPE_RELATIVE],
        horizontal=True,
        help=(
            "Absolute: fixed percentage point change. "
            "Relative: proportional lift over baseline. E.g., 0.05 = 5% lift."
        ),
    )


def render_common_inputs():
    col_alpha, col_power = st.columns(2)

    with col_alpha:
        alpha = st.number_input(
            "Significance Level (α)",
            min_value=MIN_PROBABILITY,
            max_value=MAX_PROBABILITY,
            value=DEFAULT_ALPHA,
            step=0.001,
            format="%.3f",
            help="Probability of false positive (Type I error). Commonly 0.05.",
        )
    with col_power:
        power = st.number_input(
            "Statistical Power (1 - β)",
            min_value=MIN_POWER,
            max_value=MAX_PROBABILITY,
            value=DEFAULT_POWER,
            step=0.05,
            format="%.2f",
            help="Probability of detecting a real effect. Standard: 0.80.",
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
            help="Current conversion rate of your control group.",
        )
    with col_mde:
        mde = st.number_input(
            f"Minimum Detectable Effect ({'relative' if is_relative else 'absolute'})",
            min_value=MIN_MDE,
            max_value=0.50 if is_relative else MAX_PROBABILITY,
            value=0.05 if is_relative else 0.02,
            step=0.005,
            format="%.3f",
            help=(
                "Relative lift over baseline. E.g., 0.05 = 5% improvement."
                if is_relative
                else "Absolute change in rate. E.g., 0.02 = 2 percentage points."
            ),
        )

    return baseline_rate, mde


def render_continuous_inputs():
    col_baseline, col_mde, col_std = st.columns(3)

    with col_baseline:
        baseline_mean = st.number_input(
            "Baseline Mean",
            value=50.0,
            step=1.0,
            format="%.2f",
            help="Current average of your control metric.",
        )
    with col_mde:
        mde = st.number_input(
            "Minimum Detectable Effect (absolute)",
            min_value=MIN_MDE,
            value=5.0,
            step=0.5,
            format="%.2f",
            help="Smallest absolute difference worth detecting.",
        )
    with col_std:
        std_dev = st.number_input(
            "Standard Deviation (σ)",
            min_value=MIN_STD_DEV,
            value=20.0,
            step=1.0,
            format="%.2f",
            help="Expected standard deviation of the metric.",
        )

    return baseline_mean, mde, std_dev


# ---- Result Display ----


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
        treatment_rate = (
            baseline_rate * (1 + mde)
            if mde_type_key == "relative"
            else baseline_rate + mde
        )
        display_results(
            per_group,
            f"You need **{per_group:,} users per group** "
            f"(**{per_group * 2:,} total**) to detect a change "
            f"from **{baseline_rate:.1%}** to **{treatment_rate:.1%}** "
            f"with **{power:.0%} power** at **α = {alpha}**.",
        )
    except ValueError as error:
        st.error(f"Invalid input: {error}")


def run_continuous_calculation(
    baseline_mean, mde, std_dev, alpha, power, mde_type_key
):
    try:
        per_group = calculate_sample_size_continuous(
            baseline_mean=baseline_mean,
            minimum_detectable_effect=mde,
            std_dev=std_dev,
            alpha=alpha,
            power=power,
            mde_type=mde_type_key,
        )
        treatment_mean = (
            baseline_mean * (1 + mde)
            if mde_type_key == "relative"
            else baseline_mean + mde
        )
        display_results(
            per_group,
            f"You need **{per_group:,} observations per group** "
            f"(**{per_group * 2:,} total**) to detect a mean shift "
            f"from **{baseline_mean:.2f}** to **{treatment_mean:.2f}** "
            f"(σ = {std_dev:.2f}) with **{power:.0%} power** at **α = {alpha}**.",
        )
    except ValueError as error:
        st.error(f"Invalid input: {error}")


# === Pages ===


def render_sample_size_calculator():
    st.title("🧪 A/B Testing Intelligence Platform")
    st.caption("Sample Size Calculator")

    test_type = render_test_type_selector()
    mde_type_label = render_mde_type_selector()
    mde_type_key = mde_type_label.lower()
    alpha, power = render_common_inputs()

    if test_type == TEST_TYPE_PROPORTION:
        baseline_rate, mde = render_proportion_inputs(mde_type_label)
        run_proportion_calculation(
            baseline_rate, mde, alpha, power, mde_type_key
        )
    elif test_type == TEST_TYPE_CONTINUOUS:
        baseline_mean, mde, std_dev = render_continuous_inputs()
        run_continuous_calculation(
            baseline_mean, mde, std_dev, alpha, power, mde_type_key
        )
    with st.expander("📚 Understanding these numbers"):
        st.markdown("""
    **Sample size increases when:**
    - You want to detect smaller effects (lower MDE)
    - Baseline rate is closer to 0% or 100% (more variance)
    - You demand higher power (lower β)
    - You use stricter significance (lower α)
    
    **Real-world context:**
    - Most A/B tests detect 2-10% relative lifts
    - Detecting a 2% lift requires ~4x more users than a 4% lift
    - Industry standard: 80% power, 5% significance
    """)
    with st.expander("📊 Compare Multiple Scenarios"):
        mde_options = st.multiselect(
        "Select MDE values",
        [0.01, 0.02, 0.03, 0.05, 0.10],
        default=[0.02, 0.05]
    )
    
    if mde_options and test_type == TEST_TYPE_PROPORTION:
        results = []
        for m in mde_options:
            n = calculate_sample_size_proportion(baseline_rate, m, alpha, power, mde_type_key)
            results.append({"MDE": m, "Per Group": n, "Total": n*2})
        st.dataframe(results)


def render_power_analysis():
    st.title("🧪 A/B Testing Intelligence Platform")
    st.caption("Power Analysis")
    st.write(
        "How does achieved power change as a function of MDE "
        "at a fixed sample size?"
    )

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
        value=3_839,
        step=100,
        help="Number of users in each group (control and treatment).",
    )
    alpha = st.number_input(
        "Significance Level (α)",
        min_value=MIN_PROBABILITY,
        max_value=MAX_PROBABILITY,
        value=DEFAULT_ALPHA,
        step=0.001,
        format="%.3f",
        help="Probability of false positive (Type I error). Commonly 0.05.",
    )

    mde_range = np.linspace(
        POWER_CURVE_MDE_MIN, POWER_CURVE_MDE_MAX, POWER_CURVE_POINTS
    )

    try:
        powers = [
            calculate_power(baseline, mde, sample_size_per_group, alpha)
            for mde in mde_range
        ]
    except ValueError as error:
        st.error(f"Invalid input: {error}")
        return

    _render_power_curve(mde_range, powers)
    _render_crossover_callout(mde_range, powers)


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
        annotation_text=f"{int(TARGET_POWER * 100)}% Target",
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
    """Surface the MDE at which power first crosses the target threshold."""
    crossover = next(
        (mde for mde, pwr in zip(mde_range, powers) if pwr >= TARGET_POWER),
        None,
    )
    if crossover is None:
        st.warning(
            "Power never reaches 80% in this MDE range. "
            "Increase your sample size."
        )
        return

    st.info(
        f"At this sample size, you reach **{int(TARGET_POWER * 100)}% power** "
        f"at an MDE of **{crossover:.3f}** (absolute)."
    )


# === Router — single source of truth for pages ===

PAGE_ROUTER = {
    "Sample Size Calculator": render_sample_size_calculator,
    "Power Analysis": render_power_analysis,
    # "Simulations": render_simulations,       # Week 2
    # "Pitfall Detector": render_pitfalls,     # Week 3
    # "Business Impact": render_business_impact, # Week 4
}


def render_sidebar():
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Select tool",
        options=list(PAGE_ROUTER.keys()),
        index=0,
    )


def main():
    page = render_sidebar()
    if render_fn := PAGE_ROUTER.get(page):
        render_fn()


if __name__ == "__main__":
    main()