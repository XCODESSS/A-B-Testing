import streamlit as st 
from src.sample_size import (
    calculate_sample_size_proportion,
    calculate_sample_size_continuous,
    DEFAULT_ALPHA,
    DEFAULT_POWER
)
 
#----Page Config----
st.set_page_config(
    page_title="A/B Test Sample Size Calculator",
    page_icon="📊",
    layout="centered",
)

#----UI Constants----

TEST_TYPE_PROPORTION = "Coversion Rate"
TEST_TYPE_CONTINUOUS = "Continuous Metric"
MIN_PROBABILITY = 0.001
MAX_PROBABILITY = 0.999
MIN_MDE = 0.01
MIN_STD_DEV = 0.01
MIN_POWER = 0.5

#----Page Registry----

PAGES = {
    "Sample Size Calculator"
}

#====Sidebar====

def render_sidebar():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Select tool", options= PAGES, index = 0)

#==== Input Widgets====

def render_test_type_selector():
    return st.radio(
        "Test Type",
        options=[TEST_TYPE_PROPORTION, TEST_TYPE_CONTINUOUS],
        horizontal=True,
        help="Conversion Rate: binary outcomes(click/no-click). Continuous Metric: numeric outcomes (revenue, time spent, etc.)."
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
            help="Probability of false positive (Type I error). Commonly set at 0.05."
        
        )
    with col_power:
        power = st.number_input(
            "Statistical Power (1 - β)",
            min_value=MIN_POWER,
            max_value=MAX_PROBABILITY,
            value=DEFAULT_POWER,
            step=0.05,
            format="%.2f",
            help="Probability of detecting a real effect."
            "Standard: 0.8." 
        )
    return alpha, power

def render_proportion_inputs():
    col_baseline, col_mde = st.columns(2)
    with col_baseline:
        baseline_rate = st.number_input(
            "Baseline Converion Rate",
            min_value=MIN_PROBABILITY,
            max_value=MAX_PROBABILITY,
            value=0.10,
            step=0.01,
            format="%.3f",
            help="Current conversion rate of your control group"
        )
    with col_mde:
        mde = st.number_input(
            "Minimum Detectable Effect (absolute)",
            min_value=MIN_MDE,
            max_value=MAX_PROBABILITY,
            value=0.02,
            step=0.005,
            format="%.3f",
            help="Smallest change in conversion rate you want to detect (e.g., 0.02 = 2 % points)."
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

#----Result Display----

def display_results(per_group, summary_text):
    """Shared display logic for both test types"""
    total = per_group * 2

    st.divider()

    col_per, col_total = st.columns(2)

    with col_per:
        st.metric("Per Group", f"{per_group:,}")
    with col_total:
        st.metric("Total", f"{total:,}")
    
    st.info(summary_text)

def run_proportion_test(baseline_rate, mde, alpha, power):
    try:
        per_group = calculate_sample_size_proportion(
            baseline_rate=baseline_rate,
            minimum_detectable_effect=mde,
            alpha=alpha,
            power=power,
        )
        treatment_rate = baseline_rate + mde

        display_results(
            per_group,
            f"You need **{per_group:,} users per group** "
            f"(**{per_group * 2:,} total**) to detect a change "
            f"from **{baseline_rate:.1%}** to "
            f"**{treatment_rate:.1%}** with "
            f"**{power:.0%} power** at **α = {alpha}**.",
        )

    except ValueError as error:
        st.error(f"Invalid input: {error}")


def run_continuous_calculation(
    baseline_mean, mde, std_dev, alpha, power
):
    try:
        per_group = calculate_sample_size_continuous(
            baseline_mean=baseline_mean,
            minimum_detectable_effect=mde,
            std_dev=std_dev,
            alpha=alpha,
            power=power,
        )
        treatment_mean = baseline_mean + mde

        display_results(
            per_group,
            f"You need **{per_group:,} observations per group** "
            f"(**{per_group * 2:,} total**) to detect a mean "
            f"shift from **{baseline_mean:.2f}** to "
            f"**{treatment_mean:.2f}** (σ = {std_dev:.2f}) with "
            f"**{power:.0%} power** at **α = {alpha}**.",
        )

    except ValueError as error:
        st.error(f"Invalid input: {error}")


# === Pages ===


def render_sample_size_calculator():
    st.title("🧪 A/B Testing Intelligence Platform")
    st.caption("Sample Size Calculator")

    test_type = render_test_type_selector()
    alpha, power = render_common_inputs()

    if test_type == TEST_TYPE_PROPORTION:
        baseline_rate, mde = render_proportion_inputs()
        run_proportion_test(baseline_rate, mde, alpha, power)

    elif test_type == TEST_TYPE_CONTINUOUS:
        baseline_mean, mde, std_dev = render_continuous_inputs()
        run_continuous_calculation(
            baseline_mean, mde, std_dev, alpha, power
        )
# === Router ===

PAGE_ROUTER = {
    "Sample Size Calculator": render_sample_size_calculator,
    # "Statistical Tests": render_stat_tests,        # Week 2
    # "Simulations": render_simulations,              # Week 2
    # "Pitfall Detector": render_pitfalls,            # Week 3
    # "Business Impact": render_business_impact,      # Week 4
}


def main():
    page = render_sidebar()
    if render_fn := PAGE_ROUTER.get(page):
        render_fn()


if __name__ == "__main__":
    main()