import math
import pathlib
import sys

import numpy as np
from scipy import stats

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stat_tests import chi_square_test, mann_whitney_test, t_test


def test_t_test_matches_scipy_welch():
    group_a = np.array([12.1, 11.3, 13.4, 10.9, 12.7, 11.8])
    group_b = np.array([10.1, 9.4, 11.2, 10.8, 9.9, 10.5, 9.8])

    result = t_test(group_a, group_b, alpha=0.05)
    scipy_result = stats.ttest_ind(group_a, group_b, equal_var=False)

    assert math.isclose(result["statistic"], float(scipy_result.statistic), rel_tol=1e-9)
    assert math.isclose(result["p_value"], float(scipy_result.pvalue), rel_tol=1e-9)
    assert result["significant"] is (result["p_value"] < 0.05)
    assert isinstance(result["confidence_interval"], tuple)
    assert len(result["confidence_interval"]) == 2


def test_chi_square_matches_scipy():
    contingency = np.array([[35, 65], [50, 50]])

    result = chi_square_test(contingency, alpha=0.05)
    statistic, p_value, _, _ = stats.chi2_contingency(
        contingency, correction=False
    )

    assert math.isclose(result["statistic"], float(statistic), rel_tol=1e-9)
    assert math.isclose(result["p_value"], float(p_value), rel_tol=1e-9)
    assert result["significant"] is (result["p_value"] < 0.05)


def test_mann_whitney_matches_scipy():
    group_a = np.array([1.2, 2.1, 2.4, 3.8, 4.0])
    group_b = np.array([0.5, 1.1, 1.7, 2.0, 2.2, 2.9])

    result = mann_whitney_test(group_a, group_b, alpha=0.05)
    scipy_result = stats.mannwhitneyu(
        group_a, group_b, alternative="two-sided", method="auto"
    )

    assert math.isclose(result["statistic"], float(scipy_result.statistic), rel_tol=1e-9)
    assert math.isclose(result["p_value"], float(scipy_result.pvalue), rel_tol=1e-9)
    assert result["significant"] is (result["p_value"] < 0.05)
