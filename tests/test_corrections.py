import math
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corrections import apply_benjamini_hochberg, apply_bonferroni


def test_bonferroni_correction_flags_only_values_below_adjusted_threshold():
    results = apply_bonferroni([0.01, 0.02, 0.20], alpha=0.05)

    assert [entry["significant"] for entry in results] == [True, False, False]
    assert math.isclose(results[0]["corrected_p"], 0.03, rel_tol=1e-9)
    assert math.isclose(results[1]["corrected_p"], 0.06, rel_tol=1e-9)
    assert math.isclose(results[2]["corrected_p"], 0.60, rel_tol=1e-9)


def test_benjamini_hochberg_returns_monotonic_adjusted_pvalues_in_original_order():
    results = apply_benjamini_hochberg([0.20, 0.01, 0.04, 0.02], alpha=0.05)

    assert [entry["significant"] for entry in results] == [False, True, False, True]
    assert math.isclose(results[0]["corrected_p"], 0.20, rel_tol=1e-9)
    assert math.isclose(results[1]["corrected_p"], 0.04, rel_tol=1e-9)
    assert math.isclose(results[2]["corrected_p"], 0.05333333333333334, rel_tol=1e-9)
    assert math.isclose(results[3]["corrected_p"], 0.04, rel_tol=1e-9)
