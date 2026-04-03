import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.peeking_detector import detect_peeking, find_significance_crossings


def test_detect_peeking_returns_low_risk_when_only_final_check_is_significant():
    result = detect_peeking([0.21, 0.12, 0.07, 0.04], alpha=0.05)

    assert result["peeking_risk"] == "low"
    assert result["first_significant_at"] == 3
    assert result["final_significant"] is True
    assert "cleanest pattern" in result["false_positive_risk_note"]


def test_detect_peeking_returns_medium_risk_for_single_interim_crossing():
    result = detect_peeking([0.18, 0.03, 0.02], alpha=0.05)

    assert result["peeking_risk"] == "medium"
    assert result["first_significant_at"] == 1
    assert result["final_significant"] is True
    assert "interim check 1" in result["false_positive_risk_note"]


def test_detect_peeking_returns_high_risk_for_multiple_crossings():
    pvalues = [0.20, 0.03, 0.10, 0.04, 0.07]
    result = detect_peeking(pvalues, alpha=0.05)

    assert result["peeking_risk"] == "high"
    assert result["first_significant_at"] == 1
    assert result["final_significant"] is False
    assert find_significance_crossings(pvalues, alpha=0.05) == [1, 3]


def test_detect_peeking_validates_inputs():
    with pytest.raises(ValueError):
        detect_peeking([], alpha=0.05)

    with pytest.raises(ValueError):
        detect_peeking([0.10, 1.20], alpha=0.05)

    with pytest.raises(ValueError):
        detect_peeking([0.10, 0.02], alpha=1.0)
