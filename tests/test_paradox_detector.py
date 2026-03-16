import pathlib
import sys

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paradox_detector import detect_simpsons_paradox


def _build_rows(group, converted, segment, count):
    return [
        {"variant": group, "converted": converted, "segment": segment}
        for _ in range(count)
    ]


def test_detects_simpsons_paradox_when_aggregate_reverses_subgroups():
    rows = []
    rows += _build_rows("treatment", 1, "easy", 81)
    rows += _build_rows("treatment", 0, "easy", 6)
    rows += _build_rows("control", 1, "easy", 234)
    rows += _build_rows("control", 0, "easy", 36)
    rows += _build_rows("treatment", 1, "hard", 192)
    rows += _build_rows("treatment", 0, "hard", 71)
    rows += _build_rows("control", 1, "hard", 55)
    rows += _build_rows("control", 0, "hard", 25)

    df = pd.DataFrame(rows)
    result = detect_simpsons_paradox(df, "variant", "converted", "segment")

    assert result["paradox_detected"] is True
    assert result["aggregate_result"]["winner"] == "control"
    assert [group["winner"] for group in result["subgroup_results"]] == ["treatment", "treatment"]
    assert "Simpson's paradox" in result["reversal_explanation"]


def test_reports_clean_pattern_when_aggregate_matches_subgroups():
    rows = []
    rows += _build_rows("control", 1, "desktop", 40)
    rows += _build_rows("control", 0, "desktop", 60)
    rows += _build_rows("treatment", 1, "desktop", 48)
    rows += _build_rows("treatment", 0, "desktop", 52)
    rows += _build_rows("control", 1, "mobile", 20)
    rows += _build_rows("control", 0, "mobile", 80)
    rows += _build_rows("treatment", 1, "mobile", 24)
    rows += _build_rows("treatment", 0, "mobile", 76)

    df = pd.DataFrame(rows)
    result = detect_simpsons_paradox(df, "variant", "converted", "segment")

    assert result["paradox_detected"] is False
    assert result["aggregate_result"]["winner"] == "treatment"
    assert [group["winner"] for group in result["subgroup_results"]] == ["treatment", "treatment"]
