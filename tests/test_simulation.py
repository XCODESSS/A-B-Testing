import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation import simulate_pvalue_distribution


SIMULATION_CASES = (
    ("T-test", 200),
    ("Chi-square", 500),
    ("Mann-Whitney", 200),
)


def test_null_effect_produces_nearly_uniform_pvalues():
    for test_type, sample_size in SIMULATION_CASES:
        p_values = simulate_pvalue_distribution(
            n_simulations=2500,
            sample_size=sample_size,
            effect_size=0.0,
            test_type=test_type,
            random_state=123,
        )
        assert p_values.shape == (2500,)
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)
        assert 0.45 <= float(np.mean(p_values)) <= 0.55

        for cutoff in (0.25, 0.50, 0.75):
            observed_share = float(np.mean(p_values <= cutoff))
            assert abs(observed_share - cutoff) <= 0.05

        assert 0.03 <= float(np.mean(p_values < 0.05)) <= 0.08


def test_effect_size_pushes_pvalues_toward_zero():
    for test_type, sample_size in SIMULATION_CASES:
        null_p = simulate_pvalue_distribution(
            n_simulations=2500,
            sample_size=sample_size,
            effect_size=0.0,
            test_type=test_type,
            random_state=123,
        )
        effect_p = simulate_pvalue_distribution(
            n_simulations=2500,
            sample_size=sample_size,
            effect_size=0.2,
            test_type=test_type,
            random_state=123,
        )

        null_rate = float(np.mean(null_p < 0.05))
        effect_rate = float(np.mean(effect_p < 0.05))

        assert effect_rate > null_rate + 0.20
        assert float(np.quantile(effect_p, 0.75)) < 0.25
