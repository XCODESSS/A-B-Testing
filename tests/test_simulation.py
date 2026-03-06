import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation import simulate_pvalue_distribution


def test_pvalues_are_bounded_for_all_test_types():

    for test_type in ("T-test", "Chi-square", "Mann-Whitney"):
        p_values = simulate_pvalue_distribution(
            n_simulations=200,
            sample_size=80,
            effect_size=0.0,
            test_type=test_type,
            random_state=42,
        )
        assert p_values.shape == (200,)
        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)


def test_effect_increases_rejection_rate_for_t_test():
    null_p = simulate_pvalue_distribution(
        n_simulations=1000,
        sample_size=120,
        effect_size=0.0,
        test_type="T-test",
        random_state=123,
    )
    effect_p = simulate_pvalue_distribution(
        n_simulations=1000,
        sample_size=120,
        effect_size=0.35,
        test_type="T-test",
        random_state=123,
    )

    null_rate = float(np.mean(null_p < 0.05))
    effect_rate = float(np.mean(effect_p < 0.05))

    assert 0.02 <= null_rate <= 0.08
    assert effect_rate > null_rate + 0.20
