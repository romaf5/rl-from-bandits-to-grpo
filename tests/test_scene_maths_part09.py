import numpy as np
import pytest

import part09_ppo as p09


def test_part09_positive_advantage_flat_above_1_plus_eps():
    assert p09.clip_grad(1.3, 1.0) == pytest.approx(0) and p09.clip_grad(1.1, 1.0) == pytest.approx(1.0)


def test_part09_negative_advantage_flat_below_1_minus_eps():
    assert p09.clip_grad(0.7, -1.0) == pytest.approx(0) and p09.clip_grad(0.9, -1.0) == pytest.approx(-1.0)


def test_part09_pessimistic_bound():
    r = np.linspace(0.5, 1.5, 11)
    assert np.all(p09.clip_objective(r, 1.0) <= r * 1.0 + 1e-12)
