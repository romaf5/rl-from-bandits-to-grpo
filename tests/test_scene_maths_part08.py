import numpy as np
import pytest

import part08_trpo as p08


def test_part08_same_param_step_different_kl():
    assert p08.kl_bern(0.0, 1.0) > 5 * p08.kl_bern(4.0, 5.0)


def test_part08_quadratic_kl_model():
    d = 1e-2; assert p08.kl_bern(0.3, 0.3 + d) == pytest.approx(0.5 * p08.fisher(0.3) * d * d, rel=1e-2)


# The natural step hits max_kl only to second order (KL 0.00895 vs 0.01 at t = 2).
def test_part08_natural_step_hits_max_kl():
    t = 2.0; d = p08.natural_step(t, 1.0); assert p08.kl_bern(t, t + d) == pytest.approx(0.01, rel=0.15)
