import numpy as np
import pytest

import part07_actor_critic as p07


def _ep():
    rng = np.random.default_rng(0); r = rng.random(8); v = rng.random(8)
    return r, v, np.append(v[1:], 0.0)


def test_part07_recursion_matches_direct_sum():
    r, v, vn = _ep(); assert np.allclose(p07.gae(r, v, vn), p07.gae_direct(r, v))


def test_part07_lambda_0_is_td_error_lambda_1_is_mc():
    r, v, vn = _ep()
    assert np.allclose(p07.gae(r, v, vn, lam=0.0), r + 0.99 * vn - v)
    G = np.array([sum(0.99 ** k * r[t + k] for k in range(8 - t)) for t in range(8)])
    assert np.allclose(p07.gae(r, v, vn, lam=1.0), G - v)
