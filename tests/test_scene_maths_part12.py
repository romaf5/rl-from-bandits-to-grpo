import numpy as np
import pytest

import part12_ppo_llm as p12


def test_part12_kl_penalty_every_token_reward_at_end():
    r = p12.per_token_rewards([-1, -1, -1], [-1.5, -1.0, -2.0], R=1.0, beta=0.1)
    assert r == pytest.approx([-0.05, 0.0, 1.0 - 0.1])


def test_part12_gae_lambda1_is_return_minus_value():
    r = np.array([0.0, 0.0, 1.0]); v = np.array([0.2, 0.5, 0.7])
    assert np.allclose(p12.gae_tokens(r, v, lam=1.0), r[::-1].cumsum()[::-1] - v)
