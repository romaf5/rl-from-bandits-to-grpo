import numpy as np
import pytest

import part16_beyond_grpo as p16


def test_part16_seq_mean_makes_long_answers_cheaper_per_token():
    w = p16.token_weights([10, 40], "seq"); assert w[1] < w[0]


def test_part16_token_level_is_uniform():
    w = p16.token_weights([10, 40], "token"); assert w[0] == w[1] == pytest.approx(1 / 50)


def test_part16_gspo_is_geometric_mean():
    assert p16.gspo_ratio(np.log([1.1, 0.9, 1.0])) == pytest.approx((1.1 * 0.9 * 1.0) ** (1 / 3))


def test_part16_dapo_clip_higher():
    assert p16.clip_bounds("DAPO")[1] > p16.clip_bounds("GRPO")[1]
