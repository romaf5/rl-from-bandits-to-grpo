import numpy as np
import pytest

import part13_dpo as p13


def test_part13_zero_margin_is_log2():
    assert p13.dpo_loss(-5, -5, -5, -5) == pytest.approx(np.log(2))


def test_part13_raising_chosen_lowers_loss():
    assert p13.dpo_loss(-4, -5, -5, -5) < p13.dpo_loss(-5, -5, -5, -5) < p13.dpo_loss(-5, -4, -5, -5)
