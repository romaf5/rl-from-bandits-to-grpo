import numpy as np
import pytest

import part11_reward_model as p11


def test_part11_bt_half_at_equal_rewards():
    assert p11.bt_prob(1.0, 1.0) == pytest.approx(0.5) and p11.bt_loss(0, 0) == pytest.approx(np.log(2))


def test_part11_goodhart_proxy_rises_true_peaks_inside():
    ns, px, tr = p11.best_of_n_curves()
    assert np.all(np.diff(px) > 0) and 0 < int(np.argmax(tr)) < len(ns) - 1
