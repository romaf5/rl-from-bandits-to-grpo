import numpy as np
import pytest

import part14_rloo as p14


def test_part14_rloo_example():
    assert p14.rloo_advantages([1, 0, 0, 0]) == pytest.approx([1, -1 / 3, -1 / 3, -1 / 3])


def test_part14_rloo_sums_to_zero():
    assert p14.rloo_advantages([0.3, 0.9, 0.1, 0.5]).sum() == pytest.approx(0)


def test_part14_batch_norm_is_global():
    a = p14.batch_normalised([1, 0, 1, 1, 0, 0, 0, 0])
    assert a.mean() == pytest.approx(0, abs=1e-9) and a.std() == pytest.approx(1)
