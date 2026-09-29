import numpy as np
import pytest

import part06_reinforce as p06


def test_part06_unbiased():
    th = np.zeros(3)
    assert np.allclose(p06.reinforce_samples(th, n=40000).mean(0), p06.true_grad(th), atol=0.03)


def test_part06_baseline_reduces_variance():
    th = np.zeros(3); b = p06.softmax(th) @ p06.Q_TRUE
    v0 = p06.reinforce_samples(th, baseline=0).var(0).sum(); v1 = p06.reinforce_samples(th, baseline=b).var(0).sum()
    assert v1 < v0
