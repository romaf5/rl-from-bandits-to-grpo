import numpy as np
import pytest

import part10_llm as p10


def test_part10_reward_is_sparse_and_terminal():
    r = p10.terminal_rewards(6, 1.0); assert r.sum() == 1.0 and r[-1] == 1.0 and np.count_nonzero(r) == 1


def test_part10_mapping_has_six_rows():
    assert len(p10.MAPPING) == 6


def test_part10_tree_branches_sum_to_one():
    for prefix, branches in p10.NEXT_TOKEN_PROBS.items():
        assert sum(p for _, p in branches) == pytest.approx(1.0), prefix
