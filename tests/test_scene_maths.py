import numpy as np
import pytest

import part15_grpo as p15


def test_part15_lone_correct_gets_2_47():
    a = p15.group_adv(p15.rewards_with(1))
    assert a[0] == pytest.approx(2.474, abs=1e-2) and a[-1] == pytest.approx(-0.3535, abs=1e-2)


def test_part15_mean_only_is_0_875():
    assert p15.group_adv(p15.rewards_with(1), normalize=False)[0] == pytest.approx(0.875)


def test_part15_dead_group_is_zero():
    assert np.allclose(p15.group_adv(p15.rewards_with(8)), 0)


import part02_bandits as p02


def test_part02_incremental_mean_equals_sample_average():
    q = p02.make_bandit(); h = p02.run_agent(q, "eps", steps=200)
    a = h["a"][-1]
    assert h["Q"][-1][a] == pytest.approx(h["r"][h["a"] == a].mean())


def test_part02_ucb_bonus_infinite_for_untried():
    b = p02.ucb_bonus(5, np.array([0, 2.0]))
    assert np.isinf(b[0]) and b[1] == pytest.approx(2 * np.sqrt(np.log(5) / 2))


def test_part02_eps_greedy_beats_greedy_on_this_seed():
    q = p02.make_bandit()
    g = p02.run_agent(q, "greedy", steps=300); e = p02.run_agent(q, "eps", steps=300)
    assert (e["a"][-100:] == q.argmax()).mean() > (g["a"][-100:] == q.argmax()).mean()


import part01_foundations as p01


def test_part01_grid_has_22_states_and_stochastic_rows():
    cells, P, R, term = p01.gridworld()
    assert len(cells) == 22 and np.allclose(P.sum(-1), 1)  # 25 cells - 3 walls (brief said 21)


def test_part01_discounted_return():
    assert p01.discounted_return([1, 1, 1], 0.5) == pytest.approx(1.75)


def test_part01_far_sighted_start_value_higher():
    cells, *_ = p01.gridworld(); s0 = cells.index((0, 0))
    assert p01.optimal_values(0.99)[s0] > p01.optimal_values(0.5)[s0]


import part03_dp as p03


def test_part03_error_contracts_by_gamma():
    _, H = p03.value_iteration_history(0.9, 200)
    err = np.abs(H - H[-1]).max(1)
    for k in range(30):
        assert err[k + 1] <= 0.9 * err[k] + 1e-9


def test_part03_policy_from_goal_neighbour_points_to_goal():
    cells, H = p03.value_iteration_history()
    pi = p03.greedy_policy(H[-1])
    assert pi[cells.index((4, 3))] == 1          # right, into G


import part04_tabular as p04


def test_part04_q_learning_takes_the_edge():
    path = p04.greedy_path(p04.train("q"))
    assert path[-1] == p04.GOAL and len(path) == 14      # 13 moves: up, 11 right, down


def test_part04_sarsa_takes_a_longer_safer_path():
    q_path = p04.greedy_path(p04.train("q")); s_path = p04.greedy_path(p04.train("sarsa"))
    assert s_path[-1] == p04.GOAL and len(s_path) > len(q_path)


def test_part04_td_error():
    assert p04.td_error(1.0, 2.0, 0.5, gamma=0.9) == pytest.approx(2.3)


import part05_dqn as p05


def test_part05_deadly_triad_diverges():
    assert abs(p05.two_state_divergence(0.9)[-1]) > 10 and abs(p05.two_state_divergence(0.4)[-1]) < 1


def test_part05_frozen_target_is_piecewise_constant():
    _, ts = p05.chase(10)
    assert len(set(np.round(ts[:10], 9))) == 1
