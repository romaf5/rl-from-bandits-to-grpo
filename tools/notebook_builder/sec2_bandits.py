from common import md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 2 · Bandits: exploration versus exploitation

Strip the MDP down to a single state and you get the **k-armed bandit**: each step you pull one of $k$
arms and receive a noisy reward whose mean $q^*(a)$ you don't know. There is no state to worry about,
which isolates one core problem of RL: **should I pull the arm that looks best (exploit), or try
another one to learn more (explore)?**

We estimate action values by averaging observed rewards. The incremental form, which every later
algorithm reuses, is

$$Q_{n+1}(a) = Q_n(a) + \alpha\,\big[r_n - Q_n(a)\big], \qquad \alpha = \tfrac{1}{N(a)} \text{ (sample average) or a constant.}$$

*New estimate = old estimate + step size × (target − old estimate)*. Memorise that shape.

Three classic strategies:

* **ε-greedy**: with probability ε pick a random arm, otherwise the greedy one.
* **UCB** (upper confidence bound): pick $\arg\max_a\; Q(a) + c\sqrt{\ln t / N(a)}$. The bonus is large for
  arms we've tried rarely: *optimism in the face of uncertainty*.
* **Optimistic initialisation**: start all $Q(a)$ high, so every arm looks disappointing after a try and
  the greedy agent explores by itself.
"""))

CELLS.append(code(r"""
class Bandit:
    '''k arms with Gaussian rewards. q_star ~ N(0, 1), reward ~ N(q_star[a], 1).'''
    def __init__(self, k=10):
        self.k = k
        self.q_star = np.random.randn(k)
        self.best = int(self.q_star.argmax())
    def pull(self, a):
        return self.q_star[a] + np.random.randn()

class EpsGreedy:
    def __init__(self, k, eps=0.1, q0=0.0, alpha=None):
        self.eps, self.alpha = eps, alpha
        self.Q = np.full(k, float(q0)); self.N = np.zeros(k)
    def act(self, t):
        if np.random.rand() < self.eps:
            return np.random.randint(len(self.Q))
        return int(np.argmax(self.Q + 1e-9 * np.random.rand(len(self.Q))))   # random tie-break
    def update(self, a, r):
        self.N[a] += 1
        step = (1.0 / self.N[a]) if self.alpha is None else self.alpha
        self.Q[a] += step * (r - self.Q[a])

class UCB(EpsGreedy):
    def __init__(self, k, c=2.0):
        super().__init__(k, eps=0.0); self.c = c
    def act(self, t):
        if (self.N == 0).any():                       # try every arm once first
            return int(np.argmin(self.N))
        return int(np.argmax(self.Q + self.c * np.sqrt(np.log(t + 1) / self.N)))

def run_bandits(make_agent, n_runs=200, n_steps=1000, k=10):
    '''Returns (reward[n_runs, n_steps], optimal[n_runs, n_steps]) for a factory of agents.'''
    rewards = np.zeros((n_runs, n_steps)); optimal = np.zeros((n_runs, n_steps))
    for run in range(n_runs):
        bandit, agent = Bandit(k), make_agent(k)
        for t in range(n_steps):
            a = agent.act(t); r = bandit.pull(a); agent.update(a, r)
            rewards[run, t] = r; optimal[run, t] = (a == bandit.best)
    return rewards, optimal

seed_everything(0)
with Timer("bandit experiments"):
    results = {
        "greedy (ε=0)":        run_bandits(lambda k: EpsGreedy(k, eps=0.0)),
        "ε-greedy (ε=0.1)":    run_bandits(lambda k: EpsGreedy(k, eps=0.1)),
        "ε-greedy (ε=0.01)":   run_bandits(lambda k: EpsGreedy(k, eps=0.01)),
        "UCB (c=2)":           run_bandits(lambda k: UCB(k, c=2.0)),
        "optimistic (Q0=5, ε=0)": run_bandits(lambda k: EpsGreedy(k, eps=0.0, q0=5.0, alpha=0.1)),
    }
"""))

CELLS.append(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for name, (rew, opt) in results.items():
    plot_runs(axes[0], [rew.mean(0)], label=name, k=5)
    plot_runs(axes[1], [100 * opt.mean(0)], label=name, k=5)
finish(axes[0], "Average reward per step", "step", "reward")
finish(axes[1], "% of pulls on the optimal arm", "step", "% optimal", legend=False)
plt.tight_layout(); plt.show()
"""))

CELLS.append(md(r"""
**Reading the plot.** Pure greedy locks onto the first arm that looked good and never recovers.
ε = 0.1 learns fast but pays a permanent 10% exploration tax; ε = 0.01 is still catching up after 1000
steps, and only overtakes ε = 0.1 on much longer runs because its tax is ten times smaller (try `n_steps=5000`).
UCB and optimistic initialisation explore *systematically* and then stop exploring, which is why
they win in the long run. There is no free lunch: every exploration strategy trades short-term reward for
information.

**Why this matters later.** When we do RL on language models (Parts 10–17) the "arms" are the
astronomically many possible completions. Sampling $k$ completions per prompt and keeping the good
ones (GRPO's *group*) is exactly a bandit-style exploration, and the pass@k metric measures how much
"exploration" the base model still has in it.

### Test your knowledge
"""))

CELLS += quiz_cells("2.1", "In UCB, the bonus term c*sqrt(ln t / N(a)) is largest for which arms?",
    ["Arms with the highest estimated value",
     "Arms that have been pulled the fewest times",
     "Arms with the highest observed reward variance",
     "The arm pulled most recently"], "B",
    "N(a) in the denominator makes the bonus shrink as an arm gets pulled. The ln t in the numerator "
    "grows slowly so that every arm is eventually re-tried. Rarely tried arms are treated optimistically.")

CELLS += quiz_cells("2.2", "Why does a greedy agent with optimistic initial values (Q0 = 5) explore at all?",
    ["It uses a hidden epsilon.",
     "Every arm it tries returns less than 5, so its estimate drops below the untried arms, which still look like 5.",
     "Optimistic values increase the reward variance.",
     "It doesn't; the curve is a coincidence."], "B",
    "Disappointment drives exploration: each pull lowers that arm's estimate toward its true mean (< 5), "
    "making the still-untried arms look better. Once every arm has been sampled enough, exploration stops "
    "by itself. This only works with stationary problems and a constant step size alpha.")

CELLS += exercise_cells("2.3",
    r"""
### Exercise 2.3 · Gradient bandit

Instead of estimating values, learn a **preference** $H(a)$ per arm and act with a softmax
$\pi(a) = e^{H(a)} / \sum_b e^{H(b)}$. After receiving reward $r$ for action $a$ with a baseline $\bar r$
(the running average reward), the update is

$$H(a) \leftarrow H(a) + \alpha (r - \bar r)(1 - \pi(a)), \qquad H(b) \leftarrow H(b) - \alpha (r - \bar r)\pi(b) \;\; \text{for } b \ne a.$$

Implement `gradient_bandit_update(H, a, r, baseline, alpha)` returning the **new** preference vector.
This is a policy-gradient method in disguise; Part 6 derives the general version.
""",
    r"""
def gradient_bandit_update(H, a, r, baseline, alpha):
    '''H: np.array of preferences. Return the updated copy.'''
    # YOUR CODE HERE
    raise NotImplementedError
""",
    r"""
def _tests(fn):
    H = np.array([0.0, 0.0, 0.0]); out = fn(H.copy(), 1, r=2.0, baseline=1.0, alpha=0.1)
    assert out is not None, "function returned None"
    pi = np.exp(H) / np.exp(H).sum()
    expected = H - 0.1 * (2.0 - 1.0) * pi; expected[1] += 0.1 * (2.0 - 1.0)
    assert np.allclose(out, expected), f"expected {expected}, got {out}"
    H2 = np.array([1.0, -1.0]); out2 = fn(H2.copy(), 0, r=-0.5, baseline=0.5, alpha=0.2)
    pi2 = np.exp(H2) / np.exp(H2).sum()
    exp2 = H2 - 0.2 * (-1.0) * pi2; exp2[0] += 0.2 * (-1.0)
    assert np.allclose(out2, exp2), "wrong update when reward is below the baseline"
""",
    r"""
def gradient_bandit_update(H, a, r, baseline, alpha):
    pi = np.exp(H - H.max()); pi /= pi.sum()
    onehot = np.zeros_like(H); onehot[a] = 1.0
    return H + alpha * (r - baseline) * (onehot - pi)
""", 'gradient_bandit_update')
