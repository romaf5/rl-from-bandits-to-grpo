from common import video, md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 4 · Model-free learning: Monte Carlo, TD, SARSA, Q-learning

Now throw away the model. The agent only gets samples $(s, a, r, s')$ from interaction and must learn
values from them. Two families:

* **Monte Carlo (MC)**: wait until the episode ends, compute the actual return $G_t$, and move the estimate
  toward it: $V(s_t) \leftarrow V(s_t) + \alpha[G_t - V(s_t)]$. Unbiased, but high variance (a whole
  episode of randomness enters every target).
* **Temporal difference (TD)**: don't wait; use the next state's *current estimate* as a stand-in for the
  rest of the return, i.e. **bootstrap**: $V(s_t) \leftarrow V(s_t) + \alpha[\,r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\,]$.
  Lower variance, but biased while $V$ is wrong.

The quantity $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$ is the **TD error**. It is the single most
reused idea in RL; it will be the "advantage" in actor-critic methods and the per-step ingredient of GAE.

Let's compare both for *prediction* (estimating $V^\pi$ for the random policy) against the exact answer
from Part 1.
"""))
CELLS.append(video(4))

CELLS.append(code(r"""
def run_episode(env, pi, max_steps=200):
    '''Sample one episode under a (nS, nA) stochastic policy. Returns lists of states, actions, rewards.'''
    s = env.reset(); S, A, R = [s], [], []
    for _ in range(max_steps):
        a = np.random.choice(env.nA, p=pi[s])
        s, r, done = env.step(a)
        A.append(a); R.append(r); S.append(s)
        if done: break
    return S, A, R

def mc_prediction(env, pi, n_episodes, alpha=0.05, gamma=None, V_true=None):
    '''Every-visit Monte Carlo with a constant step size. Returns RMSE-vs-episode if V_true is given.'''
    gamma = env.gamma if gamma is None else gamma
    V = np.zeros(env.nS); errs = []
    for _ in range(n_episodes):
        S, A, R = run_episode(env, pi)
        G = 0.0
        for t in reversed(range(len(R))):             # walk backwards to accumulate returns
            G = R[t] + gamma * G
            V[S[t]] += alpha * (G - V[S[t]])
        if V_true is not None: errs.append(np.sqrt(np.mean((V - V_true) ** 2)))
    return V, errs

def td0_prediction(env, pi, n_episodes, alpha=0.05, gamma=None, V_true=None):
    gamma = env.gamma if gamma is None else gamma
    V = np.zeros(env.nS); errs = []
    for _ in range(n_episodes):
        s = env.reset()
        for _ in range(200):
            a = np.random.choice(env.nA, p=pi[s])
            s2, r, done = env.step(a)
            target = r + (0.0 if done else gamma * V[s2])      # bootstrap unless terminal
            V[s] += alpha * (target - V[s])
            s = s2
            if done: break
        if V_true is not None: errs.append(np.sqrt(np.mean((V - V_true) ** 2)))
    return V, errs

seed_everything(1)
curves = {"MC": [], "TD(0)": []}
with Timer("MC vs TD prediction"):
    for run in range(10):
        curves["MC"].append(mc_prediction(env, pi_random, 300, V_true=V_random)[1])
        curves["TD(0)"].append(td0_prediction(env, pi_random, 300, V_true=V_random)[1])

fig, ax = plt.subplots(figsize=(7, 3.4))
for name, runs in curves.items():
    plot_runs(ax, runs, label=name, k=3)
finish(ax, "Prediction error of the random policy's value  (α = 0.05, 10 seeds)", "episode", "RMSE vs exact $V^\\pi$")
plt.show()
"""))

CELLS.append(md(r"""
TD(0) drops faster and settles lower at this step size: bootstrapping trades a little bias for a lot
less variance. The gap widens with longer episodes, which is why almost all deep RL bootstraps.

## Control: learning Q and acting on it

To improve the policy we need action values. Three updates for $Q(s,a)$ after observing $(s,a,r,s')$
and (for SARSA) the next action $a'$ chosen by the current policy:

| Algorithm | Target | Type |
|---|---|---|
| **SARSA** | $r + \gamma\, Q(s', a')$ | on-policy: learns the value of the policy it follows, exploration included |
| **Q-learning** | $r + \gamma\, \max_{a'} Q(s', a')$ | off-policy: learns $Q^*$ regardless of how it explores |
| **Expected SARSA** | $r + \gamma \sum_{a'} \pi(a'|s')\, Q(s', a')$ | on-policy, but averages out the randomness of $a'$ |

We'll use the classic **CliffWalk** environment (Sutton & Barto, Example 6.6): a 4×12 grid, every step
costs −1, and stepping off the cliff costs −100 and sends you back to the start. The shortest path
runs right along the cliff edge. With ε-greedy exploration, is that path really the best one to *follow*?
"""))

CELLS.append(code(r"""
class CliffWalk:
    '''4x12 grid. Start bottom-left, goal bottom-right, cliff in between. Deterministic moves.'''
    MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    ARROWS = ["↑", "→", "↓", "←"]
    def __init__(self, H=4, W=12):
        self.H, self.W, self.nA = H, W, 4
        self.nS = H * W
        self.start, self.goal = (H - 1, 0), (H - 1, W - 1)
        self.cliff = {(H - 1, c) for c in range(1, W - 1)}
    def _id(self, cell): return cell[0] * self.W + cell[1]
    def reset(self):
        self.pos = self.start; return self._id(self.pos)
    def step(self, a):
        r, c = self.pos; dr, dc = self.MOVES[a]
        nr, nc = min(max(r + dr, 0), self.H - 1), min(max(c + dc, 0), self.W - 1)
        self.pos = (nr, nc)
        if self.pos in self.cliff:
            self.pos = self.start
            return self._id(self.pos), -100.0, False
        return self._id(self.pos), -1.0, self.pos == self.goal

def render_cliff(env, Q=None, paths=None, ax=None, title=None):
    if ax is None: fig, ax = plt.subplots(figsize=(7, 2.6))
    img = np.zeros((env.H, env.W))
    for (r, c) in env.cliff: img[r, c] = 1.0
    ax.imshow(img, cmap=mpl.colors.ListedColormap(["white", "#d9d8d3"]))
    ax.set_xticks(np.arange(-.5, env.W, 1), minor=True); ax.set_yticks(np.arange(-.5, env.H, 1), minor=True)
    ax.grid(which="minor", color=AXIS, linewidth=1); ax.grid(which="major", visible=False)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(env.start[1], env.start[0], "S", ha="center", va="center", fontweight="bold")
    ax.text(env.goal[1], env.goal[0], "G", ha="center", va="center", fontweight="bold")
    ax.text((env.W - 1) / 2, env.H - 1, "cliff (-100)", ha="center", va="center", color=INK2, fontsize=9)
    if Q is not None:
        for s in range(env.nS):
            r, c = divmod(s, env.W)
            if (r, c) in env.cliff or (r, c) == env.goal: continue
            ax.text(c, r, env.ARROWS[int(Q[s].argmax())], ha="center", va="center", fontsize=11, color=INK2)
    if paths:
        for (name, path, color) in paths:
            rc = np.array([divmod(s, env.W) for s in path])
            ax.plot(rc[:, 1], rc[:, 0], color=color, lw=2.5, alpha=0.9, label=name, solid_capstyle="round")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    if title: ax.set_title(title)
    return ax

cliff = CliffWalk()
render_cliff(cliff, title="CliffWalk"); plt.show()
"""))

CELLS.append(code(r"""
def eps_greedy(Q, s, eps):
    if np.random.rand() < eps:
        return np.random.randint(Q.shape[1])
    return int(np.argmax(Q[s] + 1e-9 * np.random.rand(Q.shape[1])))

def td_control(env, method="q_learning", n_episodes=500, alpha=0.5, gamma=1.0, eps=0.1, max_steps=500):
    '''One generic loop; `method` picks the target: 'sarsa' | 'q_learning' | 'expected_sarsa'.'''
    Q = np.zeros((env.nS, env.nA)); returns = []
    for ep in range(n_episodes):
        s = env.reset(); a = eps_greedy(Q, s, eps); total = 0.0
        for _ in range(max_steps):
            s2, r, done = env.step(a); total += r
            a2 = eps_greedy(Q, s2, eps)                          # the action we will actually take next
            if done:
                target = r
            elif method == "sarsa":
                target = r + gamma * Q[s2, a2]
            elif method == "q_learning":
                target = r + gamma * Q[s2].max()
            elif method == "expected_sarsa":
                probs = np.full(env.nA, eps / env.nA); probs[Q[s2].argmax()] += 1 - eps
                target = r + gamma * (probs * Q[s2]).sum()
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s2, a2
            if done: break
        returns.append(total)
    return Q, returns

def greedy_path(env, Q, max_steps=100):
    s = env.reset(); path = [s]
    for _ in range(max_steps):
        s, _, done = env.step(int(Q[s].argmax())); path.append(s)
        if done: break
    return path

seed_everything(0)
methods = ["sarsa", "q_learning", "expected_sarsa"]
cliff_results = {m: [] for m in methods}; cliff_Q = {}
with Timer("CliffWalk control"):
    for m in methods:
        for run in range(10):
            Q, rets = td_control(cliff, m); cliff_results[m].append(rets)
        cliff_Q[m] = Q

fig, axes = plt.subplots(2, 1, figsize=(9, 6.0), gridspec_kw={"height_ratios": [1.5, 1]})
for m in methods:
    plot_runs(axes[0], cliff_results[m], label=m, k=10)
axes[0].set_ylim(-100, 0)
finish(axes[0], "CliffWalk: sum of rewards per episode while learning (ε = 0.1, 10 seeds)", "episode", "return")
render_cliff(cliff, paths=[("SARSA greedy path", greedy_path(cliff, cliff_Q["sarsa"]), PALETTE[0]),
                           ("Q-learning greedy path", greedy_path(cliff, cliff_Q["q_learning"]), PALETTE[1])],
             ax=axes[1], title="Where each learned policy walks")
plt.tight_layout(); plt.show()
"""))

CELLS.append(md(r"""
**The classic result.** Q-learning learns the *optimal* path along the cliff edge, but while it is still
exploring with ε = 0.1 it keeps falling off, so its online return is worse. SARSA learns the value of
the ε-greedy policy it actually follows, and that policy is safer if it stays away from the edge.
Expected SARSA removes the variance of sampling $a'$ and usually does best online.

This is the on-policy / off-policy distinction, and it will come back with a vengeance in Parts 8–16:
PPO and GRPO are *almost* on-policy methods that use importance sampling to reuse slightly stale data,
and the whole art of stable LLM training is managing how far off-policy you can go.

### Test your knowledge
"""))

CELLS += quiz_cells("4.1", "Q-learning is called off-policy because:",
    ["It never explores.",
     "Its update target uses max_a' Q(s', a'), the greedy action, regardless of which action the behaviour policy actually takes next.",
     "It requires a model of the environment.",
     "It updates Q for states it never visited."], "B",
    "The target policy (greedy) differs from the behaviour policy (epsilon-greedy). Off-policy learning lets you "
    "learn about one policy from data generated by another, which is what makes experience replay legal in DQN.")

CELLS += quiz_cells("4.2", "Why did TD(0) beat Monte Carlo on prediction error in our experiment?",
    ["TD is unbiased and MC is biased",
     "MC needs a model of the environment",
     "TD targets have lower variance because they contain only one random step instead of a whole episode",
     "TD uses a larger step size"], "C",
    "Both used alpha = 0.05. MC's target G_t sums many random rewards and transitions; TD's target r + gamma V(s') "
    "contains just one. TD's bias vanishes as V improves, so for long episodes the variance reduction dominates.")

CELLS += quiz_cells("4.3", "Q-learning's max over noisy estimates tends to be too optimistic ('maximisation bias'). Which fix addresses it directly?",
    ["Use a smaller epsilon",
     "Double Q-learning: keep two Q tables, select the argmax with one and evaluate it with the other",
     "Use a larger discount factor",
     "Replace max with min"], "B",
    "E[max(X_1..X_n)] >= max(E[X_1..X_n]): taking a max over noisy estimates is biased upward. Decoupling "
    "selection from evaluation removes the bias. Double DQN applies the same idea with neural networks (Part 5).")

CELLS += exercise_cells("4.4",
    r"""
### Exercise 4.4 · Double Q-learning update

Keep two tables $Q_A$ and $Q_B$. On each transition flip a coin. If heads, update $Q_A$ with the target

$$r + \gamma\, Q_B\big(s', \arg\max_{a'} Q_A(s', a')\big),$$

otherwise update $Q_B$ symmetrically. Implement `double_q_update(QA, QB, s, a, r, s2, done, alpha, gamma, update_A)`
that modifies the tables **in place** (`update_A=True` means update $Q_A$). For a terminal transition the target is just `r`.
""",
    r"""
def double_q_update(QA, QB, s, a, r, s2, done, alpha, gamma, update_A):
    # YOUR CODE HERE
    raise NotImplementedError
""",
    r"""
def _tests(fn):
    QA = np.array([[0.0, 0.0], [1.0, 3.0]]); QB = np.array([[0.0, 0.0], [5.0, 2.0]])
    fn(QA, QB, s=0, a=0, r=1.0, s2=1, done=False, alpha=0.5, gamma=0.9, update_A=True)
    # argmax_a' QA[1] = 1 (value 3), evaluated with QB -> QB[1,1] = 2 ; target = 1 + 0.9*2 = 2.8 ; QA[0,0] = 0.5*2.8 = 1.4
    assert np.isclose(QA[0, 0], 1.4), f"expected QA[0,0]=1.4, got {QA[0,0]}"
    assert np.allclose(QB, [[0, 0], [5, 2]]), "QB must not change when update_A=True"
    fn(QA, QB, s=0, a=1, r=-1.0, s2=1, done=False, alpha=1.0, gamma=1.0, update_A=False)
    # argmax QB[1] = 0 (value 5), evaluated with QA -> QA[1,0] = 1 ; target = -1 + 1 = 0 ; QB[0,1] = 0
    assert np.isclose(QB[0, 1], 0.0), f"expected QB[0,1]=0.0, got {QB[0,1]}"
    fn(QA, QB, s=1, a=0, r=7.0, s2=0, done=True, alpha=1.0, gamma=0.9, update_A=True)
    assert np.isclose(QA[1, 0], 7.0), "terminal transitions should use target = r"
""",
    r"""
def double_q_update(QA, QB, s, a, r, s2, done, alpha, gamma, update_A):
    Qsel, Qeval = (QA, QB) if update_A else (QB, QA)
    if done:
        target = r
    else:
        a_star = int(np.argmax(Qsel[s2]))          # select with one table ...
        target = r + gamma * Qeval[s2, a_star]     # ... evaluate with the other
    Qsel[s, a] += alpha * (target - Qsel[s, a])
""", 'double_q_update')
