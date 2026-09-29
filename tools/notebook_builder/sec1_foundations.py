from common import video, md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 1 · Foundations: the RL problem

Reinforcement learning is learning **what to do** from **evaluative feedback**: the environment does not
tell the agent the correct action (that would be supervised learning), it only tells it how good the
outcome was. The agent must discover good behaviour by trial and error, and it must deal with
*delayed* consequences: an action now can pay off much later.
"""))
CELLS.append(video(1))

CELLS.append(md(r"""
## The agent–environment loop

At every step $t$ the agent sees a state $s_t$, picks an action $a_t$, and receives a reward $r_{t+1}$ and
the next state $s_{t+1}$. That's it. Everything else in this notebook is about how to pick $a_t$ well.
"""))

CELLS.append(code(r"""
def draw_rl_loop():
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off"); ax.grid(False)
    for (x, label) in [(1.0, "Agent\n(policy $\\pi$)"), (6.5, "Environment\n(dynamics $p$)")]:
        ax.add_patch(FancyBboxPatch((x, 1.3), 2.5, 1.4, boxstyle="round,pad=0.05,rounding_size=0.2",
                                    fc="white", ec=PALETTE[0], lw=1.5))
        ax.text(x + 1.25, 2.0, label, ha="center", va="center", fontsize=11, color=INK)
    ax.add_patch(FancyArrowPatch((3.5, 2.4), (6.5, 2.4), arrowstyle="-|>", mutation_scale=16, lw=1.5, color=PALETTE[1]))
    ax.text(5.0, 2.65, "action $a_t$", ha="center", color=INK2)
    ax.add_patch(FancyArrowPatch((6.5, 1.6), (3.5, 1.6), arrowstyle="-|>", mutation_scale=16, lw=1.5, color=PALETTE[2]))
    ax.text(5.0, 1.15, "state $s_{t+1}$, reward $r_{t+1}$", ha="center", color=INK2)
    ax.set_title("The RL interaction loop")
    plt.show()
draw_rl_loop()
"""))

CELLS.append(md(r"""
## Markov decision processes

The standard formalisation is a **Markov decision process (MDP)** $(\mathcal S, \mathcal A, p, r, \gamma)$:

* $\mathcal S$: set of states, $\mathcal A$: set of actions.
* $p(s' \mid s, a)$: transition dynamics, the probability of landing in $s'$ after doing $a$ in $s$.
* $r(s, a, s')$: reward for that transition.
* $\gamma \in [0, 1)$: **discount factor**, how much we care about the future.

The **Markov property** says the next state and reward depend on the past *only through the current state*:
$p(s_{t+1} \mid s_t, a_t) = p(s_{t+1} \mid s_0, a_0, \dots, s_t, a_t)$. The state is a sufficient statistic
of history. (In Part 10 the "state" will be an entire text prefix, which makes this trivially true.)

### Our first environment: a stochastic GridWorld

A 5×5 grid. `S` is the start, `G` a goal worth +1, `P` a pit worth −1 (both terminal), `#` are walls.
Every step costs −0.04 so the agent has a reason to hurry. Moves are noisy: with probability `slip` the
agent goes sideways instead of where it intended. Because we build the transition tensor
`P[s, a, s']` explicitly, we can later do exact **planning** (Part 3) before doing any **learning** (Part 4).
"""))

CELLS.append(code(r"""
class GridWorld:
    '''Small stochastic grid MDP with a known model (P, R) so we can plan exactly.
    Actions: 0=up, 1=right, 2=down, 3=left.  With prob `slip` the move goes sideways.'''
    MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    ARROWS = ["↑", "→", "↓", "←"]

    def __init__(self, layout=("S....", ".#.#.", ".....", ".#.P.", "....G"),
                 slip=0.1, step_reward=-0.04, gamma=0.9):
        self.layout = [list(row) for row in layout]
        self.H, self.W = len(layout), len(layout[0])
        self.slip, self.step_reward, self.gamma = slip, step_reward, gamma
        # states = all non-wall cells, numbered row-major
        self.cells = [(r, c) for r in range(self.H) for c in range(self.W) if self.layout[r][c] != "#"]
        self.idx = {cell: i for i, cell in enumerate(self.cells)}
        self.nS, self.nA = len(self.cells), 4
        self.start = self.idx[next(cell for cell in self.cells if self._ch(cell) == "S")]
        self.terminal = np.array([self._ch(cell) in "GP" for cell in self.cells])
        self._build_model()

    def _ch(self, cell): return self.layout[cell[0]][cell[1]]

    def _move(self, cell, a):
        r, c = cell; dr, dc = self.MOVES[a]
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.H and 0 <= nc < self.W and self.layout[nr][nc] != "#":
            return (nr, nc)
        return cell                       # bumping into a wall or the border: stay put

    def _build_model(self):
        '''Fill P[s,a,s'] (transition probabilities) and R[s,a,s'] (rewards).'''
        self.P = np.zeros((self.nS, self.nA, self.nS))
        self.R = np.zeros((self.nS, self.nA, self.nS))
        for s, cell in enumerate(self.cells):
            for a in range(self.nA):
                if self.terminal[s]:      # terminal states absorb with zero reward
                    self.P[s, a, s] = 1.0
                    continue
                outcomes = [(a, 1 - self.slip), ((a + 1) % 4, self.slip / 2), ((a - 1) % 4, self.slip / 2)]
                for a_actual, prob in outcomes:
                    s2 = self.idx[self._move(cell, a_actual)]
                    self.P[s, a, s2] += prob
                    ch = self._ch(self.cells[s2])
                    self.R[s, a, s2] = self.step_reward + (1.0 if ch == "G" else -1.0 if ch == "P" else 0.0)

    # --- the interaction interface used by learning algorithms ---
    def reset(self):
        self.s = self.start
        return self.s

    def step(self, a):
        s2 = np.random.choice(self.nS, p=self.P[self.s, a])
        r = self.R[self.s, a, s2]
        self.s = s2
        return s2, r, bool(self.terminal[s2])

env = GridWorld()
print(f"{env.nS} states, {env.nA} actions, start state {env.start}")
print("P sums to 1 over s' for every (s,a):", np.allclose(env.P.sum(-1), 1.0))
"""))

CELLS.append(code(r"""
def render_grid(env, V=None, policy=None, title=None, ax=None, annotate=True, cmap="Blues"):
    '''Draw the grid. V: array of state values -> heatmap. policy: array of actions (or nS x nA probs) -> arrows.'''
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
    img = np.full((env.H, env.W), np.nan)
    for s, (r, c) in enumerate(env.cells):
        if not env.terminal[s]:                                # terminal cells are drawn separately
            img[r, c] = V[s] if V is not None else 0.0
    cm = plt.get_cmap(cmap if V is not None else "Greys").copy()
    cm.set_bad("white")
    if V is not None:
        vmin, vmax = float(np.nanmin(img)), float(np.nanmax(img))
        if vmax - vmin < 1e-9: vmax = vmin + 1e-9
    else:
        vmin, vmax = 0.0, 1.0
    ax.imshow(img, cmap=cm, vmin=vmin, vmax=vmax)
    for r in range(env.H):                                     # walls
        for c in range(env.W):
            if env.layout[r][c] == "#":
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc="#d9d8d3", ec="none"))
    for s, (r, c) in enumerate(env.cells):                     # goal / pit washes
        if env.terminal[s]:
            col = "#0ca30c" if env._ch((r, c)) == "G" else "#d03b3b"
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fc=col, alpha=0.25, ec="none"))
    ax.set_xticks(np.arange(-.5, env.W, 1), minor=True); ax.set_yticks(np.arange(-.5, env.H, 1), minor=True)
    ax.grid(which="minor", color=AXIS, linewidth=1); ax.grid(which="major", visible=False)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values(): spine.set_visible(False)
    if policy is not None:
        acts = np.asarray(policy).argmax(-1) if np.ndim(policy) == 2 else np.asarray(policy)
    for s, (r, c) in enumerate(env.cells):
        ch = env._ch((r, c))
        dark = V is not None and not env.terminal[s] and (img[r, c] - vmin) / (vmax - vmin) > 0.6
        col = "white" if dark else INK
        if ch in "SGP":
            ax.text(c, r - 0.28, ch, ha="center", va="center", fontsize=11, fontweight="bold", color=col)
        if policy is not None and not env.terminal[s]:
            ax.text(c, r + 0.05, env.ARROWS[acts[s]], ha="center", va="center", fontsize=15, color=col)
        if V is not None and annotate and not env.terminal[s]:
            ax.text(c, r + 0.34, f"{V[s]:.2f}", ha="center", va="center", fontsize=7, color=col)
    if title: ax.set_title(title)
    return ax

render_grid(env, title="GridWorld layout  (S start, G goal +1, P pit -1)")
plt.show()
"""))

CELLS.append(md(r"""
## Returns and discounting

The agent wants to maximise the **return**, the discounted sum of future rewards:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

Discounting does three jobs: it keeps the sum finite, it expresses a preference for sooner rewards,
and it sets an **effective horizon** of roughly $1/(1-\gamma)$ steps. Note the recursion
$G_t = r_{t+1} + \gamma G_{t+1}$, which is the seed of every Bellman equation below.
"""))

CELLS.append(code(r"""
fig, ax = plt.subplots(figsize=(7, 3.2))
t = np.arange(0, 120)
for g in [0.5, 0.9, 0.99]:
    ax.plot(t, g ** t, label=f"$\\gamma$ = {g}   (horizon ≈ {1/(1-g):.0f} steps)")
finish(ax, "Weight of a reward k steps in the future:  $\\gamma^k$", "k (steps ahead)", "weight")
plt.show()
"""))

CELLS.append(md(r"""
## Policies and value functions

A **policy** $\pi(a \mid s)$ is a distribution over actions in each state. Given a policy, two functions
summarise how good things are:

$$V^\pi(s) = \mathbb E_\pi\big[G_t \mid s_t = s\big], \qquad Q^\pi(s, a) = \mathbb E_\pi\big[G_t \mid s_t = s, a_t = a\big].$$

Using $G_t = r_{t+1} + \gamma G_{t+1}$ and taking expectations gives the **Bellman expectation equation**:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} p(s' \mid s, a)\big[r(s,a,s') + \gamma V^\pi(s')\big].$$

For a finite MDP this is a *linear* system. Writing $P_\pi[s, s'] = \sum_a \pi(a|s) p(s'|s,a)$ and
$R_\pi[s] = \sum_a \pi(a|s) \sum_{s'} p(s'|s,a) r(s,a,s')$:

$$V^\pi = R_\pi + \gamma P_\pi V^\pi \quad \Longrightarrow \quad V^\pi = (I - \gamma P_\pi)^{-1} R_\pi.$$

Let's evaluate the **uniform random policy** exactly.
"""))

CELLS.append(code(r"""
def evaluate_policy_exact(env, pi, gamma=None):
    '''Solve the Bellman expectation equation as a linear system. pi: (nS, nA) probabilities.'''
    gamma = env.gamma if gamma is None else gamma
    P_pi = np.einsum("sa,sat->st", pi, env.P)                       # (nS, nS)
    R_pi = np.einsum("sa,sat,sat->s", pi, env.P, env.R)             # (nS,)
    return np.linalg.solve(np.eye(env.nS) - gamma * P_pi, R_pi)

def q_from_v(env, V, gamma=None):
    '''Q(s,a) = sum_s' P[s,a,s'] (R[s,a,s'] + gamma V[s'])'''
    gamma = env.gamma if gamma is None else gamma
    return np.einsum("sat,sat->sa", env.P, env.R + gamma * V[None, None, :])

pi_random = np.full((env.nS, env.nA), 0.25)
V_random = evaluate_policy_exact(env, pi_random)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
render_grid(env, V_random, ax=axes[0], title="$V^\\pi$ of the random policy")
render_grid(env, V_random, policy=q_from_v(env, V_random), ax=axes[1], title="greedy w.r.t. $Q^\\pi$ (one improvement step)")
plt.tight_layout(); plt.show()
print(f"value of the start state under the random policy: {V_random[env.start]:+.3f}")
"""))

CELLS.append(md(r"""
Even one step of *acting greedily with respect to the random policy's Q-values* already points most
arrows toward the goal and away from the pit. That is the **policy improvement theorem**, and Part 3
turns it into an algorithm.

## Optimality

The **optimal** value functions satisfy the **Bellman optimality equations**, which replace the
expectation over $\pi$ with a max:

$$V^*(s) = \max_a \sum_{s'} p(s'|s,a)\big[r + \gamma V^*(s')\big], \qquad
Q^*(s,a) = \sum_{s'} p(s'|s,a)\big[r + \gamma \max_{a'} Q^*(s',a')\big].$$

Once we have $Q^*$ the optimal policy is trivial: $\pi^*(s) = \arg\max_a Q^*(s,a)$. Almost every
algorithm in this notebook is a way of approximating $V^*$, $Q^*$, or $\pi^*$ directly.

### Test your knowledge
"""))

CELLS += quiz_cells("1.1", "What does the Markov property say?",
    ["The reward depends only on the action taken.",
     "The future depends on the past only through the current state.",
     "The environment dynamics are deterministic.",
     "The agent must remember all previous states to act optimally."], "B",
    "Given s_t and a_t, the distribution of s_{t+1} and r_{t+1} does not change if you also condition on "
    "earlier history. The current state already contains everything relevant, which is exactly what makes "
    "value functions of the *state* meaningful.")

CELLS += quiz_cells("1.2", "Roughly how many steps into the future does an agent with gamma = 0.99 'care about'?",
    ["About 10 steps", "About 100 steps", "About 1000 steps", "Infinitely many, equally"], "B",
    "The effective horizon is about 1/(1-gamma) = 100 steps: rewards beyond that are weighted by less than "
    "e^-1 of their nominal value. This is why LLM-RL, where episodes are a few thousand tokens, typically uses gamma = 1.")

CELLS += quiz_cells("1.3", "Suppose we change GridWorld's step reward from -0.04 to +0.5 (gamma stays 0.9). What does the optimal policy do?",
    ["Still rushes to the goal, rewards are relative.",
     "Avoids both terminal states so it can collect +0.5 forever.",
     "Goes to the pit, because -1 is now less bad than before.",
     "Becomes undefined because returns are infinite."], "B",
    "Staying alive forever is worth 0.5/(1-0.9) = 5, far more than the goal's +1. The optimal policy hugs walls "
    "and never terminates. Reward design determines behaviour; this is the toy version of 'reward hacking'.")
