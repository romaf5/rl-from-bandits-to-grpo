from common import video, md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 3 · Dynamic programming: planning with a known model

When we *know* $p$ and $r$ we don't need to interact at all; we can compute $V^*$ by turning the Bellman
equations into update rules. This is **dynamic programming (DP)**. It doesn't scale to real problems
(you rarely know the model, and the state space is usually enormous), but every sample-based method
in later parts is an approximation of one of these two algorithms.

### Policy evaluation as a fixed-point iteration

Define the Bellman operator $(T^\pi V)(s) = \sum_a \pi(a|s)\sum_{s'} p(s'|s,a)[r + \gamma V(s')]$.
It is a **γ-contraction** in the max-norm: $\|T^\pi V - T^\pi U\|_\infty \le \gamma \|V - U\|_\infty$.
So iterating $V \leftarrow T^\pi V$ from any start converges to the unique fixed point $V^\pi$, and the
error shrinks by a factor $\gamma$ per sweep.

### Policy iteration

Alternate two steps until the policy stops changing:

1. **Evaluate** the current policy: compute $V^\pi$ (iteratively or by solving the linear system).
2. **Improve** it: $\pi'(s) = \arg\max_a Q^\pi(s,a)$. The policy improvement theorem guarantees $V^{\pi'} \ge V^\pi$.

### Value iteration

Skip the full evaluation and apply the *optimality* operator directly:
$V(s) \leftarrow \max_a \sum_{s'} p(s'|s,a)[r + \gamma V(s')]$. Also a γ-contraction, converging to $V^*$.
"""))
CELLS.append(video(3))

CELLS.append(code(r"""
def policy_evaluation(env, pi, gamma=None, theta=1e-8, max_sweeps=10_000, V0=None):
    '''Iterative policy evaluation: sweep V <- T^pi V until the max change < theta.'''
    gamma = env.gamma if gamma is None else gamma
    V = np.zeros(env.nS) if V0 is None else V0.copy()
    for sweep in range(max_sweeps):
        Q = q_from_v(env, V, gamma)                # (nS, nA)
        V_new = (pi * Q).sum(1)
        delta = np.abs(V_new - V).max()
        V = V_new
        if delta < theta:
            break
    return V, sweep + 1

def greedy_policy(env, V, gamma=None):
    '''Deterministic greedy policy w.r.t. V, returned as a (nS, nA) one-hot matrix.'''
    Q = q_from_v(env, V, gamma)
    pi = np.zeros_like(Q); pi[np.arange(env.nS), Q.argmax(1)] = 1.0
    return pi

def policy_iteration(env, gamma=None, verbose=True):
    pi = np.full((env.nS, env.nA), 1.0 / env.nA)
    history = []
    for it in range(100):
        V, sweeps = policy_evaluation(env, pi, gamma)
        pi_new = greedy_policy(env, V, gamma)
        history.append((V, pi_new))
        if verbose:
            print(f"iteration {it}: evaluation took {sweeps:4d} sweeps, V(start) = {V[env.start]:+.3f}")
        if np.array_equal(pi_new, pi):
            break
        pi = pi_new
    return V, pi, history

V_pi, pi_star, hist = policy_iteration(env)
"""))

CELLS.append(code(r"""
fig, axes = plt.subplots(1, len(hist), figsize=(3.4 * len(hist), 3.6))
for k, (ax, (V, pi)) in enumerate(zip(np.atleast_1d(axes), hist)):
    render_grid(env, V, policy=pi, ax=ax, title=f"policy iteration, step {k}", annotate=False)
plt.tight_layout(); plt.show()
"""))

CELLS.append(code(r"""
def value_iteration(env, gamma=None, theta=1e-8, max_sweeps=10_000, track=None):
    '''V <- max_a sum_s' P (R + gamma V).  Optionally records ||V_k - track||_inf per sweep.'''
    gamma = env.gamma if gamma is None else gamma
    V = np.zeros(env.nS); errors = []
    for sweep in range(max_sweeps):
        V_new = q_from_v(env, V, gamma).max(1)
        if track is not None:
            errors.append(np.abs(V_new - track).max())
        delta = np.abs(V_new - V).max(); V = V_new
        if delta < theta:
            break
    return V, greedy_policy(env, V, gamma), errors

V_star, pi_vi, errs = value_iteration(env, track=V_pi)
print("value iteration agrees with policy iteration:", np.allclose(V_star, V_pi, atol=1e-6),
      "| same greedy policy:", np.array_equal(pi_vi, pi_star))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for g in [0.5, 0.9, 0.99]:
    Vg, _, _ = value_iteration(env, gamma=g)
    _, _, e = value_iteration(env, gamma=g, track=Vg)
    axes[0].plot(e[:-1], label=f"γ = {g}")
axes[0].set_yscale("log")
finish(axes[0], "Value iteration error  $\\|V_k - V^*\\|_\\infty$  (slope = log γ)", "sweep", "max error")
render_grid(env, V_star, policy=pi_vi, ax=axes[1], title="$V^*$ and $\\pi^*$ from value iteration")
plt.tight_layout(); plt.show()
"""))

CELLS.append(md(r"""
The error curve on the left is a straight line on a log scale: that is the contraction at work, with
slope $\log\gamma$. A larger $\gamma$ means a longer horizon **and** slower convergence, a tension
that reappears in every algorithm (bootstrapping over long horizons is slow and unstable).

Notice in the optimal policy how the agent, at the cell just left of the pit, prefers to move *up* or
*down* rather than *right* toward the goal: with 10% slip it is worth a detour to avoid a chance of falling in.

### Test your knowledge
"""))

CELLS += quiz_cells("3.1", "Why does iterative policy evaluation converge from any initial V?",
    ["Because rewards are bounded",
     "Because the Bellman operator is a gamma-contraction in the max-norm, so it has a unique fixed point",
     "Because the policy is deterministic",
     "Because V is initialised to zero"], "B",
    "Banach's fixed-point theorem: a contraction on a complete metric space has exactly one fixed point, and "
    "iterating the map converges to it geometrically. That fixed point is V^pi. The same argument covers value "
    "iteration (the optimality operator is also a gamma-contraction).")

CELLS += quiz_cells("3.2", "Policy iteration converged in a handful of outer iterations while each evaluation took many sweeps. Value iteration needs no inner loop. When is value iteration preferable?",
    ["Never, policy iteration is always cheaper",
     "When the model is unknown",
     "When a full policy evaluation per iteration is too expensive and we are happy to improve the policy from partially-evaluated values",
     "When gamma is exactly 1"], "C",
    "Value iteration is policy iteration with a single evaluation sweep per improvement (a 'truncated' evaluation). "
    "Generalised policy iteration interleaves evaluation and improvement at any granularity, which is what "
    "actor-critic methods do with function approximation.")

CELLS += exercise_cells("3.3",
    r"""
### Exercise 3.3 · Q-value iteration

Run value iteration directly on **Q** instead of V:

$$Q(s,a) \leftarrow \sum_{s'} p(s'|s,a)\big[r(s,a,s') + \gamma \max_{a'} Q(s',a')\big].$$

Implement `q_value_iteration(env, gamma, theta)` returning the converged `Q` of shape `(nS, nA)`.
Hint: `env.P`, `env.R` have shape `(nS, nA, nS)`; `np.einsum("sat,sat->sa", ...)` is your friend.
""",
    r"""
def q_value_iteration(env, gamma=0.9, theta=1e-8):
    Q = np.zeros((env.nS, env.nA))
    # YOUR CODE HERE
    raise NotImplementedError
    return Q
""",
    r"""
def _tests(fn):
    Q = fn(env, gamma=0.9)
    assert Q is not None and Q.shape == (env.nS, env.nA), "wrong shape"
    V_ref, _, _ = value_iteration(env, gamma=0.9)
    assert np.allclose(Q.max(1), V_ref, atol=1e-5), "max_a Q(s,a) should equal V*(s)"
    assert np.all(Q.max(1) >= Q.min(1)), "sanity"
""",
    r"""
def q_value_iteration(env, gamma=0.9, theta=1e-8):
    Q = np.zeros((env.nS, env.nA))
    while True:
        target = env.R + gamma * Q.max(1)[None, None, :]        # (nS, nA, nS')
        Q_new = np.einsum("sat,sat->sa", env.P, target)
        if np.abs(Q_new - Q).max() < theta:
            return Q_new
        Q = Q_new
""", 'q_value_iteration')
