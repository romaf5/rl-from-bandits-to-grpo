from common import video, md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 7 · Actor-critic and generalised advantage estimation
"""))
CELLS.append(video(7))

CELLS.append(md(r"""
## Baselines

Subtracting a **baseline** $b(s_t)$ from the return does not change the expected gradient:

$$\mathbb E_{a \sim \pi}\big[\nabla_\theta \log \pi_\theta(a|s)\, b(s)\big] = b(s)\, \nabla_\theta \sum_a \pi_\theta(a|s) = b(s)\, \nabla_\theta 1 = 0.$$

The best simple choice is the state value $b(s) = V^\pi(s)$, which turns the weight into the
**advantage** $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$: *how much better was this action than what the policy
usually gets from here?* Actions that are merely "in a good state" no longer get credit; only actions
that beat expectations do.

## Actor-critic

We don't know $V^\pi$, so we learn it: a second network, the **critic**, trained by regression on
returns (or TD targets), while the **actor** $\pi_\theta$ is trained with the advantage-weighted policy
gradient. This is *generalised policy iteration* with neural networks: the critic evaluates, the actor improves.

## Generalised advantage estimation (GAE)

How should we estimate $A_t$? Two extremes:

* Monte Carlo: $\hat A_t = G_t - V(s_t)$. Unbiased given $V$, but high variance.
* One-step TD: $\hat A_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. Low variance, biased if $V$ is wrong.

GAE (Schulman et al., 2016) interpolates with a parameter $\lambda \in [0,1]$ exactly like TD(λ):

$$\hat A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}, \qquad
\text{computed backwards as } \hat A_t = \delta_t + \gamma\lambda\, \hat A_{t+1}.$$

$\lambda = 0$ gives the TD error, $\lambda = 1$ gives the Monte Carlo advantage. $\lambda \approx 0.95$
is the usual sweet spot. Almost every modern on-policy algorithm (A2C, TRPO, PPO, PPO-for-LLMs) uses it.
"""))

CELLS.append(code(r"""
def compute_gae(rew, values, next_values, term, done, gamma, lam):
    '''Returns (advantages, value targets). term[t]=1: s_{t+1} is terminal (no bootstrap).
    done[t]=1: episode ended after t (terminal OR truncated), so the GAE recursion must restart.'''
    T, adv, last = len(rew), np.zeros(len(rew), np.float32), 0.0
    for t in reversed(range(T)):
        delta = rew[t] + gamma * (1.0 - term[t]) * next_values[t] - values[t]     # TD error
        last = delta + gamma * lam * (1.0 - done[t]) * last
        adv[t] = last
    return adv, adv + values          # value target = advantage + baseline = lambda-return

def a2c_update(policy, value, opt_pi, opt_v, ro, gamma=0.99, lam=0.95, ent_coef=0.0, value_steps=5):
    with torch.no_grad():
        v, v_next = value(ro.obs).numpy(), value(ro.next_obs).numpy()
    adv, ret = compute_gae(ro.rew, v, v_next, ro.term, ro.done, gamma, lam)
    adv, ret = torch.as_tensor(adv), torch.as_tensor(ret)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)                       # advantage normalisation
    dist = policy.dist(ro.obs)
    pi_loss = -(dist.log_prob(ro.act) * adv).mean() - ent_coef * dist.entropy().mean()
    opt_pi.zero_grad(); pi_loss.backward(); opt_pi.step()
    for _ in range(value_steps):                                         # the critic can afford more steps
        v_loss = F.mse_loss(value(ro.obs), ret)
        opt_v.zero_grad(); v_loss.backward(); opt_v.step()
    return pi_loss.item(), v_loss.item()

def train_a2c(seed=0, iters=60, batch_steps=1024, lr=3e-3, lr_v=3e-3, gamma=0.99, lam=0.95):
    seed_everything(seed)
    env = CartPole(); policy, value = Policy(env.obs_dim, env.n_actions), ValueNet(env.obs_dim)
    opt_pi, opt_v = torch.optim.Adam(policy.parameters(), lr=lr), torch.optim.Adam(value.parameters(), lr=lr_v)
    curve, steps, total = [], [], 0
    for it in range(iters):
        ro = collect_rollout(env, policy, batch_steps, complete_episodes=True)
        a2c_update(policy, value, opt_pi, opt_v, ro, gamma, lam)
        total += len(ro.rew); curve.append(np.mean(ro.ep_returns)); steps.append(total)
    return policy, value, np.array(curve), np.array(steps)
"""))

CELLS.append(code(r"""
with Timer("A2C"):
    curves, steps = [], []
    for seed in range(N_SEEDS):
        pol_a2c, val_a2c, c, s = train_a2c(seed, iters=TOTAL_STEPS // 1024); curves.append(c); steps.append(s)
    pg_results["A2C (GAE λ=0.95)"] = (curves, steps)
    print(f"A2C final return (mean over seeds): {np.mean([c[-1] for c in curves]):6.1f}")

fig, ax = plt.subplots(figsize=(8, 3.6))
for name, (curves, steps) in pg_results.items():
    plot_runs(ax, curves, label=name, k=3, x=steps[0])
finish(ax, "Adding a learned baseline: REINFORCE vs actor-critic", "environment steps", "mean episode return")
plt.show()
"""))

CELLS.append(code(r"""
# What does lambda do? Advantages of one episode under the trained critic for several lambdas.
seed_everything(3)
ro = collect_rollout(CartPole(), pol_a2c, 400, complete_episodes=True)
end = int(np.where(ro.done)[0][0]) + 1                              # first complete episode
with torch.no_grad():
    v, v_next = val_a2c(ro.obs).numpy(), val_a2c(ro.next_obs).numpy()
fig, ax = plt.subplots(figsize=(9, 3.4))
for lam in [0.0, 0.5, 0.95, 1.0]:
    adv, _ = compute_gae(ro.rew, v, v_next, ro.term, ro.done, 0.99, lam)
    ax.plot(adv[:end], label=f"λ = {lam}" + ("  (TD error)" if lam == 0 else "  (Monte Carlo)" if lam == 1 else ""))
ax.axhline(0, color=AXIS, lw=1)
finish(ax, f"GAE advantages along one episode ({end} steps): λ trades variance for bias", "time step", "advantage")
plt.show()
"""))

CELLS.append(md(r"""
With $\lambda = 1$ the advantage carries the *entire* remaining episode's randomness; at
$\lambda = 0$ it is a single TD error, small and smooth but only as accurate as the critic. Values in
between blend the two. The last few steps before termination are where all curves agree: the critic
sees the fall coming.

### Test your knowledge
"""))

CELLS += quiz_cells("7.1", "Which statement about a state-dependent baseline b(s) is true?",
    ["It reduces the variance of the gradient estimate and leaves its expectation unchanged",
     "It makes the estimator biased but faster",
     "It only works if b(s) equals the true value function",
     "It increases the variance but improves exploration"], "A",
    "E_a[nabla log pi(a|s) b(s)] = 0 for any b that does not depend on the action. The closer b is to V^pi, "
    "the smaller the variance, but any b is 'legal'. This is why GRPO's group-mean baseline works even though it is a crude estimate of V.")

CELLS += quiz_cells("7.2", "GAE with lambda = 0 gives the advantage estimate:",
    ["G_t - V(s_t)", "r_t + gamma V(s_{t+1}) - V(s_t)", "r_t", "Q(s_t, a_t)"], "B",
    "Only the first TD error survives when (gamma*lambda)^l = 0 for l >= 1. This is the lowest-variance, "
    "highest-bias estimator; lambda = 1 recovers the Monte Carlo advantage.")

CELLS += quiz_cells("7.3", "Why do we bootstrap from V(s_{t+1}) at a truncated (time-limit) step but not at a terminal one?",
    ["Truncated steps are rare",
     "Because after a true terminal state there is no future reward, while after truncation the episode would have continued",
     "To save computation",
     "Because V is undefined at terminal states"], "B",
    "term[t] controls the bootstrap (V of a terminal state is 0 by definition); done[t] controls where the recursion "
    "restarts. Confusing the two teaches the critic that time itself is dangerous.")

CELLS += exercise_cells("7.4",
    r"""
### Exercise 7.4 · Implement GAE

Write `my_gae(rew, values, next_values, term, done, gamma, lam)` returning **only** the advantage array.
Check the recursion: $\delta_t = r_t + \gamma (1-\text{term}_t) V(s_{t+1}) - V(s_t)$ and
$\hat A_t = \delta_t + \gamma\lambda (1-\text{done}_t) \hat A_{t+1}$.
""",
    r"""
def my_gae(rew, values, next_values, term, done, gamma, lam):
    # YOUR CODE HERE
    raise NotImplementedError
""",
    r"""
def _tests(fn):
    rew = np.array([1.0, 1.0, 1.0]); values = np.array([0.5, 0.4, 0.3]); next_values = np.array([0.4, 0.3, 0.2])
    term = np.array([0.0, 0.0, 1.0]); done = np.array([0.0, 0.0, 1.0])
    A = fn(rew, values, next_values, term, done, gamma=0.9, lam=0.5)
    assert A is not None and len(A) == 3, "wrong length"
    d2 = 1.0 - 0.3                                   # terminal: no bootstrap
    d1 = 1.0 + 0.9 * 0.3 - 0.4; d0 = 1.0 + 0.9 * 0.4 - 0.5
    exp = np.array([d0 + 0.45 * (d1 + 0.45 * d2), d1 + 0.45 * d2, d2])
    assert np.allclose(A, exp, atol=1e-6), f"expected {exp}, got {A}"
    # truncation: bootstrap but restart the recursion
    A2 = fn(np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([2.0, 2.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]), 0.5, 0.9)
    assert np.allclose(A2, [2.0, 2.0]), f"truncation handling wrong: {A2}"
""",
    r"""
def my_gae(rew, values, next_values, term, done, gamma, lam):
    adv, last = np.zeros(len(rew)), 0.0
    for t in reversed(range(len(rew))):
        delta = rew[t] + gamma * (1 - term[t]) * next_values[t] - values[t]
        last = delta + gamma * lam * (1 - done[t]) * last
        adv[t] = last
    return adv
""", "my_gae")
