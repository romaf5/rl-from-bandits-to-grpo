from common import video, md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 6 · Policy gradients: REINFORCE

Value-based methods learn $Q$ and act greedily. **Policy gradient** methods skip the detour and
optimise the policy $\pi_\theta(a|s)$ directly by gradient ascent on the expected return
$J(\theta) = \mathbb E_{\tau \sim \pi_\theta}[R(\tau)]$. This handles continuous actions, stochastic
optimal policies, and, crucially for Parts 10–17, policies that are *language models*.
"""))
CELLS.append(video(6))

CELLS.append(md(r"""
## The policy gradient theorem

Write $p_\theta(\tau) = p(s_0) \prod_t \pi_\theta(a_t|s_t)\, p(s_{t+1}|s_t, a_t)$. Then, using the
**log-derivative trick** $\nabla p = p \nabla \log p$,

$$\nabla_\theta J = \int \nabla_\theta p_\theta(\tau) R(\tau)\, d\tau
= \mathbb E_{\tau}\big[ R(\tau)\, \nabla_\theta \log p_\theta(\tau) \big]
= \mathbb E_{\tau}\Big[ R(\tau) \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \Big].$$

The dynamics $p(s'|s,a)$ vanished because they don't depend on $\theta$. **We never need to know
the environment model.** The estimator says: *push up the log-probability of every action taken in
proportion to how good the whole trajectory turned out.*

Two refinements that don't change the expectation but slash the variance:

* **Reward-to-go / causality.** The action at time $t$ cannot affect rewards before $t$, so
  replace $R(\tau)$ by $G_t = \sum_{t' \ge t} \gamma^{t'-t} r_{t'}$.
* **Baselines** (next Part): subtract any function $b(s_t)$ from $G_t$.

The resulting algorithm is **REINFORCE** (Williams, 1992): sample episodes, compute returns, take a
gradient step on $-\frac{1}{N}\sum_t G_t \log \pi_\theta(a_t|s_t)$.
"""))

CELLS.append(code(r"""
class Policy(nn.Module):
    '''Categorical policy: MLP -> action logits.'''
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n_actions))
    def dist(self, obs):
        return torch.distributions.Categorical(logits=self.net(obs))
    @torch.no_grad()
    def act(self, obs):
        probs = torch.softmax(self.net(torch.as_tensor(obs)), -1).numpy()
        return int(np.random.choice(len(probs), p=probs))

class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, obs):
        return self.net(obs).squeeze(-1)

@dataclass
class Rollout:
    obs: torch.Tensor; act: torch.Tensor; next_obs: torch.Tensor
    rew: np.ndarray; term: np.ndarray; done: np.ndarray   # term: true terminal; done: term OR truncated
    ep_returns: list

def collect_rollout(env, policy, n_steps, complete_episodes=False):
    '''Run the policy for >= n_steps. If complete_episodes, keep going until the current episode ends.'''
    O, A, O2, R, T, D, ep_returns = [], [], [], [], [], [], []
    obs, ep_ret, t = env.reset(), 0.0, 0
    while True:
        a = policy.act(obs)
        next_obs, r, term, trunc = env.step(a)
        O.append(obs); A.append(a); O2.append(next_obs); R.append(r); T.append(term); D.append(term or trunc)
        obs, ep_ret, t = next_obs, ep_ret + r, t + 1
        if term or trunc:
            ep_returns.append(ep_ret); ep_ret, obs = 0.0, env.reset()
            if t >= n_steps: break
        elif t >= n_steps and not complete_episodes:
            break
    return Rollout(torch.as_tensor(np.array(O)), torch.as_tensor(np.array(A)), torch.as_tensor(np.array(O2)),
                   np.array(R, np.float32), np.array(T, np.float32), np.array(D, np.float32), ep_returns)

def rewards_to_go(rew, done, gamma):
    '''G_t = r_t + gamma * G_{t+1}, restarting at episode boundaries (done[t] = 1 means t was the last step).'''
    G, running = np.zeros(len(rew), np.float32), 0.0
    for t in reversed(range(len(rew))):
        running = rew[t] + gamma * running * (1.0 - done[t])
        G[t] = running
    return G

def episode_return_per_step(rew, done, gamma):
    '''The naive estimator: every step of an episode gets that episode's full discounted return.'''
    out, start = np.zeros(len(rew), np.float32), 0
    for t in range(len(rew)):
        if done[t] or t == len(rew) - 1:
            disc = gamma ** np.arange(t - start + 1)
            out[start:t + 1] = (rew[start:t + 1] * disc).sum(); start = t + 1
    return out

# quick sanity check of the return computation
_r = np.array([1, 1, 1, 1], np.float32); _d = np.array([0, 1, 0, 1], np.float32)
print("reward-to-go (γ=0.5):", rewards_to_go(_r, _d, 0.5), "  episode return per step:", episode_return_per_step(_r, _d, 0.5))
"""))

CELLS.append(code(r"""
def reinforce_update(policy, opt, ro, gamma, mode="rtg"):
    G = rewards_to_go(ro.rew, ro.done, gamma) if mode == "rtg" else episode_return_per_step(ro.rew, ro.done, gamma)
    G = torch.as_tensor(G)
    logp = policy.dist(ro.obs).log_prob(ro.act)
    loss = -(logp * G).mean()                    # gradient ascent on E[G log pi]
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

def train_reinforce(seed=0, mode="rtg", iters=60, batch_steps=1024, lr=3e-3, gamma=0.99, snapshot_at=None):
    seed_everything(seed)
    env = CartPole(); policy = Policy(env.obs_dim, env.n_actions)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    curve, steps, snapshot, total = [], [], None, 0
    for it in range(iters):
        ro = collect_rollout(env, policy, batch_steps, complete_episodes=True)
        reinforce_update(policy, opt, ro, gamma, mode)
        total += len(ro.rew); curve.append(np.mean(ro.ep_returns)); steps.append(total)
        if snapshot_at is not None and it == snapshot_at: snapshot = copy.deepcopy(policy)
    return policy, np.array(curve), np.array(steps), snapshot
"""))

CELLS.append(code(r"""
# Shared experiment settings for all CartPole policy-gradient runs (lower N_SEEDS to 1 for a faster notebook)
N_SEEDS = 3
TOTAL_STEPS = 60 * 1024          # same environment budget for every algorithm
pg_results = {}                  # name -> (list of return curves, list of step curves)

with Timer("REINFORCE"):
    for mode, name in [("total", "REINFORCE (total return)"), ("rtg", "REINFORCE (reward-to-go)")]:
        curves, steps = [], []
        for seed in range(N_SEEDS):
            pol, c, s, snap = train_reinforce(seed, mode, iters=TOTAL_STEPS // 1024, snapshot_at=15)
            curves.append(c); steps.append(s)
            if mode == "rtg" and seed == 0: policy_snapshot = snap      # kept for the variance experiment
        pg_results[name] = (curves, steps)
        print(f"{name:30s} final return (mean over seeds): {np.mean([c[-1] for c in curves]):6.1f}")
"""))

CELLS.append(code(r"""
fig, ax = plt.subplots(figsize=(8, 3.6))
for name, (curves, steps) in pg_results.items():
    plot_runs(ax, curves, label=name, k=3, x=steps[0])
finish(ax, f"REINFORCE on CartPole ({N_SEEDS} seeds, mean ± std)", "environment steps", "mean episode return")
plt.show()
"""))

CELLS.append(md(r"""
## Measuring the variance of the gradient estimate

Let's make the "variance" talk concrete. Freeze a partially trained policy, draw many independent
batches of the same size, compute the gradient estimate for each, and look at how much the estimates
disagree with each other. We compare the naive total-return estimator, reward-to-go, and reward-to-go
minus a constant baseline (the batch mean return).
"""))

CELLS.append(code(r"""
def flat_grad(y, model, **kw):
    grads = torch.autograd.grad(y, list(model.parameters()), **kw)
    return torch.cat([g.reshape(-1) for g in grads])

def gradient_variance(policy, estimator, n_batches=30, batch_steps=500):
    env, grads = CartPole(), []
    for _ in range(n_batches):
        ro = collect_rollout(env, policy, batch_steps, complete_episodes=True)
        G = torch.as_tensor(estimator(ro))
        logp = policy.dist(ro.obs).log_prob(ro.act)
        grads.append(flat_grad(-(logp * G).mean(), policy))
    grads = torch.stack(grads)
    return grads.var(0).sum().item(), grads.mean(0)
"""))

CELLS.append(code(r"""
seed_everything(0)
estimators = {
    "total return":            lambda ro: episode_return_per_step(ro.rew, ro.done, 0.99),
    "reward-to-go":            lambda ro: rewards_to_go(ro.rew, ro.done, 0.99),
    "reward-to-go − mean":     lambda ro: (lambda G: G - G.mean())(rewards_to_go(ro.rew, ro.done, 0.99)),
}
with Timer("gradient variance"):
    var = {name: gradient_variance(policy_snapshot, est)[0] for name, est in estimators.items()}

fig, ax = plt.subplots(figsize=(6.5, 3))
names = list(var); vals = [var[n] for n in names]
bars = ax.barh(names, vals, color=PALETTE[0], height=0.5)
for b, v in zip(bars, vals):
    ax.text(v, b.get_y() + b.get_height() / 2, f"  {v:.2g}", va="center", fontsize=9, color=INK2)
ax.set_xscale("log"); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
finish(ax, "Total variance of the policy-gradient estimate (same policy, same batch size)", "variance (log scale)", legend=False)
plt.show()
"""))

CELLS.append(md(r"""
Each refinement removes a large chunk of variance **without changing what the gradient points at**.
That is the whole game in policy-gradient research: keep the estimator unbiased (or nearly so) and
make it as low-variance as possible so that fewer samples give a usable direction. GAE (Part 7), TRPO's
trust region (Part 8), PPO's clipping (Part 9) and GRPO's group baseline (Part 15) are all moves in
this game.

### Test your knowledge
"""))

CELLS += quiz_cells("6.1", "Why does the environment's transition model p(s'|s,a) disappear from the policy gradient?",
    ["Because we assume deterministic dynamics",
     "Because log p(tau) is a sum, and the dynamics terms don't depend on theta so their gradient is zero",
     "Because we use the reward-to-go trick",
     "Because the baseline cancels it"], "B",
    "log p_theta(tau) = log p(s0) + sum_t [log pi_theta(a_t|s_t) + log p(s_{t+1}|s_t,a_t)]. Differentiating with "
    "respect to theta kills everything except the policy terms. This is why policy gradients are model-free.")

CELLS += quiz_cells("6.2", "Replacing R(tau) with the reward-to-go G_t is allowed because:",
    ["Past rewards are always zero",
     "For any t, E[ nabla log pi(a_t|s_t) * (rewards before t) ] = 0: an action cannot influence rewards that came before it",
     "It makes the gradient larger",
     "Returns are always positive in CartPole"], "B",
    "Conditioned on s_t, the earlier rewards are fixed constants, and E_a[nabla log pi(a|s)] = nabla sum_a pi(a|s) = "
    "nabla 1 = 0. So the removed terms had zero mean; dropping them keeps the estimator unbiased and removes noise.")

CELLS += quiz_cells("6.3", "REINFORCE is on-policy. What does that imply for sample efficiency?",
    ["Every batch of data can be used only for the policy that generated it, then must be thrown away",
     "It can reuse a replay buffer like DQN",
     "It needs a model of the environment",
     "It converges in one update"], "A",
    "The gradient is an expectation under the current policy. After one update the samples come from an old policy "
    "and the estimate is biased. TRPO and PPO fix this partially by importance-weighting the samples for a few "
    "epochs while keeping the policy close to the old one.")

CELLS += exercise_cells("6.4",
    r"""
### Exercise 6.4 · Discounted reward-to-go with episode boundaries

Write `my_rewards_to_go(rew, done, gamma)` from scratch (don't call the notebook's version). `rew` and
`done` are 1-D arrays; `done[t] = 1` means step `t` was the **last** step of its episode. Return an array
`G` with $G_t = r_t + \gamma G_{t+1}$ inside an episode and $G_t = r_t$ at the last step.
""",
    r"""
def my_rewards_to_go(rew, done, gamma):
    # YOUR CODE HERE
    raise NotImplementedError
""",
    r"""
def _tests(fn):
    rew = np.array([1.0, 2.0, 3.0, 4.0, 5.0]); done = np.array([0, 0, 1, 0, 1])
    G = fn(rew, done, 0.5)
    assert G is not None and len(G) == 5, "wrong length"
    assert np.allclose(G, [1 + 0.5 * (2 + 0.5 * 3), 2 + 0.5 * 3, 3.0, 4 + 0.5 * 5, 5.0]), f"got {G}"
    G2 = fn(np.ones(4), np.array([0, 0, 0, 1]), 1.0)
    assert np.allclose(G2, [4, 3, 2, 1]), f"undiscounted case failed: {G2}"
""",
    r"""
def my_rewards_to_go(rew, done, gamma):
    G = np.zeros(len(rew)); running = 0.0
    for t in reversed(range(len(rew))):
        running = rew[t] + gamma * running * (1.0 - done[t])
        G[t] = running
    return G
""", "my_rewards_to_go")
