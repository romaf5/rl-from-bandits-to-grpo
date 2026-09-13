from common import md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 5 · Deep Q-learning (DQN)

Tables stop working the moment the state is continuous or high-dimensional. The fix is
**function approximation**: represent $Q(s, a; \theta)$ with a neural network and move $\theta$ by
gradient descent on the squared TD error. In principle this is Q-learning with a new data structure.
In practice, combining **bootstrapping**, **off-policy** updates and **function approximation** (the
"deadly triad") can diverge, and DQN (Mnih et al., 2013/2015) added two tricks that make it work:

1. **Experience replay.** Store transitions in a buffer and train on random minibatches. This breaks
   the correlation between consecutive samples and reuses each transition many times.
2. **Target network.** Compute the target $r + \gamma \max_{a'} Q(s', a'; \theta^-)$ with a *slowly moving copy*
   $\theta^-$ (refreshed only every 10,000 updates in the 2015 Nature paper; we use the now-common Polyak average
   $\theta^- \leftarrow (1-\tau)\theta^- + \tau\theta$), so the regression target stops jumping under our feet.

## The environment: CartPole

A pole hinged on a cart. The observation is $[x, \dot x, \theta, \dot \theta]$, the actions are push
left / push right, the reward is +1 per step, and the episode ends when the pole falls past 12° or the
cart drifts 2.4 units, or after 500 steps. We implement the physics in NumPy so the notebook needs no
extra packages.
"""))

CELLS.append(code(r"""
class CartPole:
    '''NumPy port of the classic CartPole-v1 dynamics (Euler integration, dt = 0.02).'''
    def __init__(self, max_steps=500):
        self.g, self.mc, self.mp, self.l, self.force, self.tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
        self.theta_thr, self.x_thr, self.max_steps = 12 * 2 * math.pi / 360, 2.4, max_steps
        self.obs_dim, self.n_actions = 4, 2

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, size=4); self.t = 0
        return self.state.astype(np.float32)

    def step(self, a):
        x, xd, th, thd = self.state
        f = self.force if a == 1 else -self.force
        cos, sin = math.cos(th), math.sin(th)
        total = self.mc + self.mp
        temp = (f + self.mp * self.l * thd ** 2 * sin) / total
        thacc = (self.g * sin - cos * temp) / (self.l * (4.0 / 3.0 - self.mp * cos ** 2 / total))
        xacc = temp - self.mp * self.l * thacc * cos / total
        x, xd, th, thd = x + self.tau * xd, xd + self.tau * xacc, th + self.tau * thd, thd + self.tau * thacc
        self.state = np.array([x, xd, th, thd]); self.t += 1
        terminated = bool(abs(x) > self.x_thr or abs(th) > self.theta_thr)   # a real end: pole fell
        truncated = bool(self.t >= self.max_steps and not terminated)        # time limit: NOT a real end
        return self.state.astype(np.float32), 1.0, terminated, truncated

def render_cartpole(state, ax, t=None):
    x, _, th, _ = state
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-0.4, 1.4); ax.set_aspect("equal"); ax.grid(False)
    ax.axhline(0, color=AXIS, lw=1)
    ax.add_patch(Rectangle((x - 0.25, 0), 0.5, 0.25, fc=PALETTE[0], ec="none"))
    ax.plot([x, x + 1.0 * math.sin(th)], [0.25, 0.25 + 1.0 * math.cos(th)], color=PALETTE[1], lw=4, solid_capstyle="round")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    if t is not None: ax.set_title(f"t = {t}", fontsize=10, loc="center")

seed_everything(0)
cp = CartPole(); obs = cp.reset(); frames = [(0, obs)]
for t in range(1, 40):
    obs, r, term, trunc = cp.step(np.random.randint(2))
    if t % 8 == 0: frames.append((t, obs))
    if term: frames.append((t, obs)); break
fig, axes = plt.subplots(1, len(frames), figsize=(2.4 * len(frames), 2.3))
for ax, (t, s) in zip(axes, frames): render_cartpole(s, ax, t)
fig.suptitle("A random policy loses the pole within a few dozen steps", x=0.01, ha="left", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()
"""))

CELLS.append(md(r"""
## Building DQN step by step

Three components: a Q-network, a replay buffer, and the update rule. The loss for a minibatch of
transitions $(s, a, r, s', d)$ ($d = 1$ if $s'$ is terminal) is

$$\mathcal L(\theta) = \mathbb E\Big[\, \ell_{\text{Huber}}\big(Q(s,a;\theta),\; r + \gamma (1-d) \max_{a'} Q(s',a';\theta^-)\big)\Big].$$

The Huber loss behaves like MSE near zero and like an absolute loss for large errors, which keeps
occasional wild targets from dominating the gradient.

A subtle but important detail: **time-limit truncation is not termination.** When CartPole stops at
step 500 the pole is still up, so the correct target still bootstraps from $Q(s', \cdot)$. Only a
genuinely terminal transition (pole fell) gets $d = 1$.
"""))

CELLS.append(code(r"""
class QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))
    def forward(self, obs):
        return self.net(obs)

class ReplayBuffer:
    '''A ring buffer of transitions stored as NumPy arrays; sample() returns torch tensors.'''
    def __init__(self, capacity, obs_dim):
        self.obs = np.zeros((capacity, obs_dim), np.float32); self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros(capacity, np.int64); self.rew = np.zeros(capacity, np.float32); self.done = np.zeros(capacity, np.float32)
        self.capacity, self.ptr, self.size = capacity, 0, 0
    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = o, a, r, o2, d
        self.ptr = (self.ptr + 1) % self.capacity; self.size = min(self.size + 1, self.capacity)
    def sample(self, n):
        idx = np.random.randint(0, self.size, size=n)
        return (torch.as_tensor(self.obs[idx]), torch.as_tensor(self.act[idx]), torch.as_tensor(self.rew[idx]),
                torch.as_tensor(self.next_obs[idx]), torch.as_tensor(self.done[idx]))

def dqn_update(q, q_target, opt, batch, gamma, double=False):
    obs, act, rew, next_obs, done = batch
    q_sa = q(obs).gather(1, act[:, None]).squeeze(1)                    # Q(s, a) for the actions taken
    with torch.no_grad():
        if double:   # Double DQN: select with the online net, evaluate with the target net
            best = q(next_obs).argmax(1, keepdim=True)
            q_next = q_target(next_obs).gather(1, best).squeeze(1)
        else:        # DQN: max over the target net
            q_next = q_target(next_obs).max(1).values
        target = rew + gamma * (1.0 - done) * q_next
    loss = F.smooth_l1_loss(q_sa, target)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(q.parameters(), 10.0)
    opt.step()
    return loss.item()

def evaluate_policy(env, act_fn, n_episodes=5):
    '''Average undiscounted return of a deterministic act_fn(obs) -> action.'''
    rets = []
    for _ in range(n_episodes):
        obs, total = env.reset(), 0.0
        while True:
            obs, r, term, trunc = env.step(act_fn(obs)); total += r
            if term or trunc: break
        rets.append(total)
    return float(np.mean(rets))
"""))

CELLS.append(code(r"""
def train_dqn(seed=0, total_steps=40_000, gamma=0.99, lr=1e-3, batch_size=128, buffer_size=50_000,
              learning_starts=1_000, train_every=4, tau=0.005, eps_start=1.0, eps_end=0.05, eps_decay_steps=10_000,
              double=False, eval_every=2_000, verbose=True):
    '''tau: Polyak (soft) target update rate, theta_target <- (1-tau) theta_target + tau theta, every step.'''
    seed_everything(seed)
    env, eval_env = CartPole(), CartPole()
    q = QNet(env.obs_dim, env.n_actions); q_target = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_size, env.obs_dim)
    log = dict(ep_end_step=[], ep_return=[], loss=[], eval_step=[], eval_return=[], eps=[])
    greedy = lambda o: int(q(torch.as_tensor(o)).argmax())

    obs, ep_ret = env.reset(), 0.0
    for step in range(1, total_steps + 1):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)
        with torch.no_grad():
            a = np.random.randint(env.n_actions) if np.random.rand() < eps else greedy(obs)
        next_obs, r, term, trunc = env.step(a); ep_ret += r
        buf.add(obs, a, r, next_obs, float(term))            # bootstrap through truncation, not termination
        obs = next_obs
        if term or trunc:
            log["ep_end_step"].append(step); log["ep_return"].append(ep_ret); log["eps"].append(eps)
            obs, ep_ret = env.reset(), 0.0
        if step >= learning_starts and step % train_every == 0:
            log["loss"].append(dqn_update(q, q_target, opt, buf.sample(batch_size), gamma, double))
        with torch.no_grad():                                  # soft target update
            for p, p_t in zip(q.parameters(), q_target.parameters()):
                p_t.mul_(1 - tau).add_(tau * p)
        if step % eval_every == 0:
            with torch.no_grad():
                ev = evaluate_policy(eval_env, greedy)
            log["eval_step"].append(step); log["eval_return"].append(ev)
            if verbose and step % (eval_every * 5) == 0:
                print(f"step {step:6d} | eps {eps:.2f} | last-10 train return {np.mean(log['ep_return'][-10:]):6.1f} | eval {ev:6.1f}")
    return q, log
"""))

CELLS.append(code(r"""
with Timer("DQN (3 seeds)"):
    dqn_runs = [train_dqn(seed=s, verbose=(s == 0)) for s in range(3)]
q_dqn, dqn_log = max(dqn_runs, key=lambda r: r[1]["eval_return"][-1])      # keep the best final model for the Q-value plot
"""))

CELLS.append(code(r"""
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
ax = axes[0]
for s, (_, lg) in enumerate(dqn_runs):
    ax.plot(lg["eval_step"], lg["eval_return"], marker="o", ms=3, label=f"seed {s}")
finish(ax, "DQN on CartPole: greedy evaluation return", "environment steps", "episode return")
ax = axes[1]
_, lg0 = dqn_runs[0]
ax.plot(lg0["ep_end_step"], smooth(lg0["ep_return"], 10), label="training episodes (ε-greedy), seed 0")
finish(ax, "Training-episode return (ε-greedy, seed 0)", "environment steps", "episode return", legend=False)
ax = axes[2]
ax.plot(np.arange(len(lg0["loss"])) * 4 + 1_000, smooth(lg0["loss"], 200))
finish(ax, "Huber TD loss (seed 0, smoothed)", "environment steps", "loss")
plt.tight_layout(); plt.show()
"""))

CELLS.append(code(r"""
# What does the learned Q-function believe? Trace Q(s, left), Q(s, right) along an episode where the agent
# acts greedily 70% of the time and randomly otherwise, so that the pole eventually falls.
seed_everything(1)
env = CartPole(); obs = env.reset(); qs, angles = [], []
with torch.no_grad():
    while True:
        qv = q_dqn(torch.as_tensor(obs)); qs.append(qv.numpy()); angles.append(obs[2])
        a = int(qv.argmax()) if np.random.rand() < 0.7 else np.random.randint(2)
        obs, r, term, trunc = env.step(a)
        if term or trunc: break
qs = np.array(qs)
fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
axes[0].plot(qs[:, 0], label="Q(s, push left)"); axes[0].plot(qs[:, 1], label="Q(s, push right)")
axes[0].axhline(1 / (1 - 0.99), color=MUTED, lw=1); axes[0].text(0, 1 / (1 - 0.99) - 6, "1/(1-γ) = 100: value of balancing forever", color=INK2, fontsize=9)
axes[0].set_ylim(0, 108); finish(axes[0], f"Learned Q-values along a noisy episode (length {len(qs)}, pole falls at the end)", None, "Q")
axes[1].plot(np.degrees(angles), color=PALETTE[2]); axes[1].axhline(12, color=MUTED, lw=1); axes[1].axhline(-12, color=MUTED, lw=1)
finish(axes[1], "pole angle (episode ends beyond ±12°)", "time step", "degrees", legend=False)
plt.tight_layout(); plt.show()
print("While the pole is balanced both Q-values sit near the ceiling; as the angle drifts toward the limit the values fall,")
print("and the gap between them says which push the network thinks will rescue the pole.")
"""))

CELLS.append(md(r"""
**What to notice.** Q-values sit near the discounted-return ceiling $1/(1-\gamma) = 100$ while the pole is
safe and drop as failure approaches. Look at the three seeds in the evaluation plot: DQN on CartPole is
notoriously **seed-sensitive**. Curves climb, collapse and recover, because a greedy policy flips
abruptly whenever two Q-values cross, and every flip changes the data distribution in the buffer.
Soft (Polyak) target updates, larger batches, Double DQN and more steps all make it steadier; none make it
monotone. Reporting several seeds is not optional in RL.

**The deadly triad in one sentence.** Bootstrapping (targets depend on our own estimates) +
off-policy data (replay contains actions the current policy would not take) + function approximation
(an update to one state changes others) can create a feedback loop where estimates grow without
bound. DQN's tricks don't eliminate the triad; they slow the loop enough for learning to win.

### Test your knowledge
"""))

CELLS += quiz_cells("5.1", "What is the purpose of the target network in DQN?",
    ["To act in the environment while the online network trains",
     "To keep the regression target fixed for a while so the update does not chase a moving target",
     "To reduce the number of parameters",
     "To make the algorithm on-policy"], "B",
    "Without it the target r + gamma max Q(s') changes after every gradient step, because it is computed with the "
    "same weights being updated. Freezing a copy turns each interval into an ordinary supervised regression.")

CELLS += quiz_cells("5.2", "Why is experience replay legal for Q-learning but NOT for the policy-gradient methods coming up in Part 6?",
    ["Replay buffers only store discrete actions",
     "Q-learning is off-policy: its target does not depend on which policy generated the transition",
     "Policy gradients don't use neural networks",
     "It is legal for both; DQN just happens to use it"], "B",
    "Q-learning's target uses max over a', so old transitions from any behaviour policy are valid samples. The vanilla "
    "policy gradient is an expectation under the *current* policy, so stale samples bias it, which is why PPO needs "
    "importance ratios to reuse data even a few epochs old.")

CELLS += quiz_cells("5.3", "CartPole ends after 500 steps by a time limit. If we stored done=1 for that transition, what would go wrong?",
    ["Nothing; 500 steps is the maximum anyway",
     "The network would learn that the state at step 500 is worth zero future reward, even though the pole is still balanced, biasing Q-values near the time limit",
     "The replay buffer would overflow",
     "Epsilon would stop decaying"], "B",
    "Truncation is a property of the experiment, not the environment. The observation at step 500 looks like any "
    "other balanced state, so labelling it terminal teaches a contradiction. Bootstrapping through truncation (done=0) "
    "is correct; a common bug in tutorials.")

CELLS += exercise_cells("5.4",
    r"""
### Exercise 5.4 · The Double DQN target

Implement `double_dqn_target(q, q_target, rew, next_obs, done, gamma)` returning the tensor of targets

$$y = r + \gamma (1-d)\, Q_{\text{target}}\big(s', \arg\max_{a'} Q_{\text{online}}(s', a')\big).$$

`q` and `q_target` are callables mapping a batch of observations to a `(batch, n_actions)` tensor.
""",
    r"""
def double_dqn_target(q, q_target, rew, next_obs, done, gamma):
    with torch.no_grad():
        # YOUR CODE HERE
        raise NotImplementedError
""",
    r"""
def _tests(fn):
    q = lambda o: torch.tensor([[1.0, 2.0], [3.0, 0.0]])
    qt = lambda o: torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    y = fn(q, qt, torch.tensor([1.0, 1.0]), torch.zeros(2, 4), torch.tensor([0.0, 0.0]), 0.5)
    assert y is not None and tuple(y.shape) == (2,), "return a 1-D tensor of shape (batch,)"
    assert torch.allclose(y, torch.tensor([11.0, 16.0])), f"expected [11, 16], got {y.tolist()} (are you selecting with the ONLINE net?)"
    y2 = fn(q, qt, torch.tensor([1.0, 1.0]), torch.zeros(2, 4), torch.tensor([0.0, 1.0]), 0.5)
    assert torch.allclose(y2, torch.tensor([11.0, 1.0])), "terminal transitions must not bootstrap"
""",
    r"""
def double_dqn_target(q, q_target, rew, next_obs, done, gamma):
    with torch.no_grad():
        best = q(next_obs).argmax(1, keepdim=True)                 # selection: online net
        q_next = q_target(next_obs).gather(1, best).squeeze(1)      # evaluation: target net
        return rew + gamma * (1.0 - done) * q_next
""", "double_dqn_target")
