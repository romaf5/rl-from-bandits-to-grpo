from common import md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 8 · TRPO: trust region policy optimisation

## The step-size problem

In supervised learning a too-large step just makes the loss go up for one iteration. In on-policy RL
it is worse: a bad policy update **changes the data you collect next**. Collapse the policy into a bad
region and you may never sample your way out. The Adam learning rate in Parts 6–7 was a fragile
compromise; we want a step size defined in terms of *how much the policy changes*, not how much the
parameters change.

## The surrogate objective

The **performance difference lemma** (Kakade & Langford, 2002) says, for any two policies,

$$J(\pi') - J(\pi) = \frac{1}{1-\gamma}\,\mathbb E_{s \sim d^{\pi'},\, a \sim \pi'}\big[A^\pi(s,a)\big].$$

The catch is $d^{\pi'}$: it needs states visited by the *new* policy, which we don't have. If $\pi'$ is
close to $\pi$ we can approximate $d^{\pi'} \approx d^\pi$ and importance-sample the actions:

$$L_\pi(\pi') = \mathbb E_{s \sim d^\pi,\, a \sim \pi}\Big[\frac{\pi'(a|s)}{\pi(a|s)} A^\pi(s,a)\Big].$$

TRPO (Schulman et al., 2015) proves $J(\pi') \ge J(\pi) + L_\pi(\pi') - C \cdot \max_s \mathrm{KL}(\pi \| \pi')$
and turns it into a practical constrained problem:

$$\max_\theta\; L_{\theta_{\text{old}}}(\theta) \quad \text{s.t.} \quad \bar{\mathrm{KL}}(\theta_{\text{old}} \| \theta) \le \delta.$$

## Solving it: natural gradient + conjugate gradient + line search

Expand to second order around $\theta_{\text{old}}$: the objective is linear, $g^\top \Delta\theta$, and
the KL is quadratic, $\tfrac12 \Delta\theta^\top F \Delta\theta$ with $F$ the **Fisher information
matrix** (the KL's Hessian). The solution is the **natural gradient** step

$$\Delta\theta = \sqrt{\frac{2\delta}{g^\top F^{-1} g}}\; F^{-1} g .$$

$F$ is (#params)² so we never form it. Instead we solve $F x = g$ with **conjugate gradient**, which
only needs Fisher-vector products $Fv$, and those come for free from automatic differentiation as
$\nabla_\theta \big( (\nabla_\theta \bar{\mathrm{KL}})^\top v \big)$. Finally, because the quadratic model is
only approximate, a **backtracking line search** shrinks the step until the real KL is within $\delta$
and the real surrogate improves.
"""))

CELLS.append(code(r"""
def flat_params(model):
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])

def set_flat_params(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel(); p.data.copy_(flat[i:i + n].view_as(p)); i += n

def conjugate_gradient(Avp, b, iters=10, tol=1e-10):
    '''Solve A x = b for symmetric positive-definite A given only the matrix-vector product Avp(v) = A v.'''
    x, r = torch.zeros_like(b), b.clone()
    p, rr = r.clone(), r @ r
    for _ in range(iters):
        Ap = Avp(p)
        alpha = rr / (p @ Ap + 1e-12)
        x = x + alpha * p; r = r - alpha * Ap
        rr_new = r @ r
        if rr_new < tol: break
        p = r + (rr_new / rr) * p; rr = rr_new
    return x

def trpo_update(policy, value, opt_v, ro, gamma=0.99, lam=0.95, max_kl=0.01, damping=0.1,
                cg_iters=10, backtrack_iters=10, value_steps=20):
    with torch.no_grad():
        v, v_next = value(ro.obs).numpy(), value(ro.next_obs).numpy()
    adv, ret = compute_gae(ro.rew, v, v_next, ro.term, ro.done, gamma, lam)
    adv, ret = torch.as_tensor(adv), torch.as_tensor(ret)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    obs, act = ro.obs, ro.act
    with torch.no_grad():
        old_dist = policy.dist(obs); old_logp = old_dist.log_prob(act)
    old_dist = torch.distributions.Categorical(probs=old_dist.probs.detach())

    def surrogate():                       # L(theta) = E[ pi_theta / pi_old * A ]
        return (torch.exp(policy.dist(obs).log_prob(act) - old_logp) * adv).mean()
    def mean_kl():                         # KL(pi_old || pi_theta), zero at theta_old with Hessian = Fisher
        return torch.distributions.kl_divergence(old_dist, policy.dist(obs)).mean()
    def fisher_vector_product(vec):
        grad_kl = flat_grad(mean_kl(), policy, create_graph=True)
        return flat_grad(grad_kl @ vec, policy) + damping * vec

    L_old = surrogate()
    g = flat_grad(L_old, policy)                                  # policy gradient
    step_dir = conjugate_gradient(fisher_vector_product, g, cg_iters)    # ~ F^{-1} g
    shs = 0.5 * step_dir @ fisher_vector_product(step_dir)               # 1/2 s^T F s
    full_step = torch.sqrt(max_kl / shs) * step_dir                      # scaled so the quadratic KL model hits max_kl
    expected_improve = g @ full_step
    theta_old = flat_params(policy)
    info = dict(kl=0.0, improve=0.0, frac=0.0, accepted=False)
    for i in range(backtrack_iters):                                     # backtracking line search
        frac = 0.5 ** i
        set_flat_params(policy, theta_old + frac * full_step)
        with torch.no_grad():
            improve, kl = (surrogate() - L_old).item(), mean_kl().item()
        if kl <= 1.5 * max_kl and improve > 0 and improve / (frac * expected_improve.item() + 1e-12) > 0.1:
            info.update(kl=kl, improve=improve, frac=frac, accepted=True); break
    if not info["accepted"]:
        set_flat_params(policy, theta_old)
    for _ in range(value_steps):
        v_loss = F.mse_loss(value(obs), ret); opt_v.zero_grad(); v_loss.backward(); opt_v.step()
    return info, (obs, act, adv, old_logp, old_dist, g, full_step)

def train_trpo(seed=0, iters=30, batch_steps=2048, gamma=0.99, lam=0.95, max_kl=0.01, snapshot_at=None):
    seed_everything(seed)
    env = CartPole(); policy, value = Policy(env.obs_dim, env.n_actions), ValueNet(env.obs_dim)
    opt_v = torch.optim.Adam(value.parameters(), lr=3e-3)
    curve, steps, kls, total, snapshot = [], [], [], 0, None
    for it in range(iters):
        ro = collect_rollout(env, policy, batch_steps)
        if snapshot_at is not None and it == snapshot_at:
            snapshot = (copy.deepcopy(policy), copy.deepcopy(value), ro)
        info, _ = trpo_update(policy, value, opt_v, ro, gamma, lam, max_kl)
        total += len(ro.rew); curve.append(np.mean(ro.ep_returns) if ro.ep_returns else curve[-1]); steps.append(total); kls.append(info["kl"])
    return policy, value, np.array(curve), np.array(steps), np.array(kls), snapshot
"""))

CELLS.append(code(r"""
with Timer("TRPO"):
    curves, steps, kl_curves = [], [], []
    for seed in range(N_SEEDS):
        pol_trpo, val_trpo, c, s, k, snap = train_trpo(seed, iters=TOTAL_STEPS // 2048, snapshot_at=6)
        curves.append(c); steps.append(s); kl_curves.append(k)
        if seed == 0: trpo_snapshot = snap
    pg_results["TRPO (δ=0.01)"] = (curves, steps)
    print(f"TRPO final return (mean over seeds): {np.mean([c[-1] for c in curves]):6.1f}")
"""))

CELLS.append(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
for name, (curves, steps) in pg_results.items():
    plot_runs(axes[0], curves, label=name, k=3, x=steps[0])
finish(axes[0], "TRPO joins the comparison", "environment steps", "mean episode return")
plot_runs(axes[1], kl_curves, label="KL(old‖new) after each update", color=PALETTE[0])
axes[1].axhline(0.01, color=MUTED, lw=1); axes[1].text(0, 0.0103, "δ = 0.01", color=INK2, fontsize=9)
finish(axes[1], "The constraint is respected", "update", "mean KL", legend=False)
plt.tight_layout(); plt.show()
"""))

CELLS.append(md(r"""
## Why the natural gradient direction is special

Take a batch from a snapshot of training and walk along two directions from the current parameters:
the natural-gradient step $F^{-1} g$ (scaled so that $s=1$ is exactly TRPO's step) and the plain
gradient $g$ (scaled to the same Euclidean length). We measure the surrogate objective and the true KL
along each ray.
"""))

CELLS.append(code(r"""
pol_s, val_s, ro_s = trpo_snapshot
pol_s = copy.deepcopy(pol_s); opt_dummy = torch.optim.Adam(val_s.parameters(), lr=0.0)
_, (obs, act, adv, old_logp, old_dist, g, full_step) = trpo_update(pol_s, val_s, opt_dummy, ro_s, backtrack_iters=0, value_steps=0)
theta0 = flat_params(pol_s)
grad_step = g / g.norm() * full_step.norm()                 # plain gradient, same length as the TRPO step

def measure(direction, scales):
    Ls, KLs = [], []
    for s in scales:
        set_flat_params(pol_s, theta0 + s * direction)
        with torch.no_grad():
            d = pol_s.dist(obs)
            Ls.append((torch.exp(d.log_prob(act) - old_logp) * adv).mean().item())
            KLs.append(torch.distributions.kl_divergence(old_dist, d).mean().item())
    set_flat_params(pol_s, theta0)
    return np.array(Ls), np.array(KLs)

scales = np.linspace(0, 4, 41)
L_nat, KL_nat = measure(full_step, scales); L_grad, KL_grad = measure(grad_step, scales)
fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
axes[0].plot(scales, L_nat, label="natural gradient direction  $F^{-1}g$"); axes[0].plot(scales, L_grad, label="plain gradient direction  $g$")
axes[0].axvline(1, color=MUTED, lw=1); axes[0].text(1.05, axes[0].get_ylim()[0], "TRPO step", color=INK2, fontsize=9)
finish(axes[0], "Surrogate objective along each ray", "step scale s", "$L(\\theta_0 + s\\, d)$")
axes[1].plot(scales, KL_nat); axes[1].plot(scales, KL_grad)
axes[1].axhline(0.01, color=MUTED, lw=1); axes[1].text(0.05, 0.011, "δ = 0.01", color=INK2, fontsize=9); axes[1].axvline(1, color=MUTED, lw=1)
finish(axes[1], "KL to the old policy along each ray", "step scale s", "mean KL", legend=False)
axes[2].plot(KL_nat, L_nat); axes[2].plot(KL_grad, L_grad); axes[2].set_xlim(0, 0.1)
axes[2].axvline(0.01, color=MUTED, lw=1); axes[2].text(0.011, axes[2].get_ylim()[0], "δ", color=INK2, fontsize=9)
finish(axes[2], "Improvement bought per unit of KL", "mean KL (zoomed to 0-0.1)", "surrogate objective", legend=False)
plt.tight_layout(); plt.show()
"""))

CELLS.append(md(r"""
Per unit of *parameter* length (left two panels) the plain gradient improves the surrogate faster, but it
also moves the **policy distribution** many times further: its KL explodes while the natural direction's
KL grows gently and hits $\delta$ exactly at the TRPO step. The fair comparison is the right panel,
improvement per unit of KL: for any given policy change we are willing to tolerate, the natural direction buys
more objective. That is what "measuring distance in policy space, not parameter space" means, and it is
why a large plain-gradient step can wreck a policy that a same-length natural step would only nudge.

**Why not just use TRPO everywhere?** Every update needs several Fisher-vector products (each a
double backward pass) and a line search, it is awkward with shared actor-critic parameters and with
dropout / batch-norm, and it does not scale gracefully to billion-parameter models. PPO keeps the
idea and drops the machinery.

### Test your knowledge
"""))

CELLS += quiz_cells("8.1", "Why does TRPO measure the step in KL divergence rather than in Euclidean parameter distance?",
    ["KL is cheaper to compute than a norm",
     "The same parameter change can alter the action distribution a lot or a little depending on where you are; KL measures the change that actually matters for the data you collect",
     "Because the Fisher matrix is diagonal",
     "Euclidean distance is undefined for neural networks"], "B",
    "Policy performance depends on the action distribution, not on the parameterisation. A trust region in KL is "
    "invariant to reparameterisation, which is exactly the property a step-size rule for on-policy learning should have.")

CELLS += quiz_cells("8.2", "What does conjugate gradient compute in TRPO, and why not just invert the Fisher matrix?",
    ["It computes the policy gradient g; inversion is impossible for non-square matrices",
     "It solves F x = g using only Fisher-vector products; forming or inverting F (#params squared) would be far too expensive",
     "It computes the KL divergence exactly",
     "It replaces the line search"], "B",
    "Each Fisher-vector product is one extra backward pass, so ~10 CG iterations cost about 10 backward passes, "
    "independent of the parameter count. Forming F would need (#params)^2 memory.")

CELLS += quiz_cells("8.3", "The line search rejects a step when the real KL exceeds 1.5*delta or the surrogate does not improve. Why is that check necessary at all?",
    ["Because CG is randomised",
     "Because the step was computed from a second-order Taylor model of the KL and a first-order model of the objective, which can be wrong for a step of finite size",
     "Because the value function is inaccurate",
     "It isn't; it only saves computation"], "B",
    "The natural-gradient step is exact only for the quadratic approximation. Backtracking guarantees the two things we "
    "actually care about, monotone improvement of the surrogate and a bounded policy change, hold for the real functions.")

CELLS += exercise_cells("8.4",
    r"""
### Exercise 8.4 · Conjugate gradient

Implement `my_cg(Avp, b, iters=20)` that solves $Ax = b$ for a symmetric positive-definite $A$ given
only a function `Avp(v) -> A @ v`. Use the standard recurrences (residual $r$, search direction $p$,
step $\alpha = r^\top r / p^\top A p$, $\beta = r_{\text{new}}^\top r_{\text{new}} / r^\top r$).
""",
    r"""
def my_cg(Avp, b, iters=20):
    # YOUR CODE HERE
    raise NotImplementedError
""",
    r"""
def _tests(fn):
    torch.manual_seed(0)
    M = torch.randn(6, 6); A = M @ M.T + 0.5 * torch.eye(6)     # SPD
    b = torch.randn(6)
    x = fn(lambda v: A @ v, b, iters=20)
    assert x is not None and x.shape == b.shape, "wrong shape"
    assert torch.allclose(A @ x, b, atol=1e-3), f"residual too large: {(A @ x - b).norm():.4f}"
""",
    r"""
def my_cg(Avp, b, iters=20):
    x, r = torch.zeros_like(b), b.clone(); p, rr = r.clone(), r @ r
    for _ in range(iters):
        Ap = Avp(p); alpha = rr / (p @ Ap)
        x = x + alpha * p; r = r - alpha * Ap
        rr_new = r @ r
        if rr_new < 1e-10: break
        p = r + (rr_new / rr) * p; rr = rr_new
    return x
""", "my_cg")
