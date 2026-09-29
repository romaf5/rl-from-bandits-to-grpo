from common import video, md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 9 · PPO: proximal policy optimisation

PPO (Schulman et al., 2017) keeps TRPO's insight, *don't let the policy move too far from the one that
collected the data*, but enforces it with a first-order trick instead of a constrained second-order
solve. Define the probability ratio $\rho_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$
and use the **clipped surrogate objective**

$$L^{\text{CLIP}}(\theta) = \mathbb E_t\Big[\min\big(\rho_t \hat A_t,\; \text{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon)\, \hat A_t\big)\Big].$$

Read it case by case:

* $\hat A_t > 0$ (good action): we would like to raise $\rho_t$, but once $\rho_t > 1 + \epsilon$ the clipped
  term is flat and the min picks it, so **the gradient is zero**. No incentive to move further.
* $\hat A_t < 0$ (bad action): we would like to lower $\rho_t$, but below $1 - \epsilon$ the gradient again vanishes.
* The `min` makes the objective a **pessimistic lower bound**: the clip only ever removes incentives,
  never adds them, so the unclipped objective is always used when it is *worse*.

Because clipping kills the gradient of samples that already moved far, we can safely take **many
gradient steps on the same batch** (several epochs of minibatches). That is the source of PPO's sample
efficiency compared to A2C, and the whole reason it became the default for everything from robotics to
RLHF.
"""))
CELLS.append(video(9))

CELLS.append(code(r"""
ratio = np.linspace(0.5, 1.5, 300); eps = 0.2
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
for ax, A in zip(axes, [1.0, -1.0]):
    unclipped = ratio * A
    clipped = np.minimum(ratio * A, np.clip(ratio, 1 - eps, 1 + eps) * A)
    ax.plot(ratio, unclipped, color=MUTED, lw=1.5, label="unclipped  ρ·A")
    ax.plot(ratio, clipped, label="$L^{CLIP}$")
    ax.axvline(1 - eps, color=AXIS, lw=1); ax.axvline(1 + eps, color=AXIS, lw=1)
    flat = ratio > 1 + eps if A > 0 else ratio < 1 - eps
    ax.fill_between(ratio, clipped.min() - 0.1, clipped.max() + 0.1, where=flat, color=PALETTE[1], alpha=0.10, linewidth=0)
    ax.text(1 + eps + 0.02 if A > 0 else 0.52, clipped.max() + 0.02, "zero gradient", color=INK2, fontsize=9)
    finish(ax, f"advantage A = {A:+.0f}", "probability ratio ρ", "objective")
plt.tight_layout(); plt.show()
"""))

CELLS.append(md(r"""
## The full recipe

The objective above is the heart; a production PPO adds several details, all of which matter:

| Ingredient | Why |
|---|---|
| GAE advantages, normalised per minibatch | low-variance credit assignment; scale-free updates |
| Value loss $\tfrac12 (V_\theta(s_t) - \hat R_t)^2$ | the critic is trained jointly (optionally clipped too) |
| Entropy bonus $-\beta\, \mathcal H[\pi_\theta(\cdot|s)]$ | discourages premature collapse to a deterministic policy |
| Several epochs of minibatch SGD per batch | reuses data; the clip makes it safe |
| Global gradient-norm clipping (0.5) | robustness against rare huge advantages |
| Diagnostics: approx. KL, clip fraction, explained variance | tells you whether the update is too aggressive or the critic is useless |

We'll implement all of it in about 40 lines.
"""))

CELLS.append(code(r"""
def ppo_update(policy, value, opt, ro, gamma=0.99, lam=0.95, clip=0.2, epochs=10, minibatch=256,
               vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5, record_ratios=False):
    with torch.no_grad():
        v, v_next = value(ro.obs).numpy(), value(ro.next_obs).numpy()
        old_logp = policy.dist(ro.obs).log_prob(ro.act)
    adv, ret = compute_gae(ro.rew, v, v_next, ro.term, ro.done, gamma, lam)
    adv, ret = torch.as_tensor(adv), torch.as_tensor(ret)
    N, params = len(ro.rew), list(policy.parameters()) + list(value.parameters())
    stats = defaultdict(list); ratio_snapshots = {}
    for epoch in range(epochs):
        perm = torch.randperm(N)
        for start in range(0, N, minibatch):
            idx = perm[start:start + minibatch]
            dist = policy.dist(ro.obs[idx]); logp = dist.log_prob(ro.act[idx])
            ratio = torch.exp(logp - old_logp[idx])                         # pi_theta / pi_old
            mb_adv = adv[idx]; mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            pg_loss = -torch.min(ratio * mb_adv, torch.clamp(ratio, 1 - clip, 1 + clip) * mb_adv).mean()
            v_loss = 0.5 * F.mse_loss(value(ro.obs[idx]), ret[idx])
            entropy = dist.entropy().mean()
            loss = pg_loss + vf_coef * v_loss - ent_coef * entropy
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, max_grad_norm); opt.step()
            with torch.no_grad():                                           # diagnostics
                stats["approx_kl"].append(((ratio - 1) - torch.log(ratio)).mean().item())   # k3 estimator of KL(old||new)
                stats["clipfrac"].append(((ratio - 1).abs() > clip).float().mean().item())
                stats["entropy"].append(entropy.item())
        if record_ratios and epoch in (0, epochs - 1):
            with torch.no_grad():
                ratio_snapshots[epoch] = torch.exp(policy.dist(ro.obs).log_prob(ro.act) - old_logp).numpy()
    with torch.no_grad():                                                   # explained variance of the critic
        v_pred = value(ro.obs); ev = 1 - (ret - v_pred).var() / (ret.var() + 1e-8)
    out = {k: float(np.mean(vals)) for k, vals in stats.items()}; out["explained_var"] = ev.item()
    return out, ratio_snapshots

def train_ppo(seed=0, iters=30, batch_steps=2048, lr=1e-3, record_at=None, **kw):
    seed_everything(seed)
    env = CartPole(); policy, value = Policy(env.obs_dim, env.n_actions), ValueNet(env.obs_dim)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=lr, eps=1e-5)
    curve, steps, diag, total, ratios = [], [], defaultdict(list), 0, None
    for it in range(iters):
        ro = collect_rollout(env, policy, batch_steps)
        stats, snaps = ppo_update(policy, value, opt, ro, record_ratios=(it == record_at), **kw)
        if snaps: ratios = snaps
        total += len(ro.rew); curve.append(np.mean(ro.ep_returns) if ro.ep_returns else curve[-1]); steps.append(total)
        for k, v_ in stats.items(): diag[k].append(v_)
    return policy, value, np.array(curve), np.array(steps), diag, ratios
"""))

CELLS.append(code(r"""
with Timer("PPO"):
    curves, steps, diags = [], [], []
    for seed in range(N_SEEDS):
        pol_ppo, val_ppo, c, s, d, r_snap = train_ppo(seed, iters=TOTAL_STEPS // 2048, record_at=8)
        curves.append(c); steps.append(s); diags.append(d)
        if seed == 0: ppo_ratios = r_snap
    pg_results["PPO (clip 0.2)"] = (curves, steps)
    print(f"PPO final return (mean over seeds): {np.mean([c[-1] for c in curves]):6.1f}")
"""))

CELLS.append(code(r"""
fig, ax = plt.subplots(figsize=(9, 4))
for name, (curves, steps) in pg_results.items():
    plot_runs(ax, curves, label=name, k=3, x=steps[0])
finish(ax, f"Policy-gradient family on CartPole, equal environment budget ({N_SEEDS} seeds)", "environment steps", "mean episode return")
plt.show()
"""))

CELLS.append(code(r"""
fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for ax, key, title in zip(axes, ["approx_kl", "clipfrac", "entropy", "explained_var"],
                          ["approx. KL per update", "fraction of clipped ratios", "policy entropy", "critic explained variance"]):
    plot_runs(ax, [d[key] for d in diags], color=PALETTE[0])
    finish(ax, title, "update", None, legend=False)
axes[3].set_ylim(-0.1, 1.05)
plt.suptitle("PPO diagnostics: what a healthy run looks like", x=0.01, ha="left", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(8, 3))
bins = np.linspace(0.6, 1.4, 60)
for ep, r in ppo_ratios.items():
    ax.hist(r, bins=bins, alpha=0.6, label=f"after epoch {ep + 1}", edgecolor="none")
ax.axvline(0.8, color=MUTED, lw=1); ax.axvline(1.2, color=MUTED, lw=1)
finish(ax, "Probability ratios within one PPO update (iteration 9)", "ρ = π_new / π_old", "count")
plt.show()
"""))

CELLS.append(md(r"""
**Reading the diagnostics.** Before the update every ratio is exactly 1. After the first epoch (8 minibatch
steps) they have already spread a little; by the last epoch a visible fraction sits at or beyond the clip
boundaries, and those samples contribute no gradient any more. Healthy runs show approximate KL around 0.005–0.03 per update,
clip fraction 0.05–0.3, entropy decreasing slowly, and explained variance climbing toward 1. If KL
explodes, lower the learning rate or the number of epochs; if explained variance stays near 0 *while
returns are still improving*, the critic is useless and the advantages are mostly noise.

One trap in the last panel: once every episode reaches the 500-step cap, every state is worth about
$1/(1-\gamma)$ and the return targets have almost no variance left to explain, so explained variance
collapses toward 0 (or goes negative) even though the critic is fine. Diagnostics need context.

## Variants you will meet in the wild

* **PPO-penalty**: replace the clip with $-\beta\, \mathrm{KL}(\pi_{\text{old}} \| \pi_\theta)$ and adapt
  $\beta$ up or down depending on whether the measured KL exceeds a target. Rarely used for control,
  but the per-token KL penalty in RLHF (Part 12) is its cousin.
* **Value clipping**: clip the value prediction around its old value the same way. Popular in
  implementations, of dubious benefit.
* **Dual-clip PPO**: for $\hat A_t < 0$, also cap $\rho_t$ from above (e.g. at 3) so that a single
  very-off-policy sample cannot dominate.
* **Early stopping on KL**: stop the epochs when approximate KL exceeds a target (e.g. 0.015–0.05).
  Simple and effective; many LLM trainers do this.

### Test your knowledge
"""))

CELLS += quiz_cells("9.1", "For a sample with negative advantage and ratio rho = 0.7 (clip eps = 0.2), what is the gradient contribution of the PPO objective?",
    ["Large negative, pushing the probability down further",
     "Zero: the clipped term is active and it is constant in theta",
     "Positive, pushing the probability back up",
     "Undefined"], "B",
    "For A < 0 the objective is max(rho*A, clip(rho)*A) in absolute terms; below 1 - eps the clipped branch wins and "
    "is flat, so this sample has already been pushed down 'enough' and is ignored. Note the asymmetry: if rho were 0.7 "
    "with A > 0 the unclipped branch (which is lower) would be active, and the gradient would pull the probability back up.")

CELLS += quiz_cells("9.2", "Why can PPO run 10 epochs on one batch while REINFORCE / A2C take a single gradient step?",
    ["PPO uses a replay buffer",
     "The importance ratio corrects for the policy drifting away from the data-collecting policy, and the clip prevents that drift from becoming large enough to make the correction unreliable",
     "PPO's learning rate is smaller",
     "PPO's critic makes the data on-policy again"], "B",
    "The surrogate L is an importance-sampled estimate that is valid only while the new policy stays close to the old. "
    "Clipping (plus gradient clipping and KL monitoring) keeps it in that regime, so re-using the batch is cheap extra signal.")

CELLS += quiz_cells("9.3", "Your PPO run shows approx_kl = 0.2 per update and a clip fraction of 0.8. What is the most likely fix?",
    ["Increase the number of epochs",
     "Lower the learning rate or the number of epochs, or add early stopping on KL; the policy is moving far too much per update",
     "Increase gamma",
     "Remove advantage normalisation"], "B",
    "Those numbers mean nearly every sample is outside the trust region and the policy changed drastically. That is the "
    "instability PPO exists to prevent. Typical healthy values are KL ~ 0.01 and clip fraction ~ 0.1-0.3.")

CELLS += exercise_cells("9.4",
    r"""
### Exercise 9.4 · The clipped surrogate loss

Implement `ppo_clip_loss(logp_new, logp_old, adv, eps)` returning the **scalar loss to minimise**
(the negative clipped objective, averaged over the batch). All inputs are 1-D tensors.
""",
    r"""
def ppo_clip_loss(logp_new, logp_old, adv, eps=0.2):
    # YOUR CODE HERE
    raise NotImplementedError
""",
    r"""
def _tests(fn):
    logp_old = torch.log(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    logp_new = torch.log(torch.tensor([0.5, 0.75, 0.25, 0.75]))     # ratios 1.0, 1.5, 0.5, 1.5
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    loss = fn(logp_new, logp_old, adv, 0.2)
    assert loss is not None and loss.dim() == 0, "return a scalar"
    # per-sample objective: min(1*1, 1*1)=1 ; min(1.5, 1.2)=1.2 ; min(-0.5, -0.8)=-0.8 ; min(-1.5, -1.2)=-1.5
    expected = -torch.tensor([1.0, 1.2, -0.8, -1.5]).mean()
    assert torch.isclose(loss, expected, atol=1e-6), f"expected {expected.item():.4f}, got {loss.item():.4f}"
""",
    r"""
def ppo_clip_loss(logp_new, logp_old, adv, eps=0.2):
    ratio = torch.exp(logp_new - logp_old)
    return -torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv).mean()
""", "ppo_clip_loss")
