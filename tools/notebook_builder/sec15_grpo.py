from common import video, md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 15 · GRPO: group relative policy optimisation

GRPO (Shao et al., 2024, DeepSeekMath) is the algorithm behind DeepSeek-R1 and, in modified form, most
open reasoning models of 2025. It is PPO with the critic replaced by a **group baseline**:

1. For each prompt $x$, sample a group of $G$ responses $\{y_1, \dots, y_G\}$ from the old policy.
2. Score them, $R_i$, and compute **group-normalised advantages**
   $$\hat A_i = \frac{R_i - \operatorname{mean}(R_1..R_G)}{\operatorname{std}(R_1..R_G)},$$
   shared by every token of $y_i$.
3. Take PPO's clipped objective per token, average over each response's tokens, then over the group,
   and subtract a KL penalty **in the loss** using the unbiased, always-positive "k3" estimator:
   $$\mathcal J_{\text{GRPO}} = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|}
   \Big[\min\big(\rho_{i,t}\hat A_i,\; \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat A_i\big)
   - \beta\, \mathbb D_{i,t}\Big], \qquad
   \mathbb D_{i,t} = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1 .$$

That's it. Compared with PPO-RLHF it needs no value model and no GAE, and the baseline is a
Monte Carlo estimate of $V(x)$ computed from the group itself. The division by the group std is
GRPO's distinctive (and controversial, see Part 16) choice: it turns every prompt's rewards into
z-scores, so a lone correct answer among 8 failures gets a huge advantage.

**DeepSeek-R1's recipe on top of GRPO** was almost embarrassingly simple: rule-based rewards only
(accuracy + a format reward for `<think>...</think>` tags), no neural reward model, thousands of steps,
and the reasoning chains grew on their own ("aha moments"). The lesson was that the algorithm matters
less than having a verifiable reward and enough compute.
"""))
CELLS.append(video(15))

CELLS.append(code(r"""
def group_advantages(R, G, normalize_std=True, eps=1e-4):
    '''R: (B,) rewards in blocks of G per prompt -> (B,) advantages.'''
    Rg = R.view(-1, G)
    adv = Rg - Rg.mean(1, keepdim=True)
    if normalize_std:
        adv = adv / (Rg.std(1, keepdim=True) + eps)
    return adv.reshape(-1)

def make_grpo_step(policy, ref, reward_fn=verifier_reward, n_samples=8, clip=0.2, beta=0.04, lr=3e-4,
                   epochs=1, normalize_std=True, gen_fn=None):
    opt = torch.optim.Adam(policy.parameters(), lr=lr); gen_fn = gen_fn or generate
    def step(pairs):
        prompts = batch_prompts(pairs, n_samples)
        seq, mask = gen_fn(policy, prompts)                                   # 1. sample a group per prompt
        R = batch_rewards(pairs, n_samples, seq, reward_fn)                   # 2. score
        adv = group_advantages(R, n_samples, normalize_std)[:, None]          #    group-relative advantages (B, 1)
        with torch.no_grad():
            old_logp = completion_logprobs(policy, seq); ref_logp = completion_logprobs(ref, seq)
        for _ in range(epochs):                                               # 3. clipped objective + KL in the loss
            logp = completion_logprobs(policy, seq)
            ratio = torch.exp(logp - old_logp)
            surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
            kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1           # k3 estimator of KL(pi_theta || pi_ref), per token
            per_token = surrogate - beta * kl
            loss = -((per_token * mask).sum(1) / mask.sum(1)).mean()          # 1/|y_i| inside, 1/G outside
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        Rg = R.view(-1, n_samples)
        return dict(reward=R.mean().item(), kl=((old_logp - ref_logp) * mask).sum(1).mean().item(), length=mask.sum(1).mean().item(),
                    dead_groups=(Rg.std(1) == 0).float().mean().item(), entropy=-(old_logp * mask).sum().item() / mask.sum().item())
    return step

seed_everything(0)
policy_grpo = copy.deepcopy(sft_model)
with Timer("GRPO"):
    log_grpo = train_llm_rl(make_grpo_step(policy_grpo, sft_model), policy_grpo)
llm_results["GRPO"] = log_grpo
plot_llm_runs(llm_results, "GRPO joins the comparison")
"""))

CELLS.append(code(r"""
# Look inside one group: samples, rewards, and the advantages GRPO assigns.
seed_everything(5)
a, b = 14, 17
prompts = batch_prompts([(a, b)], 8); seq, mask = generate(sft_model, prompts)
R = batch_rewards([(a, b)], 8, seq, verifier_reward)
adv_std, adv_mean = group_advantages(R, 8, True), group_advantages(R, 8, False)
print(f"prompt {decode(encode_prompt(a, b))}  (true answer {a + b}), SFT policy samples:")
for i in range(8):
    print(f"  {decode(seq[i, PROMPT_LEN:]):8s} reward {R[i]:.0f}   advantage: GRPO {adv_std[i]:+.2f}   mean-only {adv_mean[i]:+.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
ks = np.arange(1, 8)                                                 # number of correct samples in a group of 8
for normalize, ax, title in [(True, axes[0], "GRPO (divide by group std)"), (False, axes[1], "mean-only baseline (Dr. GRPO)")]:
    a_correct = [group_advantages(torch.tensor([1.0] * k + [0.0] * (8 - k)), 8, normalize)[0].item() for k in ks]
    a_wrong = [group_advantages(torch.tensor([1.0] * k + [0.0] * (8 - k)), 8, normalize)[-1].item() for k in ks]
    ax.plot(ks, a_correct, marker="o", label="advantage of a correct sample"); ax.plot(ks, a_wrong, marker="o", label="advantage of a wrong sample")
    ax.axhline(0, color=AXIS, lw=1); finish(ax, title, "number of correct samples in the group (of 8)", "advantage", legend=(ax is axes[0]))
plt.suptitle("How the group baseline weights samples by prompt difficulty", x=0.01, ha="left", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(smooth(log_grpo["dead_groups"], 5)); finish(ax, "Fraction of groups with identical rewards (zero gradient)", "GRPO step", "fraction", legend=False)
plt.show()
"""))

CELLS.append(md(r"""
Two things to take from these plots:

* **Std-normalisation is a difficulty weighting.** With one correct sample out of eight, GRPO gives it an
  advantage of about +2.5 and each wrong one −0.35; with seven correct, the roles flip. Mean-only
  advantages are symmetric and much smaller for lopsided groups. GRPO therefore pushes hardest on
  prompts the model almost never (or almost always) solves, which Dr. GRPO argues is a *bias* rather than a
  feature (Part 16).
* **Dead groups grow as the model improves.** Once most prompts are solved 8/8 the batch is mostly zero
  advantage: compute spent on samples that teach nothing. This is why DAPO resamples until the batch is full
  of informative groups, and why real training curricula keep feeding harder problems.

### Test your knowledge
"""))

CELLS += quiz_cells("15.1", "What replaces the critic in GRPO?",
    ["A reward model",
     "The mean (and std) of the rewards of the other samples in the same prompt's group, a Monte Carlo estimate of the prompt's value",
     "A moving average of all rewards",
     "Nothing; GRPO has no baseline"], "B",
    "Sampling G responses per prompt gives a free estimate of V(x). It costs extra sampling instead of an extra "
    "model, and sampling is cheap relative to training a value LM of the same size.")

CELLS += quiz_cells("15.2", "GRPO's KL term uses the estimator pi_ref/pi_theta - log(pi_ref/pi_theta) - 1. Why this form instead of the plain log-ratio log(pi_theta/pi_ref)?",
    ["It is cheaper to compute",
     "It is an unbiased estimator of KL that is always non-negative and has much lower variance than the log-ratio sample, whose expectation is KL only after averaging many samples of both signs",
     "It is the exact KL",
     "The plain log-ratio is biased"], "B",
    "Both are unbiased for KL(pi_theta || pi_ref) under samples from pi_theta, but the plain log-ratio k1 = log(pi/pi_ref) "
    "takes both signs and is noisy; k3 = r - log r - 1 (with r = pi_ref/pi) is >= 0 and low-variance (Schulman's 'approximating KL' note).")

CELLS += quiz_cells("15.3", "With G = 8 samples, a prompt the model solves exactly once gets advantages of roughly +2.5 (correct) and -0.35 (each wrong). What does this imply for training dynamics?",
    ["Easy prompts dominate the gradient",
     "Prompts near the edge of the model's ability get large gradient signal, while prompts solved 0/8 or 8/8 contribute nothing",
     "The KL penalty is ignored",
     "The policy becomes deterministic immediately"], "B",
    "GRPO implicitly implements a curriculum focused on 'just barely solvable' prompts. Its std normalisation amplifies "
    "this; Dr. GRPO removes the amplification, and DAPO removes the uninformative groups entirely.")

CELLS += exercise_cells("15.4",
    r"""
### Exercise 15.4 · k3 KL estimator

Implement `k3_kl(logp, ref_logp)` returning the per-token estimate $\frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$
from log-probability tensors of the same shape. It must be non-negative everywhere and zero when the two agree.
""",
    r"""
def k3_kl(logp, ref_logp):
    # YOUR CODE HERE
    raise NotImplementedError
""",
    r"""
def _tests(fn):
    lp, rp = torch.tensor([-1.0, -2.0, -0.5]), torch.tensor([-1.0, -1.0, -1.5])
    out = fn(lp, rp)
    exp = torch.exp(rp - lp) - (rp - lp) - 1
    assert out is not None and torch.allclose(out, exp, atol=1e-6), f"expected {exp.tolist()}, got {out.tolist()}"
    assert (out >= 0).all() and torch.isclose(out[0], torch.tensor(0.0)), "must be >= 0 and zero when equal"
""",
    r"""
def k3_kl(logp, ref_logp):
    log_ratio = ref_logp - logp
    return torch.exp(log_ratio) - log_ratio - 1
""", "k3_kl")
