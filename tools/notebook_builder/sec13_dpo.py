from common import video, md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 13 · DPO: direct preference optimisation

PPO-RLHF is a lot of machinery just to nudge a model toward what humans prefer. **DPO** (Rafailov et al.,
2023) noticed that the RLHF objective has a *closed-form* solution, and that plugging it back into the
Bradley–Terry model gives a loss you can minimise on preference pairs directly. No reward model, no
sampling, no critic.
"""))
CELLS.append(video(13))

CELLS.append(md(r"""
## The derivation in four lines

1. The KL-regularised objective $\max_\pi \mathbb E_{y\sim\pi}[r(x,y)] - \beta\,\mathrm{KL}(\pi\|\pi_{\text{ref}})$
   is maximised by the Gibbs distribution
   $$\pi^*(y|x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y|x)\, \exp\!\big(r(x,y)/\beta\big).$$
2. Solve for the reward: $r(x,y) = \beta \log \dfrac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$.
3. Plug into Bradley–Terry. The intractable $Z(x)$ **cancels** because both responses share the prompt:
   $$P(y_w \succ y_l) = \sigma\!\Big(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big).$$
4. Replace $\pi^*$ by $\pi_\theta$ and maximise the likelihood of the observed preferences:
   $$\mathcal L_{\text{DPO}}(\theta) = -\mathbb E\Big[\log \sigma\Big(\beta \big[\underbrace{\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}}_{\text{implicit reward of } y_w} - \log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\big]\Big)\Big].$$

The gradient weights each pair by $\sigma(\hat r_l - \hat r_w)$: pairs the implicit reward model gets
*wrong* get the largest updates. $\beta$ plays the same role as in RLHF (how far from the reference we are
allowed to go).

## DPO on our task

We build preference pairs the cheap way: sample completions from the SFT model and let the *verifier*
label a correct one as chosen and an incorrect one as rejected. That is the "offline RLVR" setting
several 2025 papers explored; with human labels the code is identical.
"""))

CELLS.append(code(r"""
def sequence_logprob(model, seq, mask):
    return (completion_logprobs(model, seq) * mask).sum(1)

def dpo_loss(policy, ref, W, Mw, L, Ml, beta=0.1):
    pi_w, pi_l = sequence_logprob(policy, W, Mw), sequence_logprob(policy, L, Ml)
    with torch.no_grad():
        ref_w, ref_l = sequence_logprob(ref, W, Mw), sequence_logprob(ref, L, Ml)
    margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))          # difference of implicit rewards
    loss = -F.logsigmoid(margin).mean()
    return loss, dict(margin=margin.mean().item(), pref_acc=(margin > 0).float().mean().item(),
                      logp_chosen=pi_w.mean().item(), logp_rejected=pi_l.mean().item())

def build_verifier_pairs(model, pairs, n_samples=8, max_pairs=4):
    '''chosen = a correct sample, rejected = an incorrect sample of the same prompt.'''
    seq, mask = sample_completions(model, pairs, n_samples)
    W, Mw, L, Ml = [], [], [], []
    for p, (a, b) in enumerate(pairs):
        idx = range(p * n_samples, (p + 1) * n_samples)
        good = [i for i in idx if verify(a, b, seq[i, PROMPT_LEN:].tolist())]; bad = [i for i in idx if i not in good]
        for _ in range(min(max_pairs, len(good), len(bad))):
            i, j = np.random.choice(good), np.random.choice(bad)
            W.append(seq[i]); Mw.append(mask[i]); L.append(seq[j]); Ml.append(mask[j])
    return torch.stack(W), torch.stack(Mw), torch.stack(L), torch.stack(Ml)

seed_everything(0)
dpo_pairs = build_verifier_pairs(sft_model, ALL_PAIRS)
print(f"{len(dpo_pairs[0])} preference pairs; example chosen/rejected: {decode(dpo_pairs[0][0])} / {decode(dpo_pairs[2][0])}")
"""))

CELLS.append(code(r"""
def train_dpo(pairs_data, steps=300, batch_size=64, beta=0.1, lr=2e-4, eval_every=20):
    policy = copy.deepcopy(sft_model); opt = torch.optim.Adam(policy.parameters(), lr=lr)
    W, Mw, L, Ml = pairs_data; log = defaultdict(list)
    for step in range(steps):
        idx = torch.randint(0, len(W), (batch_size,))
        loss, info = dpo_loss(policy, sft_model, W[idx], Mw[idx], L[idx], Ml[idx], beta)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        for k, v in info.items(): log[k].append(v)
        if step % eval_every == 0 or step == steps - 1:
            _, ev = evaluate_lm(policy, n_samples=4)
            log["eval_step"].append(step); log["eval_acc"].append(ev["acc"]); log["eval_easy"].append(ev["easy"]); log["eval_hard"].append(ev["hard"])
    return policy, log
"""))

CELLS.append(code(r"""
dpo_runs = {}
with Timer("DPO (three values of β)"):
    for beta in [0.1, 0.3, 1.0]:
        seed_everything(0)
        dpo_runs[beta] = train_dpo(dpo_pairs, beta=beta)
        pol, lg = dpo_runs[beta]
        print(f"  β = {beta}: accuracy {lg['eval_acc'][0]:.3f} -> {lg['eval_acc'][-1]:.3f}  (easy {lg['eval_easy'][-1]:.2f}, hard {lg['eval_hard'][-1]:.2f}) "
              f"| final log π: chosen {lg['logp_chosen'][-1]:.2f}, rejected {lg['logp_rejected'][-1]:.2f}")
policy_dpo, log_dpo = dpo_runs[1.0]

fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.3))
for beta, (pol, lg) in dpo_runs.items():
    axes[0].plot(lg["eval_step"], lg["eval_acc"], marker="o", ms=3, label=f"β = {beta}")
    axes[1].plot(lg["eval_step"], lg["eval_easy"], marker="o", ms=3, label=f"β = {beta}")
finish(axes[0], "verifier accuracy, all prompts", "DPO step", "accuracy")
finish(axes[1], "verifier accuracy, EASY prompts", "DPO step", "accuracy", legend=False)
lg = dpo_runs[0.1][1]
axes[2].plot(smooth(lg["logp_chosen"], 10), label="chosen"); axes[2].plot(smooth(lg["logp_rejected"], 10), label="rejected")
finish(axes[2], "log π(y | x) under DPO with β = 0.1", "DPO step", "sequence log-prob")
plt.tight_layout(); plt.show()
seed_everything(1); print("samples from the β = 0.1 policy on easy prompts:"); show_samples(dpo_runs[0.1][0], [(2, 5), (9, 9), (3, 4)])
"""))

CELLS.append(md(r"""
**DPO can over-optimise, and β is the leash.** With the textbook β = 0.1 the hard prompts improve but the
*easy* ones collapse: look at the samples, the model starts emitting empty or truncated answers. The right
panel explains why. DPO only cares about the *gap* between chosen and rejected, and the cheapest way to
widen it is to drive the rejected sequences to log-probabilities of −20 and below. With only a few
hundred easy-prompt pairs to anchor it, the probability mass that leaves the rejected answers does not go
to the chosen ones (whose log-prob is already ≈ 0) but to *unseen* strings. This is **likelihood
displacement** (Razin et al., 2024), a well-documented DPO failure mode, and a large β (a tighter KL leash)
or early stopping prevents it. Production runs use β ≈ 0.1 with learning rates a thousand times smaller
than ours, which amounts to the same thing: a small total policy change.

## The DPO family

| Method | Loss idea | Needs $\pi_{\text{ref}}$? | Notes |
|---|---|---|---|
| **DPO** (2023) | $-\log\sigma(\beta\,\Delta)$, $\Delta$ = chosen-minus-rejected difference of $\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ | yes | the baseline; sensitive to over-fitting on small data |
| **IPO** (Azar et al., 2023) | $(\Delta - \tfrac{1}{2\beta})^2$ | yes | bounded target instead of pushing the margin to ∞; more robust to deterministic preferences |
| **KTO** (Ethayarajh et al., 2024) | prospect-theory utility on *unpaired* good/bad examples | yes | uses thumbs-up/down data, no pairs required |
| **SimPO** (Meng et al., 2024) | length-normalised log-prob as the reward, plus a target margin $\gamma$ | **no** | cheaper, fights length bias, popular for open models |
| **ORPO** (Hong et al., 2024) | odds-ratio penalty added to the SFT loss | no | single-stage SFT + alignment |
| **Online / iterative DPO** | sample from the current policy, label with an RM or verifier, DPO step, repeat | yes | recovers most of the benefit of on-policy RL |

The offline methods share one limitation: they can only reinforce responses that already exist in the
dataset. On-policy RL (Parts 12, 14–17) generates its own data, explores, and finds behaviours no
annotator wrote down. Empirically, for reasoning tasks on-policy RL wins; for cheap alignment of
chat models, DPO-style methods are often good enough and far simpler to run.

### Test your knowledge
"""))

CELLS += quiz_cells("13.1", "Why can DPO drop the partition function Z(x) that appears in the optimal policy?",
    ["Z(x) is always 1",
     "Both responses in a pair share the same prompt, so log Z(x) appears in both implicit rewards and cancels in their difference",
     "It is estimated with a critic",
     "Because beta is small"], "B",
    "This cancellation is the whole trick. It is also why DPO needs PAIRS (or a reference point per prompt): "
    "an absolute reward for a single response would still contain the intractable Z(x).")

CELLS += quiz_cells("13.2", "What is the 'implicit reward' in DPO?",
    ["The reward model's output", "beta * log( pi_theta(y|x) / pi_ref(y|x) )", "The verifier's score", "log pi_theta(y|x)"], "B",
    "DPO trains the policy so that this quantity behaves like a Bradley-Terry reward on the preference data. "
    "You can even use a DPO-trained model as a reward model by reading off this ratio.")

CELLS += quiz_cells("13.3", "Compared with on-policy RL (PPO/GRPO), the main limitation of offline DPO is:",
    ["It needs a critic",
     "It cannot use a reference model",
     "It only ever sees the fixed dataset's responses, so it cannot discover and reinforce new behaviours that the current policy would generate",
     "It requires verifiable rewards"], "C",
    "Distribution shift: as the policy changes, the dataset's responses become stale. Iterative/online DPO addresses "
    "this by regenerating data, at which point you are most of the way back to RL.")

CELLS += exercise_cells("13.4",
    r"""
### Exercise 13.4 · DPO loss from log-probabilities

Implement `my_dpo_loss(pi_w, pi_l, ref_w, ref_l, beta)` given four `(B,)` tensors of **sequence**
log-probabilities (policy and reference, chosen and rejected). Return the mean loss.
""",
    r"""
def my_dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    # YOUR CODE HERE
    raise NotImplementedError
""",
    r"""
def _tests(fn):
    pi_w, pi_l = torch.tensor([-1.0, -2.0]), torch.tensor([-3.0, -1.0])
    ref_w, ref_l = torch.tensor([-1.5, -2.0]), torch.tensor([-2.5, -2.0])
    out = fn(pi_w, pi_l, ref_w, ref_l, 0.5)
    margin = 0.5 * ((pi_w - ref_w) - (pi_l - ref_l))
    exp = -F.logsigmoid(margin).mean()
    assert out is not None and torch.isclose(out, exp, atol=1e-6), f"expected {exp.item():.4f}, got {out.item():.4f}"
""",
    r"""
def my_dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(margin).mean()
""", "my_dpo_loss")
