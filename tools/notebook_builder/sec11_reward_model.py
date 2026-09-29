from common import video, md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 11 · Reward models and the Bradley–Terry model

For arithmetic we have a verifier. For "write a helpful, harmless answer" we don't, so RLHF learns a
**reward model** $r_\phi(x, y)$ from human comparisons. Annotators see two responses to a prompt and pick
the better one; the **Bradley–Terry** model turns the scalar reward into a preference probability:

$$P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big), \qquad
\mathcal L_{\text{RM}} = -\mathbb E\big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\big].$$

The reward model is usually the LM itself (or a copy of it) with a scalar head on the last token.
"""))
CELLS.append(video(11))

CELLS.append(md(r"""
## A synthetic annotator

We simulate human raters on our arithmetic task. The rater prefers the correct answer 85% of the time
(humans err), and, when both answers are equally right or wrong, prefers the **longer** one 85% of the
time. That second rule is the toy version of a real, well-documented bias: human raters (and LLM
judges) like longer, more elaborate answers. Watch what it does to the learned reward.
"""))

CELLS.append(code(r"""
class RewardModel(nn.Module):
    '''LM backbone (initialised from the SFT model) + scalar head read at the last generated token.'''
    def __init__(self, init_from):
        super().__init__()
        self.backbone = copy.deepcopy(init_from); self.head = nn.Linear(64, 1)
    def forward(self, seq, mask):
        h = self.backbone.hidden(seq)                                   # (B, L, d)
        last = (PROMPT_LEN + mask.sum(1).long() - 1).clamp(max=seq.shape[1] - 1)
        return self.head(h[torch.arange(len(seq)), last]).squeeze(-1)   # (B,)

def annotator_prefers(a, b, tok_i, tok_j, p_correct=0.85, p_longer=0.85):
    '''Returns True if the simulated rater prefers completion i over j.'''
    ci, cj = verify(a, b, tok_i), verify(a, b, tok_j)
    if ci != cj:
        return (ci > cj) if np.random.rand() < p_correct else (ci < cj)
    li, lj = len(parse_completion(tok_i)[0] or ""), len(parse_completion(tok_j)[0] or "")
    if li != lj:
        return (li > lj) if np.random.rand() < p_longer else (li < lj)
    return np.random.rand() < 0.5

@torch.no_grad()
def sample_completions(model, pairs, n, temperature=1.0, gen_fn=None):
    prompts = torch.tensor([encode_prompt(a, b) for a, b in pairs]).repeat_interleave(n, 0)
    return (gen_fn or generate)(model, prompts, temperature=temperature)

def build_preference_data(model, pairs, n_samples=6, pairs_per_prompt=4):
    '''Sample n completions per prompt and let the annotator label random pairs of DISTINCT completions
    (two identical strings carry no preference information).'''
    seq, mask = sample_completions(model, pairs, n_samples)
    W, Lo, Mw, Ml = [], [], [], []
    for p, (a, b) in enumerate(pairs):
        cands = seq[p * n_samples:(p + 1) * n_samples]
        distinct = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples) if not torch.equal(cands[i], cands[j])]
        for k in np.random.permutation(len(distinct))[:pairs_per_prompt]:
            i, j = distinct[k]
            si, sj = seq[p * n_samples + i], seq[p * n_samples + j]
            mi, mj = mask[p * n_samples + i], mask[p * n_samples + j]
            if annotator_prefers(a, b, si[PROMPT_LEN:].tolist(), sj[PROMPT_LEN:].tolist()):
                W.append(si); Lo.append(sj); Mw.append(mi); Ml.append(mj)
            else:
                W.append(sj); Lo.append(si); Mw.append(mj); Ml.append(mi)
    return torch.stack(W), torch.stack(Mw), torch.stack(Lo), torch.stack(Ml)

seed_everything(0)
perm = np.random.permutation(len(ALL_PAIRS)); train_pairs = [ALL_PAIRS[i] for i in perm[:320]]; heldout_pairs = [ALL_PAIRS[i] for i in perm[320:]]
pref_train = build_preference_data(sft_model, train_pairs); pref_test = build_preference_data(sft_model, heldout_pairs)
print(f"{len(pref_train[0])} training pairs, {len(pref_test[0])} held-out pairs (different prompts)")
"""))

CELLS.append(code(r"""
def train_reward_model(pref, steps=600, batch_size=64, lr=1e-3):
    rm = RewardModel(sft_model); opt = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.01)
    W, Mw, L, Ml = pref; losses = []
    for step in range(steps):
        idx = torch.randint(0, len(W), (batch_size,))
        margin = rm(W[idx], Mw[idx]) - rm(L[idx], Ml[idx])
        loss = -F.logsigmoid(margin).mean()                             # Bradley-Terry negative log-likelihood
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    return rm, losses

@torch.no_grad()
def rm_accuracy(rm, pref):
    W, Mw, L, Ml = pref
    return ((rm(W, Mw) - rm(L, Ml)) > 0).float().mean().item()
"""))

CELLS.append(code(r"""
with Timer("reward model"):
    reward_model, rm_losses = train_reward_model(pref_train)
print(f"pairwise accuracy: train {rm_accuracy(reward_model, pref_train):.3f} | held-out prompts {rm_accuracy(reward_model, pref_test):.3f} "
      f"(the annotator itself is only ~85% consistent, so ~0.85 is the ceiling)")
fig, ax = plt.subplots(figsize=(6, 2.6)); ax.plot(smooth(rm_losses, 20)); ax.axhline(math.log(2), color=MUTED, lw=1)
ax.text(len(rm_losses), math.log(2) + 0.01, "log 2 (chance)", ha="right", color=INK2, fontsize=8)
finish(ax, "Bradley-Terry loss", "step", "loss", legend=False); plt.show()
"""))

CELLS.append(md(r"""
## Optimising against a proxy: best-of-n and Goodhart's law

The simplest way to "optimise" a policy against a reward model is **best-of-n**: sample $n$ responses,
keep the one the RM likes most. It is a real deployment technique (rejection sampling) and a clean
experimental probe, because the KL divergence between best-of-$n$ selection and the base policy has a
closed form, $\mathrm{KL} \approx \log n - \frac{n-1}{n}$, independent of the task.

We measure two things as $n$ grows: the **proxy** reward (what the RM says) and the **true** reward
(what the verifier says). Gao et al. (2022) showed that in real RLHF the true reward eventually peaks and
declines while the proxy keeps rising: the policy finds responses the RM over-values. This is
**Goodhart's law**: "when a measure becomes a target, it ceases to be a good measure."
"""))

CELLS.append(code(r"""
seed_everything(0)
N_BON = 64
seq_bon, mask_bon = sample_completions(sft_model, ALL_PAIRS, N_BON)
with torch.no_grad():
    proxy = reward_model(seq_bon, mask_bon).numpy().reshape(len(ALL_PAIRS), N_BON)
true = np.array([verify(a, b, seq_bon[p * N_BON + j, PROMPT_LEN:].tolist()) for p, (a, b) in enumerate(ALL_PAIRS) for j in range(N_BON)]).reshape(len(ALL_PAIRS), N_BON)
lengths = np.array([len(parse_completion(seq_bon[i, PROMPT_LEN:].tolist())[0] or "") for i in range(len(seq_bon))]).reshape(len(ALL_PAIRS), N_BON)

ns = [1, 2, 4, 8, 16, 32, 64]; rows = []
for n in ns:
    best = proxy[:, :n].argmax(1)                                        # index of the RM-preferred sample among the first n
    rows.append(dict(n=n, kl=np.log(n) - (n - 1) / n, proxy=proxy[np.arange(len(ALL_PAIRS)), best].mean(),
                     true=true[np.arange(len(ALL_PAIRS)), best].mean(), length=lengths[np.arange(len(ALL_PAIRS)), best].mean()))
fig, axes = plt.subplots(1, 3, figsize=(14, 3.3))
kls = [r["kl"] for r in rows]
axes[0].plot(kls, [r["proxy"] for r in rows], marker="o"); finish(axes[0], "proxy: RM score of the selected sample", "KL(best-of-n ‖ π_ref)", "RM score", legend=False)
axes[1].plot(kls, [r["true"] for r in rows], marker="o", color=PALETTE[1]); finish(axes[1], "truth: verifier accuracy of the selected sample", "KL(best-of-n ‖ π_ref)", "accuracy", legend=False)
for ax in axes[:2]:
    for r in rows: ax.annotate(f"n={r['n']}", (r["kl"], ax.lines[0].get_ydata()[rows.index(r)]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color=INK2)
bins = np.linspace(proxy.min(), proxy.max(), 40)
axes[2].hist(proxy[true == 1], bins=bins, alpha=0.6, label="verifier-correct samples", edgecolor="none")
axes[2].hist(proxy[true == 0], bins=bins, alpha=0.6, label="incorrect samples", edgecolor="none")
finish(axes[2], "RM scores of all 25,600 samples", "RM score", "count")
plt.tight_layout(); plt.show()
print("best-of-n table:"); [print(f"  n={r['n']:3d}  KL={r['kl']:.2f}  proxy={r['proxy']:+.2f}  true={r['true']:.3f}") for r in rows]
print("what best-of-64 picks for a few prompts:")
for p in [0, 137, 261, 399]:
    a, b = ALL_PAIRS[p]; j = int(proxy[p].argmax())
    print(f"  {decode(encode_prompt(a, b))} (true {a + b:2d}) -> {decode(seq_bon[p * N_BON + j, PROMPT_LEN:]):8s} RM score {proxy[p, j]:+.2f}  verifier {true[p, j]:.0f}")
"""))

CELLS.append(md(r"""
The proxy rises monotonically, as it must (we are selecting on it). The true reward rises for small $n$,
peaks, and then **falls**: the textbook Goodhart curve. The right panel shows why. The reward model
separates correct from incorrect samples only partially (it was trained on a few hundred noisy pairs), and
the two score distributions overlap in the upper tail. Mild selection (n = 2–4) mostly picks correct samples;
heavy selection (n = 64) reaches into the tail where the highest-scoring sample for many prompts is a wrong
answer the RM happens to over-value. Real RLHF applies vastly more optimisation pressure than
best-of-64, which is why practitioners (a) keep the KL small, (b) retrain the RM on fresh samples from the
current policy in several rounds, (c) ensemble RMs, and (d) use verifiable rewards whenever they can.

### Test your knowledge
"""))

CELLS += quiz_cells("11.1", "The Bradley-Terry loss only depends on the DIFFERENCE r(y_w) - r(y_l). What consequence does that have?",
    ["The reward model cannot be trained with gradient descent",
     "The absolute scale and offset of the reward are unidentified: adding a per-prompt constant changes nothing, which is fine because the policy gradient only uses differences too",
     "It requires exactly two responses per prompt",
     "It makes the reward model linear"], "B",
    "Preferences only reveal relative quality. This is also why DPO can drop the partition function Z(x) (Part 13) and why "
    "GRPO can subtract the group mean (Part 15): only differences within a prompt carry information.")

CELLS += quiz_cells("11.2", "In the best-of-n experiment, why does the proxy reward keep rising even where the true reward does not?",
    ["The reward model is trained on the wrong prompts",
     "Selecting the sample with the highest RM score guarantees the score increases with n; if the RM over-values some feature (length), selection finds samples that have that feature whether or not they are correct",
     "Because the verifier is stochastic",
     "Because KL grows with n"], "B",
    "Optimisation exploits the errors of the proxy in proportion to how hard it optimises. Goodhart's law is a statement "
    "about the tail of a distribution, which selection and RL both explore aggressively.")

CELLS += exercise_cells("11.3",
    r"""
### Exercise 11.3 · Bradley–Terry loss with label noise

Implement `bt_loss(r_w, r_l, label_smoothing=0.0)` returning the mean loss over a batch. With label
smoothing $\epsilon$ the target probability that $y_w$ is better becomes $1-\epsilon$, so the loss is
$-(1-\epsilon)\log\sigma(r_w - r_l) - \epsilon \log\sigma(r_l - r_w)$. This is what many production RM
trainers use to cope with annotator noise.
""",
    r"""
def bt_loss(r_w, r_l, label_smoothing=0.0):
    # YOUR CODE HERE
    raise NotImplementedError
""",
    r"""
def _tests(fn):
    r_w, r_l = torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0])
    out = fn(r_w, r_l)
    assert out is not None and out.dim() == 0, "return a scalar"
    exp = -(F.logsigmoid(torch.tensor(2.0)) + F.logsigmoid(torch.tensor(-1.0))) / 2
    assert torch.isclose(out, exp, atol=1e-6), f"expected {exp.item():.4f}, got {out.item():.4f}"
    out2 = fn(r_w, r_l, label_smoothing=0.1)
    exp2 = (-(0.9 * F.logsigmoid(r_w - r_l) + 0.1 * F.logsigmoid(r_l - r_w))).mean()
    assert torch.isclose(out2, exp2, atol=1e-6), "label smoothing is wrong"
""",
    r"""
def bt_loss(r_w, r_l, label_smoothing=0.0):
    margin = r_w - r_l
    return (-(1 - label_smoothing) * F.logsigmoid(margin) - label_smoothing * F.logsigmoid(-margin)).mean()
""", "bt_loss")
