from common import video, md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 16 · After GRPO: the 2025–2026 landscape

GRPO's success triggered a wave of papers that each fixed one weakness. Almost all of them can be
expressed as small edits to three decisions: **how advantages are computed**, **how the importance
ratio is formed and clipped**, and **how token losses are aggregated**. Here are the ones that shaped
the recipes used in current systems.

### Dr. GRPO — "GRPO done right" (Liu et al., March 2025)
Two of GRPO's normalisations are *biases*, not features:
* The $1/|y_i|$ term makes long wrong answers cheaper per token than short wrong answers, so under
  negative advantage the model learns to **ramble** (the "response length keeps growing" effect).
* Dividing by the group std up-weights very easy and very hard prompts (Part 15's plot).

Fix: drop both. Use $A_i = R_i - \text{mean}(R)$ and replace the per-response $1/|y_i|$ by a normaliser that
does not depend on the response's own length (a constant in the paper; in our implementation the batch's
total token count, i.e. DAPO's token-level mean, which differs only by a scale Adam ignores).

### DAPO — Decoupled clip and dynamic sampling (Yu et al., ByteDance Seed, March 2025)
An open-sourced large-scale recipe with four tricks:
1. **Clip-Higher**: asymmetric clipping, $\epsilon_{\text{low}} = 0.2$, $\epsilon_{\text{high}} = 0.28$. The
   upper clip is what stops low-probability tokens from growing, which drives **entropy collapse**;
   loosening it keeps exploration alive.
2. **Dynamic sampling**: throw away groups whose rewards are all equal (zero advantage) and keep
   sampling until the batch is full of informative groups.
3. **Token-level loss**: average over all tokens in the batch rather than per response, so long
   responses are not down-weighted (same fix as Dr. GRPO's first point).
4. **Overlong reward shaping**: a soft length penalty instead of a hard zero for truncated responses.
DAPO also drops the KL term entirely, arguing that reasoning training is *supposed* to drift far.

### GSPO — Group Sequence Policy Optimisation (Zheng et al., Qwen team, July 2025)
Token-level importance ratios are noisy estimates: each one is a single sample from the token's
distribution, and the noise accumulates over thousands of tokens. For mixture-of-experts models it
is worse (expert routing can flip between old and new policy). GSPO instead uses **one ratio per
sequence**, length-normalised, and clips at the sequence level:

$$s_i(\theta) = \Big(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}\Big)^{1/|y_i|}
= \exp\Big(\frac{1}{|y_i|}\sum_t \log\frac{\pi_\theta(y_{i,t}|\cdot)}{\pi_{\theta_{\text{old}}}(y_{i,t}|\cdot)}\Big).$$

Every token in $y_i$ shares the weight $s_i$, and whole sequences are either kept or clipped. This
is what trained the Qwen3 models.

### CISPO — Clipped importance-sampling policy optimisation (MiniMax-M1, June 2025)
PPO-style clipping *zeroes the gradient* of clipped tokens. Those are often exactly the rare
"reflective" tokens ("wait", "however") that matter most for reasoning. CISPO clips the
**importance weight** but keeps every token's gradient:

$$\mathcal J_{\text{CISPO}} = \mathbb E\Big[\sum_t \text{sg}\big(\text{clip}(\rho_{i,t}, 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}})\big)\, \hat A_i \log \pi_\theta(y_{i,t}|\cdot)\Big],$$

where sg is stop-gradient. It is REINFORCE with clipped, detached importance weights. Meta's
"ScaleRL" study (Khatri et al., Oct 2025) also picked CISPO (plus prompt-level averaging and
batch-level normalisation) as the most scalable loss in its comparison.

### Others worth knowing
| Method | Idea |
|---|---|
| **VAPO** (ByteDance, Apr 2025) | a value-based recipe that beats DAPO: critic pre-training, separate λ for critic and policy, length-adaptive GAE, clip-higher, token-level loss. Critics are not dead. |
| **Lite PPO** (Liu et al., Aug 2025) | after ablating every trick: group-mean advantage with batch-level std, token-level loss, and nothing else, works about as well as the full recipes. |
| **ProRL** (NVIDIA, May 2025) | prolonged RL (thousands of steps) with periodic reference-policy resets and KL to fight collapse; argues RL *does* expand reasoning beyond the base model. |
| **Magistral** (Mistral, Jun 2025) | GRPO minus KL, plus clip-higher, minibatch advantage normalisation, dynamic sampling, length penalties; an asynchronous training system. |
| **Kimi K1.5 / K2** (Moonshot, 2025) | online policy mirror descent: a squared loss $(r - \tau\log\frac{\pi}{\pi_{\text{old}}} - b)^2$ instead of a clipped ratio; also length penalties and curriculum. |
| **OPO** (May 2025) | strictly on-policy training with the *optimal* reward baseline; argues that most instability comes from being off-policy at all. |
| **TOPR** (Mar 2025) | tapered off-policy REINFORCE: clip ratios asymmetrically for positive vs negative advantages. |
| **Entropy control** (Cui et al., 2025) | Clip-Cov / KL-Cov: regulate the tokens whose probability–advantage covariance is largest, since those drive entropy collapse. |

The names differ; the dials are the same. Let's build one loss function with all of them and run the
variants side by side.
"""))
CELLS.append(video(16))

CELLS.append(code(r"""
@dataclass
class RLConfig:
    name: str = "GRPO"
    adv: str = "group_std"        # group_std (GRPO) | group_mean (Dr. GRPO, DAPO) | rloo | batch_norm (REINFORCE++)
    ratio: str = "token"          # token (PPO/GRPO) | sequence (GSPO)
    clip_low: float = 0.2
    clip_high: float = 0.2
    cispo: bool = False           # clip the weight but keep every token's gradient
    agg: str = "sequence"         # sequence (1/|y_i| then mean over group) | token (mean over all tokens)
    beta: float = 0.04            # KL coefficient in the loss (0 = no KL, as in DAPO)
    dynamic_sampling: bool = False   # drop zero-advantage groups (DAPO)
    epochs: int = 1

def compute_advantages(R, G, cfg):
    Rg = R.view(-1, G)
    if cfg.adv == "group_std":   adv = (Rg - Rg.mean(1, keepdim=True)) / (Rg.std(1, keepdim=True) + 1e-4)
    elif cfg.adv == "group_mean": adv = Rg - Rg.mean(1, keepdim=True)
    elif cfg.adv == "rloo":      adv = Rg - (Rg.sum(1, keepdim=True) - Rg) / (G - 1)
    elif cfg.adv == "batch_norm": adv = (Rg - Rg.mean(1, keepdim=True)); adv = adv / (adv.std() + 1e-4)
    return adv.reshape(-1)

def policy_loss(logp, old_logp, ref_logp, adv, mask, cfg):
    '''All the 2025 variants in one place. logp/old_logp/ref_logp/mask: (B, T); adv: (B,).'''
    A = adv[:, None]
    if cfg.ratio == "sequence":                                        # GSPO: one length-normalised ratio per sequence
        log_ratio = ((logp - old_logp) * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        ratio = torch.exp(log_ratio).expand_as(logp)
    else:                                                              # PPO/GRPO: per-token ratio
        ratio = torch.exp(logp - old_logp)
    clipped = torch.clamp(ratio, 1 - cfg.clip_low, 1 + cfg.clip_high)
    if cfg.cispo:                                                      # REINFORCE with clipped, detached weights
        surrogate = clipped.detach() * A * logp
    else:                                                              # PPO-style pessimistic min
        surrogate = torch.min(ratio * A, clipped * A)
    kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1 if cfg.beta > 0 else torch.zeros_like(logp)
    per_token = surrogate - cfg.beta * kl
    if cfg.agg == "sequence":
        return -((per_token * mask).sum(1) / mask.sum(1)).mean()
    return -(per_token * mask).sum() / mask.sum()

def make_rl_step(policy, ref, cfg, reward_fn=verifier_reward, n_samples=8, lr=3e-4, gen_fn=None):
    opt = torch.optim.Adam(policy.parameters(), lr=lr); gen_fn = gen_fn or generate
    def step(pairs):
        prompts = batch_prompts(pairs, n_samples)
        seq, mask = gen_fn(policy, prompts)
        R = batch_rewards(pairs, n_samples, seq, reward_fn)
        adv = compute_advantages(R, n_samples, cfg)
        if cfg.dynamic_sampling:                                       # keep only groups with signal
            keep = (R.view(-1, n_samples).std(1) > 0).repeat_interleave(n_samples)
            if keep.sum() == 0: return dict(reward=R.mean().item(), kl=0.0, length=mask.sum(1).mean().item(), entropy=float("nan"))
            seq, mask, adv = seq[keep], mask[keep], adv[keep]
        with torch.no_grad():
            old_logp = completion_logprobs(policy, seq); ref_logp = completion_logprobs(ref, seq)
        for _ in range(cfg.epochs):
            loss = policy_loss(completion_logprobs(policy, seq), old_logp, ref_logp, adv, mask, cfg)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        with torch.no_grad():                                          # entropy of the next-token distribution over the completion
            logits = policy(seq[:, :-1])[:, PROMPT_LEN - 1:]
            ent = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1)
        return dict(reward=R.mean().item(), kl=((old_logp - ref_logp) * mask).sum(1).mean().item(),
                    length=mask.sum(1).mean().item(), entropy=(ent * mask).sum().item() / mask.sum().item())
    return step

VARIANTS = [
    RLConfig("GRPO"),
    RLConfig("Dr. GRPO", adv="group_mean", agg="token"),
    RLConfig("DAPO-style", adv="group_mean", agg="token", clip_high=0.28, beta=0.0, dynamic_sampling=True),
    RLConfig("GSPO", ratio="sequence", clip_low=0.003, clip_high=0.004, beta=0.0),   # paper: 3e-4 / 4e-4 for long sequences
    RLConfig("CISPO", adv="group_mean", agg="token", cispo=True, clip_high=5.0, beta=0.0),
]
"""))

CELLS.append(code(r"""
variant_logs = {}
with Timer("GRPO variants"):
    for cfg in VARIANTS:
        seed_everything(0)
        pol = copy.deepcopy(sft_model)
        variant_logs[cfg.name] = train_llm_rl(make_rl_step(pol, sft_model, cfg), pol, steps=120, verbose=False)
        lg = variant_logs[cfg.name]
        print(f"  {cfg.name:11s} final accuracy {lg['eval_acc'][-1]:.3f} (hard {lg['eval_hard'][-1]:.3f}) | KL {np.mean(lg['kl'][-10:]):.2f} | entropy {np.nanmean(lg['entropy'][-10:]):.3f} | length {np.mean(lg['length'][-10:]):.2f}")

plot_llm_runs(variant_logs, "GRPO and its 2025 descendants on the toy task (same seed, same data budget)",
              keys=("eval_acc", "entropy", "kl"), titles=("accuracy on all prompts", "policy entropy over completion tokens", "KL(π‖π_ref) per sequence"))
"""))

CELLS.append(md(r"""
**What to look for.** On a three-token task every variant solves the problem, so the accuracy curves
bunch together; the *side channels* are where the design choices show. The DAPO-style run keeps a
visibly higher entropy: partly clip-higher doing its job, and partly a selection effect, because with
dynamic sampling the entropy is measured only on the informative (not-yet-solved) groups. Be suspicious of
any single-metric comparison of RL recipes; they interact with what is in the batch. Where these
differences really matter is over tens of thousands of steps of long chain-of-thought training, where
entropy collapse means the model stops exploring and plateaus, and where token-ratio noise over thousands
of tokens makes GSPO/CISPO-style weighting the difference between a stable and a divergent run.

**GSPO's clip range** looks absurd (the paper uses $\epsilon_{\text{low}} = 3\cdot10^{-4}$,
$\epsilon_{\text{high}} = 4\cdot10^{-4}$; we use ten times that for our three-token sequences) until you
remember its ratio is a *geometric mean* over tokens: a sequence-level ratio of 1.0004 means the token
ratios drifted by that factor *on average*, which over thousands of tokens is a substantial change.

### Off-policy corrections for asynchronous systems (2025–26)

A practical development behind the newest recipes: production trainers generate rollouts on a separate
inference engine (vLLM, SGLang) that runs a few steps *behind* the trainer, often with different numerics.
That makes every batch slightly off-policy even at "epoch 1", and token-level ratios blow up on long
sequences. The fixes are all importance-sampling corrections: sequence-level ratios (GSPO), truncated
importance sampling against the inference engine's log-probs (Yao et al., 2025, "Your efficient RL
framework secretly brings you off-policy RL training"), and decoupled clipping. If you read a 2026
training report and see "TIS", "MIS", "IcePop" or "sequence-level IS", this is the problem being solved.

### Test your knowledge
"""))

CELLS += quiz_cells("16.1", "Why does dividing each response's loss by its length |y_i| (as in GRPO) tend to make responses longer?",
    ["Longer responses get more tokens of positive advantage",
     "For a wrong answer (negative advantage) the per-token penalty is spread thinner over a long response, so being long is 'cheaper' when wrong; for a right answer, being short concentrates the reward",
     "Because the KL term grows with length",
     "It doesn't; length grows because of the clip"], "B",
    "Dr. GRPO and DAPO both switch to a token-level (constant-normaliser) aggregation for exactly this reason. "
    "Length growth in R1-style training is partly real reasoning and partly this artefact.")

CELLS += quiz_cells("16.2", "What is the key difference between PPO-style clipping and CISPO?",
    ["CISPO uses a smaller learning rate",
     "PPO's clip zeroes the gradient of tokens outside the trust region; CISPO clips the importance WEIGHT but detaches it, so every token still contributes a (bounded) gradient",
     "CISPO removes importance sampling entirely",
     "CISPO clips at the sequence level"], "B",
    "Rare pivotal tokens ('wait', 'however') often have large ratios after one update and would be silenced by PPO's clip. "
    "CISPO keeps their signal while bounding their weight, which MiniMax found accelerated reasoning training.")

CELLS += quiz_cells("16.3", "GSPO computes ONE importance ratio per sequence, the length-normalised product of token ratios. What problem does this address?",
    ["The KL estimator's bias",
     "Token-level ratios are single-sample estimates whose noise accumulates over long sequences (and flips with MoE expert routing), causing unstable updates; sequence-level weighting matches the sequence-level reward",
     "Slow generation",
     "The need for a critic"], "B",
    "GSPO's authors argue the unit of optimisation should match the unit of reward: the reward is per sequence, so the "
    "importance weight and the clipping decision should be per sequence too.")

CELLS += exercise_cells("16.4",
    r"""
### Exercise 16.4 · GSPO's sequence-level ratio

Implement `gspo_ratio(logp, old_logp, mask)` returning a `(B,)` tensor: the length-normalised sequence ratio
$s_i = \exp\big(\frac{1}{|y_i|}\sum_t (\log\pi_\theta - \log\pi_{\text{old}})\big)$ using only masked tokens.
""",
    r"""
def gspo_ratio(logp, old_logp, mask):
    # YOUR CODE HERE
    raise NotImplementedError
""",
    r"""
def _tests(fn):
    logp = torch.tensor([[-1.0, -1.0, -5.0], [-2.0, -1.0, -1.0]]); old = torch.tensor([[-1.5, -0.5, 0.0], [-2.0, -2.0, -1.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    out = fn(logp, old, mask)
    exp = torch.tensor([math.exp(((-1 + 1.5) + (-1 + 0.5)) / 2), math.exp((0 + 1 + 0) / 3)])
    assert out is not None and torch.allclose(out, exp, atol=1e-6), f"expected {exp.tolist()}, got {out.tolist()} (did you ignore masked tokens?)"
""",
    r"""
def gspo_ratio(logp, old_logp, mask):
    log_ratio = ((logp - old_logp) * mask).sum(1) / mask.sum(1)
    return torch.exp(log_ratio)
""", "gspo_ratio")
