from common import video, md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 14 · Critic-free policy gradients: RLOO and REINFORCE++

Look again at what PPO-RLHF does with its critic. The "environment" is deterministic, the reward is
one number at the end, and we can sample as many responses per prompt as we like. In that setting the
Monte Carlo return is *exact* (it's just $R$), so the critic exists only to provide a **baseline**.
Ahmadian et al. (2024, "Back to Basics") argued that a well-chosen baseline plus plain REINFORCE is as
good as PPO for LLMs and far simpler. The sequence-level policy gradient is

$$\nabla_\theta J = \mathbb E\Big[(R(y) - b)\, \nabla_\theta \log \pi_\theta(y|x)\Big], \qquad \log\pi_\theta(y|x) = \sum_t \log \pi_\theta(y_t \mid x, y_{<t}).$$
"""))
CELLS.append(video(14))

CELLS.append(md(r"""
## RLOO: REINFORCE leave-one-out

Sample $K$ responses $y_1..y_K$ for the same prompt and use the **mean of the other $K-1$ rewards** as the
baseline for each one:

$$A_i = R_i - \frac{1}{K-1}\sum_{j \ne i} R_j .$$

Because $b_i$ does not depend on $y_i$, the estimator stays unbiased (Part 7), and it adapts to the
prompt's difficulty automatically. The KL penalty is folded into the sequence reward:
$R_i \leftarrow R_i - \beta\, \mathrm{KL}_i$.

## REINFORCE++

Hu (2025) kept PPO's *token-level* clipped objective but removed the critic: per-token rewards with the
KL shaping of Part 12, reward-to-go advantages (with $\gamma = 1$ the advantage of every token is just
the sequence's total shaped reward from that token on), and **normalisation over the whole batch** rather
than per prompt. Its "-baseline" variant additionally subtracts each prompt's mean reward first, like RLOO.
The clip makes it safe to take a couple of epochs per batch.
"""))

CELLS.append(code(r"""
def make_rloo_step(policy, ref, reward_fn=verifier_reward, n_samples=8, beta=0.02, lr=3e-4, gen_fn=None):
    opt = torch.optim.Adam(policy.parameters(), lr=lr); gen_fn = gen_fn or generate
    def step(pairs):
        prompts = batch_prompts(pairs, n_samples)
        seq, mask = gen_fn(policy, prompts)
        R = batch_rewards(pairs, n_samples, seq, reward_fn)
        with torch.no_grad():
            old_logp = completion_logprobs(policy, seq); ref_logp = completion_logprobs(ref, seq)
            kl_seq = ((old_logp - ref_logp) * mask).sum(1)
            R_shaped = (R - beta * kl_seq).view(-1, n_samples)                       # KL penalty inside the reward
            baseline = (R_shaped.sum(1, keepdim=True) - R_shaped) / (n_samples - 1)  # leave-one-out mean
            adv = (R_shaped - baseline).reshape(-1)
        logp_seq = (completion_logprobs(policy, seq) * mask).sum(1)                   # log pi(y|x)
        loss = -(adv * logp_seq).mean()                                               # one REINFORCE step
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        return dict(reward=R.mean().item(), kl=kl_seq.mean().item(), length=mask.sum(1).mean().item())
    return step

def make_reinforce_pp_step(policy, ref, reward_fn=verifier_reward, n_samples=8, beta=0.02, clip=0.2, lr=3e-4,
                           epochs=2, minibatch=128, prompt_baseline=True, gen_fn=None):
    opt = torch.optim.Adam(policy.parameters(), lr=lr); gen_fn = gen_fn or generate
    def step(pairs):
        prompts = batch_prompts(pairs, n_samples)
        seq, mask = gen_fn(policy, prompts)
        R = batch_rewards(pairs, n_samples, seq, reward_fn)
        with torch.no_grad():
            old_logp = completion_logprobs(policy, seq); ref_logp = completion_logprobs(ref, seq)
            if prompt_baseline:                                                       # REINFORCE++-baseline
                R = (R.view(-1, n_samples) - R.view(-1, n_samples).mean(1, keepdim=True)).reshape(-1)
            rewards = shape_rewards(R, old_logp, ref_logp, mask, beta)
            adv = torch.flip(torch.cumsum(torch.flip(rewards, [1]), 1), [1]) * mask      # reward-to-go, gamma = 1
            valid = mask > 0
            adv = (adv - adv[valid].mean()) / (adv[valid].std() + 1e-8) * mask           # global batch normalisation
        N = len(seq)
        for _ in range(epochs):
            perm = torch.randperm(N)
            for s in range(0, N, minibatch):
                idx = perm[s:s + minibatch]; m = mask[idx]
                ratio = torch.exp(completion_logprobs(policy, seq[idx]) - old_logp[idx])
                pg = -torch.min(ratio * adv[idx], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[idx])
                loss = (pg * m).sum() / m.sum()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        return dict(reward=batch_rewards(pairs, n_samples, seq, reward_fn).mean().item(),
                    kl=((old_logp - ref_logp) * mask).sum(1).mean().item(), length=mask.sum(1).mean().item())
    return step
"""))

CELLS.append(code(r"""
seed_everything(0)
policy_rloo = copy.deepcopy(sft_model)
with Timer("RLOO"):
    log_rloo = train_llm_rl(make_rloo_step(policy_rloo, sft_model), policy_rloo)
seed_everything(0)
policy_rpp = copy.deepcopy(sft_model)
with Timer("REINFORCE++"):
    log_rpp = train_llm_rl(make_reinforce_pp_step(policy_rpp, sft_model), policy_rpp)
llm_results["RLOO"] = log_rloo; llm_results["REINFORCE++ (baseline)"] = log_rpp
plot_llm_runs(llm_results, "Critic-free methods vs PPO on the toy LM (same 256 samples per step)")
"""))

CELLS.append(md(r"""
On this task the critic-free methods match or beat PPO, at a fraction of the cost: no value model, no
GAE, and one forward pass fewer per step. This is the empirical finding that drove the field toward
GRPO-style algorithms in 2024–25.

### Test your knowledge
"""))

CELLS += quiz_cells("14.1", "Why is the leave-one-out baseline unbiased while using the mean of ALL K rewards (including R_i itself) is not?",
    ["The full mean has higher variance",
     "A valid baseline must be independent of the action being scored; R_i depends on y_i, so including it correlates the baseline with the sample and biases the gradient (by a factor (K-1)/K, as it turns out)",
     "The leave-one-out mean is always larger",
     "Both are biased"], "B",
    "E[grad log pi(y_i) * f(y_i)] is not zero for a function of y_i. With the full mean the bias is a constant "
    "rescaling of the gradient, which is why GRPO (which uses the full-group mean) still works in practice.")

CELLS += quiz_cells("14.2", "REINFORCE++ normalises advantages over the entire batch, RLOO/GRPO per prompt. What is a consequence of per-prompt normalisation?",
    ["Prompts where all samples get the same reward contribute no gradient at all",
     "The batch must contain a single prompt",
     "Per-prompt normalisation is biased",
     "It requires a critic"], "A",
    "If every sample of a prompt scores 0 (too hard) or 1 (too easy), the per-prompt advantages are all zero. "
    "That wastes compute and is the motivation for DAPO's dynamic sampling (Part 16).")

CELLS += exercise_cells("14.3",
    r"""
### Exercise 14.3 · RLOO advantages

Implement `rloo_advantages(R, K)` where `R` is a `(B,)` tensor of rewards ordered so that consecutive
blocks of `K` entries belong to the same prompt. Return the `(B,)` tensor of leave-one-out advantages.
""",
    r"""
def rloo_advantages(R, K):
    # YOUR CODE HERE
    raise NotImplementedError
""",
    r"""
def _tests(fn):
    R = torch.tensor([1.0, 0.0, 0.0, 1.0,   1.0, 1.0, 1.0, 0.0])
    out = fn(R, 4)
    exp = torch.tensor([1 - 1/3, 0 - 2/3, 0 - 2/3, 1 - 1/3,   1 - 2/3, 1 - 2/3, 1 - 2/3, 0 - 1.0])
    assert out is not None and torch.allclose(out, exp, atol=1e-6), f"expected {exp.tolist()}, got {out.tolist()}"
""",
    r"""
def rloo_advantages(R, K):
    Rg = R.view(-1, K)
    baseline = (Rg.sum(1, keepdim=True) - Rg) / (K - 1)
    return (Rg - baseline).reshape(-1)
""", "rloo_advantages")
