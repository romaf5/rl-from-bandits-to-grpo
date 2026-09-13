from common import md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 12 · RLHF with PPO

The classic RLHF objective (Ziegler et al., 2019; Ouyang et al., 2022) is

$$\max_\theta\; \mathbb E_{x \sim \mathcal D,\, y \sim \pi_\theta(\cdot|x)}\big[\, r(x, y)\,\big] \;-\; \beta\, \mathbb E_x\big[\mathrm{KL}\big(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\big)\big].$$

Implemented with PPO over tokens:

1. Sample a response per prompt from the current policy.
2. Score it with the reward function (RM or verifier) to get one scalar $R$.
3. Turn it into **per-token rewards**: $r_t = -\beta\big(\log\pi_\theta(a_t|s_t) - \log\pi_{\text{ref}}(a_t|s_t)\big)$ for
   every token, plus $R$ on the final token. The KL penalty becomes part of the reward the critic must predict.
4. Compute GAE advantages over tokens with a **value model** (a second LM with a scalar head) and $\gamma = 1$.
5. Run PPO's clipped update over tokens for a couple of epochs.

That is four large models in memory at once (policy, reference, critic, reward model), which is the main
reason critic-free methods (Parts 14–16) took over for reasoning models.

## Building it
"""))

CELLS.append(code(r"""
class Critic(nn.Module):
    '''Value model: LM backbone + scalar head, one value per prefix position.'''
    def __init__(self, init_from):
        super().__init__()
        self.backbone = copy.deepcopy(init_from); self.head = nn.Linear(64, 1)
    def forward(self, seq):                        # (B, L) -> V for the prefix ending at each position: (B, L)
        return self.head(self.backbone.hidden(seq)).squeeze(-1)

def shape_rewards(final_reward, logp, ref_logp, mask, beta):
    '''Per-token rewards: -beta * (log pi - log pi_ref) on every real token, plus the final reward on the last real token.'''
    rewards = -beta * (logp - ref_logp) * mask
    last = (mask.sum(1).long() - 1).clamp(min=0)
    rewards[torch.arange(len(mask)), last] += final_reward
    return rewards

def gae_tokens(rewards, values, mask, gamma=1.0, lam=0.95):
    '''GAE over completion tokens. values[:, t] = V(prefix before token t). Terminal after the last real token.'''
    B, T = rewards.shape; adv = torch.zeros(B, T); last = torch.zeros(B)
    for t in reversed(range(T)):
        next_v = values[:, t + 1] * mask[:, t + 1] if t + 1 < T else torch.zeros(B)   # V = 0 after the last real token
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last = (delta + gamma * lam * last) * mask[:, t]        # the mask is a prefix, so this resets the recursion at padding
        adv[:, t] = last
    return adv, (adv + values) * mask

def batch_prompts(pairs, n_samples):
    return torch.tensor([encode_prompt(a, b) for a, b in pairs]).repeat_interleave(n_samples, 0)

def batch_rewards(pairs, n_samples, seq, reward_fn):
    return torch.tensor([reward_fn(a, b, seq[i * n_samples + j]) for i, (a, b) in enumerate(pairs) for j in range(n_samples)], dtype=torch.float32)

verifier_reward = lambda a, b, seq: verify(a, b, seq[PROMPT_LEN:].tolist())
"""))

CELLS.append(code(r"""
def make_ppo_llm_step(policy, ref, critic, reward_fn=verifier_reward, n_samples=8, beta=0.02, clip=0.2,
                      epochs=2, minibatch=128, lr=3e-4, lr_v=1e-3, vf_coef=0.5, lam=0.95, gen_fn=None):
    opt = torch.optim.Adam(policy.parameters(), lr=lr); opt_v = torch.optim.Adam(critic.parameters(), lr=lr_v)
    gen_fn = gen_fn or generate
    def step(pairs):
        prompts = batch_prompts(pairs, n_samples)
        seq, mask = gen_fn(policy, prompts)                                   # 1. sample
        R = batch_rewards(pairs, n_samples, seq, reward_fn)                   # 2. score
        with torch.no_grad():
            old_logp = completion_logprobs(policy, seq); ref_logp = completion_logprobs(ref, seq)
            values = critic(seq[:, :-1])[:, PROMPT_LEN - 1:]                  # V(prefix) aligned with each completion token
            rewards = shape_rewards(R, old_logp, ref_logp, mask, beta)        # 3. per-token rewards with KL penalty
            adv, ret = gae_tokens(rewards, values, mask, lam=lam)             # 4. GAE over tokens
        N = len(seq); stats = defaultdict(list)
        for _ in range(epochs):                                               # 5. PPO epochs
            perm = torch.randperm(N)
            for s in range(0, N, minibatch):
                idx = perm[s:s + minibatch]; m = mask[idx]
                mb_adv = adv[idx]; mb_adv = (mb_adv - mb_adv[m > 0].mean()) / (mb_adv[m > 0].std() + 1e-8)
                logp = completion_logprobs(policy, seq[idx]); ratio = torch.exp(logp - old_logp[idx])
                pg = -torch.min(ratio * mb_adv, torch.clamp(ratio, 1 - clip, 1 + clip) * mb_adv)
                pg_loss = (pg * m).sum() / m.sum()
                v_pred = critic(seq[idx][:, :-1])[:, PROMPT_LEN - 1:]
                v_loss = (((v_pred - ret[idx]) ** 2) * m).sum() / m.sum()
                opt.zero_grad(); pg_loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
                opt_v.zero_grad(); (vf_coef * v_loss).backward(); opt_v.step()
                stats["clipfrac"].append((((ratio - 1).abs() > clip).float() * m).sum().item() / m.sum().item())
        with torch.no_grad():
            ent = -(completion_logprobs(policy, seq) * mask).sum() / mask.sum()   # mean negative log-prob of sampled tokens (an entropy proxy)
        return dict(reward=R.mean().item(), kl=((old_logp - ref_logp) * mask).sum(1).mean().item(),
                    length=mask.sum(1).mean().item(), nll=ent.item(), clipfrac=float(np.mean(stats["clipfrac"])))
    return step

def train_llm_rl(step_fn, policy, steps=120, batch_size=32, eval_every=10, gen_fn=None, verify_fn=None, extra_eval=None, verbose=True):
    '''Generic driver: sample a batch of prompts, call step_fn, log, periodically evaluate on all prompts.'''
    log = defaultdict(list)
    for step in range(steps):
        pairs = [ALL_PAIRS[i] for i in np.random.choice(len(ALL_PAIRS), batch_size, replace=False)]
        for k, v in step_fn(pairs).items(): log[k].append(v)
        if step % eval_every == 0 or step == steps - 1:
            _, ev = evaluate_lm(policy, n_samples=4, gen_fn=gen_fn, verify_fn=verify_fn)
            log["eval_step"].append(step); log["eval_acc"].append(ev["acc"]); log["eval_easy"].append(ev["easy"]); log["eval_hard"].append(ev["hard"])
            if extra_eval is not None:
                for k, v in extra_eval(policy).items(): log[k].append(v)
            if verbose and (step % (eval_every * 4) == 0 or step == steps - 1):
                print(f"  step {step:4d} | batch reward {np.mean(log['reward'][-eval_every:]):.3f} | eval acc {ev['acc']:.3f} "
                      f"(easy {ev['easy']:.2f}, hard {ev['hard']:.2f}) | KL {np.mean(log['kl'][-eval_every:]):.3f} | len {np.mean(log['length'][-eval_every:]):.2f}")
    return log
"""))

CELLS.append(code(r"""
seed_everything(0)
policy_ppo = copy.deepcopy(sft_model); critic_ppo = Critic(sft_model)
ppo_llm_step = make_ppo_llm_step(policy_ppo, sft_model, critic_ppo)
with Timer("PPO-RLHF (verifier reward)"):
    log_ppo_llm = train_llm_rl(ppo_llm_step, policy_ppo)
llm_results = {"PPO (critic + KL reward)": log_ppo_llm}
"""))

CELLS.append(code(r"""
def plot_llm_runs(results, title="", keys=("eval_acc", "kl", "length"), titles=("accuracy on all prompts (4 samples each)", "KL(π‖π_ref) per sequence", "response length (tokens)")):
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.3))
    for name, log in results.items():
        for ax, key in zip(axes, keys):
            if key.startswith("eval"): ax.plot(log["eval_step"], log[key], marker="o", ms=3, label=name)
            else: ax.plot(smooth(log[key], 5), label=name)
    for ax, t in zip(axes, titles): finish(ax, t, "RL step", None, legend=(ax is axes[0]))
    if title: fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()

plot_llm_runs(llm_results, "PPO on the toy LM: sharpening the SFT policy with a verifier")
"""))

CELLS.append(md(r"""
Accuracy climbs from the SFT level toward 100% while the KL to the reference grows slowly: the policy
concentrates probability on answers it could already produce. The response length stays put because the
verifier does not reward verbosity. Now let's optimise against the **reward model** from Part 11 instead
and watch the same run through two lenses: the proxy it optimises and the truth.
"""))

CELLS.append(code(r"""
@torch.no_grad()
def rm_reward_factory(rm):
    def rm_reward(a, b, seq):
        m = torch.zeros(1, MAX_NEW)
        toks = seq[PROMPT_LEN:].tolist(); n = next((i + 1 for i, t in enumerate(toks) if t == EOS), MAX_NEW); m[0, :n] = 1
        return rm(seq[None], m).item()
    return rm_reward

seed_everything(0)
policy_rm = copy.deepcopy(sft_model); critic_rm = Critic(sft_model)
rm_reward = rm_reward_factory(reward_model)
rm_step = make_ppo_llm_step(policy_rm, sft_model, critic_rm, reward_fn=rm_reward, beta=0.01)
with Timer("PPO-RLHF (reward-model reward)"):
    log_rm = train_llm_rl(rm_step, policy_rm, steps=150, verbose=False)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
axes[0].plot(smooth(log_rm["reward"], 5), color=PALETTE[0]); finish(axes[0], "proxy: reward-model score of samples", "RL step", "RM score", legend=False)
axes[1].plot(log_rm["eval_step"], log_rm["eval_acc"], marker="o", ms=3, color=PALETTE[1]); finish(axes[1], "truth: verifier accuracy", "RL step", "accuracy", legend=False)
axes[2].plot(smooth(log_rm["length"], 5), color=PALETTE[2]); finish(axes[2], "response length (tokens)", "RL step", "tokens", legend=False)
plt.suptitle("Optimising the reward model instead of the verifier", x=0.01, ha="left", fontsize=12, fontweight="bold"); plt.tight_layout(); plt.show()
seed_everything(2); print("samples from the RM-optimised policy:"); show_samples(policy_rm, [(3, 4), (2, 5), (13, 8), (17, 19)])
"""))

CELLS.append(md(r"""
The proxy climbs while the truth collapses. Look at the samples: the policy has drifted to strings the
reward model never saw in training (a lone digit, or malformed outputs containing `=` or `+`), which the RM
scores highly by *extrapolation*. Nothing in the Bradley–Terry loss constrains scores off the training
distribution, and the policy gradient is an efficient search for exactly those inputs. Compare with the SFT
samples in Part 10. In production RLHF this is the failure mode people mean by "reward hacking", and it
is why every recipe since has combined a learned reward with a KL leash (we used a small β = 0.01 here on
purpose), verifiers, or both.

### Test your knowledge
"""))

CELLS += quiz_cells("12.1", "Why is the KL penalty put INTO the per-token reward rather than added as a separate loss term?",
    ["It is cheaper to compute",
     "So the critic learns to predict it and GAE distributes it over tokens; the policy is then optimised against a single consistent objective (reward minus KL)",
     "Because PPO cannot have two loss terms",
     "It is not; PPO always adds it as a loss"], "B",
    "Both placements exist. Reward shaping (OpenAI's original recipe) folds the constraint into the return so credit "
    "assignment handles it; GRPO instead adds a KL term to the loss directly (Part 15). Each choice has its own bias/variance trade-offs.")

CELLS += quiz_cells("12.2", "In PPO-RLHF the critic predicts the value of a text prefix. Why is this harder than predicting the value of a CartPole state?",
    ["Text is discrete",
     "The reward is a single sparse scalar at the end of a long sequence, and the 'state' is a whole prefix whose future depends on subtle wording, so the value regression is noisy and needs a model as large as the policy",
     "Because gamma = 1",
     "It isn't harder"], "B",
    "A value model for language is essentially another LLM that must judge partial responses. It is expensive, and its "
    "errors inject bias into every advantage. This is the core motivation for GRPO and RLOO's Monte Carlo baselines.")

CELLS += quiz_cells("12.3", "Which set of models must be held in memory for classic PPO-RLHF?",
    ["Policy only",
     "Policy and reward model",
     "Policy, reference policy, critic (value model), and reward model",
     "Policy and critic"], "C",
    "Four models, two of which (policy, critic) are trained. With 70B-parameter models this dominates the engineering. "
    "GRPO drops the critic; RLVR replaces the reward model by a program; DPO drops both plus the sampling loop.")

CELLS += exercise_cells("12.4",
    r"""
### Exercise 12.4 · Per-token reward shaping

Implement `my_shape_rewards(final_reward, logp, ref_logp, mask, beta)` from scratch. `logp`, `ref_logp`, `mask`
are `(B, T)` tensors; `final_reward` is `(B,)`. Every real token gets $-\beta(\log\pi - \log\pi_{\text{ref}})$;
the **last real token** of each sequence additionally gets the final reward. Padding tokens get exactly 0.
""",
    r"""
def my_shape_rewards(final_reward, logp, ref_logp, mask, beta):
    # YOUR CODE HERE
    raise NotImplementedError
""",
    r"""
def _tests(fn):
    logp = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -0.5, -0.5]]); ref = torch.tensor([[-1.5, -2.0, -2.0], [-1.0, -1.0, -1.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]); R = torch.tensor([1.0, 0.0])
    out = fn(R, logp, ref, mask, beta=0.1)
    exp = torch.tensor([[-0.1 * 0.5, -0.1 * 0.0 + 1.0, 0.0], [-0.05, -0.05, -0.05]])
    assert out is not None and torch.allclose(out, exp, atol=1e-6), f"expected {exp.tolist()}, got {out.tolist()}"
""",
    r"""
def my_shape_rewards(final_reward, logp, ref_logp, mask, beta):
    r = -beta * (logp - ref_logp) * mask
    last = (mask.sum(1).long() - 1).clamp(min=0)
    r[torch.arange(len(mask)), last] += final_reward
    return r
""", "my_shape_rewards")
