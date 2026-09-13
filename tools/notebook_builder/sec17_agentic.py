from common import md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 17 · Agentic RL: multi-turn episodes, tools, and loss masking

Coding assistants, research agents and computer-use models are trained with the same algorithms as
Parts 12–16, but the *episode* looks different. A single trajectory is now:

```
prompt → model turn → tool call → tool output → model turn → tool call → … → final answer → reward
```

The tool outputs (a compiler error, a search result, a file's contents) are **observations from the
environment**, not actions. That changes four things in the implementation:

1. **Loss masking.** Only tokens the model generated get a gradient (and an importance ratio). Tool
   outputs are masked out exactly like the prompt. Getting this wrong makes the model learn to
   *predict* tool outputs, which is both useless and a source of hallucinated tool results.
2. **Rewards are outcome-based and often verifiable**: tests pass, the task's checker succeeds, the
   answer matches. Costs (tokens, tool latency, number of turns) are usually folded in as small penalties.
3. **Credit assignment spans many turns.** Most systems still assign the same sequence-level advantage
   to every model token of the whole multi-turn trajectory (GRPO-style); finer per-turn credit is an
   open research area.
4. **Rollouts are long, slow and asynchronous**, so training is partially off-policy, which is what the
   importance-sampling corrections at the end of Part 16 are for.

## A calculator tool for our toy model

We give the model a `<call>` token. When it emits one, the environment appends `<obs> digits <obs>` with
the correct sum and the model continues. The tool is free to *use* but we charge a small cost
(0.25 reward) per call, so the interesting question is: **does the policy learn to call the tool only when
it needs to?** We fine-tune a fresh model on demonstrations that use the tool half the time regardless of
difficulty; its *direct* answers are reliable on easy prompts and right only one time in five on hard ones.
"""))

CELLS.append(code(r"""
def calculator_obs(prompt_tokens):
    a = 10 * int(prompt_tokens[0]) + int(prompt_tokens[1]); b = 10 * int(prompt_tokens[3]) + int(prompt_tokens[4])
    return [OBS] + [int(ch) for ch in str(a + b)] + [OBS]

@torch.no_grad()
def generate_with_tool(model, prompts, max_len=CTX, temperature=1.0, greedy=False):
    '''Agentic generation loop. On <call> the environment inserts the calculator's observation tokens.
    Returns seq (B, max_len) and gen_mask (B, max_len - PROMPT_LEN): 1 for MODEL tokens, 0 for env tokens / padding.'''
    B = prompts.shape[0]
    seqs = [p.tolist() for p in prompts]; roles = [[] for _ in range(B)]; alive = [True] * B
    while any(alive):
        idx = [i for i in range(B) if alive[i]]
        L = max(len(seqs[i]) for i in idx)
        batch = torch.full((len(idx), L), PAD, dtype=torch.long)
        for k, i in enumerate(idx): batch[k, :len(seqs[i])] = torch.tensor(seqs[i])
        logits = model(batch)[torch.arange(len(idx)), torch.tensor([len(seqs[i]) - 1 for i in idx])] / temperature
        nxt = logits.argmax(-1) if greedy else torch.multinomial(F.softmax(logits, -1), 1).squeeze(1)
        for k, i in enumerate(idx):
            t = int(nxt[k]); seqs[i].append(t); roles[i].append(1)                  # model token
            if t == EOS or len(seqs[i]) >= max_len:
                alive[i] = False
            elif t == CALL:                                                          # environment turn
                for o in calculator_obs(seqs[i][:PROMPT_LEN]):
                    if len(seqs[i]) < max_len: seqs[i].append(o); roles[i].append(0)
                if len(seqs[i]) >= max_len: alive[i] = False
    seq = torch.full((B, max_len), PAD, dtype=torch.long); gen_mask = torch.zeros(B, max_len - PROMPT_LEN)
    for i in range(B):
        seq[i, :len(seqs[i])] = torch.tensor(seqs[i]); gen_mask[i, :len(roles[i])] = torch.tensor(roles[i], dtype=torch.float32)
    return seq, gen_mask

def used_tool(tokens): return CALL in [int(t) for t in tokens]

def verify_tool(a, b, completion_tokens):
    '''Final answer = digits after the last <obs> (if the tool was used), else the whole completion.'''
    toks = [int(t) for t in completion_tokens]
    if OBS in toks:
        last_obs = max(i for i, t in enumerate(toks) if t == OBS); toks = toks[last_obs + 1:]
    return verify(a, b, toks)

TOOL_COST = 0.25
tool_reward = lambda a, b, seq: verify_tool(a, b, seq[PROMPT_LEN:].tolist()) - TOOL_COST * used_tool(seq[PROMPT_LEN:].tolist())

def roles_for_completion(y):
    '''1 for model tokens, 0 for the environment's <obs>...<obs> segment.'''
    roles, in_obs, seen = [], False, 0
    for t in y:
        if t == OBS:
            roles.append(0); seen += 1; in_obs = (seen % 2 == 1)
        else:
            roles.append(0 if in_obs else 1)
    return roles

def make_tool_sft_dataset(pairs, per_prompt=8, tool_prob=0.5, hard_correct_prob=0.5):
    X, Y, R = [], [], []
    for a, b in pairs:
        for _ in range(per_prompt):
            X.append(encode_prompt(a, b))
            if np.random.rand() < tool_prob:                                        # demonstration with the tool
                y = [CALL] + calculator_obs(X[-1]) + encode_answer(a + b)
            else:                                                                   # direct answer, noisy on hard prompts
                ans = a + b
                if not is_easy(a, b) and np.random.rand() > hard_correct_prob:
                    ans = max(0, ans + int(np.random.choice([d for d in range(-9, 10) if d != 0])))
                y = encode_answer(ans)
            Y.append(y); R.append(roles_for_completion(y))
    return X, Y, R
"""))

CELLS.append(code(r"""
seed_everything(0)
X_tool, Y_tool, R_tool = make_tool_sft_dataset(ALL_PAIRS, hard_correct_prob=0.2)   # direct hard answers: right 1 time in 5
i = next(k for k in range(len(Y_tool)) if CALL in Y_tool[k])
print("tool demonstration:", decode(X_tool[i]), "->", decode(Y_tool[i]), "| loss mask:", R_tool[i])
tool_sft_model = TinyLM()
with Timer("tool SFT"):
    sft_train(tool_sft_model, X_tool, Y_tool, roles=R_tool, total_len=CTX)
"""))

CELLS.append(code(r"""
@torch.no_grad()
def tool_stats(model, n_samples=4):
    out = {}
    for tier, pairs in [("easy", EASY_PAIRS), ("hard", HARD_PAIRS)]:
        seq, _ = generate_with_tool(model, batch_prompts(pairs, n_samples))
        out[f"tool_rate_{tier}"] = float(np.mean([used_tool(s[PROMPT_LEN:].tolist()) for s in seq]))
    return out

_, ev = evaluate_lm(tool_sft_model, n_samples=4, gen_fn=generate_with_tool, verify_fn=verify_tool)
print("tool-SFT model: accuracy {acc:.3f} (easy {easy:.2f}, hard {hard:.2f})".format(**ev), "| tool use:", tool_stats(tool_sft_model))
seed_everything(3)
seq, gm = generate_with_tool(tool_sft_model, batch_prompts([(3, 4), (16, 17)], 3))
for s, m in zip(seq, gm):
    print(f"  {decode(s[:PROMPT_LEN]):8s} {decode(s[PROMPT_LEN:]):22s} model-token mask: {m[:int((s != PAD).sum()) - PROMPT_LEN].int().tolist()}")
"""))

CELLS.append(md(r"""
The SFT model uses the tool about half the time regardless of difficulty, because that is what the
demonstrations did. Now we run GRPO with the tool-aware reward. Everything from Part 16 is reused
unchanged; the only differences are the generation function and the mask it returns.
"""))

CELLS.append(code(r"""
seed_everything(0)
policy_tool = copy.deepcopy(tool_sft_model)
cfg_tool = RLConfig("GRPO (tool)", beta=0.02)
with Timer("agentic GRPO"):
    log_tool = train_llm_rl(make_rl_step(policy_tool, tool_sft_model, cfg_tool, reward_fn=tool_reward, gen_fn=generate_with_tool),
                            policy_tool, steps=150, gen_fn=generate_with_tool, verify_fn=verify_tool, extra_eval=tool_stats, verbose=False)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
axes[0].plot(log_tool["eval_step"], log_tool["eval_easy"], marker="o", ms=3, label="easy prompts"); axes[0].plot(log_tool["eval_step"], log_tool["eval_hard"], marker="o", ms=3, label="hard prompts")
finish(axes[0], "final-answer accuracy", "RL step", "accuracy")
axes[1].plot(log_tool["eval_step"], log_tool["tool_rate_easy"], marker="o", ms=3, label="easy prompts"); axes[1].plot(log_tool["eval_step"], log_tool["tool_rate_hard"], marker="o", ms=3, label="hard prompts")
finish(axes[1], f"tool-call rate (tool cost {TOOL_COST})", "RL step", "tool-call rate", legend=False)
axes[2].plot(smooth(log_tool["reward"], 5)); finish(axes[2], "mean shaped reward per sample", "RL step", "reward", legend=False)
plt.suptitle("Learning when to use a tool", x=0.01, ha="left", fontsize=12, fontweight="bold"); plt.tight_layout(); plt.show()
print("final tool-use rates:", tool_stats(policy_tool))
seed_everything(4); seq, _ = generate_with_tool(policy_tool, batch_prompts([(2, 5), (7, 1), (14, 18), (19, 9)], 2))
for s in seq: print(f"  {decode(s[:PROMPT_LEN]):8s} -> {decode(s[PROMPT_LEN:])}")
"""))

CELLS.append(md(r"""
The policy was never told which prompts are hard. It discovered from reward alone that a 0.25 tool
cost is worth paying when its own answer is unreliable and not otherwise. Scale this up (thousands of
tools, tests as the verifier, long multi-turn trajectories, an asynchronous rollout fleet) and you have
the training loop behind current coding agents.

Try `hard_correct_prob=0.5` in the SFT data: the direct answers are then good enough that RL *sharpens
them past the tool's 0.75 payoff* and the policy abandons the tool on hard prompts too. Whether an agent
internalises a skill or outsources it depends on the tool's cost relative to its own reliability, and RL
will find that break-even point.

### What is still hard (and where the field is going)
* **Credit assignment across turns.** One scalar for a 50-turn trajectory is a weak signal. Process
  reward models, turn-level advantages and learned critics for agents are active work.
* **Reward hacking with real tools.** An agent rewarded for "tests pass" can delete the tests, mock the
  network, or special-case the checker. Sandboxes, held-out tests and judge models are part of the reward.
* **Environment engineering.** Most of the effort in agentic RL now goes into building diverse,
  verifiable environments and curricula, not into the loss function.
* **Off-policy stability at scale.** Long rollouts force asynchrony; sequence-level importance
  sampling (GSPO), truncated IS, and periodic reference resets (ProRL) are the current tools.

### Test your knowledge
"""))

CELLS += quiz_cells("17.1", "Why must tool-output tokens be excluded from the policy-gradient loss?",
    ["They are too long",
     "They were produced by the environment, not sampled from the policy, so they carry no action for the gradient to reinforce and including them would train the model to imitate (hallucinate) tool outputs",
     "Because they contain digits",
     "They should be included; masking is only for padding"], "B",
    "In MDP terms the observation is part of s_{t+1}, not a_t. The log-probability ratio and the advantage only make "
    "sense for tokens the policy chose. Prompt tokens are masked for the same reason.")

CELLS += quiz_cells("17.2", "In our experiment, why did the policy keep calling the tool on hard prompts instead of eventually learning to add directly?",
    ["The tool cost was zero",
     "Once it routes hard prompts to the tool it stops generating direct answers for them, so it collects no data that would improve its direct accuracy: the equilibrium is self-reinforcing",
     "The KL penalty forbids direct answers",
     "Hard prompts have no correct answer"], "B",
    "RL only improves behaviours it samples. This is a small example of a general phenomenon: the policy's "
    "exploration is shaped by its current strategy, and a cheap tool can crowd out learning the underlying skill.")

CELLS += quiz_cells("17.3", "A coding agent is rewarded when the repository's tests pass. Which is a realistic reward-hacking risk?",
    ["The agent writes code that is too well documented",
     "The agent modifies or deletes the failing tests, or special-cases the test inputs, so the checker passes without solving the task",
     "The agent refuses to use tools",
     "None; test passing is a verifiable reward and cannot be hacked"], "B",
    "Verifiable does not mean un-hackable. Production setups hold out tests the agent cannot see, run in sandboxes, "
    "and add judge models or rule checks for such behaviours, and those checks themselves become part of the reward.")

CELLS += exercise_cells("17.4",
    r"""
### Exercise 17.4 · Masked sequence log-probability with roles

Implement `masked_seq_logprob(logp, roles)` where `logp` is `(B, T)` per-token log-probs of a completion and
`roles` is `(B, T)` with 1 for model tokens and 0 for environment tokens or padding. Return the `(B,)` tensor of
summed log-probs over **model tokens only**. (This is the quantity RLOO/GRPO use for agentic trajectories.)
""",
    r"""
def masked_seq_logprob(logp, roles):
    # YOUR CODE HERE
    raise NotImplementedError
""",
    r"""
def _tests(fn):
    logp = torch.tensor([[-1.0, -2.0, -3.0, -4.0], [-0.5, -0.5, -0.5, -0.5]]); roles = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
    out = fn(logp, roles)
    assert out is not None and torch.allclose(out, torch.tensor([-5.0, -1.0])), f"got {out.tolist() if out is not None else None}"
""",
    r"""
def masked_seq_logprob(logp, roles):
    return (logp * roles).sum(1)
""", "masked_seq_logprob")
