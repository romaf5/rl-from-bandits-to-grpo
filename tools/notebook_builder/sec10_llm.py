from common import md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 10 · RL for language models: the setting

Everything so far was about balancing poles. The same machinery now trains the systems you talk to.
The mapping is exact once you see it:

| RL concept | In an LLM |
|---|---|
| state $s_t$ | the prompt plus the tokens generated so far |
| action $a_t$ | the next token (vocabulary of ~100k "arms") |
| policy $\pi_\theta(a_t \mid s_t)$ | the LM's next-token distribution |
| transition | deterministic: append the token |
| episode / trajectory | one complete response |
| reward | usually a single number at the end: a reward model's score, a unit test passing, a correct final answer |
| discount $\gamma$ | 1 (a few thousand steps, all of which matter) |

Two things are different from CartPole and shape every algorithm in Parts 11–17:

1. **The reward is sparse and terminal.** One scalar for a thousand-token response. Credit assignment
   across tokens is brutal, so most methods give every token of a response the *same* advantage.
2. **We start from a very good policy.** Pretraining plus supervised fine-tuning already produce
   sensible text. RL's job is to *sharpen* and *steer* that policy, not to learn from scratch, and
   drifting too far from the starting point destroys capabilities. Hence the ubiquitous **KL penalty to a
   reference model**.

## The post-training pipeline

The classic RLHF recipe (Ouyang et al., 2022, InstructGPT) has three stages after pretraining.
Newer "reasoning" recipes (DeepSeek-R1, 2025) replace the reward model with a **verifier** (RL with
verifiable rewards, RLVR), and DPO-style methods (Part 13) skip the RL loop entirely.
"""))

CELLS.append(code(r"""
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.8)); ax.set_xlim(0, 12); ax.set_ylim(0.3, 3.7); ax.axis("off"); ax.grid(False)
    def box(x, y, w, h, text, color, sub=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.15", fc="white", ec=color, lw=1.6))
        ax.text(x + w / 2, y + h / 2 + (0.16 if sub else 0), text, ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold")
        if sub: ax.text(x + w / 2, y + h / 2 - 0.22, sub, ha="center", va="center", fontsize=8.5, color=INK2)
    def arrow(x0, y0, x1, y1, color=MUTED, style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14, lw=1.4, color=color, connectionstyle="arc3,rad=0.0"))
    box(0.2, 2.2, 2.2, 1.0, "Pretraining", PALETTE[0], "next-token prediction\non the internet")
    box(2.9, 2.2, 2.2, 1.0, "SFT", PALETTE[0], "imitate demonstrations\n(Part 10)")
    box(5.6, 2.2, 2.4, 1.0, "Reward model", PALETTE[1], "Bradley-Terry on\nhuman preferences (Part 11)")
    box(8.5, 2.2, 3.2, 1.0, "RL: PPO / GRPO / ...", PALETTE[2], "maximise reward - β·KL(π‖π_ref)\n(Parts 12, 14-17)")
    arrow(2.4, 2.7, 2.9, 2.7); arrow(5.1, 2.7, 5.6, 2.7); arrow(8.0, 2.7, 8.5, 2.7)
    box(5.6, 0.5, 2.4, 1.0, "Verifier", PALETTE[3], "unit tests, exact answers\n(RLVR: DeepSeek-R1 style)")
    arrow(8.0, 1.0, 9.2, 2.2)
    box(2.9, 0.5, 2.2, 1.0, "DPO and friends", PALETTE[4], "preferences straight into\nthe policy, no RL (Part 13)")
    arrow(4.0, 2.2, 4.0, 1.5)
    ax.text(4.1, 1.85, "skip RM + RL", fontsize=8, color=INK2)
    ax.set_title("LLM post-training: where each Part of this notebook fits")
    plt.show()
draw_pipeline()
"""))

CELLS.append(md(r"""
## A toy language model we can train in seconds

We need a "language model" small enough to train on a CPU in seconds but real enough that
token-level RL makes sense. We'll use a **two-layer transformer** with a 16-token vocabulary on a
mini arithmetic task:

> prompt: `07+15=` → completion: `22<eos>`

* **Easy prompts** have both operands below 10 (100 prompts). **Hard** prompts have at least one
  two-digit operand (300 prompts). This is our stand-in for problems of varying difficulty.
* The **verifier** is exact-match against the true sum in canonical form (`22`, not `022`).
* Two extra tokens, `<call>` and `<obs>`, are reserved for the tool-use agent in Part 17.

We deliberately fine-tune the model on **noisy demonstrations**: hard prompts get the correct answer
only half the time. That mimics a real base model that "knows" the answer sometimes, which is exactly
when RL helps: the model already contains the right behaviour at some probability, and RL amplifies it.
"""))

CELLS.append(code(r"""
# ---- Vocabulary and task ----
VOCAB = [str(d) for d in range(10)] + ["+", "=", "<eos>", "<pad>", "<call>", "<obs>"]
PLUS, EQ, EOS, PAD, CALL, OBS = 10, 11, 12, 13, 14, 15
V_SIZE, PROMPT_LEN, MAX_NEW, CTX = len(VOCAB), 6, 4, 16
MAX_OPERAND = 19
ALL_PAIRS = [(a, b) for a in range(MAX_OPERAND + 1) for b in range(MAX_OPERAND + 1)]
def is_easy(a, b): return a < 10 and b < 10
EASY_PAIRS = [p for p in ALL_PAIRS if is_easy(*p)]; HARD_PAIRS = [p for p in ALL_PAIRS if not is_easy(*p)]

def encode_prompt(a, b): return [a // 10, a % 10, PLUS, b // 10, b % 10, EQ]
def encode_answer(n, zero_pad=False): return [int(ch) for ch in (f"{n:02d}" if zero_pad else str(n))] + [EOS]
def decode(tokens): return "".join(VOCAB[int(t)] for t in tokens if int(t) != PAD)

def parse_completion(tokens):
    '''Digits up to <eos>. Returns (string, well_formed).'''
    digits = []
    for t in tokens:
        t = int(t)
        if t == EOS: return "".join(map(str, digits)), True
        if t > 9: return None, False                          # any non-digit token before <eos> is malformed
        digits.append(t)
    return None, False                                        # ran out of tokens without <eos>

def verify(a, b, completion_tokens):
    '''The verifiable reward: 1.0 iff the completion is exactly the canonical sum followed by <eos>.'''
    s, ok = parse_completion(completion_tokens)
    return float(ok and s == str(a + b))

print(f"{len(ALL_PAIRS)} prompts ({len(EASY_PAIRS)} easy, {len(HARD_PAIRS)} hard). Example prompt:", decode(encode_prompt(7, 15)),
      "-> answer", decode(encode_answer(22)), "| reward for '22<eos>':", verify(7, 15, encode_answer(22)),
      "| for '022<eos>':", verify(7, 15, [0, 2, 2, EOS]))
"""))

CELLS.append(code(r"""
# ---- A tiny GPT ----
class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) for t in (q, k, v)]
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)      # causal self-attention
        x = x + self.proj(att.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))

class TinyLM(nn.Module):
    def __init__(self, vocab=V_SIZE, d=64, n_layers=2, n_heads=4, ctx=CTX):
        super().__init__()
        self.tok_emb, self.pos_emb = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln_f, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)
    def hidden(self, idx):                        # (B, T) -> (B, T, d)
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(idx.shape[1]))
        for blk in self.blocks: x = blk(x)
        return self.ln_f(x)
    def forward(self, idx):                       # (B, T) -> logits (B, T, vocab)
        return self.head(self.hidden(idx))

def token_logprobs(model, seq):
    '''log pi(seq[:, t+1] | seq[:, :t+1]) for every position: (B, L) -> (B, L-1).'''
    logits = model(seq[:, :-1])
    return F.log_softmax(logits, -1).gather(-1, seq[:, 1:, None]).squeeze(-1)

def completion_logprobs(model, seq, prompt_len=PROMPT_LEN):
    '''Per-token log-probs of the generated part only: (B, L-prompt_len).'''
    return token_logprobs(model, seq)[:, prompt_len - 1:]

@torch.no_grad()
def generate(model, prompts, max_new=MAX_NEW, temperature=1.0, greedy=False):
    '''Sample completions. Returns seq (B, P+max_new) and mask (B, max_new): 1 for real generated tokens
    (up to and including <eos>), 0 for padding after <eos>.'''
    seq, B = prompts.clone(), prompts.shape[0]
    alive, mask = torch.ones(B, dtype=torch.bool), torch.zeros(B, max_new)
    for t in range(max_new):
        logits = model(seq)[:, -1] / temperature
        nxt = logits.argmax(-1) if greedy else torch.multinomial(F.softmax(logits, -1), 1).squeeze(1)
        nxt = torch.where(alive, nxt, torch.full_like(nxt, PAD))
        mask[:, t] = alive
        seq = torch.cat([seq, nxt[:, None]], 1)
        alive = alive & (nxt != EOS)
    return seq, mask

lm = TinyLM()
print(f"TinyLM parameters: {sum(p.numel() for p in lm.parameters()):,}")
seq, mask = generate(lm, torch.tensor([encode_prompt(7, 15)]))
print("untrained sample:", repr(decode(seq[0, PROMPT_LEN:])), "| mask:", mask[0].tolist())
"""))

CELLS.append(md(r"""
### Supervised fine-tuning (SFT)

SFT is plain maximum likelihood on demonstration completions, with the loss masked to the completion
tokens (we don't train the model to predict the prompt). We build the dataset with the noise described above,
plus a small "formatting quirk": 5% of single-digit answers are written zero-padded (`07`). The verifier
counts those as wrong, so the SFT model starts with a small format problem that RL can fix.
"""))

CELLS.append(code(r"""
def make_sft_dataset(pairs, per_prompt=8, hard_correct_prob=0.5, zero_pad_prob=0.05):
    X, Y = [], []
    for a, b in pairs:
        for _ in range(per_prompt):
            ans = a + b
            if not is_easy(a, b) and np.random.rand() > hard_correct_prob:              # noisy label on hard prompts
                ans = max(0, ans + int(np.random.choice([d for d in range(-9, 10) if d != 0])))
            X.append(encode_prompt(a, b)); Y.append(encode_answer(ans, zero_pad=(ans < 10 and np.random.rand() < zero_pad_prob)))
    return X, Y

def pad_batch(X, Y, total_len=PROMPT_LEN + MAX_NEW, roles=None):
    '''Right-pad prompt+completion. loss_mask[i, t] = 1 where seq[i, t+1] is a completion token the MODEL should predict
    (roles, if given, mark which completion tokens belong to the model (1) vs. the environment (0); see Part 17).'''
    seq = torch.full((len(X), total_len), PAD, dtype=torch.long); loss_mask = torch.zeros(len(X), total_len - 1)
    for i, (x, y) in enumerate(zip(X, Y)):
        toks = x + y; seq[i, :len(toks)] = torch.tensor(toks)
        r = roles[i] if roles is not None else [1] * len(y)
        loss_mask[i, len(x) - 1:len(toks) - 1] = torch.tensor(r, dtype=torch.float32)   # predict completion tokens only
    return seq, loss_mask

def sft_train(model, X, Y, steps=1500, batch_size=128, lr=3e-3, verbose=True, roles=None, total_len=PROMPT_LEN + MAX_NEW):
    seq, loss_mask = pad_batch(X, Y, total_len, roles)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(seq), (batch_size,))
        logp = token_logprobs(model, seq[idx])
        loss = -(logp * loss_mask[idx]).sum() / loss_mask[idx].sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        losses.append(loss.item())
        if verbose and (step + 1) % 500 == 0: print(f"  sft step {step + 1}: loss {np.mean(losses[-100:]):.3f}")
    return losses

seed_everything(0)
X_sft, Y_sft = make_sft_dataset(ALL_PAIRS)
print(f"{len(X_sft)} SFT examples, e.g. {decode(X_sft[900])} -> {decode(Y_sft[900])}")
sft_model = TinyLM()
with Timer("SFT"):
    sft_losses = sft_train(sft_model, X_sft, Y_sft)
"""))

CELLS.append(code(r"""
from math import comb

@torch.no_grad()
def evaluate_lm(model, pairs=ALL_PAIRS, n_samples=8, temperature=1.0, greedy=False, gen_fn=None, verify_fn=None):
    '''Sample n completions per prompt. Returns per-prompt correctness matrix (len(pairs), n) and summary dict.'''
    gen_fn, verify_fn = gen_fn or generate, verify_fn or verify
    prompts = torch.tensor([encode_prompt(a, b) for a, b in pairs]).repeat_interleave(n_samples, 0)
    seq, mask = gen_fn(model, prompts, temperature=temperature, greedy=greedy)
    rewards = np.array([verify_fn(a, b, seq[i * n_samples + j, PROMPT_LEN:].tolist()) for i, (a, b) in enumerate(pairs) for j in range(n_samples)])
    R = rewards.reshape(len(pairs), n_samples)
    easy = np.array([is_easy(a, b) for a, b in pairs])
    return R, dict(acc=R.mean(), easy=R[easy].mean(), hard=R[~easy].mean(), length=(mask.sum(1).mean().item()))

def pass_at_k(R, k):
    '''Unbiased pass@k estimator (Chen et al., 2021) from n samples with c correct per prompt.'''
    n = R.shape[1]; c = R.sum(1)
    return np.mean([1.0 - comb(int(n - ci), k) / comb(n, k) if n - ci >= k else 1.0 for ci in c])

R_sft, stats = evaluate_lm(sft_model, n_samples=32)
print("SFT model, sampling at T=1:  accuracy (pass@1) = {acc:.3f}   easy = {easy:.3f}   hard = {hard:.3f}   mean length = {length:.2f}".format(**stats))
_, greedy_stats = evaluate_lm(sft_model, n_samples=1, greedy=True)
print("SFT model, greedy decoding:  accuracy = {acc:.3f}   easy = {easy:.3f}   hard = {hard:.3f}".format(**greedy_stats))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].plot(smooth(sft_losses, 20)); finish(axes[0], "SFT loss (masked cross-entropy)", "step", "loss", legend=False)
ks = [1, 2, 4, 8, 16, 32]
axes[1].plot(ks, [pass_at_k(R_sft, k) for k in ks], marker="o", label="all prompts")
axes[1].plot(ks, [pass_at_k(R_sft[[is_easy(*p) for p in ALL_PAIRS]], k) for k in ks], marker="o", label="easy")
axes[1].plot(ks, [pass_at_k(R_sft[[not is_easy(*p) for p in ALL_PAIRS]], k) for k in ks], marker="o", label="hard")
axes[1].set_xscale("log", base=2); axes[1].set_ylim(0, 1.02)
finish(axes[1], "pass@k of the SFT model: the right answer is in there", "k samples", "pass@k")
plt.tight_layout(); plt.show()
"""))

CELLS.append(md(r"""
**This plot is the whole motivation for RLVR.** With one sample the SFT model gets a hard prompt right
about half the time; with 8 samples it almost always produces the right answer *somewhere*. The
knowledge is in the model; the sampling distribution is just not concentrated on it. RL with a
verifier takes the samples that happened to be right and makes them more likely. Whether RL can teach
*genuinely new* skills beyond what pass@k reveals is an active research debate (Yue et al., 2025); for our
toy the sharpening view is exactly right.

We'll keep `sft_model` frozen as the **reference policy** $\pi_{\text{ref}}$ for every method that follows.
Let's also look at a few actual samples so the task feels concrete.
"""))

CELLS.append(code(r"""
def show_samples(model, pairs, n=4, temperature=1.0):
    prompts = torch.tensor([encode_prompt(a, b) for a, b in pairs]).repeat_interleave(n, 0)
    seq, _ = generate(model, prompts, temperature=temperature)
    for i, (a, b) in enumerate(pairs):
        outs = [decode(seq[i * n + j, PROMPT_LEN:]) for j in range(n)]
        marks = ["✓" if verify(a, b, seq[i * n + j, PROMPT_LEN:].tolist()) else "✗" for j in range(n)]
        print(f"{decode(encode_prompt(a, b)):8s} (true {a + b:2d}) : " + "   ".join(f"{o:8s}{m}" for o, m in zip(outs, marks)))
seed_everything(1)
show_samples(sft_model, [(3, 4), (7, 2), (13, 8), (17, 19), (9, 16)])
"""))

CELLS.append(md(r"""
### Test your knowledge
"""))

CELLS += quiz_cells("10.1", "In the LLM-as-MDP view, what is the 'state' at step t?",
    ["The current token only",
     "The hidden activations of the last layer",
     "The prompt plus all tokens generated so far",
     "The reward received so far"], "C",
    "The next-token distribution is conditioned on the whole prefix, so the prefix is the state. It grows every step, "
    "the transition is deterministic (append the sampled token), and the Markov property holds trivially.")

CELLS += quiz_cells("10.2", "Why is the KL penalty to a reference model so central in LLM RL but absent in CartPole?",
    ["CartPole has a discrete action space",
     "We start from a strong pretrained policy whose capabilities we must not destroy, and the reward signal is too narrow to define good behaviour on its own",
     "LLMs have more parameters",
     "Because gamma = 1 for LLMs"], "B",
    "The reward (a reward model score, a unit test) covers a sliver of what makes a response good. Unconstrained "
    "optimisation finds outputs that score well and are otherwise degenerate (reward hacking). Staying close to pi_ref "
    "keeps the rest of the behaviour intact and keeps the reward model's inputs in-distribution.")

CELLS += quiz_cells("10.3", "pass@8 is far above pass@1 for the SFT model. What does that tell you about what RL with a verifier can do cheaply?",
    ["Nothing, they measure different things",
     "That the model must learn a new algorithm for addition",
     "That the correct answer already has substantial probability, so RL mainly needs to reallocate probability mass toward it",
     "That greedy decoding is always better"], "C",
    "If the model can produce the right answer in 1 of 8 samples, the verifier will find those samples and the policy "
    "gradient will raise their probability. This 'sharpening' is fast. Teaching skills the base model cannot sample at all is much harder.")
