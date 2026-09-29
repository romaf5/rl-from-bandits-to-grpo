from common import video, md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
---
# Part 18 · Wrap-up: the map
"""))
CELLS.append(video(18))

CELLS.append(md(r"""
## One table

| Algorithm | Family | On/off-policy | Learns | Baseline / critic | Key trick | Where you meet it |
|---|---|---|---|---|---|---|
| Value iteration / policy iteration | DP | model-based | $V^*$ | exact | Bellman contraction | planning, textbooks |
| MC / TD(0) / SARSA | tabular | on-policy | $V$, $Q$ | none | bootstrapping | foundations |
| Q-learning / DQN / Double DQN | value-based | off-policy | $Q^*$ | none (target net) | replay + target network | Atari, discrete control |
| REINFORCE | policy gradient | on-policy | $\pi$ | optional constant | log-derivative trick, reward-to-go | teaching, bandits |
| A2C / A3C | actor-critic | on-policy | $\pi$, $V$ | learned $V$ | GAE advantages | classic deep RL |
| TRPO | trust region | on-policy | $\pi$, $V$ | learned $V$ | natural gradient + KL constraint | robotics (historical) |
| PPO | trust region (1st order) | almost on-policy | $\pi$, $V$ | learned $V$ | clipped ratio, many epochs | robotics, games, RLHF |
| PPO-RLHF | PPO + LLM | almost on-policy | $\pi$, $V$ | value LM | per-token KL reward | InstructGPT, ChatGPT era |
| DPO / IPO / KTO / SimPO | preference optimisation | offline | $\pi$ | implicit reward | closed-form RLHF solution | open chat models |
| RLOO / REINFORCE++ | critic-free PG | on-policy | $\pi$ | leave-one-out / batch stats | Monte Carlo baseline | 2024–25 LLM RL |
| GRPO | critic-free PG | almost on-policy | $\pi$ | group mean/std | group-normalised advantages, k3 KL | DeepSeek-R1 and descendants |
| Dr. GRPO / DAPO / Lite PPO | GRPO fixes | almost on-policy | $\pi$ | group mean | token-level loss, clip-higher, dynamic sampling | 2025 open reasoning models |
| GSPO | sequence-level PG | almost on-policy | $\pi$ | group mean/std | sequence-level ratio & clip | Qwen3 |
| CISPO | REINFORCE + clipped IS | almost on-policy | $\pi$ | group | detached clipped weights, no dead tokens | MiniMax-M1, ScaleRL |
| VAPO | value-based LLM RL | almost on-policy | $\pi$, $V$ | value LM | critic pre-training, decoupled GAE | ByteDance Seed |

## A decision guide

* **Small discrete problem, known model?** Value iteration. Done.
* **Discrete actions, cheap simulator, want sample efficiency?** DQN family (add Double DQN, prioritised replay, n-step, dueling: "Rainbow").
* **Continuous control or you just want something that works?** PPO. Tune epochs, clip, entropy, and *always* look at approx-KL and clip fraction.
* **Aligning an LLM to preferences with limited compute?** DPO (or SimPO) on a good preference set; iterate online if you can.
* **Improving an LLM on a task with a verifier (maths, code, agents)?** GRPO-style with the 2025 fixes: token-level loss, clip-higher, dynamic sampling, no or small KL, sequence-level or detached importance weights if your rollouts are async. Spend your effort on the environment and the reward.
* **Something breaks?** Check, in order: the loss mask, the done/truncation flags, the advantage normalisation, the KL/clip diagnostics, the reward function for exploits.

## Reading list (in the order the notebook covered them)

* Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Parts I–II: everything in Parts 1–4.
* Mnih et al., 2015, "Human-level control through deep reinforcement learning" (DQN); van Hasselt et al., 2016 (Double DQN).
* Williams, 1992 (REINFORCE); Sutton et al., 2000 (policy gradient theorem); Schulman et al., 2016 (GAE).
* Schulman et al., 2015 (TRPO); Schulman et al., 2017 (PPO); Engstrom et al., 2020, "Implementation matters in deep policy gradients".
* Ziegler et al., 2019 and Ouyang et al., 2022 (RLHF / InstructGPT); Gao et al., 2022 (reward-model over-optimisation).
* Rafailov et al., 2023 (DPO); Azar et al., 2023 (IPO); Ethayarajh et al., 2024 (KTO); Meng et al., 2024 (SimPO).
* Ahmadian et al., 2024 (RLOO, "Back to basics"); Hu, 2025 (REINFORCE++).
* Shao et al., 2024 (GRPO, DeepSeekMath); DeepSeek-AI, 2025 (DeepSeek-R1).
* Liu et al., 2025 (Dr. GRPO); Yu et al., 2025 (DAPO); Yue et al., 2025 (VAPO); Zheng et al., 2025 (GSPO); MiniMax, 2025 (CISPO); Khatri et al., 2025 (ScaleRL); Liu et al., 2025 (Lite PPO); Yue et al., 2025, "Does RL really incentivize reasoning capacity beyond the base model?".

## Where to go next (project ideas)

1. Add prioritised replay and dueling heads to the DQN and measure the gain on CartPole with 3 seeds.
2. Give the CartPole PPO a *continuous* action version (Gaussian policy) on a NumPy pendulum.
3. Replace the arithmetic verifier with a **learned judge** and measure how fast each GRPO variant hacks it.
4. Make the calculator tool *unreliable* (wrong 20% of the time) and see whether the policy learns to double-check.
5. Port the GRPO variants to a real small model (e.g. a 0.5B parameter instruction model with HuggingFace TRL) on GSM8K, and see which of the toy-task differences survive.

## Final self-test
"""))

CELLS += quiz_cells("18.1", "Which pair correctly matches an algorithm with the quantity it uses as a baseline?",
    ["PPO-RLHF: the group mean reward", "GRPO: a learned value model", "RLOO: the mean reward of the other K-1 samples for the same prompt", "REINFORCE: the KL to the reference"], "C",
    "PPO-RLHF uses a value LM, GRPO the group mean (divided by std), RLOO the leave-one-out mean, plain REINFORCE none.")

CELLS += quiz_cells("18.2", "You are training a 30B reasoning model with 8k-token responses and rollouts generated asynchronously by a separate inference cluster. Which combination is most in line with 2025-26 practice?",
    ["Tabular Q-learning with a large table",
     "TRPO with conjugate gradient over all 30B parameters",
     "A GRPO-family loss with token-level aggregation, clip-higher or detached clipped weights (CISPO), sequence-level or truncated importance sampling to handle the off-policy rollouts, dynamic sampling, and verifiable rewards",
     "Offline DPO on a fixed dataset"], "C",
    "That list is essentially the union of DAPO, GSPO/CISPO and ScaleRL's recommendations, which is what current open reports converge on.")

CELLS += quiz_cells("18.3", "Across the whole notebook, which single idea appears in the incremental bandit update, TD learning, GAE, PPO's critic, and PPO-RLHF's value model?",
    ["Importance sampling", "New estimate = old estimate + step size * (target - old estimate)", "Clipping", "Group normalisation"], "B",
    "The same error-driven update shape, with different targets: a reward sample, a bootstrapped TD target, a lambda-return, a "
    "shaped token return. Recognising it is most of the way to reading any RL paper.")

CELLS.append(code(r"""
score()
print("Re-run any answer(...) cell to improve your score. Exercises: run solution('<id>') to compare with the reference implementation.")
"""))

CELLS.append(md(r"""
*Built as a self-contained learning notebook. Every algorithm here is a faithful but minimal
implementation; production systems add distributed rollouts, mixed precision, KV caches and a great
deal of engineering, but the objectives are the ones you just implemented.*
"""))
