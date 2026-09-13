# Reinforcement Learning, from Bandits to GRPO

A single self-contained Jupyter notebook that teaches the main RL algorithms by building each one
from scratch in NumPy and PyTorch, running it, plotting what happens, and quizzing you on it.

**Notebook:** [`rl_from_bandits_to_grpo.ipynb`](rl_from_bandits_to_grpo.ipynb)

| Part | Topic | Environment |
|---|---|---|
| 1–4 | MDPs, bandits, dynamic programming, MC/TD, SARSA, Q-learning, Double Q | GridWorld, CliffWalk |
| 5 | DQN and Double DQN | CartPole (pure NumPy) |
| 6–9 | REINFORCE, actor-critic + GAE, TRPO, PPO | CartPole |
| 10–13 | A tiny transformer LM, SFT, pass@k, reward models, PPO-RLHF, DPO | arithmetic task |
| 14–16 | RLOO, REINFORCE++, GRPO, Dr. GRPO, DAPO, GSPO, CISPO | arithmetic task |
| 17 | Agentic RL: multi-turn tool calls with loss masking | arithmetic + calculator tool |
| 18 | Comparison table, decision guide, reading list, final quiz | |

Every part has **quiz cells** (`quiz(...)` / `answer(...)`) and **coding exercises** with hidden tests
(`check(...)`) and reference solutions (`solution(...)`).

## Running it

**Google Colab / Drive:** upload the `.ipynb`, open it with Colab, `Runtime → Run all`.
Everything needed (`numpy`, `matplotlib`, `torch`) is preinstalled there.

**Mac laptop (CPU is fine):**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib torch jupyterlab
jupyter lab rl_from_bandits_to_grpo.ipynb
```

A full top-to-bottom run takes roughly 10–15 minutes on a laptop CPU. The heaviest cells have a
`N_SEEDS` or `steps` setting at the top you can lower.

The notebook ships with outputs already executed, so you can also read it like a book first.

## Regenerating the notebook

`tools/notebook_builder/` holds the per-section source files the notebook is assembled from (see its README;
it contains the quiz answer keys, so avoid it if you want to take the quizzes cold).
