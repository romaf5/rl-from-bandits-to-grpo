from common import md, code, quiz_cells, exercise_cells

CELLS = []

CELLS.append(md(r"""
# Reinforcement Learning, from Bandits to GRPO

**A hands-on, build-it-yourself course in one notebook.** You will implement every
algorithm from scratch in NumPy and PyTorch, watch it learn, and test yourself along the way.

| Part | What you build | Environment |
|---|---|---|
| 1. Foundations | MDPs, returns, value functions, Bellman equations | 5×5 GridWorld |
| 2. Bandits | ε-greedy, UCB, optimistic init | 10-armed bandit |
| 3. Dynamic programming | Policy evaluation, policy iteration, value iteration | GridWorld |
| 4. Model-free tabular | Monte Carlo, TD(0), SARSA, Q-learning, Double Q | GridWorld, CliffWalk |
| 5. Deep Q-learning | DQN (replay, target net), Double DQN | CartPole (NumPy) |
| 6. Policy gradients | REINFORCE, reward-to-go, gradient variance | CartPole |
| 7. Actor-critic | Baselines, advantages, GAE | CartPole |
| 8. TRPO | Natural gradient, conjugate gradient, line search | CartPole |
| 9. PPO | Clipped objective, all the diagnostics | CartPole |
| 10. RL for language models | A tiny transformer LM, SFT, pass@k | Arithmetic task |
| 11. Reward models | Bradley–Terry, best-of-n, Goodhart | Arithmetic task |
| 12. RLHF with PPO | Token-level KL penalty, value head | Arithmetic task |
| 13. DPO | Direct preference optimization, IPO/KTO/SimPO | Arithmetic task |
| 14. Critic-free RL | RLOO, REINFORCE++ | Arithmetic task |
| 15. GRPO | Group-relative advantages, the DeepSeek recipe | Arithmetic task |
| 16. After GRPO (2025–26) | Dr. GRPO, DAPO, GSPO, CISPO, and friends | Arithmetic task |
| 17. Agentic RL | Multi-turn, tool calls, loss masking | Arithmetic + calculator tool |
| 18. Wrap-up | Comparison table, decision guide, reading list | |

### How to use this notebook

* **Run it top to bottom.** Every section reuses helpers from earlier sections. `Runtime → Run all`
  (Colab) or `Kernel → Restart & Run All` (Jupyter) takes about 10–15 minutes on a laptop CPU.
* **Read the code.** Each algorithm is built in small cells with comments. The prose explains *why*,
  the code shows *how*.
* **Test your knowledge.** Cells marked `quiz(...)` ask a question; type your answer into the
  `answer(...)` cell below it. Cells marked `exercise(...)` ask you to implement something;
  `check(...)` runs hidden tests and `solution(...)` reveals a reference implementation.
* **Tinker.** Every experiment has hyperparameters at the top of the cell. Change them and re-run.

### Requirements

`numpy`, `matplotlib`, `torch` (CPU is fine). Google Colab has all three preinstalled.
On a Mac: `pip install numpy matplotlib torch` in a fresh virtualenv, then `pip install jupyterlab`.
"""))

CELLS.append(code(r"""
# If a dependency is missing (e.g. a fresh Mac environment), install it.
import importlib, subprocess, sys
for pkg in ["numpy", "matplotlib", "torch"]:
    if importlib.util.find_spec(pkg) is None:
        print(f"installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("dependencies OK")
"""))

CELLS.append(code(r"""
import math, time, random, copy, base64, hashlib, textwrap
from dataclasses import dataclass, field
from collections import deque, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
DEVICE = torch.device("cpu")          # everything here is small enough for CPU
print("torch", torch.__version__, "| numpy", np.__version__)

def seed_everything(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

seed_everything(0)
"""))

CELLS.append(code(r"""
# ---- Plot style: one consistent look for every figure in the notebook ----
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.prop_cycle": mpl.cycler(color=PALETTE),
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.fontsize": 9,
    "figure.dpi": 100, "font.size": 10, "font.family": "sans-serif",
})

def smooth(x, k=10):
    '''Trailing moving average that keeps the array length (early values use a shorter window).'''
    x = np.asarray(x, dtype=float)
    if k <= 1 or len(x) == 0:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - k)
    return (c[idx] - c[lo]) / (idx - lo)

def plot_runs(ax, runs, label=None, color=None, k=1, x=None):
    '''Plot the mean of several runs with a ±1 std band. `runs`: list of 1-D arrays (one per seed).'''
    n = min(len(r) for r in runs)
    arr = np.stack([smooth(np.asarray(r[:n], dtype=float), k) for r in runs])
    mu, sd = arr.mean(0), arr.std(0)
    xs = np.arange(n) if x is None else np.asarray(x[:n])
    line, = ax.plot(xs, mu, label=label, color=color)
    if len(runs) > 1:
        ax.fill_between(xs, mu - sd, mu + sd, color=line.get_color(), alpha=0.12, linewidth=0)
    return line

def finish(ax, title=None, xlabel=None, ylabel=None, legend=True):
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend()

class Timer:
    def __init__(self, name=""): self.name = name
    def __enter__(self): self.t = time.time(); return self
    def __exit__(self, *a): print(f"[{self.name}] {time.time() - self.t:.1f}s")

print("plot helpers ready")
"""))

CELLS.append(code(r"""
# ---- Quiz & exercise framework (answers are hashed so you can't peek by accident) ----
_QUIZ, _EX, _SCORE = {}, {}, {"correct": 0, "attempted": 0}

def _dec(b64): return base64.b64decode(b64.encode()).decode()

def register_quiz(qid, question, options, answer_hash, explanation_b64):
    _QUIZ[qid] = dict(q=question, opts=options, h=answer_hash, e=explanation_b64, solved=False)

def quiz(qid):
    q = _QUIZ[qid]
    print(f"QUIZ {qid}: {q['q']}\n")
    for letter, opt in zip("ABCD", q["opts"]):
        print(f"   {letter})  {opt}")
    print(f"\n-> put your choice in the answer('{qid}', ...) cell below and run it.")

def answer(qid, choice):
    q = _QUIZ[qid]
    choice = str(choice).strip().upper()
    if choice not in "ABCD" or len(choice) != 1:
        print("Type a single letter: A, B, C or D."); return
    _SCORE["attempted"] += 1
    ok = hashlib.sha256(f"{qid}:{choice}".encode()).hexdigest() == q["h"]
    if ok:
        if not q["solved"]:
            _SCORE["correct"] += 1
        q["solved"] = True
        print(f"Correct!  {_dec(q['e'])}")
    else:
        print(f"Not quite ({choice}). Think about it once more, then re-run. "
              f"Hint is in the explanation once you get it right.")

def score():
    print(f"quiz score: {_SCORE['correct']} solved / {len(_QUIZ)} registered "
          f"({_SCORE['attempted']} attempts)")

def register_exercise(eid, solution_b64):
    _EX[eid] = dict(sol=solution_b64)

def check(eid, fn, tests):
    try:
        if fn is None:
            raise NotImplementedError
        tests(fn)
    except NotImplementedError:
        print(f"[{eid}] not implemented yet. Fill in the cell above (or run solution('{eid}')).")
    except AssertionError as e:
        print(f"[{eid}] FAIL: {e}")
    except Exception as e:
        print(f"[{eid}] ERROR: {type(e).__name__}: {e}")
    else:
        print(f"[{eid}] PASS - all tests passed.")

def solution(eid):
    print(_dec(_EX[eid]["sol"]))

print("quiz framework ready - questions are checked with answer('<id>', 'A'/'B'/'C'/'D')")
"""))
