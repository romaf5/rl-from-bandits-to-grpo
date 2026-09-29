# Companion videos for every Part of the notebook

## Goal

One short ManimCE animation (60–120 s) for each of the notebook's 18 Parts, shown at the top of that Part.
Each video conveys **one core intuition** before the reader meets the code. The style follows the GRPO
advantages prototype in `~/workdir/grpo-visuals`: dark background, 3b1b colour palette, on-screen captions and no voiceover.

**Success criteria**

- 18 videos in `videos/partNN.mp4`, 1080p at 30 fps, each playable inline in **local Jupyter**.
- Every number shown on screen is computed with the notebook's formulas. There are no hand-typed values that could drift.
- No quiz answers appear on screen, since the videos come before the quizzes.
- The embeds survive a rebuild (`assemble.py`) *and* reach the shipped, executed notebook without re-running it.

**Non-goals:** voiceover, Colab/GitHub rendering, and using real training logs (see "Rejected approaches").

## Layout

```
tools/manim_scenes/
  style.py              shared palette, caption helper, bar/axes helpers, title card, roadmap outro
  partNN_<slug>.py      one Scene class `PartNN` per Part; small NumPy maths at the top of the file
  render.py             render one/all parts at 1080p30 -> copies to videos/partNN.mp4
  preview.py            contact sheet (12 frames) per part for visual review -> /tmp or media/
  requirements.txt      manim==0.21.*
  README.md             how to install (apt deps + pip) and render
tests/test_scene_maths.py   checks each part's maths helpers against the notebook's formulas
videos/partNN.mp4       committed outputs (~2–4 MB each)
tools/notebook_builder/common.py   + video(n) -> markdown cell <video src="videos/partNN.mp4" controls width="100%">
tools/notebook_builder/secNN_*.py  + CELLS.append(video(NN)) right after the Part header cell
tools/notebook_builder/insert_videos.py   inserts the same cells into the executed notebook, preserving outputs (idempotent)
```

`media/` (Manim's scratch output) is added to `.gitignore`.

### Units

- **style.py** is the only place that defines colours, fonts, caption position, the title card
  (`title_card(n, title)`), the captions (`caption(scene, text)`, which replaces the previous caption in one fixed slot
  under the title) and the outro (`roadmap_outro(scene, n)`, a strip of 18 dots with Part n lit up for 3 s).
  It also provides the bar and axes helpers taken from the GRPO prototype. Scenes may not hard-code colours.
- **partNN files** import only `style` and `numpy`. The maths lives in plain functions at the top of the file
  (for example `group_adv`, `gae`, `clip_objective`) so tests can import it without rendering.
- **render.py** runs `manim -qh --frame_rate 30` and then copies the result. `python render.py 2 15` renders only those
  parts, and `python render.py all` renders everything.
- **insert_videos.py** finds each `# Part N ·` header cell in the `.ipynb` and inserts the video cell after it, unless one
  is already there. It never touches outputs.

## Visual language

- Colours: correct/positive = GREEN, wrong/negative = RED, mean/baseline = YELLOW, std/uncertainty = BLUE,
  policy = TEAL, reference/old policy = GREY, critic/value = PURPLE.
- Every video is built as: title card, then 2–3 short acts of about 20–40 s each, a one-line takeaway, and the roadmap outro.
- **The "update shape"** `estimate ← estimate + step × (target − estimate)` recurs as the same coloured
  MathTex in Parts 2, 4, 5 and 7, with only the *target* term changing.
- Captions are short, one line and lower case, like the prototype.

## Storyboards

| Part | Core intuition | Key animation |
|---|---|---|
| 1 Foundations | RL is a loop with delayed reward, and γ sets the horizon | agent⇄env loop on the 5×5 GridWorld; a trajectory's discounted return G_t built term by term; γ swept 0 → 0.99 while the value heatmap shifts from short- to far-sighted |
| 2 Bandits | exploration costs reward now and buys information later | 10 arms as reward distributions; Q̂ markers move toward q* via the update shape; greedy locks onto a bad arm, ε-greedy finds the best one, UCB confidence bars shrink |
| 3 Dynamic programming | the Bellman update is a γ-contraction, so values flow out from the goal | value-iteration heatmap filling sweep by sweep, the max-error bar shrinking by about γ per sweep, policy arrows snapping into place; policy iteration done in a few steps |
| 4 MC / TD / SARSA / Q | TD bootstraps (less variance, some bias); on- and off-policy learners take different paths | split screen: MC update at episode end vs TD update each step (δ_t highlighted); CliffWalk with Q-learning hugging the edge vs SARSA's safe path |
| 5 DQN | the deadly triad destabilises learning; replay and a target network stabilise it | Q estimate chasing a target that runs away, then the frozen target network; replay buffer shuffling correlated transitions into random minibatches |
| 6 REINFORCE | push up the probability of sampled actions in proportion to their return | policy as probability bars; a sampled action's bar grows with its return; gradient-estimate arrows scattered around the true gradient, and the scatter shrinking with reward-to-go and a baseline |
| 7 Actor-critic + GAE | a baseline keeps credit only for beating expectations; λ trades bias for variance | bars shifted by V(s) (the GRPO move, now with a learned critic); GAE weights (γλ)^l as decaying bars over δ's, with a λ slider from 0 (TD) to 1 (MC) |
| 8 TRPO | measure step size in policy space (KL), not in parameter space | two parameter steps of equal length giving very different policy changes; KL trust-region ellipse vs Euclidean circle; natural-gradient arrow bending to the metric |
| 9 PPO | clipping removes the incentive to move the ratio past 1±ε | L^CLIP vs ρ drawn as min(ρA, clip·A) for A>0 and A<0; a ball rolling up the objective and stopping at the flat part; gradient shown as zero there |
| 10 LLMs as RL | tokens are actions, prefixes are states, and the reward is one number at the end | CartPole's RL table morphing into the LLM version; token-by-token generation tree; a single terminal reward lighting up at EOS |
| 11 Reward models | preferences give a reward via Bradley–Terry, and optimising a proxy runs into Goodhart | sigmoid of the reward difference with a pair sliding along it; best-of-n curve where proxy reward keeps rising while true reward peaks and falls (length bias) |
| 12 RLHF with PPO | four models, and a KL penalty paid per token | diagram of policy / ref / reward / value models; token strip with small −β·KL rewards on every token and R on the last one; GAE sweeping backwards over tokens |
| 13 DPO | the optimal policy's form lets Z(x) cancel, leaving a loss on preference pairs | four-line derivation with Z(x) struck out; implicit reward β log π/π_ref; chosen log-prob rising and rejected one falling as the margin moves along the sigmoid |
| 14 RLOO / REINFORCE++ | the other samples are a free baseline | K samples, and for each one the *others'* rewards highlighted and averaged; contrast with REINFORCE++'s batch-wide normalisation |
| 15 GRPO | group-normalised advantages act as a difficulty weighting | port of the existing prototype onto `style.py` |
| 16 After GRPO | every successor is a setting of three dials: advantage, ratio/clip, aggregation | three dials; Dr. GRPO, DAPO, GSPO and CISPO each turn them; per-token aggregation shown making long wrong answers cheaper per token (the rambling bias) |
| 17 Agentic RL | only the model's own tokens get a gradient | multi-turn trajectory ribbon (model turns vs tool outputs); loss mask sweeping over it with tool tokens greyed out; what goes wrong without the mask (the model learns to predict tool output) |
| 18 The map | all 17 parts are one family tree | the algorithm graph builds up node by node with labelled edges ("+critic", "−critic", "+clip", "group baseline", …), ending on the full map |

Numbers, names and hyperparameters shown on screen come from the matching Part of the notebook. For example, Part 15 uses
G = 8, and Part 9 uses the notebook's ε.

## Testing and review

- `tests/test_scene_maths.py` checks each maths helper against a value the notebook states or computes, for example
  GRPO's +2.47 for 1 correct answer out of 8, the GAE recursion against a direct sum, or PPO clip gradients being zero past 1+ε. It runs with pytest.
- Every scene must render at `-ql` without errors, and `preview.py` makes a contact sheet that I inspect for
  overlaps, clipping, leftover mobjects and captions going off-screen. This inspection caught five layout bugs in the prototype.
- The final 1080p30 render is spot-checked on one frame per act.

## Delivery plan

1. **Foundation:** `style.py`, `render.py`, `preview.py`, the tests scaffold, the builder `video()` helper and
   `insert_videos.py`, plus the reference pair **Part 2 and Part 15**. Then you review the look.
2. **Batches** of about 4 parts, built by parallel subagents that each follow `style.py` and this storyboard. I review
   every contact sheet before accepting a part.
3. Insert the videos into the notebook, update the README with the install and render instructions, and commit.

## Rejected approaches

- **Data from real notebook runs:** every video would depend on a full 10–15 minute run and its seeds, and most of the
  intuitions are conceptual. This can be revisited for a single part (for example Part 5) if its synthetic data feels flat.
- **Rebuilding the notebook to add embeds:** `assemble.py` drops outputs, and re-running takes 10–15 minutes. `insert_videos.py`
  patches the executed notebook instead, and the section files are updated so future builds match.
