# Notebook Companion Videos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One 60–120 s ManimCE companion video per notebook Part (18 total), embedded at the top of each Part for local Jupyter.

**Architecture:** A shared `style.py` (palette, captions, bars, title card, roadmap outro) plus one `partNN_<slug>.py` per Part.
Each Part file begins with pure-NumPy maths helpers that are unit-tested against the notebook's formulas, followed by one `PartNN(Scene)`.
`render.py` renders at 1080p30 into `videos/`. `preview.py` builds contact sheets for visual review.
`insert_videos.py` patches the executed notebook, and the builder's `video()` helper keeps rebuilds in sync.

**Tech Stack:** Python 3.12, ManimCE 0.21, NumPy, pytest, nbformat, ffmpeg (system), TeX Live (system).

**Spec:** `docs/superpowers/specs/2026-09-27-notebook-videos-design.md`

## Global Constraints

- Repo root: `~/workdir/rl-from-bandits-to-grpo`, branch `notebook-videos`. All paths below are relative to the repo root.
- Python env: `.venv/` at the repo root (already git-ignored), created in Task 1. Always call `.venv/bin/python`, `.venv/bin/manim` and `.venv/bin/pytest`.
- System deps are already installed on this machine: `libcairo2-dev libpango1.0-dev pkg-config ffmpeg texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvisvgm cm-super`.
- Output: `videos/partNN.mp4` (two-digit NN), 1080p, **30 fps**, 60–120 s long.
- Scene class name `PartNN` (e.g. `Part02`) in `tools/manim_scenes/partNN_<slug>.py`. Slugs mirror the builder: 01 foundations, 02 bandits, 03 dp, 04 tabular, 05 dqn, 06 reinforce, 07 actor_critic, 08 trpo, 09 ppo, 10 llm, 11 reward_model, 12 ppo_llm, 13 dpo, 14 rloo, 15 grpo, 16 beyond_grpo, 17 agentic, 18 wrapup.
- Part files import only `numpy`, `manim` and `style` (Part 03 may also import Part 01's gridworld helpers). Scenes never hard-code colours: use the `style` constants.
- Every video: `title_card`, then 2–3 acts, a `takeaway`, and the `roadmap_outro`.
- Captions: one line, lower case, and at most 70 characters (the `Captioner` scales down anything wider than the frame, but aim short).
- **No quiz answers on screen.** Do not open `quiz_cells(...)` or `exercise_cells(...)` blocks in `tools/notebook_builder/sec*.py` for content. Use only the Part's markdown text and its code cells.
- Every on-screen number comes from a maths helper in the Part file, never typed by hand.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Git identity is already set repo-locally.

## Review Focus

- **Leftover mobjects between acts** (the prototype's worst bug): `LaggedStart` adds children individually, so removing the parent leaves them on screen. `clear_act` must remove everything except `keep`. It is tested in Task 1.
- **Text running off-screen** at 1080p: long captions or labels. `Captioner` shrinks anything wider than 13 units. Tested in Task 1, and contact sheets are checked in every Part task.
- **`insert_videos.py` run twice**, or run after a rebuild that already contains the cells: it must not duplicate cells. Tested in Task 2.
- **A Part without a rendered video yet**: `insert_videos.py` must skip it with a message, not crash or embed a broken `<video>`. Tested in Task 2.
- **Executed outputs must survive the insertion**: the output count before and after must be equal. Tested in Task 2.

---

### Task 1: Tooling foundation (env, style, render, preview)

**Files:**
- Create: `tools/manim_scenes/requirements.txt`, `tools/manim_scenes/style.py`, `tools/manim_scenes/render.py`, `tools/manim_scenes/preview.py`, `tools/manim_scenes/README.md`, `tests/conftest.py`, `tests/test_style.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces (used by every Part task), all in `style.py`:
  - Colours: `CORRECT, WRONG, MEAN_C, STD_C, POLICY_C, REF_C, VALUE_C, MUTED`
  - `PART_TITLES: list[str]` (18 short names, index 0 = Part 1)
  - `title_card(scene, n: int, title: str) -> Text`: animates "Part n" plus the title, which settles at the top. Returns the title mobject.
  - `class Captioner(scene, anchor: Mobject)` with `.say(text: str, color=MUTED, wait: float = 0) -> Text` and `.clear()`
  - `clear_act(scene, keep: list[Mobject] = ()) -> None`
  - `bar(ax: Axes, x: float, v: float, color, width=0.64, outline=False) -> Polygon`
  - `value_label(ax: Axes, x: float, v: float, font_size=26) -> DecimalNumber`
  - `update_shape(target_tex: str, target_color) -> MathTex`, which renders `Q ← Q + α(target − Q)`
  - `takeaway(scene, text: str, keep=()) -> None`
  - `roadmap_outro(scene, n: int) -> None`
- `render.py` CLI: `python tools/manim_scenes/render.py [--preview] 2 15 | all`
- `preview.py` CLI: `python tools/manim_scenes/preview.py 2 [--hq]`, which writes `media/previews/partNN.png` and prints the duration

- [ ] **Step 1: Create the venv and requirements**

`tools/manim_scenes/requirements.txt`:
```
manim==0.21.*
numpy
pytest
nbformat
```
Run:
```bash
cd ~/workdir/rl-from-bandits-to-grpo && python3 -m venv .venv && .venv/bin/pip install -q -r tools/manim_scenes/requirements.txt && .venv/bin/manim --version
```
Expected: `Manim Community v0.21.x`

Append to `.gitignore`:
```
media/
```

- [ ] **Step 2: Write the failing style tests**

`tests/conftest.py`:
```python
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "manim_scenes"))
sys.path.insert(0, str(ROOT / "tools" / "notebook_builder"))
```

`tests/test_style.py`:
```python
from manim import Scene, Square, Circle, VGroup, FadeIn, LaggedStart, config
import style


def _scene():
    config.dry_run = True          # no files written, animations are not rendered
    return Scene()


def test_part_titles_has_18_entries():
    assert len(style.PART_TITLES) == 18


def test_clear_act_removes_lagged_children_but_keeps_keep():
    s = _scene()
    keep = Circle()
    squares = VGroup(*[Square() for _ in range(4)])
    s.add(keep)
    s.play(LaggedStart(*[FadeIn(q) for q in squares]))   # adds children individually
    style.clear_act(s, keep=[keep])
    assert s.mobjects == [keep]


def test_captioner_shrinks_long_text_to_frame():
    s = _scene()
    anchor = Square().to_edge(style.UP)
    cap = style.Captioner(s, anchor)
    t = cap.say("x" * 300)
    assert t.width <= 13.0 + 1e-6


def test_captioner_replaces_previous_caption():
    s = _scene()
    cap = style.Captioner(s, Square())
    cap.say("first"); t2 = cap.say("second")
    texts = [m for m in s.mobjects if m is t2 or getattr(m, "text", None) == "first"]
    assert texts == [t2]
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `.venv/bin/pytest tests/test_style.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'style'`

- [ ] **Step 4: Implement `style.py`**

```python
"""Shared look for every Part video. Scenes import constants and helpers from here only."""
import numpy as np
from manim import *

CORRECT, WRONG = GREEN_C, RED_C
MEAN_C, STD_C = YELLOW, BLUE_C
POLICY_C, REF_C, VALUE_C = TEAL_C, GREY_B, PURPLE_B
MUTED = GREY_A
MAX_TEXT_WIDTH = 13.0

PART_TITLES = [
    "Foundations", "Bandits", "Dynamic programming", "MC / TD / Q-learning", "DQN", "REINFORCE",
    "Actor-critic + GAE", "TRPO", "PPO", "LLMs as RL", "Reward models", "RLHF with PPO", "DPO",
    "RLOO / REINFORCE++", "GRPO", "After GRPO", "Agentic RL", "The map",
]


def fit(mob, max_width=MAX_TEXT_WIDTH):
    if mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def title_card(scene, n, title):
    kicker = Text(f"Part {n}", font_size=30, color=MUTED)
    head = fit(Text(title, font_size=56))
    VGroup(kicker, head).arrange(DOWN, buff=0.3)
    scene.play(FadeIn(kicker, shift=0.2 * UP), Write(head))
    scene.wait(0.8)
    top = fit(Text(f"{n} · {title}", font_size=40)).to_edge(UP)
    scene.play(FadeOut(kicker), ReplacementTransform(head, top))
    return top


class Captioner:
    """One caption slot under `anchor`; each say() replaces the previous caption."""

    def __init__(self, scene, anchor):
        self.scene, self.anchor, self.current = scene, anchor, None

    def say(self, text, color=MUTED, wait=0):
        new = fit(Text(text, font_size=26, color=color)).next_to(self.anchor, DOWN, buff=0.25)
        anims = [FadeIn(new)]
        if self.current is not None:
            anims.append(FadeOut(self.current))
        self.scene.play(*anims)
        if self.current is not None:
            self.scene.remove(self.current)
        self.current = new
        if wait:
            self.scene.wait(wait)
        return new

    def clear(self):
        if self.current is not None:
            self.scene.play(FadeOut(self.current))
            self.scene.remove(self.current)
            self.current = None


def clear_act(scene, keep=()):
    """Fade out and fully remove everything except `keep` (children added by LaggedStart included)."""
    keep_ids = {id(m) for k in keep for m in k.get_family()}
    doomed = [m for m in scene.mobjects if id(m) not in keep_ids]
    if doomed:
        for m in doomed:
            m.clear_updaters()
        scene.play(*[FadeOut(m) for m in doomed])
    scene.remove(*doomed)
    scene.mobjects = [m for m in scene.mobjects if id(m) in keep_ids]


def bar(ax, x, v, color, width=0.64, outline=False):
    v = v if abs(v) > 1e-3 else 1e-3
    x0, x1 = x - width / 2, x + width / 2
    poly = Polygon(ax.c2p(x0, 0), ax.c2p(x1, 0), ax.c2p(x1, v), ax.c2p(x0, v))
    if outline:
        return poly.set_stroke(WHITE, 2, opacity=0.8).set_fill(opacity=0)
    return poly.set_stroke(color, 2).set_fill(color, 0.75)


def value_label(ax, x, v, font_size=26):
    lab = DecimalNumber(v, num_decimal_places=2, include_sign=True, font_size=font_size)
    return lab.next_to(ax.c2p(x, v), UP if v >= 0 else DOWN, buff=0.1)


def update_shape(target_tex, target_color=YELLOW):
    m = MathTex(r"Q", r"\leftarrow", r"Q", "+", r"\alpha", r"\big(", target_tex, "-", r"Q", r"\big)", font_size=44)
    m[6].set_color(target_color); m[4].set_color(STD_C)
    return m


def takeaway(scene, text, keep=()):
    clear_act(scene, keep=keep)
    t = fit(Text(text, font_size=34))
    scene.play(FadeIn(t, shift=0.2 * UP))
    scene.wait(2.5)
    scene.play(FadeOut(t))
    scene.remove(t)


def roadmap_outro(scene, n):
    clear_act(scene)
    dots = VGroup(*[Dot(radius=0.12, color=GREY_D) for _ in PART_TITLES]).arrange(RIGHT, buff=0.45)
    line = Line(dots[0].get_center(), dots[-1].get_center(), color=GREY_D, stroke_width=2)
    dots[n - 1].set_color(YELLOW).scale(1.6)
    for i in range(n - 1):
        dots[i].set_color(GREY_B)
    here = Text(PART_TITLES[n - 1], font_size=28, color=YELLOW).next_to(dots[n - 1], UP, buff=0.3)
    nxt = (Text(f"next: {PART_TITLES[n]}", font_size=22, color=MUTED).next_to(dots[n], DOWN, buff=0.35)
           if n < len(PART_TITLES) else VMobject())
    scene.play(Create(line), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.03), run_time=1)
    scene.play(FadeIn(here), FadeIn(nxt))
    scene.wait(1.5)
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `.venv/bin/pytest tests/test_style.py -q`
Expected: `4 passed`. If `test_captioner_replaces_previous_caption` fails because Manim's `Text` has no `.text` attribute, change the check to count `Text` instances in `s.mobjects` (there should be exactly 1).

- [ ] **Step 6: Write `render.py`**

```python
"""Render Part videos. Usage: render.py [--preview] 2 15 | all"""
import pathlib, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MEDIA, OUT = ROOT / "media", ROOT / "videos"
MANIM = ROOT / ".venv" / "bin" / "manim"


def part_files():
    return {int(p.name[4:6]): p for p in sorted(HERE.glob("part[0-9][0-9]_*.py"))}


def render(n, preview=False):
    f = part_files().get(n)
    if f is None:
        sys.exit(f"no scene file for part {n}")
    cls = f"Part{n:02d}"
    q = ["-ql"] if preview else ["-qh", "--frame_rate", "30"]
    subprocess.run([str(MANIM), *q, "--media_dir", str(MEDIA), str(f), cls], check=True, cwd=HERE)
    sub = "480p15" if preview else "1080p30"
    mp4 = MEDIA / "videos" / f.stem / sub / f"{cls}.mp4"
    if not preview:
        OUT.mkdir(exist_ok=True)
        shutil.copy(mp4, OUT / f"part{n:02d}.mp4")
        print("wrote", OUT / f"part{n:02d}.mp4")
    return mp4


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--preview"]
    nums = sorted(part_files()) if args == ["all"] else [int(a) for a in args]
    for n in nums:
        render(n, preview="--preview" in sys.argv)
```

- [ ] **Step 7: Write `preview.py`**

```python
"""Contact sheet of 12 frames for one Part: preview.py N [--hq] -> media/previews/partNN.png"""
import pathlib, subprocess, sys, tempfile
from render import MEDIA, part_files


def duration(mp4):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def sheet(n, hq=False):
    f = part_files()[n]
    mp4 = MEDIA / "videos" / f.stem / ("1080p30" if hq else "480p15") / f"Part{n:02d}.mp4"
    d = duration(mp4)
    tmp = pathlib.Path(tempfile.mkdtemp())
    for i in range(12):
        t = d * (i + 0.5) / 12
        subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(mp4), "-frames:v", "1",
                        str(tmp / f"{i}.png")], check=True)
    ins = sum([["-i", str(tmp / f"{i}.png")] for i in range(12)], [])
    rows = ";".join(f"[{3*r}][{3*r+1}][{3*r+2}]hstack=3[r{r}]" for r in range(4))
    out = MEDIA / "previews" / f"part{n:02d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *ins, "-filter_complex",
                    rows + ";[r0][r1][r2][r3]vstack=4", str(out)], check=True)
    flag = "" if 55 <= d <= 130 else "  <-- OUTSIDE 60-120 s TARGET"
    print(f"part {n}: {d:.1f} s{flag}\n{out}")
    return out


if __name__ == "__main__":
    sheet(int(sys.argv[1]), hq="--hq" in sys.argv)
```

- [ ] **Step 8: Write `tools/manim_scenes/README.md`**

```markdown
# Companion videos

One ManimCE scene per notebook Part. Outputs go to `videos/partNN.mp4` (1080p30).

## Install (Ubuntu / WSL)

    sudo apt-get install libcairo2-dev libpango1.0-dev pkg-config ffmpeg \
        texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvisvgm cm-super
    python3 -m venv .venv && .venv/bin/pip install -r tools/manim_scenes/requirements.txt

## Render

    .venv/bin/python tools/manim_scenes/render.py --preview 9   # fast 480p check
    .venv/bin/python tools/manim_scenes/preview.py 9            # contact sheet -> media/previews/part09.png
    .venv/bin/python tools/manim_scenes/render.py 9             # final -> videos/part09.mp4
    .venv/bin/python tools/manim_scenes/render.py all
    .venv/bin/python tools/notebook_builder/insert_videos.py rl_from_bandits_to_grpo.ipynb

`style.py` holds the palette and helpers. Each `partNN_*.py` starts with its maths helpers,
which `tests/test_scene_maths.py` checks against the notebook's formulas.
```

- [ ] **Step 9: Commit**

```bash
git add .gitignore tools/manim_scenes tests
git commit -m "Add Manim tooling: shared style, render and preview scripts

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Notebook embedding (builder helper + insert_videos.py)

**Files:**
- Modify: `tools/notebook_builder/common.py` (add `video`)
- Create: `tools/notebook_builder/insert_videos.py`, `tests/test_insert_videos.py`
- Modify (Task 21, once the videos exist): `tools/notebook_builder/sec1_foundations.py` … `sec18_wrapup.py`

**Interfaces:**
- Produces: `common.video(n: int) -> NotebookNode` (a markdown cell whose metadata has `{"companion_video": n}`)
- Produces: `insert_videos.insert(nb_path: str, videos_dir: str) -> int` (returns the number of cells inserted). CLI: `insert_videos.py NOTEBOOK.ipynb`

- [ ] **Step 1: Write the failing tests**

`tests/test_insert_videos.py`:
```python
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output
from common import video
import insert_videos


def _nb(tmp_path, parts=(1, 2)):
    nb = new_notebook()
    for p in parts:
        nb.cells.append(new_markdown_cell(f"---\n# Part {p} · Title {p}\n\ntext"))
        c = new_code_cell("print(1)"); c.outputs = [new_output("stream", name="stdout", text="1\n")]
        nb.cells.append(c)
    path = tmp_path / "nb.ipynb"; nbformat.write(nb, path)
    return path


def _videos(tmp_path, parts):
    d = tmp_path / "videos"; d.mkdir()
    for p in parts:
        (d / f"part{p:02d}.mp4").write_bytes(b"x")
    return d


def test_video_cell_points_at_relative_path_and_is_tagged():
    c = video(3)
    assert 'src="videos/part03.mp4"' in c.source and c.metadata["companion_video"] == 3


def test_inserts_after_part_header_and_keeps_outputs(tmp_path):
    path = _nb(tmp_path); vids = _videos(tmp_path, [1, 2])
    assert insert_videos.insert(str(path), str(vids)) == 2
    nb = nbformat.read(path, as_version=4)
    assert nb.cells[1].metadata["companion_video"] == 1
    assert nb.cells[4].metadata["companion_video"] == 2
    assert sum(len(c.get("outputs", [])) for c in nb.cells) == 2


def test_idempotent(tmp_path):
    path = _nb(tmp_path); vids = _videos(tmp_path, [1, 2])
    insert_videos.insert(str(path), str(vids))
    assert insert_videos.insert(str(path), str(vids)) == 0
    assert len(nbformat.read(path, as_version=4).cells) == 6


def test_skips_parts_without_video(tmp_path, capsys):
    path = _nb(tmp_path); vids = _videos(tmp_path, [2])
    assert insert_videos.insert(str(path), str(vids)) == 1
    assert "part 1: no video" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `.venv/bin/pytest tests/test_insert_videos.py -q`
Expected: FAIL with `ImportError: cannot import name 'video' from 'common'`

- [ ] **Step 3: Implement**

Append to `tools/notebook_builder/common.py`:
```python
def video(n):
    """Companion animation for Part n; the notebook lives next to videos/ so the path is relative."""
    cell = md(f'<video src="videos/part{n:02d}.mp4" controls width="100%"></video>\n\n'
              f'*Companion animation for Part {n}. Rendered with Manim from `tools/manim_scenes/`.*')
    cell.metadata["companion_video"] = n
    return cell
```

`tools/notebook_builder/insert_videos.py`:
```python
"""Insert companion-video cells into an (executed) notebook without touching outputs.
Usage: insert_videos.py NOTEBOOK.ipynb     (videos are looked up in <notebook dir>/videos)"""
import pathlib, re, sys
import nbformat
from common import video

HEADER = re.compile(r"^---\s*\n#\s*Part\s+(\d+)\s*·", re.M)


def insert(nb_path, videos_dir):
    nb = nbformat.read(nb_path, as_version=4)
    have = {c.metadata.get("companion_video") for c in nb.cells}
    out, added = [], 0
    for c in nb.cells:
        out.append(c)
        m = HEADER.match(c.source) if c.cell_type == "markdown" else None
        if not m:
            continue
        n = int(m.group(1))
        if n in have:
            continue
        if not (pathlib.Path(videos_dir) / f"part{n:02d}.mp4").exists():
            print(f"part {n}: no video, skipped"); continue
        out.append(video(n)); added += 1
    nb.cells = out
    nbformat.write(nb, nb_path)
    print(f"inserted {added} video cells into {nb_path}")
    return added


if __name__ == "__main__":
    p = pathlib.Path(sys.argv[1])
    insert(str(p), str(p.parent / "videos"))
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `.venv/bin/pytest tests/test_insert_videos.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add tools/notebook_builder/common.py tools/notebook_builder/insert_videos.py tests/test_insert_videos.py
git commit -m "Add video() cell helper and idempotent insert_videos.py

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## The Part task template (Tasks 3–20)

Every Part task has the same five steps. Each task below gives the **maths helpers** (exact code), the **tests** (exact code, appended to `tests/test_scene_maths.py`) and the **storyboard** (acts, the key animation and captions). Scene code is written against `style.py` by following the storyboard.

- [ ] **Step 1:** Append the task's tests to `tests/test_scene_maths.py`. Run `.venv/bin/pytest tests/test_scene_maths.py -q -k partNN` and confirm FAIL (`ModuleNotFoundError`).
- [ ] **Step 2:** Create `tools/manim_scenes/partNN_<slug>.py` with the maths helpers verbatim at the top, then re-run the tests and confirm PASS.
- [ ] **Step 3:** Write `class PartNN(Scene)` below the helpers, following the storyboard. The skeleton is:
  ```python
  class PartNN(Scene):
      def construct(self):
          self.title = title_card(self, NN, "<title from PART_TITLES or longer>")
          self.cap = Captioner(self, self.title)
          self.act1(); clear_act(self, keep=[self.title])
          self.act2(); clear_act(self, keep=[self.title])
          self.act3()                                   # optional
          takeaway(self, "<takeaway line>")
          roadmap_outro(self, NN)
  ```
- [ ] **Step 4:** Run `.venv/bin/python tools/manim_scenes/render.py --preview NN && .venv/bin/python tools/manim_scenes/preview.py NN`, then **open `media/previews/partNN.png` and look at it**. Fix any overlaps, text off-screen, leftover mobjects, unreadably small labels or empty frames, and re-render until it's clean. The duration must be 60–120 s (the script flags it otherwise).
- [ ] **Step 5:** Run `.venv/bin/python tools/manim_scenes/render.py NN` (writes `videos/partNN.mp4`), then commit:
  ```bash
  git add tools/manim_scenes/partNN_*.py tests/test_scene_maths.py videos/partNN.mp4
  git commit -m "Add Part NN companion video: <topic>

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

`tests/test_scene_maths.py` starts with:
```python
import numpy as np
import pytest
```
Each task appends its own `import partNN_<slug> as pNN` and its tests, and names every test `test_partNN_*`.

---

### Task 3: Part 15 · GRPO (reference video: port of the prototype)

**Source:** port `~/workdir/grpo-visuals/grpo_advantages.py` onto `style.py`: use `Captioner` instead of hand-placed notes, `bar`/`value_label` from `style`, `clear_act` between acts, `title_card` and `roadmap_outro`. Keep all three acts and the same numbers.

**Maths** (`part15_grpo.py`):
```python
def group_adv(R, normalize=True, eps=1e-4):
    """Notebook's group_advantages for one group (torch .std() is unbiased: ddof=1)."""
    R = np.asarray(R, float)
    a = R - R.mean()
    return a / (R.std(ddof=1) + eps) if normalize else a


def rewards_with(k, G=8):
    return [1.0] * k + [0.0] * (G - k)
```
**Tests:**
```python
import part15_grpo as p15

def test_part15_lone_correct_gets_2_47():
    a = p15.group_adv(p15.rewards_with(1))
    assert a[0] == pytest.approx(2.474, abs=1e-2) and a[-1] == pytest.approx(-0.3535, abs=1e-2)

def test_part15_mean_only_is_0_875():
    assert p15.group_adv(p15.rewards_with(1), normalize=False)[0] == pytest.approx(0.875)

def test_part15_dead_group_is_zero():
    assert np.allclose(p15.group_adv(p15.rewards_with(8)), 0)
```
**Storyboard:** identical to the prototype. Act 1: sample and grade the group. Act 2: subtract the mean, then divide by the std. Act 3: sweep k = 1…8, comparing with the mean-only baseline, ending on the dead group. Takeaway: "the group is its own critic, and ÷std turns reward into a difficulty weighting".

### Task 4: Part 2 · Bandits (reference video) — **CHECKPOINT: after this task, stop and show your human partner both contact sheets and both videos before starting Task 5**

**Maths** (`part02_bandits.py`):
```python
def make_bandit(k=10, seed=3):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(k)                       # q_star ~ N(0, 1)


def run_agent(q_star, kind="eps", eps=0.1, c=2.0, steps=300, seed=0):
    """Sample-average agent. kind in {"greedy", "eps", "ucb"}. Returns dict of per-step histories."""
    rng = np.random.default_rng(seed)
    k = len(q_star); Q = np.zeros(k); N = np.zeros(k)
    Qs, Ns, acts, rews = [], [], [], []
    for t in range(1, steps + 1):
        if kind == "ucb":
            a = int(np.argmax(Q + ucb_bonus(t, N, c)))
        elif kind == "eps" and rng.random() < eps:
            a = int(rng.integers(k))
        else:
            a = int(np.argmax(Q + 1e-9 * rng.random(k)))
        r = q_star[a] + rng.standard_normal()
        N[a] += 1; Q[a] += (r - Q[a]) / N[a]            # the update shape, alpha = 1/N
        Qs.append(Q.copy()); Ns.append(N.copy()); acts.append(a); rews.append(r)
    return {"Q": np.array(Qs), "N": np.array(Ns), "a": np.array(acts), "r": np.array(rews)}


def ucb_bonus(t, N, c=2.0):
    with np.errstate(divide="ignore"):
        return np.where(N == 0, np.inf, c * np.sqrt(np.log(t) / np.maximum(N, 1)))
```
**Tests:**
```python
import part02_bandits as p02

def test_part02_incremental_mean_equals_sample_average():
    q = p02.make_bandit(); h = p02.run_agent(q, "eps", steps=200)
    a = h["a"][-1]
    assert h["Q"][-1][a] == pytest.approx(h["r"][h["a"] == a].mean())

def test_part02_ucb_bonus_infinite_for_untried():
    b = p02.ucb_bonus(5, np.array([0, 2.0]))
    assert np.isinf(b[0]) and b[1] == pytest.approx(2 * np.sqrt(np.log(5) / 2))

def test_part02_eps_greedy_beats_greedy_on_this_seed():
    q = p02.make_bandit()
    g = p02.run_agent(q, "greedy", steps=300); e = p02.run_agent(q, "eps", steps=300)
    assert (e["a"][-100:] == q.argmax()).mean() > (g["a"][-100:] == q.argmax()).mean()
```
If the last test fails for `seed=3`, change the `make_bandit` default seed until greedy locks onto a suboptimal arm (the story needs that), and keep the test.

**Storyboard:**
- Act 1 "ten slot machines": 10 arms as vertical Gaussian curves (violins) centred on q*. The true means stay hidden (drawn dashed) and a Q̂ dot sits at 0 on each. Captions: "each arm pays a noisy reward around an unknown mean", "we only see the rewards we sample".
- Act 2 "the update shape": pulls animate as sample dots dropping onto an arm while its Q̂ dot moves. Show `update_shape(r"r_n", CORRECT)` with the α term reading `1/N`. Caption: "new estimate = old + step × (target − old)".
- Act 3 "greedy vs ε-greedy vs UCB": three rows of small arm strips (or one strip with the agent switched), each running `run_agent` for 300 steps, sped up, with a pull-count bar under each arm. Greedy's counts pile onto one wrong arm. ε-greedy finds `argmax q*`. For UCB, draw error bars `Q ± ucb_bonus` that shrink as N grows. Captions: "greedy locks onto the first arm that looked good", "ε-greedy pays a 10% exploration tax, and finds the best arm", "ucb: optimism in the face of uncertainty".
- Takeaway: "exploration costs reward now and buys information later".

### Task 5: Part 1 · Foundations

**Maths** (`part01_foundations.py`), copied from the notebook's GridWorld:
```python
LAYOUT = ("S....", ".#.#.", ".....", ".#.P.", "....G")
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def gridworld(slip=0.1, step_reward=-0.04):
    """Returns cells, P[s,a,s'], R[s,a,s'], terminal[s] exactly as the notebook's GridWorld."""
    H, W = len(LAYOUT), len(LAYOUT[0])
    cells = [(r, c) for r in range(H) for c in range(W) if LAYOUT[r][c] != "#"]
    idx = {cell: i for i, cell in enumerate(cells)}
    ch = lambda cell: LAYOUT[cell[0]][cell[1]]
    terminal = np.array([ch(cell) in "GP" for cell in cells])
    nS = len(cells); P = np.zeros((nS, 4, nS)); R = np.zeros((nS, 4, nS))

    def move(cell, a):
        r, c = cell; nr, nc = r + MOVES[a][0], c + MOVES[a][1]
        return (nr, nc) if 0 <= nr < H and 0 <= nc < W and LAYOUT[nr][nc] != "#" else cell

    for s, cell in enumerate(cells):
        for a in range(4):
            if terminal[s]:
                P[s, a, s] = 1.0; continue
            for a2, p in [(a, 1 - slip), ((a + 1) % 4, slip / 2), ((a - 1) % 4, slip / 2)]:
                s2 = idx[move(cell, a2)]
                P[s, a, s2] += p
                R[s, a, s2] = step_reward + (1.0 if ch(cells[s2]) == "G" else -1.0 if ch(cells[s2]) == "P" else 0.0)
    return cells, P, R, terminal


def discounted_return(rewards, gamma):
    return float(sum(r * gamma ** t for t, r in enumerate(rewards)))


def optimal_values(gamma, sweeps=500):
    cells, P, R, _ = gridworld()
    V = np.zeros(len(cells))
    for _ in range(sweeps):
        V = (P * (R + gamma * V[None, None, :])).sum(-1).max(1)
    return V
```
**Tests:**
```python
import part01_foundations as p01

def test_part01_grid_has_21_states_and_stochastic_rows():
    cells, P, R, term = p01.gridworld()
    assert len(cells) == 21 and np.allclose(P.sum(-1), 1)

def test_part01_discounted_return():
    assert p01.discounted_return([1, 1, 1], 0.5) == pytest.approx(1.75)

def test_part01_far_sighted_start_value_higher():
    cells, *_ = p01.gridworld(); s0 = cells.index((0, 0))
    assert p01.optimal_values(0.99)[s0] > p01.optimal_values(0.5)[s0]
```
**Storyboard:**
- Act 1 "the loop": boxes for agent and environment with arrows labelled a_t, then r_{t+1}, s_{t+1}, cycling 3 times. Caption: "the agent acts, the world answers with a reward and a new state".
- Act 2 "a trajectory": the 5×5 grid (S, walls, pit P in red, goal G in green). A dot walks a fixed path to G, and each step's reward appears in a strip. Then G_t is built term by term with γ^t weights (`discounted_return`, γ = 0.9). Caption: "the return adds rewards, discounting the future by γ".
- Act 3 "γ is the horizon": a heatmap of `optimal_values(γ)` with a ValueTracker γ from 0.1 to 0.99. Caption: "small γ: short-sighted. γ → 1: far-sighted".
- Takeaway: "rl learns what to do from delayed, evaluative feedback".

### Task 6: Part 3 · Dynamic programming

**Maths** (`part03_dp.py`):
```python
from part01_foundations import gridworld


def value_iteration_history(gamma=0.9, sweeps=40):
    cells, P, R, _ = gridworld()
    V = np.zeros(len(cells)); hist = [V]
    for _ in range(sweeps):
        V = (P * (R + gamma * V[None, None, :])).sum(-1).max(1); hist.append(V)
    return cells, np.array(hist)


def greedy_policy(V, gamma=0.9):
    cells, P, R, _ = gridworld()
    return (P * (R + gamma * V[None, None, :])).sum(-1).argmax(1)
```
**Tests:**
```python
import part03_dp as p03

def test_part03_error_contracts_by_gamma():
    _, H = p03.value_iteration_history(0.9, 200)
    err = np.abs(H - H[-1]).max(1)
    for k in range(30):
        assert err[k + 1] <= 0.9 * err[k] + 1e-9

def test_part03_policy_from_goal_neighbour_points_to_goal():
    cells, H = p03.value_iteration_history()
    pi = p03.greedy_policy(H[-1])
    assert pi[cells.index((4, 3))] == 1          # right, into G
```
**Storyboard:**
- Act 1 "values flow from the goal": grid heatmap for sweep k = 0…15 with the sweep counter shown. The Bellman backup is drawn once as arrows from 4 neighbours into one cell. Caption: "each sweep: V(s) ← max over actions of reward + γ·V(next)".
- Act 2 "a contraction": a bar of `max|V_k − V*|` next to the grid, shrinking about ×0.9 per sweep, with a log-scale mini-plot. Caption: "the error shrinks by γ every sweep, from any starting point".
- Act 3 "policy falls out": arrows from `greedy_policy` fade in over the final heatmap. Caption: "act greedily on V* and you have the optimal policy".
- Takeaway: "every later algorithm is a sampled approximation of this".

### Task 7: Part 4 · MC, TD, SARSA, Q-learning

**Maths** (`part04_tabular.py`):
```python
H, W = 4, 12
START, GOAL = (3, 0), (3, 11)
CLIFF = {(3, c) for c in range(1, 11)}
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def step(pos, a):
    r, c = pos; nr = min(max(r + MOVES[a][0], 0), H - 1); nc = min(max(c + MOVES[a][1], 0), W - 1)
    if (nr, nc) in CLIFF:
        return START, -100.0, False
    return (nr, nc), -1.0, (nr, nc) == GOAL


def train(method, episodes=500, alpha=0.5, eps=0.1, seed=0):
    rng = np.random.default_rng(seed); Q = np.zeros((H, W, 4))
    pol = lambda s: int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[s] + 1e-9 * rng.random(4)))
    for _ in range(episodes):
        s = START; a = pol(s); done = False
        while not done:
            s2, r, done = step(s, a); a2 = pol(s2)
            target = r if done else r + (Q[s2].max() if method == "q" else Q[s2][a2])
            Q[s][a] += alpha * (target - Q[s][a]); s, a = s2, a2
    return Q


def greedy_path(Q, max_len=60):
    s, path = START, [START]
    for _ in range(max_len):
        s, _, done = step(s, int(np.argmax(Q[s]))); path.append(s)
        if done:
            break
    return path


def td_error(r, v_next, v, gamma=1.0):
    return r + gamma * v_next - v
```
**Tests:**
```python
import part04_tabular as p04

def test_part04_q_learning_takes_the_edge():
    path = p04.greedy_path(p04.train("q"))
    assert path[-1] == p04.GOAL and len(path) == 14      # 13 moves: up, 11 right, down

def test_part04_sarsa_takes_a_longer_safer_path():
    q_path = p04.greedy_path(p04.train("q")); s_path = p04.greedy_path(p04.train("sarsa"))
    assert s_path[-1] == p04.GOAL and len(s_path) > len(q_path)

def test_part04_td_error():
    assert p04.td_error(1.0, 2.0, 0.5, gamma=0.9) == pytest.approx(2.3)
```
**Storyboard:**
- Act 1 "wait vs bootstrap": a 5-state chain on the left (MC) and on the right (TD). MC's update arrow fires only when the episode ends, with target G_t. TD's fires every step with target `r + γV(s')`. Use `update_shape` twice, changing only the target term. Caption: "mc waits for the real return; td uses its own next guess".
- Act 2 "the td error": a δ_t = r + γV(s') − V(s) MathTex with each term highlighted on the chain. Caption: "this δ is the most reused quantity in rl".
- Act 3 "cliff walk": the 4×12 grid with the cliff in red. Draw the Q-learning greedy path (along the edge) and the SARSA path (safer), then show an ε-greedy stumble off the edge on the Q-learning path. Captions: "q-learning learns the optimal edge path… and falls off while exploring", "sarsa learns the policy it actually follows: keep your distance".
- Takeaway: "bootstrapping trades variance for bias; on- vs off-policy decides what you learn".

### Task 8: Part 5 · DQN

**Maths** (`part05_dqn.py`):
```python
def two_state_divergence(gamma=0.9, alpha=0.1, steps=60):
    """Tsitsiklis & Van Roy: V(s1)=w, V(s2)=2w, only s1->s2 (reward 0) is ever updated (off-policy).
    TD update w += alpha * (gamma*2w - w) * 1 diverges when 2*gamma > 1."""
    w = 1.0; ws = [w]
    for _ in range(steps):
        w += alpha * (gamma * 2 * w - w); ws.append(w)
    return np.array(ws)


def chase(target_every, steps=60, alpha=0.3, seed=0):
    """Regress q toward a target that is a noisy copy of q's own value.
    target_every=1: target moves every step; >1: frozen target network refreshed every k steps."""
    rng = np.random.default_rng(seed); q, tgt = 0.0, 0.0; qs, ts = [], []
    for t in range(steps):
        if t % target_every == 0:
            tgt = 1.0 + 0.9 * q                      # r + gamma * Q_target, true fixed point q* = 10
        q += alpha * (tgt + 0.5 * rng.standard_normal() - q); qs.append(q); ts.append(tgt)
    return np.array(qs), np.array(ts)
```
**Tests:**
```python
import part05_dqn as p05

def test_part05_deadly_triad_diverges():
    assert abs(p05.two_state_divergence(0.9)[-1]) > 10 and abs(p05.two_state_divergence(0.4)[-1]) < 1

def test_part05_frozen_target_is_piecewise_constant():
    _, ts = p05.chase(10)
    assert len(set(np.round(ts[:10], 9))) == 1
```
**Storyboard:**
- Act 1 "the deadly triad": three badges (bootstrapping, off-policy, function approximation) join together. Then the two-state w/2w picture with a plot of `two_state_divergence` blowing up. Caption: "any two are fine; all three can diverge".
- Act 2 "a moving target": a q dot chasing a target dot on a number line (`chase(1)`, jittery), then `chase(10)` with the target frozen and stepping. Caption: "freeze the target network, so you regress toward something that holds still".
- Act 3 "experience replay": a stream of consecutive, correlated transitions (coloured by time) poured into a buffer, then a random minibatch pulled out with mixed colours. Caption: "replay breaks correlations and reuses every transition".
- Takeaway: "dqn = q-learning + a network + two stabilisers".

### Task 9: Part 6 · REINFORCE

**Maths** (`part06_reinforce.py`):
```python
Q_TRUE = np.array([1.0, 2.0, 0.5])          # expected reward of 3 actions (one-step bandit)


def softmax(theta):
    z = np.exp(theta - theta.max()); return z / z.sum()


def true_grad(theta):
    p = softmax(theta); return p * (Q_TRUE - p @ Q_TRUE)


def reinforce_samples(theta, n=400, baseline=0.0, noise=1.0, seed=0):
    """n single-sample gradient estimates (R - b) * grad log pi(a)."""
    rng = np.random.default_rng(seed); p = softmax(theta); out = []
    for _ in range(n):
        a = rng.choice(3, p=p); R = Q_TRUE[a] + noise * rng.standard_normal()
        g = -p.copy(); g[a] += 1                 # grad of log softmax
        out.append((R - baseline) * g)
    return np.array(out)
```
**Tests:**
```python
import part06_reinforce as p06

def test_part06_unbiased():
    th = np.zeros(3)
    assert np.allclose(p06.reinforce_samples(th, n=40000).mean(0), p06.true_grad(th), atol=0.03)

def test_part06_baseline_reduces_variance():
    th = np.zeros(3); b = p06.softmax(th) @ p06.Q_TRUE
    v0 = p06.reinforce_samples(th, baseline=0).var(0).sum(); v1 = p06.reinforce_samples(th, baseline=b).var(0).sum()
    assert v1 < v0
```
**Storyboard:**
- Act 1 "push up what paid off": policy as 3 probability bars (POLICY_C). Sample an action, reveal its reward, and the bar grows by an amount ∝ R while the others shrink (renormalised). Repeat 4 times. Caption: "sample an action; raise its probability in proportion to the reward".
- Act 2 "the log-derivative trick": MathTex ∇J = E[R ∇log π] with ∇p = p∇log p highlighted. Caption: "no model of the world needed, just samples".
- Act 3 "noisy but unbiased": in 2-D (first two components), scatter 200 sample gradients from `reinforce_samples` as small arrows/dots around the true gradient (YELLOW arrow). Then switch on the baseline and the cloud tightens. Caption: "each estimate is noisy; a baseline shrinks the noise without moving the mean".
- Takeaway: "policy gradients optimise the policy directly, one noisy sample at a time".

### Task 10: Part 7 · Actor-critic and GAE

**Maths** (`part07_actor_critic.py`):
```python
def gae(rew, values, next_values, gamma=0.99, lam=0.95):
    """Single episode, terminal at the end (next_values[-1] ignored)."""
    T = len(rew); adv = np.zeros(T); last = 0.0
    for t in reversed(range(T)):
        nv = 0.0 if t == T - 1 else next_values[t]
        delta = rew[t] + gamma * nv - values[t]
        last = delta + gamma * lam * last; adv[t] = last
    return adv


def gae_direct(rew, values, gamma=0.99, lam=0.95):
    T = len(rew); v_next = np.append(values[1:], 0.0)
    delta = rew + gamma * v_next - values
    return np.array([sum((gamma * lam) ** l * delta[t + l] for l in range(T - t)) for t in range(T)])


def gae_weights(gamma, lam, n=12):
    return (gamma * lam) ** np.arange(n)
```
**Tests:**
```python
import part07_actor_critic as p07

def _ep():
    rng = np.random.default_rng(0); r = rng.random(8); v = rng.random(8)
    return r, v, np.append(v[1:], 0.0)

def test_part07_recursion_matches_direct_sum():
    r, v, vn = _ep(); assert np.allclose(p07.gae(r, v, vn), p07.gae_direct(r, v))

def test_part07_lambda_0_is_td_error_lambda_1_is_mc():
    r, v, vn = _ep()
    assert np.allclose(p07.gae(r, v, vn, lam=0.0), r + 0.99 * vn - v)
    G = np.array([sum(0.99 ** k * r[t + k] for k in range(8 - t)) for t in range(8)])
    assert np.allclose(p07.gae(r, v, vn, lam=1.0), G - v)
```
**Storyboard:**
- Act 1 "credit only for beating expectations": bars of returns G for 6 actions from different states, then bars of V(s) (VALUE_C) drawn behind them. Subtract, reusing the GRPO shift animation, to get the advantages. Caption: "a critic V(s) says what usually happens from here; the actor gets A = G − V".
- Act 2 "actor and critic": two boxes, actor (POLICY_C) and critic (VALUE_C), with arrows: the critic's A goes into the actor's gradient, and the returns go into the critic's regression. Caption: "the critic evaluates, the actor improves".
- Act 3 "λ dials bias vs variance": a row of δ_t tiles along an episode, with weight bars `gae_weights(0.99, λ)` above them and a λ ValueTracker from 0 to 1. Labels at the ends: "λ=0: td error (low variance, biased)" and "λ=1: monte carlo (unbiased, noisy)". Caption: "gae blends every n-step estimate with weights (γλ)^l".
- Takeaway: "advantages = credit for beating expectations; λ picks how far to trust the critic".

### Task 11: Part 8 · TRPO

**Maths** (`part08_trpo.py`):
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def kl_bern(t1, t2):
    """KL(pi_t1 || pi_t2) for a 2-action policy with P(a=1) = sigmoid(theta)."""
    p, q = sigmoid(t1), sigmoid(t2)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def fisher(t):
    p = sigmoid(t); return p * (1 - p)


def natural_step(t, grad, max_kl=0.01):
    """Step length whose quadratic KL model 0.5*F*d^2 equals max_kl (the notebook's max_kl)."""
    F = fisher(t); d = np.sqrt(2 * max_kl / F)
    return np.sign(grad) * d
```
**Tests:**
```python
import part08_trpo as p08

def test_part08_same_param_step_different_kl():
    assert p08.kl_bern(0.0, 1.0) > 5 * p08.kl_bern(4.0, 5.0)

def test_part08_quadratic_kl_model():
    d = 1e-2; assert p08.kl_bern(0.3, 0.3 + d) == pytest.approx(0.5 * p08.fisher(0.3) * d * d, rel=1e-2)

def test_part08_natural_step_hits_max_kl():
    t = 2.0; d = p08.natural_step(t, 1.0); assert p08.kl_bern(t, t + d) == pytest.approx(0.01, rel=0.1)
```
**Storyboard:**
- Act 1 "the step-size problem": policy bars P(a=1) = σ(θ). The same parameter step Δθ = 1 applied at θ = 0 and at θ = 4, showing a big change vs a tiny change in the policy, with KL numbers from `kl_bern`. Caption: "equal steps in parameters ≠ equal changes in behaviour".
- Act 2 "a trust region in policy space": a 2-D parameter plane with a Euclidean circle vs the KL ellipse (Fisher metric; use a fixed 2×2 matrix like [[3,1],[1,0.5]] for the picture). The gradient arrow vs the natural-gradient arrow F⁻¹g landing on the ellipse boundary. Caption: "trpo takes the biggest step whose kl stays under δ = 0.01".
- Act 3 "why it matters": a short depiction of collapse, where a bad big step makes the policy collect bad data. Show a sample loop degrading. Caption: "on-policy, a bad step also ruins the next batch of data".
- Takeaway: "measure steps by how much the policy changes, not the parameters".

### Task 12: Part 9 · PPO

**Maths** (`part09_ppo.py`):
```python
EPS = 0.2


def clip_objective(rho, A, eps=EPS):
    rho = np.asarray(rho, float)
    return np.minimum(rho * A, np.clip(rho, 1 - eps, 1 + eps) * A)


def clip_grad(rho, A, eps=EPS, h=1e-6):
    return (clip_objective(rho + h, A, eps) - clip_objective(rho - h, A, eps)) / (2 * h)
```
**Tests:**
```python
import part09_ppo as p09

def test_part09_positive_advantage_flat_above_1_plus_eps():
    assert p09.clip_grad(1.3, 1.0) == pytest.approx(0) and p09.clip_grad(1.1, 1.0) == pytest.approx(1.0)

def test_part09_negative_advantage_flat_below_1_minus_eps():
    assert p09.clip_grad(0.7, -1.0) == pytest.approx(0) and p09.clip_grad(0.9, -1.0) == pytest.approx(-1.0)

def test_part09_pessimistic_bound():
    r = np.linspace(0.5, 1.5, 11)
    assert np.all(p09.clip_objective(r, 1.0) <= r * 1.0 + 1e-12)
```
**Storyboard:**
- Act 1 "the ratio": ρ = π_θ/π_old shown as two probability bars with ρ as their quotient. Caption: "ρ measures how far the new policy moved on this action".
- Act 2 "A > 0": axes ρ ∈ [0, 2]. Draw ρA (dashed) and clip(ρ)A (dashed), then their min (solid, POLICY_C) with shaded band 1±ε. A ball rolls up the solid curve and stops on the flat part past 1.2, with a gradient arrow shrinking to zero (`clip_grad`). Caption: "good action: raise it, but not past 1 + ε, where the gradient goes to zero".
- Act 3 "A < 0": the mirror image, with the ball stopping at 0.8. Then both panels side by side. Caption: "bad action: lower it, but not below 1 − ε".
- Takeaway: "ppo = trpo's idea with one min and one clip".

### Task 13: Part 10 · LLMs as RL

**Maths** (`part10_llm.py`):
```python
MAPPING = [
    ("state", "prompt + tokens so far"),
    ("action", "next token"),
    ("policy", "next-token distribution"),
    ("transition", "append the token"),
    ("episode", "one full response"),
    ("reward", "one number at the end"),
]


def terminal_rewards(n_tokens, R):
    r = np.zeros(n_tokens); r[-1] = R; return r
```
**Tests:**
```python
import part10_llm as p10

def test_part10_reward_is_sparse_and_terminal():
    r = p10.terminal_rewards(6, 1.0); assert r.sum() == 1.0 and r[-1] == 1.0 and np.count_nonzero(r) == 1

def test_part10_mapping_has_six_rows():
    assert len(p10.MAPPING) == 6
```
**Storyboard:**
- Act 1 "same loop, new agent": the Part 1 agent⇄env loop redrawn, then morphed so the agent becomes an LM box and the environment becomes "append token". Then the table from `MAPPING` fills in, row by row, with the RL concept on the left and the LLM meaning on the right. Caption: "an llm is a policy; each token is an action".
- Act 2 "generation is a tree": prompt `14+17=`. Branching next-token choices with probabilities, and one path highlighted to `31<eos>`. Caption: "the state is everything generated so far".
- Act 3 "one reward at the end": a token strip with rewards all 0 except the last (`terminal_rewards`), lighting up green at EOS. Caption: "a single scalar for the whole response: credit assignment is hard".
- Takeaway: "every algorithm from parts 6–9 now trains a language model".

### Task 14: Part 11 · Reward models

**Maths** (`part11_reward_model.py`):
```python
def bt_prob(r_w, r_l):
    return 1 / (1 + np.exp(-(r_w - r_l)))


def bt_loss(r_w, r_l):
    return -np.log(bt_prob(r_w, r_l))


def best_of_n_curves(ns=(1, 2, 4, 8, 16, 32, 64, 128, 256), trials=4000, seed=0):
    """Candidates: true quality q ~ N(0,1), length L ~ N(0,1). Proxy RM rewards length (bias 0.8),
    true reward punishes rambling. Pick best-of-n by proxy; report mean proxy and mean true reward."""
    rng = np.random.default_rng(seed); proxy_m, true_m = [], []
    for n in ns:
        q = rng.standard_normal((trials, n)); L = rng.standard_normal((trials, n))
        proxy = q + 0.8 * L; true = q - 0.35 * np.maximum(L, 0) ** 2
        pick = proxy.argmax(1); rows = np.arange(trials)
        proxy_m.append(proxy[rows, pick].mean()); true_m.append(true[rows, pick].mean())
    return np.array(ns), np.array(proxy_m), np.array(true_m)
```
**Tests:**
```python
import part11_reward_model as p11

def test_part11_bt_half_at_equal_rewards():
    assert p11.bt_prob(1.0, 1.0) == pytest.approx(0.5) and p11.bt_loss(0, 0) == pytest.approx(np.log(2))

def test_part11_goodhart_proxy_rises_true_peaks_inside():
    ns, px, tr = p11.best_of_n_curves()
    assert np.all(np.diff(px) > 0) and 0 < int(np.argmax(tr)) < len(ns) - 1
```
If the Goodhart test fails, tune only the 0.8 / 0.35 constants until the true reward peaks at an interior n.

**Storyboard:**
- Act 1 "preferences, not scores": two answer cards; an annotator picks one (checkmark). Caption: "humans compare two answers; they can't give absolute scores".
- Act 2 "bradley–terry": a sigmoid plot of P(y_w ≻ y_l) vs r_w − r_l, with a dot sliding as the reward model trains and the gap widening. Show the loss −log σ(Δ) as a second curve. Caption: "turn a reward difference into a preference probability, and fit it".
- Act 3 "goodhart": `best_of_n_curves` plotted on a log-n axis. The proxy keeps rising while the true reward rises and then falls, with the peak marked. Caption: "optimise the proxy too hard and you optimise its mistakes (here: length)".
- Takeaway: "a reward model is a learned, imperfect stand-in for human judgment".

### Task 15: Part 12 · RLHF with PPO

**Maths** (`part12_ppo_llm.py`):
```python
def per_token_rewards(logp, ref_logp, R, beta=0.02):
    r = -beta * (np.asarray(logp) - np.asarray(ref_logp)); r = r.copy(); r[-1] += R; return r


def gae_tokens(rewards, values, lam=0.95):
    """gamma = 1 over one response; terminal after the last token."""
    T = len(rewards); adv = np.zeros(T); last = 0.0
    for t in reversed(range(T)):
        nv = values[t + 1] if t + 1 < T else 0.0
        last = rewards[t] + nv - values[t] + lam * last; adv[t] = last
    return adv
```
**Tests:**
```python
import part12_ppo_llm as p12

def test_part12_kl_penalty_every_token_reward_at_end():
    r = p12.per_token_rewards([-1, -1, -1], [-1.5, -1.0, -2.0], R=1.0, beta=0.1)
    assert r == pytest.approx([-0.05, 0.0, 1.0 - 0.1])

def test_part12_gae_lambda1_is_return_minus_value():
    r = np.array([0.0, 0.0, 1.0]); v = np.array([0.2, 0.5, 0.7])
    assert np.allclose(p12.gae_tokens(r, v, lam=1.0), r[::-1].cumsum()[::-1] - v)
```
**Storyboard:**
- Act 1 "four models": four boxes appear: policy (POLICY_C, trained), reference (REF_C, frozen), reward model (MEAN_C, frozen) and value model (VALUE_C, trained). Arrows show the data flow for one response. Caption: "ppo-rlhf juggles four copies of a language model".
- Act 2 "kl as a per-token reward": a token strip of the response. Each token gets a small −β·(log π − log π_ref) bar (from `per_token_rewards`), with the final token getting +R. Caption: "stay close to the reference: pay a small kl toll on every token".
- Act 3 "credit flows backwards": GAE sweeping from the last token to the first (`gae_tokens`), with value-model bars behind. Caption: "the critic spreads the final reward back over the tokens".
- Takeaway: "powerful, but heavy: the next parts remove pieces of this machine".

### Task 16: Part 13 · DPO

**Maths** (`part13_dpo.py`):
```python
def implicit_reward(logp, ref_logp, beta=0.1):
    return beta * (np.asarray(logp) - np.asarray(ref_logp))


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    margin = implicit_reward(pi_w, ref_w, beta) - implicit_reward(pi_l, ref_l, beta)
    return np.log1p(np.exp(-margin))             # -log sigmoid(margin)
```
**Tests:**
```python
import part13_dpo as p13

def test_part13_zero_margin_is_log2():
    assert p13.dpo_loss(-5, -5, -5, -5) == pytest.approx(np.log(2))

def test_part13_raising_chosen_lowers_loss():
    assert p13.dpo_loss(-4, -5, -5, -5) < p13.dpo_loss(-5, -5, -5, -5) < p13.dpo_loss(-5, -4, -5, -5)
```
**Storyboard:**
- Act 1 "the optimal policy has a formula": show π* = π_ref · exp(r/β) / Z(x). Solve for r = β log π*/π_ref + β log Z(x), as a TransformMatchingTex step. Caption: "the kl-regularised objective has a closed-form optimum".
- Act 2 "Z cancels": plug into Bradley–Terry with two terms r(y_w) − r(y_l). Both β log Z(x) terms get highlighted, then struck out and faded. Caption: "both answers share the prompt, so the intractable Z(x) cancels".
- Act 3 "the loss": log-prob bars for the chosen answer (CORRECT) rising and the rejected one (WRONG) falling, relative to the reference (REF_C ghost bars). The margin dot slides up the sigmoid and `dpo_loss` counts down from log 2. Caption: "no reward model, no sampling, no critic: just pairs".
- Takeaway: "dpo turns rlhf into a classification loss on preference pairs".

### Task 17: Part 14 · RLOO and REINFORCE++

**Maths** (`part14_rloo.py`):
```python
def rloo_advantages(R):
    R = np.asarray(R, float); K = len(R)
    return R - (R.sum() - R) / (K - 1)


def batch_normalised(R_batch):
    """REINFORCE++-style: one mean/std across the whole batch (all prompts), not per group."""
    R = np.asarray(R_batch, float); return (R - R.mean()) / (R.std() + 1e-8)
```
**Tests:**
```python
import part14_rloo as p14

def test_part14_rloo_example():
    assert p14.rloo_advantages([1, 0, 0, 0]) == pytest.approx([1, -1 / 3, -1 / 3, -1 / 3])

def test_part14_rloo_sums_to_zero():
    assert p14.rloo_advantages([0.3, 0.9, 0.1, 0.5]).sum() == pytest.approx(0)

def test_part14_batch_norm_is_global():
    a = p14.batch_normalised([1, 0, 1, 1, 0, 0, 0, 0])
    assert a.mean() == pytest.approx(0, abs=1e-9) and a.std() == pytest.approx(1)
```
**Storyboard:**
- Act 1 "the critic only gave a baseline": recap of Part 12's four boxes, then the value model fading out. Caption: "with a single end-of-answer reward, the critic's only job was a baseline".
- Act 2 "leave one out": K = 4 sample cards with rewards. For each card in turn, highlight the *other three*, average them into a YELLOW baseline tick, and draw the advantage bar (`rloo_advantages`). Caption: "each sample's baseline is the mean of the others: free and unbiased".
- Act 3 "reinforce++": several prompts' groups laid out together, with one global mean and std line across the whole batch (`batch_normalised`). Caption: "reinforce++ normalises across the whole batch instead of per prompt".
- Takeaway: "sample more, and let the samples be each other's critic". This is the setup for GRPO (Part 15).
- Do **not** state *why* including R_i in its own baseline biases the gradient (that is a quiz answer).

### Task 18: Part 16 · After GRPO

**Maths** (`part16_beyond_grpo.py`):
```python
def token_weights(lengths, agg):
    """Weight each token gets in the loss. 'seq': GRPO's (1/G) * (1/|y_i|); 'token': 1/sum|y| (Dr. GRPO / DAPO)."""
    L = np.asarray(lengths, float)
    if agg == "seq":
        return 1.0 / (len(L) * L)
    return np.full(len(L), 1.0 / L.sum())


def gspo_ratio(token_log_ratios):
    return float(np.exp(np.mean(token_log_ratios)))


def clip_bounds(name):
    """(low, high) clip range on the ratio, from the notebook's RLConfig table."""
    return {"GRPO": (0.8, 1.2), "DAPO": (0.8, 1.28), "GSPO": (1 - 0.003, 1 + 0.004)}[name]
```
**Tests:**
```python
import part16_beyond_grpo as p16

def test_part16_seq_mean_makes_long_answers_cheaper_per_token():
    w = p16.token_weights([10, 40], "seq"); assert w[1] < w[0]

def test_part16_token_level_is_uniform():
    w = p16.token_weights([10, 40], "token"); assert w[0] == w[1] == pytest.approx(1 / 50)

def test_part16_gspo_is_geometric_mean():
    assert p16.gspo_ratio(np.log([1.1, 0.9, 1.0])) == pytest.approx((1.1 * 0.9 * 1.0) ** (1 / 3))

def test_part16_dapo_clip_higher():
    assert p16.clip_bounds("DAPO")[1] > p16.clip_bounds("GRPO")[1]
```
**Storyboard:**
- Act 1 "three dials": three dials labelled *advantage*, *ratio & clip* and *aggregation*, each with GRPO's setting (mean/std, token ratio with ±0.2, 1/|y| per answer). Caption: "every successor is a different setting of three dials".
- Act 2 "dr. grpo": two wrong answers (10 and 40 tokens) as token strips, with per-token penalty weights from `token_weights(..., "seq")`. The long one's tokens get lighter, i.e. cheaper. Caption: "averaging per answer makes long wrong answers cheaper per token → rambling". Switch to "token" and the weights equalise. Dr. GRPO also turns the std dial off (show a mini version of Part 15's plot).
- Act 3 "dapo, gspo, cispo": turn the dials in turn. DAPO: clip-higher (the band stretches to 1.28) plus dynamic sampling (dead groups get dropped and resampled). GSPO: the ratio dial switches from per-token to one sequence ratio (`gspo_ratio`, geometric mean animation). CISPO: clip the importance *weight* but keep every token's gradient. Captions are one line per method.
- Takeaway: "advantage, ratio, aggregation: know the three dials and you can read any new paper".

### Task 19: Part 17 · Agentic RL

**Maths** (`part17_agentic.py`):
```python
SEGMENTS = [("prompt", 6), ("model", 5), ("tool", 4), ("model", 4), ("tool", 3), ("model", 3)]


def loss_mask(segments=SEGMENTS):
    return np.concatenate([np.full(n, 1.0 if kind == "model" else 0.0) for kind, n in segments])
```
**Tests:**
```python
import part17_agentic as p17

def test_part17_only_model_tokens_trained():
    m = p17.loss_mask(); assert m.sum() == 12 and len(m) == 25

def test_part17_mask_zero_on_tool_output():
    m = p17.loss_mask([("model", 2), ("tool", 3)]); assert list(m) == [1, 1, 0, 0, 0]
```
**Storyboard:**
- Act 1 "a multi-turn episode": a ribbon of token tiles built left to right with segment labels: prompt (REF_C), model turn (POLICY_C), tool call → tool output (GREY/orange), model turn, …, final answer, reward. Use the notebook's arithmetic + calculator example. Caption: "an agent's trajectory interleaves its own tokens with the world's".
- Act 2 "loss masking": a mask row slides under the ribbon (`loss_mask`). Tool and prompt tiles fade to grey with ∇ = 0 labels, and model tiles glow. Caption: "only tokens the model wrote get a gradient".
- Act 3 "without the mask": the tool output tiles also get gradient arrows, and the model learns to *predict* the calculator's answer: show a hallucinated tool result. Caption: "unmasked, the model learns to fake tool outputs".
- Takeaway: "same algorithms, longer episodes: mask what you didn't generate".

### Task 20: Part 18 · The map

**Maths** (`part18_wrapup.py`):
```python
NODES = {  # name -> (part, family)
    "Bandits": (2, "tabular"), "Value/Policy iteration": (3, "dp"), "TD / SARSA / Q-learning": (4, "tabular"),
    "DQN": (5, "value"), "REINFORCE": (6, "pg"), "Actor-critic + GAE": (7, "pg"), "TRPO": (8, "pg"),
    "PPO": (9, "pg"), "Reward model": (11, "llm"), "PPO-RLHF": (12, "llm"), "DPO": (13, "llm"),
    "RLOO / REINFORCE++": (14, "llm"), "GRPO": (15, "llm"), "Dr. GRPO / DAPO / GSPO / CISPO": (16, "llm"),
    "Agentic RL": (17, "llm"),
}
EDGES = [  # (from, to, label)
    ("Value/Policy iteration", "TD / SARSA / Q-learning", "sample, don't plan"),
    ("TD / SARSA / Q-learning", "DQN", "+ network"),
    ("Bandits", "REINFORCE", "learn the policy"),
    ("REINFORCE", "Actor-critic + GAE", "+ critic"),
    ("Actor-critic + GAE", "TRPO", "+ trust region"),
    ("TRPO", "PPO", "clip instead"),
    ("PPO", "PPO-RLHF", "+ LM, + KL"),
    ("Reward model", "PPO-RLHF", "reward"),
    ("Reward model", "DPO", "closed form"),
    ("PPO-RLHF", "RLOO / REINFORCE++", "− critic"),
    ("RLOO / REINFORCE++", "GRPO", "group baseline"),
    ("GRPO", "Dr. GRPO / DAPO / GSPO / CISPO", "fix the dials"),
    ("GRPO", "Agentic RL", "multi-turn"),
]


def connected(nodes=NODES, edges=EDGES):
    adj = {n: set() for n in nodes}
    for a, b, _ in edges:
        adj[a].add(b); adj[b].add(a)
    seen, stack = set(), [next(iter(nodes))]
    while stack:
        n = stack.pop()
        if n not in seen:
            seen.add(n); stack.extend(adj[n] - seen)
    return seen == set(nodes)
```
**Tests:**
```python
import part18_wrapup as p18

def test_part18_graph_is_connected_and_edges_valid():
    assert all(a in p18.NODES and b in p18.NODES for a, b, _ in p18.EDGES) and p18.connected()

def test_part18_covers_parts_2_to_17_except_1_and_10():
    assert {p for p, _ in p18.NODES.values()} == set(range(2, 18)) - {10}
```
**Storyboard:**
- Act 1 "the family tree": nodes appear in part order with rounded boxes coloured by family and edges drawn with their labels (use a fixed hand-placed layout: tabular/DP column left, policy gradient centre, LLM right). The camera stays still; the graph fits the frame (shrink fonts rather than overflow). Caption per era: "planning → sampling", "values → policies", "policies → language models".
- Act 2 "three questions": highlight paths answering "do you have a verifier?", "can you afford a critic?" and "on- or off-policy?" by lighting subsets of nodes. Caption: "pick by what you have: a model, a verifier, preferences, compute".
- Takeaway: "one idea throughout: move toward what was better than expected".
- Outro: `roadmap_outro(self, 18)` with every dot lit.

---

### Task 21: Embed everything and ship

**Files:**
- Modify: `tools/notebook_builder/sec1_foundations.py` … `sec18_wrapup.py`, `rl_from_bandits_to_grpo.ipynb`, `README.md`

- [ ] **Step 1: Render all finals and check durations**

Run: `.venv/bin/python tools/manim_scenes/render.py all && for n in $(seq 1 18); do ffprobe -v error -show_entries format=duration -of csv=p=0 videos/part$(printf %02d $n).mp4; done`
Expected: 18 files, each 60–120 s.

- [ ] **Step 2: Add `video(N)` to each section file right after its Part header**

For each `secN_*.py` (N = 1…18), change `from common import md, code, ...` to also import `video`, and insert `CELLS.append(video(N))` directly after the `CELLS.append(md(r"""\n---\n# Part N · ...`)` block. Verify with an assembled copy (not the shipped notebook):
```bash
cd tools/notebook_builder && ../../.venv/bin/python assemble.py /tmp/check.ipynb && \
../../.venv/bin/python -c "import nbformat; nb=nbformat.read('/tmp/check.ipynb',4); print(sorted(c.metadata['companion_video'] for c in nb.cells if 'companion_video' in c.metadata))"
```
Expected: `[1, 2, ..., 18]`, with each video cell immediately after its Part header.

- [ ] **Step 3: Patch the executed notebook**

Run: `.venv/bin/python tools/notebook_builder/insert_videos.py rl_from_bandits_to_grpo.ipynb` and then run it once more.
Expected: `inserted 18 video cells`, then `inserted 0 video cells`. Check outputs are intact: the count of cells with outputs is still 171.

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 5: README**

Add a section to `README.md` after "Running it":
```markdown
## Companion videos

Every Part opens with a short Manim animation (`videos/partNN.mp4`) that plays inline in local Jupyter.
Sources and render instructions: [`tools/manim_scenes/`](tools/manim_scenes/README.md).
```

- [ ] **Step 6: Commit**

```bash
git add tools/notebook_builder/sec*.py rl_from_bandits_to_grpo.ipynb README.md videos
git commit -m "Embed companion videos in every Part of the notebook

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
