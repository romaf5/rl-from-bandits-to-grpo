"""Part 17 · Agentic RL: multi-turn episodes, tool outputs as observations, and loss masking."""
import numpy as np
from style import *

SEGMENTS = [("prompt", 6), ("model", 5), ("tool", 4), ("model", 4), ("tool", 3), ("model", 3)]


def loss_mask(segments=SEGMENTS):
    return np.concatenate([np.full(n, 1.0 if kind == "model" else 0.0) for kind, n in segments])


# ---------------------------------------------------------------- the on-screen episode
# Tokens come from the notebook's vocabulary (digits, +, =, <call>, <obs>, <eos>). On <call> the
# environment appends "<obs> digits <obs>". Here the calculator adds the expression the model wrote
# before <call>, so one episode can make two calls: 16 + 7 (units), then 2 + 1 (tens).
A, B = 16, 17
TOOL_COST = 0.25                       # notebook: reward = correct − 0.25 · used_tool
CALL, OBS, EOS = "<call>", "<obs>", "<eos>"
FAKE_OFFSET = 1                        # act 3: the unmasked model's made-up first observation is off by this


def digits(n):
    return [str(d) for d in str(n)]


def calculator(expr):
    x, y = "".join(expr).split("+")
    return [OBS] + digits(int(x) + int(y)) + [OBS]


def episode(fake=False):
    """[(kind, tokens)]. fake=True: the model writes the first observation itself instead of calling the tool."""
    prompt = digits(A) + ["+"] + digits(B) + ["="]
    units = digits(A) + ["+"] + digits(B % 10)
    obs1 = calculator(units)
    if fake:
        obs1 = [OBS] + digits(int("".join(obs1[1:-1])) + FAKE_OFFSET) + [OBS]
    partial = int("".join(obs1[1:-1]))
    tens = [str(partial // 10), "+", str(B // 10)]
    obs2 = calculator(tens)
    final = obs2[1:-1] + [str(partial % 10), EOS]
    if fake:
        return [("prompt", prompt), ("model", units + obs1 + tens + [CALL]), ("tool", obs2), ("model", final)]
    return [("prompt", prompt), ("model", units + [CALL]), ("tool", obs1), ("model", tens + [CALL]),
            ("tool", obs2), ("model", final)]


def final_answer(segs):
    return int("".join(segs[-1][1][:-1]))


def true_obs():
    return int("".join(calculator(digits(A) + ["+"] + digits(B % 10))[1:-1]))


def reward(segs):
    used = any(CALL in toks for kind, toks in segs if kind == "model")
    return float(final_answer(segs) == A + B) - TOOL_COST * used


assert [(k, len(t)) for k, t in episode()] == SEGMENTS    # the ribbon is exactly loss_mask(SEGMENTS)

KIND_C = {"prompt": REF_C, "model": POLICY_C, "tool": STD_C}
RIBBON_Y = 0.95
TILE_H = 0.62


def num(v):
    return f"{v:+.2f}" if v else "0"         # for MathTex (ASCII minus)


def tile(tok, color, stroke=None):
    t = Text(tok, font="DejaVu Sans Mono", font_size=26 if len(tok) == 1 else 17)
    box = RoundedRectangle(corner_radius=0.07, width=max(0.4, t.width + 0.14), height=TILE_H)
    box.set_stroke(stroke or color, 2.5 if stroke else 2).set_fill(color, 0.3)
    return VGroup(box, t.move_to(box))


def ribbon(segs, fake_seg=None):
    """VGroup of segment VGroups of tiles. fake_seg: index of the segment whose <obs>…<obs> the model faked."""
    groups = VGroup()
    for s, (kind, toks) in enumerate(segs):
        g = VGroup()
        in_obs = False
        for tok in toks:
            if s == fake_seg and tok == OBS:
                in_obs = not in_obs
            fake = s == fake_seg and (in_obs or tok == OBS)
            g.add(tile(tok, KIND_C[kind], stroke=WRONG if fake else None))
        groups.add(g.arrange(RIGHT, buff=0.05))
    return groups.arrange(RIGHT, buff=0.16)


SEG_LABELS = ["prompt", "model turn", "tool output", "model turn", "tool output", "final answer"]


class Part17(Scene):
    def construct(self):
        self.title = title_card(self, 17, "Agentic RL: tools and loss masking")
        self.cap = Captioner(self, self.title)
        self.act1()
        clear_act(self, keep=[self.title, self.rib, self.labels, self.rew])
        self.act2()
        clear_act(self, keep=[self.title, self.rib, self.labels, self.rew])
        self.act3()
        takeaway(self, "same algorithms, longer episodes: mask what you didn't generate")
        roadmap_outro(self, 17)

    # ------------------------------------------------------------------ act 1: a multi-turn episode
    def act1(self):
        cap = self.cap
        segs = episode()
        R = reward(segs)
        rib = ribbon(segs)
        rew = VGroup(RoundedRectangle(corner_radius=0.1, width=1.5, height=TILE_H).set_stroke(CORRECT, 2.5)
                     .set_fill(CORRECT, 0.2),
                     MathTex(rf"R = {R:.2f}", font_size=30, color=CORRECT))
        rew[1].move_to(rew[0])
        VGroup(rib, rew).arrange(RIGHT, buff=0.3)
        fit(VGroup(rib, rew), 12.8).move_to(RIBBON_Y * UP)
        labels = VGroup()
        for i, (g, name) in enumerate(zip(rib, SEG_LABELS)):
            col = KIND_C[segs[i][0]]
            line = Line(g.get_corner(UL), g.get_corner(UR), color=col, stroke_width=3).shift(0.14 * UP)
            lab = Text(name, font_size=19, color=col).next_to(line, UP, buff=0.1)
            if i % 2:
                lab.shift(0.34 * UP)
            labels.add(VGroup(line, lab))
        rew_lab = Text("reward", font_size=19, color=CORRECT).next_to(rew, UP, buff=0.24)
        self.rib, self.labels, self.rew = rib, labels, VGroup(rew, rew_lab)

        cap.say("an agent's trajectory interleaves its own tokens with the world's")
        calc = VGroup(RoundedRectangle(corner_radius=0.12, width=2.4, height=0.7).set_stroke(STD_C, 2),
                      Text("calculator", font_size=24, color=STD_C))
        calc[1].move_to(calc[0])
        calc.move_to(1.3 * DOWN)
        legend = VGroup(*[
            VGroup(Square(0.25).set_fill(c, 0.3).set_stroke(c, 2), Text(t, font_size=20, color=c)).arrange(RIGHT, buff=0.12)
            for t, c in [("prompt", REF_C), ("written by the model", POLICY_C), ("written by the tool", STD_C)]
        ]).arrange(RIGHT, buff=0.6).to_edge(DOWN, buff=0.5)

        note = VGroup(Text("illustrative episode: two tool calls for clarity", font_size=21, color=MUTED),
                      Text("(the notebook's calculator makes one call and returns a + b)", font_size=21,
                           color=MUTED)).arrange(DOWN, buff=0.1).next_to(legend, UP, buff=0.3)
        self.play(FadeIn(legend), FadeIn(note))
        for i, g in enumerate(rib):
            kind = segs[i][0]
            if kind == "tool":
                if calc not in self.mobjects:
                    self.play(FadeIn(calc))
                call_tile = rib[i - 1][-1]
                expr = "".join(segs[i - 1][1][:-1])
                q = Text(expr, font="DejaVu Sans Mono", font_size=22, color=POLICY_C).next_to(calc, LEFT, buff=0.3)
                a1 = Arrow(call_tile.get_bottom(), calc.get_top() + 0.4 * LEFT, buff=0.1, color=POLICY_C,
                           stroke_width=3, max_tip_length_to_length_ratio=0.12)
                a2 = Arrow(calc.get_top() + 0.4 * RIGHT, g.get_bottom(), buff=0.1, color=STD_C,
                           stroke_width=3, max_tip_length_to_length_ratio=0.12)
                self.play(Indicate(call_tile, color=POLICY_C), GrowArrow(a1), FadeIn(q))
                self.play(GrowArrow(a2), Indicate(calc, color=STD_C, scale_factor=1.05))
                self.play(LaggedStart(*[FadeIn(t, shift=0.3 * UP) for t in g], lag_ratio=0.15), FadeIn(labels[i]))
                self.wait(0.4)
                self.play(FadeOut(a1), FadeOut(a2), FadeOut(q))
            else:
                self.play(LaggedStart(*[FadeIn(t, shift=0.2 * RIGHT) for t in g], lag_ratio=0.1), FadeIn(labels[i]),
                          run_time=1.2)
            if i == 1:
                cap.say("the model writes <call>: that token is its action")
            if i == 2:
                cap.say("the environment answers with <obs> tokens: an observation", color=STD_C)
        self.play(FadeOut(calc))
        cap.say(f"reward: {A} + {B} = {final_answer(segs)} is correct, minus the {TOOL_COST} tool cost",
                color=CORRECT)
        self.play(FadeIn(rew, shift=0.2 * LEFT), FadeIn(rew_lab))
        self.wait(1.5)
        n_model = int(loss_mask().sum())
        cap.say(f"{len(loss_mask())} tokens in the episode, but only {n_model} were written by the model", wait=2)

    # ------------------------------------------------------------------ act 2: loss masking
    def act2(self):
        cap, rib = self.cap, self.rib
        m = loss_mask()
        tiles = [t for g in rib for t in g]
        for t in tiles:
            t.save_state()
        cap.say("only tokens the model wrote get a gradient")

        mask_lab = MathTex(r"m_t", font_size=32, color=MUTED)
        digits_row = VGroup(*[Text(f"{v:.0f}", font="DejaVu Sans Mono", font_size=24,
                                   color=POLICY_C if v else GREY_C).next_to(t, DOWN, buff=0.28)
                              for t, v in zip(tiles, m)])
        mask_lab.next_to(digits_row, LEFT, buff=0.15)
        self.play(LaggedStart(*[FadeIn(d, shift=0.3 * RIGHT) for d in digits_row], lag_ratio=0.05),
                  FadeIn(mask_lab), run_time=2)
        self.wait(0.5)

        # Masked tiles go grey; model tiles glow.
        dim, glow = [], []
        for t, v in zip(tiles, m):
            if v:
                glow.append(t[0].animate.set_stroke(POLICY_C, 4).set_fill(POLICY_C, 0.55))
            else:
                dim.append(t.animate.set_opacity(0.35))
        self.play(*dim, *glow, run_time=1.2)

        grads = VGroup()
        for i, g in enumerate(rib):
            ref = VGroup(*[digits_row[j] for j in range(len(tiles)) if tiles[j] in g])
            if SEGMENTS[i][0] == "model":
                arr = VGroup(*[Arrow(d.get_bottom() + 0.55 * DOWN, d.get_bottom() + 0.05 * DOWN, buff=0,
                                     color=POLICY_C, stroke_width=3, max_tip_length_to_length_ratio=0.35)
                               for d in ref])
                grads.add(VGroup(arr, MathTex(r"\nabla", font_size=30, color=POLICY_C).next_to(arr, DOWN, buff=0.1)))
            else:
                grads.add(MathTex(r"\nabla = 0", font_size=28, color=GREY_B).next_to(ref, DOWN, buff=0.35))
        self.play(LaggedStart(*[FadeIn(g) for g in grads], lag_ratio=0.15), run_time=1.5)

        count = VGroup(Integer(int(m.sum()), font_size=34, color=POLICY_C),
                       Text(f"of {len(m)} tokens trained", font_size=26)).arrange(RIGHT, buff=0.15)
        formula = MathTex(r"\mathcal{L} = -\,\hat{A}\sum_t", r"m_t", r"\,\log \pi_\theta(y_t \mid y_{<t})",
                          font_size=36)
        formula[1].set_color(POLICY_C)
        VGroup(formula, count).arrange(RIGHT, buff=1.0).to_edge(DOWN, buff=0.35)
        self.play(Write(formula), FadeIn(count))
        cap.say("tool outputs are masked out exactly like the prompt", wait=1.5)
        cap.say("<call> stays in: choosing to use the tool is the model's action", color=POLICY_C)
        calls = [t for t, tok in zip(tiles, [tok for _, toks in episode() for tok in toks]) if tok == CALL]
        self.play(*[Indicate(c, color=POLICY_C, scale_factor=1.25) for c in calls], run_time=1.2)
        self.wait(1.5)

        # Restore the ribbon for act 3.
        self.play(*[Restore(t) for t in tiles])

    # ------------------------------------------------------------------ act 3: without the mask
    def act3(self):
        cap, rib = self.cap, self.rib
        m = loss_mask([("model" if k == "tool" else k, n) for k, n in SEGMENTS])   # tool outputs left unmasked
        cap.say("now drop the mask on tool outputs: they get a gradient too", color=WRONG)
        tool_idx = [i for i, (k, _) in enumerate(SEGMENTS) if k == "tool"]
        arrows = VGroup()
        for i, (kind, _) in enumerate(SEGMENTS):
            if kind == "prompt":
                continue
            for t in rib[i]:
                arrows.add(Arrow(t.get_bottom() + 0.6 * DOWN, t.get_bottom() + 0.05 * DOWN, buff=0,
                                 color=WRONG if kind == "tool" else POLICY_C, stroke_width=3,
                                 max_tip_length_to_length_ratio=0.35))
        count = VGroup(Integer(int(m.sum()), font_size=34, color=WRONG),
                       Text(f"of {len(m)} tokens trained", font_size=26)).arrange(RIGHT, buff=0.15)
        count.next_to(arrows, DOWN, buff=0.25).set_x(0)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.08),
                  *[Indicate(t, color=WRONG, scale_factor=1.1) for i in tool_idx for t in rib[i]],
                  FadeIn(count), run_time=1.8)
        cap.say("the model is trained to predict the calculator's answers", wait=1.5)

        # A later rollout from the unmasked model: it writes the observation itself.
        fake = episode(fake=True)
        frib = ribbon(fake, fake_seg=1)
        frib.scale(rib[0][0].height / frib[0][0].height).next_to(count, DOWN, buff=0.45).align_to(rib, LEFT)
        R = reward(fake)
        frew = VGroup(RoundedRectangle(corner_radius=0.1, width=1.5, height=TILE_H).set_stroke(WRONG, 2.5)
                      .set_fill(WRONG, 0.2), MathTex(rf"R = {num(R)}", font_size=30, color=WRONG))
        frew[1].move_to(frew[0])
        frew.match_height(self.rew[0]).next_to(frib, RIGHT, buff=0.3)
        self.play(FadeOut(arrows), FadeOut(count))
        frib.shift((count.get_top()[1] - frib.get_top()[1]) * UP)
        frew.match_y(frib)
        head = Text("a later rollout:", font_size=20, color=MUTED).next_to(frib, UP, buff=0.12).align_to(frib, LEFT)
        self.play(FadeIn(head), LaggedStart(*[FadeIn(t, shift=0.2 * RIGHT) for g in frib for t in g],
                                            lag_ratio=0.04), run_time=2.5)
        cap.say("unmasked, the model learns to fake tool outputs", color=WRONG)
        start = len(digits(A)) + 1 + len(digits(B % 10))           # after the model's "16+7"
        fake_tiles = VGroup(*frib[1][start:start + len(digits(true_obs() + FAKE_OFFSET)) + 2])
        brace = Brace(fake_tiles, DOWN, color=WRONG)
        note = Text(f"no <call>, no calculator: it wrote {true_obs() + FAKE_OFFSET}, the true sum is {true_obs()}",
                    font_size=22, color=WRONG)
        note.next_to(brace, DOWN, buff=0.1)
        if note.get_left()[0] < -config.frame_width / 2 + 0.3:
            note.to_edge(LEFT, buff=0.3)
        self.play(GrowFromCenter(brace), FadeIn(note), Indicate(fake_tiles, color=WRONG))
        self.wait(1.5)
        self.play(FadeIn(frew, shift=0.2 * LEFT))
        cap.say(f"final answer {final_answer(fake)} instead of {A + B}: a hallucinated tool result", color=WRONG,
                wait=2.5)
