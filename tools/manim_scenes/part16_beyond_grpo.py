"""Part 16 · After GRPO: the 2025–26 landscape as three dials (advantage, ratio & clip, aggregation)."""
import numpy as np
from style import *


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


def group_adv(R, normalize=True, eps=1e-4):
    """Local copy of Part 15's helper: the notebook's group advantages for one group (unbiased std)."""
    R = np.asarray(R, float)
    a = R - R.mean()
    return a / (R.std(ddof=1) + eps) if normalize else a


def ppo_grad_weight(ratios, lo, hi, A=1.0):
    """Coefficient on ∇log π of min(ρA, clip(ρ)A): ρ where the unclipped branch is active, 0 where clipped."""
    r = np.asarray(ratios, float)
    active = r <= hi if A > 0 else r >= lo
    return np.where(active, r, 0.0)


def cispo_grad_weight(ratios, lo, hi):
    """CISPO: sg(clip(ρ, lo, hi)) multiplies ∇log π for every token, so none is silenced."""
    return np.clip(np.asarray(ratios, float), lo, hi)


def eps_of(bounds):
    lo, hi = bounds
    return float(f"{1 - lo:.3g}"), float(f"{hi - 1:.3g}")


# The notebook's VARIANTS (RLConfig) as dial settings. eps = (ε_low, ε_high) on the ratio.
CISPO_EPS = (0.2, 5.0)                 # RLConfig("CISPO", clip_high=5.0), clip_low keeps its 0.2 default
GSPO_PAPER_EPS = (3e-4, 4e-4)          # the paper's values for long sequences (the notebook uses 10x)
VARIANTS = {
    "GRPO":       dict(adv="group_std",  ratio="token",    eps=eps_of(clip_bounds("GRPO")), agg="seq",   beta=0.04, dyn=False),
    "Dr. GRPO":   dict(adv="group_mean", ratio="token",    eps=eps_of(clip_bounds("GRPO")), agg="token", beta=0.04, dyn=False),
    "DAPO-style": dict(adv="group_mean", ratio="token",    eps=eps_of(clip_bounds("DAPO")), agg="token", beta=0.0,  dyn=True),
    "GSPO":       dict(adv="group_std",  ratio="sequence", eps=eps_of(clip_bounds("GSPO")), agg="seq",   beta=0.0,  dyn=False),
    "CISPO":      dict(adv="group_mean", ratio="cispo",    eps=CISPO_EPS,                   agg="token", beta=0.0,  dyn=False),
}
ADV_TEXT = {"group_std": "(R − mean) / std", "group_mean": "R − mean"}
RATIO_TEXT = {"token": "token ratio", "sequence": "sequence ratio", "cispo": "sg(clip ρ) weight"}
AGG_TEXT = {"seq": "1/|y| per answer", "token": "token-level"}
ADV_IDX = {"group_std": 0, "group_mean": 1}
RATIO_IDX = {"token": 0, "sequence": 1, "cispo": 2}
AGG_IDX = {"seq": 0, "token": 1}

LENGTHS = [10, 40]                      # act 2: two wrong answers
G = 8
HARD = [1.0] + [0.0] * (G - 1)          # act 2 mini plot: 1 of 8 correct (Part 15's hard prompt)
DS_GROUPS = [[1, 0, 0, 1, 0, 0, 0, 0], [1] * 8, [0, 1, 1, 0, 1, 0, 1, 1], [0] * 8]
DS_RESAMPLED = {1: [1, 1, 0, 1, 1, 1, 0, 1], 3: [0, 0, 1, 0, 0, 0, 0, 0]}
GSPO_TOKENS = ["so", "17", "+", "14", "is", "31"]
GSPO_TOKEN_RATIOS = [1.15, 0.88, 1.25, 0.93, 1.02, 0.84]    # noisy per-token ratios
CISPO_TOKENS = ["so", "it", "is", "wait", "31", "."]
CISPO_TOKEN_RATIOS = [1.05, 0.97, 1.10, 1.60, 1.02, 0.93]   # the rare reflective token has grown a lot
PANEL_LEFT = 2.85  # left edge of the right-hand dial panel
LEFT_X = -2.2    # x-centre of the left-hand visual


def g1(x):
    """Shortest form, but keep one decimal on whole numbers (5.0, not 5)."""
    s = f"{x:g}"
    return s if "." in s or "e" in s else s + ".0"


def fmt_eps(eps):
    lo, hi = eps
    return f"ε = ±{g1(lo)}" if lo == hi else f"ε = {g1(lo)} / {g1(hi)}"


def dial_texts(v):
    return ([ADV_TEXT[v["adv"]]], [RATIO_TEXT[v["ratio"]], fmt_eps(v["eps"])], [AGG_TEXT[v["agg"]]])


def extras_text(v):
    s = f"β = {v['beta']:g}"
    return s + ("  ·  dynamic sampling" if v["dyn"] else "")


def num(v, d=2):
    return f"{v:.{d}f}".replace("-", "−")


class Dial(VGroup):
    """A knob with a pointer, a name and a one- or two-line setting to its right."""

    def __init__(self, name, n_opts, lines, idx=0):
        super().__init__()
        self.n = n_opts
        self.knob = Circle(radius=0.34).set_stroke(MUTED, 3).set_fill(BLACK, 1)
        self.ticks = VGroup(*[Dot(self.knob.get_center() + 0.46 * np.array([np.cos(a), np.sin(a), 0]),
                                  radius=0.035, color=MUTED) for a in self.angles()])
        self.angle = self.angles()[idx]
        self.pointer = Line(ORIGIN, 0.27 * RIGHT, stroke_width=5, color=WHITE).rotate(
            self.angle, about_point=ORIGIN)
        self.name = Text(name, font_size=20, color=MUTED)
        self.txt = self.make_txt(lines, WHITE)
        self.add(self.knob, self.ticks, self.pointer, self.name, self.txt)
        self.layout()

    def angles(self):
        return [PI / 2 + 0.9 - i * 1.8 / max(self.n - 1, 1) for i in range(self.n)]

    def make_txt(self, lines, color):
        return VGroup(*[Text(l, font_size=24 if i == 0 else 22, color=color) for i, l in enumerate(lines)]
                      ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

    def layout(self):
        self.name.next_to(self.knob, RIGHT, buff=0.35).align_to(self.knob, UP).shift(0.05 * UP)
        self.txt.next_to(self.name, DOWN, aligned_edge=LEFT, buff=0.12)

    def turn(self, idx, lines, color):
        """Animations that rotate the pointer to option idx and swap in the new setting text."""
        new_angle = self.angles()[idx]
        new = self.make_txt(lines, color).next_to(self.name, DOWN, aligned_edge=LEFT, buff=0.12)
        anims = [Transform(self.txt, new), self.knob.animate.set_stroke(color if color != WHITE else MUTED)]
        if abs(new_angle - self.angle) > 1e-6:
            anims.append(Rotate(self.pointer, new_angle - self.angle, about_point=self.knob.get_center()))
            self.angle = new_angle
        return anims


class Part16(Scene):
    def construct(self):
        self.title = title_card(self, 16, "After GRPO: the 2025–26 landscape")
        self.cap = Captioner(self, self.title)
        self.act1()
        keep = [self.title, self.dials, self.header, self.extras]
        clear_act(self, keep=keep)
        self.act2()
        clear_act(self, keep=keep)
        self.act3()
        takeaway(self, "advantage, ratio, aggregation: know the three dials and you can read any new paper")
        roadmap_outro(self, 16)

    # ------------------------------------------------------------------ dial panel
    def set_method(self, name, color, focus=(), extra=()):
        """Turn every dial to `name`'s notebook settings; dials that differ from GRPO take `color`."""
        v, base = VARIANTS[name], VARIANTS["GRPO"]
        texts = dial_texts(v)
        diff = [v["adv"] != base["adv"], (v["ratio"], v["eps"]) != (base["ratio"], base["eps"]), v["agg"] != base["agg"]]
        idx = [ADV_IDX[v["adv"]], RATIO_IDX[v["ratio"]], AGG_IDX[v["agg"]]]
        anims = []
        for d, i, t, changed in zip(self.dials, idx, texts, diff):
            anims += d.turn(i, t, color if changed else WHITE)
        new_head = Text(name, font_size=36, color=color).move_to(self.header).align_to(self.header, LEFT)
        new_extra = Text(extras_text(v), font_size=22, color=color if v["beta"] != base["beta"] or v["dyn"] else MUTED
                         ).move_to(self.extras).align_to(self.extras, LEFT)
        self.play(Transform(self.header, new_head), Transform(self.extras, new_extra), *anims, *extra, run_time=1.2)
        for k in focus:
            self.play(Indicate(self.dials[k], color=color, scale_factor=1.08))

    # ------------------------------------------------------------------ act 1: three dials
    def act1(self):
        cap = self.cap
        v = VARIANTS["GRPO"]
        texts = dial_texts(v)
        self.dials = VGroup(
            Dial("advantage", 2, texts[0], ADV_IDX[v["adv"]]),
            Dial("ratio & clip", 3, texts[1], RATIO_IDX[v["ratio"]]),
            Dial("aggregation", 2, texts[2], AGG_IDX[v["agg"]]),
        )
        self.dials.arrange(RIGHT, buff=0.5, aligned_edge=UP)
        self.k = min(1.2, 12.6 / self.dials.width)
        self.dials.scale(self.k).move_to(0.3 * DOWN)
        self.header = Text("GRPO", font_size=44).next_to(self.dials, UP, buff=0.9)
        self.extras = Text(extras_text(v), font_size=26, color=MUTED).next_to(self.dials, DOWN, buff=0.8)

        cap.say("GRPO makes three decisions in its loss")
        self.play(FadeIn(self.header, shift=0.2 * DOWN))
        for d in self.dials:
            self.play(FadeIn(d, shift=0.2 * UP), run_time=0.8)
            self.wait(0.6)
        self.play(FadeIn(self.extras))
        cap.say("every successor is a different setting of three dials", color=MEAN_C, wait=2)

        # Move the dial panel to the right; it stays there for the rest of the video.
        self.play(
            self.dials.animate.scale(1 / self.k).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
                .move_to(0.35 * DOWN).align_to(PANEL_LEFT * RIGHT, LEFT),
            self.header.animate.scale(36 / 44).move_to(1.85 * UP).align_to(PANEL_LEFT * RIGHT, LEFT),
            self.extras.animate.scale(22 / 26).move_to(3.0 * DOWN).align_to(PANEL_LEFT * RIGHT, LEFT),
            run_time=1.3,
        )
        self.divider = Line(2.55 * RIGHT + 2.2 * UP, 2.55 * RIGHT + 3.4 * DOWN, color=GREY_D, stroke_width=2)
        self.play(Create(self.divider))

    # ------------------------------------------------------------------ act 2: Dr. GRPO
    def act2(self):
        cap = self.cap
        seq_w, tok_w = token_weights(LENGTHS, "seq"), token_weights(LENGTHS, "token")
        wmax = max(seq_w.max(), tok_w.max())
        t = ValueTracker(0)                     # 0: per-answer (seq), 1: token-level
        w = lambda j: (1 - t.get_value()) * seq_w[j] + t.get_value() * tok_w[j]
        side, gap = 0.17, 0.035
        ys = [1.1, -1.1]

        def strip(j):
            sq = VGroup(*[Square(side).set_stroke(WRONG, 1).set_fill(WRONG, 0.08 + 0.92 * w(j) / wmax)
                          for _ in range(LENGTHS[j])]).arrange(RIGHT, buff=gap)
            return sq.move_to([LEFT_X, ys[j], 0]).align_to([-6.6, 0, 0], LEFT)

        strips = always_redraw(lambda: VGroup(strip(0), strip(1)))
        heads = VGroup(*[Text(f"wrong answer, |y| = {L}", font_size=24, color=WRONG)
                         .next_to([-6.6, ys[j] + 0.5, 0], RIGHT, buff=0).align_to([-6.6, 0, 0], LEFT)
                         for j, L in enumerate(LENGTHS)])

        def readout(j):
            row = VGroup(Text("per-token penalty", font_size=22, color=MUTED),
                         DecimalNumber(w(j), num_decimal_places=4, font_size=28),
                         Text("  whole answer", font_size=22, color=MUTED),
                         DecimalNumber(w(j) * LENGTHS[j], num_decimal_places=2, font_size=28)).arrange(RIGHT, buff=0.15)
            return row.next_to([-6.6, ys[j] - 0.5, 0], RIGHT, buff=0)

        reads = always_redraw(lambda: VGroup(readout(0), readout(1)))
        cap.say(f"two wrong answers, {LENGTHS[0]} and {LENGTHS[1]} tokens: what does each token pay?")
        self.play(FadeIn(heads), FadeIn(strips), FadeIn(reads))
        self.play(Indicate(self.dials[2], color=WRONG, scale_factor=1.08))
        self.wait(1)
        cap.say("1/|y| per answer: long wrong answers are cheaper per token → rambling", color=WRONG, wait=2.5)

        self.set_method("Dr. GRPO", MEAN_C, focus=[2])
        cap.say(f"token-level: every token pays 1/Σ|y| = 1/{sum(LENGTHS)}, length buys nothing", color=MEAN_C)
        self.play(t.animate.set_value(1), run_time=2)
        self.wait(2)
        strips.clear_updaters(); reads.clear_updaters()
        self.play(FadeOut(VGroup(strips, reads, heads)))
        self.remove(strips, reads, heads)

        # Mini version of Part 15's plot: the hard prompt's advantages with and without ÷ std.
        std_adv, mean_adv = group_adv(HARD), group_adv(HARD, normalize=False)
        ax = Axes(x_range=[0, 2 * G + 2, 1], y_range=[-1, 2.75, 1], x_length=7.0, y_length=4.2, tips=False,
                  axis_config={"color": REF_C}, x_axis_config={"include_ticks": False},
                  y_axis_config={"numbers_to_include": [-1, 0, 1, 2], "font_size": 22}).move_to([LEFT_X, -0.55, 0])
        ylab = Text("advantage", font_size=22, color=REF_C).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.4)
        col = lambda r: CORRECT if r else WRONG
        g1 = VGroup(*[bar(ax, i + 1, a, col(HARD[i]), width=0.7) for i, a in enumerate(std_adv)])
        g2 = VGroup(*[bar(ax, G + 2 + i, a, col(HARD[i]), width=0.7) for i, a in enumerate(mean_adv)])
        l1 = Text("÷ std (GRPO)", font_size=22, color=STD_C).move_to(ax.c2p((G + 1) / 2, -1) + 0.35 * DOWN)
        l2 = Text("mean only (Dr. GRPO)", font_size=22, color=MEAN_C).move_to(ax.c2p(G + 1 + (G + 1) / 2, -1) + 0.35 * DOWN)
        v1, v2 = value_label(ax, 1, std_adv[0], 24), value_label(ax, G + 2, mean_adv[0], 24)
        cap.say(f"a hard prompt, 1 of {G} correct (Part 15's plot)")
        self.play(Create(ax), FadeIn(ylab))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in g1], lag_ratio=0.06), FadeIn(l1))
        self.play(FadeIn(v1))
        cap.say(f"÷ std blows the lone success up to {num(std_adv[0])}: a hidden difficulty weight",
                color=STD_C, wait=1.5)
        self.play(Indicate(self.dials[0], color=MEAN_C, scale_factor=1.08))
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in g2], lag_ratio=0.06), FadeIn(l2))
        self.play(FadeIn(v2))
        cap.say(f"Dr. GRPO drops it: A = R − mean gives just {num(mean_adv[0])}", color=MEAN_C, wait=2.5)

    # ------------------------------------------------------------------ act 3: DAPO, GSPO, CISPO
    def act3(self):
        self.dapo(); self.clear_left()
        self.gspo(); self.clear_left()
        self.cispo()

    def clear_left(self):
        clear_act(self, keep=[self.title, self.dials, self.header, self.extras, self.divider])

    def dapo(self):
        cap = self.cap
        (lo, hi0), (_, hi1) = clip_bounds("GRPO"), clip_bounds("DAPO")
        nl = NumberLine(x_range=[0.6, 1.5, 0.1], length=6.8, color=REF_C, include_numbers=False
                        ).move_to([LEFT_X, 1.2, 0])
        hi = ValueTracker(hi0)
        band = always_redraw(lambda: Rectangle(width=nl.n2p(hi.get_value())[0] - nl.n2p(lo)[0], height=0.45)
                             .set_stroke(width=0).set_fill(POLICY_C, 0.3)
                             .move_to((nl.n2p(lo) + nl.n2p(hi.get_value())) / 2))
        lo_lab = Text(f"{lo:g}", font_size=22).next_to(nl.n2p(lo), DOWN, buff=0.3)
        hi_lab = always_redraw(lambda: DecimalNumber(hi.get_value(), num_decimal_places=2, font_size=26,
                                                     color=POLICY_C).next_to(nl.n2p(hi.get_value()), DOWN, buff=0.3))
        rho = Text("ratio ρ", font_size=22, color=MUTED).next_to(nl, UP, aligned_edge=LEFT, buff=0.35)
        self.set_method("DAPO-style", POLICY_C, focus=[1],
                        extra=[Create(nl), FadeIn(band), FadeIn(lo_lab), FadeIn(hi_lab), FadeIn(rho)])
        self.play(hi.animate.set_value(hi1), run_time=1.5)
        cap.say(f"clip-higher: ε_high {hi1 - 1:.2f} lets rare tokens grow, entropy stays up", color=POLICY_C, wait=1.5)
        band.clear_updaters(); hi_lab.clear_updaters()

        # Dynamic sampling: all-equal groups carry zero advantage, drop and resample them.
        def row(R, y):
            return VGroup(*[Dot(radius=0.12, color=CORRECT if r else WRONG) for r in R]).arrange(RIGHT, buff=0.22
                          ).move_to([LEFT_X - 0.9, y, 0])
        yrows = [-0.35 - 0.75 * i for i in range(len(DS_GROUPS))]
        rows = [row(R, y) for R, y in zip(DS_GROUPS, yrows)]
        tags = []
        for R, r in zip(DS_GROUPS, rows):
            a = group_adv(R)
            tags.append(Text(f"max |A| = {np.abs(a).max():.2f}", font_size=22,
                             color=MUTED if np.abs(a).max() > 0 else WRONG).next_to(r, RIGHT, buff=0.5))
        cap.say("dynamic sampling: all-equal groups have zero advantage", color=POLICY_C)
        self.play(LaggedStart(*[AnimationGroup(FadeIn(r), FadeIn(tg)) for r, tg in zip(rows, tags)], lag_ratio=0.2))
        dead = [i for i, R in enumerate(DS_GROUPS) if np.allclose(group_adv(R), 0)]
        crosses = [Cross(rows[i], stroke_width=4, stroke_color=WRONG) for i in dead]
        self.play(*[Create(c) for c in crosses])
        cap.say("drop them and resample until the batch is informative", color=POLICY_C)
        new_rows, new_tags = [], []
        for i in dead:
            R = DS_RESAMPLED[i]
            new_rows.append(row(R, yrows[i]))
            new_tags.append(Text(f"max |A| = {np.abs(group_adv(R)).max():.2f}", font_size=22, color=MUTED)
                            .next_to(new_rows[-1], RIGHT, buff=0.5))
        self.play(*[FadeOut(VGroup(rows[i], tags[i], c), shift=0.3 * RIGHT) for i, c in zip(dead, crosses)])
        self.play(*[FadeIn(VGroup(r, tg), shift=0.3 * RIGHT) for r, tg in zip(new_rows, new_tags)])
        self.wait(1.5)

    def gspo(self):
        cap = self.cap
        rs = np.array(GSPO_TOKEN_RATIOS)
        s = gspo_ratio(np.log(rs))
        chips = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.1, width=0.85, height=0.6).set_stroke(MUTED, 2),
                                Text(tk, font="Monospace", font_size=24)) for tk in GSPO_TOKENS]
                       ).arrange(RIGHT, buff=0.2).move_to([LEFT_X + 0.6, 0.2, 0])
        rl = VGroup(*[Text(f"{r:.2f}", font_size=24, color=MUTED).next_to(c, UP, buff=0.2) for r, c in zip(rs, chips)])
        rlab = Text("per-token ρ", font_size=22, color=MUTED).next_to(rl, LEFT, buff=0.3)
        self.set_method("GSPO", VALUE_C, focus=[1], extra=[FadeIn(chips), FadeIn(rl), FadeIn(rlab)])
        cap.say("token ratios are noisy single samples, and the noise adds up", wait=1.5)

        formula = MathTex(r"s_i = \exp\Big(\tfrac{1}{|y_i|}\sum_t \log \rho_{i,t}\Big) = ", f"{s:.4f}",
                          font_size=36).move_to([LEFT_X, 1.75, 0])
        formula[1].set_color(VALUE_C)
        cap.say("gspo: one geometric-mean ratio per sequence, every token shares it", color=VALUE_C)
        self.play(Write(formula[0]), ReplacementTransform(rl.copy(), formula[1]), run_time=1.5)
        shared = VGroup(*[Text(f"{s:.4f}", font_size=18, color=VALUE_C).next_to(c, DOWN, buff=0.2) for c in chips])
        slab = Text("shared s", font_size=22, color=VALUE_C).next_to(shared, LEFT, buff=0.3)
        self.play(LaggedStart(*[TransformFromCopy(formula[1], x) for x in shared], lag_ratio=0.1), FadeIn(slab),
                  rl.animate.set_opacity(0.35), rlab.animate.set_opacity(0.35))
        lo, hi = clip_bounds("GSPO")
        kept = lo <= s <= hi
        e_lo, e_hi = eps_of((lo, hi))
        verdict = Text(f"clip band [{lo:.3f}, {hi:.3f}]: whole sequence {'kept' if kept else 'clipped'}",
                       font_size=24, color=VALUE_C).move_to([LEFT_X, -1.3, 0])
        note = VGroup(
            Text(f"notebook: ε = {g1(e_lo)} / {g1(e_hi)}  (3-token answers)", font_size=22, color=MUTED),
            Text(f"paper: ε = {g1(GSPO_PAPER_EPS[0])} / {g1(GSPO_PAPER_EPS[1])}  (long sequences)", font_size=22, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([LEFT_X, -2.3, 0])
        self.play(FadeIn(verdict))
        self.play(FadeIn(note))
        self.wait(2.5)

    def cispo(self):
        cap = self.cap
        self.set_method("GRPO", WHITE)
        rs = np.array(CISPO_TOKEN_RATIOS)
        (plo, phi), (clo, chi) = clip_bounds("GRPO"), (1 - CISPO_EPS[0], 1 + CISPO_EPS[1])
        ppo_g, cis_g = ppo_grad_weight(rs, plo, phi), cispo_grad_weight(rs, clo, chi)
        chips = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.1, width=0.95, height=0.6).set_stroke(MUTED, 2),
                                Text(tk, font="Monospace", font_size=21)) for tk in CISPO_TOKENS]
                       ).arrange(RIGHT, buff=0.2).move_to([LEFT_X + 1.05, 1.4, 0])
        rl = VGroup(*[Text(f"{r:.2f}", font_size=22, color=MUTED).next_to(c, DOWN, buff=0.15) for r, c in zip(rs, chips)])
        rlab = Text("ρ", font_size=26, color=MUTED).next_to(rl, LEFT, buff=0.3)
        self.play(FadeIn(chips), FadeIn(rl), FadeIn(rlab))

        def arrows(gw, y, color):
            grp = VGroup()
            for g, c in zip(gw, chips):
                x = c.get_center()[0]
                if g > 0:
                    grp.add(Arrow([x, y, 0], [x, y + 0.75 * g, 0], buff=0, color=color, stroke_width=6,
                                  max_tip_length_to_length_ratio=0.3))
                else:
                    grp.add(Text("0", font_size=26, color=WRONG).move_to([x, y + 0.25, 0]))
            return grp

        y1, y2 = -0.95, -2.95
        ppo_a = arrows(ppo_g, y1, REF_C)
        ppo_lab = Text(f"PPO clip ±{eps_of((plo, phi))[1]:g}", font_size=22, color=REF_C
                       ).next_to([chips.get_left()[0], y1 + 0.35, 0], LEFT, buff=0.3)
        cap.say("A > 0: gradient on each token's log-prob under PPO's clip")
        self.play(FadeIn(ppo_lab), LaggedStart(*[GrowArrow(a) if isinstance(a, Arrow) else FadeIn(a)
                                                 for a in ppo_a], lag_ratio=0.1))
        wait_i = int(np.argmax(rs))
        self.play(Indicate(chips[wait_i], color=WRONG), Indicate(ppo_a[wait_i], color=WRONG))
        cap.say(f"\"{CISPO_TOKENS[wait_i]}\" left the band (ρ = {rs[wait_i]:.2f}): its gradient is zeroed",
                color=WRONG, wait=1.5)

        self.set_method("CISPO", CORRECT, focus=[1])
        cis_a = arrows(cis_g, y2, CORRECT)
        cis_lab = Text("CISPO", font_size=22, color=CORRECT).next_to([chips.get_left()[0], y2 + 0.35, 0], LEFT, buff=0.3)
        self.play(FadeIn(cis_lab), LaggedStart(*[GrowArrow(a) for a in cis_a], lag_ratio=0.1))
        cap.say("cispo: clip the weight, stop its gradient, keep every token's gradient", color=CORRECT, wait=2.5)
