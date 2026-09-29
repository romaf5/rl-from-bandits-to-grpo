"""Part 15 · GRPO: group-relative advantages (port of grpo-visuals/grpo_advantages.py)."""
import numpy as np
from style import *


def group_adv(R, normalize=True, eps=1e-4):
    """Notebook's group_advantages for one group (torch .std() is unbiased: ddof=1)."""
    R = np.asarray(R, float)
    a = R - R.mean()
    return a / (R.std(ddof=1) + eps) if normalize else a


def rewards_with(k, G=8):
    return [1.0] * k + [0.0] * (G - k)


G = 8
PANEL_X = 4.7  # x-centre of the right-hand panel (formula, readouts, legend)
ANSWERS = ["31", "41", "31", "21", "30", "31", "13", "32"]  # group sampled for "14 + 17 = ?"


def signed(v):
    return f"{v:+.2f}".replace("-", "−")


class Part15(Scene):
    def construct(self):
        self.title = title_card(self, 15, "GRPO: group-relative advantages")
        self.cap = Captioner(self, self.title)
        self.act1()
        clear_act(self, keep=[self.title, self.ax, self.card_labels])
        self.act2()
        clear_act(self, keep=[self.title, self.ax, self.card_labels, self.ylab, self.bars])
        self.act3()
        takeaway(self, "the group is its own critic, and ÷std turns reward into a difficulty weighting")
        roadmap_outro(self, 15)

    # ------------------------------------------------------------------ act 1: sample and grade
    def act1(self):
        cap = self.cap
        prompt = MathTex(r"14 + 17 = \,?", font_size=56).move_to(1.5 * UP)
        self.play(FadeIn(prompt, shift=DOWN))
        cap.say(f"the policy samples a group of G = {G} answers")

        self.R = [1.0 if a == "31" else 0.0 for a in ANSWERS]
        cards = VGroup(*[
            VGroup(RoundedRectangle(corner_radius=0.12, width=1.2, height=0.8).set_stroke(MUTED, 2),
                   Text(a, font="Monospace", font_size=32))
            for a in ANSWERS
        ]).arrange(RIGHT, buff=0.3).next_to(prompt, DOWN, buff=0.6)
        self.play(LaggedStart(*[FadeIn(c, shift=0.3 * DOWN) for c in cards], lag_ratio=0.12))

        marks = VGroup()
        for c, r in zip(cards, self.R):
            col = CORRECT if r else WRONG
            m = MathTex(r"\checkmark" if r else r"\times", color=col, font_size=44)
            rl = MathTex(f"R={int(r)}", font_size=30, color=col)
            marks.add(VGroup(m, rl).arrange(DOWN, buff=0.15).next_to(c, DOWN, buff=0.25))
        cap.say("a rule-based verifier scores each one")
        self.play(LaggedStart(*[AnimationGroup(FadeIn(m, scale=1.5), c[0].animate.set_stroke(m[0].get_color()))
                                for m, c in zip(marks, cards)], lag_ratio=0.1))
        self.wait(1.5)

        # Collapse the cards into labels under a bar chart.
        self.ax = Axes(x_range=[0, G + 0.6, 1], y_range=[-2.75, 2.75, 1], x_length=8.4, y_length=5.2,
                       tips=False, axis_config={"color": REF_C},
                       y_axis_config={"numbers_to_include": [-2, -1, 0, 1, 2], "font_size": 24},
                       x_axis_config={"include_ticks": False}).shift(0.35 * DOWN + 1.45 * LEFT)
        ax = self.ax
        self.card_labels = VGroup(*[
            c[1].copy().scale(0.7).set_color(CORRECT if r else WRONG) for c, r in zip(cards, self.R)
        ])
        for i, lab in enumerate(self.card_labels):
            lab.move_to(ax.c2p(i + 1, -2.75) + 0.3 * DOWN)
        self.play(
            FadeOut(prompt), FadeOut(marks), FadeOut(VGroup(*[c[0] for c in cards])),
            ReplacementTransform(VGroup(*[c[1] for c in cards]), self.card_labels), run_time=1.2,
        )
        self.play(Create(ax), run_time=1.2)

    # ------------------------------------------------------------------ act 2: subtract mean, divide by std
    def act2(self):
        ax, R, cap = self.ax, self.R, self.cap
        m, s = np.mean(R), np.std(R, ddof=1)
        t = ValueTracker(0)   # 0 -> 1: subtract the mean
        u = ValueTracker(0)   # 0 -> 1: divide by the std
        scale = lambda: 1 + u.get_value() * (s - 1)
        val = lambda i: (R[i] - t.get_value() * m) / scale()

        self.ylab = always_redraw(lambda: Text(
            "reward" if t.get_value() < 0.5 else ("R − mean" if u.get_value() < 0.5 else "advantage"),
            font_size=24, color=REF_C).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.45))
        bars = always_redraw(lambda: VGroup(*[bar(ax, i + 1, val(i), CORRECT if R[i] else WRONG) for i in range(G)]))
        bars_static = VGroup(*[bar(ax, i + 1, R[i], CORRECT if R[i] else WRONG) for i in range(G)])
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars_static], lag_ratio=0.08), FadeIn(self.ylab))
        self.remove(*bars_static)
        self.add(bars)

        formula = MathTex(r"\hat{A}_i = \frac{ {{R_i}} - {{\mathrm{mean}(R)}} }{ {{\mathrm{std}(R)}} }", font_size=40)
        formula[1].set_color(CORRECT); formula[3].set_color(MEAN_C); formula[5].set_color(STD_C)
        formula.move_to(RIGHT * PANEL_X + 1.8 * UP)
        self.play(Write(formula))

        # Mean: the group's own estimate of how well the model does on this prompt (replaces the critic).
        y_mean = lambda: (1 - t.get_value()) * m / scale()
        mean_line = always_redraw(lambda: DashedLine(ax.c2p(0.4, y_mean()), ax.c2p(G + 0.6, y_mean()),
                                                     color=MEAN_C, stroke_width=3))
        mean_lab = always_redraw(lambda: MathTex(rf"\mathrm{{mean}} = {m:.3f}", font_size=28, color=MEAN_C)
                                 .next_to(ax.c2p(G + 0.6, y_mean()), RIGHT, buff=0.15))
        cap.say("the group mean is a free baseline: no critic network needed", color=MEAN_C)
        self.play(Create(mean_line), FadeIn(mean_lab), Indicate(formula[3], color=MEAN_C))
        self.wait(1.5)

        # Step 1: subtract the mean.
        cap.say("subtract it: better than the group → positive, worse → negative")
        self.play(Indicate(formula[2:4], color=MEAN_C))
        self.play(t.animate.set_value(1), run_time=2)
        self.wait(1)

        # Step 2: divide by the std (band shows ±1 std around the mean, which becomes ±1).
        band = always_redraw(lambda: Rectangle(
            width=ax.c2p(G + 0.6, 0)[0] - ax.c2p(0.4, 0)[0],
            height=abs(ax.c2p(0, 2 * s / scale())[1] - ax.c2p(0, 0)[1]),
        ).set_stroke(width=0).set_fill(STD_C, 0.15).move_to(ax.c2p((G + 1) / 2, y_mean())))
        std_lab = always_redraw(lambda: MathTex(r"\pm\,\mathrm{std}" if u.get_value() < 0.99 else r"\pm 1",
                                                font_size=28, color=STD_C)
                                .next_to(ax.c2p(G + 0.6, y_mean() + s / scale()), RIGHT, buff=0.15))
        cap.say(f"divide by the std ({s:.2f}): rewards become z-scores", color=STD_C)
        self.play(FadeIn(band), FadeIn(std_lab), Indicate(formula[5], color=STD_C))
        self.play(u.animate.set_value(1), run_time=2)

        adv = group_adv(R)
        labels = VGroup(*[value_label(ax, i + 1, adv[i]) for i in range(G)])
        self.play(LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.05))
        cap.say("every token of an answer is pushed up or down by its advantage", wait=2.5)

        bars.clear_updaters()
        self.ylab.clear_updaters()
        self.bars = bars

    # ------------------------------------------------------------------ act 3: sweep the difficulty
    def act3(self):
        ax, cap = self.ax, self.cap
        k = ValueTracker(3)

        def interp_adv(normalize):
            kf = k.get_value()
            lo, hi = int(np.floor(kf)), int(np.ceil(kf))
            a, b = group_adv(rewards_with(lo, G), normalize), group_adv(rewards_with(hi, G), normalize)
            return a + (kf - lo) * (b - a)

        def color_of(i):
            # blend while bar i flips from wrong to correct
            return interpolate_color(WRONG, CORRECT, np.clip(k.get_value() - i, 0, 1))

        bars = always_redraw(lambda: VGroup(*[bar(ax, i + 1, v, color_of(i)) for i, v in enumerate(interp_adv(True))]))
        ghosts = always_redraw(lambda: VGroup(*[bar(ax, i + 1, v, WHITE, outline=True)
                                                for i, v in enumerate(interp_adv(False))]))
        new_labels = VGroup(*[Text(f"#{i + 1}", font_size=22, color=REF_C).move_to(l)
                              for i, l in enumerate(self.card_labels)])
        cap.say("same algorithm, different prompt difficulty")
        self.play(FadeOut(self.bars), FadeIn(bars), ReplacementTransform(self.card_labels, new_labels))

        counter = always_redraw(lambda: VGroup(
            Integer(int(round(k.get_value())), font_size=34, color=CORRECT),
            Text(f"of {G} correct", font_size=26),
        ).arrange(RIGHT, buff=0.15).move_to(RIGHT * PANEL_X + 1.8 * UP))
        readout = always_redraw(lambda: VGroup(
            VGroup(Text("correct:", font_size=24, color=CORRECT),
                   DecimalNumber(interp_adv(True)[0] if k.get_value() > 0.5 else 0, include_sign=True,
                                 font_size=28)).arrange(RIGHT, buff=0.15),
            VGroup(Text("wrong:", font_size=24, color=WRONG),
                   DecimalNumber(interp_adv(True)[-1], include_sign=True, font_size=28)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(counter, DOWN, aligned_edge=LEFT))
        self.play(FadeIn(counter), FadeIn(readout))

        # Hard prompt: a lone correct answer gets a huge push.
        self.play(k.animate.set_value(1), run_time=2)
        hard = group_adv(rewards_with(1, G))[0]
        cap.say(f"hard prompt: the lone correct answer gets {signed(hard)}", color=CORRECT, wait=1.5)

        # Compare against the mean-only baseline (Dr. GRPO).
        legend = VGroup(
            VGroup(Square(0.25).set_fill(REF_C, 0.75).set_stroke(REF_C), Text("GRPO (÷ std)", font_size=22)
                   ).arrange(RIGHT, buff=0.15),
            VGroup(Square(0.25).set_stroke(WHITE, 2), Text("mean-only (Dr. GRPO)", font_size=22)
                   ).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to(RIGHT * PANEL_X + 1.6 * DOWN)
        self.play(FadeIn(ghosts), FadeIn(legend))
        mean_only = group_adv(rewards_with(1, G), normalize=False)[0]
        cap.say(f"mean-only would give it just {signed(mean_only)}: ÷std reweights by difficulty", wait=2)

        # Sweep to an easy prompt: roles flip.
        self.play(k.animate.set_value(G - 1), run_time=5, rate_func=linear)
        easy = group_adv(rewards_with(G - 1, G))[-1]
        cap.say(f"easy prompt: now the lone wrong answer gets {signed(easy)}", color=WRONG, wait=2)

        # All correct: std = 0, nothing to learn.
        self.play(k.animate.set_value(G), run_time=1.5)
        dead = group_adv(rewards_with(G, G))
        assert np.allclose(dead, 0)
        std0 = np.std(rewards_with(G, G), ddof=1)
        cap.say(f"{G} of {G} correct: std = {std0:.0f}, every advantage is {abs(dead).max():.0f}, a dead group",
                color=MEAN_C, wait=2.5)
