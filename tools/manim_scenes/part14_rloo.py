"""Part 14 · RLOO and REINFORCE++: critic-free baselines (leave-one-out, then global batch normalisation)."""
import numpy as np
from style import *


def rloo_advantages(R):
    R = np.asarray(R, float); K = len(R)
    return R - (R.sum() - R) / (K - 1)


def batch_normalised(R_batch):
    """REINFORCE++-style: one mean/std across the whole batch (all prompts), not per group."""
    R = np.asarray(R_batch, float); return (R - R.mean()) / (R.std() + 1e-8)


def loo_baselines(R):
    """The baseline each sample gets in RLOO: R_i − A_i, i.e. the mean of the other K − 1 rewards."""
    R = np.asarray(R, float)
    return R - rloo_advantages(R)


REWARDS = [0.9, 0.2, 0.6, 0.1]          # act 2: reward-model scores for K = 4 samples of one prompt
K = len(REWARDS)
BATCH = [[0.9, 0.7, 0.8, 0.5],          # act 3: three prompts x four samples in one batch
         [0.4, 0.1, 0.3, 0.2],
         [0.0, 0.2, 0.0, 0.1]]
PANEL_X = 4.7      # x-centre of the right-hand panel (formula, readouts)


def signed(v):
    return f"{v:+.2f}".replace("-", "−")


def num(v):
    return f"{v:.2f}"


class Part14(Scene):
    def construct(self):
        self.title = title_card(self, 14, "RLOO and REINFORCE++: critic-free baselines")
        self.cap = Captioner(self, self.title)
        self.act1(); clear_act(self, keep=[self.title])
        self.act2(); clear_act(self, keep=[self.title])
        self.act3()
        takeaway(self, "sample more, and let the samples be each other's critic")
        roadmap_outro(self, 14)

    # ------------------------------------------------------------------ act 1: the critic only gave a baseline
    def act1(self):
        cap = self.cap
        specs = [("policy", "trained", POLICY_C), ("reference", "frozen", REF_C),
                 ("reward model", "frozen", MEAN_C), ("value model", "trained", VALUE_C)]
        boxes = VGroup()
        for name, role, col in specs:
            rect = RoundedRectangle(corner_radius=0.15, width=3.0, height=1.3).set_stroke(col, 3).set_fill(col, 0.12)
            lab = Text(name, font_size=26, color=col).move_to(rect.get_center() + 0.18 * UP)
            sub = Text(role, font_size=20, color=MUTED).move_to(rect.get_center() + 0.3 * DOWN)
            boxes.add(VGroup(rect, lab, sub))
        boxes.arrange(RIGHT, buff=0.3).move_to(0.9 * UP)
        cap.say("recap of part 12: ppo-rlhf keeps four models in memory")
        self.play(LaggedStart(*[FadeIn(b, shift=0.3 * UP) for b in boxes], lag_ratio=0.2), run_time=1.6)
        self.wait(1)

        rm_note = VGroup(Text("one number R,", font_size=22, color=MEAN_C),
                         Text("at the very end", font_size=22, color=MEAN_C)).arrange(DOWN, buff=0.1)
        rm_note.next_to(boxes[2], DOWN, buff=0.35)
        v_note = VGroup(Text("V(s): a baseline", font_size=22, color=VALUE_C),
                        Text("for that R", font_size=22, color=VALUE_C)).arrange(DOWN, buff=0.1)
        v_note.next_to(boxes[3], DOWN, buff=0.35)
        self.play(Indicate(boxes[2][0], color=MEAN_C), FadeIn(rm_note, shift=0.2 * DOWN))
        self.play(Indicate(boxes[3][0], color=VALUE_C), FadeIn(v_note, shift=0.2 * DOWN))
        cap.say("with one end-of-answer reward, the critic's only job was a baseline", wait=2)

        grad = MathTex(r"\nabla_\theta J = \mathbb{E}\big[\,(", "R", "-", "b", r")\,\nabla_\theta \log \pi_\theta(y|x)\,\big]",
                       font_size=44).move_to(2.2 * DOWN)
        grad[1].set_color(CORRECT); grad[3].set_color(MEAN_C)
        self.play(Write(grad))
        self.wait(1)

        cross = VGroup(Line(UL, DR), Line(UR, DL)).set_stroke(WRONG, 6)
        cross.scale_to_fit_width(1.0).move_to(boxes[3][0])
        cap.say("a whole trained network, just to produce b")
        self.play(Create(cross))
        self.play(FadeOut(boxes[3]), FadeOut(cross), FadeOut(v_note), FadeOut(rm_note))
        self.play(boxes[:3].animate.move_to(0.9 * UP))
        cap.say("drop it: can the samples themselves supply b?", color=MEAN_C)
        self.play(Indicate(grad[3], color=MEAN_C, scale_factor=1.5))
        self.wait(1.2)

    # ------------------------------------------------------------------ act 2: leave one out
    def act2(self):
        cap = self.cap
        R = np.asarray(REWARDS, float)
        base, adv = loo_baselines(R), rloo_advantages(R)

        ax = Axes(x_range=[0, K + 1, 1], y_range=[-0.6, 1.0, 0.2], x_length=7.0, y_length=4.6,
                  tips=False, axis_config={"color": REF_C},
                  y_axis_config={"numbers_to_include": [-0.4, 0, 0.4, 0.8], "font_size": 22,
                                 "decimal_number_config": {"num_decimal_places": 1}},
                  x_axis_config={"include_ticks": False}).shift(0.1 * DOWN + 1.9 * LEFT)
        ylab = Text("reward", font_size=22, color=REF_C).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.5)

        cards = VGroup()
        for i, r in enumerate(R):
            rect = RoundedRectangle(corner_radius=0.1, width=1.25, height=0.75).set_stroke(MUTED, 2)
            txt = VGroup(MathTex(f"y_{i + 1}", font_size=28), MathTex(f"R={num(r)}", font_size=24)
                         ).arrange(DOWN, buff=0.06).move_to(rect)
            cards.add(VGroup(rect, txt).move_to(ax.c2p(i + 1, -0.6) + 0.5 * DOWN))

        cap.say(f"sample K = {K} answers to the same prompt and score each one")
        self.play(Create(ax), FadeIn(ylab))
        self.play(LaggedStart(*[FadeIn(c, shift=0.2 * UP) for c in cards], lag_ratio=0.15))
        rbars = VGroup(*[bar(ax, i + 1, R[i], MUTED) for i in range(K)])
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in rbars], lag_ratio=0.1))

        formula = MathTex(r"A_i = ", "R_i", "-", r"\frac{1}{K-1}\sum_{j\neq i} R_j", font_size=40)
        formula[1].set_color(CORRECT); formula[3].set_color(MEAN_C)
        formula.move_to(RIGHT * PANEL_X + 1.3 * UP)
        self.play(Write(formula))
        cap.say("each sample's baseline is the mean of the others: free and unbiased", color=MEAN_C)

        ticks, segs, readout = VGroup(), VGroup(), None
        for i in range(K):
            others = [j for j in range(K) if j != i]
            col = CORRECT if adv[i] > 0 else WRONG
            self.play(cards[i][0].animate.set_stroke(WHITE, 4),
                      *[cards[j][0].animate.set_stroke(MEAN_C, 3) for j in others],
                      *[rbars[j].animate.set_fill(MEAN_C, 0.55).set_stroke(MEAN_C) for j in others],
                      run_time=0.6)
            # the other three average into a yellow tick over sample i
            ghosts = VGroup(*[Line(ax.c2p(j + 1 - 0.32, R[j]), ax.c2p(j + 1 + 0.32, R[j]), color=MEAN_C, stroke_width=4)
                              for j in others])
            tick = Line(ax.c2p(i + 1 - 0.42, base[i]), ax.c2p(i + 1 + 0.42, base[i]), color=MEAN_C, stroke_width=6)
            new_read = VGroup(
                MathTex(rf"b_{i + 1} = {num(base[i])}", font_size=32, color=MEAN_C),
                MathTex(rf"A_{i + 1} = {num(R[i])} - {num(base[i])} = {adv[i]:+.2f}", font_size=32),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(formula, DOWN, buff=0.6)
            new_read[1][0][-5:].set_color(col)
            self.play(ReplacementTransform(ghosts, tick), FadeIn(new_read[0]),
                      *([FadeOut(readout)] if readout is not None else []), run_time=1.0)
            lo, hi = sorted([base[i], R[i]])
            seg = Polygon(ax.c2p(i + 0.86, lo), ax.c2p(i + 1.14, lo), ax.c2p(i + 1.14, hi), ax.c2p(i + 0.86, hi)
                          ).set_stroke(col, 2).set_fill(col, 0.8)
            self.play(GrowFromEdge(seg, DOWN if R[i] > base[i] else UP), FadeIn(new_read[1]), run_time=0.8)
            self.wait(0.6)
            self.play(cards[i][0].animate.set_stroke(MUTED, 2),
                      *[cards[j][0].animate.set_stroke(MUTED, 2) for j in others],
                      *[rbars[j].animate.set_fill(MUTED, 0.75).set_stroke(MUTED) for j in others],
                      run_time=0.4)
            ticks.add(tick); segs.add(seg); readout = new_read

        # drop every advantage segment so it stands on zero
        cap.say("drop each gap onto zero: those are the advantages")
        outlines = VGroup(*[bar(ax, i + 1, R[i], MUTED, outline=True) for i in range(K)])
        abars = VGroup(*[bar(ax, i + 1, adv[i], CORRECT if adv[i] > 0 else WRONG) for i in range(K)])
        ylab2 = Text("reward / advantage", font_size=22, color=REF_C).rotate(PI / 2).move_to(ylab)
        self.play(FadeOut(readout), ReplacementTransform(rbars, outlines), FadeOut(ticks),
                  ReplacementTransform(segs, abars), ReplacementTransform(ylab, ylab2), run_time=1.6)
        labels = VGroup(*[value_label(ax, i + 1, adv[i], font_size=24) for i in range(K)])
        self.play(LaggedStart(*[FadeIn(l) for l in labels], lag_ratio=0.1))
        total = MathTex(rf"\textstyle\sum_i A_i = {abs(adv.sum()):.2f}", font_size=36).next_to(formula, DOWN, buff=0.7)
        self.play(FadeIn(total))
        cap.say("better than the others → pushed up, worse → pushed down", wait=2.5)

    # ------------------------------------------------------------------ act 3: reinforce++ batch normalisation
    def act3(self):
        cap = self.cap
        R = np.asarray(BATCH, float).reshape(-1)
        P, S = len(BATCH), len(BATCH[0])
        xs = [g * (S + 1) + j + 1 for g in range(P) for j in range(S)]
        m, s = R.mean(), R.std()
        z = batch_normalised(R)

        ax = Axes(x_range=[0, P * (S + 1), 1], y_range=[-2.0, 2.0, 0.5], x_length=7.2, y_length=4.6,
                  tips=False, axis_config={"color": REF_C},
                  y_axis_config={"numbers_to_include": [-2, -1, 0, 1, 2], "font_size": 22},
                  x_axis_config={"include_ticks": False}).shift(0.35 * DOWN + 1.55 * LEFT)
        t = ValueTracker(0)
        val = lambda k: (1 - t.get_value()) * R[k] + t.get_value() * z[k]
        ylab = always_redraw(lambda: Text("reward" if t.get_value() < 0.5 else "advantage", font_size=22,
                                          color=REF_C).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.45))
        glabs = VGroup(*[Text(f"prompt {g + 1}", font_size=22, color=MUTED)
                         .move_to(ax.c2p(g * (S + 1) + (S + 1) / 2, -2.0) + 0.35 * DOWN) for g in range(P)])
        seps = VGroup(*[DashedLine(ax.c2p(g * (S + 1), -2.0), ax.c2p(g * (S + 1), 2.0), color=GREY_D, stroke_width=2)
                        for g in range(1, P)])

        cap.say(f"reinforce++: {P} prompts × {S} samples in one batch")
        self.play(Create(ax), FadeIn(glabs), FadeIn(seps), FadeIn(ylab))
        static = VGroup(*[bar(ax, xs[k], R[k], MUTED, width=0.7) for k in range(len(R))])
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in static], lag_ratio=0.05))
        colr = lambda k: CORRECT if z[k] > 0 else WRONG
        bars = always_redraw(lambda: VGroup(*[
            bar(ax, xs[k], val(k), interpolate_color(ManimColor(MUTED), ManimColor(colr(k)), t.get_value()), width=0.7)
            for k in range(len(R))]))
        self.remove(*static); self.add(bars)

        formula = MathTex(r"A = \frac{ {{R}} - {{\mathrm{mean}_{\text{batch}}}} }{ {{\mathrm{std}_{\text{batch}}}} }",
                          font_size=40)
        formula[1].set_color(CORRECT); formula[3].set_color(MEAN_C); formula[5].set_color(STD_C)
        formula.move_to(RIGHT * PANEL_X + 1.5 * UP)
        self.play(Write(formula))

        y_mean = lambda: (1 - t.get_value()) * m
        half = lambda: (1 - t.get_value()) * s + t.get_value() * 1.0
        x0, x1 = 0.3, P * (S + 1) - 0.3
        mean_line = always_redraw(lambda: DashedLine(ax.c2p(x0, y_mean()), ax.c2p(x1, y_mean()),
                                                     color=MEAN_C, stroke_width=3))
        band = always_redraw(lambda: Rectangle(
            width=ax.c2p(x1, 0)[0] - ax.c2p(x0, 0)[0],
            height=abs(ax.c2p(0, 2 * half())[1] - ax.c2p(0, 0)[1]),
        ).set_stroke(width=0).set_fill(STD_C, 0.15).move_to(ax.c2p((x0 + x1) / 2, y_mean())))
        cap.say("reinforce++ normalises across the whole batch instead of per prompt", color=MEAN_C)
        readout = VGroup(
            MathTex(rf"\mathrm{{mean}}_{{\text{{batch}}}} = {num(m)}", font_size=32, color=MEAN_C),
            MathTex(rf"\mathrm{{std}}_{{\text{{batch}}}} = {num(s)}", font_size=32, color=STD_C),
            Text(f"shared by all {len(R)} samples", font_size=20, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(formula, DOWN, buff=0.5).align_to(formula, LEFT)
        self.play(FadeIn(band), Create(mean_line), FadeIn(readout[0]), FadeIn(readout[1]))
        self.play(FadeIn(readout[2]))
        self.wait(1.5)

        cap.say("subtract the batch mean, divide by the batch std", color=STD_C)
        self.play(t.animate.set_value(1), run_time=2.5)
        self.wait(1)
        note = VGroup(
            Text("in the notebook: pooled", font_size=20, color=MUTED),
            Text("over every token's", font_size=20, color=MUTED),
            Text("reward-to-go in the batch", font_size=20, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).next_to(readout, DOWN, aligned_edge=LEFT, buff=0.5)
        cap.say("it keeps ppo's clipped token-level objective, minus the critic")
        self.play(FadeIn(note))
        self.wait(2.5)
        bars.clear_updaters(); ylab.clear_updaters(); mean_line.clear_updaters(); band.clear_updaters()
