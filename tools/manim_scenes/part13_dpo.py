"""Part 13 · DPO: the closed-form optimum, Z(x) cancelling, and the loss on preference pairs."""
import numpy as np
from style import *


def implicit_reward(logp, ref_logp, beta=0.1):
    return beta * (np.asarray(logp) - np.asarray(ref_logp))


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    margin = implicit_reward(pi_w, ref_w, beta) - implicit_reward(pi_l, ref_l, beta)
    return np.log1p(np.exp(-margin))             # -log sigmoid(margin)


BETA = 0.1                       # as in the notebook
REF_W, REF_L = -7.0, -8.0        # illustrative sequence log-probs under the frozen reference
END_W, END_L = -4.0, -16.0       # ... and under the policy after DPO: chosen up, rejected down
LOGP_MIN = -18.0                 # log-prob axis floor
PANEL_X = 3.7                    # x-centre of the right-hand panel (sigmoid, readouts)
CHART_X = -3.4                   # x-centre of the left-hand log-prob chart

# Colour roles: pi* / pi_theta = POLICY_C, pi_ref = REF_C, reward = VALUE_C, Z(x) = MEAN_C.
TEX_COLOURS = [("Z(x)", MEAN_C), (r"\pi_{\text{ref}}", REF_C), (r"\pi^*", POLICY_C), ("r(x,", VALUE_C)]


def paint(eq):
    """Colour isolated substrings of a MathTex by the role of the symbol they hold."""
    for part in eq.submobjects:
        for key, col in TEX_COLOURS:
            if part.tex_string.strip().startswith(key):
                part.set_color(col)
                break
    return eq


def sigmoid(m):
    return 1 / (1 + np.exp(-m))


def logp_at(s):
    """Policy log-probs at progress s in [0, 1] (s = 0: policy == reference)."""
    return REF_W + s * (END_W - REF_W), REF_L + s * (END_L - REF_L)


class Part13(Scene):
    def construct(self):
        self.title = title_card(self, 13, "DPO: direct preference optimisation")
        self.cap = Captioner(self, self.title)
        self.act1()
        clear_act(self, keep=[self.title, self.r_eq])
        self.act2()
        clear_act(self, keep=[self.title, self.loss_eq])
        self.act3()
        takeaway(self, "dpo turns rlhf into a classification loss on preference pairs")
        roadmap_outro(self, 13)

    # ------------------------------------------------------------------ act 1: the optimal policy has a formula
    def act1(self):
        cap = self.cap
        objective = MathTex(r"\max_\pi\ \mathbb E_{y\sim\pi}\big[r(x,y)\big]", r"-\ \beta\,\mathrm{KL}(\pi\,\|\,",
                            r"\pi_{\text{ref}}", r")", font_size=40).move_to(1.5 * UP)
        objective[2].set_color(REF_C)
        cap.say("rlhf: maximise reward, but stay close to the reference model")
        self.play(Write(objective))
        beta_lab = MathTex(rf"\beta = {BETA}", font_size=32, color=MUTED).next_to(objective, RIGHT, buff=0.6)
        self.play(FadeIn(beta_lab))
        self.wait(1)

        pi_star = paint(MathTex(
            r"{{\pi^*(y|x)}} = \frac{1}{ {{Z(x)}} }\, {{\pi_{\text{ref}}(y|x)}}\, \exp\!\big( {{r(x,y)}} /\beta \big)",
            font_size=48)).move_to(0.9 * DOWN)
        arrow = Arrow(objective.get_bottom(), pi_star.get_top(), buff=0.25, color=MUTED)
        cap.say("the kl-regularised objective has a closed-form optimum")
        self.play(GrowArrow(arrow), Write(pi_star), run_time=1.6)
        self.wait(1.5)

        z = pi_star.submobjects[3]
        box = SurroundingRectangle(z, color=MEAN_C, buff=0.08)
        cap.say("Z(x) sums over every possible answer: intractable", color=MEAN_C)
        self.play(Create(box), Indicate(z, color=MEAN_C))
        self.wait(1.5)

        # Solve for r: take logs, then rearrange (TransformMatchingTex keeps each symbol in place).
        self.play(FadeOut(objective), FadeOut(beta_lab), FadeOut(arrow), FadeOut(box),
                  pi_star.animate.move_to(0.6 * UP))
        log_eq = paint(MathTex(
            r"\log {{\pi^*(y|x)}} = \log {{\pi_{\text{ref}}(y|x)}} + {{r(x,y)}} /\beta - \log {{Z(x)}}",
            font_size=48)).move_to(0.6 * UP)
        cap.say("take logs and solve for the reward")
        self.play(TransformMatchingTex(pi_star, log_eq), run_time=1.8)
        self.wait(1)
        r_eq = paint(MathTex(
            r"{{r(x,y)}} = \beta \log \frac{ {{\pi^*(y|x)}} }{ {{\pi_{\text{ref}}(y|x)}} } + \beta \log {{Z(x)}}",
            font_size=48)).move_to(0.6 * UP)
        self.play(TransformMatchingTex(log_eq, r_eq), run_time=1.8)
        cap.say("any reward can be written through its own optimal policy", wait=2)
        self.r_eq = r_eq

    # ------------------------------------------------------------------ act 2: Z cancels
    def act2(self):
        cap = self.cap
        r_eq = self.r_eq
        self.play(r_eq.animate.scale(0.75).move_to(1.6 * UP))

        bt = MathTex(r"P(y_w \succ y_l) = \sigma\big(", r"r(x,y_w)", r"-", r"r(x,y_l)", r"\big)",
                     font_size=44).move_to(0.1 * DOWN)
        bt[1].set_color(VALUE_C); bt[3].set_color(VALUE_C)
        cap.say("Bradley–Terry: preference is a sigmoid of the reward gap")
        self.play(Write(bt))
        self.wait(1.5)

        w_term = r"\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)}"
        l_term = r"\beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}"
        sub = MathTex(r"P(y_w \succ y_l) = \sigma\!\Big(", r"\Big[", w_term, r"+ \beta \log Z(x)", r"\Big]",
                      r"-", r"\Big[", l_term, r"+ \beta \log Z(x)", r"\Big]", r"\Big)", font_size=40)
        fit(sub).move_to(0.1 * DOWN)
        cap.say("plug in the reward for both answers")
        self.play(ReplacementTransform(bt[0], sub[0]), ReplacementTransform(bt[1], sub[1:5]),
                  ReplacementTransform(bt[2], sub[5]), ReplacementTransform(bt[3], sub[6:10]),
                  ReplacementTransform(bt[4], sub[10]), Indicate(r_eq, color=VALUE_C, scale_factor=1.05),
                  run_time=1.8)
        self.wait(1)

        zs = [sub[3], sub[8]]
        boxes = VGroup(*[SurroundingRectangle(z, color=MEAN_C, buff=0.08) for z in zs])
        cap.say("both answers share the prompt, so the intractable Z(x) cancels", color=MEAN_C)
        self.play(*[z.animate.set_color(MEAN_C) for z in zs], Create(boxes),
                  Indicate(r_eq.submobjects[-1], color=MEAN_C))
        self.wait(1)
        strikes = VGroup(*[Line(b.get_corner(DL), b.get_corner(UR), color=WRONG, stroke_width=5) for b in boxes])
        self.play(Create(strikes), run_time=0.8)
        self.wait(0.8)

        fin = MathTex(r"P(y_w \succ y_l) = \sigma\!\Big(", w_term, r"-", l_term, r"\Big)", font_size=44)
        fit(fin).move_to(0.1 * DOWN)
        self.play(FadeOut(VGroup(*zs, boxes, strikes, sub[1], sub[4], sub[6], sub[9])),
                  ReplacementTransform(sub[0], fin[0]), ReplacementTransform(sub[2], fin[1]),
                  ReplacementTransform(sub[5], fin[2]), ReplacementTransform(sub[7], fin[3]),
                  ReplacementTransform(sub[10], fin[4]), FadeOut(r_eq), run_time=1.6)
        self.wait(1.5)

        self.loss_eq = MathTex(
            r"\mathcal L_{\text{DPO}} = -\log \sigma\Big(\beta\Big[", r"\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}",
            r"-", r"\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}", r"\Big]\Big)", font_size=44)
        fit(self.loss_eq).move_to(2.0 * DOWN)
        self.loss_eq[1].set_color(CORRECT); self.loss_eq[3].set_color(WRONG)
        cap.say("swap π* for the model and maximise the likelihood of the pairs")
        self.play(TransformFromCopy(fin, self.loss_eq), run_time=1.6)
        self.wait(2)

    # ------------------------------------------------------------------ act 3: the loss
    def act3(self):
        cap = self.cap
        self.play(self.loss_eq.animate.scale(0.8).move_to(1.85 * UP))
        s = ValueTracker(0)

        ax = Axes(x_range=[0, 3, 1], y_range=[LOGP_MIN, 0, 3], x_length=4.6, y_length=3.7, tips=False,
                  axis_config={"color": REF_C},
                  y_axis_config={"numbers_to_include": [-15, -10, -5, 0], "font_size": 22},
                  x_axis_config={"include_ticks": False}).move_to(RIGHT * CHART_X + 0.75 * DOWN)
        ylab = MathTex(r"\log \pi(y|x)", font_size=28, color=REF_C).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.5)
        names = VGroup(Text("chosen", font_size=24, color=CORRECT).move_to(ax.c2p(1, LOGP_MIN) + 0.35 * DOWN),
                       Text("rejected", font_size=24, color=WRONG).move_to(ax.c2p(2, LOGP_MIN) + 0.35 * DOWN))
        ghosts = VGroup(*[bar(ax, x, v, REF_C, width=0.78).set_fill(REF_C, 0.18).set_stroke(REF_C, 2, opacity=0.9)
                          for x, v in ((1, REF_W), (2, REF_L))])
        bars = always_redraw(lambda: VGroup(*[bar(ax, x, v, c, width=0.5)
                                              for x, v, c in zip((1, 2), logp_at(s.get_value()), (CORRECT, WRONG))]))
        legend = VGroup(
            VGroup(Square(0.22).set_fill(REF_C, 0.18).set_stroke(REF_C, 2), Text("reference (frozen)", font_size=20)
                   ).arrange(RIGHT, buff=0.12),
            VGroup(Square(0.22).set_fill(CORRECT, 0.75).set_stroke(CORRECT, 2),
                   Square(0.22).set_fill(WRONG, 0.75).set_stroke(WRONG, 2), Text("policy (trained)", font_size=20)
                   ).arrange(RIGHT, buff=0.12),
        ).arrange(RIGHT, buff=0.4).next_to(names, DOWN, buff=0.25)
        cap.say("the loss only needs log-probs from the policy and a frozen reference")
        self.play(Create(ax), FadeIn(ylab), FadeIn(names), FadeIn(ghosts), FadeIn(legend))
        self.play(FadeIn(bars))

        # Right panel: the sigmoid with the margin dot, and live readouts from the maths helpers.
        sax = Axes(x_range=[-2, 2, 1], y_range=[0, 1, 0.5], x_length=4.2, y_length=1.8, tips=False,
                   axis_config={"color": REF_C, "font_size": 20},
                   x_axis_config={"numbers_to_include": [-2, -1, 0, 1, 2]},
                   y_axis_config={"numbers_to_include": [0, 0.5, 1], "decimal_number_config": {"num_decimal_places": 1}},
                   ).move_to(RIGHT * PANEL_X + 0.15 * DOWN)
        curve = sax.plot(sigmoid, x_range=[-2, 2], color=MUTED, stroke_width=3)
        slab = MathTex(r"\sigma(\text{margin})", font_size=26, color=MUTED).move_to(sax.c2p(-1.1, 0.8))

        def margin():
            w, l = logp_at(s.get_value())
            return float(implicit_reward(w, REF_W, BETA) - implicit_reward(l, REF_L, BETA))

        dot = always_redraw(lambda: Dot(sax.c2p(margin(), sigmoid(margin())), radius=0.1, color=MEAN_C))

        def row(tex, value, col, size=30):
            return VGroup(MathTex(tex, font_size=size, color=col),
                          DecimalNumber(value, num_decimal_places=2, include_sign=True, font_size=size))\
                .arrange(RIGHT, buff=0.15)

        def readouts():
            w, l = logp_at(s.get_value())
            rw, rl = float(implicit_reward(w, REF_W, BETA)), float(implicit_reward(l, REF_L, BETA))
            loss = float(dpo_loss(w, l, REF_W, REF_L, BETA))
            g = VGroup(row(r"\hat r_w =", rw, CORRECT), row(r"\hat r_l =", rl, WRONG),
                       row(r"\text{margin} = \hat r_w - \hat r_l =", rw - rl, MEAN_C),
                       VGroup(MathTex(r"\mathcal L_{\text{DPO}} =", font_size=34),
                              DecimalNumber(loss, num_decimal_places=3, font_size=34)).arrange(RIGHT, buff=0.15))
            g.arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to(RIGHT * PANEL_X + 2.6 * DOWN)
            return g

        panel = always_redraw(readouts)
        self.play(Create(sax), Create(curve), FadeIn(slab))
        self.play(FadeIn(dot), FadeIn(panel))
        loss0 = float(dpo_loss(REF_W, REF_L, REF_W, REF_L, BETA))
        cap.say(f"at the start the policy is the reference: margin 0, loss log 2 = {loss0:.3f}")
        self.wait(2)

        cap.say("chosen log-prob up, rejected down, relative to the reference")
        self.play(s.animate.set_value(1), run_time=6, rate_func=smooth)
        w, l = logp_at(1)
        loss1 = float(dpo_loss(w, l, REF_W, REF_L, BETA))
        m1 = float(implicit_reward(w, REF_W, BETA) - implicit_reward(l, REF_L, BETA))
        self.wait(1)
        cap.say(f"margin {m1:.2f}: the dot climbs the sigmoid, loss falls to {loss1:.3f}", color=MEAN_C, wait=2.5)
        cap.say("no reward model, no sampling, no critic: just pairs", wait=3)
