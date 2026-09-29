"""Part 9 · PPO: the probability ratio and the clipped surrogate objective."""
import numpy as np
from style import *

EPS = 0.2


def clip_objective(rho, A, eps=EPS):
    rho = np.asarray(rho, float)
    return np.minimum(rho * A, np.clip(rho, 1 - eps, 1 + eps) * A)


def clip_grad(rho, A, eps=EPS, h=1e-6):
    return (clip_objective(rho + h, A, eps) - clip_objective(rho - h, A, eps)) / (2 * h)


def unclipped_term(rho, A):
    """The plain importance-weighted surrogate ρ·A."""
    return np.asarray(rho, float) * A


def clipped_term(rho, A, eps=EPS):
    """The clipped surrogate clip(ρ, 1 − ε, 1 + ε)·A (before the min)."""
    return np.clip(np.asarray(rho, float), 1 - eps, 1 + eps) * A


def ratio(p_new, p_old):
    return p_new / p_old


def ascent_path(rho0, A, lr, steps, eps=EPS):
    """Plain gradient ascent on ρ under the clipped objective: stops once the gradient is zero."""
    rs = [float(rho0)]
    for _ in range(steps):
        rs.append(rs[-1] + lr * float(clip_grad(rs[-1], A, eps)))
    return np.array(rs)


# Act 1: one action's probability under the old and the new policy.
P_OLD, P_UP, P_DOWN, P_NEW = 0.40, 0.56, 0.28, 0.46
# Acts 2-3: gradient ascent on ρ, starting on the far side of the trust band.
LR, N_STEPS = 0.15, 7
RHO0 = {+1.0: 0.5, -1.0: 1.5}
ARROW_RHO = 0.3      # arrow length in ρ-units per unit of gradient
PANEL_X = 4.4        # x-centre of the right-hand panel
CHART_CENTER = 2.5 * LEFT + 0.75 * DOWN
RHO_MAX = 2.0


def num(v, places=2, sign=False, font_size=30, color=WHITE):
    return DecimalNumber(v, num_decimal_places=places, include_sign=sign, font_size=font_size, color=color)


def swatch(kind, color):
    if kind == "dashed":
        return DashedLine(ORIGIN, 0.6 * RIGHT, color=color, stroke_width=4, dash_length=0.08)
    if kind == "solid":
        return Line(ORIGIN, 0.6 * RIGHT, color=color, stroke_width=6)
    return Rectangle(width=0.6, height=0.3).set_stroke(width=0).set_fill(color, 0.3)


def make_panel(A):
    """Axes for the clipped objective with advantage A, the 1 ± ε band and the three sampled curves."""
    lo, hi = (0.0, 2.2) if A > 0 else (-2.2, 0.0)
    ticks = [0.5, 1, 1.5, 2] if A > 0 else [-2, -1.5, -1, -0.5]
    ax = Axes(x_range=[0, RHO_MAX + 0.1, 0.5], y_range=[lo, hi, 0.5], x_length=7.4, y_length=4.6, tips=False,
              axis_config={"color": REF_C, "font_size": 24},
              x_axis_config={"numbers_to_include": [0.5, 1, 1.5, 2],
                             "decimal_number_config": {"num_decimal_places": 1},
                             "label_direction": DOWN if A > 0 else UP},
              y_axis_config={"numbers_to_include": ticks,
                             "decimal_number_config": {"num_decimal_places": 1}}).move_to(CHART_CENTER)
    xlab = MathTex(r"\rho", font_size=34).next_to(ax.c2p(RHO_MAX + 0.1, 0), RIGHT, buff=0.15)
    ylab = MathTex(r"L", font_size=34).next_to(ax.c2p(0, hi if A > 0 else lo), LEFT, buff=0.55)
    band = Polygon(ax.c2p(1 - EPS, lo), ax.c2p(1 + EPS, lo), ax.c2p(1 + EPS, hi), ax.c2p(1 - EPS, hi))
    band.set_stroke(width=0).set_fill(STD_C, 0.18)
    band_lab = MathTex(r"1 \pm \epsilon", font_size=28, color=STD_C)
    band_lab.next_to(ax.c2p(1, hi if A > 0 else lo), DOWN if A > 0 else UP, buff=0.12)

    rho_s = np.linspace(0, RHO_MAX, 801)

    def curve(f, color, width):
        return ax.plot_line_graph(rho_s, f(rho_s), add_vertex_dots=False, line_color=color, stroke_width=width)["line_graph"]

    raw = DashedVMobject(curve(lambda r: unclipped_term(r, A), REF_C, 3), num_dashes=60)
    clp = DashedVMobject(curve(lambda r: clipped_term(r, A), VALUE_C, 3), num_dashes=60)
    obj = curve(lambda r: clip_objective(r, A), POLICY_C, 7)
    return {"ax": ax, "labels": VGroup(xlab, ylab), "band": VGroup(band, band_lab),
            "raw": raw, "clp": clp, "obj": obj}


def panel_group(p):
    return VGroup(p["band"], p["ax"], p["labels"], p["raw"], p["clp"], p["obj"])


class Part09(Scene):
    def construct(self):
        self.title = title_card(self, 9, "PPO: clip the ratio, keep the step small")
        self.cap = Captioner(self, self.title)
        self.act1()
        clear_act(self, keep=[self.title])
        self.act2()
        self.act3()
        takeaway(self, "ppo = trpo's idea with one min and one clip")
        roadmap_outro(self, 9)

    # ------------------------------------------------------------------ act 1: the ratio
    def act1(self):
        cap = self.cap
        ax = Axes(x_range=[0, 3, 1], y_range=[0, 0.8, 0.2], x_length=4.6, y_length=4.4, tips=False,
                  axis_config={"color": REF_C},
                  y_axis_config={"numbers_to_include": [0.2, 0.4, 0.6, 0.8], "font_size": 22,
                                 "decimal_number_config": {"num_decimal_places": 1}},
                  x_axis_config={"include_ticks": False}).move_to(3.3 * LEFT + 0.8 * DOWN)
        ylab = Text("probability of action a", font_size=22, color=REF_C).rotate(PI / 2)
        ylab.next_to(ax.y_axis, LEFT, buff=0.5)
        names = VGroup(MathTex(r"\pi_{\text{old}}(a|s)", font_size=32, color=REF_C).move_to(ax.c2p(1, 0) + 0.45 * DOWN),
                       MathTex(r"\pi_\theta(a|s)", font_size=32, color=POLICY_C).move_to(ax.c2p(2, 0) + 0.45 * DOWN))
        p = ValueTracker(P_OLD)
        old_bar = bar(ax, 1, P_OLD, REF_C, width=0.7)
        old_val = num(P_OLD, font_size=28).next_to(old_bar, UP, buff=0.1)
        new_bar = always_redraw(lambda: bar(ax, 2, p.get_value(), POLICY_C, width=0.7))
        new_val = always_redraw(lambda: num(p.get_value(), font_size=28).next_to(new_bar, UP, buff=0.1))

        cap.say("same state, same action: old policy vs new policy")
        self.play(Create(ax), FadeIn(ylab), FadeIn(names), run_time=1.2)
        self.play(GrowFromEdge(old_bar, DOWN), FadeIn(old_val))
        self.play(FadeIn(new_bar), FadeIn(new_val))

        defn = MathTex(r"\rho", "=", r"{\pi_\theta(a|s) \over \pi_{\text{old}}(a|s)}", font_size=44)
        defn[0].set_color(MEAN_C)
        defn.move_to(PANEL_X * RIGHT + 1.3 * UP)
        live = always_redraw(lambda: MathTex(
            rf"\rho = {{{p.get_value():.2f} \over {P_OLD:.2f}}} = {ratio(p.get_value(), P_OLD):.2f}",
            font_size=40).next_to(defn, DOWN, buff=0.5))
        self.play(Write(defn))
        self.play(FadeIn(live))

        line = NumberLine(x_range=[0, RHO_MAX, 0.5], length=4.0, color=REF_C, include_numbers=True,
                          font_size=26, decimal_number_config={"num_decimal_places": 1})
        line.move_to((PANEL_X - 0.3) * RIGHT + 1.9 * DOWN)
        line_lab = MathTex(r"\rho", font_size=32, color=MEAN_C).next_to(line, RIGHT, buff=0.2)
        marker = always_redraw(lambda: Triangle(color=MEAN_C, fill_opacity=1).scale(0.12).rotate(PI)
                               .next_to(line.n2p(ratio(p.get_value(), P_OLD)), UP, buff=0.05))
        self.play(Create(line), FadeIn(line_lab), FadeIn(marker))
        cap.say("ρ measures how far the new policy moved on this action", wait=1)

        self.play(p.animate.set_value(P_UP), run_time=2)
        self.wait(0.5)
        self.play(p.animate.set_value(P_DOWN), run_time=2.5)
        self.wait(0.5)
        self.play(p.animate.set_value(P_OLD), run_time=1.5)
        cap.say(f"ρ = {ratio(P_OLD, P_OLD):.0f}: no change. ρ > 1: more likely. ρ < 1: less likely", wait=1.5)

        band = Rectangle(width=line.n2p(1 + EPS)[0] - line.n2p(1 - EPS)[0], height=0.5)
        band.set_stroke(width=0).set_fill(STD_C, 0.3).move_to(line.n2p(1))
        band_lab = MathTex(rf"1 \pm \epsilon = [{1 - EPS:.1f},\ {1 + EPS:.1f}]", font_size=30, color=STD_C)
        band_lab.next_to(band, DOWN, buff=0.55)
        cap.say(f"ppo wants every ρ to stay near 1, within ε = {EPS:g}", color=STD_C)
        self.play(FadeIn(band), FadeIn(band_lab))
        self.play(p.animate.set_value(P_NEW), run_time=1.5)
        self.wait(1.5)
        for m in (new_bar, new_val, live, marker):
            m.clear_updaters()

    # ------------------------------------------------------------------ helpers for acts 2-3
    def build_curves(self, A, words):
        """Draw the panel for advantage A one curve at a time, with captions and a legend."""
        cap = self.cap
        p = make_panel(A)
        ax = p["ax"]
        formula = MathTex(r"L", "=", r"\min\big(", r"\rho A", ",\ ", r"\mathrm{clip}(\rho, 1{-}\epsilon, 1{+}\epsilon)\,A",
                          r"\big)", font_size=30)
        formula[3].set_color(REF_C); formula[5].set_color(VALUE_C); formula[2].set_color(POLICY_C)
        fit(formula, 4.9).move_to(PANEL_X * RIGHT + 1.75 * UP)
        adv = MathTex(rf"A = {A:+.0f}", font_size=40, color=CORRECT if A > 0 else WRONG)
        adv.next_to(formula, DOWN, buff=0.3)
        legend = VGroup(
            VGroup(swatch("dashed", REF_C), MathTex(r"\rho A", font_size=28), Text("unclipped", font_size=20, color=MUTED)),
            VGroup(swatch("dashed", VALUE_C), MathTex(r"\mathrm{clip}(\rho)\,A", font_size=28)),
            VGroup(swatch("solid", POLICY_C), Text("min of the two", font_size=22)),
            VGroup(swatch("band", STD_C), MathTex(rf"1 \pm \epsilon,\ \epsilon = {EPS:g}", font_size=28)),
        )
        for row in legend:
            row.arrange(RIGHT, buff=0.2)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(adv, DOWN, buff=0.35)
        legend.set_x(PANEL_X)

        cap.say(words["intro"], color=CORRECT if A > 0 else WRONG)
        self.play(Create(ax), FadeIn(p["labels"]), Write(formula), FadeIn(adv), run_time=1.5)
        self.play(FadeIn(p["band"]), FadeIn(legend[3]))
        cap.say(words["raw"])
        self.play(Create(p["raw"]), FadeIn(legend[0]), Indicate(formula[3], color=REF_C), run_time=1.5)
        cap.say(words["clp"])
        self.play(Create(p["clp"]), FadeIn(legend[1]), Indicate(formula[5], color=VALUE_C), run_time=1.5)
        cap.say("ppo keeps the smaller of the two: a pessimistic bound", color=POLICY_C)
        self.play(Create(p["obj"]), FadeIn(legend[2]), Indicate(formula[2], color=POLICY_C), run_time=1.8)
        self.play(p["raw"].animate.set_stroke(opacity=0.45), p["clp"].animate.set_stroke(opacity=0.45))
        self.wait(0.8)
        return p, VGroup(formula, adv, legend)

    def roll_ball(self, p, A, side_panel):
        """Gradient ascent on ρ: ball at (ρ, L(ρ)), arrow of length ∂L/∂ρ."""
        ax = p["ax"]
        path = ascent_path(RHO0[A], A, LR, N_STEPS)
        t = ValueTracker(0)

        def rho():
            s = t.get_value()
            i = min(int(np.floor(s)), N_STEPS - 1)
            return path[i] + (s - i) * (path[i + 1] - path[i])

        ball = always_redraw(lambda: Dot(ax.c2p(rho(), float(clip_objective(rho(), A))), radius=0.14,
                                         color=MEAN_C).set_stroke(WHITE, 2).set_z_index(3))

        def grad_arrow():
            r = rho()
            g = float(clip_grad(r, A))
            if abs(g) < 1e-6:
                return VMobject()
            y = float(clip_objective(r, A))
            return Arrow(ax.c2p(r, y), ax.c2p(r + ARROW_RHO * g, y), buff=0, stroke_width=6,
                         max_tip_length_to_length_ratio=0.35, color=CORRECT if A > 0 else WRONG).set_z_index(2)

        arrow = always_redraw(grad_arrow)
        gcol = CORRECT if A > 0 else WRONG
        readout = always_redraw(lambda: VGroup(
            VGroup(Text("step", font_size=30, color=MUTED),
                   Integer(int(np.floor(t.get_value())), font_size=38)).arrange(RIGHT, buff=0.2),
            VGroup(MathTex(r"\rho =", font_size=40, color=MEAN_C), num(rho(), font_size=38)).arrange(RIGHT, buff=0.15),
            VGroup(MathTex(r"L =", font_size=40, color=POLICY_C),
                   num(float(clip_objective(rho(), A)), sign=True, font_size=38)).arrange(RIGHT, buff=0.15),
            VGroup(MathTex(r"\partial L / \partial \rho =", font_size=40, color=gcol),
                   num(float(clip_grad(rho(), A)), sign=True, font_size=38)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to(side_panel[2], aligned_edge=UL))
        self.play(FadeOut(side_panel[2]), run_time=0.5)
        self.remove(side_panel[2])
        self.play(FadeIn(ball, scale=0.5), FadeIn(arrow), FadeIn(readout))
        return path, t, ball, arrow, readout

    def run_ball(self, t, run_time):
        self.play(t.animate.set_value(N_STEPS), run_time=run_time, rate_func=linear)

    # ------------------------------------------------------------------ act 2: A > 0
    def act2(self):
        cap = self.cap
        A = +1.0
        p, side = self.build_curves(A, {
            "intro": f"a good action: advantage A = {A:+.0f}",
            "raw": "unclipped ρA: the higher ρ, the better, forever",
            "clp": f"clipped: flat once ρ passes 1 + ε = {1 + EPS:.1f}",
        })
        cap.say("now let gradient ascent push ρ, one step at a time")
        path, t, ball, arrow, readout = self.roll_ball(p, A, side)
        self.run_ball(t, 6)
        stop = path[-1]
        assert stop > 1 + EPS and abs(float(clip_grad(stop, A))) < 1e-6
        cap.say("good action: raise it until ρ passes 1 + ε; there the gradient dies", color=CORRECT)
        self.play(Indicate(readout[3], color=CORRECT), Flash(ball, color=MEAN_C))
        self.wait(2)
        for m in (ball, arrow, readout):
            m.clear_updaters()
        self.pos = {"p": p, "ball": ball, "arrow": arrow}
        self.remove(arrow)   # zero gradient: the arrow has no points left
        self.play(FadeOut(VGroup(panel_group(p), ball, readout, side[:2])))
        self.remove(panel_group(p), ball, arrow, readout, side)

    # ------------------------------------------------------------------ act 3: A < 0, then side by side
    def act3(self):
        cap = self.cap
        A = -1.0
        p, side = self.build_curves(A, {
            "intro": f"a bad action: advantage A = {A:+.0f}",
            "raw": "unclipped ρA: the lower ρ, the better, forever",
            "clp": f"clipped: flat once ρ drops below 1 − ε = {1 - EPS:.1f}",
        })
        cap.say("gradient ascent now pushes ρ toward smaller values")
        path, t, ball, arrow, readout = self.roll_ball(p, A, side)
        self.run_ball(t, 6)
        stop = path[-1]
        assert stop < 1 - EPS and abs(float(clip_grad(stop, A))) < 1e-6
        cap.say("bad action: lower it until ρ passes 1 − ε; there the gradient dies", color=WRONG)
        self.play(Indicate(readout[3], color=WRONG), Flash(ball, color=MEAN_C))
        self.wait(2)
        for m in (ball, arrow, readout):
            m.clear_updaters()
        self.remove(arrow)   # zero gradient: the arrow has no points left

        # Both panels side by side.
        neg = VGroup(panel_group(p), ball)
        pos = VGroup(panel_group(self.pos["p"]), self.pos["ball"])
        self.play(FadeOut(readout), FadeOut(side[:2]))
        self.remove(readout, side)
        pos.scale(0.72).move_to(3.55 * LEFT + 0.55 * DOWN)
        self.play(neg.animate.scale(0.72).move_to(3.55 * RIGHT + 0.55 * DOWN), run_time=1.3)
        heads = VGroup(
            MathTex(r"A > 0", font_size=36, color=CORRECT).next_to(pos, UP, buff=0.2),
            MathTex(r"A < 0", font_size=36, color=WRONG).next_to(neg, UP, buff=0.2),
        )
        self.play(FadeIn(pos), FadeIn(heads))
        cap.say("the clip only removes incentives, it never adds them", wait=3)
