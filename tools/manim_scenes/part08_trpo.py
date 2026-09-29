"""Part 8 · TRPO: steps measured in policy change (KL), the trust region, and why on-policy collapse hurts."""
import numpy as np
from style import *


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


# ---------------------------------------------------------------- act 1: same Δθ, different KL
MAX_KL = 0.01                    # the notebook's δ
STEP = 1.0                       # the parameter step Δθ applied in both cases
STARTS = (0.0, 4.0)              # θ where the step is applied

# ---------------------------------------------------------------- act 2: an ILLUSTRATIVE 2-D plane
# A fixed 2x2 matrix standing in for a Fisher matrix, purely to draw the picture.
# It is NOT the Fisher matrix of the act-1 policy (that one is 1-D).
FISHER_2D = np.array([[3.0, 1.0], [1.0, 0.5]])
GRAD_2D = np.array([1.0, 0.2])


def quad_kl(d, F=FISHER_2D):
    """Quadratic KL model 0.5 d^T F d."""
    d = np.asarray(d, float)
    return 0.5 * d @ F @ d


def natural_step_2d(g=GRAD_2D, F=FISHER_2D, max_kl=MAX_KL):
    """TRPO's full step sqrt(2δ / gᵀF⁻¹g) F⁻¹g: the natural direction scaled onto the ellipse ½dᵀFd = δ."""
    x = np.linalg.solve(F, g)
    return np.sqrt(2 * max_kl / (g @ x)) * x


def kl_ellipse(F=FISHER_2D, max_kl=MAX_KL, n=240):
    """Points d on the boundary ½ dᵀF d = max_kl (eigen-decomposition of F)."""
    lam, V = np.linalg.eigh(F)
    ang = np.linspace(0, 2 * np.pi, n)
    unit = np.stack([np.cos(ang), np.sin(ang)])
    return (V @ (np.sqrt(2 * max_kl / lam)[:, None] * unit)).T


def grad_step_same_length(g=GRAD_2D, length=None):
    """Plain gradient direction with the same Euclidean length as the natural step (as in the notebook)."""
    length = np.linalg.norm(natural_step_2d()) if length is None else length
    return g / np.linalg.norm(g) * length


def grad_step_in_region(g=GRAD_2D, F=FISHER_2D, max_kl=MAX_KL):
    """Plain gradient direction shrunk until its quadratic KL equals max_kl."""
    return np.sqrt(2 * max_kl / (g @ F @ g)) * g


# ---------------------------------------------------------------- act 3: schematic collapse
COLLAPSE_THETAS = [2.0, -2.5, -3.5, -4.5]   # schematic θ of P(good action) per iteration; a big bad step after iter 1
BATCH = 12


def sample_batch(p_good, n=BATCH, seed=0):
    """Which of n on-policy samples picked the good action."""
    return np.random.default_rng(seed).random(n) < p_good


PANEL_X = 4.2


def fmt(v, k=3):
    return f"{v:.{k}f}"


class Part08(Scene):
    def construct(self):
        self.title = title_card(self, 8, "TRPO: trust region policy optimisation")
        self.cap = Captioner(self, self.title)
        self.act1(); clear_act(self, keep=[self.title])
        self.act2(); clear_act(self, keep=[self.title])
        self.act3()
        takeaway(self, "measure steps by how much the policy changes, not the parameters")
        roadmap_outro(self, 8)

    # ------------------------------------------------------------------ act 1: the step-size problem
    def act1(self):
        cap = self.cap
        ax = Axes(x_range=[-2, 7, 1], y_range=[0, 1.05, 0.25], x_length=7.2, y_length=4.6, tips=False,
                  axis_config={"color": REF_C, "font_size": 22},
                  x_axis_config={"numbers_to_include": list(range(-2, 8))},
                  y_axis_config={"numbers_to_include": [0, 0.5, 1], "decimal_number_config": {"num_decimal_places": 1}},
                  ).move_to(2.3 * LEFT + 0.55 * DOWN)
        xl = MathTex(r"\theta", font_size=34, color=REF_C).next_to(ax.x_axis, RIGHT, buff=0.15)
        yl = MathTex(r"P(a{=}1)", font_size=30, color=REF_C).next_to(ax.y_axis, UP, buff=0.15)
        curve = ax.plot(sigmoid, x_range=[-2, 7], color=POLICY_C, stroke_width=4)
        form = MathTex(r"P(a{=}1) = \sigma(\theta)", font_size=32, color=POLICY_C).next_to(
            ax.c2p(2.2, 0.3), RIGHT, buff=0)
        cap.say("a two-action policy with one parameter θ")
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=1.2)
        self.play(Create(curve), FadeIn(form))

        th = ValueTracker(STARTS[0])
        dot = always_redraw(lambda: Dot(ax.c2p(th.get_value(), sigmoid(th.get_value())), radius=0.09,
                                        color=MEAN_C).set_stroke(WHITE, 1.5))

        # right panel: the policy itself, as two bars
        bx = Axes(x_range=[0, 3, 1], y_range=[0, 1, 0.5], x_length=3.0, y_length=2.2, tips=False,
                  axis_config={"color": REF_C, "font_size": 20},
                  y_axis_config={"numbers_to_include": [0, 1]},
                  x_axis_config={"include_ticks": False}).move_to(PANEL_X * RIGHT + 0.2 * UP)
        blabs = VGroup(*[MathTex(f"a{{=}}{a}", font_size=26, color=MUTED).next_to(bx.c2p(a + 1, 0), DOWN, buff=0.15)
                         for a in range(2)])
        bars = always_redraw(lambda: VGroup(
            bar(bx, 1, 1 - sigmoid(th.get_value()), REF_C, width=0.7),
            bar(bx, 2, sigmoid(th.get_value()), POLICY_C, width=0.7)))
        ptxt = always_redraw(lambda: MathTex(
            rf"\theta = {th.get_value():.2f},\;\; P(a{{=}}1) = {sigmoid(th.get_value()):.3f}", font_size=26
        ).next_to(bx, UP, buff=0.2))
        self.play(FadeIn(dot), FadeIn(bx), FadeIn(blabs), FadeIn(bars), FadeIn(ptxt))
        self.wait(0.5)

        rows = VGroup()
        steps = VGroup()
        for i, t0 in enumerate(STARTS):
            t1 = t0 + STEP
            p0, p1, kl = sigmoid(t0), sigmoid(t1), kl_bern(t0, t1)
            if i == 1:
                self.play(th.animate.set_value(t0), run_time=1.2)
            col = WRONG if i == 0 else CORRECT
            # parameter step along the θ axis
            arr = Arrow(ax.c2p(t0, 0) + 0.35 * DOWN, ax.c2p(t1, 0) + 0.35 * DOWN, buff=0, color=STD_C,
                        stroke_width=5, max_tip_length_to_length_ratio=0.25)
            arr_lab = MathTex(rf"\Delta\theta = {STEP:.0f}", font_size=26, color=STD_C).next_to(arr, DOWN, buff=0.08)
            # policy change on the P axis
            dp = Line(ax.c2p(t1, p0), ax.c2p(t1, p1), color=col, stroke_width=7)
            guide = DashedLine(ax.c2p(t0, p0), ax.c2p(t1, p0), color=col, stroke_width=2, dash_length=0.06)
            dp_lab = MathTex(rf"\Delta P = {p1 - p0:.3f}", font_size=26, color=col).next_to(
                dp, RIGHT if i == 0 else DOWN, buff=0.12)
            if i == 1:
                dp_lab.next_to(ax.c2p(t1, p0), DOWN, buff=0.35)
            cap.say(f"step θ from {t0:.0f} to {t1:.0f}: Δθ = {STEP:.0f}")
            self.play(GrowArrow(arr), FadeIn(arr_lab))
            self.play(th.animate.set_value(t1), Create(guide), run_time=1.5)
            self.play(Create(dp), FadeIn(dp_lab))
            row = MathTex(rf"\theta: {t0:.0f}\to{t1:.0f}", r"\quad", rf"\mathrm{{KL}} = {fmt(kl, 4)}",
                          font_size=30)
            row[2].set_color(col)
            rows.add(row)
            rows.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(PANEL_X * RIGHT + 2.0 * DOWN)
            self.play(FadeIn(row, shift=0.2 * LEFT))
            steps.add(VGroup(arr, arr_lab, dp, guide, dp_lab))
            self.wait(1.2)

        ratio = kl_bern(STARTS[0], STARTS[0] + STEP) / kl_bern(STARTS[1], STARTS[1] + STEP)
        rtxt = Text(f"same Δθ, {ratio:.0f}× the KL", font_size=26, color=MEAN_C).next_to(rows, DOWN, buff=0.35)
        self.play(FadeIn(rtxt), Indicate(rows[0][2], color=WRONG))
        cap.say("equal steps in parameters ≠ equal changes in behaviour", color=MEAN_C, wait=2.5)

        # the notebook's fix in 1-D: size the step by the Fisher information, not by a fixed Δθ
        nat = [natural_step(t, 1.0, MAX_KL) for t in STARTS]
        cap.say(f"fix a KL budget δ = {MAX_KL} instead, and the step size adapts")
        self.play(FadeOut(steps), FadeOut(rtxt))
        nrows = VGroup(*[MathTex(rf"\theta = {t:.0f}:", r"\;\Delta\theta = \sqrt{2\delta/F} =",
                                 rf"{d:.2f}", font_size=28) for t, d in zip(STARTS, nat)]
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(rows)
        for r in nrows:
            r[2].set_color(STD_C)
        self.play(ReplacementTransform(rows, nrows))
        nat_arrs = VGroup()
        for t, d in zip(STARTS, nat):
            a = Arrow(ax.c2p(t, 0) + 0.35 * DOWN, ax.c2p(t + d, 0) + 0.35 * DOWN, buff=0, color=STD_C,
                      stroke_width=5, max_tip_length_to_length_ratio=0.25)
            nat_arrs.add(a)
        self.play(th.animate.set_value(STARTS[0]), run_time=0.8)
        self.play(GrowArrow(nat_arrs[0]), th.animate.set_value(STARTS[0] + nat[0]), run_time=1.2)
        self.play(th.animate.set_value(STARTS[1]), run_time=1)
        self.play(GrowArrow(nat_arrs[1]), th.animate.set_value(STARTS[1] + nat[1]), run_time=1.5)
        kls = [kl_bern(t, t + d) for t, d in zip(STARTS, nat)]
        cap.say(f"real KL: {kls[0]:.3f} and {kls[1]:.3f}, both near the budget δ", wait=2.5)
        for m in (dot, bars, ptxt):
            m.clear_updaters()

    # ------------------------------------------------------------------ act 2: a trust region in policy space
    def act2(self):
        cap = self.cap
        R = 0.42
        ax = Axes(x_range=[-R, R, 0.2], y_range=[-R, R, 0.2], x_length=5.2, y_length=5.2, tips=False,
                  axis_config={"color": REF_C, "stroke_opacity": 0.6, "include_ticks": False}
                  ).move_to(2.9 * LEFT + 0.95 * DOWN)
        xl = MathTex(r"\Delta\theta_1", font_size=30, color=REF_C).next_to(ax.x_axis, RIGHT, buff=0.12)
        yl = MathTex(r"\Delta\theta_2", font_size=30, color=REF_C).next_to(ax.y_axis, UP, buff=0.1)
        tag = Text("illustrative 2-D parameter plane", font_size=20, color=MUTED).next_to(ax, DOWN, buff=0.1)
        cap.say("now two parameters: which step Δθ should we take?")
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(tag), run_time=1.2)

        p = lambda v: ax.c2p(v[0], v[1])
        d_nat = natural_step_2d()
        d_grad = grad_step_same_length()
        d_fit = grad_step_in_region()
        assert np.isclose(quad_kl(d_nat), MAX_KL) and np.isclose(quad_kl(d_fit), MAX_KL)
        r = np.linalg.norm(d_nat)

        F = FISHER_2D
        panel = VGroup(
            MathTex(rf"F = \begin{{bmatrix}} {F[0,0]:g} & {F[0,1]:g} \\ {F[1,0]:g} & {F[1,1]:g} \end{{bmatrix}}",
                    font_size=32),
            Text("(a made-up F, just for the picture)", font_size=18, color=MUTED),
            MathTex(rf"g = ({GRAD_2D[0]:g},\ {GRAD_2D[1]:g})", font_size=32, color=MEAN_C),
        ).arrange(DOWN, buff=0.18).move_to(3.4 * RIGHT + 1.35 * UP)
        self.play(FadeIn(panel))

        # Euclidean circle and the plain gradient step
        circle = Circle(radius=abs(p([r, 0])[0] - p([0, 0])[0]), color=REF_C, stroke_width=3).move_to(p([0, 0]))
        circle.set_stroke(opacity=0.8)
        circ_lab = Text("‖Δθ‖ = r", font_size=22, color=REF_C).next_to(circle.point_at_angle(1.2 * PI), DL, buff=0.08)
        cap.say("a plain step size limits ‖Δθ‖: a circle in parameter space")
        self.play(Create(circle), FadeIn(circ_lab))
        g_arr = Arrow(p([0, 0]), p(d_grad), buff=0, color=MEAN_C, stroke_width=6, max_tip_length_to_length_ratio=0.2)
        g_lab = MathTex("g", font_size=34, color=MEAN_C).next_to(g_arr.get_end(), UR, buff=0.08)
        self.play(GrowArrow(g_arr), FadeIn(g_lab))

        kl_line = MathTex(r"\mathrm{KL} \approx \tfrac12\,\Delta\theta^\top F\,\Delta\theta", font_size=32
                          ).next_to(panel, DOWN, buff=0.4)
        self.play(Write(kl_line))
        g_row = VGroup(MathTex("g", font_size=30, color=MEAN_C), Text("step:", font_size=22),
                       MathTex(rf"\mathrm{{KL}} = {fmt(quad_kl(d_grad))}", font_size=30, color=WRONG)
                       ).arrange(RIGHT, buff=0.15)
        g_row.next_to(kl_line, DOWN, buff=0.35).align_to(kl_line, LEFT)
        self.play(FadeIn(g_row, shift=0.2 * LEFT))
        self.wait(1)

        # the KL ellipse
        ell = Polygon(*[p(v) for v in kl_ellipse()], color=POLICY_C, stroke_width=4).set_fill(POLICY_C, 0.12)
        ell_lab = MathTex(rf"\tfrac12\,\Delta\theta^\top F\Delta\theta = \delta", font_size=26, color=POLICY_C
                          ).next_to(p(kl_ellipse()[np.argmin(kl_ellipse()[:, 1])]), RIGHT, buff=0.15)
        cap.say(f"the KL budget ½ ΔθᵀFΔθ ≤ δ = {MAX_KL} is an ellipse", color=POLICY_C)
        self.play(DrawBorderThenFill(ell), FadeIn(ell_lab), run_time=1.5)
        self.play(Indicate(g_arr, color=WRONG), Indicate(g_row[2], color=WRONG))
        cap.say(f"the plain step leaves it: KL = {fmt(quad_kl(d_grad))}, {quad_kl(d_grad) / MAX_KL:.1f}× the budget",
                color=WRONG, wait=1.5)

        # the natural gradient
        n_arr = Arrow(p([0, 0]), p(d_nat), buff=0, color=CORRECT, stroke_width=6, max_tip_length_to_length_ratio=0.2)
        n_lab = MathTex(r"F^{-1}g", font_size=32, color=CORRECT).next_to(n_arr.get_end(), RIGHT, buff=0.2
                                                                           ).shift(0.3 * DOWN)
        n_dot = Dot(p(d_nat), radius=0.08, color=CORRECT).set_stroke(WHITE, 1.5)
        n_row = VGroup(MathTex(r"F^{-1}g", font_size=30, color=CORRECT), Text("step:", font_size=22),
                       MathTex(rf"\mathrm{{KL}} = {fmt(quad_kl(d_nat))} = \delta", font_size=30, color=CORRECT)
                       ).arrange(RIGHT, buff=0.15).next_to(g_row, DOWN, buff=0.3, aligned_edge=LEFT)
        cap.say("the natural gradient F⁻¹g, scaled to land on the boundary")
        self.play(GrowArrow(n_arr), FadeIn(n_lab))
        self.play(FadeIn(n_dot, scale=2), Flash(n_dot, color=CORRECT), FadeIn(n_row, shift=0.2 * LEFT))
        self.wait(1)
        cap.say(f"trpo takes the biggest step whose kl stays under δ = {MAX_KL}", color=CORRECT, wait=2)

        # fair comparison: the gradient direction shrunk to fit the same budget
        fit_arr = DashedLine(p([0, 0]), p(d_fit), color=MEAN_C, stroke_width=5).add_tip(tip_length=0.18)
        gain = VGroup(
            Text("objective gain gᵀΔθ at KL = δ", font_size=20, color=MUTED),
            MathTex(rf"g\ \text{{shrunk}}:\ {fmt(GRAD_2D @ d_fit)}", font_size=30, color=MEAN_C),
            MathTex(rf"F^{{-1}}g:\ {fmt(GRAD_2D @ d_nat)}", font_size=30, color=CORRECT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(n_row, DOWN, buff=0.4, aligned_edge=LEFT)
        cap.say("shrink g to fit the ellipse and it buys less improvement")
        self.play(g_arr.animate.set_opacity(0.3), g_lab.animate.set_opacity(0.3), Create(fit_arr))
        self.play(FadeIn(gain, shift=0.2 * LEFT))
        self.wait(2.5)

    # ------------------------------------------------------------------ act 3: why it matters (schematic)
    def act3(self):
        cap = self.cap
        # the on-policy loop
        c = 3.9 * LEFT + 0.8 * DOWN
        names = [r"\pi_\theta", r"\text{batch}", r"\text{update}"]
        angs = [PI / 2, PI / 2 + 2 * PI / 3, PI / 2 + 4 * PI / 3]
        nodes = VGroup(*[
            VGroup(Circle(radius=0.62, color=MUTED, stroke_width=3), MathTex(nm, font_size=30)).move_to(
                c + 1.75 * np.array([np.cos(a), np.sin(a), 0]))
            for nm, a in zip(names, angs)])
        nodes[0][0].set_color(POLICY_C)
        links = VGroup(*[
            CurvedArrow(nodes[i].get_center(), nodes[(i + 1) % 3].get_center(), angle=PI / 3, color=REF_C,
                        stroke_width=3).scale(0.5)
            for i in range(3)])
        # place each arrow midway between its two nodes, nudged outward
        for i, l in enumerate(links):
            mid = (nodes[i].get_center() + nodes[(i + 1) % 3].get_center()) / 2
            l.move_to(mid + 0.45 * (mid - c) / np.linalg.norm(mid - c))
        tag = Text("schematic", font_size=20, color=MUTED).next_to(VGroup(nodes, links), DOWN, buff=0.25)
        cap.say("on-policy: the policy collects the very data it learns from")
        self.play(LaggedStart(*[FadeIn(n) for n in nodes], lag_ratio=0.2), FadeIn(tag))
        self.play(LaggedStart(*[Create(l) for l in links], lag_ratio=0.3))

        # iteration log on the right: P(good) bar + a batch of samples
        head = VGroup(Text("iter", font_size=20, color=MUTED), Text("P(good)", font_size=20, color=MUTED),
                      Text(f"batch of {BATCH} actions", font_size=20, color=MUTED))
        rows = VGroup()
        x_it, x_bar, x_bat = 0.2, 1.35, 4.4
        y0, dy = 1.35, 0.95
        for m, x in zip(head, (x_it, x_bar, x_bat)):
            m.move_to([x, y0 + 0.2, 0])
        self.play(FadeIn(head))
        for k, t in enumerate(COLLAPSE_THETAS):
            pg = sigmoid(t)
            good = sample_batch(pg, seed=k)
            y = y0 - dy * (k + 1) + 0.35
            back = Rectangle(width=1.0, height=0.3).set_stroke(REF_C, 1.5).move_to([x_bar, y, 0])
            fill = Rectangle(width=max(pg, 0.02) * 1.0, height=0.3).set_stroke(width=0).set_fill(POLICY_C, 0.85)
            fill.align_to(back, LEFT).set_y(y)
            dots = VGroup(*[Dot(radius=0.1, color=CORRECT if gd else WRONG) for gd in good]
                          ).arrange(RIGHT, buff=0.1).move_to([x_bat, y, 0])
            row = VGroup(Text(str(k + 1), font_size=24).move_to([x_it, y, 0]), back, fill, dots)
            rows.add(row)
            if k == 1:
                bad = Text("big bad step", font_size=22, color=WRONG).next_to(links[2], RIGHT, buff=0.1)
                cap.say("one step too far pushes the policy off a cliff", color=WRONG)
                self.play(FadeIn(bad), links[2].animate.set_color(WRONG).set_stroke(width=6),
                          nodes[0][0].animate.set_color(WRONG))
            self.play(Indicate(nodes[1], color=MEAN_C, scale_factor=1.1), FadeIn(row[:3]),
                      LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.05), run_time=1.2)
            if k == 0:
                cap.say("a healthy policy samples mostly good actions")
                self.wait(0.8)
            if k == 2:
                cap.say("now it rarely tries the good action, so it can't learn it back", color=WRONG)
        cap.say("on-policy, a bad step also ruins the next batch of data", color=MEAN_C, wait=2.5)
