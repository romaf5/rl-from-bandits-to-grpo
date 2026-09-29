"""Part 6 · REINFORCE: push up what paid off, the log-derivative trick, and noisy-but-unbiased gradients."""
import numpy as np
from style import *

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


def policy_steps(theta, steps=4, lr=0.5, noise=0.5, seed=0):
    """Act 1: single-sample REINFORCE updates theta <- theta + lr * R * grad log pi(a).
    Returns a list of (action, reward, probs before, probs after)."""
    rng = np.random.default_rng(seed); th = np.array(theta, float); out = []
    for _ in range(steps):
        p = softmax(th); a = int(rng.choice(3, p=p)); R = Q_TRUE[a] + noise * rng.standard_normal()
        g = -p.copy(); g[a] += 1
        th = th + lr * R * g
        out.append((a, R, p, softmax(th)))
    return out


def spread(samples):
    """Summed variance of the gradient estimates (the 'noise' number shown on screen)."""
    return samples.var(0).sum()


def cov_ellipse(samples, k=2.0):
    """(centre, width, height, angle) of the k-sigma covariance ellipse of 2-D samples."""
    c = samples.mean(0); w, v = np.linalg.eigh(np.cov(samples.T))
    return c, 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), np.arctan2(v[1, 1], v[0, 1])


THETA0 = np.zeros(3)                     # uniform starting policy
ACT1_LR, ACT1_NOISE, ACT1_SEED, ACT1_STEPS = 0.5, 0.5, 0, 4
N_EST, EST_SEED = 200, 0                 # act 3: gradient estimates in the scatter
PANEL_X = 4.1                            # x-centre of the right-hand panel
STATS_X = 0.6                            # left edge of the act-3 stats table
CHART_X = -2.7                           # x-centre of the left-hand chart


def num(v, d=2):
    return f"{v:.{d}f}".replace("-", "−")


def pair(v):
    return f"({num(v[0])}, {num(v[1])})"


class Part06(Scene):
    def construct(self):
        self.title = title_card(self, 6, "REINFORCE: the policy gradient")
        self.cap = Captioner(self, self.title)
        self.act1(); clear_act(self, keep=[self.title])
        self.act2(); clear_act(self, keep=[self.title])
        self.act3()
        takeaway(self, "policy gradients optimise the policy directly, one noisy sample at a time")
        roadmap_outro(self, 6)

    # ------------------------------------------------------------------ act 1: push up what paid off
    def act1(self):
        cap = self.cap
        ax = Axes(x_range=[0, 4, 1], y_range=[0, 1, 0.25], x_length=5.6, y_length=4.4, tips=False,
                  axis_config={"color": REF_C, "stroke_opacity": 0.6},
                  y_axis_config={"numbers_to_include": [0, 0.5, 1], "font_size": 22,
                                 "decimal_number_config": {"num_decimal_places": 1}},
                  x_axis_config={"include_ticks": False}).move_to((CHART_X + 0.4) * RIGHT + 0.75 * DOWN)
        ylab = MathTex(r"\pi_\theta(a)", font_size=34, color=POLICY_C).next_to(ax.y_axis, LEFT, buff=0.3)
        acts = VGroup(*[MathTex(f"a_{i + 1}", font_size=34, color=MUTED).move_to(ax.c2p(i + 1, 0) + 0.4 * DOWN)
                        for i in range(3)])

        def bars(p):
            return VGroup(*[bar(ax, i + 1, p[i], POLICY_C, width=0.8) for i in range(3)])

        def labels(p):
            return VGroup(*[Text(num(p[i]), font_size=26).next_to(ax.c2p(i + 1, p[i]), UP, buff=0.12)
                            for i in range(3)])

        p0 = softmax(THETA0)
        bs, ls = bars(p0), labels(p0)
        cap.say("a policy is a probability for each action")
        self.play(Create(ax), FadeIn(ylab), FadeIn(acts), run_time=1.0)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bs], lag_ratio=0.15), FadeIn(ls))

        rule = MathTex(r"\theta", r"\leftarrow", r"\theta", "+", r"\alpha", r"\,R\,", r"\nabla_\theta \log \pi_\theta(a)",
                       font_size=36).move_to(PANEL_X * RIGHT + 1.5 * UP)
        rule[5].set_color(CORRECT); rule[4].set_color(STD_C)
        alpha = MathTex(rf"\alpha = {ACT1_LR:g}", font_size=30, color=STD_C).next_to(rule, DOWN, buff=0.25)
        cap.say("sample an action; raise its probability in proportion to the reward")
        self.play(Write(rule), FadeIn(alpha))

        readout = VGroup()
        for k, (a, R, p, p_new) in enumerate(policy_steps(THETA0, ACT1_STEPS, ACT1_LR, ACT1_NOISE, ACT1_SEED)):
            new_readout = VGroup(
                Text(f"step {k + 1}", font_size=28),
                VGroup(Text("sampled", font_size=24, color=MUTED), MathTex(f"a_{a + 1}", font_size=34, color=POLICY_C)
                       ).arrange(RIGHT, buff=0.2),
                VGroup(Text("reward", font_size=24, color=MUTED), MathTex(f"R = {num(R)}", font_size=34, color=CORRECT)
                       ).arrange(RIGHT, buff=0.2),
                VGroup(MathTex(rf"\pi(a_{a + 1}):", font_size=32),
                       MathTex(rf"{num(p[a])} \to {num(p_new[a])}", font_size=32, color=POLICY_C)).arrange(RIGHT, buff=0.2),
            ).arrange(DOWN, buff=0.28).move_to(PANEL_X * RIGHT + 1.15 * DOWN)
            pointer = Arrow(ax.c2p(a + 1, p[a]) + 1.3 * UP, ax.c2p(a + 1, p[a]) + 0.45 * UP, buff=0,
                            color=POLICY_C, stroke_width=5, max_tip_length_to_length_ratio=0.35)
            anims = [FadeIn(new_readout[:2])] + ([FadeOut(readout)] if len(readout) else [])
            self.play(GrowArrow(pointer), Indicate(acts[a], color=POLICY_C, scale_factor=1.4), *anims, run_time=0.9)
            tag = MathTex(f"R = {num(R)}", font_size=32, color=CORRECT).next_to(pointer, UP, buff=0.1)
            self.play(FadeIn(tag, shift=0.2 * DOWN), FadeIn(new_readout[2]), Indicate(rule[5], color=CORRECT),
                      run_time=0.8)
            nb, nl = bars(p_new), labels(p_new)
            shift = (ax.c2p(0, p_new[a])[1] - ax.c2p(0, p[a])[1]) * UP
            self.play(Transform(bs, nb), Transform(ls, nl), pointer.animate.shift(shift), tag.animate.shift(shift),
                      FadeIn(new_readout[3]), run_time=1.2)
            self.wait(0.8)
            self.play(FadeOut(pointer), FadeOut(tag), run_time=0.4)
            readout = new_readout

        best = int(Q_TRUE.argmax())
        cap.say(f"after {ACT1_STEPS} samples, a_{best + 1} (the best action) is favoured", color=POLICY_C)
        self.play(Indicate(acts[best], color=CORRECT, scale_factor=1.5), bs[best].animate.set_stroke(CORRECT, 4))
        self.wait(1.8)

    # ------------------------------------------------------------------ act 2: the log-derivative trick
    def act2(self):
        cap = self.cap
        fs = 40
        l1 = MathTex(r"J(\theta)", "=", r"\sum_a", r"\pi_\theta(a)", r"\, r(a)", font_size=fs)
        l2 = MathTex(r"\nabla_\theta J", "=", r"\sum_a", r"\nabla_\theta \pi_\theta(a)", r"\, r(a)", font_size=fs)
        l3 = MathTex(r"\nabla_\theta J", "=", r"\sum_a", r"\pi_\theta(a)\, \nabla_\theta \log \pi_\theta(a)", r"\, r(a)",
                     font_size=fs)
        l4 = MathTex(r"\nabla_\theta J", "=", r"\mathbb{E}_{a \sim \pi_\theta}", r"\big[\, R \; \nabla_\theta \log \pi_\theta(a) \,\big]",
                     font_size=fs + 4)
        lines = VGroup(l1, l2, l3, l4).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(2.2 * LEFT + 0.55 * DOWN)
        for l in (l2, l3, l4):
            l.shift((l1[1].get_x() - l[1].get_x()) * RIGHT)

        cap.say("goal: climb the expected reward J(θ)")
        self.play(Write(l1))
        self.wait(0.8)
        self.play(TransformMatchingTex(l1.copy(), l2), run_time=1.2)
        self.wait(0.5)

        trick = MathTex(r"\nabla p", "=", r"p \, \nabla \log p", font_size=46, color=MEAN_C)
        box = SurroundingRectangle(trick, color=MEAN_C, buff=0.25, corner_radius=0.1)
        trick_name = Text("log-derivative trick", font_size=26, color=MEAN_C).next_to(box, UP, buff=0.2)
        tgroup = VGroup(trick_name, trick, box).move_to(PANEL_X * RIGHT + 1.1 * UP)
        cap.say("the log-derivative trick puts π back in front of the gradient")
        self.play(FadeIn(trick_name), Write(trick), Create(box))
        self.play(Indicate(l2[3], color=MEAN_C))
        self.play(TransformFromCopy(l2, l3), run_time=1.2)
        self.play(Indicate(l3[2:4], color=MEAN_C, scale_factor=1.05))
        self.wait(0.5)

        cap.say("a sum weighted by π is an expectation, so we can sample it")
        self.play(TransformFromCopy(l3, l4), run_time=1.2)
        frame = SurroundingRectangle(l4, color=CORRECT, buff=0.18, corner_radius=0.1)
        self.play(Create(frame))

        soft = VGroup(
            Text("softmax policy:", font_size=24, color=MUTED),
            MathTex(r"\nabla_\theta \log \pi_\theta(a) = e_a - \pi_\theta", font_size=34, color=POLICY_C),
            Text("one sample of the bracket:", font_size=24, color=MUTED),
            MathTex(r"g = R\,(e_a - \pi_\theta)", font_size=36, color=STD_C),
        ).arrange(DOWN, buff=0.22).move_to(PANEL_X * RIGHT + 1.7 * DOWN)
        self.play(FadeIn(soft[:2], shift=0.2 * UP))
        self.play(FadeIn(soft[2:], shift=0.2 * UP))
        cap.say("no model of the world needed, just samples", color=CORRECT)
        self.play(Indicate(l2[4], color=CORRECT), Indicate(l3[4], color=CORRECT))
        note = Text("r(a) is never differentiated", font_size=24, color=CORRECT).next_to(frame, DOWN, buff=0.3)
        self.play(FadeIn(note))
        self.wait(2.5)

    # ------------------------------------------------------------------ act 3: noisy but unbiased
    def act3(self):
        cap = self.cap
        s0 = reinforce_samples(THETA0, n=N_EST, seed=EST_SEED)
        b = softmax(THETA0) @ Q_TRUE
        s1 = reinforce_samples(THETA0, n=N_EST, baseline=b, seed=EST_SEED)
        g_true = true_grad(THETA0)
        x0, y0 = s0[:, :2], s1[:, :2]
        lo = np.floor(np.minimum(x0.min(0), y0.min(0)))
        hi = np.ceil(np.maximum(x0.max(0), y0.max(0)))
        unit = min(5.3 / (hi[1] - lo[1]), 6.0 / (hi[0] - lo[0]))
        ax = Axes(x_range=[lo[0], hi[0], 1], y_range=[lo[1], hi[1], 1],
                  x_length=unit * (hi[0] - lo[0]), y_length=unit * (hi[1] - lo[1]), tips=False,
                  axis_config={"color": REF_C, "stroke_opacity": 0.6, "font_size": 20},
                  x_axis_config={"numbers_to_include": [v for v in np.arange(lo[0], hi[0] + 1) if v]},
                  y_axis_config={"numbers_to_include": [v for v in np.arange(lo[1], hi[1] + 1) if v]})
        ax.move_to(CHART_X * RIGHT + 0.95 * DOWN)
        xl = MathTex(r"g_1", font_size=30, color=MUTED).next_to(ax.x_axis.get_end(), UP, buff=0.15)
        yl = MathTex(r"g_2", font_size=30, color=MUTED).next_to(ax.y_axis.get_end(), RIGHT, buff=0.2)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=1.0)

        dot = lambda v: Dot(ax.c2p(*v), radius=0.045, color=STD_C).set_opacity(0.45)
        arrow = Arrow(ax.c2p(0, 0), ax.c2p(*g_true[:2]), buff=0, color=MEAN_C, stroke_width=7,
                      max_tip_length_to_length_ratio=0.45, max_stroke_width_to_length_ratio=30)
        mean_mark = lambda m: Circle(radius=0.08, color=WHITE, stroke_width=2.5).move_to(ax.c2p(*m))

        legend = VGroup(
            VGroup(Dot(radius=0.06, color=STD_C), Text("one estimate g", font_size=22)).arrange(RIGHT, buff=0.2),
            VGroup(Arrow(ORIGIN, 0.5 * RIGHT, buff=0, color=MEAN_C, stroke_width=6), Text("true gradient ∇J", font_size=22)
                   ).arrange(RIGHT, buff=0.2),
            VGroup(Circle(radius=0.09, color=WHITE, stroke_width=3), Text(f"sample average (n = {N_EST})", font_size=22)
                   ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).move_to(PANEL_X * RIGHT + 1.35 * UP)

        cap.say(f"{N_EST} single-sample gradient estimates (first two components)")
        glab = MathTex(r"\nabla J", font_size=32, color=MEAN_C).next_to(arrow.get_end(), UR, buff=0.12)
        glab.add_background_rectangle(opacity=0.75, buff=0.04)
        self.play(GrowArrow(arrow), FadeIn(glab), FadeIn(legend[1]))
        n = ValueTracker(0)
        cloud = always_redraw(lambda: VGroup(*[dot(v) for v in x0[:max(int(n.get_value()), 1)]]))
        run_mean = always_redraw(lambda: mean_mark(x0[:max(int(n.get_value()), 1)].mean(0)))
        counter = always_redraw(lambda: VGroup(Text("estimates", font_size=24, color=MUTED),
                                               Integer(int(n.get_value()), font_size=30)).arrange(RIGHT, buff=0.2)
                                .move_to(PANEL_X * RIGHT + 0.35 * DOWN))
        self.add(cloud, counter)
        self.play(FadeIn(legend[0]), n.animate.set_value(12), run_time=1.5, rate_func=linear)
        self.add(run_mean)
        self.play(FadeIn(legend[2]), n.animate.set_value(N_EST), run_time=5, rate_func=linear)
        for m in (cloud, run_mean, counter):
            m.clear_updaters()
        self.bring_to_front(arrow, glab, run_mean)

        def stats_row(label, value, color):
            return VGroup(Text(label, font_size=22, color=MUTED), Text(value, font_size=24, color=color))

        def table(*rows):
            """Two columns: labels left-aligned at STATS_X, values left-aligned at STATS_X + 3.45."""
            g = VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.24)
            g.move_to(STATS_X * RIGHT + 1.1 * DOWN, aligned_edge=LEFT)
            for r in rows:
                if len(r) == 2:
                    r[0].move_to([STATS_X, r.get_y(), 0], aligned_edge=LEFT)
                    r[1].move_to([STATS_X + 3.45, r.get_y(), 0], aligned_edge=LEFT)
            return g

        stats0 = table(
            stats_row("true ∇J", pair(g_true), MEAN_C),
            stats_row(f"sample avg (n = {N_EST})", pair(x0.mean(0)), WHITE),
            stats_row("total variance", num(spread(x0)), STD_C),
        )
        cap.say("each estimate is noisy, but on average it points the right way")
        self.play(FadeOut(counter), FadeIn(stats0))

        c, w, h, ang = cov_ellipse(x0)
        ell = lambda c, w, h, ang: Ellipse(width=w * unit, height=h * unit, color=STD_C, stroke_width=3
                                           ).rotate(ang).move_to(ax.c2p(*c))
        e0 = ell(c, w, h, ang)
        self.play(Create(e0))
        self.wait(2.0)

        form = VGroup(Text("with a baseline:", font_size=24, color=CORRECT),
                      MathTex(r"g = (R - b)\,(e_a - \pi_\theta)", font_size=34, color=CORRECT),
                      MathTex(rf"b = \pi_\theta \cdot r = {num(b)}", font_size=32, color=CORRECT),
                      ).arrange(DOWN, buff=0.18).move_to(PANEL_X * RIGHT + 1.5 * UP)
        cap.say("a baseline shrinks the noise; the expected gradient stays put", color=CORRECT)
        self.play(FadeOut(legend), FadeIn(form))
        stats1 = table(
            stats_row("true ∇J", pair(g_true), MEAN_C),
            stats_row("sample avg, no b", pair(x0.mean(0)), MUTED),
            stats_row("sample avg, with b", pair(y0.mean(0)), WHITE),
            stats_row("total variance", f"{num(spread(x0))} → {num(spread(y0))}", CORRECT),
            Text("the averages differ only by sampling noise", font_size=20, color=MUTED),
        )
        target = VGroup(*[dot(v).set_color(CORRECT).set_opacity(0.45) for v in y0])
        e1 = ell(*cov_ellipse(y0)).set_color(CORRECT)
        self.play(Transform(cloud, target), Transform(e0, e1), run_mean.animate.move_to(ax.c2p(*y0.mean(0))),
                  Transform(stats0, stats1), run_time=2.5)
        self.bring_to_front(arrow, glab, run_mean)
        self.play(Indicate(arrow, color=MEAN_C, scale_factor=1.3))
        self.wait(3.0)
