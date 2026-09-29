"""Part 2 · Bandits: sample averages, the update shape, and exploration (greedy vs ε-greedy vs UCB)."""
import numpy as np
from style import *


def make_bandit(k=10, seed=2):
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


K, STEPS, EPS, C = 10, 300, 0.1, 2.0
PANEL_X = 4.7      # x-centre of the right-hand panel (formula, readouts, legend)
CHART_X = -1.7     # x-centre of the left-hand chart
Y_MAX = 4.5        # reward axis half-range in act 1/2
QY = 3.5           # estimate axis half-range in act 3
N_SLOW, N_FAST = 3, 30   # act 2: pulls shown one by one, then fast-forwarded to N_FAST


def signed(v):
    return f"{v:+.2f}".replace("-", "−")


def tex_num(v):
    return f"{v:.2f}"


def violin(ax, x, mu, color, half=0.42, span=2.4):
    """Vertical Gaussian (sd 1) centred on mu, clipped to the axis range."""
    ys = np.linspace(max(mu - span, -Y_MAX), min(mu + span, Y_MAX), 60)
    w = half * np.exp(-0.5 * (ys - mu) ** 2)
    right = [ax.c2p(x + wi, y) for y, wi in zip(ys, w)]
    left = [ax.c2p(x - wi, y) for y, wi in zip(ys[::-1], w[::-1])]
    return Polygon(*right, *left).set_stroke(color, 1.5).set_fill(color, 0.3)


def mean_tick(ax, x, mu, half=0.42):
    return DashedLine(ax.c2p(x - half, mu), ax.c2p(x + half, mu), color=MEAN_C, stroke_width=3, dash_length=0.07)


def q_dot(point):
    return Dot(point, radius=0.1, color=VALUE_C).set_stroke(WHITE, 1.5)


def arm_labels(ax, y, best=None, font_size=24):
    return VGroup(*[Text(str(a + 1), font_size=font_size, color=CORRECT if a == best else MUTED)
                    .move_to(ax.c2p(a + 1, y) + 0.3 * DOWN) for a in range(K)])


def state_at(h, n):
    """(Q, N) after n pulls of history h (n = 0 is the untouched start)."""
    if n <= 0:
        return np.zeros(K), np.zeros(K)
    return h["Q"][n - 1], h["N"][n - 1]


class Part02(Scene):
    def construct(self):
        self.title = title_card(self, 2, "Bandits: exploration vs exploitation")
        self.cap = Captioner(self, self.title)
        self.q = make_bandit()
        self.act1()
        self.act2()
        clear_act(self, keep=[self.title])
        self.act3()
        takeaway(self, "exploration costs reward now and buys information later")
        roadmap_outro(self, 2)

    # ------------------------------------------------------------------ act 1: ten slot machines
    def act1(self):
        q, cap = self.q, self.cap
        self.ax = ax = Axes(x_range=[0, K + 1, 1], y_range=[-Y_MAX, Y_MAX, 1], x_length=8.0, y_length=5.4,
                            tips=False, axis_config={"color": REF_C, "stroke_opacity": 0.6},
                            y_axis_config={"numbers_to_include": [-4, -2, 0, 2, 4], "font_size": 22},
                            x_axis_config={"include_ticks": False}).move_to(CHART_X * RIGHT + 0.3 * DOWN)
        self.ylab = Text("reward", font_size=24, color=REF_C).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.45)
        self.arms = arm_labels(ax, -Y_MAX)
        arm_word = Text("arm", font_size=22, color=MUTED).next_to(self.arms[0], LEFT, buff=0.45)
        self.arms.add(arm_word)
        self.play(Create(ax), FadeIn(self.ylab), FadeIn(self.arms), run_time=1.2)

        self.violins = VGroup(*[violin(ax, a + 1, q[a], STD_C) for a in range(K)])
        self.ticks = VGroup(*[mean_tick(ax, a + 1, q[a]) for a in range(K)])
        cap.say("each arm pays a noisy reward around an unknown mean")
        self.play(LaggedStart(*[GrowFromCenter(v) for v in self.violins], lag_ratio=0.08), run_time=1.6)
        self.play(LaggedStart(*[Create(t) for t in self.ticks], lag_ratio=0.05), run_time=0.8)

        self.qdots = VGroup(*[q_dot(ax.c2p(a + 1, 0)) for a in range(K)])
        v_icon = violin(ax, 0, 0, STD_C).copy().scale(0.35)
        self.legend = VGroup(
            VGroup(v_icon, Text("reward distribution", font_size=22)).arrange(RIGHT, buff=0.2),
            VGroup(DashedLine(ORIGIN, 0.5 * RIGHT, color=MEAN_C, stroke_width=3, dash_length=0.07),
                   Text("true mean", font_size=22), MathTex(r"q^*(a)", font_size=30, color=MEAN_C)
                   ).arrange(RIGHT, buff=0.2),
            VGroup(q_dot(ORIGIN), Text("our estimate", font_size=22), MathTex(r"\hat{Q}(a)", font_size=30, color=VALUE_C)
                   ).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(PANEL_X * RIGHT + 0.6 * UP)
        self.play(LaggedStart(*[FadeIn(d, scale=0.5) for d in self.qdots], lag_ratio=0.04), FadeIn(self.legend))
        self.wait(1.5)

        cap.say("we only see the rewards we sample")
        self.play(self.violins.animate.set_fill(opacity=0.07).set_stroke(opacity=0.25),
                  self.ticks.animate.set_stroke(opacity=0.45),
                  self.legend[0].animate.set_opacity(0.3), self.legend[1].animate.set_opacity(0.45))
        self.wait(1.5)

    # ------------------------------------------------------------------ act 2: the update shape
    def act2(self):
        ax, q, cap = self.ax, self.q, self.cap
        self.play(FadeOut(self.legend))
        self.remove(self.legend)
        h = run_agent(q, "eps", eps=EPS, c=C, steps=STEPS)

        formula = update_shape(r"r_n", CORRECT).move_to(PANEL_X * RIGHT + 1.6 * UP)
        if formula.width > 4.2:
            formula.scale_to_fit_width(4.2)
        alpha = MathTex(r"\alpha = 1/N", font_size=34, color=STD_C).next_to(formula[4], DOWN, buff=0.3)
        cap.say("new estimate = old + step × (target − old)")
        self.play(Write(formula))
        self.play(FadeIn(alpha, shift=0.2 * UP))

        rng = np.random.default_rng(1)
        jit = rng.uniform(-0.2, 0.2, STEPS)
        sample = lambda j: Dot(ax.c2p(h["a"][j] + 1 + jit[j], h["r"][j]), radius=0.055,
                               color=CORRECT).set_opacity(0.85)
        samples = VGroup()
        readout = VGroup()
        for i in range(N_SLOW):
            a, r = h["a"][i], h["r"][i]
            N = int(h["N"][i][a])
            old, new = state_at(h, i)[0][a], h["Q"][i][a]
            x = a + 1
            dot = sample(i)
            start = dot.copy().move_to(ax.c2p(x + jit[i], Y_MAX))
            new_readout = VGroup(
                Text(f"pull {i + 1} · arm {a + 1}", font_size=26),
                MathTex(rf"r_{{{N}}} = {tex_num(r)}", font_size=32, color=CORRECT),
                MathTex(rf"\hat{{Q}} \leftarrow {tex_num(old)} + \tfrac{{1}}{{{N}}}\big({tex_num(r)} - {tex_num(old)}\big)",
                        rf"= {tex_num(new)}", font_size=32).arrange(DOWN, buff=0.15),
            ).arrange(DOWN, buff=0.25).move_to(PANEL_X * RIGHT + 1.0 * DOWN)
            new_readout[2][1].set_color(VALUE_C)
            fit(new_readout, 4.2)
            self.play(FadeIn(start), run_time=0.3)
            self.play(start.animate.move_to(dot), run_time=0.6, rate_func=rush_into)
            self.remove(start); self.add(dot); samples.add(dot)
            err = Line(ax.c2p(x, old), ax.c2p(x, r), color=CORRECT, stroke_width=5)
            anims = [FadeIn(new_readout)] + ([FadeOut(readout)] if len(readout) else [])
            self.play(Create(err), *anims, run_time=0.8)
            readout = new_readout
            self.play(self.qdots[a].animate.move_to(ax.c2p(x, new)), Indicate(formula[4], color=STD_C),
                      FadeOut(err), run_time=1.2)
            self.wait(0.6)

        # Fast-forward: the remaining pulls, drawn from the same history.
        cap.say(f"{N_FAST - N_SLOW} more pulls, same rule every time")
        s = ValueTracker(N_SLOW)
        live_samples = always_redraw(lambda: VGroup(*[sample(j) for j in range(N_SLOW, int(s.get_value()))]))
        live_dots = always_redraw(lambda: VGroup(*[q_dot(ax.c2p(a + 1, state_at(h, int(s.get_value()))[0][a]))
                                                  for a in range(K)]))
        counter = always_redraw(lambda: Text(f"pull {int(s.get_value())}", font_size=26)
                                .move_to(readout[0]))
        self.remove(self.qdots)
        self.add(live_samples, live_dots)
        self.play(FadeOut(readout), FadeIn(counter), run_time=0.4)
        self.play(s.animate.set_value(N_FAST), run_time=4, rate_func=linear)
        for m in (live_samples, live_dots, counter):
            m.clear_updaters()

        # With alpha = 1/N the running estimate IS the sample average.
        a = int(np.bincount(h["a"][:N_FAST], minlength=K).argmax())
        rs = h["r"][:N_FAST][h["a"][:N_FAST] == a]
        Qa = h["Q"][N_FAST - 1][a]
        assert np.isclose(rs.mean(), Qa)
        avg_line = DashedLine(ax.c2p(a + 1 - 0.42, rs.mean()), ax.c2p(a + 1 + 0.42, rs.mean()),
                              color=VALUE_C, stroke_width=4, dash_length=0.06)
        proof = VGroup(
            Text(f"arm {a + 1}: {len(rs)} rewards", font_size=26),
            MathTex(rf"\text{{average}} = {tex_num(rs.mean())} = \hat{{Q}}({a + 1})", font_size=32, color=VALUE_C),
        ).arrange(DOWN, buff=0.25).move_to(PANEL_X * RIGHT + 0.9 * DOWN)
        cap.say("with α = 1/N the estimate is exactly the average reward", color=VALUE_C)
        self.play(FadeOut(counter), FadeIn(proof), Create(avg_line))
        self.play(Indicate(live_dots[a], color=VALUE_C, scale_factor=1.6))
        self.wait(2.5)

    # ------------------------------------------------------------------ act 3: greedy vs ε-greedy vs UCB
    def act3(self):
        q, cap = self.q, self.cap
        best = int(q.argmax())
        qax = Axes(x_range=[0, K + 1, 1], y_range=[-QY, QY, 1], x_length=8.0, y_length=3.0, tips=False,
                   axis_config={"color": REF_C, "stroke_opacity": 0.6},
                   y_axis_config={"numbers_to_include": [-2, 0, 2], "font_size": 22},
                   x_axis_config={"include_ticks": False})
        qax.shift(CHART_X * RIGHT - qax.c2p(K / 2 + 0.5, 0)[0] * RIGHT + (2.35 - qax.c2p(0, QY)[1]) * UP)
        cax = Axes(x_range=[0, K + 1, 1], y_range=[0, STEPS, 100], x_length=8.0, y_length=1.9, tips=False,
                   axis_config={"color": REF_C, "stroke_opacity": 0.6},
                   y_axis_config={"numbers_to_include": [100, 200, 300], "font_size": 20},
                   x_axis_config={"include_ticks": False})
        cax.shift((qax.c2p(0, 0)[0] - cax.c2p(0, 0)[0]) * RIGHT + (-1.45 - cax.c2p(0, STEPS)[1]) * UP)
        qlab = MathTex(r"\hat{Q}", font_size=34, color=VALUE_C).next_to(qax.y_axis, LEFT, buff=0.25)
        clab = Text("pulls", font_size=22, color=REF_C).rotate(PI / 2).next_to(cax.y_axis, LEFT, buff=0.25)
        arms = arm_labels(cax, 0, best=best, font_size=22)
        ticks = VGroup(*[mean_tick(qax, a + 1, q[a], half=0.3) for a in range(K)])
        self.play(Create(qax), Create(cax), FadeIn(qlab), FadeIn(clab), FadeIn(arms), FadeIn(ticks), run_time=1.2)

        T = ValueTracker(0)
        self.h, self.kind = None, "greedy"
        cur = lambda: state_at(self.h, int(round(T.get_value()))) if self.h is not None else state_at(None, 0)

        def err_bars():
            Q, N = cur()
            if self.kind != "ucb":
                return VGroup()
            b = ucb_bonus(max(int(round(T.get_value())), 1), N, C)
            g = VGroup()
            for a in range(K):
                lo, hi = np.clip([Q[a] - b[a], Q[a] + b[a]], -QY, QY)
                p0, p1 = qax.c2p(a + 1, lo), qax.c2p(a + 1, hi)
                g.add(VGroup(Line(p0, p1), Line(p0 + 0.1 * LEFT, p0 + 0.1 * RIGHT),
                             Line(p1 + 0.1 * LEFT, p1 + 0.1 * RIGHT)).set_stroke(STD_C, 3))
            return g

        ebars = always_redraw(err_bars)
        dots = always_redraw(lambda: VGroup(*[q_dot(qax.c2p(a + 1, cur()[0][a])) for a in range(K)]))
        cbars = always_redraw(lambda: VGroup(*[bar(cax, a + 1, cur()[1][a], CORRECT if a == best else POLICY_C,
                                                   width=0.5) for a in range(K)]))
        self.add(ebars, cbars, dots)

        def pct_best():
            n = int(round(T.get_value()))
            return (self.h["a"][:n] == best).mean() if n else 0.0

        def avg_r():
            n = int(round(T.get_value()))
            return self.h["r"][:n].mean() if n else 0.0

        def row(label, value, color=WHITE):
            return VGroup(Text(label, font_size=24, color=MUTED), value.set_color(color)).arrange(RIGHT, buff=0.2)

        readout = always_redraw(lambda: VGroup(
            row("step", Integer(int(round(T.get_value())), font_size=30)),
            row("best-arm pulls", Text(f"{pct_best():.0%}", font_size=26), CORRECT),
            row("avg reward", Text(signed(avg_r()), font_size=26), MEAN_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to((PANEL_X - 1.9) * RIGHT + 0.35 * DOWN, aligned_edge=LEFT))

        agents = [
            ("greedy", "greedy", MathTex(r"a_t = \arg\max_a \hat{Q}(a)", font_size=32)),
            ("eps", "ε-greedy", VGroup(MathTex(rf"\text{{prob. }} \varepsilon = {EPS:g}:\ \text{{random arm}}", font_size=32),
                                       MathTex(r"\text{else } \arg\max_a \hat{Q}(a)", font_size=32)
                                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)),
            ("ucb", "UCB", MathTex(r"\arg\max_a \hat{Q}(a) + c\sqrt{\ln t / N(a)}", font_size=30)),
        ]
        board_head = VGroup(Text("avg reward", font_size=20, color=MUTED), Text("best arm", font_size=20, color=MUTED))
        board = VGroup()
        name_m = rule_m = None
        results = {}
        for kind, name, rule in agents:
            new_name = Text(name, font_size=34, color=POLICY_C).move_to(PANEL_X * RIGHT + 2.1 * UP)
            if rule.width > 4.3:
                rule.scale_to_fit_width(4.3)
            rule.next_to(new_name, DOWN, buff=0.25)
            if name_m is None:
                self.play(FadeIn(new_name), FadeIn(rule), FadeIn(readout))
            else:
                self.play(T.animate.set_value(0), run_time=0.7)
                self.play(ReplacementTransform(name_m, new_name), FadeOut(rule_m), FadeIn(rule))
            name_m, rule_m = new_name, rule
            self.kind, self.h = kind, run_agent(q, kind, eps=EPS, c=C, steps=STEPS)
            h = self.h
            results[kind] = (h["r"].mean(), (h["a"] == best).mean())

            if kind == "greedy":
                cap.say("greedy locks onto the first arm that looked good", color=WRONG)
            elif kind == "eps":
                cap.say(f"ε-greedy pays a {EPS:.0%} exploration tax, and finds the best arm", color=CORRECT)
            else:
                cap.say("ucb: optimism in the face of uncertainty", color=STD_C)
                self.play(FadeIn(ebars), run_time=0.1)
            self.play(T.animate.set_value(STEPS), run_time=6, rate_func=linear)

            if kind == "greedy":
                lock = int(h["N"][-1].argmax())
                first = int(np.argmax(h["a"] == lock))
                cap.say(f"arm {lock + 1} paid {signed(h['r'][first])} on its first pull; greedy never left it",
                        color=WRONG)
                self.play(Indicate(cbars[lock], color=WRONG), Indicate(arms[best], color=CORRECT, scale_factor=1.5))
                self.wait(1.5)
            elif kind == "eps":
                self.wait(1.5)
            else:
                b = ucb_bonus(STEPS, h["N"][-1], C)
                rare = int(h["N"][-1].argmin())
                cap.say(f"after {STEPS} steps: bonus {b[best]:.2f} on arm {best + 1}, "
                        f"{b[rare]:.2f} on rarely tried arm {rare + 1}", color=STD_C)
                self.wait(2)

            avg, frac = results[kind]
            r = VGroup(Text(name, font_size=22, color=POLICY_C), Text(signed(avg), font_size=22, color=MEAN_C),
                       Text(f"{frac:.0%}", font_size=22, color=CORRECT))
            board.add(r)
            self._layout_board(board_head, board)
            if len(board) == 1:
                self.play(FadeIn(board_head), FadeIn(r))
            else:
                self.play(FadeIn(r))
        self.wait(1.5)
        for m in (ebars, dots, cbars, readout):
            m.clear_updaters()

    def _layout_board(self, head, board):
        cols = [PANEL_X - 2.0, PANEL_X + 0.05, PANEL_X + 1.6]
        y0 = -1.9
        head[0].move_to([cols[1], y0, 0]); head[1].move_to([cols[2], y0, 0])
        for i, r in enumerate(board):
            y = y0 - 0.45 * (i + 1)
            r[0].move_to([cols[0], y, 0], aligned_edge=LEFT)
            r[1].move_to([cols[1], y, 0]); r[2].move_to([cols[2], y, 0])
