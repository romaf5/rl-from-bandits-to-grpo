"""Part 7 · Actor-critic and GAE: advantages as credit for beating the critic, and λ as a bias/variance dial."""
import numpy as np
from style import *


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


# Act 1: six actions taken from six different states, with the return G that followed and the critic's V(s).
G_RET = np.array([8.0, 3.0, 6.5, 1.0, 9.0, 2.5])
V_S = np.array([7.0, 1.5, 7.5, 2.0, 6.5, 3.0])
ADV = G_RET - V_S

# Act 3: one CartPole-like episode (reward 1 per step, terminal after T steps) and an imperfect critic.
GAMMA, T_EP = 0.99, 12
EP_R = np.ones(T_EP)
EP_V_ERR = np.array([0.6, -0.5, 0.4, 0.9, -0.3, -0.8, 0.5, 0.2, -0.6, 0.4, -0.2, 0.3])
EP_V = (1 - GAMMA ** (T_EP - np.arange(T_EP))) / (1 - GAMMA) + EP_V_ERR   # true value-to-go + critic error
EP_VN = np.append(EP_V[1:], 0.0)
EP_DELTA = gae(EP_R, EP_V, EP_VN, GAMMA, 0.0)                              # λ = 0 is the TD error
EP_MC = gae(EP_R, EP_V, EP_VN, GAMMA, 1.0)                                 # λ = 1 is G − V
LAM_MID, LAM_SWEET = 0.5, 0.95                                             # stops on the λ sweep

PANEL_X = 4.7      # x-centre of the right-hand panel
N_ACT = len(G_RET)


class Part07(Scene):
    def construct(self):
        self.title = title_card(self, 7, "Actor-critic and GAE")
        self.cap = Captioner(self, self.title)
        self.act1(); clear_act(self, keep=[self.title])
        self.act2(); clear_act(self, keep=[self.title])
        self.act3()
        takeaway(self, "advantages = credit for beating expectations; λ picks how far to trust the critic")
        roadmap_outro(self, 7)

    # ------------------------------------------------------------------ act 1: subtract V(s)
    def act1(self):
        cap = self.cap
        ax = Axes(x_range=[0, N_ACT + 0.7, 1], y_range=[-2, 10, 2], x_length=7.6, y_length=4.9,
                  tips=False, axis_config={"color": REF_C},
                  y_axis_config={"numbers_to_include": [-2, 0, 2, 4, 6, 8, 10], "font_size": 22},
                  x_axis_config={"include_ticks": False}).shift(0.55 * DOWN + 1.9 * LEFT)
        states = VGroup(*[MathTex(f"s_{i + 1}", font_size=30, color=MUTED).move_to(ax.c2p(i + 1, -2) + 0.3 * DOWN)
                          for i in range(N_ACT)])
        t = ValueTracker(0)   # 0 -> 1: subtract V(s)
        ylab = always_redraw(lambda: Text("return G" if t.get_value() < 0.5 else "advantage A", font_size=24,
                                          color=REF_C).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.45))
        cap.say("six actions, taken from six different states")
        self.play(Create(ax), FadeIn(states), FadeIn(ylab), run_time=1.2)

        def g_color(i):
            return interpolate_color(STD_C, CORRECT if ADV[i] > 0 else WRONG, t.get_value())

        gval = lambda i: G_RET[i] - t.get_value() * V_S[i]
        gbars = always_redraw(lambda: VGroup(*[bar(ax, i + 1, gval(i), g_color(i), width=0.5) for i in range(N_ACT)]))
        g_static = VGroup(*[bar(ax, i + 1, G_RET[i], STD_C, width=0.5) for i in range(N_ACT)])
        g_labs = VGroup(*[DecimalNumber(G_RET[i], num_decimal_places=1, font_size=24, color=STD_C)
                          .next_to(ax.c2p(i + 1, G_RET[i]), UP, buff=0.1) for i in range(N_ACT)])
        cap.say("the return G that followed each action")
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in g_static], lag_ratio=0.1), FadeIn(g_labs))
        self.remove(*g_static)
        self.add(gbars)
        self.wait(0.8)

        legend = VGroup(
            VGroup(Square(0.25).set_fill(STD_C, 0.75).set_stroke(STD_C), Text("return G", font_size=22)
                   ).arrange(RIGHT, buff=0.15),
            VGroup(Square(0.25).set_fill(VALUE_C, 0.3).set_stroke(VALUE_C), Text("critic V(s)", font_size=22)
                   ).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(RIGHT * PANEL_X + 0.2 * DOWN)
        self.play(FadeIn(legend))
        cap.say("but some states are simply better to be in than others", wait=1.5)

        # V(s) bars behind the returns, with a dashed top that slides to zero as V is subtracted.
        vtop = lambda i: (1 - t.get_value()) * V_S[i]

        def vbar(i):
            return bar(ax, i + 1, vtop(i), VALUE_C, width=0.86).set_fill(VALUE_C, 0.3)

        vbars = always_redraw(lambda: VGroup(*[vbar(i) for i in range(N_ACT)]))
        vticks = always_redraw(lambda: VGroup(*[
            DashedLine(ax.c2p(i + 0.57, vtop(i)), ax.c2p(i + 1.43, vtop(i)), color=VALUE_C, stroke_width=4,
                       dash_length=0.08) for i in range(N_ACT)]))
        v_static = VGroup(*[vbar(i) for i in range(N_ACT)])
        v_labs = VGroup(*[DecimalNumber(V_S[i], num_decimal_places=1, font_size=24, color=VALUE_C)
                          .move_to(ax.c2p(i + 1, 0) + 0.28 * DOWN) for i in range(N_ACT)])   # under the axis
        cap.say("a critic V(s) predicts what usually happens; the actor gets A = G − V", color=VALUE_C)
        self.bring_to_back(v_static)
        g_up = [g_labs[i].animate.next_to(ax.c2p(i + 1, max(G_RET[i], V_S[i])), UP, buff=0.1) for i in range(N_ACT)]
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in v_static], lag_ratio=0.1), FadeIn(v_labs), *g_up)
        self.remove(*v_static)
        self.add(vbars, vticks)
        self.bring_to_back(vbars)
        self.add(gbars)   # keep the returns in front
        self.wait(1.5)

        formula = MathTex("{{A}} = {{G}} - {{V(s)}}", font_size=48).move_to(RIGHT * PANEL_X + 1.3 * UP)
        formula[2].set_color(STD_C); formula[4].set_color(VALUE_C)
        self.play(Write(formula))

        # The GRPO move: slide every bar down by its own baseline.
        self.play(FadeOut(g_labs), FadeOut(v_labs))
        self.play(t.animate.set_value(1), formula[0].animate.set_color(CORRECT), run_time=2.5)
        a_labs = VGroup(*[value_label(ax, i + 1, ADV[i], font_size=24) for i in range(N_ACT)])
        self.play(LaggedStart(*[FadeIn(l) for l in a_labs], lag_ratio=0.08))
        self.wait(1)

        neg = [i for i in range(N_ACT) if ADV[i] < 0]
        pos = [i for i in range(N_ACT) if ADV[i] > 0]
        i_bad = max(neg, key=lambda i: G_RET[i])     # big return, still below expectations
        i_good = min(pos, key=lambda i: G_RET[i])    # small return, but beat expectations
        box = lambda i: SurroundingRectangle(VGroup(states[i], a_labs[i]), color=MEAN_C, buff=0.12)
        hb = box(i_bad)
        self.play(Create(hb))
        cap.say(f"s{i_bad + 1}: return {G_RET[i_bad]:.1f}, but V = {V_S[i_bad]:.1f}"
                f" expected more: A = {ADV[i_bad]:+.1f}".replace("-", "−"), color=WRONG, wait=2)
        self.play(Transform(hb, box(i_good)))
        cap.say(f"s{i_good + 1}: return only {G_RET[i_good]:.1f}, but it beat V = {V_S[i_good]:.1f}:"
                f" A = {ADV[i_good]:+.1f}", color=CORRECT, wait=2)
        self.play(FadeOut(hb))
        cap.say("only actions that beat expectations get credit", wait=1.5)
        for m in (gbars, vbars, vticks, ylab):
            m.clear_updaters()

    # ------------------------------------------------------------------ act 2: two networks
    def act2(self):
        cap = self.cap

        def node(name, sym, color, w=3.0):
            box = RoundedRectangle(corner_radius=0.2, width=w, height=1.35).set_stroke(color, 3).set_fill(color, 0.12)
            txt = VGroup(Text(name, font_size=30, color=color), MathTex(sym, font_size=36)).arrange(DOWN, buff=0.12)
            return VGroup(box, txt.move_to(box))

        actor = node("actor", r"\pi_\theta(a\,|\,s)", POLICY_C).move_to(np.array([-0.9, 1.0, 0]))
        critic = node("critic", r"V_\phi(s)", VALUE_C).move_to(np.array([-0.9, -2.3, 0]))
        env = node("rollouts", r"s_t,\ a_t,\ r_t", MUTED).move_to(np.array([-5.0, -0.65, 0]))

        a_env = Arrow(actor[0].get_left(), env[0].get_top(), buff=0.1, color=POLICY_C)
        a_ret = Arrow(env[0].get_bottom(), critic[0].get_left(), buff=0.1, color=STD_C)
        a_adv = Arrow(critic[0].get_top(), actor[0].get_bottom(), buff=0.1, color=VALUE_C)
        l_env = Text("acts", font_size=24, color=POLICY_C).next_to(a_env.get_center(), UL, buff=0.1)
        l_ret = VGroup(Text("returns", font_size=24, color=STD_C), MathTex("G", font_size=34, color=STD_C)
                       ).arrange(RIGHT, buff=0.12).next_to(a_ret.get_center(), DL, buff=0.1)
        l_adv = MathTex(r"A = G - V", font_size=34, color=VALUE_C).next_to(a_adv, RIGHT, buff=0.2)

        cap.say("two networks share the work")
        self.play(FadeIn(actor, shift=0.2 * DOWN), FadeIn(critic, shift=0.2 * UP))
        self.play(FadeIn(env), GrowArrow(a_env), FadeIn(l_env))

        pi_loss = VGroup(Text("actor: policy gradient", font_size=24, color=POLICY_C),
                         MathTex(r"\nabla_\theta \log \pi_\theta(a|s)\ \cdot\ {{A}}", font_size=36)
                         ).arrange(DOWN, buff=0.2)
        pi_loss[1][1].set_color(VALUE_C)
        v_loss = VGroup(Text("critic: regression", font_size=24, color=VALUE_C),
                        MathTex(r"\big(V_\phi(s) - {{G}}\big)^2", font_size=36)).arrange(DOWN, buff=0.2)
        v_loss[1][1].set_color(STD_C)
        panel = VGroup(pi_loss, v_loss).arrange(DOWN, buff=0.8).move_to(RIGHT * PANEL_X + 0.6 * DOWN)

        def pulse(arrow, color):
            d = Dot(arrow.get_start(), radius=0.1, color=color).set_stroke(WHITE, 1.5)
            return Succession(FadeIn(d, run_time=0.1), MoveAlongPath(d, Line(arrow.get_start(), arrow.get_end())),
                              FadeOut(d, run_time=0.1))

        cap.say("the returns go into the critic's regression", color=STD_C)
        self.play(GrowArrow(a_ret), FadeIn(l_ret), FadeIn(v_loss))
        self.play(pulse(a_ret, STD_C), Indicate(critic, color=VALUE_C, scale_factor=1.05), run_time=1.2)
        self.wait(0.8)

        cap.say("the critic's A goes into the actor's gradient", color=VALUE_C)
        self.play(GrowArrow(a_adv), FadeIn(l_adv), FadeIn(pi_loss))
        self.play(pulse(a_adv, VALUE_C), Indicate(actor, color=POLICY_C, scale_factor=1.05), run_time=1.2)
        self.wait(0.8)

        cap.say("the critic evaluates, the actor improves")
        for _ in range(2):
            self.play(pulse(a_env, POLICY_C), run_time=0.8)
            self.play(pulse(a_ret, STD_C), Indicate(v_loss[1], color=STD_C), run_time=0.8)
            self.play(pulse(a_adv, VALUE_C), Indicate(pi_loss[1], color=VALUE_C), run_time=0.8)
        self.wait(1.5)

    # ------------------------------------------------------------------ act 3: the λ dial
    def act3(self):
        cap = self.cap
        lam = ValueTracker(0.0)
        ax = Axes(x_range=[-0.7, T_EP - 0.3, 1], y_range=[0, 1.05, 0.5], x_length=7.6, y_length=2.3,
                  tips=False, axis_config={"color": REF_C},
                  y_axis_config={"numbers_to_include": [0, 0.5, 1], "font_size": 22},
                  x_axis_config={"include_ticks": False}).move_to(np.array([-1.1, 0.55, 0]))
        ylab = MathTex(r"(\gamma\lambda)^l", font_size=30, color=STD_C).next_to(ax.y_axis, LEFT, buff=0.45)
        w = lambda: gae_weights(GAMMA, lam.get_value(), T_EP)
        wbars = always_redraw(lambda: VGroup(*[bar(ax, l, w()[l], STD_C, width=0.5) for l in range(T_EP)]))

        dmax = np.abs(EP_DELTA).max()
        tile_w = ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0] - 0.08

        def tiles():
            ws = w()
            g = VGroup()
            for l in range(T_EP):
                col = CORRECT if EP_DELTA[l] >= 0 else WRONG
                sq = Square(tile_w).set_stroke(col, 2).set_fill(col, 0.1 + 0.55 * ws[l] * abs(EP_DELTA[l]) / dmax)
                sq.move_to(np.array([ax.c2p(l, 0)[0], -1.25, 0]))
                g.add(VGroup(sq, MathTex(rf"\delta_{{{l}}}", font_size=26).move_to(sq)))
            return g

        tile_row = always_redraw(tiles)
        ep_lab = Text("TD errors along one episode", font_size=22, color=MUTED).next_to(
            np.array([ax.c2p(0, 0)[0] - tile_w / 2, -1.25 - tile_w / 2, 0]), DOWN, buff=0.12, aligned_edge=LEFT)

        cap.say("one episode: a TD error δ at every step")
        self.play(FadeIn(tile_row), FadeIn(ep_lab))
        self.play(Create(ax), FadeIn(ylab), FadeIn(wbars))

        formula = MathTex(r"\hat A_0 = \sum_l (\gamma\lambda)^l\, \delta_l", font_size=38
                          ).move_to(RIGHT * PANEL_X + 1.4 * UP)
        readout = always_redraw(lambda: VGroup(
            VGroup(MathTex(r"\lambda =", font_size=34), DecimalNumber(lam.get_value(), num_decimal_places=2,
                                                                     font_size=34)).arrange(RIGHT, buff=0.15),
            VGroup(MathTex(r"\hat A_0 =", font_size=34),
                   DecimalNumber(gae(EP_R, EP_V, EP_VN, GAMMA, lam.get_value())[0], num_decimal_places=2,
                                 include_sign=True, font_size=34, color=MEAN_C)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(formula, DOWN, buff=0.45))
        refs = VGroup(
            MathTex(rf"\text{{TD: }}\delta_0 = {EP_DELTA[0]:+.2f}", font_size=28, color=MUTED),
            MathTex(rf"\text{{MC: }}G_0 - V_0 = {EP_MC[0]:+.2f}", font_size=28, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(readout, DOWN, buff=0.45, aligned_edge=LEFT)
        self.play(Write(formula), FadeIn(readout))

        # λ slider with its two ends labelled.
        slider = NumberLine(x_range=[0, 1, 0.25], length=7.0, color=REF_C, include_numbers=False
                           ).move_to(np.array([-1.35, -2.4, 0]))
        knob = always_redraw(lambda: Dot(slider.n2p(lam.get_value()), radius=0.12, color=MEAN_C))
        end0 = VGroup(Text("λ=0: td error", font_size=22),
                      Text("(low variance, biased)", font_size=20, color=MUTED)).arrange(DOWN, buff=0.06)
        end1 = VGroup(Text("λ=1: monte carlo", font_size=22),
                      Text("(unbiased, noisy)", font_size=20, color=MUTED)).arrange(DOWN, buff=0.06)
        end0.next_to(slider.n2p(0), DOWN, buff=0.3)
        end1.next_to(slider.n2p(1), DOWN, buff=0.3)
        self.play(FadeOut(ep_lab), Create(slider), FadeIn(knob), FadeIn(end0), FadeIn(end1))

        cap.say(f"λ = 0: only δ_0 counts, weight 1 (A = {EP_DELTA[0]:+.2f})".replace("-", "−"), wait=1)
        self.play(FadeIn(refs[0]), Indicate(end0))
        self.wait(1)

        cap.say("raise λ: later TD errors join in, with shrinking weights")
        self.play(lam.animate.set_value(LAM_MID), run_time=2.5, rate_func=linear)
        self.wait(1)
        cap.say(f"λ = {LAM_SWEET} is the usual sweet spot (A2C, PPO)")
        self.play(lam.animate.set_value(LAM_SWEET), run_time=2.5, rate_func=linear)
        self.wait(1.5)
        cap.say(f"λ = 1: every step counts, A = G − V ({EP_MC[0]:+.2f})".replace("-", "−"))
        self.play(lam.animate.set_value(1.0), run_time=1.5)
        self.play(FadeIn(refs[1]), Indicate(end1))
        self.wait(1)
        cap.say("gae blends every n-step estimate with weights (γλ)^l", wait=1)
        self.play(lam.animate.set_value(0.0), run_time=2.5)
        self.play(lam.animate.set_value(LAM_SWEET), run_time=2.5)
        self.wait(1.5)
        for m in (wbars, tile_row, readout, knob):
            m.clear_updaters()
