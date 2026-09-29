"""Part 12 · RLHF with PPO: four models, the KL toll as a per-token reward, and GAE over tokens."""
import numpy as np
from style import *


def per_token_rewards(logp, ref_logp, R, beta=0.02):
    r = -beta * (np.asarray(logp) - np.asarray(ref_logp)); r = r.copy(); r[-1] += R; return r


def gae_tokens(rewards, values, lam=0.95):
    """gamma = 1 over one response; terminal after the last token."""
    T = len(rewards); adv = np.zeros(T); last = 0.0
    for t in reversed(range(T)):
        nv = values[t + 1] if t + 1 < T else 0.0
        last = rewards[t] + nv - values[t] + lam * last; adv[t] = last
    return adv


# Illustrative numbers for one sampled response (the notebook's arithmetic-task token format).
BETA, LAM = 0.02, 0.95                    # notebook defaults (gamma = 1)
PAIR = (14, 17)                           # prompt tokens: 1 4 + 1 7 =
PROMPT = f"{PAIR[0]}+{PAIR[1]}="
TOKENS = list(str(sum(PAIR))) + ["<eos>"]  # response tokens: 3 1 <eos>
LOGP = [-0.25, -0.10, -0.02]              # log π_θ(a_t | s_t) of the sampled tokens
REF_LOGP = [-2.55, -1.60, -0.07]          # log π_ref(a_t | s_t)
R_FINAL = 1.0                             # reward-model score of the whole response
VALUES = [0.45, 0.85, 0.90]               # V(s_t): value model on the prefix before token t
T_LEN = len(TOKENS)

PANEL_X = 4.4     # x-centre of the right-hand panel
CHART_X = -2.6    # x-centre of the left-hand chart


def kl_only():
    """The KL toll alone: per_token_rewards with no final reward."""
    return per_token_rewards(LOGP, REF_LOGP, R=0.0, beta=BETA)


def shaped():
    return per_token_rewards(LOGP, REF_LOGP, R=R_FINAL, beta=BETA)


def advantages():
    return gae_tokens(shaped(), VALUES, lam=LAM)


def fmt(v, d=3, sign=True):
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "−")


def tex(v, d=3, sign=True):
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def model_box(name, color, trained, width=3.2, height=0.95):
    rect = RoundedRectangle(corner_radius=0.15, width=width, height=height)
    if trained:
        rect.set_stroke(color, 4).set_fill(color, 0.18)
    else:
        rect = DashedVMobject(rect.set_stroke(color, 3), num_dashes=36).set_fill(opacity=0)
    label = Text(name, font_size=28, color=color).move_to(rect)
    tag = Text("trained" if trained else "frozen", font_size=18,
               color=CORRECT if trained else REF_C).next_to(rect, DOWN, buff=0.08).align_to(rect, RIGHT)
    return VGroup(rect, label, tag)


def token_card(s, color=WHITE, stroke=MUTED, font_size=30):
    t = Text(s, font="Monospace", font_size=font_size, color=color)
    box = RoundedRectangle(corner_radius=0.1, width=max(0.7, t.width + 0.3), height=0.7).set_stroke(stroke, 2)
    return VGroup(box, t.move_to(box))


class Part12(Scene):
    def construct(self):
        self.title = title_card(self, 12, "RLHF with PPO: four models, one response")
        self.cap = Captioner(self, self.title)
        self.act1(); clear_act(self, keep=[self.title])
        self.act2(); clear_act(self, keep=[self.title])
        self.act3()
        takeaway(self, "powerful, but heavy: the next parts remove pieces of this machine")
        roadmap_outro(self, 12)

    # ------------------------------------------------------------------ act 1: four models
    def act1(self):
        cap = self.cap
        cap.say("ppo-rlhf juggles four copies of a language model")

        prompt = token_card(PROMPT, color=MUTED).move_to(LEFT * 5.4 + 1.2 * UP)
        policy = model_box("policy", POLICY_C, True, width=2.4).move_to(LEFT * 5.4 + 0.6 * DOWN)
        response = VGroup(*[token_card(t, color=POLICY_C, stroke=POLICY_C) for t in TOKENS]
                          ).arrange(RIGHT, buff=0.1).move_to(LEFT * 1.25 + 0.6 * DOWN)
        ref = model_box("reference", REF_C, False).move_to(RIGHT * 2.6 + 1.25 * UP)
        rm = model_box("reward model", MEAN_C, False).move_to(RIGHT * 2.6 + 0.6 * DOWN)
        vm = model_box("value model", VALUE_C, True).move_to(RIGHT * 2.6 + 2.45 * DOWN)

        self.play(FadeIn(prompt, shift=0.2 * DOWN))
        a_prompt = Arrow(prompt.get_bottom(), policy.get_top(), buff=0.1, color=MUTED, stroke_width=3)
        self.play(GrowArrow(a_prompt), FadeIn(policy))
        a_gen = Arrow(policy[0].get_right(), response.get_left(), buff=0.1, color=POLICY_C, stroke_width=3)
        gen_lab = MathTex(r"\log\pi_\theta(a_t)", font_size=34, color=POLICY_C).next_to(response, UP, buff=0.2)
        self.play(GrowArrow(a_gen), LaggedStart(*[FadeIn(c, shift=0.2 * RIGHT) for c in response], lag_ratio=0.25))
        self.play(FadeIn(gen_lab))
        cap.say("the policy samples a response, token by token")
        self.wait(0.8)

        # The response fans out to the three other models.
        outs = [
            (ref, MathTex(r"\log\pi_{\mathrm{ref}}(a_t)", font_size=36, color=REF_C)),
            (rm, MathTex(r"R", font_size=40, color=MEAN_C)),
            (vm, MathTex(r"V(s_t)", font_size=36, color=VALUE_C)),
        ]
        cap.say("three more models read the same response")
        fan = VGroup()
        for box, lab in outs:
            a_in = Arrow(response.get_right(), box[0].get_left(), buff=0.12, color=MUTED, stroke_width=3)
            a_out = Arrow(box[0].get_right(), box[0].get_right() + 0.8 * RIGHT, buff=0.08,
                          color=box[1].get_color(), stroke_width=3)
            lab.next_to(a_out, RIGHT, buff=0.1)
            self.play(GrowArrow(a_in), FadeIn(box), run_time=0.7)
            self.play(GrowArrow(a_out), FadeIn(lab), run_time=0.6)
            fan.add(a_in, a_out, lab)
        self.wait(1)

        # Trained vs frozen.
        trained = VGroup(policy, vm)
        frozen = VGroup(ref, rm)
        cap.say("two are trained: the policy and the value model", color=CORRECT)
        self.play(*[Indicate(b[0], color=b[1].get_color(), scale_factor=1.06) for b in trained],
                  *[Indicate(b[2], color=CORRECT) for b in trained])
        self.wait(1)
        cap.say("two stay frozen: the reference and the reward model", color=REF_C)
        self.play(frozen.animate.set_opacity(0.45), run_time=0.8)
        self.play(frozen.animate.set_opacity(1), run_time=0.8)
        self.wait(1)

        # PPO update loop back into the two trained models.
        loop = CurvedArrow(vm[0].get_bottom() + 0.1 * DOWN + 0.6 * RIGHT, policy[2].get_bottom() + 0.1 * DOWN,
                           angle=-0.55, color=CORRECT, stroke_width=3)
        loop_lab = Text("ppo update", font_size=22, color=CORRECT).next_to(loop, DOWN, buff=0.05)
        loop_lab.set_y(-3.55)
        cap.say("all four sit in memory at once for every training step")
        self.play(Create(loop), FadeIn(loop_lab), run_time=1.2)
        self.wait(2)

    # ------------------------------------------------------------------ act 2: the KL toll
    def act2(self):
        cap = self.cap
        kl, r = kl_only(), shaped()
        zoom0 = 0.7 / np.abs(kl).max()                 # KL bars fill ~70 % of the axis while zoomed
        z = ValueTracker(zoom0)                        # display scale
        k = ValueTracker(0)                            # 0 -> 1: add R on the last token

        ax = Axes(x_range=[0, T_LEN + 0.6, 1], y_range=[-1.0, 1.25, 0.5], x_length=6.2, y_length=4.4,
                  tips=False, axis_config={"color": REF_C, "stroke_opacity": 0.6},
                  x_axis_config={"include_ticks": False}, y_axis_config={"include_ticks": False}
                  ).move_to(CHART_X * RIGHT + 0.6 * DOWN)
        cards = VGroup(*[token_card(t, color=POLICY_C, stroke=POLICY_C).scale(0.85).move_to(ax.c2p(i + 1, -1.0) + 0.45 * DOWN)
                         for i, t in enumerate(TOKENS)])
        pcard = token_card(PROMPT, color=MUTED).scale(0.85).next_to(cards[0], LEFT, buff=0.55)
        ylab = Text("per-token reward", font_size=22, color=REF_C).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.25)
        ylab.set_y(ax.c2p(0, 0.1)[1])
        self.play(FadeIn(pcard), LaggedStart(*[FadeIn(c) for c in cards], lag_ratio=0.2), Create(ax), FadeIn(ylab))

        f1 = MathTex(r"r_t", "=", r"-\beta\,\big(\log\pi_\theta(a_t) - \log\pi_{\mathrm{ref}}(a_t)\big)", font_size=36)
        f2 = MathTex(r"+\,R\ \text{on the last token}", font_size=36, color=MEAN_C)
        f2.next_to(f1, DOWN, buff=0.15).align_to(f1[2], LEFT)
        formula = VGroup(*f1, f2)
        fit(formula, 5.2).move_to(PANEL_X * RIGHT + 1.35 * UP)
        formula[2].set_color(POLICY_C)
        beta_lab = MathTex(rf"\beta = {BETA}", font_size=34, color=POLICY_C).next_to(formula, DOWN, buff=0.25)
        cap.say("stay close to the reference: pay a small kl toll on every token")
        self.play(Write(formula[:3]), FadeIn(beta_lab))

        # Log-prob table (right panel), one column per token.
        rows = [("token", TOKENS, WHITE), ("log π", [fmt(v, 2, False) for v in LOGP], POLICY_C),
                ("log π_ref", [fmt(v, 2, False) for v in REF_LOGP], REF_C)]
        table = VGroup()
        for name, vals, col in rows:
            row = VGroup(Text(name, font_size=24, color=col),
                         *[Text(v, font_size=24, font="Monospace" if name == "token" else "") for v in vals])
            table.add(row)
        col_x = [PANEL_X - 1.8, PANEL_X - 0.4, PANEL_X + 0.75, PANEL_X + 1.9]
        for j, row in enumerate(table):
            for i, m in enumerate(row):
                m.move_to([col_x[i], -0.45 - 0.55 * j, 0])
            row[0].align_to([col_x[0] + 0.55, 0, 0], RIGHT)
        self.play(FadeIn(table, shift=0.2 * UP))

        disp = lambda i: z.get_value() * (kl[i] + (k.get_value() * R_FINAL if i == T_LEN - 1 else 0.0))
        bars = always_redraw(lambda: VGroup(*[bar(ax, i + 1, disp(i), POLICY_C if disp(i) < 0 else MEAN_C, width=0.55)
                                              for i in range(T_LEN)]))
        labels = always_redraw(lambda: VGroup(*[
            Text(fmt(kl[i] + (k.get_value() * R_FINAL if i == T_LEN - 1 else 0.0)), font_size=22)
            .next_to(ax.c2p(i + 1, disp(i)), UP if disp(i) >= 0 else DOWN, buff=0.1) for i in range(T_LEN)]))
        zoom_tag = Text(f"zoomed ×{zoom0:.0f}", font_size=22, color=MEAN_C).next_to(ax.c2p(T_LEN + 0.6, 1.25), DL, buff=0.1)
        self.play(Indicate(formula[2], color=POLICY_C))
        self.add(bars)
        self.play(FadeIn(labels), FadeIn(zoom_tag), run_time=1)
        self.wait(1.5)
        big = int(np.argmin(kl))
        cap.say(f"“{TOKENS[big]}” moved furthest from the reference, so it pays the most")
        self.play(Indicate(cards[big], color=POLICY_C), Indicate(table[1][big + 1]), Indicate(table[2][big + 1]))
        self.wait(1.5)

        # Zoom out and add the reward model's score on the final token.
        cap.say(f"the reward model's score R = {R_FINAL:.1f} lands on the final token", color=MEAN_C)
        self.play(Write(formula[3]))
        self.play(z.animate.set_value(1.0), FadeOut(zoom_tag), run_time=1.5)
        self.play(k.animate.set_value(1.0), run_time=1.5)
        self.wait(1)
        cap.say("at true scale the toll is tiny, but it adds up as the policy drifts")
        self.wait(2.5)
        bars.clear_updaters(); labels.clear_updaters()
        assert np.allclose([kl[i] + (R_FINAL if i == T_LEN - 1 else 0) for i in range(T_LEN)], r)

    # ------------------------------------------------------------------ act 3: GAE over tokens
    def act3(self):
        cap = self.cap
        r, V, A = shaped(), np.asarray(VALUES), advantages()

        ax = Axes(x_range=[0, T_LEN + 0.6, 1], y_range=[-0.25, 1.25, 0.5], x_length=6.2, y_length=4.4,
                  tips=False, axis_config={"color": REF_C, "stroke_opacity": 0.6},
                  x_axis_config={"include_ticks": False},
                  y_axis_config={"numbers_to_include": [0, 0.5, 1.0], "font_size": 22, "decimal_number_config": {"num_decimal_places": 1}}
                  ).move_to(CHART_X * RIGHT + 0.6 * DOWN)
        cards = VGroup(*[token_card(t, color=POLICY_C, stroke=POLICY_C).scale(0.85).move_to(ax.c2p(i + 1, -0.25) + 0.45 * DOWN)
                         for i, t in enumerate(TOKENS)])
        pcard = token_card(PROMPT, color=MUTED).scale(0.85).next_to(cards[0], LEFT, buff=0.55)
        self.play(FadeIn(pcard), FadeIn(cards), Create(ax))

        vbars = VGroup(*[bar(ax, i + 1, V[i], VALUE_C, width=0.8).set_fill(VALUE_C, 0.25).set_stroke(VALUE_C, 2, 0.6)
                         for i in range(T_LEN)])
        legend = VGroup(
            VGroup(Square(0.25).set_fill(VALUE_C, 0.25).set_stroke(VALUE_C, 2), MathTex(r"V(s_t)", font_size=28, color=VALUE_C),
                   Text("value model", font_size=20, color=MUTED)).arrange(RIGHT, buff=0.15),
            VGroup(Square(0.25).set_fill(CORRECT, 0.75).set_stroke(CORRECT, 2), MathTex(r"\hat A_t", font_size=28, color=CORRECT),
                   Text("advantage", font_size=20, color=MUTED)).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(PANEL_X * RIGHT + 1.25 * UP)
        cap.say("the critic guesses how good each prefix is before the next token")
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in vbars], lag_ratio=0.2), FadeIn(legend[0]))
        self.wait(1)

        formulas = VGroup(
            MathTex(r"\delta_t", "=", "r_t", "+", r"V(s_{t+1})", "-", r"V(s_t)", font_size=32),
            MathTex(r"\hat A_t", "=", r"\delta_t", "+", r"\lambda", r"\hat A_{t+1}", font_size=32),
        ).arrange(DOWN, buff=0.2).next_to(legend, DOWN, buff=0.35)
        for f in formulas:
            f.set_x(PANEL_X)
        formulas[0][4].set_color(VALUE_C); formulas[0][6].set_color(VALUE_C); formulas[1][4].set_color(STD_C)
        params = MathTex(rf"\gamma = 1,\ \lambda = {LAM}", font_size=28, color=STD_C).next_to(formulas, DOWN, buff=0.2)
        cap.say("the critic spreads the final reward back over the tokens")
        self.play(FadeIn(legend[1]), Write(formulas), FadeIn(params))

        # Sweep from the last token back to the first.
        cursor = Arrow(ax.c2p(T_LEN + 0.5, 1.17), ax.c2p(T_LEN - 0.5, 1.17), buff=0, color=MEAN_C, stroke_width=4,
                       max_tip_length_to_length_ratio=0.3)
        self.play(GrowArrow(cursor))
        abars, alabs, readouts = VGroup(), VGroup(), VGroup()
        for t in reversed(range(T_LEN)):
            nv = V[t + 1] if t + 1 < T_LEN else 0.0
            delta = r[t] + nv - V[t]
            nxt = A[t + 1] if t + 1 < T_LEN else 0.0
            head = Text(f"token “{TOKENS[t]}”", font_size=24, color=POLICY_C)
            line = MathTex(rf"\delta = {tex(r[t])} + {tex(nv, 2, False)} - {tex(V[t], 2, False)} = {tex(delta)}", font_size=30)
            line2 = MathTex(rf"\hat A = {tex(delta)} + {LAM} \times {tex(nxt)} = {tex(A[t])}", font_size=30, color=CORRECT)
            ro = VGroup(head, line, line2).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            fit(ro, 4.9).next_to(params, DOWN, buff=0.35).set_x(PANEL_X)
            b = bar(ax, t + 1, A[t], CORRECT if A[t] >= 0 else WRONG, width=0.45)
            lab = Text(fmt(A[t]), font_size=22, color=CORRECT if A[t] >= 0 else WRONG
                       ).next_to(ax.c2p(t + 1, A[t]), UP, buff=0.1)
            anims = [cursor.animate.move_to(ax.c2p(t + 1, 1.17)), Indicate(cards[t], color=MEAN_C)]
            if len(readouts):
                anims.append(FadeOut(readouts[-1]))
            self.play(*anims, FadeIn(ro), run_time=1)
            self.play(GrowFromEdge(b, DOWN), FadeIn(lab), run_time=1)
            abars.add(b); alabs.add(lab); readouts.add(ro)
            self.wait(1.2)
        self.play(FadeOut(cursor))

        best = int(np.argmax(A))
        cap.say(f"the decisive token “{TOKENS[best]}” gets the most credit, not the one scored")
        self.play(Indicate(abars[T_LEN - 1 - best], color=CORRECT, scale_factor=1.15), Indicate(cards[best], color=CORRECT))
        self.wait(1.5)
        cap.say("the critic's jump at that token is where the credit comes from", color=VALUE_C)
        self.play(*[Indicate(vbars[j], color=VALUE_C) for j in (best, best + 1) if j < T_LEN])
        self.wait(2)
