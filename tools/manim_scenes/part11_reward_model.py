"""Part 11 · Reward models: preferences, the Bradley–Terry model and Goodhart's law."""
import numpy as np
from style import *


def bt_prob(r_w, r_l):
    return 1 / (1 + np.exp(-(r_w - r_l)))


def bt_loss(r_w, r_l):
    return -np.log(bt_prob(r_w, r_l))


def best_of_n_curves(ns=(1, 2, 4, 8, 16, 32, 64, 128, 256), trials=4000, seed=0):
    """Candidates: true quality q ~ N(0,1), length L ~ N(0,1). Proxy RM rewards length (bias 0.8),
    true reward punishes rambling. Pick best-of-n by proxy; report mean proxy and mean true reward."""
    rng = np.random.default_rng(seed); proxy_m, true_m = [], []
    for n in ns:
        q = rng.standard_normal((trials, n)); L = rng.standard_normal((trials, n))
        proxy = q + 0.8 * L; true = q - 0.5 * np.maximum(L, 0) ** 2
        pick = proxy.argmax(1); rows = np.arange(trials)
        proxy_m.append(proxy[rows, pick].mean()); true_m.append(true[rows, pick].mean())
    return np.array(ns), np.array(proxy_m), np.array(true_m)


def bt_train(r_w0=-0.5, r_l0=0.5, lr=0.8, steps=30):
    """Gradient descent on bt_loss for one pair: dL/dr_w = -(1 - σ(Δ)), dL/dr_l = +(1 - σ(Δ))."""
    rw, rl = [r_w0], [r_l0]
    for _ in range(steps):
        g = 1 - bt_prob(rw[-1], rl[-1])
        rw.append(rw[-1] + lr * g); rl.append(rl[-1] - lr * g)
    return np.array(rw), np.array(rl)


# The notebook's synthetic annotator (sec11): prefers the correct answer, and on ties the longer one.
P_CORRECT, P_LONGER = 0.85, 0.85
PANEL_X = 4.6      # x-centre of the right-hand panel (formulas, readouts, legend)
CHART_X = -2.3     # x-centre of the left-hand chart
PROMPT = "14 + 17 = ?"


def signed(v):
    return f"{v:+.2f}".replace("-", "−")


def answer_card(text, width=4.6):
    body = Text(text, font="Monospace", font_size=30)
    box = RoundedRectangle(corner_radius=0.15, width=max(width, body.width + 0.6), height=1.1).set_stroke(MUTED, 2)
    return VGroup(box, body.move_to(box))


class Part11(Scene):
    def construct(self):
        self.title = title_card(self, 11, "Reward models and Bradley–Terry")
        self.cap = Captioner(self, self.title)
        self.act1(); clear_act(self, keep=[self.title])
        self.act2(); clear_act(self, keep=[self.title])
        self.act3()
        takeaway(self, "a reward model is a learned, imperfect stand-in for human judgment", keep=[self.title])
        roadmap_outro(self, 11)

    # ------------------------------------------------------------------ act 1: preferences, not scores
    def pick(self, cards, win, extra=()):
        """Annotator ticks cards[win]; labels y_w / y_l appear under the pair."""
        tick = MathTex(r"\checkmark", color=CORRECT, font_size=60).next_to(cards[win], RIGHT, buff=0.2)
        cross = MathTex(r"\times", color=WRONG, font_size=52).next_to(cards[1 - win], RIGHT, buff=0.2)
        self.play(FadeIn(tick, scale=1.5), cards[win][0].animate.set_stroke(CORRECT, 3),
                  cards[1 - win][0].animate.set_stroke(WRONG, 2), FadeIn(cross), *extra)
        yw = MathTex("y_w", color=CORRECT, font_size=36).next_to(cards[win], LEFT, buff=0.25)
        yl = MathTex("y_l", color=WRONG, font_size=36).next_to(cards[1 - win], LEFT, buff=0.25)
        self.play(FadeIn(yw), FadeIn(yl))
        return VGroup(tick, cross, yw, yl)

    def act1(self):
        cap = self.cap
        prompt = Text(PROMPT, font="Monospace", font_size=40).move_to(1.6 * UP + 1.2 * LEFT)
        cards = VGroup(answer_card("31"), answer_card("32")).arrange(DOWN, buff=0.35).next_to(prompt, DOWN, buff=0.5)
        # a minimal annotator: head + shoulders
        head = Circle(0.32).set_stroke(MUTED, 3)
        body = Arc(radius=0.6, start_angle=0, angle=PI).set_stroke(MUTED, 3).next_to(head, DOWN, buff=0.08)
        rater = VGroup(head, body)
        rater_lab = Text("annotator", font_size=24, color=MUTED).next_to(rater, DOWN, buff=0.15)
        who = VGroup(rater, rater_lab).move_to(RIGHT * PANEL_X + 0.3 * DOWN)
        self.play(FadeIn(prompt, shift=DOWN), LaggedStart(*[FadeIn(c, shift=0.3 * LEFT) for c in cards], lag_ratio=0.2),
                  FadeIn(who))
        cap.say("humans compare two answers; they can't give absolute scores")

        # "7/10?" is exactly what raters are bad at: show the score slot and strike it.
        score = Text("score: ? / 10", font_size=28, color=MUTED).next_to(who, UP, buff=0.4)
        strike = Line(score.get_left(), score.get_right(), color=WRONG, stroke_width=4)
        self.play(FadeIn(score))
        self.play(Create(strike))
        better = Text("which is better?", font_size=28, color=MEAN_C).move_to(score)
        self.play(FadeOut(VGroup(score, strike)), FadeIn(better))
        marks = self.pick(cards, 0)
        self.wait(1.5)

        # Second pair: both correct, so the notebook's annotator falls back to length.
        cap.say("our toy annotator prefers correct answers, and on ties longer ones")
        cards2 = VGroup(answer_card("31"), answer_card("31, as 14 + 17 = 31")).arrange(DOWN, buff=0.35).move_to(cards)
        self.play(FadeOut(marks), ReplacementTransform(cards, cards2))
        marks2 = self.pick(cards2, 1)
        rule = VGroup(
            Text(f"correct wins: {P_CORRECT:.0%}", font_size=24, color=CORRECT),
            Text(f"tie → longer: {P_LONGER:.0%}", font_size=24, color=MEAN_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(who, DOWN, buff=0.45)
        self.play(FadeIn(rule, shift=0.2 * UP))
        self.wait(1)
        cap.say("each label is just a pair: (prompt, y_w, y_l)", wait=2)

    # ------------------------------------------------------------------ act 2: Bradley–Terry
    def act2(self):
        cap = self.cap
        rw, rl = bt_train()
        dmin, dmax = -4.5, 4.5
        cfg = dict(tips=False, axis_config={"color": REF_C, "font_size": 22})
        top = Axes(x_range=[dmin, dmax, 1], y_range=[0, 1, 0.5], x_length=7.2, y_length=2.2,
                   y_axis_config={"numbers_to_include": [0, 0.5, 1], "decimal_number_config": {"num_decimal_places": 1}},
                   x_axis_config={"include_ticks": True}, **cfg).move_to(CHART_X * RIGHT + 0.55 * UP)
        bot = Axes(x_range=[dmin, dmax, 1], y_range=[0, 4, 1], x_length=7.2, y_length=2.0,
                   y_axis_config={"numbers_to_include": [0, 2, 4]},
                   x_axis_config={"numbers_to_include": [-4, -2, 0, 2, 4]}, **cfg).next_to(top, DOWN, buff=0.35)
        bot.align_to(top, LEFT)
        ylab_t = MathTex(r"P(y_w \succ y_l)", font_size=28, color=POLICY_C).next_to(top, UP, buff=0.1).align_to(top, LEFT)
        ylab_b = MathTex(r"-\log\sigma(\Delta)", font_size=28, color=WRONG).move_to(bot.c2p(1.5, 3.3))
        xlab = MathTex(r"\Delta = r_w - r_l", font_size=28, color=REF_C).next_to(bot, DOWN, buff=0.15)
        sig = top.plot(lambda d: bt_prob(d, 0), x_range=[dmin, dmax], color=POLICY_C, stroke_width=4)
        # loss exceeds 4 for Δ < about -4; clip the plotted range to the axis
        dlo = -np.log(np.exp(4) - 1)
        loss = bot.plot(lambda d: bt_loss(d, 0), x_range=[dlo, dmax], color=WRONG, stroke_width=4)

        f1 = MathTex(r"P(y_w \succ y_l) = \sigma(r_w - r_l)", font_size=34)
        f2 = MathTex(r"\mathcal{L}_{\mathrm{RM}} = -\log \sigma(r_w - r_l)", font_size=34)
        VGroup(f1, f2).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * PANEL_X + 1.55 * UP)

        cap.say("turn a reward difference into a preference probability, and fit it")
        self.play(Create(top), FadeIn(ylab_t), Write(f1))
        self.play(Create(sig), run_time=1.5)
        mid = Dot(top.c2p(0, bt_prob(0, 0)), color=MEAN_C)
        mid_lab = MathTex(rf"\Delta=0 \Rightarrow {bt_prob(0, 0):.1f}", font_size=26, color=MEAN_C) \
            .next_to(mid, RIGHT, buff=0.15).shift(0.25 * DOWN)
        self.play(FadeIn(mid), FadeIn(mid_lab))
        self.wait(1)
        self.play(FadeOut(mid), FadeOut(mid_lab), Create(bot), FadeIn(ylab_b), FadeIn(xlab), Write(f2))
        self.play(Create(loss), run_time=1.5)

        # Reward bars for the pair in the right panel.
        rax = Axes(x_range=[0, 3, 1], y_range=[-2.5, 2.5, 1], x_length=2.6, y_length=2.6, tips=False,
                   axis_config={"color": REF_C}, x_axis_config={"include_ticks": False}) \
            .move_to(RIGHT * (PANEL_X - 0.9) + 1.55 * DOWN)
        t = ValueTracker(0)
        idx = lambda: t.get_value()
        r_w = lambda: np.interp(idx(), np.arange(len(rw)), rw)
        r_l = lambda: np.interp(idx(), np.arange(len(rl)), rl)
        dlt = lambda: r_w() - r_l()
        bars = always_redraw(lambda: VGroup(bar(rax, 1, r_w(), CORRECT, width=0.7), bar(rax, 2, r_l(), WRONG, width=0.7)))
        blabs = VGroup(MathTex("r_w", color=CORRECT, font_size=28).move_to(rax.c2p(1, -2.5) + 0.3 * DOWN),
                       MathTex("r_l", color=WRONG, font_size=28).move_to(rax.c2p(2, -2.5) + 0.3 * DOWN))
        readout = always_redraw(lambda: VGroup(*[
            VGroup(MathTex(lab, font_size=28, color=c), DecimalNumber(v, num_decimal_places=2, include_sign=s,
                                                                      font_size=28, color=c)).arrange(RIGHT, buff=0.15)
            for lab, v, c, s in [(r"\Delta =", dlt(), MEAN_C, True), (r"P =", bt_prob(r_w(), r_l()), POLICY_C, False),
                                 (r"\mathcal{L} =", bt_loss(r_w(), r_l()), WRONG, False)]
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(rax, RIGHT, buff=0.3))
        d_top = always_redraw(lambda: Dot(top.c2p(dlt(), bt_prob(r_w(), r_l())), color=MEAN_C, radius=0.1))
        d_bot = always_redraw(lambda: Dot(bot.c2p(dlt(), bt_loss(r_w(), r_l())), color=MEAN_C, radius=0.1))
        guide = always_redraw(lambda: DashedLine(bot.c2p(dlt(), 0), top.c2p(dlt(), bt_prob(r_w(), r_l())),
                                                 color=MEAN_C, stroke_width=2, stroke_opacity=0.6))
        cap.say(f"untrained RM ranks the pair wrong: Δ = {signed(rw[0] - rl[0])}", color=WRONG)
        self.play(Create(rax), FadeIn(bars), FadeIn(blabs), FadeIn(readout), FadeIn(d_top), FadeIn(d_bot),
                  FadeIn(guide))
        self.wait(1)
        cap.say("each gradient step raises the winner, lowers the loser: the gap widens")
        self.play(t.animate.set_value(len(rw) - 1), run_time=6, rate_func=smooth)
        n = len(rw) - 1
        cap.say(f"after {n} steps: P = {bt_prob(rw[-1], rl[-1]):.2f}, loss = {bt_loss(rw[-1], rl[-1]):.2f}", wait=1.5)
        cap.say("only the difference matters: the loss flattens once Δ is large", wait=2)
        for m in (bars, readout, d_top, d_bot, guide):
            m.clear_updaters()

    # ------------------------------------------------------------------ act 3: Goodhart
    def act3(self):
        cap = self.cap
        ns, px, tr = best_of_n_curves()
        lx = np.log2(ns)
        ax = Axes(x_range=[0, lx[-1] + 0.3, 1], y_range=[-0.5, 4, 1], x_length=7.4, y_length=4.6, tips=False,
                  axis_config={"color": REF_C, "font_size": 22},
                  y_axis_config={"numbers_to_include": [0, 1, 2, 3, 4]},
                  x_axis_config={"include_ticks": True}).move_to(CHART_X * RIGHT + 0.75 * DOWN)
        xt = VGroup(*[Text(f"{n}", font_size=20, color=REF_C).next_to(ax.c2p(l, -0.5), DOWN, buff=0.15)
                      for n, l in zip(ns, lx) if int(l) % 2 == 0])
        xlab = Text("n (best-of-n, log scale)", font_size=22, color=REF_C).next_to(xt, DOWN, buff=0.12)
        ylab = Text("mean reward of the pick", font_size=22, color=REF_C).rotate(PI / 2).next_to(ax.y_axis, LEFT, buff=0.4)
        toy = Text("toy model: q, L ~ N(0,1)", font_size=20, color=MUTED).next_to(ax.c2p(0, 4), RIGHT, buff=0.3) \
            .shift(0.15 * DOWN)

        cap.say("best-of-n: sample n answers, keep the one the RM likes most")
        self.play(Create(ax), FadeIn(xt), FadeIn(xlab), FadeIn(ylab), FadeIn(toy))

        legend = VGroup(
            VGroup(Line(ORIGIN, 0.5 * RIGHT, color=MEAN_C, stroke_width=5), Text("proxy (RM score)", font_size=22)
                   ).arrange(RIGHT, buff=0.15),
            VGroup(Line(ORIGIN, 0.5 * RIGHT, color=CORRECT, stroke_width=5), Text("true reward", font_size=22)
                   ).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(ax, RIGHT, buff=0.6).set_y(1.75)
        f = Text("RM gives a bonus for length", font_size=20, color=MEAN_C)
        g = Text("truth penalises rambling", font_size=20, color=CORRECT)
        forms = VGroup(f, g).arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(legend, DOWN, buff=0.3).align_to(legend, LEFT)
        self.play(FadeIn(legend), FadeIn(forms))

        t = ValueTracker(0)

        def partial(ys, color):
            k = t.get_value()
            xs = np.linspace(0, k, max(2, int(40 * k) + 2))
            return VMobject().set_points_as_corners([ax.c2p(x, np.interp(x, lx, ys)) for x in xs]) \
                .set_stroke(color, 5)
        p_curve = always_redraw(lambda: partial(px, MEAN_C))
        t_curve = always_redraw(lambda: partial(tr, CORRECT))
        p_dots = VGroup(*[Dot(ax.c2p(l, v), color=MEAN_C, radius=0.07) for l, v in zip(lx, px)])
        t_dots = VGroup(*[Dot(ax.c2p(l, v), color=CORRECT, radius=0.07) for l, v in zip(lx, tr)])
        for i, l in enumerate(lx):
            for d in (p_dots[i], t_dots[i]):
                d.add_updater(lambda m, l=l: m.set_opacity(1 if t.get_value() >= l - 1e-6 else 0))
        readout = always_redraw(lambda: VGroup(
            VGroup(Text("n =", font_size=24), Integer(int(round(2 ** t.get_value())), font_size=28)).arrange(RIGHT, buff=0.15),
            VGroup(Text("proxy", font_size=24, color=MEAN_C),
                   DecimalNumber(np.interp(t.get_value(), lx, px), include_sign=True, font_size=28, color=MEAN_C)
                   ).arrange(RIGHT, buff=0.15),
            VGroup(Text("true", font_size=24, color=CORRECT),
                   DecimalNumber(np.interp(t.get_value(), lx, tr), include_sign=True, font_size=28, color=CORRECT)
                   ).arrange(RIGHT, buff=0.15),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(forms, DOWN, buff=0.3).align_to(legend, LEFT))
        self.add(p_curve, t_curve, p_dots, t_dots)
        self.play(FadeIn(readout))
        self.play(t.animate.set_value(lx[3]), run_time=3, rate_func=linear)
        cap.say("mild selection helps: both proxy and true reward go up", color=CORRECT)
        self.play(t.animate.set_value(lx[-1]), run_time=5, rate_func=linear)
        for m in (p_curve, t_curve, readout, *p_dots, *t_dots):
            m.clear_updaters()

        k = int(np.argmax(tr))
        peak = Dot(ax.c2p(lx[k], tr[k]), color=WHITE, radius=0.11)
        vline = DashedLine(ax.c2p(lx[k], -0.5), ax.c2p(lx[k], tr[k]), color=WHITE, stroke_width=2)
        plab = Text(f"true peak: n = {ns[k]}", font_size=22).next_to(peak, UP, buff=0.35)
        cap.say("optimise the proxy too hard and you optimise its mistakes: length", color=WRONG)
        self.play(FadeIn(peak, scale=1.6), Create(vline), FadeIn(plab))

        drop = VGroup(
            Text("true reward of the pick", font_size=22, color=CORRECT),
            Text(f"peak, n = {ns[k]}:  {tr[k]:+.2f}", font_size=22),
            Text(f"at n = {ns[-1]}:  {tr[-1]:+.2f}", font_size=22, color=WRONG),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).next_to(readout, DOWN, buff=0.5).align_to(legend, LEFT)
        self.play(FadeIn(drop, shift=0.2 * UP))
        self.wait(1.5)
        gain = px[-1] - px[k]
        cap.say(f"past n = {ns[k]} the proxy still gains {gain:.2f}, the truth drops: Goodhart", wait=2.5)
