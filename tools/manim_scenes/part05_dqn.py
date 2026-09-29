"""Part 5 · DQN: the deadly triad, a moving target vs a frozen target network, and experience replay."""
import numpy as np
from style import *


def two_state_divergence(gamma=0.9, alpha=0.1, steps=60):
    """Tsitsiklis & Van Roy: V(s1)=w, V(s2)=2w, only s1->s2 (reward 0) is ever updated (off-policy).
    TD update w += alpha * (gamma*2w - w) * 1 diverges when 2*gamma > 1."""
    w = 1.0; ws = [w]
    for _ in range(steps):
        w += alpha * (gamma * 2 * w - w); ws.append(w)
    return np.array(ws)


def chase(target_every, steps=60, alpha=0.3, seed=0):
    """Regress q toward a target that is a noisy copy of q's own value.
    target_every=1: target moves every step; >1: frozen target network refreshed every k steps."""
    rng = np.random.default_rng(seed); q, tgt = 0.0, 0.0; qs, ts = [], []
    for t in range(steps):
        if t % target_every == 0:
            tgt = 1.0 + 0.9 * q                      # r + gamma * Q_target, true fixed point q* = 10
        q += alpha * (tgt + 0.5 * rng.standard_normal() - q); qs.append(q); ts.append(tgt)
    return np.array(qs), np.array(ts)


def replay_batches(n_buffer=24, batch=6, n_batches=3, seed=0):
    """Uniform minibatches (no repeats inside one batch) drawn from a buffer of n_buffer transitions."""
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n_buffer, batch, replace=False)) for _ in range(n_batches)]


# ---------------------------------------------------------------------- scene constants and small helpers
GAMMA_BAD, GAMMA_OK, ALPHA_TD, TD_STEPS = 0.9, 0.4, 0.1, 60   # two_state_divergence's inputs
REWARD_12 = 0.0                        # two_state_divergence: the only updated transition s1 -> s2 pays 0
R_CHASE, G_CHASE = 1.0, 0.9            # chase(): target = r + gamma * q_target
Q_STAR = R_CHASE / (1 - G_CHASE)       # its fixed point
CHASE_STEPS, K_FROZEN = 60, 10         # chase()'s default length; target network copied every K_FROZEN steps
N_BUF, BATCH, N_BATCH = 24, 6, 3       # act 3: transitions in the stream, minibatch size, minibatches drawn
TRIAD = [("bootstrapping", "targets use\nour own estimates", VALUE_C),
         ("off-policy", "data from a\ndifferent policy", MEAN_C),
         ("function approximation", "one update moves\nmany states", POLICY_C)]
COL_W = 4.1                            # act 1: badge-row column width (3 columns, 4.5 apart, inside the frame)


def num(v, d=2):
    return f"{v:.{d}f}".replace("-", "−")


def growth(gamma, alpha=ALPHA_TD):
    """w is multiplied by this every TD step: 1 + alpha * (2 gamma - 1)."""
    ws = two_state_divergence(gamma, alpha, 1)
    return ws[1] / ws[0]


def n_changes(ts):
    return int(np.sum(np.abs(np.diff(ts)) > 1e-12)) + 1


def badge(name, color, font_size=32):
    t = Text(name, font_size=font_size, color=color)
    box = RoundedRectangle(corner_radius=0.18, width=t.width + 0.6, height=t.height + 0.45)
    box.set_stroke(color, 3).set_fill(color, 0.12)
    bg = BackgroundRectangle(box, fill_opacity=1, buff=0)
    return VGroup(bg, box, t.move_to(box)).set_z_index(1)


class Part05(Scene):
    def construct(self):
        self.title = title_card(self, 5, "Deep Q-learning (DQN)")
        self.cap = Captioner(self, self.title)
        self.act1(); clear_act(self, keep=[self.title])
        self.act2(); clear_act(self, keep=[self.title])
        self.act3()
        takeaway(self, "dqn = q-learning + a network + two stabilisers")
        roadmap_outro(self, 5)

    # ------------------------------------------------------------------ act 1: the deadly triad
    def act1(self):
        cap = self.cap
        cap.say("put a network where the Q-table was, and three ingredients meet")
        badges, notes, shrink = VGroup(), VGroup(), []
        for i, (name, note, col) in enumerate(TRIAD):
            b = badge(name, col)
            shrink.append(min(1.0, COL_W / b.width))
            b.scale(shrink[-1]).move_to([(i - 1) * 4.5, 0.4, 0])
            badges.add(b)
            lines = VGroup(*[Text(ln, font_size=26, color=MUTED) for ln in note.split("\n")]).arrange(DOWN, buff=0.12)
            notes.add(fit(lines, COL_W).move_to([b.get_x(), 0, 0]).align_to(badges, UP).shift(1.2 * DOWN))
        for b, n in zip(badges, notes):
            self.play(FadeIn(b, shift=0.2 * UP), FadeIn(n), run_time=0.8)
            self.wait(0.5)
        self.wait(0.5)

        spots = [np.array([0, 1.5, 0]), np.array([-4.0, -2.6, 0]), np.array([4.0, -2.6, 0])]
        edges = VGroup(*[Line(spots[i], spots[j], color=WRONG, stroke_width=5)
                         for i, j in ((0, 1), (1, 2), (2, 0))]).set_z_index(-1)
        centre = Text("the deadly triad", font_size=34, color=WRONG).move_to((spots[0] + spots[1] + spots[2]) / 3)
        self.play(FadeOut(notes), *[b.animate.scale(1 / k).move_to(p) for b, p, k in zip(badges, spots, shrink)], run_time=1.2)
        self.play(Create(edges), FadeIn(centre), run_time=1)
        cap.say("any two are fine; all three can diverge", color=WRONG, wait=2.5)
        self.play(FadeOut(edges), FadeOut(centre), FadeOut(badges))

        # A two-state example (Tsitsiklis & Van Roy) that has all three.
        cap.say("a two-state example (Tsitsiklis & Van Roy) with all three")
        px = 3.9
        s1 = Circle(0.42, color=REF_C).move_to([px - 1.45, 1.6, 0])
        s2 = Circle(0.42, color=REF_C).move_to([px + 1.45, 1.6, 0])
        n1 = MathTex("s_1", font_size=34).move_to(s1); n2 = MathTex("s_2", font_size=34).move_to(s2)
        v1 = MathTex("V = w", font_size=32, color=POLICY_C).next_to(s1, DOWN, buff=0.15)
        v2 = MathTex("V = 2w", font_size=32, color=POLICY_C).next_to(s2, DOWN, buff=0.15)
        arr = Arrow(s1.get_right(), s2.get_left(), buff=0.08, color=MEAN_C, stroke_width=4)
        rlab = MathTex(f"r = {REWARD_12:g}", font_size=30, color=MEAN_C).next_to(arr, UP, buff=0.1)
        tags = VGroup(
            Text("function approx: one weight w for both", font_size=22, color=POLICY_C),
            Text("off-policy: only s₁ → s₂ is ever updated", font_size=22, color=MEAN_C),
            Text("bootstrapping: target γ·V(s₂) = γ·2w", font_size=22, color=VALUE_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([px, -0.1, 0])
        fit(tags, 6.0)
        self.play(Create(s1), Create(s2), FadeIn(n1), FadeIn(n2), run_time=0.8)
        self.play(FadeIn(v1), FadeIn(v2))
        self.play(GrowArrow(arr), FadeIn(rlab))
        for t in tags:
            self.play(FadeIn(t, shift=0.1 * RIGHT), run_time=0.6)
        self.wait(0.8)

        upd = MathTex(r"w", r"\leftarrow", r"w", "+", r"\alpha", r"\big(", r"\gamma\cdot 2w", "-", "w", r"\big)",
                      font_size=38)
        upd[4].set_color(STD_C); upd[6].set_color(VALUE_C)
        fac = MathTex(r"=\big(1 + \alpha(2\gamma - 1)\big)\, w", font_size=34)
        VGroup(upd, fac).arrange(DOWN, buff=0.2).move_to([px, -1.55, 0])
        a_lab = MathTex(rf"\alpha = {ALPHA_TD:g}", font_size=30, color=STD_C).next_to(fac, DOWN, buff=0.25)
        self.play(Write(upd))
        self.play(FadeIn(fac), FadeIn(a_lab))

        # The plot: w over TD steps for a large and a small gamma.
        bad, ok = two_state_divergence(GAMMA_BAD), two_state_divergence(GAMMA_OK)
        ymax = float(np.ceil(bad.max() / 20) * 20)
        ax = Axes(x_range=[0, TD_STEPS, 10], y_range=[0, ymax, 20], x_length=6.4, y_length=4.4, tips=False,
                  axis_config={"color": REF_C, "font_size": 22},
                  x_axis_config={"numbers_to_include": list(range(0, TD_STEPS + 1, 20))},
                  y_axis_config={"numbers_to_include": list(range(0, int(ymax) + 1, 40))}).move_to([-3.3, -0.85, 0])
        xl = Text("TD updates", font_size=22, color=MUTED).next_to(ax.x_axis, DOWN, buff=0.35)
        yl = MathTex("w", font_size=32, color=MUTED).next_to(ax.y_axis, UP, buff=0.12)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        k = ValueTracker(0)

        def curve(ws, col):
            def make():
                n = max(int(k.get_value()), 1)
                return VMobject().set_points_as_corners([ax.c2p(i, ws[i]) for i in range(n + 1)]) \
                    .set_stroke(col, 5)
            return always_redraw(make)
        c_bad, c_ok = curve(bad, WRONG), curve(ok, CORRECT)
        leg_bad = MathTex(rf"\gamma = {GAMMA_BAD:g}:\ \times {num(growth(GAMMA_BAD))}\ \text{{per step}}",
                          font_size=30, color=WRONG)
        leg_ok = MathTex(rf"\gamma = {GAMMA_OK:g}:\ \times {num(growth(GAMMA_OK))}\ \text{{per step}}",
                         font_size=30, color=CORRECT)
        VGroup(leg_bad, leg_ok).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([px, -3.25, 0])
        self.add(c_bad, c_ok)
        cap.say(f"with γ = {GAMMA_BAD:g}, 2γ > 1: every update makes w bigger", color=WRONG)
        self.play(FadeIn(leg_bad), FadeIn(leg_ok))
        self.play(k.animate.set_value(TD_STEPS), run_time=5, rate_func=linear)
        end_bad = Text(num(bad[-1]), font_size=24, color=WRONG).next_to(ax.c2p(TD_STEPS, bad[-1]), LEFT, buff=0.15)
        end_ok = Text(num(ok[-1]), font_size=24, color=CORRECT).next_to(ax.c2p(TD_STEPS, ok[-1]), UP, buff=0.15)
        self.play(FadeIn(end_bad), FadeIn(end_ok))
        self.wait(1.5)
        cap.say("dqn's two tricks don't remove the triad; they tame it in practice")
        self.wait(2.5)

    # ------------------------------------------------------------------ act 2: a moving target
    def chase_row(self, qs, ts, y, head, col):
        nl = NumberLine(x_range=[0, Q_STAR, 1], length=6.6, color=REF_C, include_numbers=True, font_size=22,
                        numbers_to_include=[0, 5, int(Q_STAR)]).move_to([-2.9, y, 0])
        star = MathTex("q^*", font_size=28, color=CORRECT).next_to(nl.n2p(Q_STAR), UP, buff=0.18)
        lab = Text(head, font_size=24, color=col).move_to([-6.4, y + 0.95, 0], aligned_edge=LEFT)
        ax = Axes(x_range=[0, CHASE_STEPS, 20], y_range=[0, Q_STAR, 5], x_length=4.2, y_length=1.7, tips=False,
                  axis_config={"color": REF_C, "font_size": 18},
                  x_axis_config={"numbers_to_include": [0, CHASE_STEPS]},
                  y_axis_config={"numbers_to_include": [0, int(Q_STAR)]}).move_to([4.55, y + 0.2, 0])
        k = ValueTracker(0)
        idx = lambda: min(int(k.get_value()), CHASE_STEPS - 1)

        def q_at():
            v = k.get_value()
            return float(np.interp(v - 1, np.arange(CHASE_STEPS), qs)) if v >= 1 else 0.0
        qd = always_redraw(lambda: Dot(nl.n2p(q_at()), radius=0.12, color=VALUE_C).set_z_index(2))
        ql = always_redraw(lambda: MathTex("q", font_size=30, color=VALUE_C).next_to(qd, DOWN, buff=0.4))
        tri = always_redraw(lambda: Triangle(color=MEAN_C).set_fill(MEAN_C, 1).scale(0.14).rotate(PI)
                            .next_to(nl.n2p(ts[idx()]), UP, buff=0.05))
        tl = always_redraw(lambda: Text("target", font_size=22, color=MEAN_C).next_to(tri, UP, buff=0.08))

        def traces():
            n = idx() + 1
            tp = []
            for i in range(n):
                tp += [ax.c2p(i, ts[i]), ax.c2p(i + 1, ts[i])]
            qp = [ax.c2p(0, 0)] + [ax.c2p(i + 1, qs[i]) for i in range(n)]
            return VGroup(VMobject().set_points_as_corners(tp).set_stroke(MEAN_C, 3),
                          VMobject().set_points_as_corners(qp).set_stroke(VALUE_C, 3))
        tr = always_redraw(traces)
        cnt = always_redraw(lambda: Text(f"step {idx() + 1}   target changed {n_changes(ts[:idx() + 1])} times",
                                         font_size=22, color=MUTED).move_to([-2.9, y - 1.05, 0]))
        static = VGroup(nl, star, lab, ax)
        return k, static, VGroup(qd, ql, tri, tl, tr, cnt)

    def act2(self):
        cap = self.cap
        form = MathTex(r"\text{target} = r + \gamma\, q_{\text{target}}", "=", f"{R_CHASE:g}", "+",
                       f"{G_CHASE:g}", r"\cdot q_{\text{target}}", r"\qquad q^* =", f"{Q_STAR:g}", font_size=34)
        form.move_to([0, 1.95, 0]); form[4].set_color(STD_C); form[7].set_color(CORRECT)
        cap.say("a moving target: q regresses toward a value built from q itself")
        self.play(Write(form))
        q1, t1 = chase(1); qk, tk = chase(K_FROZEN)
        k1, st1, dy1 = self.chase_row(q1, t1, 0.4, "target recomputed from q every step", WRONG)
        kk, stk, dyk = self.chase_row(qk, tk, -2.15, f"frozen target network, copied every {K_FROZEN} steps", POLICY_C)
        self.play(FadeIn(st1))
        self.add(dy1)
        cap.say("every step q moves, so the target moves too: it jitters", color=WRONG)
        self.play(k1.animate.set_value(CHASE_STEPS), run_time=8, rate_func=linear)
        self.wait(0.5)
        self.play(FadeIn(stk))
        self.add(dyk)
        cap.say("freeze the target network: regress toward something that holds still", color=POLICY_C)
        self.play(kk.animate.set_value(CHASE_STEPS), run_time=8, rate_func=linear)
        self.wait(1)
        note = Text(f"after {CHASE_STEPS} steps q = {num(q1[-1])} (moving) vs {num(qk[-1])} (frozen); "
                    f"q* = {Q_STAR:g}", font_size=22, color=MUTED)
        fit(note, 12.5).to_edge(DOWN, buff=0.2)
        cap.say("slower in this toy, but each regression now has a fixed goal")
        self.play(FadeIn(note))
        self.wait(3)

    # ------------------------------------------------------------------ act 3: experience replay
    def act3(self):
        cap = self.cap
        cols = color_gradient([STD_C, CORRECT, MEAN_C, WRONG], N_BUF)
        size, gap = 0.4, 0.48
        stream = VGroup(*[Square(size).set_stroke(WHITE, 1, 0.6).set_fill(cols[i], 0.9)
                          .move_to([(i - (N_BUF - 1) / 2) * gap, 1.7, 0]) for i in range(N_BUF)])
        t0 = MathTex("t = 1", font_size=26, color=MUTED).next_to(stream[0], DOWN, buff=0.15)
        t1 = MathTex(f"t = {N_BUF}", font_size=26, color=MUTED).next_to(stream[-1], DOWN, buff=0.15)
        tarr = Arrow(t0.get_right(), t1.get_left(), buff=0.3, color=MUTED, stroke_width=2,
                     max_tip_length_to_length_ratio=0.02)
        tlab = Text("time", font_size=20, color=MUTED).next_to(tarr, DOWN, buff=0.05)
        cap.say("the agent sees a stream of transitions (s, a, r, s′), one per step")
        self.play(LaggedStart(*[FadeIn(s, shift=0.3 * RIGHT) for s in stream], lag_ratio=0.12), run_time=2.5)
        self.play(FadeIn(t0), FadeIn(t1), GrowArrow(tarr), FadeIn(tlab))
        cap.say("consecutive transitions are correlated: neighbours look alike")
        self.wait(1)

        # Right-hand panel: the minibatches we would train on.
        rx, ry = 3.9, [0.2, -0.75, -1.55, -2.35]

        def batch_row(idxs, y):
            return VGroup(*[Square(size).set_stroke(WHITE, 1, 0.6).set_fill(cols[i], 0.9)
                            .move_to([rx + (j - (BATCH - 1) / 2) * gap, y, 0]) for j, i in enumerate(idxs)])

        def row_label(text, idxs, y, col):
            return VGroup(Text(text, font_size=22, color=col),
                          Text(f"t {idxs[0] + 1}–{idxs[-1] + 1}", font_size=20, color=MUTED)) \
                .arrange(DOWN, aligned_edge=LEFT, buff=0.06).move_to([0.3, y, 0], aligned_edge=LEFT)
        latest = list(range(N_BUF - BATCH, N_BUF))
        on_row = batch_row(latest, ry[0])
        on_lab = row_label(f"latest {BATCH}", latest, ry[0], WRONG)
        self.play(*[TransformFromCopy(stream[i], on_row[j]) for j, i in enumerate(latest)], FadeIn(on_lab))
        cap.say("train on them in order and every minibatch is one narrow moment", color=WRONG)
        self.wait(1.5)

        # Pour the stream into the buffer.
        bw, bh = 6, N_BUF // 6
        bc = np.array([-3.6, -1.0, 0])
        slots = [bc + np.array([(i % bw - (bw - 1) / 2) * gap, ((bh - 1) / 2 - i // bw) * gap, 0]) for i in range(N_BUF)]
        frame = SurroundingRectangle(VGroup(*[Square(size).move_to(p) for p in slots]), buff=0.2,
                                     color=REF_C, corner_radius=0.1)
        blab = Text("replay buffer", font_size=26, color=REF_C).next_to(frame, UP, buff=0.12)
        cap.say("instead, store every transition in a replay buffer")
        self.play(FadeOut(t0), FadeOut(t1), FadeOut(tarr), FadeOut(tlab), Create(frame), FadeIn(blab))
        self.play(LaggedStart(*[stream[i].animate.move_to(slots[i]) for i in range(N_BUF)], lag_ratio=0.06),
                  run_time=2.5)

        # Draw random minibatches.
        batches = replay_batches(N_BUF, BATCH, N_BATCH)
        cap.say("then train on random minibatches: every batch mixes old and new", color=POLICY_C)
        for b, idxs in enumerate(batches):
            row = batch_row(idxs, ry[b + 1])
            lab = row_label(f"minibatch {b + 1}", idxs, ry[b + 1], POLICY_C)
            rings = VGroup(*[SurroundingRectangle(stream[i], buff=0.04, color=WHITE, stroke_width=3) for i in idxs])
            self.play(Create(rings), run_time=0.5)
            self.play(*[TransformFromCopy(stream[i], row[j]) for j, i in enumerate(idxs)], FadeIn(lab),
                      FadeOut(rings), run_time=1)
            self.wait(0.4)
        uses = np.bincount(np.concatenate(batches), minlength=N_BUF)
        read = Text(f"{N_BATCH} minibatches from {N_BUF} transitions: {int((uses > 1).sum())} used more than once",
                    font_size=22, color=MUTED)
        fit(read, 12.5).to_edge(DOWN, buff=0.2)
        cap.say("replay breaks correlations and reuses every transition", color=POLICY_C)
        self.play(FadeIn(read))
        self.wait(3)
