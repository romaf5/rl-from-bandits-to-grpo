"""Part 3 · Dynamic programming: value iteration, the γ-contraction, and the greedy policy."""
import numpy as np
from style import *
from part01_foundations import gridworld


def value_iteration_history(gamma=0.9, sweeps=40):
    cells, P, R, _ = gridworld()
    V = np.zeros(len(cells)); hist = [V]
    for _ in range(sweeps):
        V = (P * (R + gamma * V[None, None, :])).sum(-1).max(1); hist.append(V)
    return cells, np.array(hist)


def greedy_policy(V, gamma=0.9):
    cells, P, R, _ = gridworld()
    return (P * (R + gamma * V[None, None, :])).sum(-1).argmax(1)


# ---------------------------------------------------------------------- scene constants and small helpers
from part01_foundations import LAYOUT, MOVES, CELL, cell_center, grid_mob, num

GAMMA = 0.9                        # the notebook's default discount
K_SHOW = 15                        # acts 1-2 animate sweeps 0 … K_SHOW
BACKUP_CELL, BACKUP_K = (2, 2), 5  # act 1's worked Bellman backup: this cell, reading V_k at this sweep
PANEL_X = 3.3                      # x-centre of the right-hand panel
ARROWS = "↑→↓←"                    # action names in MOVES order


def action_values(V, s, gamma=GAMMA):
    """Q(s, a) = sum_s' p(s'|s,a) [r + γ V(s')] for every action a."""
    _, P, R, _ = gridworld()
    return (P[s] * (R[s] + gamma * V[None, :])).sum(-1)


def sweep_errors(H):
    """max-norm distance of every sweep's V_k from the final one (V*)."""
    return np.abs(H - H[-1]).max(1)


def screen_dir(a):
    dr, dc = MOVES[a]
    return np.array([dc, -dr, 0.0])


class Part03(Scene):
    def construct(self):
        self.cells, self.H = value_iteration_history(GAMMA)
        _, _, _, self.terminal = gridworld()
        live = ~self.terminal
        self.vmin, self.vmax = self.H[:, live].min(), self.H[:, live].max()
        self._heat_cache = {}

        self.title = title_card(self, 3, "Dynamic programming: planning with a model")
        self.cap = Captioner(self, self.title)
        self.act1(); clear_act(self, keep=[self.title])
        self.act2(); clear_act(self, keep=[self.title])
        self.act3()
        takeaway(self, "every later algorithm is a sampled approximation of this")
        roadmap_outro(self, 3)

    # ------------------------------------------------------------------ shared grid pieces
    def board(self):
        """Walls, P and G squares plus the S/P/G letters (the heatmap covers every other cell)."""
        squares, letters = grid_mob()
        fixed = VGroup(*[sq for sq in squares if sq.ch in "#PG"])
        return fixed, letters

    def heat(self, k, labels=True):
        key = (k, labels)
        if key not in self._heat_cache:
            V = self.H[k]
            grp = VGroup()
            for s, cell in enumerate(self.cells):
                if self.terminal[s]:
                    continue
                t = (V[s] - self.vmin) / (self.vmax - self.vmin)
                sq = Square(CELL).move_to(cell_center(cell)).set_stroke(REF_C, 1.5, opacity=0.6) \
                    .set_fill(VALUE_C, 0.06 + 0.84 * t)
                parts = [sq]
                if labels:
                    parts.append(Text(num(V[s]), font_size=17).move_to(sq.get_center() + 0.12 * DOWN))
                grp.add(VGroup(*parts))
            self._heat_cache[key] = grp
        return self._heat_cache[key].copy()   # always_redraw mutates what it is given

    def sweep_counter(self, k_tracker, pos):
        return always_redraw(lambda: MathTex(rf"\text{{sweep }} k = {int(round(k_tracker.get_value()))}",
                                             font_size=38, color=MEAN_C).move_to(pos))

    # ------------------------------------------------------------------ act 1: values flow from the goal
    def act1(self):
        cap, H, cells = self.cap, self.H, self.cells
        k = ValueTracker(0)
        heatmap = always_redraw(lambda: self.heat(int(round(k.get_value()))))
        fixed, letters = self.board()
        cap.say("value iteration: start from V = 0 everywhere and sweep")
        self.play(FadeIn(fixed), FadeIn(heatmap), FadeIn(letters))

        bellman = MathTex(r"V_{k+1}(s) \leftarrow \max_a \sum_{s'} p(s' \mid s,a)\big[r + \gamma\, V_k(s')\big]",
                          font_size=30)
        fit(bellman, 6.4).move_to([PANEL_X, 1.5, 0])
        counter = self.sweep_counter(k, [PANEL_X, 0.5, 0])
        gam = MathTex(rf"\gamma = {GAMMA:g}", font_size=32, color=STD_C).move_to([PANEL_X, -0.2, 0])
        self.play(Write(bellman), FadeIn(counter), FadeIn(gam))
        self.wait(0.5)

        cap.say("each sweep: V(s) ← max over actions of reward + γ·V(next)")
        for _ in range(3):
            self.play(k.animate.set_value(k.get_value() + 1), run_time=0.9)
            self.wait(0.5)
        cap.say("the goal's reward leaks one step further every sweep", color=VALUE_C)
        self.play(k.animate.set_value(BACKUP_K), run_time=1.6, rate_func=linear)
        self.wait(0.5)

        # One backup, drawn once: four arrows from the neighbours into the centre cell.
        s = cells.index(BACKUP_CELL)
        q = action_values(H[BACKUP_K], s)
        best = np.isclose(q, q.max())
        centre = cell_center(BACKUP_CELL)
        arrows, qlabs = VGroup(), VGroup()
        for a in range(4):
            d = screen_dir(a)
            col = CORRECT if best[a] else MUTED
            lift = 0.2 * UP if a % 2 else ORIGIN          # horizontal arrows ride above the value labels
            start = 0.95 if a % 2 else 0.78               # vertical arrows start below the neighbour's label
            arrows.add(Arrow(centre + d * CELL * start + lift, centre + d * CELL * 0.33 + lift, buff=0, color=col,
                             stroke_width=8, max_tip_length_to_length_ratio=0.35))
        ring = Square(CELL).move_to(centre).set_stroke(MEAN_C, 5)
        cap.say("one backup: look at every action's next cells, keep the best", color=MEAN_C)
        self.play(Create(ring), LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.2), run_time=1.5)

        head = MathTex(rf"Q(s,a) \text{{ from }} V_{{{BACKUP_K}}}", font_size=30, color=MUTED)
        rows = VGroup(*[VGroup(Text(ARROWS[a], font_size=28, color=CORRECT if best[a] else MUTED),
                               Text(num(q[a], 3), font_size=26, color=CORRECT if best[a] else MUTED)
                               ).arrange(RIGHT, buff=0.3) for a in range(4)])
        rows.arrange_in_grid(rows=2, cols=2, buff=(0.8, 0.2))
        table = VGroup(head, rows).arrange(DOWN, buff=0.25).move_to([PANEL_X, -1.35, 0])
        self.play(FadeIn(head), LaggedStart(*[FadeIn(r) for r in rows], lag_ratio=0.2), run_time=1.3)
        result = MathTex(rf"V_{{{BACKUP_K + 1}}}(s) = \max_a Q(s,a) = {num(q.max(), 3)}", font_size=32,
                         color=CORRECT).move_to([PANEL_X, -2.65, 0])
        self.play(Write(result))
        self.play(k.animate.set_value(BACKUP_K + 1), Indicate(ring, color=CORRECT), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(arrows), FadeOut(ring), FadeOut(table), FadeOut(result))

        cap.say("every cell does this at once, sweep after sweep", color=VALUE_C)
        self.play(k.animate.set_value(K_SHOW), run_time=5, rate_func=linear)
        cap.say("after a few sweeps nothing changes any more: that is V*", color=CORRECT)
        self.wait(2)
        heatmap.clear_updaters(); counter.clear_updaters()

    # ------------------------------------------------------------------ act 2: a contraction
    def act2(self):
        cap, H = self.cap, self.H
        err = sweep_errors(H)
        worst = np.abs(H - H[-1]).argmax(1)
        k = ValueTracker(0)
        ki = lambda: int(round(k.get_value()))
        heatmap = always_redraw(lambda: self.heat(ki()))
        fixed, letters = self.board()
        marker = always_redraw(lambda: Square(CELL).move_to(cell_center(self.cells[worst[ki()]]))
                               .set_stroke(WRONG, 5))
        counter = self.sweep_counter(k, [PANEL_X - 1.7, 1.8, 0])

        # Error bar: full width = the error at sweep 0.
        bar_w, bar_y, bar_x0 = 4.2, 0.55, PANEL_X - 2.6
        frame = Rectangle(width=bar_w, height=0.36).set_stroke(REF_C, 1.5).move_to(
            [bar_x0 + bar_w / 2, bar_y, 0])
        norm = MathTex(r"\text{error}_k = \max_s |V_k(s) - V^*(s)|", font_size=30, color=WRONG) \
            .next_to(frame, UP, buff=0.2, aligned_edge=LEFT)
        errbar = always_redraw(lambda: Rectangle(width=max(bar_w * err[ki()] / err[0], 1e-3), height=0.36)
                               .set_stroke(width=0).set_fill(WRONG, 0.8)
                               .move_to([bar_x0, bar_y, 0], aligned_edge=LEFT))
        errnum = always_redraw(lambda: Text(f"{err[ki()]:.4f}", font_size=24, color=WRONG)
                               .next_to(frame, RIGHT, buff=0.2))

        # Log-scale mini-plot of the error with the γ^k ceiling.
        lo = int(np.floor(np.log10(err[K_SHOW])))
        ax = Axes(x_range=[0, K_SHOW, 5], y_range=[0, -lo, 1], x_length=5.4, y_length=2.7, tips=False,
                  axis_config={"color": REF_C, "include_ticks": True},
                  x_axis_config={"numbers_to_include": list(range(0, K_SHOW + 1, 5)), "font_size": 22})
        ax.move_to([PANEL_X + 0.1, -1.75, 0])
        ylabs = VGroup(*[MathTex(rf"10^{{{e}}}", font_size=22, color=MUTED).next_to(ax.c2p(0, e - lo), LEFT, buff=0.12)
                         for e in range(lo, 1, 2)])
        xlab = Text("sweep k", font_size=20, color=MUTED).next_to(ax.x_axis, DOWN, buff=0.4).align_to(ax, RIGHT)
        yv = lambda e: np.log10(e) - lo                  # log10(error), shifted so the axis sits at the bottom
        ceiling = DashedLine(ax.c2p(0, yv(err[0])), ax.c2p(K_SHOW, yv(err[0] * GAMMA ** K_SHOW)),
                             color=STD_C, stroke_width=3)
        ceil_lab = MathTex(r"\gamma^k \cdot \text{error}_0", font_size=26, color=STD_C) \
            .next_to(ceiling.get_end(), DOWN, buff=0.15).shift(0.4 * LEFT)

        def curve():
            n = ki()
            pts = [ax.c2p(j, yv(err[j])) for j in range(n + 1)]
            dots = VGroup(*[Dot(p, radius=0.05, color=WRONG) for p in pts])
            line = VMobject().set_points_as_corners(pts).set_stroke(WRONG, 3) if n else VMobject()
            return VGroup(line, dots)

        plot = always_redraw(curve)

        cap.say("how far is sweep k from the answer? measure the worst cell", color=WRONG)
        self.play(FadeIn(fixed), FadeIn(heatmap), FadeIn(letters), FadeIn(counter))
        self.play(Create(frame), FadeIn(norm), FadeIn(errbar), FadeIn(errnum), FadeIn(marker))
        self.play(Create(ax), FadeIn(ylabs), FadeIn(xlab), FadeIn(plot))
        self.wait(1)

        cap.say("each sweep multiplies the error by at most γ", color=STD_C)
        self.play(Create(ceiling), FadeIn(ceil_lab))
        for _ in range(3):
            self.play(k.animate.set_value(k.get_value() + 1), run_time=0.9)
            self.wait(0.4)
        n = int(k.get_value())
        ratio = MathTex(rf"\text{{error}}_{{{n}}} / \text{{error}}_{{{n - 1}}} \approx {err[n] / err[n - 1]:.2f}"
                        rf" \le \gamma",
                        font_size=30, color=STD_C).move_to([PANEL_X + 1.35, 1.8, 0])
        self.play(FadeIn(ratio))
        self.wait(1.5)
        self.play(k.animate.set_value(K_SHOW), run_time=6, rate_func=linear)
        cap.say("γᵏ is only a ceiling: the real error can fall even faster", color=STD_C)
        self.wait(2.5)
        for m in (heatmap, marker, counter, errbar, errnum, plot):
            m.clear_updaters()

    # ------------------------------------------------------------------ act 3: the policy falls out
    def act3(self):
        cap, cells, V = self.cap, self.cells, self.H[-1]
        pi = greedy_policy(V, GAMMA)
        fixed, letters = self.board()
        heatmap = self.heat(len(self.H) - 1)
        cap.say("V* is known: now read a policy off it", color=VALUE_C)
        self.play(FadeIn(fixed), FadeIn(heatmap), FadeIn(letters))

        rule = MathTex(r"\pi^*(s) = \arg\max_a \sum_{s'} p(s' \mid s,a)\big[r + \gamma\, V^*(s')\big]",
                       font_size=30)
        fit(rule, 6.4).move_to([PANEL_X, 1.2, 0])
        self.play(Write(rule))

        arrows = VGroup()
        for s, cell in enumerate(cells):
            if self.terminal[s]:
                continue
            d, c = screen_dir(pi[s]), cell_center(cell) + 0.08 * RIGHT + 0.05 * DOWN
            arrows.add(Arrow(c - d * 0.2, c + d * 0.2, buff=0, color=POLICY_C, stroke_width=6,
                             max_tip_length_to_length_ratio=0.5))
        labels = VGroup(*[g[1] for g in heatmap])
        cap.say("act greedily on V* and you have the optimal policy", color=POLICY_C)
        self.play(labels.animate.set_opacity(0.0),
                  LaggedStart(*[FadeIn(a, scale=0.6) for a in arrows], lag_ratio=0.06), run_time=3)
        self.wait(1.5)

        g_nb = cells.index((4, 3))
        q = action_values(V, g_nb)
        spot = Square(CELL).move_to(cell_center((4, 3))).set_stroke(MEAN_C, 5)
        read = VGroup(*[VGroup(Text(ARROWS[a], font_size=28, color=POLICY_C if a == pi[g_nb] else MUTED),
                               Text(num(q[a], 3), font_size=26, color=POLICY_C if a == pi[g_nb] else MUTED)
                               ).arrange(RIGHT, buff=0.3) for a in range(4)])
        read.arrange_in_grid(rows=2, cols=2, buff=(0.8, 0.2))
        head = Text("next to the goal", font_size=26, color=MUTED)
        box = VGroup(head, read).arrange(DOWN, buff=0.25).move_to([PANEL_X, -1.0, 0])
        cap.say("beside the goal the best action steps right, straight into G", color=CORRECT)
        self.play(Create(spot), FadeIn(box))
        live_ids = [s for s in range(len(cells)) if not self.terminal[s]]
        self.play(Indicate(arrows[live_ids.index(g_nb)], color=CORRECT))
        self.wait(1.5)
        cap.say("no trial and error: the model p(s′|s,a) did all the work", color=MUTED)
        self.wait(2.5)
