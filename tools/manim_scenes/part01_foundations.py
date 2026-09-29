"""Part 1 · Foundations: the agent-environment loop, returns and discounting, and γ as the horizon."""
import numpy as np
from style import *

LAYOUT = ("S....", ".#.#.", ".....", ".#.P.", "....G")
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def gridworld(slip=0.1, step_reward=-0.04):
    """Returns cells, P[s,a,s'], R[s,a,s'], terminal[s] exactly as the notebook's GridWorld."""
    H, W = len(LAYOUT), len(LAYOUT[0])
    cells = [(r, c) for r in range(H) for c in range(W) if LAYOUT[r][c] != "#"]
    idx = {cell: i for i, cell in enumerate(cells)}
    ch = lambda cell: LAYOUT[cell[0]][cell[1]]
    terminal = np.array([ch(cell) in "GP" for cell in cells])
    nS = len(cells); P = np.zeros((nS, 4, nS)); R = np.zeros((nS, 4, nS))

    def move(cell, a):
        r, c = cell; nr, nc = r + MOVES[a][0], c + MOVES[a][1]
        return (nr, nc) if 0 <= nr < H and 0 <= nc < W and LAYOUT[nr][nc] != "#" else cell

    for s, cell in enumerate(cells):
        for a in range(4):
            if terminal[s]:
                P[s, a, s] = 1.0; continue
            for a2, p in [(a, 1 - slip), ((a + 1) % 4, slip / 2), ((a - 1) % 4, slip / 2)]:
                s2 = idx[move(cell, a2)]
                P[s, a, s2] += p
                R[s, a, s2] = step_reward + (1.0 if ch(cells[s2]) == "G" else -1.0 if ch(cells[s2]) == "P" else 0.0)
    return cells, P, R, terminal


def discounted_return(rewards, gamma):
    return float(sum(r * gamma ** t for t, r in enumerate(rewards)))


def optimal_values(gamma, sweeps=500):
    cells, P, R, _ = gridworld()
    V = np.zeros(len(cells))
    for _ in range(sweeps):
        V = (P * (R + gamma * V[None, None, :])).sum(-1).max(1)
    return V


# ---------------------------------------------------------------------- scene constants and small helpers
PATH_ACTIONS = [1, 1, 2, 2, 1, 1, 2, 2]     # act 2's fixed walk from S to G: → → ↓ ↓ → → ↓ ↓
GAMMA = 0.9                                  # act 2's discount (the notebook's default)
GAMMA_LO, GAMMA_HI = 0.1, 0.99               # act 3's sweep
GAMMA_STOPS = [GAMMA_LO, 0.5, GAMMA, GAMMA_HI]
N_LOOPS = 3                                  # act 1: turns of the agent-environment loop
CELL = 0.9                                   # grid cell side
GRID_C = np.array([-3.4, -0.75, 0])          # grid centre
PANEL_X = 3.3                                # x-centre of the right-hand panel


def trajectory(actions=PATH_ACTIONS):
    """Cells visited and rewards collected when each move goes where intended (no slip)."""
    cells, P, R, _ = gridworld()
    idx = {cell: i for i, cell in enumerate(cells)}
    path, rewards = [(0, 0)], []
    for a in actions:
        r, c = path[-1]
        nxt = (r + MOVES[a][0], c + MOVES[a][1])
        rewards.append(float(R[idx[path[-1]], a, idx[nxt]]))
        path.append(nxt)
    return path, rewards


def horizon(gamma):
    return 1 / (1 - gamma)


SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def hz(gamma):
    h = horizon(gamma)
    return f"{h:.1f} steps" if h < 10 else f"{h:.0f} steps"


def num(v, d=2):
    return f"{v:+.{d}f}".replace("-", "−")


def tnum(v, d=3):
    return f"{v:+.{d}f}"


def cell_center(rc):
    r, c = rc
    return GRID_C + CELL * np.array([c - 2, 2 - r, 0])


def grid_mob():
    """Squares for every cell (walls dark grey, P in WRONG, G in CORRECT) plus S/P/G letters."""
    squares, letters = VGroup(), VGroup()
    for r, row in enumerate(LAYOUT):
        for c, ch in enumerate(row):
            sq = Square(CELL).move_to(cell_center((r, c))).set_stroke(REF_C, 1.5, opacity=0.6)
            if ch == "#":
                sq.set_fill(GREY_D, 1).set_stroke(GREY_D, 1.5)
            elif ch == "P":
                sq.set_fill(WRONG, 0.55)
            elif ch == "G":
                sq.set_fill(CORRECT, 0.55)
            else:
                sq.set_fill(BLACK, 0)
            sq.ch = ch
            squares.add(sq)
            if ch in "SPG":
                letters.add(Text(ch, font_size=26, weight=BOLD).move_to(sq.get_corner(UL) + 0.2 * (RIGHT + DOWN)))
    return squares, letters


class Part01(Scene):
    def construct(self):
        self.title = title_card(self, 1, "Foundations: the RL problem")
        self.cap = Captioner(self, self.title)
        self.act1(); clear_act(self, keep=[self.title])
        self.act2(); clear_act(self, keep=[self.title])
        self.act3()
        takeaway(self, "rl learns what to do from delayed, evaluative feedback")
        roadmap_outro(self, 1)

    # ------------------------------------------------------------------ act 1: the loop
    def act1(self):
        cap = self.cap

        def box(name, sub, x):
            rect = RoundedRectangle(corner_radius=0.2, width=3.6, height=1.7).set_stroke(REF_C, 2.5)
            body = VGroup(Text(name, font_size=34), sub).arrange(DOWN, buff=0.2)
            return VGroup(rect, body).move_to([x, 0.2, 0])

        agent = box("agent", MathTex(r"\text{policy } \pi(a \mid s)", font_size=32, color=POLICY_C), -4.0)
        env = box("environment", MathTex(r"\text{dynamics } p(s', r \mid s, a)", font_size=30, color=VALUE_C), 4.0)
        top = Arrow(agent.get_right() + 0.45 * UP, env.get_left() + 0.45 * UP, buff=0.1, color=POLICY_C)
        bot = Arrow(env.get_left() + 0.45 * DOWN, agent.get_right() + 0.45 * DOWN, buff=0.1, color=MEAN_C)
        cap.say("the agent acts, the world answers with a reward and a new state")
        self.play(FadeIn(agent), FadeIn(env))
        self.play(GrowArrow(top), GrowArrow(bot))

        def lab_a(t):
            return MathTex(rf"a_{{{t}}}", font_size=40, color=POLICY_C).next_to(top, UP, buff=0.15)

        def lab_r(t):
            return MathTex(rf"r_{{{t + 1}}}", ",\\ ", rf"s_{{{t + 1}}}", font_size=40).next_to(bot, DOWN, buff=0.15) \
                .set_color_by_tex(f"r_", MEAN_C)

        clock = MathTex("t = 0", font_size=36, color=MUTED).move_to([0, 1.9, 0])
        state = MathTex(r"s_0", font_size=40).next_to(agent, DOWN, buff=0.35)
        trail = VGroup(MathTex(r"s_0", font_size=40)).move_to([0, -2.4, 0])
        self.play(FadeIn(clock), FadeIn(state), FadeIn(trail))

        la = lr = None
        for t in range(N_LOOPS):
            rt = 0.9 if t == 0 else 0.6
            new_clock = MathTex(f"t = {t}", font_size=36, color=MUTED).move_to(clock)
            new_a = lab_a(t)
            pulse = Dot(top.get_start(), radius=0.12, color=POLICY_C)
            self.play(Transform(clock, new_clock), *( [FadeOut(la)] if la else [] ), FadeIn(new_a),
                      Indicate(agent[0], color=POLICY_C, scale_factor=1.05), run_time=rt)
            self.play(MoveAlongPath(pulse, Line(top.get_start(), top.get_end())), run_time=rt)
            self.remove(pulse)
            la = new_a
            new_r = lab_r(t)
            pulse = Dot(bot.get_start(), radius=0.12, color=MEAN_C)
            self.play(Indicate(env[0], color=VALUE_C, scale_factor=1.05), *( [FadeOut(lr)] if lr else [] ),
                      FadeIn(new_r), run_time=rt)
            self.play(MoveAlongPath(pulse, Line(bot.get_start(), bot.get_end())), run_time=rt)
            self.remove(pulse)
            lr = new_r
            new_state = MathTex(rf"s_{{{t + 1}}}", font_size=40).move_to(state)
            items = [MathTex(rf"a_{{{t}}}", font_size=40, color=POLICY_C),
                     MathTex(rf"r_{{{t + 1}}}", font_size=40, color=MEAN_C),
                     MathTex(rf"s_{{{t + 1}}}", font_size=40)]
            new_trail = VGroup(*trail.copy(), *items).arrange(RIGHT, buff=0.3).move_to(trail)
            new_items = new_trail[len(trail):]
            self.play(Transform(state, new_state), trail.animate.move_to(new_trail[:len(trail)]),
                      LaggedStart(*[FadeIn(m, shift=0.2 * DOWN) for m in new_items], lag_ratio=0.2), run_time=rt)
            trail = VGroup(*trail, *new_items)
        dots = MathTex(r"\dots", font_size=40).next_to(trail, RIGHT, buff=0.3)
        self.play(FadeIn(dots))
        brace = Brace(VGroup(trail, dots), DOWN, buff=0.15, color=MUTED)
        word = Text("a trajectory", font_size=26, color=MUTED).next_to(brace, DOWN, buff=0.12)
        cap.say("that stream of states, actions and rewards is a trajectory")
        self.play(GrowFromCenter(brace), FadeIn(word))
        self.wait(2)

    # ------------------------------------------------------------------ act 2: a trajectory and its return
    def act2(self):
        cap = self.cap
        path, rewards = trajectory()
        T = len(rewards)
        squares, letters = grid_mob()
        cap.say("a 5×5 gridworld: reach the goal, avoid the pit, each step costs")
        self.play(LaggedStart(*[FadeIn(s) for s in squares], lag_ratio=0.02), FadeIn(letters), run_time=1.2)

        # Table in the right-hand panel: t | r_{t+1} | γ^t | γ^t r_{t+1}
        cols = PANEL_X + np.array([-2.3, -0.9, 0.6, 2.2])
        y0, dy = 1.1, 0.42
        head = VGroup(MathTex("t", font_size=32, color=MUTED), MathTex(r"r_{t+1}", font_size=32, color=MEAN_C),
                      MathTex(r"\gamma^t", font_size=32, color=STD_C),
                      MathTex(r"\gamma^t r_{t+1}", font_size=32, color=VALUE_C))
        for m, x in zip(head, cols):
            m.move_to([x, y0, 0])
        rule = Line([cols[0] - 0.5, y0 - 0.27, 0], [cols[-1] + 0.8, y0 - 0.27, 0], color=REF_C, stroke_width=1.5)
        row_y = lambda t: y0 - dy * (t + 1) - 0.05
        tcol = VGroup(*[Text(str(t), font_size=24, color=MUTED).move_to([cols[0], row_y(t), 0]) for t in range(T)])
        self.play(FadeIn(head[:2]), Create(rule))

        agent = Dot(cell_center(path[0]), radius=0.17, color=POLICY_C).set_stroke(WHITE, 2)
        self.play(FadeIn(agent, scale=0.5))
        cap.say("the agent walks to the goal; each step pays a reward")
        rcol = VGroup()
        for t in range(T):
            lab = Text(num(rewards[t]), font_size=24, color=CORRECT if rewards[t] > 0 else MEAN_C)
            lab.move_to([cols[1], row_y(t), 0])
            pop = lab.copy().next_to(agent.get_center() + (cell_center(path[t + 1]) - agent.get_center()), UP, buff=0.3)
            self.play(agent.animate.move_to(cell_center(path[t + 1])), FadeIn(tcol[t]), run_time=0.45)
            self.play(FadeIn(pop, shift=0.15 * UP), run_time=0.25)
            self.play(ReplacementTransform(pop, lab), run_time=0.4)
            rcol.add(lab)
        self.play(Indicate(squares[-1], color=CORRECT))

        # Build G_0 term by term.
        formula = MathTex(r"G_t = r_{t+1} + \gamma\, r_{t+2} + \gamma^2 r_{t+3} + \dots", font_size=34)
        formula.move_to([PANEL_X, 1.75, 0])
        gam = MathTex(rf"\gamma = {GAMMA:g}", font_size=32, color=STD_C).next_to(rule, DOWN, buff=0)
        gam.move_to([cols[-1] + 0.35, row_y(T) - 0.15, 0])
        cap.say("the return adds rewards, discounting the future by γ")
        self.play(Write(formula), FadeIn(head[2:]))
        total = MathTex(r"G_0 = ", tnum(0.0), font_size=36).move_to([PANEL_X - 0.6, row_y(T) - 0.35, 0])
        total[1].set_color(VALUE_C)
        gam.next_to(total, RIGHT, buff=0.6)
        self.play(FadeIn(total), FadeIn(gam))
        for t in range(T):
            w = Text(f"{GAMMA ** t:.3f}", font_size=24, color=STD_C).move_to([cols[2], row_y(t), 0])
            term = Text(num(GAMMA ** t * rewards[t], 3), font_size=24, color=VALUE_C).move_to([cols[3], row_y(t), 0])
            partial = MathTex(r"G_0 = ", tnum(discounted_return(rewards[:t + 1], GAMMA)), font_size=36)
            partial[1].set_color(VALUE_C)
            partial.move_to(total, aligned_edge=LEFT)
            rt = 0.7 if t < 2 or t == T - 1 else 0.35
            self.play(FadeIn(w), FadeIn(term), Indicate(rcol[t], scale_factor=1.15), Transform(total, partial),
                      run_time=rt)
        G0, G1 = discounted_return(rewards, GAMMA), discounted_return(rewards, 1.0)
        self.play(Circumscribe(total, color=VALUE_C))
        cap.say(f"the goal is {T - 1} steps after the first reward: weight γ{str(T - 1).translate(SUP)} = {GAMMA ** (T - 1):.2f}",
                color=STD_C)
        undisc = MathTex(rf"\text{{(}}\gamma = 1\text{{: }} {tnum(G1)}\text{{)}}", font_size=30, color=MUTED)
        undisc.next_to(total, DOWN, buff=0.2, aligned_edge=LEFT)
        self.play(FadeIn(undisc), Indicate(rcol[-1], color=CORRECT))
        self.wait(2.5)

    # ------------------------------------------------------------------ act 3: γ is the horizon
    def act3(self):
        cap = self.cap
        cells, *_ , terminal = gridworld()
        gammas = np.linspace(GAMMA_LO, GAMMA_HI, 90)
        Vs = np.array([optimal_values(g) for g in gammas])
        live = ~terminal
        vmin, vmax = Vs[:, live].min(), Vs[:, live].max()
        s0 = cells.index((0, 0))
        g = ValueTracker(GAMMA_LO)
        V_at = lambda: Vs[int(np.argmin(np.abs(gammas - g.get_value())))]

        squares, letters = grid_mob()
        by_cell = {cell: sq for sq, cell in zip(squares, [(r, c) for r in range(5) for c in range(5)])}
        heat_cells = [(s, cell) for s, cell in enumerate(cells) if not terminal[s]]

        def heat():
            V = V_at()
            grp = VGroup()
            for s, cell in heat_cells:
                k = (V[s] - vmin) / (vmax - vmin)
                sq = Square(CELL).move_to(cell_center(cell)).set_stroke(REF_C, 1.5, opacity=0.6) \
                    .set_fill(VALUE_C, 0.06 + 0.84 * k)
                lab = Text(num(V[s]), font_size=17).move_to(sq.get_center() + 0.12 * DOWN)
                grp.add(VGroup(sq, lab))
            return grp

        heatmap = always_redraw(heat)
        fixed = VGroup(*[by_cell[c] for c in by_cell if by_cell[c].ch in "#PG"])
        cap.say("the optimal value of every cell, V*(s), for a given γ")
        self.play(FadeIn(fixed), FadeIn(heatmap), FadeIn(letters))

        bellman = MathTex(r"V^*(s) = \max_a \sum_{s'} p(s' \mid s,a)\big[r + \gamma V^*(s')\big]", font_size=30)
        bellman.move_to([PANEL_X, 1.6, 0])
        fit(bellman, 5.8)
        g_row = always_redraw(lambda: MathTex(rf"\gamma = {g.get_value():.2f}", font_size=44, color=STD_C)
                              .move_to([PANEL_X, 0.4, 0]))
        h_row = always_redraw(lambda: VGroup(Text("horizon ≈ 1/(1−γ) =", font_size=26, color=MUTED),
                                             Text(hz(g.get_value()), font_size=26))
                              .arrange(RIGHT, buff=0.2).move_to([PANEL_X, -0.5, 0]))

        def v_row():
            v = V_at()[s0]
            return VGroup(MathTex(r"V^*(S) =", font_size=36),
                          Text(num(v), font_size=30, color=CORRECT if v > 0 else WRONG)
                          ).arrange(RIGHT, buff=0.2).move_to([PANEL_X, -1.4, 0])

        vrow = always_redraw(v_row)
        self.play(Write(bellman), FadeIn(g_row), FadeIn(h_row), FadeIn(vrow))
        self.wait(1)
        cap.say("small γ: short-sighted. γ → 1: far-sighted", color=STD_C)
        self.wait(2)
        cap.say(f"at γ = {GAMMA_LO:g}, S sees only the step cost; the goal is too far away",
                color=WRONG)
        self.play(Indicate(by_cell[(0, 0)], color=WRONG), run_time=0.8)
        self.wait(1.5)
        mid = GAMMA_STOPS[1]
        cap.say(f"at γ = {mid:g} the horizon is {hz(mid)}: S still can't see the goal", color=WRONG)
        self.play(g.animate.set_value(mid), run_time=3, rate_func=smooth)
        self.wait(1.5)
        cap.say(f"at γ = {GAMMA:g} the horizon is {hz(GAMMA)}: the goal pulls S upward", color=STD_C)
        self.play(g.animate.set_value(GAMMA), run_time=3.5, rate_func=smooth)
        self.wait(1.5)
        cap.say(f"at γ = {GAMMA_HI:g}, the goal's value reaches all the way back to S", color=CORRECT)
        self.play(g.animate.set_value(GAMMA_HI), run_time=4, rate_func=smooth)
        self.play(Circumscribe(vrow, color=CORRECT))
        self.wait(2)
        for m in (heatmap, g_row, h_row, vrow):
            m.clear_updaters()
