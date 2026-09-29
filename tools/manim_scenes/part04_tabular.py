"""Part 4 · Model-free learning: MC vs TD, the TD error, and SARSA vs Q-learning on CliffWalk."""
import numpy as np
from style import *

H, W = 4, 12
START, GOAL = (3, 0), (3, 11)
CLIFF = {(3, c) for c in range(1, 11)}
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def step(pos, a):
    r, c = pos; nr = min(max(r + MOVES[a][0], 0), H - 1); nc = min(max(c + MOVES[a][1], 0), W - 1)
    if (nr, nc) in CLIFF:
        return START, -100.0, False
    return (nr, nc), -1.0, (nr, nc) == GOAL


def train(method, episodes=500, alpha=0.5, eps=0.1, seed=0):
    rng = np.random.default_rng(seed); Q = np.zeros((H, W, 4))
    pol = lambda s: int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[s] + 1e-9 * rng.random(4)))
    for _ in range(episodes):
        s = START; a = pol(s); done = False
        while not done:
            s2, r, done = step(s, a); a2 = pol(s2)
            target = r if done else r + (Q[s2].max() if method == "q" else Q[s2][a2])
            Q[s][a] += alpha * (target - Q[s][a]); s, a = s2, a2
    return Q


def greedy_path(Q, max_len=60):
    s, path = START, [START]
    for _ in range(max_len):
        s, _, done = step(s, int(np.argmax(Q[s]))); path.append(s)
        if done:
            break
    return path


def td_error(r, v_next, v, gamma=1.0):
    return r + gamma * v_next - v


# ---------------------------------------------------------------------- scene constants and small helpers
GAMMA, ALPHA, EPS = 1.0, 0.5, 0.1      # CliffWalk runs undiscounted; α and ε are train()'s defaults
CHAIN_R = [1.0, 0.0, 2.0, 1.0]          # acts 1-2: rewards on the 4 transitions of a 5-state chain
CHAIN_V = [0.5, 1.0, 2.0, 1.5, 0.0]     # current guesses V(s_0 … s_3), and V(end) = 0
DELTA_T = 2                             # act 2: the TD error worked out at this step
N_ROLL = 1000                           # act 3: ε-greedy episodes used for the online readout
ARROWS = "↑→↓←"                         # action names in MOVES order
CELL = 0.95                             # act 3: grid cell size


def mc_returns(rewards, gamma=GAMMA):
    """G_t for every step of one finished episode."""
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G; out.append(G)
    return out[::-1]


def mc_update(V=CHAIN_V, rewards=CHAIN_R, alpha=ALPHA, gamma=GAMMA):
    """Every state moves toward its real return, all at once, after the episode ends."""
    G = mc_returns(rewards, gamma)
    return [V[t] + alpha * (G[t] - V[t]) for t in range(len(rewards))] + [V[-1]]


def td_targets(V=CHAIN_V, rewards=CHAIN_R, gamma=GAMMA):
    return [rewards[t] + gamma * V[t + 1] for t in range(len(rewards))]


def td_update(V=CHAIN_V, rewards=CHAIN_R, alpha=ALPHA, gamma=GAMMA):
    """One step at a time: V(s_t) moves by α·δ_t as soon as s_{t+1} is seen."""
    V = list(V)
    for t in range(len(rewards)):
        V[t] += alpha * td_error(rewards[t], V[t + 1], V[t], gamma)
    return V


def eps_rollout(Q, eps=EPS, seed=0, max_len=200):
    """One ε-greedy episode on a learned Q: list of (s, a, r, s', explored)."""
    rng = np.random.default_rng(seed); s, out = START, []
    for _ in range(max_len):
        explored = rng.random() < eps
        a = int(rng.integers(4)) if explored else int(np.argmax(Q[s]))
        s2, r, done = step(s, a); out.append((s, a, r, s2, explored)); s = s2
        if done:
            break
    return out


def first_clean_fall(Q, eps=EPS, max_seed=500):
    """The first seed whose rollout follows the greedy path and then falls on its first exploratory step."""
    for seed in range(max_seed):
        ro = eps_rollout(Q, eps, seed)
        ex = [i for i, x in enumerate(ro) if x[4]]
        if ex and ro[ex[0]][2] == step((2, 1), 2)[1] and ex[0] >= 6:
            return seed, ro[:ex[0] + 1]
    raise RuntimeError("no falling rollout found")


def online_stats(Q, eps=EPS, n=N_ROLL):
    """Mean return and fraction of episodes with a fall, following ε-greedy on Q."""
    rets, falls = [], 0
    cliff_r = step((2, 1), 2)[1]
    for seed in range(n):
        ro = eps_rollout(Q, eps, seed)
        rets.append(sum(x[2] for x in ro)); falls += any(x[2] == cliff_r for x in ro)
    return float(np.mean(rets)), falls / n


def num(v, d=2):
    return f"{v:.{d}f}".replace("-", "−")


def v_shape(target_tex, target_color):
    """style.update_shape with the estimate written V (acts 1-2 learn state values)."""
    m = update_shape(target_tex, target_color)
    for i in (0, 2, 8):
        m[i].become(MathTex("V", font_size=44).move_to(m[i]))
    return m


def cell_pos(rc, origin):
    r, c = rc
    return origin + np.array([(c - (W - 1) / 2) * CELL, ((H - 1) / 2 - r) * CELL, 0.0])


class Part04(Scene):
    def construct(self):
        self.title = title_card(self, 4, "MC, TD, SARSA and Q-learning")
        self.cap = Captioner(self, self.title)
        self.act1(); clear_act(self, keep=[self.title])
        self.act2(); clear_act(self, keep=[self.title])
        self.act3()
        takeaway(self, "bootstrapping trades variance for bias; on- vs off-policy decides what you learn")
        roadmap_outro(self, 4)

    # ------------------------------------------------------------------ shared chain pieces
    def chain(self, cx, y, gap=1.35, radius=0.34, V=CHAIN_V):
        nodes, vals, names, links, rlabs = VGroup(), VGroup(), VGroup(), VGroup(), VGroup()
        n = len(V)
        for i in range(n):
            p = np.array([cx + (i - (n - 1) / 2) * gap, y, 0])
            shape = (Square(2 * radius) if i == n - 1 else Circle(radius)).move_to(p).set_stroke(REF_C, 2.5)
            nodes.add(shape)
            vals.add(Text(num(V[i]), font_size=24, color=VALUE_C).next_to(shape, DOWN, buff=0.12))
            names.add((Text("end", font_size=20, color=MUTED) if i == n - 1 else
                       MathTex(rf"s_{{{i}}}", font_size=32, color=MUTED)).move_to(p))
        for i in range(n - 1):
            ln = Arrow(nodes[i].get_right(), nodes[i + 1].get_left(), buff=0.05, color=REF_C, stroke_width=3,
                       max_tip_length_to_length_ratio=0.3)
            links.add(ln)
            rlabs.add(MathTex(rf"r={CHAIN_R[i]:g}", font_size=26, color=MEAN_C).next_to(ln, UP, buff=0.22))
        return VGroup(nodes, vals, names, links, rlabs)

    def back_arrow(self, ch, frm, to, color, depth):
        a, b = ch[0][frm].get_bottom(), ch[0][to].get_bottom()
        a, b = a + 0.5 * DOWN, b + 0.5 * DOWN            # clear the V labels
        return CurvedArrow(a, b, angle=-depth, color=color, stroke_width=4, tip_length=0.2)

    # ------------------------------------------------------------------ act 1: wait vs bootstrap
    def act1(self):
        cap = self.cap
        ys, yf, yc = 2.0, 1.2, -0.05
        mc_head = Text("Monte Carlo", font_size=32, color=MEAN_C).move_to([-3.62, ys, 0])
        td_head = Text("TD(0)", font_size=32, color=VALUE_C).move_to([3.62, ys, 0])
        mc_f = fit(v_shape(r"G_t", MEAN_C), 5.8).move_to([-3.62, yf, 0])
        td_f = fit(v_shape(r"r + \gamma V(s')", VALUE_C), 5.8).move_to([3.62, yf, 0])
        divider = DashedLine([0, 2.3, 0], [0, -3.5, 0], color=GREY_D, stroke_width=2)
        mc, td = self.chain(-3.62, yc), self.chain(3.62, yc)
        cap.say("same episode, two ways to learn V from it")
        self.play(FadeIn(mc_head), FadeIn(td_head), Create(divider))
        self.play(Write(mc_f), Write(td_f))
        self.play(*[FadeIn(g) for g in (*mc[:3], *td[:3])], run_time=1)
        self.wait(0.5)

        # The walker steps through both chains together.
        walk = lambda ch, i: ch[0][i].get_top() + 0.22 * UP
        mw = Dot(walk(mc, 0), radius=0.1, color=POLICY_C)
        tw = Dot(walk(td, 0), radius=0.1, color=POLICY_C)
        status = Text("step 0", font_size=26, color=MUTED).move_to([0, -2.3, 0])
        cap.say("td updates after every step; mc just keeps walking")
        self.play(FadeIn(mw), FadeIn(tw), FadeIn(status))
        tgt, newV = td_targets(), td_update()
        mc_mem = VGroup()
        for t in range(len(CHAIN_R)):
            new_status = Text(f"step {t + 1}", font_size=26, color=MUTED).move_to(status)
            self.play(mw.animate.move_to(walk(mc, t + 1)), tw.animate.move_to(walk(td, t + 1)),
                      GrowArrow(mc[3][t]), GrowArrow(td[3][t]), FadeIn(mc[4][t]), FadeIn(td[4][t]),
                      Transform(status, new_status), run_time=0.7)
            arr = self.back_arrow(td, t + 1, t, VALUE_C, 1.6)
            lab = MathTex(rf"\text{{target }} {num(tgt[t])}", font_size=30, color=VALUE_C).next_to(arr, DOWN, buff=0.1)
            nv = Text(num(newV[t]), font_size=24, color=VALUE_C).move_to(td[1][t])
            wait_m = Text("…", font_size=30, color=MEAN_C).next_to(mc[0][t], DOWN, buff=0.55)
            mc_mem.add(wait_m)
            self.play(Create(arr), FadeIn(lab), Indicate(td_f[6], color=VALUE_C), FadeIn(wait_m), run_time=0.8)
            self.play(Transform(td[1][t], nv), Flash(td[0][t], color=VALUE_C, flash_radius=0.5),
                      FadeOut(arr), FadeOut(lab), run_time=0.7)
        self.wait(0.5)

        # Episode over: MC finally fires, every state at once, toward its real return.
        G, mcV = mc_returns(CHAIN_R), mc_update()
        end_status = Text("episode over", font_size=26, color=MEAN_C).move_to(status)
        cap.say("only at the end of the episode does mc see the real return G", color=MEAN_C)
        self.play(Transform(status, end_status), FadeOut(mc_mem))
        arrs = VGroup(*[self.back_arrow(mc, len(CHAIN_R), t, MEAN_C, 0.9 + 0.12 * (len(CHAIN_R) - t))
                        for t in range(len(CHAIN_R))])
        glabs = VGroup(*[MathTex(rf"G_{{{t}}}={num(G[t])}", font_size=26, color=MEAN_C)
                         .next_to(mc[0][t], DOWN, buff=0.55) for t in range(len(CHAIN_R))])
        # Labels sit under their state, arrows sweep below them.
        for a, t in zip(arrs, range(len(CHAIN_R))):
            a.shift(0.35 * DOWN)
        self.play(LaggedStart(*[Create(a) for a in arrs], lag_ratio=0.1), FadeIn(glabs),
                  Indicate(mc_f[6], color=MEAN_C), run_time=1.4)
        self.play(*[Transform(mc[1][t], Text(num(mcV[t]), font_size=24, color=VALUE_C).move_to(mc[1][t]))
                    for t in range(len(CHAIN_R))],
                  *[Flash(mc[0][t], color=MEAN_C, flash_radius=0.5) for t in range(len(CHAIN_R))], run_time=1)
        self.play(FadeOut(arrs))
        cap.say("mc waits for the real return; td uses its own next guess")
        self.wait(2.5)

    # ------------------------------------------------------------------ act 2: the TD error
    def act2(self):
        cap, t = self.cap, DELTA_T
        delta = MathTex(r"\delta_t", "=", r"r_{t+1}", "+", r"\gamma", r"V(s_{t+1})", "-", r"V(s_t)", font_size=54)
        delta.move_to(1.55 * UP)
        ch = self.chain(0, -0.35, gap=2.1, radius=0.45)
        cap.say("the bracket in td's update has a name: the td error δ")
        self.play(Write(delta))
        self.play(*[FadeIn(g) for g in ch[:3]], *[GrowArrow(a) for a in ch[3]], run_time=1)
        here = Dot(ch[0][t].get_top() + 0.25 * UP, radius=0.1, color=POLICY_C)
        self.play(FadeIn(here))
        self.play(here.animate.move_to(ch[0][t + 1].get_top() + 0.25 * UP), FadeIn(ch[4][t]), run_time=0.8)

        pieces = [(7, VGroup(ch[0][t], ch[1][t]), POLICY_C, "what we expected here"),
                  (2, ch[4][t], MEAN_C, "the reward we just got"),
                  (5, VGroup(ch[0][t + 1], ch[1][t + 1]), VALUE_C, "our guess for the rest, from s′")]
        rings = VGroup()
        for idx, target, col, words in pieces:
            ring = SurroundingRectangle(target, color=col, buff=0.1, stroke_width=4)
            rings.add(ring)
            cap.say(words, color=col)
            self.play(delta[idx].animate.set_color(col), Create(ring), run_time=0.8)
            self.wait(0.6)
        self.play(delta[4].animate.set_color(STD_C))

        d = td_error(CHAIN_R[t], CHAIN_V[t + 1], CHAIN_V[t], GAMMA)
        sub = MathTex(r"\delta_{" + str(t) + "}", "=", num(CHAIN_R[t]), "+", f"{GAMMA:g}", r"\cdot", num(CHAIN_V[t + 1]),
                      "-", num(CHAIN_V[t]), "=", num(d), font_size=44).move_to(2.0 * DOWN)
        sub[2].set_color(MEAN_C); sub[4].set_color(STD_C); sub[6].set_color(VALUE_C); sub[8].set_color(POLICY_C)
        sub[10].set_color(CORRECT if d > 0 else WRONG)
        self.play(Write(sub))
        newv = CHAIN_V[t] + ALPHA * d
        upd = MathTex(rf"V(s_{t}) \leftarrow {num(CHAIN_V[t])} + {ALPHA:g}\cdot{num(d)} = {num(newv)}",
                      font_size=38).next_to(sub, DOWN, buff=0.35)
        cap.say("δ > 0: better than expected, so nudge V(s) up", color=CORRECT)
        self.play(FadeIn(upd, shift=0.2 * UP))
        self.play(Transform(ch[1][t], Text(num(newv), font_size=24, color=VALUE_C).move_to(ch[1][t])),
                  Flash(ch[0][t], color=CORRECT, flash_radius=0.6))
        self.wait(1.5)
        cap.say("this δ is the most reused quantity in rl")
        later = Text("it returns as the advantage in actor-critic and as the building block of GAE",
                     font_size=24, color=MUTED)
        fit(later, 12).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(later))
        self.wait(2.5)

    # ------------------------------------------------------------------ act 3: cliff walk
    def act3(self):
        cap = self.cap
        o = np.array([0, 0.35, 0])
        cliff_r, step_r = step((2, 1), 2)[1], step(START, 0)[1]
        cells = VGroup()
        for r in range(H):
            for c in range(W):
                sq = Square(CELL).move_to(cell_pos((r, c), o)).set_stroke(REF_C, 1.5, opacity=0.6)
                if (r, c) in CLIFF:
                    sq.set_fill(WRONG, 0.55)
                cells.add(sq)
        s_lab = Text("S", font_size=30, weight=BOLD).move_to(cell_pos(START, o))
        g_lab = Text("G", font_size=30, weight=BOLD, color=CORRECT).move_to(cell_pos(GOAL, o))
        cliff_lab = Text(f"cliff  {num(cliff_r, 0)} and back to S", font_size=26) \
            .move_to((cell_pos((3, 1), o) + cell_pos((3, 10), o)) / 2)
        cap.say(f"cliff walk: every step costs {num(-step_r, 0)}, the cliff costs {num(-cliff_r, 0)}")
        self.play(LaggedStart(*[FadeIn(sq) for sq in cells], lag_ratio=0.005), run_time=1.2)
        self.play(FadeIn(s_lab), FadeIn(g_lab), FadeIn(cliff_lab))
        self.wait(1)

        Qq, Qs = train("q"), train("sarsa")
        qp, sp = greedy_path(Qq), greedy_path(Qs)
        line = lambda path, col, dy: VMobject().set_points_as_corners(
            [cell_pos(p, o) + dy * UP for p in path]).set_stroke(col, 6, opacity=0.9)
        q_line, s_line = line(qp, MEAN_C, -0.08), line(sp, POLICY_C, 0.08)
        base_y = cell_pos((3, 0), o)[1] - CELL / 2
        cols = [-5.7, -3.0, 1.0]
        at = lambda col, y: np.array([cols[col], base_y - y, 0])
        heads = VGroup(Text("greedy path", font_size=22, color=MUTED).move_to(at(1, 0.4), aligned_edge=LEFT),
                       Text(f"ε-greedy (ε = {EPS:g}), {N_ROLL} episodes", font_size=22, color=MUTED)
                       .move_to(at(2, 0.4), aligned_edge=LEFT))

        def legend(name, col, n, y):
            return VGroup(VGroup(Line(ORIGIN, 0.45 * RIGHT, color=col, stroke_width=6),
                                 Text(name, font_size=24, color=col)).arrange(RIGHT, buff=0.2)
                          .move_to(at(0, y), aligned_edge=LEFT),
                          Text(f"{n} moves, return {num(n * step_r, 0)}", font_size=24)
                          .move_to(at(1, y), aligned_edge=LEFT))
        q_leg = legend("Q-learning", MEAN_C, len(qp) - 1, 0.8)
        s_leg = legend("SARSA", POLICY_C, len(sp) - 1, 1.2)
        cap.say("q-learning learns the optimal edge path… and falls off while exploring", color=MEAN_C)
        self.play(Create(q_line), run_time=2)
        self.play(FadeIn(heads[0]), FadeIn(q_leg))
        self.wait(0.5)

        # A real ε-greedy episode on Q-learning's policy: greedy along the edge, then one random step.
        seed, ro = first_clean_fall(Qq)
        agent = Dot(cell_pos(START, o), radius=0.16, color=WHITE).set_z_index(3)
        self.play(FadeIn(agent, scale=0.5))
        for s, a, r, s2, ex in ro[:-1]:
            self.play(agent.animate.move_to(cell_pos(s2, o)), run_time=0.25, rate_func=linear)
        s, a, r, s2, ex = ro[-1]
        tag = Text(f"ε: random {ARROWS[a]}", font_size=26, color=WRONG).next_to(agent, UP, buff=0.15)
        self.play(FadeIn(tag, scale=1.3), agent.animate.set_color(WRONG))
        below = (s[0] + MOVES[a][0], s[1] + MOVES[a][1])
        self.play(agent.animate.move_to(cell_pos(below, o)), run_time=0.4)
        pen = Text(f"r = {num(r, 0)}", font_size=34, color=WRONG, weight=BOLD) \
            .move_to(cell_pos((below[0] - 2, below[1]), o))
        self.play(Flash(agent, color=WRONG, flash_radius=0.5), FadeIn(pen, scale=1.4), FadeOut(tag))
        self.play(agent.animate.move_to(cell_pos(s2, o)).set_color(WHITE), run_time=0.8)
        fall_txt = Text(f"ε-greedy episode (seed {seed}): {len(ro) - 1} greedy steps, one random "
                        f"{ARROWS[a]}, reward {num(r, 0)}, back to S", font_size=22, color=WRONG)
        fit(fall_txt, 12.2).move_to(at(0, 1.25), aligned_edge=LEFT)
        self.play(FadeIn(fall_txt))
        self.wait(1.5)
        self.play(FadeOut(agent), FadeOut(pen), FadeOut(fall_txt))

        cap.say("sarsa learns the policy it actually follows: keep your distance", color=POLICY_C)
        self.play(Create(s_line), run_time=2)
        self.play(FadeIn(s_leg))
        self.wait(1)

        (qm, qf), (sm, sf) = online_stats(Qq), online_stats(Qs)
        stat = lambda m, f, y: Text(f"falls in {f:.0%}, average return {num(m, 1)}", font_size=24) \
            .move_to(at(2, y), aligned_edge=LEFT)
        q_st, s_st = stat(qm, qf, 0.8), stat(sm, sf, 1.2)
        cap.say("with exploration still on, the safe path earns more", color=POLICY_C)
        self.play(FadeIn(heads[1]), FadeIn(q_st), FadeIn(s_st))
        self.play(Indicate(s_st, color=POLICY_C, scale_factor=1.05))
        self.wait(3)
