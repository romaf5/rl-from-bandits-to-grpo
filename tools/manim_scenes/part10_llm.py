"""Part 10 · LLMs as RL: an LLM is a policy, generation is a tree, and the reward arrives once at the end."""
import numpy as np
from style import *

MAPPING = [
    ("state", "prompt + tokens so far"),
    ("action", "next token"),
    ("policy", "next-token distribution"),
    ("transition", "append the token"),
    ("episode", "one full response"),
    ("reward", "one number at the end"),
]


def terminal_rewards(n_tokens, R):
    r = np.zeros(n_tokens); r[-1] = R; return r


# ---------------------------------------------------------------------- the notebook's toy task (sec10_llm.py)
VOCAB = [str(d) for d in range(10)] + ["+", "=", "<eos>", "<pad>", "<call>", "<obs>"]
PLUS, EQ, EOS = 10, 11, 12
A, B = 14, 17                                # the running example: 14+17=


def encode_prompt(a, b):
    return [a // 10, a % 10, PLUS, b // 10, b % 10, EQ]


def encode_answer(n):
    return [int(ch) for ch in str(n)] + [EOS]


def verify(a, b, completion):
    """The notebook's verifier: 1.0 iff the completion is exactly the canonical sum followed by <eos>."""
    return float(list(completion) == encode_answer(a + b))


def toks(ids):
    return [VOCAB[i] for i in ids]


# Illustrative next-token distributions for act 2's generation tree (not from a trained model).
# Keys are the completion so far (after the prompt); each branch list sums to 1.
NEXT_TOKEN_PROBS = {
    "": [("2", 0.25), ("3", 0.60), ("4", 0.15)],
    "3": [("0", 0.10), ("1", 0.70), ("2", 0.20)],
    "31": [("1", 0.10), ("<eos>", 0.90)],
}
TREE_PATH = toks(encode_answer(A + B))       # the highlighted path: 3, 1, <eos>


def path_probs(path=TREE_PATH):
    """Per-token probabilities π(a_t | s_t) along a path through the tree."""
    return [dict(NEXT_TOKEN_PROBS["".join(path[:t])])[tok] for t, tok in enumerate(path)]


def path_prob(path=TREE_PATH):
    """π(response | prompt) = product of the per-token probabilities along the path."""
    return float(np.prod(path_probs(path)))


WRONG_ANSWER = A + B + 1                     # act 3's wrong sample: 32<eos>
N_LONG = 40                                  # act 3's "real" response length (tokens drawn)


# ---------------------------------------------------------------------- scene helpers
def tok_box(s, color=WHITE, font_size=30, min_w=0.62):
    t = Text(s, font="Monospace", font_size=font_size, color=color)
    w = max(min_w, t.width + 0.3)
    rect = RoundedRectangle(corner_radius=0.1, width=w, height=min_w).set_stroke(color, 2)
    return VGroup(rect, t.move_to(rect))


def tok_row(strings, color=WHITE, buff=0.08, font_size=30, min_w=0.62):
    return VGroup(*[tok_box(s, color, font_size, min_w) for s in strings]).arrange(RIGHT, buff=buff)


class Part10(Scene):
    def construct(self):
        self.title = title_card(self, 10, "LLMs as RL: the setting")
        self.cap = Captioner(self, self.title)
        self.act1(); clear_act(self, keep=[self.title])
        self.act2(); clear_act(self, keep=[self.title])
        self.act3()
        takeaway(self, "every algorithm from parts 6–9 now trains a language model")
        roadmap_outro(self, 10)

    # ------------------------------------------------------------------ act 1: same loop, new agent
    def act1(self):
        cap = self.cap

        def box(name, sub, x):
            rect = RoundedRectangle(corner_radius=0.2, width=4.2, height=1.7).set_stroke(REF_C, 2.5)
            body = VGroup(Text(name, font_size=34), sub).arrange(DOWN, buff=0.2)
            return VGroup(rect, body).move_to([x, 0.6, 0])

        agent = box("agent", MathTex(r"\text{policy } \pi(a \mid s)", font_size=32, color=POLICY_C), -4.3)
        env = box("environment", MathTex(r"\text{dynamics } p(s', r \mid s, a)", font_size=30, color=VALUE_C), 4.3)
        top = Arrow(agent.get_right() + 0.45 * UP, env.get_left() + 0.45 * UP, buff=0.1, color=POLICY_C)
        bot = Arrow(env.get_left() + 0.45 * DOWN, agent.get_right() + 0.45 * DOWN, buff=0.1, color=MEAN_C)
        la = MathTex(r"a_t", font_size=40, color=POLICY_C).next_to(top, UP, buff=0.15)
        ls = MathTex(r"r_{t+1},\ s_{t+1}", font_size=40).next_to(bot, DOWN, buff=0.15)
        cap.say("part 1's loop: the agent acts, the world answers")
        self.play(FadeIn(agent), FadeIn(env))
        self.play(GrowArrow(top), GrowArrow(bot), FadeIn(la), FadeIn(ls))
        self.wait(1)

        # Morph: agent -> language model, environment -> append token.
        lm = box("language model", MathTex(r"\pi_\theta(a_t \mid s_t)", font_size=34, color=POLICY_C), -4.3)
        app = box("append token", MathTex(r"s_{t+1} = s_t \,\|\, a_t", font_size=34, color=VALUE_C), 4.3)
        lm[0].set_stroke(POLICY_C, 3); app[0].set_stroke(VALUE_C, 3)
        la2 = Text("next token", font_size=26, color=POLICY_C).next_to(top, UP, buff=0.15)
        ls2 = Text("prompt + tokens so far", font_size=24).next_to(bot, DOWN, buff=0.15)
        cap.say("same loop, new agent: the policy is a language model", color=POLICY_C)
        self.play(ReplacementTransform(agent, lm), ReplacementTransform(env, app),
                  ReplacementTransform(la, la2), ReplacementTransform(ls, ls2), run_time=1.6)
        self.wait(0.8)

        # Run the loop on the notebook's prompt: the state grows one token per turn.
        prompt = toks(encode_prompt(A, B))
        answer = toks(encode_answer(A + B))
        state = tok_row(prompt, REF_C).move_to([0, -1.9, 0])
        s_lab = MathTex("s_0 =", font_size=36).next_to(state, LEFT, buff=0.3)
        cap.say(f"the state starts as the prompt {''.join(prompt)}")
        self.play(FadeIn(state, shift=0.2 * UP), FadeIn(s_lab))
        for t, tok in enumerate(answer):
            new = tok_box(tok, POLICY_C).move_to(top.get_start())
            self.play(Indicate(lm[0], color=POLICY_C, scale_factor=1.04), FadeIn(new, scale=0.6), run_time=0.5)
            self.play(new.animate.move_to(top.get_end()), run_time=0.6)
            grown = VGroup(*state.copy(), new.copy())
            grown.arrange(RIGHT, buff=0.08).move_to([0, -1.9, 0])
            new_lab = MathTex(f"s_{t + 1} =", font_size=36).next_to(grown, LEFT, buff=0.3)
            self.play(Indicate(app[0], color=VALUE_C, scale_factor=1.04),
                      state.animate.move_to(grown[:-1]), new.animate.move_to(grown[-1]),
                      Transform(s_lab, new_lab), run_time=0.7)
            state = VGroup(*state, new)
        self.play(Indicate(state[-1], color=MEAN_C))
        self.wait(1)

        # Shrink the loop and fill in the mapping table.
        loop = VGroup(lm, app, top, bot, la2, ls2)
        self.play(FadeOut(state), FadeOut(s_lab),
                  loop.animate.scale(0.4).move_to([-4.3, -0.35, 0]), run_time=1.2)

        x_l, x_r, y0, dy = 0.9, 2.1, 1.4, 0.6
        head = VGroup(Text("RL", font_size=28, color=MUTED).move_to([x_l, y0, 0], aligned_edge=RIGHT),
                      Text("in an LLM", font_size=28, color=MUTED).move_to([x_r, y0, 0], aligned_edge=LEFT))
        rule = Line([x_l - 1.8, y0 - 0.3, 0], [x_r + 4.1, y0 - 0.3, 0], color=REF_C, stroke_width=1.5)
        mid = Line([(x_l + x_r) / 2, y0 + 0.25, 0], [(x_l + x_r) / 2, y0 - dy * len(MAPPING) - 0.2, 0],
                   color=REF_C, stroke_width=1.5)
        cap.say("each RL concept has an exact LLM meaning")
        self.play(FadeIn(head), Create(rule), Create(mid))
        for i, (rl, llm) in enumerate(MAPPING):
            y = y0 - dy * (i + 1) - 0.05
            left = Text(rl, font_size=26, color=MEAN_C).move_to([x_l, y, 0], aligned_edge=RIGHT)
            right = Text(llm, font_size=26).move_to([x_r, y, 0], aligned_edge=LEFT)
            self.play(FadeIn(left, shift=0.2 * RIGHT), FadeIn(right, shift=0.2 * LEFT), run_time=0.6)
            self.wait(0.35)
        cap.say("an llm is a policy; each token is an action", color=POLICY_C)
        self.wait(2.5)

    # ------------------------------------------------------------------ act 2: generation is a tree
    def act2(self):
        cap = self.cap
        xs = [-4.8, -1.6, 1.3, 4.3]
        y_mid, dy = 0.2, 1.25
        root = tok_row(toks(encode_prompt(A, B)), REF_C, buff=0.04, font_size=22, min_w=0.5).move_to([xs[0], y_mid, 0])
        cap.say(f"start from the prompt {''.join(toks(encode_prompt(A, B)))}")
        self.play(FadeIn(root, shift=0.2 * RIGHT))

        state_lab = Text("state", font_size=26, color=MUTED)

        def state_row(prefix_toks):
            row = tok_row(toks(encode_prompt(A, B)) + prefix_toks, WHITE, buff=0.05, font_size=24)
            for b in row[6:]:
                b.set_color(POLICY_C)
            return VGroup(state_lab.copy(), row).arrange(RIGHT, buff=0.3).move_to([0, -2.55, 0])

        state = state_row([])
        self.play(FadeIn(state))

        parent, prefix = root, ""
        path_nodes, path_edges = [], []
        for depth, chosen in enumerate(TREE_PATH):
            branches = NEXT_TOKEN_PROBS[prefix]
            k = [tok for tok, _ in branches].index(chosen)
            nodes, edges, labels = VGroup(), VGroup(), VGroup()
            for j, (tok, p) in enumerate(branches):
                # chosen child stays on the centre line; alternatives fan out above and below
                off = j - k
                if len(branches) == 2:
                    off = -1 if j < k else (1 if j > k else 0)
                y = y_mid - off * dy
                node = tok_box(tok, POLICY_C if j == k else MUTED, font_size=26).move_to([xs[depth + 1], y, 0])
                edge = Line(parent.get_right(), node.get_left(), buff=0.08,
                            color=POLICY_C if j == k else REF_C, stroke_width=3 if j == k else 2)
                lab = Text(f"{p:.2f}", font_size=24, color=POLICY_C if j == k else MUTED)
                lab.move_to(edge.point_from_proportion(0.5) + (0.28 * UP if y >= y_mid else 0.28 * DOWN))
                if y != y_mid:
                    lab.shift(0.25 * LEFT)
                nodes.add(node); edges.add(edge); labels.add(lab)
            if depth == 0:
                cap.say("the policy gives every next token a probability")
            self.play(LaggedStart(*[AnimationGroup(Create(e), FadeIn(n), FadeIn(l))
                                    for e, n, l in zip(edges, nodes, labels)], lag_ratio=0.25), run_time=1.3)
            if depth == 0:
                sums = Text(f"sum = {sum(p for _, p in branches):.2f}", font_size=22, color=MUTED)
                sums.next_to(nodes, DOWN, buff=0.2)
                self.play(FadeIn(sums)); self.wait(0.6); self.play(FadeOut(sums))
                cap.say("sample one; the others become roads not taken")
            others = [m for j in range(len(branches)) if j != k for m in (nodes[j], edges[j], labels[j])]
            self.play(Indicate(nodes[k], color=POLICY_C), *[m.animate.set_opacity(0.35) for m in others],
                      Transform(state, state_row(TREE_PATH[:depth + 1])), run_time=0.8)
            if depth == 0:
                cap.say("the state is everything generated so far", color=POLICY_C)
            parent, prefix = nodes[k], prefix + chosen
            path_nodes.append(nodes[k]); path_edges.append((edges[k], labels[k]))
        self.play(path_nodes[-1][0].animate.set_stroke(CORRECT, 3), path_nodes[-1][1].animate.set_color(CORRECT))
        self.wait(0.8)

        # Probability of the whole response = product along the path.
        probs = path_probs()
        answer = "".join(t for t in TREE_PATH if t != "<eos>")
        prod = MathTex(rf"\pi_\theta(\texttt{{{answer}<eos>}} \mid \texttt{{{A}+{B}=}})",
                       "=", r" \times ".join(f"{p:.2f}" for p in probs), "=", f"{path_prob():.3f}", font_size=34)
        prod[0].set_color(POLICY_C); prod[4].set_color(MEAN_C)
        prod.move_to([0, -3.35, 0]); fit(prod, 12.5)
        cap.say("a response's probability is the product of its token probabilities")
        self.play(*[Indicate(l, color=MEAN_C) for _, l in path_edges], run_time=1)
        self.play(Write(prod), run_time=1.5)
        self.wait(2.5)

    # ------------------------------------------------------------------ act 3: one reward at the end
    def act3(self):
        cap = self.cap
        prompt = toks(encode_prompt(A, B))

        def strip(answer, y):
            comp_ids = encode_answer(answer)
            comp = toks(comp_ids)
            R = verify(A, B, comp_ids)
            r = terminal_rewards(len(comp), R)
            row = VGroup(tok_row(prompt, REF_C, buff=0.06), tok_row(comp, POLICY_C, buff=0.06)).arrange(RIGHT, buff=0.3)
            row.move_to([-0.9, y, 0])
            rew = VGroup(*[Text(f"{v:g}", font_size=28, color=MUTED).next_to(b, DOWN, buff=0.25)
                           for b, v in zip(row[1], r)])
            return row, rew, r, R

        rlab = MathTex("r_t", font_size=34, color=MEAN_C)
        row, rew, r, R = strip(A + B, 0.6)
        rlab.move_to([row[1].get_left()[0] - 0.55, rew.get_center()[1], 0])
        b_p = Brace(row[0], UP, buff=0.12, color=MUTED)
        t_p = Text("prompt (given)", font_size=24, color=MUTED).next_to(b_p, UP, buff=0.1)
        b_c = Brace(row[1], UP, buff=0.12, color=POLICY_C)
        t_c = Text("response (actions)", font_size=24, color=POLICY_C).next_to(b_c, UP, buff=0.1)
        cap.say("the verifier checks the finished response against the true sum")
        self.play(FadeIn(row[0]), GrowFromCenter(b_p), FadeIn(t_p))
        self.play(LaggedStart(*[FadeIn(b, shift=0.2 * RIGHT) for b in row[1]], lag_ratio=0.3),
                  GrowFromCenter(b_c), FadeIn(t_c))
        self.play(FadeIn(rlab), LaggedStart(*[FadeIn(x) for x in rew], lag_ratio=0.35), run_time=1.3)
        eos = row[1][-1]
        glow = SurroundingRectangle(eos, color=CORRECT, buff=0.08, corner_radius=0.12)
        rew_R = Text(f"{r[-1]:g}", font_size=28, color=CORRECT, weight=BOLD).move_to(rew[-1])
        self.play(eos[0].animate.set_fill(CORRECT, 0.35).set_stroke(CORRECT, 3), Create(glow),
                  Transform(rew[-1], rew_R))
        mark = Text(f"correct: R = {R:g}", font_size=28, color=CORRECT).next_to(glow, RIGHT, buff=0.3)
        self.play(FadeIn(mark))
        cap.say("every token earns 0 except the last, which carries the whole reward", color=MEAN_C)
        self.wait(1.5)

        # A wrong sample: all zeros, and no hint of which token was to blame.
        row2, rew2, r2, R2 = strip(WRONG_ANSWER, -1.7)
        rlab2 = rlab.copy().move_to([rlab.get_center()[0], rew2.get_center()[1], 0])
        cap.say(f"a wrong answer, {''.join(toks(encode_answer(WRONG_ANSWER)))}: every reward is 0", color=WRONG)
        self.play(FadeIn(row2), FadeIn(rlab2), LaggedStart(*[FadeIn(x) for x in rew2], lag_ratio=0.3), run_time=1.3)
        self.play(row2[1][-1][0].animate.set_stroke(WRONG, 3),
                  FadeIn(Text(f"wrong: R = {R2:g}", font_size=28, color=WRONG)
                         .next_to(row2[1][-1], RIGHT, buff=0.38)))
        q = Text("?", font_size=36, color=WRONG)
        qs = VGroup(*[q.copy().next_to(b, UP, buff=0.12) for b in row2[1][:-1]])
        cap.say("which token was the mistake? the reward does not say", color=WRONG)
        self.play(LaggedStart(*[FadeIn(x, scale=1.5) for x in qs], lag_ratio=0.3))
        self.wait(2)

        # A realistic response: many tokens, one number at the very end.
        clear_act(self, keep=[self.title, self.cap.current])
        rl = terminal_rewards(N_LONG, 1.0)
        cells = VGroup(*[Square(0.26).set_stroke(POLICY_C, 1.5).set_fill(POLICY_C, 0.15) for _ in rl])
        cells.arrange(RIGHT, buff=0.04).move_to([0, 0.5, 0])
        rdots = VGroup(*[Dot(radius=0.06, color=CORRECT if v else REF_C).next_to(c, DOWN, buff=0.3)
                         for c, v in zip(cells, rl)])
        brace = Brace(cells, UP, buff=0.12, color=MUTED)
        blab = Text(f"one response: {N_LONG} tokens here, thousands in practice", font_size=24, color=MUTED)
        blab.next_to(brace, UP, buff=0.1)
        rl_lab = MathTex("r_t", font_size=32, color=MEAN_C).next_to(rdots, LEFT, buff=0.3)
        cap.say("a real response: thousands of actions, one reward")
        self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.03), GrowFromCenter(brace), FadeIn(blab),
                  run_time=1.8)
        self.play(FadeIn(rl_lab), LaggedStart(*[FadeIn(d) for d in rdots[:-1]], lag_ratio=0.02), run_time=1.2)
        self.play(cells[-1].animate.set_fill(CORRECT, 0.8).set_stroke(CORRECT, 2),
                  FadeIn(rdots[-1], scale=3), Flash(rdots[-1], color=CORRECT))
        nz = Text(f"non-zero rewards: {int(np.count_nonzero(rl))} of {N_LONG}", font_size=28, color=CORRECT)
        nz.next_to(rdots, DOWN, buff=0.6)
        self.play(FadeIn(nz))
        cap.say("a single scalar for the whole response: credit assignment is hard", color=MEAN_C)
        self.wait(3)
