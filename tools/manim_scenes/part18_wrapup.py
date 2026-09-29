"""Part 18 · The map: every algorithm in the notebook as one family tree, and how to pick one."""
import numpy as np
from style import *

NODES = {  # name -> (part, family)
    "Bandits": (2, "tabular"), "Value/Policy iteration": (3, "dp"), "TD / SARSA / Q-learning": (4, "tabular"),
    "DQN": (5, "value"), "REINFORCE": (6, "pg"), "Actor-critic + GAE": (7, "pg"), "TRPO": (8, "pg"),
    "PPO": (9, "pg"), "Reward model": (11, "llm"), "PPO-RLHF": (12, "llm"), "DPO": (13, "llm"),
    "RLOO / REINFORCE++": (14, "llm"), "GRPO": (15, "llm"), "Dr. GRPO / DAPO / GSPO / CISPO": (16, "llm"),
    "Agentic RL": (17, "llm"),
}
EDGES = [  # (from, to, label)
    ("Value/Policy iteration", "TD / SARSA / Q-learning", "sample, don't plan"),
    ("TD / SARSA / Q-learning", "DQN", "+ network"),
    ("Bandits", "REINFORCE", "learn the policy"),
    ("REINFORCE", "Actor-critic + GAE", "+ critic"),
    ("Actor-critic + GAE", "TRPO", "+ trust region"),
    ("TRPO", "PPO", "clip instead"),
    ("PPO", "PPO-RLHF", "+ LM, + KL"),
    ("Reward model", "PPO-RLHF", "reward"),
    ("Reward model", "DPO", "closed form"),
    ("PPO-RLHF", "RLOO / REINFORCE++", "− critic"),
    ("RLOO / REINFORCE++", "GRPO", "group baseline"),
    ("GRPO", "Dr. GRPO / DAPO / GSPO / CISPO", "fix the dials"),
    ("GRPO", "Agentic RL", "multi-turn"),
    # Added (not in the brief): without it {VI, TD, DQN} is a separate component and connected() is False.
    # Policy iteration's evaluate/improve loop is exactly what actor-critic does with a learned critic.
    ("Value/Policy iteration", "Actor-critic + GAE", "evaluate + improve"),
]


def connected(nodes=NODES, edges=EDGES):
    adj = {n: set() for n in nodes}
    for a, b, _ in edges:
        adj[a].add(b); adj[b].add(a)
    seen, stack = set(), [next(iter(nodes))]
    while stack:
        n = stack.pop()
        if n not in seen:
            seen.add(n); stack.extend(adj[n] - seen)
    return seen == set(nodes)


# ---------------------------------------------------------------- layout (hand-placed, frame units)
# Three columns: tabular / DP / value-based on the left, policy gradient in the centre, LLM on the right
# (the LLM column is two sub-columns, read bottom-up from PPO-RLHF). Every edge is horizontal or vertical.
COL_X = {"left": -5.65, "centre": -1.6, "llm_a": 1.9, "llm_b": 5.4}
ROW_Y = [1.75, 0.25, -1.25, -2.75]
POS = {
    "Bandits": (COL_X["left"] - 0.45, ROW_Y[0]),  # nudged left so "learn the policy" fits on one line
    "Value/Policy iteration": (COL_X["left"], ROW_Y[1]),
    "TD / SARSA / Q-learning": (COL_X["left"], ROW_Y[2]),
    "DQN": (COL_X["left"], ROW_Y[3]),
    "REINFORCE": (COL_X["centre"], ROW_Y[0]),
    "Actor-critic + GAE": (COL_X["centre"], ROW_Y[1]),
    "TRPO": (COL_X["centre"], ROW_Y[2]),
    "PPO": (COL_X["centre"], ROW_Y[3]),
    "PPO-RLHF": (COL_X["llm_a"], ROW_Y[3]),
    "RLOO / REINFORCE++": (COL_X["llm_a"], ROW_Y[2]),
    "GRPO": (COL_X["llm_a"], ROW_Y[1]),
    "Dr. GRPO / DAPO / GSPO / CISPO": (COL_X["llm_a"], ROW_Y[0]),
    "Reward model": (COL_X["llm_b"], ROW_Y[3]),
    "DPO": (COL_X["llm_b"], ROW_Y[2]),
    "Agentic RL": (COL_X["llm_b"], ROW_Y[1]),
}
PANEL = np.array([0, -3.72, 0])  # strip under the graph: legend in act 1, question in act 2
SHORT = {  # on-screen labels (two lines where the full name is long)
    "Value/Policy iteration": "value / policy\niteration",
    "TD / SARSA / Q-learning": "TD / SARSA\nQ-learning",
    "Actor-critic + GAE": "actor-critic\n+ GAE",
    "Reward model": "reward\nmodel",
    "RLOO / REINFORCE++": "RLOO /\nREINFORCE++",
    "Dr. GRPO / DAPO / GSPO / CISPO": "Dr. GRPO / DAPO\nGSPO / CISPO",
}
EDGE_SHORT = {"evaluate + improve": "evaluate\n+ improve"}
EDGE_SIDE = {"closed form": LEFT}  # default: UP for horizontal, RIGHT for vertical
FAMILY_C = {"tabular": STD_C, "dp": STD_C, "value": VALUE_C, "pg": POLICY_C, "llm": MEAN_C}
LEGEND = [("tabular / DP", "dp"), ("value-based", "value"), ("policy gradient", "pg"), ("LLM", "llm")]
ERAS = [  # (caption, parts in that era)
    ("planning → sampling", range(2, 6)),
    ("values → policies", range(6, 10)),
    ("policies → language models", range(11, 18)),
]

# Act 2: three questions from the notebook's decision guide (sec18_wrapup.py), two answers each.
QUESTIONS = [
    ("1 · do you have a verifier?", [
        ("yes: GRPO with the 2025 fixes, all the way to agents",
         {"GRPO", "Dr. GRPO / DAPO / GSPO / CISPO", "Agentic RL"}),
        ("only preferences: a reward model + PPO, or DPO on a tight budget",
         {"Reward model", "PPO-RLHF", "DPO"}),
    ]),
    ("2 · can you afford a critic?", [
        ("yes: a learned value baseline, actor-critic up to PPO-RLHF",
         {"Actor-critic + GAE", "TRPO", "PPO", "PPO-RLHF"}),
        ("no: sampled baselines, REINFORCE → RLOO → GRPO",
         {"REINFORCE", "RLOO / REINFORCE++", "GRPO", "Dr. GRPO / DAPO / GSPO / CISPO"}),
    ]),
    ("3 · on- or off-policy?", [
        ("off-policy or offline: Q-learning, DQN's replay, DPO's fixed dataset",
         {"TD / SARSA / Q-learning", "DQN", "DPO"}),
        ("(almost) on-policy: SARSA and policy gradients need fresh samples",
         {"TD / SARSA / Q-learning", "REINFORCE", "Actor-critic + GAE", "TRPO", "PPO", "PPO-RLHF", "RLOO / REINFORCE++", "GRPO",
          "Dr. GRPO / DAPO / GSPO / CISPO", "Agentic RL"}),
    ]),
]
# The combined TD box is split by the notebook's table: MC / TD(0) / SARSA are on-policy, Q-learning is off-policy.
# When it is lit in Q3, a small tag says which half of the box is meant.
TAGS = {
    QUESTIONS[2][1][0][0]: {"TD / SARSA / Q-learning": "Q-learning\nonly"},
    QUESTIONS[2][1][1][0]: {"TD / SARSA / Q-learning": "TD / SARSA\nonly"},
}
for _q, _answers in QUESTIONS:
    assert all(s <= set(NODES) and len(c) <= 70 for c, s in _answers)
assert set(POS) == set(NODES)

NODE_FS, NUM_FS, EDGE_FS = 22, 16, 18


def make_node(name):
    part, fam = NODES[name]
    col = FAMILY_C[fam]
    label = Text(SHORT.get(name, name), font_size=NODE_FS, line_spacing=0.8)
    num = Text(str(part), font_size=NUM_FS, color=MUTED)
    body = VGroup(num, label).arrange(RIGHT, buff=0.14)
    box = RoundedRectangle(corner_radius=0.12, width=body.width + 0.34, height=max(body.height + 0.26, 0.62))
    box.set_stroke(col, 3).set_fill(col, 0.15)
    return VGroup(box, body.move_to(box)).move_to([*POS[name], 0])


def edge_text(text):
    lines = EDGE_SHORT.get(text, text).split("\n")
    return VGroup(*[Text(l, font_size=EDGE_FS, color=MUTED) for l in lines]).arrange(DOWN, buff=0.06)


def make_edge(a_node, b_node, text):
    (ax_, ay), (bx, by) = a_node.get_center()[:2], b_node.get_center()[:2]
    if np.isclose(ay, by):   # horizontal edge, label above
        d = RIGHT if bx > ax_ else LEFT
        arrow = Arrow(a_node[0].get_edge_center(d), b_node[0].get_edge_center(-d), buff=0.06,
                      stroke_width=3, max_tip_length_to_length_ratio=0.12, tip_length=0.18, color=REF_C)
        lab = edge_text(text).next_to(arrow, EDGE_SIDE.get(text, UP), buff=0.08)
    else:                    # vertical edge, label to the right of the line
        assert np.isclose(ax_, bx)
        d = UP if by > ay else DOWN
        arrow = Arrow(a_node[0].get_edge_center(d), b_node[0].get_edge_center(-d), buff=0.06,
                      stroke_width=3, max_tip_length_to_length_ratio=0.3, tip_length=0.18, color=REF_C)
        lab = edge_text(text).next_to(arrow, EDGE_SIDE.get(text, RIGHT), buff=0.12)
    return VGroup(arrow, lab)


class Part18(Scene):
    def construct(self):
        self.title = title_card(self, 18, "The map: how every part connects")
        self.cap = Captioner(self, self.title)
        self.act1()
        clear_act(self, keep=[self.title, self.graph])
        self.act2()
        takeaway(self, "one idea throughout: move toward what was better than expected")
        roadmap_outro(self, 18)

    # ------------------------------------------------------------------ act 1: the family tree
    def act1(self):
        self.nodes = {n: make_node(n) for n in NODES}
        self.edges = [(a, b, make_edge(self.nodes[a], self.nodes[b], t)) for a, b, t in EDGES]
        self.graph = VGroup(*self.nodes.values(), *[e for _, _, e in self.edges])

        legend = VGroup(*[
            VGroup(RoundedRectangle(corner_radius=0.05, width=0.3, height=0.22)
                   .set_stroke(FAMILY_C[f], 2).set_fill(FAMILY_C[f], 0.4),
                   Text(lab, font_size=18, color=MUTED)).arrange(RIGHT, buff=0.12)
            for lab, f in LEGEND
        ]).arrange(RIGHT, buff=0.5).move_to(PANEL)
        self.legend = legend
        self.play(FadeIn(legend))

        shown = set()
        for caption, parts in ERAS:
            self.cap.say(caption)
            for n in sorted((n for n in NODES if NODES[n][0] in parts), key=lambda n: NODES[n][0]):
                anims = [FadeIn(self.nodes[n], scale=0.8)]
                shown.add(n)
                for a, b, e in self.edges:
                    if n in (a, b) and a in shown and b in shown:
                        anims.append(AnimationGroup(GrowArrow(e[0]), FadeIn(e[1]), lag_ratio=0.5))
                self.play(*anims, run_time=0.8)
            self.wait(1.5)
        self.cap.say(f"{len(NODES)} methods, one family tree", wait=2)

    # ------------------------------------------------------------------ act 2: three questions
    def restyle(self, lit=None):
        """lit=None: everything normal; else light the nodes in `lit` and dim the rest."""
        anims = []
        for n, node in self.nodes.items():
            fam_c = FAMILY_C[NODES[n][1]]
            node.generate_target()
            box, body = node.target
            if lit is None:
                box.set_stroke(fam_c, 3, opacity=1).set_fill(fam_c, 0.15)
                body[0].set_opacity(1); body[1].set_opacity(1)
            elif n in lit:
                box.set_stroke(WHITE, 6, opacity=1).set_fill(fam_c, 0.4)
                body[0].set_opacity(1); body[1].set_opacity(1)
            else:
                box.set_stroke(fam_c, 3, opacity=0.2).set_fill(fam_c, 0.03)
                body[0].set_opacity(0.2); body[1].set_opacity(0.2)
            anims.append(MoveToTarget(node))
        for a, b, e in self.edges:
            on = lit is None or (a in lit and b in lit)
            e.generate_target()
            e.target[0].set_color(WHITE if lit is not None and on else REF_C).set_opacity(1 if on else 0.15)
            e.target[1].set_opacity(1 if on else 0.15)
            anims.append(MoveToTarget(e))
        return anims

    def act2(self):
        cap = self.cap
        self.play(FadeOut(self.legend))
        self.remove(self.legend)
        cap.say("three questions from the decision guide", wait=1)
        question = None
        for q, answers in QUESTIONS:
            new_q = Text(q, font_size=24, color=MEAN_C).move_to(PANEL)
            self.play(FadeIn(new_q), *([FadeOut(question)] if question else []))
            if question:
                self.remove(question)
            question = new_q
            for caption, subset in answers:
                tags = VGroup(*[VGroup(*[Text(l, font_size=EDGE_FS, color=MEAN_C) for l in t.split("\n")])
                                .arrange(DOWN, buff=0.06).next_to(self.nodes[n][0], RIGHT, buff=0.15)
                                for n, t in TAGS.get(caption, {}).items()])
                self.play(*self.restyle(subset), FadeIn(tags), run_time=0.9)
                cap.say(caption, wait=2.2)
                if len(tags):
                    self.play(FadeOut(tags), run_time=0.4)
                    self.remove(tags)
        self.play(*self.restyle(None), FadeOut(question), run_time=0.9)
        self.remove(question)
        model = self.nodes["Value/Policy iteration"]
        cap.say("pick by what you have: a model, a verifier, preferences, compute")
        self.play(Indicate(model, color=WHITE, scale_factor=1.1), run_time=1.2)
        self.wait(2)
