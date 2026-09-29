"""Shared look for every Part video. Scenes import constants and helpers from here only."""
import numpy as np
from manim import *

CORRECT, WRONG = GREEN_C, RED_C
MEAN_C, STD_C = YELLOW, BLUE_C
POLICY_C, REF_C, VALUE_C = TEAL_C, GREY_B, PURPLE_B
MUTED = GREY_A
MAX_TEXT_WIDTH = 13.0

PART_TITLES = [
    "Foundations", "Bandits", "Dynamic programming", "MC / TD / Q-learning", "DQN", "REINFORCE",
    "Actor-critic + GAE", "TRPO", "PPO", "LLMs as RL", "Reward models", "RLHF with PPO", "DPO",
    "RLOO / REINFORCE++", "GRPO", "After GRPO", "Agentic RL", "The map",
]


def fit(mob, max_width=MAX_TEXT_WIDTH):
    if mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def title_card(scene, n, title):
    kicker = Text(f"Part {n}", font_size=30, color=MUTED)
    head = fit(Text(title, font_size=56))
    VGroup(kicker, head).arrange(DOWN, buff=0.3)
    scene.play(FadeIn(kicker, shift=0.2 * UP), Write(head))
    scene.wait(0.8)
    top = fit(Text(f"{n} · {title}", font_size=40)).to_edge(UP)
    scene.play(FadeOut(kicker), ReplacementTransform(head, top))
    return top


class Captioner:
    """One caption slot under `anchor`; each say() replaces the previous caption."""

    def __init__(self, scene, anchor):
        self.scene, self.anchor, self.current = scene, anchor, None

    def say(self, text, color=MUTED, wait=0):
        new = fit(Text(text, font_size=26, color=color)).next_to(self.anchor, DOWN, buff=0.25)
        stale = self.current is not None and self.current not in self.scene.mobjects
        anims = [FadeIn(new)]
        if self.current is not None and not stale:
            anims.append(FadeOut(self.current))
        self.scene.play(*anims)
        if self.current is not None and not stale:
            self.scene.remove(self.current)
        self.current = new
        if wait:
            self.scene.wait(wait)
        return new

    def clear(self):
        if self.current is not None:
            if self.current in self.scene.mobjects:
                self.scene.play(FadeOut(self.current))
                self.scene.remove(self.current)
            self.current = None


def clear_act(scene, keep=()):
    """Fade out and fully remove everything except `keep` (children added by LaggedStart included)."""
    keep_ids = {id(m) for k in keep for m in k.get_family()}
    doomed = [m for m in scene.mobjects if id(m) not in keep_ids]
    if doomed:
        for m in doomed:
            m.clear_updaters()
        scene.play(*[FadeOut(m) for m in doomed])
    scene.remove(*doomed)
    scene.mobjects = [m for m in scene.mobjects if id(m) in keep_ids]


def bar(ax, x, v, color, width=0.64, outline=False):
    v = v if abs(v) > 1e-3 else 1e-3
    x0, x1 = x - width / 2, x + width / 2
    poly = Polygon(ax.c2p(x0, 0), ax.c2p(x1, 0), ax.c2p(x1, v), ax.c2p(x0, v))
    if outline:
        return poly.set_stroke(WHITE, 2, opacity=0.8).set_fill(opacity=0)
    return poly.set_stroke(color, 2).set_fill(color, 0.75)


def value_label(ax, x, v, font_size=26):
    lab = DecimalNumber(v, num_decimal_places=2, include_sign=True, font_size=font_size)
    return lab.next_to(ax.c2p(x, v), UP if v >= 0 else DOWN, buff=0.1)


def update_shape(target_tex, target_color=YELLOW):
    m = MathTex(r"Q", r"\leftarrow", r"Q", "+", r"\alpha", r"\big(", target_tex, "-", r"Q", r"\big)", font_size=44)
    m[6].set_color(target_color); m[4].set_color(STD_C)
    return m


def takeaway(scene, text, keep=()):
    clear_act(scene, keep=keep)
    t = fit(Text(text, font_size=34))
    scene.play(FadeIn(t, shift=0.2 * UP))
    scene.wait(2.5)
    scene.play(FadeOut(t))
    scene.remove(t)


def roadmap_outro(scene, n):
    clear_act(scene)
    dots = VGroup(*[Dot(radius=0.12, color=GREY_D) for _ in PART_TITLES]).arrange(RIGHT, buff=0.45)
    line = Line(dots[0].get_center(), dots[-1].get_center(), color=GREY_D, stroke_width=2)
    dots[n - 1].set_color(YELLOW).scale(1.6)
    for i in range(n - 1):
        dots[i].set_color(GREY_B)
    here = Text(PART_TITLES[n - 1], font_size=28, color=YELLOW).next_to(dots[n - 1], UP, buff=0.3)
    nxt = (Text(f"next: {PART_TITLES[n]}", font_size=22, color=MUTED).next_to(dots[n], DOWN, buff=0.35)
           if n < len(PART_TITLES) else VMobject())
    scene.play(Create(line), LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.03), run_time=1)
    scene.play(FadeIn(here), FadeIn(nxt))
    scene.wait(1.5)
