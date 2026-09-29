from manim import Scene, Square, Circle, VGroup, FadeIn, LaggedStart, config
import style


def _scene():
    config.dry_run = True          # no files written, animations are not rendered
    return Scene()


def test_part_titles_has_18_entries():
    assert len(style.PART_TITLES) == 18


def test_clear_act_removes_lagged_children_but_keeps_keep():
    s = _scene()
    keep = Circle()
    squares = VGroup(*[Square() for _ in range(4)])
    s.add(keep)
    s.play(LaggedStart(*[FadeIn(q) for q in squares]))   # adds children individually
    style.clear_act(s, keep=[keep])
    assert s.mobjects == [keep]


def test_captioner_shrinks_long_text_to_frame():
    s = _scene()
    anchor = Square().to_edge(style.UP)
    cap = style.Captioner(s, anchor)
    t = cap.say("x" * 300)
    assert t.width <= 13.0 + 1e-6


def test_captioner_replaces_previous_caption():
    s = _scene()
    cap = style.Captioner(s, Square())
    cap.say("first"); t2 = cap.say("second")
    texts = [m for m in s.mobjects if m is t2 or getattr(m, "text", None) == "first"]
    assert texts == [t2]


def test_captioner_after_clear_act_does_not_readd_stale_caption():
    s = _scene()
    cap = style.Captioner(s, Square())
    t1 = cap.say("first")
    style.clear_act(s)  # removes t1 from the scene without telling the Captioner
    assert t1 not in s.mobjects

    played = []
    orig_play = s.play

    def spy(*anims, **kwargs):
        played.extend(anims)
        return orig_play(*anims, **kwargs)

    s.play = spy
    t2 = cap.say("second")

    # the stale caption must not be handed to play() again (that would re-add it
    # to the scene via manim's animation bookkeeping and fade it out a second time)
    animated_mobjects = [getattr(a, "mobject", None) for a in played]
    assert t1 not in animated_mobjects
    assert s.mobjects == [t2]
