# Companion videos

One ManimCE scene per notebook Part. Outputs go to `videos/partNN.mp4` (1080p30).

## Install (Ubuntu / WSL)

    sudo apt-get install libcairo2-dev libpango1.0-dev pkg-config ffmpeg python3-venv \
        texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvisvgm cm-super
    python3 -m venv .venv && .venv/bin/pip install -r tools/manim_scenes/requirements.txt

## Render

    .venv/bin/python tools/manim_scenes/render.py --preview 9   # fast 480p check
    .venv/bin/python tools/manim_scenes/preview.py 9            # contact sheet -> media/previews/part09.png
    .venv/bin/python tools/manim_scenes/render.py 9             # final -> videos/part09.mp4
    .venv/bin/python tools/manim_scenes/render.py all
    .venv/bin/python tools/notebook_builder/insert_videos.py rl_from_bandits_to_grpo.ipynb

`style.py` holds the palette and helpers. Each `partNN_*.py` starts with its maths helpers,
which `tests/test_scene_maths.py` and `tests/test_scene_maths_partNN.py` check against the
notebook's formulas.
