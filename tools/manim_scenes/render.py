"""Render Part videos. Usage: render.py [--preview] 2 15 | all"""
import pathlib, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MEDIA, OUT = ROOT / "media", ROOT / "videos"
MANIM = ROOT / ".venv" / "bin" / "manim"


def part_files():
    return {int(p.name[4:6]): p for p in sorted(HERE.glob("part[0-9][0-9]_*.py"))}


def render(n, preview=False):
    f = part_files().get(n)
    if f is None:
        sys.exit(f"no scene file for part {n}")
    cls = f"Part{n:02d}"
    q = ["-ql"] if preview else ["-qh", "--frame_rate", "30"]
    subprocess.run([str(MANIM), *q, "--media_dir", str(MEDIA), str(f), cls], check=True, cwd=HERE)
    sub = "480p15" if preview else "1080p30"
    mp4 = MEDIA / "videos" / f.stem / sub / f"{cls}.mp4"
    if not preview:
        OUT.mkdir(exist_ok=True)
        shutil.copy(mp4, OUT / f"part{n:02d}.mp4")
        print("wrote", OUT / f"part{n:02d}.mp4")
    return mp4


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--preview"]
    if not args:
        sys.exit("usage: render.py [--preview] N [N ...] | all")
    if args != ["all"] and not all(a.isdigit() for a in args):
        sys.exit(f"render.py: expected Part numbers or 'all', got {' '.join(args)}")
    nums = sorted(part_files()) if args == ["all"] else [int(a) for a in args]
    for n in nums:
        render(n, preview="--preview" in sys.argv)
