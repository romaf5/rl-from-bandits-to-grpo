"""Contact sheet of 12 frames for one Part: preview.py N [--hq] -> media/previews/partNN.png"""
import pathlib, subprocess, sys, tempfile
from render import MEDIA, part_files


def duration(mp4):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4)],
                         capture_output=True, text=True, check=True).stdout
    return float(out)


def sheet(n, hq=False):
    f = part_files().get(n)
    if f is None:
        sys.exit(f"no scene file for part {n}")
    mp4 = MEDIA / "videos" / f.stem / ("1080p30" if hq else "480p15") / f"Part{n:02d}.mp4"
    if not mp4.exists():
        sys.exit(f"{mp4} not found: render it first with render.py {'' if hq else '--preview '}{n}")
    d = duration(mp4)
    tmp = pathlib.Path(tempfile.mkdtemp())
    for i in range(12):
        t = d * (i + 0.5) / 12
        subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", str(mp4), "-frames:v", "1",
                        str(tmp / f"{i}.png")], check=True)
    ins = sum([["-i", str(tmp / f"{i}.png")] for i in range(12)], [])
    rows = ";".join(f"[{3*r}][{3*r+1}][{3*r+2}]hstack=3[r{r}]" for r in range(4))
    out = MEDIA / "previews" / f"part{n:02d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *ins, "-filter_complex",
                    rows + ";[r0][r1][r2][r3]vstack=4", str(out)], check=True)
    flag = "" if 60 <= d <= 120 else "  <-- OUTSIDE 60-120 s TARGET"
    print(f"part {n}: {d:.1f} s{flag}\n{out}")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--hq"]
    if len(args) != 1 or not args[0].isdigit():
        sys.exit("usage: preview.py N [--hq]")
    sheet(int(args[0]), hq="--hq" in sys.argv)
