"""Insert companion-video cells into an (executed) notebook without touching outputs.
Usage: insert_videos.py NOTEBOOK.ipynb     (videos are looked up in <notebook dir>/videos)

Each video sits right after its Part's intro text, before the first `## ` subsection:
a Part-header markdown cell containing a `## ` line is split in two at that line and the
video goes between the halves. Idempotent: misplaced video cells are moved, never duplicated,
and code cells (with their outputs) are never modified."""
import pathlib, re, sys
import nbformat
from nbformat.v4 import new_markdown_cell
from common import video

HEADER = re.compile(r"^---\s*\n#\s*Part\s+(\d+)\s*·", re.M)


def split_header(source):
    """(intro, rest) split at the first line starting with '## ', or (source, None)."""
    lines = source.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("## "):
            return "\n".join(lines[:i]).rstrip(), "\n".join(lines[i:])
    return source, None


def _part(c):
    m = HEADER.match(c.source) if c.cell_type == "markdown" else None
    return int(m.group(1)) if m else None


def insert(nb_path, videos_dir):
    nb = nbformat.read(nb_path, as_version=4)
    parts = {_part(c) for c in nb.cells} - {None}
    # Pull out existing video cells of Parts that have a header; they are re-placed below.
    videos, dropped, cells = {}, 0, []
    for i, c in enumerate(nb.cells):
        n = c.metadata.get("companion_video")
        if n in parts:
            if n in videos:
                dropped += 1
            else:
                prev = _part(nb.cells[i - 1]) if i else None
                videos[n] = (c, prev == n)          # (cell, already directly after its header)
            continue
        cells.append(c)
    out, added, moved, split = [], 0, 0, 0
    for c in cells:
        out.append(c)
        n = _part(c)
        if n is None:
            continue
        intro, rest = split_header(c.source)
        if rest is not None:
            c.source = intro; split += 1
        if n in videos:
            cell, in_place = videos[n]
            out.append(cell); moved += not in_place
        elif (pathlib.Path(videos_dir) / f"part{n:02d}.mp4").exists():
            out.append(video(n)); added += 1
        else:
            print(f"part {n}: no video, skipped")
        if rest is not None:
            out.append(new_markdown_cell(rest))
    changes = added + moved + split + dropped
    if changes:
        nb.cells = out
        nbformat.write(nb, nb_path)
    print(f"{nb_path}: inserted {added}, moved {moved}, split {split} header cells, "
          f"removed {dropped} duplicate video cells")
    return changes


if __name__ == "__main__":
    p = pathlib.Path(sys.argv[1])
    insert(str(p), str(p.parent / "videos"))
