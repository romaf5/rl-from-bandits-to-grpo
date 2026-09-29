import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output
from common import video
import insert_videos


def _nb(tmp_path, parts=(1, 2)):
    nb = new_notebook()
    for p in parts:
        nb.cells.append(new_markdown_cell(f"---\n# Part {p} · Title {p}\n\ntext"))
        c = new_code_cell("print(1)"); c.outputs = [new_output("stream", name="stdout", text="1\n")]
        nb.cells.append(c)
    path = tmp_path / "nb.ipynb"; nbformat.write(nb, path)
    return path


def _videos(tmp_path, parts):
    d = tmp_path / "videos"; d.mkdir()
    for p in parts:
        (d / f"part{p:02d}.mp4").write_bytes(b"x")
    return d


def test_video_cell_points_at_relative_path_and_is_tagged():
    c = video(3)
    assert 'src="videos/part03.mp4"' in c.source and c.metadata["companion_video"] == 3


def test_inserts_after_part_header_and_keeps_outputs(tmp_path):
    path = _nb(tmp_path); vids = _videos(tmp_path, [1, 2])
    assert insert_videos.insert(str(path), str(vids)) == 2
    nb = nbformat.read(path, as_version=4)
    assert nb.cells[1].metadata["companion_video"] == 1
    assert nb.cells[4].metadata["companion_video"] == 2
    assert sum(len(c.get("outputs", [])) for c in nb.cells) == 2


def test_idempotent(tmp_path):
    path = _nb(tmp_path); vids = _videos(tmp_path, [1, 2])
    insert_videos.insert(str(path), str(vids))
    assert insert_videos.insert(str(path), str(vids)) == 0
    assert len(nbformat.read(path, as_version=4).cells) == 6


def test_skips_parts_without_video(tmp_path, capsys):
    path = _nb(tmp_path); vids = _videos(tmp_path, [2])
    assert insert_videos.insert(str(path), str(vids)) == 1
    assert "part 1: no video" in capsys.readouterr().out


def _nb_with_subsections(tmp_path, video_after_header=False):
    nb = new_notebook()
    nb.cells.append(new_markdown_cell("---\n# Part 1 · Title 1\n\nintro text\n\n## Sub A\n\nbody\n\n## Sub B\n\nmore"))
    if video_after_header:
        nb.cells.append(video(1))
    c = new_code_cell("print(1)"); c.outputs = [new_output("stream", name="stdout", text="1\n")]
    nb.cells.append(c)
    path = tmp_path / "nb.ipynb"; nbformat.write(nb, path)
    return path


def _check_split(nb):
    assert [c.cell_type for c in nb.cells] == ["markdown", "markdown", "markdown", "code"]
    assert nb.cells[0].source == "---\n# Part 1 · Title 1\n\nintro text"
    assert nb.cells[1].metadata["companion_video"] == 1
    assert nb.cells[2].source == "## Sub A\n\nbody\n\n## Sub B\n\nmore"
    assert nb.cells[3].outputs[0]["text"] == "1\n"
    ids = [c.id for c in nb.cells]
    assert len(set(ids)) == len(ids)
    nbformat.validate(nb)


def test_header_with_subsection_is_split_around_video(tmp_path):
    path = _nb_with_subsections(tmp_path); vids = _videos(tmp_path, [1])
    assert insert_videos.insert(str(path), str(vids)) > 0
    _check_split(nbformat.read(path, as_version=4))


def test_misplaced_video_after_unsplit_header_is_moved_not_duplicated(tmp_path):
    path = _nb_with_subsections(tmp_path, video_after_header=True); vids = _videos(tmp_path, [1])
    assert insert_videos.insert(str(path), str(vids)) > 0
    nb = nbformat.read(path, as_version=4)
    assert sum(c.metadata.get("companion_video") == 1 for c in nb.cells) == 1
    _check_split(nb)


def test_second_run_on_split_notebook_changes_nothing(tmp_path):
    path = _nb_with_subsections(tmp_path, video_after_header=True); vids = _videos(tmp_path, [1])
    insert_videos.insert(str(path), str(vids))
    before = path.read_bytes()
    assert insert_videos.insert(str(path), str(vids)) == 0
    assert path.read_bytes() == before
    _check_split(nbformat.read(path, as_version=4))
