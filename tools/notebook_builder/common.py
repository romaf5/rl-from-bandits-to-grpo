"""Cell constructors shared by all notebook sections."""
import base64, hashlib, textwrap
from nbformat.v4 import new_markdown_cell, new_code_cell


def _clean(src):
    return textwrap.dedent(src).strip("\n")


def md(src):
    return new_markdown_cell(_clean(src))


def code(src):
    return new_code_cell(_clean(src))


def _b64(s):
    return base64.b64encode(_clean(s).encode()).decode()


def quiz_cells(qid, question, options, correct, explanation):
    """Two cells: one registers+prints the question, one is the learner's answer slot.
    Options are shuffled deterministically per question so the correct letter varies."""
    import random
    perm = list(range(len(options))); random.Random(f"shuffle-{qid}").shuffle(perm)
    options = [options[i] for i in perm]
    correct = "ABCD"[perm.index("ABCD".index(correct.upper()))]
    h = hashlib.sha256(f"{qid}:{correct.upper()}".encode()).hexdigest()
    src = (
        f"register_quiz({qid!r},\n"
        f"    {_clean(question)!r},\n"
        f"    {options!r},\n"
        f"    {h!r},\n"
        f"    {_b64(explanation)!r})\n"
        f"quiz({qid!r})"
    )
    ans = f"answer({qid!r}, '?')   # <- replace '?' with A, B, C or D and run"
    return [code(src), code(ans)]


def exercise_cells(eid, task, starter, tests, solution, fn_name):
    """Markdown task, starter cell, and a check cell (tests visible, solution encoded)."""
    check_src = (
        f"register_exercise({eid!r}, {_b64(solution)!r})\n\n"
        f"{_clean(tests)}\n\n"
        f"check({eid!r}, {fn_name}, _tests)\n"
        f"# solution({eid!r})   # <- uncomment to reveal the reference solution"
    )
    starter_cell = code(starter)
    starter_cell.metadata["exercise_solution"] = _b64(solution)
    return [md(task), starter_cell, code(check_src)]
