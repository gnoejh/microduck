"""Opt-in notebook execution tests (M5): run with  uv run pytest -m notebooks

Executes every lecture and solution notebook headless — the guarantee that a
fresh student environment can run the whole course. Worksheets (lab.ipynb)
are excluded: they contain intentional TODO placeholders.

Slow (RL labs train for real): expect ~20–30 min for the full set.
"""

import subprocess
import sys
from pathlib import Path

import pytest

LABS = sorted((Path(__file__).parent.parent / "labs").glob("lab*"))
NOTEBOOKS = [nb for lab in LABS for nb in (lab / "lecture.ipynb", lab / "solution.ipynb")]


@pytest.mark.notebooks
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_notebook_executes(notebook):
    assert notebook.exists(), f"{notebook} missing — run jupytext (see labs/README.md)"
    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--stdout", str(notebook)],
        capture_output=True, text=True, cwd=notebook.parent, timeout=1800,
    )
    assert result.returncode == 0, result.stderr[-3000:]
