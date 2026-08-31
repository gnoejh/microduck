# Labs

One directory per session, each with three notebooks:

- `lecture.ipynb` — the teaching notebook; doubles as the slide deck (present it top-to-bottom).
- `lab.ipynb` — the student worksheet: tasks T1–T5 with `# TODO` cells and self-checking `assert` cells.
- `solution.ipynb` — completed reference for instructors. Don't hand it out before the session.

## Authoring workflow (instructors)

The `.py` files (jupytext percent format) are the **source of truth** — edit those, then:

```bash
uv run jupytext --to notebook labs/labNN_*/name.py    # regenerate the .ipynb
uv run jupyter nbconvert --to notebook --execute --inplace labs/labNN_*/lecture.ipynb  # validate it runs
uv run jupyter nbconvert --clear-output --inplace labs/labNN_*/*.ipynb  # strip outputs before committing
```

Notebooks are committed **without outputs**; every image/GIF regenerates on execution
(all rendering is offscreen and CPU-only). `lab.ipynb` worksheets intentionally do NOT
execute end-to-end — they contain `...` placeholders for students.

Physical claims made in lecture text (stability times, speeds, whether something falls)
must be verified against the sim before publishing — this repo's rule is: demo it or
don't claim it.
