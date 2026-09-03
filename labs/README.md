# Labs

One directory per session, each with three notebooks:

- `lecture.ipynb` — the teaching notebook; doubles as the slide deck (present it top-to-bottom).
- `lab.ipynb` — the student worksheet: tasks T1–T5 with `# TODO` cells and self-checking `assert` cells.
- `solution.ipynb` — completed reference for instructors. Don't hand it out before the session.

## Colab-first

Every notebook opens in Google Colab from its badge (first markdown cell) and starts with
the same **setup cell** (`microduck_sim.colab.SETUP_CELL`): on Colab it clones the repo,
installs the package and sets `MUJOCO_GL=egl`; on a laptop checkout it is a no-op. Rules:

- the setup cell must not import `microduck_sim` *before* installing it (it is a
  self-contained snippet, no `!` magics, so nbconvert/pytest execute it too);
- `MUJOCO_GL` is set before the first `import mujoco` — never import mujoco above it;
- interactive 3D comes from the browser playground (`web.show_playground(scene, policy)`),
  which needs nothing installed. The native viewer scripts are laptop extras.

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
