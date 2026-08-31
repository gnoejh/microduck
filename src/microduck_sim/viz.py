"""Offscreen rendering helpers for the lab notebooks.

Everything here works headless (no window), so notebooks render images and
GIFs inline on any laptop.
"""

from __future__ import annotations

import io

import imageio.v3 as iio
import mujoco
import numpy as np

DEFAULT_SIZE = (480, 360)  # (width, height)


def render_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: str | int = -1,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> np.ndarray:
    """Render the current state to an RGB array (default: free camera)."""
    width, height = size
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        return renderer.render()
    finally:
        renderer.close()


def record(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl_fn,
    duration: float,
    fps: int = 25,
    camera: str | int = -1,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Run a control loop (see control.run_control_loop) while capturing video.

    Returns (frames, log). fps must divide the 50 Hz control rate evenly.
    """
    from microduck_sim.control import CONTROL_HZ, run_control_loop

    every = max(1, round(CONTROL_HZ / fps))
    width, height = size
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames: list[np.ndarray] = []
    tick = 0

    def on_tick(m, d):
        nonlocal tick
        if tick % every == 0:
            renderer.update_scene(d, camera=camera)
            frames.append(renderer.render())
        tick += 1

    try:
        log = run_control_loop(model, data, ctrl_fn, duration, on_tick=on_tick)
    finally:
        renderer.close()
    return frames, log


def record_policy(
    runner,
    duration: float,
    fps: int = 25,
    camera: str | int = -1,
    size: tuple[int, int] = DEFAULT_SIZE,
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Run a `PolicyRunner` (Lab 5) while capturing video. Returns (frames, log)."""
    every = max(1, round(50 / fps))
    width, height = size
    renderer = mujoco.Renderer(runner.model, height=height, width=width)
    frames: list[np.ndarray] = []
    rows = []
    try:
        for tick in range(int(round(duration * 50))):
            rows.append(runner.step())
            if tick % every == 0:
                renderer.update_scene(runner.data, camera=camera)
                frames.append(renderer.render())
    finally:
        renderer.close()
    log = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    return frames, log


def show(image: np.ndarray, title: str | None = None):
    """Display one RGB frame in a notebook (matplotlib, axes hidden)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(image.shape[1] / 96, image.shape[0] / 96))
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title)
    plt.show()


def show_grid(images: list[np.ndarray], titles: list[str] | None = None, cols: int = 4):
    """Display several frames side by side."""
    import matplotlib.pyplot as plt

    n = len(images)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows))
    for ax, img, title in zip(
        np.atleast_1d(axes).ravel(), images, titles or [""] * n
    ):
        ax.imshow(img)
        ax.set_title(title)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def show_video(frames: list[np.ndarray], fps: int = 25):
    """Display frames as an inline GIF (works in Jupyter without any plugins)."""
    from IPython.display import Image, display

    buffer = io.BytesIO()
    iio.imwrite(buffer, frames, extension=".gif", duration=1000 / fps, loop=0)
    display(Image(data=buffer.getvalue()))
