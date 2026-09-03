"""Google Colab bootstrap for the course notebooks (Colab-first platform, M6).

Every notebook's second cell is the same self-contained snippet (see
labs/README.md); it installs the repo when running on Colab and then calls
`setup()` here. On a laptop `setup()` is a no-op apart from `MUJOCO_GL`.

Rules this module keeps:
- `MUJOCO_GL` must be set BEFORE the first ``import mujoco`` or offscreen
  rendering on Colab fails (no display; EGL works on CPU and GPU runtimes).
- Nothing heavy is imported at module level so the setup cell stays fast.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/gnoejh/microduck.git"
COLAB_ROOT = Path("/content/microduck")

IN_COLAB = "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def in_colab() -> bool:
    return IN_COLAB


def set_gl(backend: str = "egl") -> None:
    """Select MuJoCo's offscreen rendering backend (must run before importing mujoco)."""
    if "mujoco" in sys.modules and os.environ.get("MUJOCO_GL") != backend:
        raise RuntimeError(
            "mujoco was imported before MUJOCO_GL was set; restart the runtime and run "
            "the setup cell first."
        )
    os.environ.setdefault("MUJOCO_GL", backend)


def setup(lab_dir: str | None = None, *, repo: str = REPO_URL, gl: str = "egl") -> Path:
    """Prepare a Colab runtime (clone + install) and `chdir` into `lab_dir`.

    Returns the repo root. On a laptop it only sets MUJOCO_GL when the platform
    is headless Linux and returns the checkout the package was installed from.
    """
    if not IN_COLAB:
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            set_gl(gl)
        return Path(__file__).resolve().parents[2]

    set_gl(gl)
    if not (COLAB_ROOT / "pyproject.toml").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", repo, str(COLAB_ROOT)], check=True)
    try:
        import microduck_sim  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(COLAB_ROOT)], check=True)
    if lab_dir:
        os.chdir(COLAB_ROOT / lab_dir)
    print(f"Colab ready: Python {sys.version.split()[0]}, MUJOCO_GL={os.environ['MUJOCO_GL']}, cwd={os.getcwd()}")
    return COLAB_ROOT


def persist(src: str = "runs", dst: str = "microduck_runs") -> Path | None:
    """Copy `src` (relative to cwd) into Google Drive so a disconnect can't eat it."""
    if not IN_COLAB:
        print("not on Colab: nothing to persist")
        return None
    from google.colab import drive  # type: ignore[import]

    drive.mount("/content/drive", force_remount=False)
    target = Path("/content/drive/MyDrive") / dst
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-r", str(Path(src)), str(target)], check=True)
    print(f"copied {src} -> {target}")
    return target


#: The setup cell every notebook carries (kept here so tests can check the copies).
SETUP_CELL = '''\
# --- setup (Colab or laptop; safe to re-run) -----------------------------------
import os, subprocess, sys
if "google.colab" in sys.modules and not os.path.exists("/content/microduck/pyproject.toml"):
    subprocess.run(["git", "clone", "-q", "--depth", "1",
                    "https://github.com/gnoejh/microduck.git", "/content/microduck"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "/content/microduck"], check=True)
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless Linux (Colab, CI); Win/mac keep their default
from microduck_sim import colab; colab.setup("{lab_dir}")
'''
