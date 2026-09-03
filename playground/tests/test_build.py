"""Runs only in the playground env (uv run --project playground pytest playground/tests)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mjswan")
REPO = Path(__file__).resolve().parents[2]


def test_minimal_build(tmp_path):
    out = tmp_path / "dist"
    subprocess.run(
        [sys.executable, str(REPO / "playground" / "build.py"), "--scenes", "walk,pose",
         "--no-community", "--out", str(out)],
        check=True, env={**__import__("os").environ, "PYTHONUTF8": "1"},
    )
    cfg = json.loads((out / "assets" / "config.json").read_text(encoding="utf-8"))
    scenes = {s["name"]: [p["name"] for p in s["policies"]] for s in cfg["projects"][0]["scenes"]}
    assert "Walk" in scenes and "Pose" in scenes
    assert any("PPO" in p for p in scenes["Walk"])
    assert (out / "main" / "assets" / "walk" / "obs" / "policy.onnx").exists()
    assert (out / "main" / "assets" / "walk" / "scene.mjz").exists()
