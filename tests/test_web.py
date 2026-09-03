"""Playground helpers that must work in the student env (no mjswan installed)."""

import pytest

from microduck_sim import colab, web


def test_playground_url_formats_params():
    url = web.playground_url("walk", "stand_ppo", panel=False)
    assert url.startswith(web.PLAYGROUND_URL)
    assert "scene=walk" in url and "policy=stand_ppo" in url and "panel=0" in url
    assert "ref=0" in web.playground_url("pose")
    assert web.playground_url("walk", base="http://localhost:8081").startswith("http://localhost:8081/?")


def test_show_playground_returns_iframe():
    frame = web.show_playground("walk", height=300)
    assert "iframe" in frame._repr_html_() and "scene=walk" in frame._repr_html_()


def test_scene_table_matches_build_script():
    assert {"walk", "pose", "backlash", "stand_pose"} <= set(web.SCENES)


def test_build_helpers_need_the_build_env():
    pytest.importorskip("mjswan", reason="student env: build helpers are exercised in playground/")


def test_colab_setup_is_noop_on_laptop(monkeypatch):
    monkeypatch.setattr(colab, "IN_COLAB", False)
    root = colab.setup()
    assert (root / "pyproject.toml").exists()
    assert colab.persist() is None
    assert "MUJOCO_GL" in colab.SETUP_CELL and "colab.setup" in colab.SETUP_CELL
