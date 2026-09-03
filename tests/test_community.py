"""Every Lab 8 submission under assets/community/ must validate."""

import pytest

from microduck_sim.evaluate import validate_submission
from microduck_sim.policy import COMMUNITY_DIR, OnnxPolicy

SUBMISSIONS = sorted(p for p in COMMUNITY_DIR.iterdir() if p.is_dir()) if COMMUNITY_DIR.exists() else []


@pytest.mark.parametrize("folder", SUBMISSIONS, ids=lambda p: p.name)
def test_submission_validates(folder):
    manifest = validate_submission(folder, quick=False)
    assert manifest["name"]


def test_community_policy_loads_by_handle():
    if not SUBMISSIONS:
        pytest.skip("no submissions yet")
    assert OnnxPolicy(f"community/{SUBMISSIONS[0].name}").metadata
