"""A release source label requires a verifiable public commit."""

import io
import json
from urllib.error import URLError

import pytest

from holoboxing.evaluation.scripts import build_image


def test_verified_release_records_the_public_commit(monkeypatch):
    revision = "a" * 40
    monkeypatch.setattr(
        build_image,
        "urlopen",
        lambda request, timeout: io.BytesIO(
            json.dumps({"sha": revision}).encode()
        ),
    )
    result = build_image.verify_public_revision(revision)
    assert result["verified"] is True
    assert result["revision"] == revision
    assert (
        result["repository"] == "https://github.com/HorizonRobotics/HoloMotion"
    )


def test_unavailable_commit_cannot_be_verified_for_release(monkeypatch):
    def unavailable(request, timeout):
        raise URLError("No commit found for SHA")

    monkeypatch.setattr(build_image, "urlopen", unavailable)
    with pytest.raises(ValueError, match="Do not publish"):
        build_image.verify_public_revision("b" * 40)


def test_wrong_commit_response_is_not_release_evidence(monkeypatch):
    monkeypatch.setattr(
        build_image,
        "urlopen",
        lambda request, timeout: io.BytesIO(
            json.dumps({"sha": "c" * 40}).encode()
        ),
    )
    with pytest.raises(ValueError, match="different commit"):
        build_image.verify_public_revision("a" * 40)
