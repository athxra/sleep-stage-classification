"""Smoke tests for the FastAPI backend used by the React dashboard."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import MODEL_PATH

pytestmark = pytest.mark.skipif(not os.path.isfile(MODEL_PATH), reason="no trained model")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.server import app
    return TestClient(app)


def test_summary(client):
    body = client.get("/api/summary").json()
    assert body["classes"] == ["Wake", "N1", "N2", "N3", "REM"]
    assert body["model"]["parameters"] > 0
    assert set(body["model"]["test_subjects"]).isdisjoint(body["model"]["train_subjects"])


def test_night_and_epoch(client):
    recs = client.get("/api/recordings").json()["recordings"]
    assert recs
    key = recs[0]["key"]
    night = client.get(f"/api/recordings/{key}/night").json()
    assert len(night["predicted"]) == night["n_epochs"] == len(night["probs"])
    start, stop = night["sleep_window"]
    assert 0 <= start < stop <= night["n_epochs"]

    epoch = client.get(f"/api/recordings/{key}/epoch/{start}").json()
    assert len(epoch["signal_uv"]) == 3000
    assert abs(sum(epoch["probs"]) - 1) < 1e-3
    assert abs(sum(epoch["bands"].values()) - 100) < 0.5


def test_unknown_recording_is_404(client):
    assert client.get("/api/recordings/SC9999/night").status_code == 404


def test_epoch_includes_power_spectrum(client):
    key = client.get("/api/recordings").json()["recordings"][0]["key"]
    psd = client.get(f"/api/recordings/{key}/epoch/0").json()["psd"]
    assert len(psd["freqs"]) == len(psd["db"]) > 0
    assert psd["freqs"][0] >= 0.5 and psd["freqs"][-1] <= 30


def test_examples_drilldown(client):
    res = client.get("/api/examples", params={"true": "N1", "pred": "REM"})
    if res.status_code == 404:
        pytest.skip("no test predictions yet")
    body = res.json()
    assert body["total"] >= len(body["examples"])
    assert client.get("/api/examples", params={"true": "N9", "pred": "REM"}).status_code == 400


def test_upload_rejects_bad_files(client):
    assert client.post("/api/upload", files={"psg": ("notes.txt", b"abc")}).status_code == 400
    assert client.post("/api/upload", files={"psg": ("fake.edf", b"not an edf")}).status_code == 400
