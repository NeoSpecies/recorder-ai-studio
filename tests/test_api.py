import importlib
import os
from pathlib import Path

from fastapi.testclient import TestClient


def make_client(tmp_path: Path):
    os.environ["RECORDER_AI_DATA"] = str(tmp_path / "data")
    os.environ["RECORDER_AI_UPLOADS"] = str(tmp_path / "uploads")
    import server.app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app)


def test_health_and_project_flow(tmp_path: Path, monkeypatch):
    client = make_client(tmp_path)
    import server.app as app_module
    monkeypatch.setattr(app_module, "local_funasr_transcript", lambda _path: (_ for _ in ()).throw(RuntimeError("FunASR unavailable in unit test")))
    assert client.get("/api/health").json()["ok"] is True
    assert "funasr" in client.get("/api/asr/status").json()

    created = client.post("/api/projects", json={"title": "API 测试", "scene": "meeting", "glossary": ["FunASR"]}).json()["project"]
    project_id = created["id"]

    upload = client.post(
        f"/api/projects/{project_id}/upload",
        files={"file": ("sample.wav", b"fake audio", "audio/wav")},
    )
    assert upload.status_code == 200
    assert upload.json()["project"]["audio"]["name"] == "sample.wav"

    transcribe = client.post(f"/api/projects/{project_id}/transcribe", json={})
    assert transcribe.status_code in (409, 502)
    assert "FunASR" in transcribe.text
    if transcribe.status_code == 409:
        assert "fallback" in transcribe.text

    insights = client.post(f"/api/projects/{project_id}/insights").json()["insights"]
    assert insights["summary"] == []

    markdown = client.get(f"/api/projects/{project_id}/export.md")
    assert markdown.status_code == 200
    assert "# API 测试" in markdown.text
