from pathlib import Path

import pytest

import recorder_mcp_server
from server import agent_tools


def test_safe_slug_handles_mixed_title():
    assert agent_tools.safe_slug("Meeting Demo 01") == "meeting-demo-01"
    assert agent_tools.safe_slug("录音 项目") == "录音-项目"


def test_parse_glossary_accepts_string_and_list():
    assert agent_tools.parse_glossary("芯片, 工具链,,AI") == ["芯片", "工具链", "AI"]
    assert agent_tools.parse_glossary(["芯片", "", "AI"]) == ["芯片", "AI"]


def test_transcribe_audio_refuses_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.wav"
    with pytest.raises(FileNotFoundError):
        agent_tools.transcribe_audio_file(missing, write_files=False)


def test_transcribe_audio_refuses_unready_model(monkeypatch, tmp_path: Path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"not real audio")
    monkeypatch.setattr(agent_tools, "get_funasr_model_status", lambda: {"ready": False, "state": "missing"})
    with pytest.raises(RuntimeError, match="not ready"):
        agent_tools.transcribe_audio_file(audio, write_files=False)


def test_model_status_includes_runtime(monkeypatch):
    monkeypatch.setattr(agent_tools, "get_funasr_model_status", lambda model_name=None: {"ready": True, "state": "ready"})
    monkeypatch.setattr(agent_tools, "get_funasr_runtime_status", lambda: {"loaded": False, "keepaliveSeconds": 600})
    status = agent_tools.model_status()
    assert status["funasr"]["ready"] is True
    assert status["runtime"]["keepaliveSeconds"] == 600



def test_transcribe_audio_builds_outputs(monkeypatch, tmp_path: Path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"not real audio")
    out_dir = tmp_path / "outputs"
    monkeypatch.setattr(agent_tools, "get_funasr_model_status", lambda: {"ready": True, "state": "ready"})
    monkeypatch.setattr(agent_tools, "get_funasr_runtime_status", lambda: {"loaded": True, "keepaliveSeconds": 600})
    monkeypatch.setattr(
        agent_tools,
        "local_funasr_transcript",
        lambda path: [
            {
                "id": "seg-1",
                "start": 0,
                "end": 3,
                "speaker": "A",
                "name": "未命名",
                "confidence": 90,
                "textRaw": "真实识别内容",
                "textCorrected": "真实识别内容",
                "tags": ["#会议"],
            }
        ],
    )
    result = agent_tools.transcribe_audio_file(audio, output_dir=out_dir, title="测试项目")
    assert result["ok"] is True
    assert result["source"] == "local_funasr"
    assert result["noMockFallback"] is True
    assert result["segmentCount"] == 1
    assert result["runtime"]["loaded"] is True
    assert Path(result["outputs"]["projectJson"]).exists()
    assert Path(result["outputs"]["markdown"]).exists()
    assert Path(result["outputs"]["report"]).exists()


def test_mcp_transcribe_submits_non_blocking_job(monkeypatch, tmp_path: Path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"not real audio")
    monkeypatch.setattr(recorder_mcp_server, "JOB_DIR", tmp_path / "jobs")

    submitted = []

    class FakeExecutor:
        def submit(self, fn, *args, **kwargs):
            submitted.append((fn, args, kwargs))
            return None

    monkeypatch.setattr(recorder_mcp_server, "_executor", FakeExecutor())
    result = recorder_mcp_server.recorder_transcribe(str(audio), title="测试项目")

    assert result["ok"] is True
    assert result["submitted"] is True
    assert result["jobId"]
    assert result["status"] == "queued"
    assert submitted and submitted[0][0] == recorder_mcp_server._run_transcription_job
    status = recorder_mcp_server.recorder_job_status(result["jobId"])
    assert status["status"] == "queued"


def test_mcp_job_result_waits_until_completed(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(recorder_mcp_server, "JOB_DIR", tmp_path / "jobs")
    job = {
        "jobId": "job-1",
        "status": "running",
        "createdAt": "now",
        "updatedAt": "now",
        "request": {"audioPath": "sample.wav"},
        "outputs": {},
        "logPath": str(tmp_path / "jobs/job-1.log"),
    }
    recorder_mcp_server._write_job(job)
    result = recorder_mcp_server.recorder_job_result("job-1")
    assert result["ok"] is False
    assert result["status"] == "running"
