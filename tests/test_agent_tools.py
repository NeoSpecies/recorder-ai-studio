from pathlib import Path

import pytest

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


def test_transcribe_audio_builds_outputs(monkeypatch, tmp_path: Path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"not real audio")
    out_dir = tmp_path / "outputs"
    monkeypatch.setattr(agent_tools, "get_funasr_model_status", lambda: {"ready": True, "state": "ready"})
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
    assert Path(result["outputs"]["projectJson"]).exists()
    assert Path(result["outputs"]["markdown"]).exists()
    assert Path(result["outputs"]["report"]).exists()
