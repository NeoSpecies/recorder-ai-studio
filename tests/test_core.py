from pathlib import Path

from server.core import ProjectStore, default_project, generate_insights, normalize_funasr_result, project_to_markdown


def test_default_project_has_no_mock_content():
    project = default_project("测试会议", "meeting", ["FunASR", "Plaud"])
    assert project["title"] == "测试会议"
    assert project["segments"] == []
    assert project["todos"] == []
    assert project["tags"] == []
    assert project["insights"] is None


def test_empty_insights_do_not_generate_fake_summary():
    project = default_project("测试会议", "meeting", ["FunASR"])
    insights = generate_insights(project)
    assert insights["summary"] == []
    assert insights["mindmap"] == []
    assert project["tags"] == []


def test_normalize_funasr_text_result():
    segments = normalize_funasr_result({"text": "第一句。第二句！"})
    assert len(segments) == 2
    assert segments[0]["textCorrected"] == "第一句"


def test_normalize_sensevoice_tokens_are_cleaned():
    segments = normalize_funasr_result({"text": "<|zh|><|NEUTRAL|><|Speech|><|woitn|>真实识别内容。"})
    assert len(segments) == 1
    assert segments[0]["textRaw"].startswith("<|zh|>")
    assert segments[0]["textCorrected"] == "真实识别内容"


def test_project_store_upload_and_markdown(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects.json", tmp_path / "uploads")
    project = store.create_project("上传测试", "meeting", ["FunASR"])
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"fake audio")
    updated = store.attach_upload(project["id"], audio, "sample.wav", "audio/wav")
    assert updated["audio"]["size"] == 10
    segments = normalize_funasr_result({"text": "真实识别第一句。真实识别第二句。"})
    updated = store.set_segments(project["id"], segments)
    generate_insights(updated)
    store.save_project(updated)
    markdown = project_to_markdown(updated)
    assert "# 上传测试" in markdown
    assert "## 转写" in markdown
