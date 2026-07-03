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
    assert Path(result["outputs"]["htmlReport"]).exists()
    assert Path(result["outputs"]["report"]).exists()
    assert "核心议题" in Path(result["outputs"]["markdown"]).read_text(encoding="utf-8")
    assert "录音智能分析报告" in Path(result["outputs"]["htmlReport"]).read_text(encoding="utf-8")
    assert result["nextSteps"]


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


def test_mcp_job_result_backfills_html_report_for_completed_job(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(recorder_mcp_server, "JOB_DIR", tmp_path / "jobs")
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    project_path = out_dir / "sample-project.json"
    markdown_path = out_dir / "sample-transcript.md"
    report_path = out_dir / "sample-report.json"
    project_path.write_text(
        """
        {
          "id": "agent-sample",
          "title": "样例会议",
          "scene": "meeting",
          "glossary": [],
          "segments": [
            {"id": "seg-1", "start": 0, "end": 5, "speaker": "A", "textCorrected": "核心目标是优化 HTML 报告展示并补齐线索追问。", "tags": ["#报告"]}
          ],
          "tags": ["#报告"],
          "todos": [],
          "insights": null,
          "transcriptionSource": "local_funasr"
        }
        """,
        encoding="utf-8",
    )
    markdown_path.write_text("# 样例会议", encoding="utf-8")
    report_path.write_text(
        '{"ok": true, "outputs": {"projectJson": "' + str(project_path) + '", "markdown": "' + str(markdown_path) + '", "report": "' + str(report_path) + '"}}',
        encoding="utf-8",
    )
    job = {
        "jobId": "job-2",
        "status": "completed",
        "createdAt": "now",
        "updatedAt": "now",
        "request": {"audioPath": "sample.wav"},
        "outputs": {"projectJson": str(project_path), "markdown": str(markdown_path), "report": str(report_path)},
        "result": {"ok": True, "outputs": {"projectJson": str(project_path), "markdown": str(markdown_path), "report": str(report_path)}},
        "logPath": str(tmp_path / "jobs/job-2.log"),
    }
    recorder_mcp_server._write_job(job)

    result = recorder_mcp_server.recorder_job_result("job-2")

    html_path = Path(result["outputs"]["htmlReport"])
    assert result["ok"] is True
    assert html_path.exists()
    assert "录音智能分析报告" in html_path.read_text(encoding="utf-8")
    assert "htmlReport" in result["result"]["outputs"]
    assert "htmlReport" in report_path.read_text(encoding="utf-8")


def test_prepare_and_apply_review_package(tmp_path: Path):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    project_path = out_dir / "review-sample-project.json"
    project_path.write_text(
        """
        {
          "id": "agent-review-sample",
          "title": "评审样例会议",
          "scene": "meeting",
          "glossary": ["FunASR"],
          "segments": [
            {"id": "seg-1", "start": 0, "end": 5, "speaker": "A", "confidence": 80, "textRaw": "这个呃我们要做转写校准", "textCorrected": "这个呃我们要做转写校准", "tags": []}
          ],
          "tags": [],
          "todos": [],
          "insights": null,
          "transcriptionSource": "local_funasr"
        }
        """,
        encoding="utf-8",
    )

    package = agent_tools.prepare_review_package(project_path, output_dir=out_dir)
    assert Path(package["reviewPackagePath"]).exists()
    assert Path(package["reviewPromptPath"]).exists()
    assert package["lowConfidenceCount"] == 1
    assert "recorder_apply_review" in Path(package["reviewPromptPath"]).read_text(encoding="utf-8")

    review_json = """
    {
      "reviewSummary": "清理口语词并补充结构化纪要。",
      "segments": [
        {"id": "seg-1", "textCorrected": "我们需要完成转写校准。", "correctionNotes": "删除口语词并调整语序。", "tags": ["#校准"]}
      ],
      "insights": {
        "brief": "会议讨论了转写校准。",
        "summary": ["需要完成转写校准。"],
        "topics": [{"title": "转写校准", "brief": "清理口语词并修正表达。", "details": ["我们需要完成转写校准。"], "weight": 1}],
        "keyPoints": ["清理口语词"],
        "details": ["修正错别字和语序。"],
        "decisions": ["推进校准流程。"],
        "risks": [],
        "clues": ["可接入 WorkBuddy 智能体处理。"],
        "questions": [],
        "actionGuidance": [{"step": "1", "title": "复核结果", "desc": "检查校准后纪要。"}],
        "keywords": ["校准"]
      },
      "todos": [{"title": "复核校准报告", "desc": "确认文本通顺", "owner": "未分配", "due": "", "done": false}]
    }
    """
    applied = agent_tools.apply_review_to_project(project_path, review_json, output_dir=out_dir)
    assert applied["ok"] is True
    assert applied["correctedSegmentCount"] == 1
    assert Path(applied["outputs"]["calibratedProjectJson"]).exists()
    assert Path(applied["outputs"]["calibratedMarkdown"]).exists()
    assert Path(applied["outputs"]["calibratedHtmlReport"]).exists()
    calibrated = Path(applied["outputs"]["calibratedProjectJson"]).read_text(encoding="utf-8")
    assert "我们需要完成转写校准" in calibrated
    assert "correctionNotes" in calibrated


def test_mcp_job_status_backfills_html_report_for_completed_job(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(recorder_mcp_server, "JOB_DIR", tmp_path / "jobs")
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    project_path = out_dir / "status-sample-project.json"
    markdown_path = out_dir / "status-sample-transcript.md"
    report_path = out_dir / "status-sample-report.json"
    project_path.write_text(
        """
        {
          "id": "agent-status-sample",
          "title": "状态查询样例会议",
          "scene": "meeting",
          "glossary": [],
          "segments": [
            {"id": "seg-1", "start": 0, "end": 5, "speaker": "A", "textCorrected": "需要确保状态查询也能返回 HTML 报告路径。", "tags": ["#报告"]}
          ],
          "tags": ["#报告"],
          "todos": [],
          "insights": null,
          "transcriptionSource": "local_funasr"
        }
        """,
        encoding="utf-8",
    )
    markdown_path.write_text("# 状态查询样例会议", encoding="utf-8")
    report_path.write_text(
        '{"ok": true, "outputs": {"projectJson": "' + str(project_path) + '", "markdown": "' + str(markdown_path) + '", "report": "' + str(report_path) + '"}}',
        encoding="utf-8",
    )
    job = {
        "jobId": "job-3",
        "status": "completed",
        "createdAt": "now",
        "updatedAt": "now",
        "request": {"audioPath": "sample.wav"},
        "outputs": {"projectJson": str(project_path), "markdown": str(markdown_path), "report": str(report_path)},
        "result": {"ok": True, "outputs": {"projectJson": str(project_path), "markdown": str(markdown_path), "report": str(report_path)}},
        "logPath": str(tmp_path / "jobs/job-3.log"),
    }
    recorder_mcp_server._write_job(job)

    status = recorder_mcp_server.recorder_job_status("job-3")

    assert "result" not in status
    assert Path(status["outputs"]["htmlReport"]).exists()
    assert "htmlReport" in report_path.read_text(encoding="utf-8")


def test_mcp_prepare_and_apply_review(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(recorder_mcp_server, "JOB_DIR", tmp_path / "jobs")
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    project_path = out_dir / "mcp-review-sample-project.json"
    markdown_path = out_dir / "mcp-review-sample-transcript.md"
    report_path = out_dir / "mcp-review-sample-report.json"
    project_path.write_text(
        """
        {
          "id": "agent-mcp-review-sample",
          "title": "MCP 评审样例会议",
          "scene": "meeting",
          "glossary": [],
          "segments": [
            {"id": "seg-1", "start": 0, "end": 5, "speaker": "A", "confidence": 90, "textRaw": "这个呃要校准文本", "textCorrected": "这个呃要校准文本", "tags": []}
          ],
          "tags": [],
          "todos": [],
          "insights": null,
          "transcriptionSource": "local_funasr"
        }
        """,
        encoding="utf-8",
    )
    markdown_path.write_text("# MCP 评审样例会议", encoding="utf-8")
    report_path.write_text(
        '{"ok": true, "outputs": {"projectJson": "' + str(project_path) + '", "markdown": "' + str(markdown_path) + '", "report": "' + str(report_path) + '"}}',
        encoding="utf-8",
    )
    job = {
        "jobId": "job-review",
        "status": "completed",
        "createdAt": "now",
        "updatedAt": "now",
        "request": {"audioPath": "sample.wav"},
        "outputs": {"projectJson": str(project_path), "markdown": str(markdown_path), "report": str(report_path)},
        "result": {"ok": True, "outputs": {"projectJson": str(project_path), "markdown": str(markdown_path), "report": str(report_path)}},
        "logPath": str(tmp_path / "jobs/job-review.log"),
    }
    recorder_mcp_server._write_job(job)

    package = recorder_mcp_server.recorder_prepare_review(job_id="job-review", output_dir=str(out_dir))
    assert package["ok"] is True
    assert Path(package["reviewPackagePath"]).exists()

    review_json = '{"segments": [{"id": "seg-1", "textCorrected": "需要校准文本。", "correctionNotes": "删除口语词。"}], "insights": {"brief": "校准文本。", "summary": ["需要校准文本。"], "topics": [{"title": "校准", "brief": "校准文本", "details": ["需要校准文本。"], "weight": 1}], "keyPoints": ["校准文本"], "details": ["删除口语词。"], "decisions": [], "risks": [], "clues": [], "questions": [], "actionGuidance": [{"step": "1", "title": "复核", "desc": "检查文本。"}], "keywords": ["校准"]}}'
    applied = recorder_mcp_server.recorder_apply_review(review_json=review_json, job_id="job-review", output_dir=str(out_dir))
    assert applied["ok"] is True
    assert Path(applied["outputs"]["calibratedHtmlReport"]).exists()
    status = recorder_mcp_server.recorder_job_status("job-review")
    assert "reviewPackage" in status["outputs"]
    assert "calibratedHtmlReport" in status["outputs"]
