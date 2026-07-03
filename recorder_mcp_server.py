from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from server.agent_tools import apply_review_to_project, model_status, prepare_review_package, release_model, transcribe_audio_file
from server.core import generate_insights, now_iso, project_to_html
from server.glossary import confirm_glossary_terms, list_glossary_terms, reject_glossary_terms, suggest_glossary_terms_for_project, upsert_glossary_term

WORKSPACE = Path(__file__).resolve().parent.parent
os.environ.setdefault("HOME", str(WORKSPACE / ".cache" / "home"))
os.environ.setdefault("MODELSCOPE_CACHE", str(WORKSPACE / ".cache" / "modelscope"))
os.environ.setdefault("MODELSCOPE_CREDENTIALS_PATH", str(WORKSPACE / ".cache" / "modelscope" / "credentials"))
os.environ.setdefault("FUNASR_MODEL", "SenseVoiceSmall")
os.environ.setdefault("FUNASR_KEEPALIVE_SECONDS", "600")

DEFAULT_JOB_DIR = WORKSPACE / "outputs" / "mcp-jobs"
JOB_DIR = Path(os.environ.get("RECORDER_AI_MCP_JOB_DIR", DEFAULT_JOB_DIR)).expanduser().resolve()
MAX_WORKERS = max(1, int(os.environ.get("RECORDER_AI_MCP_WORKERS", "1")))

mcp = FastMCP("recorder-ai-studio")
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="recorder-mcp-job")
_jobs_lock = threading.RLock()


def _job_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def _log_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.log"


def _write_job(job: dict[str, Any]) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _job_path(job["jobId"]).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_job_path(job["jobId"]))


def _read_job(job_id: str) -> dict[str, Any]:
    path = _job_path(job_id)
    if not path.exists():
        raise FileNotFoundError(f"Transcription job not found: {job_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def _update_job(job_id: str, **updates: Any) -> dict[str, Any]:
    with _jobs_lock:
        job = _read_job(job_id)
        job.update(updates)
        job["updatedAt"] = now_iso()
        _write_job(job)
        return job


def _run_transcription_job(job_id: str) -> None:
    job = _update_job(job_id, status="running", startedAt=now_iso())
    log_path = _log_path(job_id)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"[{now_iso()}] job started\n")
            log_file.flush()
            with redirect_stdout(log_file), redirect_stderr(log_file):
                result = transcribe_audio_file(
                    audio_path=job["request"]["audioPath"],
                    output_dir=job["request"].get("outputDir"),
                    title=job["request"].get("title"),
                    scene=job["request"].get("scene") or "meeting",
                    glossary=job["request"].get("glossary") or "",
                    write_files=True,
                )
        public_result = {key: value for key, value in result.items() if key != "project"}
        _update_job(
            job_id,
            status="completed",
            completedAt=now_iso(),
            result=public_result,
            outputs=public_result.get("outputs", {}),
            segmentCount=public_result.get("segmentCount"),
            source=public_result.get("source"),
            noMockFallback=public_result.get("noMockFallback"),
            logPath=str(log_path),
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            completedAt=now_iso(),
            error=f"{type(exc).__name__}: {exc}",
            logPath=str(log_path),
        )


def _submit_transcription_job(
    audio_path: str,
    title: Optional[str] = None,
    scene: str = "meeting",
    glossary: str = "",
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    audio = Path(audio_path).expanduser().resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    job_id = f"rec-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    target_output_dir = Path(output_dir).expanduser().resolve() if output_dir else JOB_DIR / job_id
    job = {
        "jobId": job_id,
        "status": "queued",
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "request": {
            "audioPath": str(audio),
            "title": title or audio.stem,
            "scene": scene,
            "glossary": glossary,
            "outputDir": str(target_output_dir),
        },
        "outputs": {},
        "logPath": str(_log_path(job_id)),
    }
    with _jobs_lock:
        _write_job(job)
    _executor.submit(_run_transcription_job, job_id)
    return {
        "ok": True,
        "submitted": True,
        "jobId": job_id,
        "status": "queued",
        "message": "Transcription job submitted. Use recorder_job_status and recorder_job_result to check progress and outputs.",
        "jobPath": str(_job_path(job_id)),
        "logPath": str(_log_path(job_id)),
        "outputDir": str(target_output_dir),
    }


@mcp.tool()
def recorder_asr_status() -> dict:
    """Return local FunASR / SenseVoiceSmall model and runtime status."""
    return model_status()


@mcp.tool()
def recorder_release_model() -> dict:
    """Release the loaded local FunASR model immediately."""
    return release_model()


@mcp.tool()
def recorder_transcribe(
    audio_path: str,
    title: Optional[str] = None,
    scene: str = "meeting",
    glossary: str = "",
    output_dir: Optional[str] = None,
) -> dict:
    """Submit a non-blocking real local FunASR transcription job.

    This tool returns immediately with a job id so MCP clients such as WorkBuddy
    do not have to keep one long JSON-RPC call open during model loading and long
    recording transcription.
    """
    return _submit_transcription_job(audio_path, title=title, scene=scene, glossary=glossary, output_dir=output_dir)


@mcp.tool()
def recorder_job_status(job_id: str) -> dict:
    """Return the current status of a recorder transcription job."""
    job = _read_job(job_id)
    if job.get("status") == "completed":
        job = _ensure_html_report(job)
    return {key: value for key, value in job.items() if key != "result"}


def _default_next_steps() -> list[str]:
    return [
        "Open outputs.htmlReport to review the visual meeting report.",
        "Use the 核心议题、重点摘要、详情整理、线索与追问 sections to校准转写后的纪要。",
        "Convert action guidance into owner/date/checkable todos before sharing.",
    ]


def _infer_html_report_path(outputs: dict[str, Any]) -> Optional[Path]:
    report_path = outputs.get("report")
    if report_path:
        return Path(report_path).expanduser().resolve().with_suffix(".html")
    markdown_path = outputs.get("markdown")
    if markdown_path:
        path = Path(markdown_path).expanduser().resolve()
        name = path.name.replace("-transcript.md", "-report.html")
        return path.with_name(name if name != path.name else f"{path.stem}-report.html")
    project_path = outputs.get("projectJson")
    if project_path:
        path = Path(project_path).expanduser().resolve()
        name = path.name.replace("-project.json", "-report.html")
        return path.with_name(name if name != path.name else f"{path.stem}-report.html")
    return None


def _ensure_html_report(job: dict[str, Any]) -> dict[str, Any]:
    """Backfill htmlReport for old completed jobs and keep job/report JSON in sync."""
    outputs = dict(job.get("outputs") or {})
    result = dict(job.get("result") or {})
    result_outputs = dict(result.get("outputs") or {})
    merged_outputs = {**outputs, **result_outputs}
    existing_html = merged_outputs.get("htmlReport")
    if existing_html and Path(existing_html).expanduser().exists():
        return job

    project_path_value = merged_outputs.get("projectJson")
    if not project_path_value:
        return job
    project_path = Path(project_path_value).expanduser().resolve()
    if not project_path.exists():
        return job

    html_path = _infer_html_report_path(merged_outputs)
    if not html_path:
        return job
    html_path.parent.mkdir(parents=True, exist_ok=True)

    project = json.loads(project_path.read_text(encoding="utf-8"))
    generate_insights(project)
    project_path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(project_to_html(project), encoding="utf-8")

    merged_outputs["htmlReport"] = str(html_path)
    result["outputs"] = merged_outputs
    result.setdefault("nextSteps", _default_next_steps())
    job["outputs"] = merged_outputs
    job["result"] = result
    job["updatedAt"] = now_iso()
    _write_job(job)

    report_path_value = merged_outputs.get("report")
    if report_path_value:
        report_path = Path(report_path_value).expanduser().resolve()
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                report = {}
            report["outputs"] = merged_outputs
            report.setdefault("nextSteps", _default_next_steps())
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return job


@mcp.tool()
def recorder_job_result(job_id: str) -> dict:
    """Return transcription job result after it completes."""
    job = _read_job(job_id)
    if job.get("status") != "completed":
        return {
            "ok": False,
            "jobId": job_id,
            "status": job.get("status"),
            "message": "Job is not completed yet. Call recorder_job_status again later.",
            "error": job.get("error"),
            "logPath": job.get("logPath"),
        }
    job = _ensure_html_report(job)
    return {"ok": True, "jobId": job_id, "status": "completed", "result": job.get("result"), "outputs": job.get("outputs", {})}


@mcp.tool()
def recorder_prepare_review(
    job_id: Optional[str] = None,
    project_json: Optional[str] = None,
    output_dir: Optional[str] = None,
    max_segments: int = 120,
    glossary_categories: Optional[str] = None,
) -> dict:
    """Prepare a review package for the WorkBuddy agent to calibrate transcript text and insights."""
    if not project_json:
        if not job_id:
            raise ValueError("Either job_id or project_json is required.")
        job = _read_job(job_id)
        if job.get("status") != "completed":
            return {"ok": False, "jobId": job_id, "status": job.get("status"), "message": "Job is not completed yet."}
        job = _ensure_html_report(job)
        project_json = (job.get("outputs") or {}).get("projectJson")
    if not project_json:
        raise ValueError("Project JSON path is not available.")
    result = prepare_review_package(project_json, output_dir=output_dir, max_segments=max_segments, glossary_categories=glossary_categories)
    if job_id:
        job = _read_job(job_id)
        outputs = dict(job.get("outputs") or {})
        outputs["reviewPackage"] = result["reviewPackagePath"]
        outputs["reviewPrompt"] = result["reviewPromptPath"]
        job["outputs"] = outputs
        job["updatedAt"] = now_iso()
        _write_job(job)
    return result


@mcp.tool()
def recorder_apply_review(
    review_json: str,
    job_id: Optional[str] = None,
    project_json: Optional[str] = None,
    output_dir: Optional[str] = None,
    suffix: str = "calibrated",
) -> dict:
    """Apply WorkBuddy agent calibrated JSON back to the project and write calibrated reports."""
    if not project_json:
        if not job_id:
            raise ValueError("Either job_id or project_json is required.")
        job = _read_job(job_id)
        if job.get("status") != "completed":
            return {"ok": False, "jobId": job_id, "status": job.get("status"), "message": "Job is not completed yet."}
        job = _ensure_html_report(job)
        project_json = (job.get("outputs") or {}).get("projectJson")
    if not project_json:
        raise ValueError("Project JSON path is not available.")
    result = apply_review_to_project(project_json, review_json, output_dir=output_dir, suffix=suffix)
    if job_id:
        job = _read_job(job_id)
        outputs = dict(job.get("outputs") or {})
        outputs.update(result.get("outputs") or {})
        job["outputs"] = outputs
        job["review"] = {key: value for key, value in result.items() if key != "outputs"}
        job["updatedAt"] = now_iso()
        _write_job(job)
    return result


@mcp.tool()
def recorder_glossary_list(categories: Optional[str] = None, keyword: str = "", limit: int = 200) -> dict:
    """List glossary terms by category and optional keyword."""
    return list_glossary_terms(categories=categories, keyword=keyword, limit=limit)


@mcp.tool()
def recorder_glossary_suggest(
    job_id: Optional[str] = None,
    project_json: Optional[str] = None,
    categories: Optional[str] = None,
    limit: int = 40,
) -> dict:
    """Suggest glossary candidates from a completed job or project JSON."""
    if not project_json:
        if not job_id:
            raise ValueError("Either job_id or project_json is required.")
        job = _read_job(job_id)
        if job.get("status") != "completed":
            return {"ok": False, "jobId": job_id, "status": job.get("status"), "message": "Job is not completed yet."}
        job = _ensure_html_report(job)
        project_json = (job.get("outputs") or {}).get("projectJson")
    if not project_json:
        raise ValueError("Project JSON path is not available.")
    project_path = Path(project_json).expanduser().resolve()
    project = json.loads(project_path.read_text(encoding="utf-8"))
    result = suggest_glossary_terms_for_project(project, categories=categories, limit=limit)
    result["projectJson"] = str(project_path)
    if job_id:
        result["jobId"] = job_id
    return result


@mcp.tool()
def recorder_glossary_confirm(terms_json: str, default_category: str = "meeting") -> dict:
    """Confirm and write glossary terms. terms_json may be a JSON array or an object with a terms field."""
    data = json.loads(terms_json)
    if isinstance(data, dict):
        terms = data.get("terms") or data.get("confirmedTerms") or []
    elif isinstance(data, list):
        terms = data
    else:
        raise ValueError("terms_json must be a JSON array or object.")
    return confirm_glossary_terms(terms, default_category=default_category)


@mcp.tool()
def recorder_glossary_reject(items_json: str, reason: str = "") -> dict:
    """Reject glossary candidates so WorkBuddy can avoid repeatedly recommending them."""
    data = json.loads(items_json)
    if isinstance(data, dict):
        items = data.get("items") or data.get("terms") or data.get("rejectedTerms") or []
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("items_json must be a JSON array or object.")
    return reject_glossary_terms(items, reason=reason)


@mcp.tool()
def recorder_glossary_update(
    term: str,
    category: str = "meeting",
    aliases: str = "",
    notes: str = "",
    priority: int = 5,
) -> dict:
    """Create or update one glossary term directly."""
    alias_values = [item.strip() for item in aliases.replace("，", ",").split(",") if item.strip()]
    return upsert_glossary_term({"term": term, "category": category, "aliases": alias_values, "notes": notes, "priority": priority, "source": "manual_update"})


if __name__ == "__main__":
    mcp.run()
