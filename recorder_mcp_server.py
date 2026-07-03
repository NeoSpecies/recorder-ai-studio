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

from server.agent_tools import apply_review_to_project, model_status, prepare_review_package, release_model, safe_slug, transcribe_audio_file
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
_scheduled_job_ids: set[str] = set()


def _job_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def _log_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.log"


def _artifact_dir(job_id: str) -> Path:
    return JOB_DIR / job_id / "artifacts"


def _iter_job_paths() -> list[Path]:
    if not JOB_DIR.exists():
        return []
    return sorted(path for path in JOB_DIR.glob("*.json") if path.is_file())


def _submit_job_worker(job_id: str) -> None:
    with _jobs_lock:
        if job_id in _scheduled_job_ids:
            return
        _scheduled_job_ids.add(job_id)
    _executor.submit(_run_transcription_job, job_id)


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


def _public_result_from_outputs(outputs: dict[str, Any]) -> Optional[dict[str, Any]]:
    report_paths: list[Path] = []
    report_path_value = outputs.get("report")
    if report_path_value:
        report_paths.append(Path(report_path_value).expanduser().resolve())
    output_dir_value = outputs.get("outputDir")
    if output_dir_value:
        output_dir = Path(output_dir_value).expanduser().resolve()
        if output_dir.exists():
            report_paths.extend(sorted(output_dir.glob("*-report.json")))
    for report_path in report_paths:
        if not report_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if report.get("ok") is True:
            report_outputs = dict(report.get("outputs") or {})
            report_outputs.setdefault("report", str(report_path))
            report_outputs.setdefault("outputDir", str(report_path.parent))
            if "sourceAudio" not in report_outputs:
                audio_candidates = [path for path in report_path.parent.iterdir() if path.is_file() and path.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}]
                if audio_candidates:
                    report_outputs["sourceAudio"] = str(audio_candidates[0])
            report["outputs"] = report_outputs
            return report
    return None


def _mark_completed_from_artifacts(job: dict[str, Any]) -> Optional[dict[str, Any]]:
    outputs = dict(job.get("outputs") or {})
    result = _public_result_from_outputs(outputs)
    if not result:
        return None
    outputs = {**outputs, **dict(result.get("outputs") or {})}
    job.update(
        status="completed",
        resumable=False,
        completedAt=job.get("completedAt") or now_iso(),
        result=result,
        outputs=outputs,
        segmentCount=result.get("segmentCount"),
        source=result.get("source"),
        noMockFallback=result.get("noMockFallback"),
        recoveredAt=now_iso(),
    )
    _write_job(job)
    return job


def _find_existing_job(audio_path: Path, title: str, scene: str, glossary: str) -> Optional[dict[str, Any]]:
    for path in reversed(_iter_job_paths()):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        request = job.get("request") or {}
        if not job.get("jobId") or not request:
            continue
        if (
            request.get("audioPath") == str(audio_path)
            and request.get("title") == title
            and request.get("scene") == scene
            and (request.get("glossary") or "") == (glossary or "")
        ):
            return job
    return None


def _recover_incomplete_jobs() -> dict[str, Any]:
    recovered: list[str] = []
    completed_from_artifacts: list[str] = []
    skipped: list[str] = []
    for path in _iter_job_paths():
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skipped.append(str(path))
            continue
        job_id = job.get("jobId")
        if not job_id or path != _job_path(job_id):
            skipped.append(str(path))
            continue
        status = job.get("status")
        if status == "completed":
            continue
        if _mark_completed_from_artifacts(job):
            completed_from_artifacts.append(job_id)
            continue
        if status in {"queued", "running"}:
            job.update(status="queued", resumable=True, recoveredAt=now_iso(), updatedAt=now_iso())
            _write_job(job)
            _submit_job_worker(job_id)
            recovered.append(job_id)
    return {"recovered": recovered, "completedFromArtifacts": completed_from_artifacts, "skipped": skipped}


def _run_transcription_job(job_id: str) -> None:
    log_path = _log_path(job_id)
    try:
        job = _update_job(
            job_id,
            status="running",
            resumable=False,
            startedAt=now_iso(),
            attempts=int(_read_job(job_id).get("attempts") or 0) + 1,
        )
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
                    output_dir_is_run_dir=True,
                )
        public_result = {key: value for key, value in result.items() if key != "project"}
        _update_job(
            job_id,
            status="completed",
            resumable=False,
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
            resumable=True,
            completedAt=now_iso(),
            error=f"{type(exc).__name__}: {exc}",
            logPath=str(log_path),
        )
    finally:
        with _jobs_lock:
            _scheduled_job_ids.discard(job_id)


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
    resolved_title = title or audio.stem
    existing_job = _find_existing_job(audio, resolved_title, scene, glossary)
    if existing_job and existing_job.get("status") == "completed":
        existing_job = _ensure_html_report(existing_job)
        return {
            "ok": True,
            "submitted": False,
            "jobId": existing_job["jobId"],
            "status": "completed",
            "message": "A completed transcription job already exists for this audio/title. Use recorder_job_result to read outputs.",
            "jobPath": str(_job_path(existing_job["jobId"])),
            "logPath": existing_job.get("logPath") or str(_log_path(existing_job["jobId"])),
            "outputDir": (existing_job.get("outputs") or {}).get("outputDir") or (existing_job.get("request") or {}).get("outputDir"),
            "outputs": existing_job.get("outputs", {}),
        }
    if existing_job and existing_job.get("status") in {"queued", "running"}:
        job_id = existing_job["jobId"]
        return {
            "ok": True,
            "submitted": False,
            "resumed": False,
            "jobId": job_id,
            "status": existing_job.get("status"),
            "message": "An incomplete transcription job already exists for this audio/title. Use recorder_job_status to continue tracking it.",
            "jobPath": str(_job_path(job_id)),
            "logPath": existing_job.get("logPath") or str(_log_path(job_id)),
            "outputDir": (existing_job.get("request") or {}).get("outputDir"),
        }
    if existing_job and existing_job.get("status") == "failed":
        job_id = existing_job["jobId"]
        job = _update_job(job_id, status="queued", resumable=True, lastError=existing_job.get("error"), error=None, recoveredAt=now_iso())
        _submit_job_worker(job_id)
        return {
            "ok": True,
            "submitted": True,
            "resumed": True,
            "jobId": job_id,
            "status": "queued",
            "message": "Existing failed transcription job resumed. Use recorder_job_status and recorder_job_result to check progress and outputs.",
            "jobPath": str(_job_path(job_id)),
            "logPath": job.get("logPath") or str(_log_path(job_id)),
            "outputDir": (job.get("request") or {}).get("outputDir"),
        }

    job_id = f"rec-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    target_output_dir = (Path(output_dir).expanduser().resolve() / job_id / "artifacts") if output_dir else _artifact_dir(job_id)
    job = {
        "jobId": job_id,
        "status": "queued",
        "resumable": True,
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "attempts": 0,
        "request": {
            "audioPath": str(audio),
            "title": resolved_title,
            "scene": scene,
            "glossary": glossary,
            "outputDir": str(target_output_dir),
            "audioSlug": safe_slug(resolved_title),
        },
        "outputs": {"outputDir": str(target_output_dir)},
        "logPath": str(_log_path(job_id)),
    }
    with _jobs_lock:
        _write_job(job)
    _submit_job_worker(job_id)
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
def recorder_resume_jobs() -> dict:
    """Scan persisted MCP jobs and resume queued/running jobs from a previous disconnected process."""
    return {"ok": True, **_recover_incomplete_jobs()}


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
    elif job.get("status") in {"queued", "running"} and job_id not in _scheduled_job_ids:
        if _mark_completed_from_artifacts(job):
            job = _ensure_html_report(_read_job(job_id))
        else:
            job = _update_job(job_id, status="queued", resumable=True, recoveredAt=now_iso())
            _submit_job_worker(job_id)
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
    status = recorder_job_status(job_id)
    job = _read_job(job_id)
    if job.get("status") != "completed":
        return {
            "ok": False,
            "jobId": job_id,
            "status": job.get("status"),
            "message": "Job is not completed yet. Call recorder_job_status again later.",
            "error": job.get("error"),
            "logPath": job.get("logPath"),
            "outputs": status.get("outputs", {}),
            "resumable": status.get("resumable"),
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
    if output_dir is None and job_id:
        output_dir = str(Path(project_json).expanduser().resolve().parent)
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
    if output_dir is None and job_id:
        output_dir = str(Path(project_json).expanduser().resolve().parent)
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
    _recover_incomplete_jobs()
    mcp.run()
