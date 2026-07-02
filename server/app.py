from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import request, error

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .core import ProjectStore, generate_insights, get_funasr_model_status, get_funasr_runtime_status, local_funasr_transcript, normalize_funasr_result, project_to_markdown

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("RECORDER_AI_DATA", ROOT / "data"))
UPLOAD_DIR = Path(os.environ.get("RECORDER_AI_UPLOADS", ROOT / "uploads"))
DB_PATH = DATA_DIR / "projects.json"

store = ProjectStore(DB_PATH, UPLOAD_DIR)
app = FastAPI(title="Recorder AI Studio", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProjectCreate(BaseModel):
    title: str = "长录音项目"
    scene: str = "meeting"
    glossary: list[str] = []


class ProjectPatch(BaseModel):
    title: Optional[str] = None
    scene: Optional[str] = None
    glossary: Optional[list[str]] = None
    segments: Optional[list[Dict[str, Any]]] = None
    tags: Optional[list[str]] = None
    todos: Optional[list[Dict[str, Any]]] = None
    insights: Optional[Dict[str, Any]] = None
    duration: Optional[float] = None


class TranscribeRequest(BaseModel):
    endpoint: Optional[str] = None


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "service": "recorder-ai-studio", "version": "0.2.0"}


@app.get("/api/asr/status")
def asr_status() -> Dict[str, Any]:
    return {"funasr": get_funasr_model_status(), "runtime": get_funasr_runtime_status()}


@app.get("/api/projects")
def list_projects() -> Dict[str, Any]:
    return {"projects": store.list_projects()}


@app.post("/api/projects")
def create_project(payload: ProjectCreate) -> Dict[str, Any]:
    return {"project": store.create_project(payload.title, payload.scene, payload.glossary)}


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> Dict[str, Any]:
    try:
        return {"project": store.get_project(project_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="project not found")


@app.put("/api/projects/{project_id}")
def update_project(project_id: str, payload: ProjectPatch) -> Dict[str, Any]:
    try:
        patch = {key: value for key, value in payload.model_dump().items() if value is not None}
        return {"project": store.update_project(project_id, patch)}
    except KeyError:
        raise HTTPException(status_code=404, detail="project not found")


@app.post("/api/projects/{project_id}/upload")
async def upload_audio(project_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        store.get_project(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="project not found")
    suffix = Path(file.filename or "audio.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        project = store.attach_upload(project_id, tmp_path, file.filename or "audio.bin", file.content_type or "")
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"project": project}


@app.post("/api/projects/{project_id}/transcribe")
def transcribe(project_id: str, payload: TranscribeRequest = TranscribeRequest()) -> Dict[str, Any]:
    try:
        project = store.get_project(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="project not found")

    segments = []
    source = "none"
    endpoint = payload.endpoint or os.environ.get("FUNASR_ENDPOINT")
    audio = project.get("audio") or {}
    audio_path = audio.get("path")
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=400, detail="audio not uploaded")
    if endpoint:
        try:
            segments = call_funasr(endpoint, Path(audio_path))
            source = "remote_funasr"
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Remote FunASR failed: {exc}") from exc
    else:
        status = get_funasr_model_status()
        if not status.get("ready"):
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Local FunASR model is still downloading; no mock or fallback transcript will be generated.",
                    "status": status,
                },
            )
        try:
            segments = local_funasr_transcript(Path(audio_path))
            source = "local_funasr"
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Local FunASR failed: {exc}") from exc
    if not segments:
        raise HTTPException(status_code=502, detail="FunASR produced no transcript segments")
    project = store.set_segments(project_id, segments)
    project["transcriptionSource"] = source
    store.save_project(project)
    return {"project": project, "segments": segments, "source": source, "runtime": get_funasr_runtime_status()}


@app.post("/api/projects/{project_id}/insights")
def insights(project_id: str) -> Dict[str, Any]:
    try:
        project = store.get_project(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="project not found")
    generated = generate_insights(project)
    store.save_project(project)
    return {"project": project, "insights": generated}


@app.get("/api/projects/{project_id}/export.md", response_class=PlainTextResponse)
def export_markdown(project_id: str) -> str:
    try:
        project = store.get_project(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="project not found")
    return project_to_markdown(project)


@app.get("/api/projects/{project_id}/audio")
def get_audio(project_id: str) -> FileResponse:
    try:
        project = store.get_project(project_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="project not found")
    audio_path = (project.get("audio") or {}).get("path")
    if not audio_path or not Path(audio_path).exists():
        raise HTTPException(status_code=404, detail="audio not found")
    return FileResponse(audio_path, media_type=(project.get("audio") or {}).get("type") or "application/octet-stream")


def call_funasr(endpoint: str, audio_path: Path) -> list[Dict[str, Any]]:
    boundary = "----RecorderAIStudioBoundary"
    audio = audio_path.read_bytes()
    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nsensevoice\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\nverbose_json\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{audio_path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode()
        + audio
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = request.Request(endpoint, data=body, method="POST", headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return normalize_funasr_result(payload)


app.mount("/", StaticFiles(directory=str(ROOT), html=True), name="static")
