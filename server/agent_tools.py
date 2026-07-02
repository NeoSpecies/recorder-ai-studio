from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .core import generate_insights, get_funasr_model_status, local_funasr_transcript, now_iso, project_to_markdown

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs" / "agent-runs"


def safe_slug(value: str) -> str:
    chars: List[str] = []
    for char in (value or "").strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {" ", "-", "_", "."}:
            chars.append("-")
    slug = re.sub(r"-+", "-", "".join(chars)).strip("-")
    return slug or "recording"


def parse_glossary(value: Optional[str | Iterable[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def model_status(model_name: Optional[str] = None) -> Dict[str, Any]:
    return {"funasr": get_funasr_model_status(model_name)}


def build_project(
    audio_path: Path,
    segments: List[Dict[str, Any]],
    title: Optional[str] = None,
    scene: str = "meeting",
    glossary: Optional[str | Iterable[str]] = None,
    source: str = "local_funasr",
) -> Dict[str, Any]:
    project = {
        "id": f"agent-{safe_slug(title or audio_path.stem)}",
        "title": title or audio_path.stem or "录音项目",
        "scene": scene,
        "glossary": parse_glossary(glossary),
        "audio": {
            "name": audio_path.name,
            "path": str(audio_path),
            "size": audio_path.stat().st_size,
            "type": "audio/mpeg" if audio_path.suffix.lower() == ".mp3" else "application/octet-stream",
        },
        "duration": 0,
        "segments": segments,
        "tags": sorted({tag for segment in segments for tag in segment.get("tags", [])}),
        "todos": [],
        "insights": None,
        "transcriptionSource": source,
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
    }
    generate_insights(project)
    return project


def transcribe_audio_file(
    audio_path: str | Path,
    output_dir: str | Path | None = None,
    title: Optional[str] = None,
    scene: str = "meeting",
    glossary: Optional[str | Iterable[str]] = None,
    write_files: bool = True,
) -> Dict[str, Any]:
    audio = Path(audio_path).expanduser().resolve()
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    status = get_funasr_model_status()
    if not status.get("ready"):
        raise RuntimeError(
            "Local FunASR model is not ready; no mock or fallback transcript will be generated. "
            f"Status: {json.dumps(status, ensure_ascii=False)}"
        )

    segments = local_funasr_transcript(audio)
    if not segments:
        raise RuntimeError("Local FunASR produced no transcript segments; refusing to create a fake result.")

    project = build_project(audio, segments, title=title, scene=scene, glossary=glossary)
    result: Dict[str, Any] = {
        "ok": True,
        "source": "local_funasr",
        "noMockFallback": True,
        "audio": {"path": str(audio), "sizeBytes": audio.stat().st_size},
        "segmentCount": len(segments),
        "project": project,
        "outputs": {},
    }

    if write_files:
        target_dir = Path(output_dir).expanduser().resolve() if output_dir else DEFAULT_OUTPUT_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        slug = safe_slug(title or audio.stem)
        project_path = target_dir / f"{slug}-project.json"
        transcript_path = target_dir / f"{slug}-transcript.md"
        report_path = target_dir / f"{slug}-report.json"
        project_path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
        transcript_path.write_text(project_to_markdown(project), encoding="utf-8")
        public_result = {key: value for key, value in result.items() if key != "project"}
        public_result["outputs"] = {
            "projectJson": str(project_path),
            "markdown": str(transcript_path),
            "report": str(report_path),
        }
        report_path.write_text(json.dumps(public_result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["outputs"] = public_result["outputs"]

    return result
