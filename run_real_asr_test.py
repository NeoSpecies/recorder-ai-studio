from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

from server.core import generate_insights, local_funasr_transcript, project_to_markdown

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
AUDIO_PATH = Path(os.environ.get("RECORDER_AI_TEST_AUDIO", ROOT / "samples" / "meeting-demo.mp3"))
OUTPUT_DIR = Path(os.environ.get("RECORDER_AI_OUTPUT_DIR", WORKSPACE / "outputs" / "real-asr-test"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HOME", str(WORKSPACE / ".cache" / "home"))
os.environ.setdefault("MODELSCOPE_CACHE", str(WORKSPACE / ".cache" / "modelscope"))
os.environ.setdefault("MODELSCOPE_CREDENTIALS_PATH", str(WORKSPACE / ".cache" / "modelscope" / "credentials"))
os.environ.setdefault("FUNASR_MODEL", "SenseVoiceSmall")
os.environ.setdefault("FUNASR_CHUNK_SECONDS", "600")
os.environ.setdefault("FUNASR_BATCH_SIZE_S", "300")

PROJECT_TITLE = os.environ.get("RECORDER_AI_TEST_TITLE", AUDIO_PATH.stem or "Real ASR Test")
BASE_URL = os.environ.get("RECORDER_AI_BASE_URL", "http://127.0.0.1:8876")
GLOSSARY = [item.strip() for item in os.environ.get("RECORDER_AI_GLOSSARY", "芯片,工具链,编译器,AI,研发,适配,版本").split(",") if item.strip()]


def safe_slug(value: str) -> str:
    chars = []
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {" ", "-", "_", "."}:
            chars.append("-")
    slug = "".join(chars).strip("-")
    return slug or "real-asr-test"


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Test audio not found: {AUDIO_PATH}. Set RECORDER_AI_TEST_AUDIO=/path/to/audio before running."
        )

    segments = local_funasr_transcript(AUDIO_PATH)
    if not segments:
        raise RuntimeError("Local FunASR produced no transcript segments; refusing to create a fake result.")

    project = {
        "id": "real-asr-local-funasr",
        "title": PROJECT_TITLE,
        "scene": "meeting",
        "glossary": GLOSSARY,
        "audio": {
            "name": AUDIO_PATH.name,
            "path": str(AUDIO_PATH),
            "size": AUDIO_PATH.stat().st_size,
            "type": "audio/mpeg",
        },
        "duration": 0,
        "segments": segments,
        "tags": sorted({tag for segment in segments for tag in segment.get("tags", [])}),
        "todos": [],
        "insights": None,
        "transcriptionSource": "local_funasr",
    }
    generate_insights(project)

    slug = safe_slug(PROJECT_TITLE)
    raw_json_path = OUTPUT_DIR / f"{slug}-local-funasr-result.json"
    md_path = OUTPUT_DIR / f"{slug}-local-funasr-transcript.md"
    report_path = OUTPUT_DIR / f"{slug}-e2e-report.json"

    write_json(raw_json_path, {"source": "local_funasr", "segment_count": len(segments), "project": project})
    md_path.write_text(project_to_markdown(project), encoding="utf-8")

    api_result = {"checked": False}
    try:
        with httpx.Client(base_url=BASE_URL, timeout=120, trust_env=False) as client:
            health = client.get("/api/health")
            health.raise_for_status()
            created = client.post("/api/projects", json={
                "title": PROJECT_TITLE,
                "scene": "meeting",
                "glossary": project["glossary"],
            })
            created.raise_for_status()
            project_id = created.json()["project"]["id"]
            with AUDIO_PATH.open("rb") as file_obj:
                uploaded = client.post(
                    f"/api/projects/{project_id}/upload",
                    files={"file": (AUDIO_PATH.name, file_obj, "audio/mpeg")},
                )
            uploaded.raise_for_status()
            updated = client.put(
                f"/api/projects/{project_id}",
                json={"segments": segments, "insights": project["insights"], "tags": project["tags"]},
            )
            updated.raise_for_status()
            exported = client.get(f"/api/projects/{project_id}/export.md")
            exported.raise_for_status()
            api_md_path = OUTPUT_DIR / f"{slug}-api-export.md"
            api_md_path.write_text(exported.text, encoding="utf-8")
            api_result = {
                "checked": True,
                "project_id": project_id,
                "health": health.json(),
                "upload_size": uploaded.json()["project"]["audio"]["size"],
                "export_path": str(api_md_path),
            }
    except Exception as exc:
        api_result = {"checked": False, "error": f"{type(exc).__name__}: {exc}"}

    report = {
        "status": "passed" if api_result.get("checked") else "asr_completed_api_check_failed",
        "source": "local_funasr",
        "audio_path": str(AUDIO_PATH),
        "audio_size": AUDIO_PATH.stat().st_size,
        "segments": len(segments),
        "outputs": {
            "json": str(raw_json_path),
            "markdown": str(md_path),
        },
        "api": api_result,
        "note": "No fallback transcript was used. Result is generated by local FunASR only.",
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
