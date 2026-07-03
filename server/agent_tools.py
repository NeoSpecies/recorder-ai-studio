from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .core import generate_insights, get_funasr_model_status, get_funasr_runtime_status, local_funasr_transcript, now_iso, project_to_html, project_to_markdown, release_funasr_model

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
    return {"funasr": get_funasr_model_status(model_name), "runtime": get_funasr_runtime_status()}


def release_model() -> Dict[str, Any]:
    return {"released": True, "runtime": release_funasr_model()}


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
        "runtime": get_funasr_runtime_status(),
    }

    if write_files:
        target_dir = Path(output_dir).expanduser().resolve() if output_dir else DEFAULT_OUTPUT_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        slug = safe_slug(title or audio.stem)
        project_path = target_dir / f"{slug}-project.json"
        transcript_path = target_dir / f"{slug}-transcript.md"
        report_path = target_dir / f"{slug}-report.json"
        html_path = target_dir / f"{slug}-report.html"
        project_path.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
        transcript_path.write_text(project_to_markdown(project), encoding="utf-8")
        html_path.write_text(project_to_html(project), encoding="utf-8")
        public_result = {key: value for key, value in result.items() if key != "project"}
        public_result["outputs"] = {
            "projectJson": str(project_path),
            "markdown": str(transcript_path),
            "htmlReport": str(html_path),
            "report": str(report_path),
        }
        public_result["nextSteps"] = [
            "Open outputs.htmlReport to review the visual meeting report.",
            "Use the 核心议题、重点摘要、详情整理、线索与追问 sections to校准转写后的纪要。",
            "Convert action guidance into owner/date/checkable todos before sharing.",
        ]
        report_path.write_text(json.dumps(public_result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["outputs"] = public_result["outputs"]
        result["nextSteps"] = public_result["nextSteps"]

    return result


def _segment_for_review(segment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": segment.get("id"),
        "start": segment.get("start"),
        "end": segment.get("end"),
        "speaker": segment.get("speaker"),
        "name": segment.get("name"),
        "confidence": segment.get("confidence"),
        "textRaw": segment.get("textRaw") or "",
        "textCurrent": segment.get("textCorrected") or segment.get("textRaw") or "",
        "tags": segment.get("tags") or [],
    }


def build_review_prompt(package: Dict[str, Any]) -> str:
    return f"""# 录音转写校准任务

你是 WorkBuddy 当前智能体。请基于下面的 review package，对本地 FunASR 转写结果进行语义校准和内容优化。不要编造原文不存在的信息；如需推断，请放入 questions 或 clues。

## 目标

1. 修正明显错别字、同音误识别、专有名词错误。
2. 清理口语词、重复语气词、断句不自然的问题。
3. 保留原意和关键事实，不删除重要信息。
4. 输出更通顺的 textCorrected，并为重要修改写 correctionNotes。
5. 重新整理 brief、summary、topics、keyPoints、details、decisions、risks、clues、questions、actionGuidance。
6. 将可执行事项整理为 todos，包含 title、desc、owner、due、done。

## 项目信息

- title: {package.get('title')}
- scene: {package.get('scene')}
- segmentCount: {package.get('segmentCount')}
- glossary: {', '.join(package.get('glossary') or [])}

## 返回 JSON Schema

请只返回 JSON，不要返回 Markdown 代码块：

{{
  "title": "可选，校准后的标题",
  "reviewSummary": "本次校准总体说明",
  "segments": [
    {{
      "id": "原 segment id",
      "textCorrected": "校准后的文本",
      "correctionNotes": "说明修正了什么，可为空",
      "tags": ["#标签"]
    }}
  ],
  "insights": {{
    "brief": "简述",
    "summary": ["重点摘要"],
    "topics": [{{"title": "议题", "brief": "说明", "details": ["证据"], "weight": 1}}],
    "keyPoints": ["关键重点"],
    "details": ["详情整理"],
    "decisions": ["决策/结论"],
    "risks": ["风险"],
    "clues": ["线索"],
    "questions": ["追问/待确认"],
    "actionGuidance": [{{"step": "1", "title": "下一步", "desc": "说明"}}],
    "keywords": ["关键词"]
  }},
  "todos": [{{"title": "待办", "desc": "说明", "owner": "未分配", "due": "", "done": false}}]
}}

## 使用方式

校准完成后，把上述 JSON 作为 recorder_apply_review 的 review_json 参数写回，生成 calibrated HTML/Markdown/JSON 报告。
"""


def prepare_review_package(
    project_path: str | Path,
    output_dir: str | Path | None = None,
    max_segments: int = 120,
) -> Dict[str, Any]:
    project_file = Path(project_path).expanduser().resolve()
    if not project_file.exists():
        raise FileNotFoundError(f"Project JSON not found: {project_file}")
    project = json.loads(project_file.read_text(encoding="utf-8"))
    segments = project.get("segments") or []
    review_segments = [_segment_for_review(segment) for segment in segments[:max_segments]]
    low_confidence = [segment for segment in review_segments if isinstance(segment.get("confidence"), (int, float)) and segment.get("confidence", 100) < 85]
    package: Dict[str, Any] = {
        "projectPath": str(project_file),
        "title": project.get("title"),
        "scene": project.get("scene") or "meeting",
        "glossary": project.get("glossary") or [],
        "segmentCount": len(segments),
        "includedSegmentCount": len(review_segments),
        "truncated": len(segments) > len(review_segments),
        "segments": review_segments,
        "lowConfidenceSegments": low_confidence,
        "rawTranscript": "\n".join(item.get("textCurrent") or "" for item in review_segments),
        "originalOutputs": {},
        "reviewWorkflow": {
            "step1": "Use the current WorkBuddy agent model to calibrate this package.",
            "step2": "Return JSON following reviewPromptPath instructions.",
            "step3": "Call recorder_apply_review with review_json to write calibrated outputs.",
        },
    }
    package["reviewPrompt"] = build_review_prompt(package)

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else project_file.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    slug = safe_slug(project.get("title") or project_file.stem.replace("-project", ""))
    package_path = target_dir / f"{slug}-review-package.json"
    prompt_path = target_dir / f"{slug}-review-prompt.md"
    package_path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    prompt_path.write_text(package["reviewPrompt"], encoding="utf-8")
    return {
        "ok": True,
        "projectJson": str(project_file),
        "reviewPackagePath": str(package_path),
        "reviewPromptPath": str(prompt_path),
        "segmentCount": len(segments),
        "includedSegmentCount": len(review_segments),
        "truncated": package["truncated"],
        "lowConfidenceCount": len(low_confidence),
        "reviewPackage": package,
        "nextSteps": [
            "Use reviewPromptPath and reviewPackagePath with the current WorkBuddy agent model to produce review_json.",
            "Call recorder_apply_review with that review_json to generate calibrated outputs.",
        ],
    }


def _parse_review_data(review_data: str | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(review_data, str):
        return json.loads(review_data)
    return dict(review_data or {})


def apply_review_to_project(
    project_path: str | Path,
    review_data: str | Dict[str, Any],
    output_dir: str | Path | None = None,
    suffix: str = "calibrated",
) -> Dict[str, Any]:
    project_file = Path(project_path).expanduser().resolve()
    if not project_file.exists():
        raise FileNotFoundError(f"Project JSON not found: {project_file}")
    review = _parse_review_data(review_data)
    original = json.loads(project_file.read_text(encoding="utf-8"))
    project = json.loads(json.dumps(original, ensure_ascii=False))
    if review.get("title"):
        project["title"] = review["title"]

    segment_reviews = {str(item.get("id")): item for item in review.get("segments") or [] if item.get("id") is not None}
    corrected_count = 0
    for segment in project.get("segments") or []:
        item = segment_reviews.get(str(segment.get("id")))
        if not item:
            continue
        corrected = str(item.get("textCorrected") or "").strip()
        if corrected:
            segment["textCorrected"] = corrected
            corrected_count += 1
        if item.get("correctionNotes"):
            segment["correctionNotes"] = item.get("correctionNotes")
        if item.get("tags"):
            segment["tags"] = sorted(set((segment.get("tags") or []) + [str(tag) for tag in item.get("tags") or []]))

    if review.get("todos") is not None:
        project["todos"] = review.get("todos") or []
    if review.get("insights"):
        project["insights"] = review["insights"]
        project["insights"].setdefault("generatedAt", now_iso())
    else:
        generate_insights(project)
    project["review"] = {
        "calibratedAt": now_iso(),
        "reviewSummary": review.get("reviewSummary") or "",
        "correctedSegmentCount": corrected_count,
        "sourceProjectJson": str(project_file),
    }
    project["updatedAt"] = now_iso()

    target_dir = Path(output_dir).expanduser().resolve() if output_dir else project_file.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    base_name = project_file.name.replace("-project.json", "")
    if base_name == project_file.stem:
        base_name = safe_slug(project.get("title") or project_file.stem)
    suffix_slug = safe_slug(suffix or "calibrated")
    output_base = f"{base_name}-{suffix_slug}"
    calibrated_project = target_dir / f"{output_base}-project.json"
    calibrated_markdown = target_dir / f"{output_base}-transcript.md"
    calibrated_html = target_dir / f"{output_base}-report.html"
    calibrated_report = target_dir / f"{output_base}-report.json"

    calibrated_project.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
    calibrated_markdown.write_text(project_to_markdown(project), encoding="utf-8")
    calibrated_html.write_text(project_to_html(project), encoding="utf-8")
    public_result = {
        "ok": True,
        "source": "workbuddy_agent_review",
        "projectJson": str(project_file),
        "correctedSegmentCount": corrected_count,
        "segmentCount": len(project.get("segments") or []),
        "outputs": {
            "calibratedProjectJson": str(calibrated_project),
            "calibratedMarkdown": str(calibrated_markdown),
            "calibratedHtmlReport": str(calibrated_html),
            "calibratedReport": str(calibrated_report),
        },
        "nextSteps": [
            "Open outputs.calibratedHtmlReport to review the final polished report.",
            "Share outputs.calibratedMarkdown as the editable meeting note if needed.",
        ],
    }
    calibrated_report.write_text(json.dumps(public_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return public_result
