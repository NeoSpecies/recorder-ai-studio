from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


KEYWORDS = [
    "FunASR", "Plaud", "长录音", "说话人分离", "本地隐私", "校对", "摘要", "脑图", "待办", "云端模型",
    "知识资产", "VAD", "时间戳", "断点续跑", "术语表", "会议", "销售", "访谈", "课程", "风险",
    "芯片", "工具链", "编译器", "算力", "大模型", "研发", "适配", "版本", "模型", "部署",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


def normalize_tag(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    return value if value.startswith("#") else f"#{value}"


def default_project(title: str = "长录音项目", scene: str = "meeting", glossary: Optional[List[str]] = None) -> Dict[str, Any]:
    glossary = glossary or []
    return {
        "id": new_id(),
        "title": title,
        "scene": scene,
        "glossary": glossary,
        "audio": None,
        "duration": 0,
        "segments": [],
        "tags": [],
        "todos": [],
        "insights": None,
        "transcriptionSource": None,
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
    }


def funasr_model_candidates(model_name: str) -> tuple[str, List[Path]]:
    aliases = {
        "SenseVoiceSmall": "iic/SenseVoiceSmall",
        "sensevoice-small": "iic/SenseVoiceSmall",
    }
    resolved_name = aliases.get(model_name, model_name)
    candidates: List[Path] = []
    explicit_dir = os.environ.get("FUNASR_MODEL_DIR")
    if explicit_dir:
        candidates.append(Path(explicit_dir))
    cache_root = os.environ.get("MODELSCOPE_CACHE")
    if cache_root:
        candidates.append(Path(cache_root) / "iic" / "SenseVoiceSmall")
    candidates.append(Path(__file__).resolve().parents[2] / ".cache" / "modelscope" / "iic" / "SenseVoiceSmall")
    return resolved_name, candidates


def get_funasr_model_status(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Return local SenseVoiceSmall weight status without starting inference."""
    resolved_name, candidates = funasr_model_candidates(model_name or os.environ.get("FUNASR_MODEL", "SenseVoiceSmall"))
    status: Dict[str, Any] = {
        "model": resolved_name,
        "ready": False,
        "path": None,
        "downloadedBytes": 0,
        "downloadedMB": 0,
        "estimatedTotalMB": 936,
        "state": "missing",
    }
    for candidate in candidates:
        if not candidate.exists():
            continue
        status["path"] = str(candidate)
        model_pt = candidate / "model.pt"
        incomplete = candidate / "model.pt.incomplete"
        if model_pt.exists():
            size = model_pt.stat().st_size
            status.update({
                "ready": True,
                "downloadedBytes": size,
                "downloadedMB": round(size / 1024 / 1024, 1),
                "state": "ready",
            })
            return status
        if incomplete.exists():
            size = incomplete.stat().st_size
            status.update({
                "downloadedBytes": size,
                "downloadedMB": round(size / 1024 / 1024, 1),
                "state": "incomplete",
            })
            return status
    return status


def resolve_funasr_model(model_name: str) -> str:
    """Resolve FunASR model id/path and fail early on incomplete local weights."""
    resolved_name, candidates = funasr_model_candidates(model_name)
    for candidate in candidates:
        if not candidate.exists():
            continue
        model_pt = candidate / "model.pt"
        incomplete = candidate / "model.pt.incomplete"
        if model_pt.exists():
            return str(candidate)
        if incomplete.exists():
            size_mb = incomplete.stat().st_size / 1024 / 1024
            raise RuntimeError(
                f"FunASR model weights are incomplete: {incomplete} ({size_mb:.1f}MB downloaded). "
                "Wait for model.pt to finish downloading, then retry."
            )
    if resolved_name == "iic/SenseVoiceSmall":
        raise RuntimeError(
            "SenseVoiceSmall weights are not downloaded yet. Download iic/SenseVoiceSmall to MODELSCOPE_CACHE first, "
            "then retry; refusing to pass the repo id directly because this FunASR version treats it as an unregistered model key."
        )
    return resolved_name



def local_funasr_transcript(audio_path: Path) -> List[Dict[str, Any]]:
    """Run local FunASR inference and normalize sentence segments.

    Long recordings are processed in deterministic chunks instead of converting
    the whole MP3 to one huge WAV. This keeps memory bounded and makes the same
    code path usable from the API and from end-to-end test scripts. It never
    fabricates transcript text; failures are surfaced to the API caller.
    """
    try:
        from funasr import AutoModel
    except Exception as exc:
        raise RuntimeError(f"funasr is not installed: {exc}") from exc

    model_name = resolve_funasr_model(os.environ.get("FUNASR_MODEL", "SenseVoiceSmall"))
    vad_model = os.environ.get("FUNASR_VAD_MODEL", "fsmn-vad")
    punc_model = os.environ.get("FUNASR_PUNC_MODEL", "ct-punc")
    spk_model = os.environ.get("FUNASR_SPK_MODEL", "") or None
    kwargs: Dict[str, Any] = {"model": model_name, "disable_update": True}
    if vad_model:
        kwargs["vad_model"] = vad_model
    if punc_model:
        kwargs["punc_model"] = punc_model
    if spk_model:
        kwargs["spk_model"] = spk_model
    model = AutoModel(**kwargs)

    chunk_seconds = int(os.environ.get("FUNASR_CHUNK_SECONDS", "600"))
    batch_size_s = int(os.environ.get("FUNASR_BATCH_SIZE_S", "300"))
    segments: List[Dict[str, Any]] = []
    for chunk_path, offset in iter_audio_chunks(audio_path, chunk_seconds=chunk_seconds):
        try:
            result = model.generate(input=str(chunk_path), batch_size_s=batch_size_s)
            chunk_segments = normalize_funasr_result(result)
            for segment in chunk_segments:
                segment["start"] = float(segment.get("start") or 0) + offset
                segment["end"] = float(segment.get("end") or 0) + offset
                segments.append(segment)
        finally:
            if chunk_path != audio_path:
                chunk_path.unlink(missing_ok=True)
    return segments



def iter_audio_chunks(audio_path: Path, chunk_seconds: int = 600):
    """Yield 16 kHz mono WAV chunks and their offsets in seconds.

    Uses librosa/soundfile so MP3 input works without Homebrew ffmpeg. For WAV
    files and short recordings this still normalizes sample rate/channel count.
    """
    try:
        import librosa
        import soundfile as sf
    except Exception as exc:
        prepared = prepare_audio_for_funasr(audio_path)
        yield prepared, 0.0
        return

    sr = int(os.environ.get("FUNASR_SAMPLE_RATE", "16000"))
    try:
        duration = float(librosa.get_duration(path=str(audio_path)))
    except Exception:
        duration = float(chunk_seconds)
    offset = 0.0
    while offset < max(duration, 0.01):
        y, loaded_sr = librosa.load(str(audio_path), sr=sr, mono=True, offset=offset, duration=chunk_seconds)
        if len(y) == 0:
            break
        tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
        sf.write(str(tmp), y, loaded_sr)
        yield tmp, offset
        offset += chunk_seconds



def prepare_audio_for_funasr(audio_path: Path) -> Path:
    """Convert unsupported audio to 16 kHz mono WAV with macOS afconvert."""
    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        return audio_path
    afconvert = shutil.which("afconvert")
    if not afconvert:
        return audio_path
    tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    cmd = [afconvert, str(audio_path), str(tmp), "-f", "WAVE", "-d", "LEI16@16000", "-c", "1"]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tmp



def clean_funasr_text(text: str) -> str:
    """Remove SenseVoice control tokens while preserving recognized words."""
    text = re.sub(r"<\|[^|>]+\|>", "", text or "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_funasr_result(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    if isinstance(data, dict):
        source = data.get("segments") or data.get("sentence_info") or data.get("sentences") or data.get("text") or []
    else:
        source = data
    if isinstance(source, str):
        parts = [item.strip() for item in re.split(r"[。！？\n]+", source) if item.strip()]
        return [
            {
                "id": new_id(),
                "start": index * 12,
                "end": index * 12 + 10,
                "speaker": "A",
                "name": "未命名",
                "confidence": 90,
                "textRaw": text,
                "textCorrected": clean_funasr_text(text),
                "tags": auto_tags(clean_funasr_text(text)),
            }
            for index, text in enumerate(parts)
            if clean_funasr_text(text)
        ]
    if not isinstance(source, list):
        return []
    segments = []
    for index, item in enumerate(source):
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("sentence") or item.get("transcript") or ""
        cleaned_text = clean_funasr_text(text)
        if not cleaned_text:
            continue
        timestamp = item.get("timestamp") or item.get("ts") or []
        if timestamp and isinstance(timestamp, list) and isinstance(timestamp[0], list):
            start_raw = timestamp[0][0]
            end_raw = timestamp[-1][1]
            has_ms = True
        else:
            start_raw = item.get("start", item.get("start_ms", index * 10))
            end_raw = item.get("end", item.get("end_ms", index * 10 + 8))
            has_ms = "start_ms" in item or "end_ms" in item or bool(timestamp)
        confidence = item.get("confidence", item.get("score", 0.9))
        if confidence <= 1:
            confidence = round(confidence * 100)
        speaker = str(item.get("speaker", item.get("spk", "A"))).replace("Speaker", "").strip() or "A"
        segments.append({
            "id": new_id(),
            "start": float(start_raw) / 1000 if has_ms else float(start_raw),
            "end": float(end_raw) / 1000 if has_ms else float(end_raw),
            "speaker": speaker,
            "name": item.get("name") or "未命名",
            "confidence": int(confidence),
            "textRaw": text,
            "textCorrected": cleaned_text,
            "tags": auto_tags(cleaned_text),
        })
    return segments


def auto_tags(text: str) -> List[str]:
    return [normalize_tag(word) for word in KEYWORDS if word in text]


def extract_keywords(text: str) -> List[str]:
    found = [word for word in KEYWORDS if word in text]
    if found:
        return found
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,6}", text)
    seen: List[str] = []
    for word in words:
        if word not in seen:
            seen.append(word)
    return seen[:8]


def generate_insights(project: Dict[str, Any]) -> Dict[str, Any]:
    segments = project.get("segments") or []
    sentences = [item.get("textCorrected") or item.get("textRaw") or "" for item in segments]
    sentences = [item.strip() for item in sentences if item.strip()]
    full_text = "\n".join(sentences)
    keywords = extract_keywords(full_text)
    if not sentences:
        insights = {
            "summary": [],
            "decisions": [],
            "risks": [],
            "mindmap": [],
            "keywords": [],
            "generatedAt": now_iso(),
        }
        project["insights"] = insights
        project["updatedAt"] = now_iso()
        return insights
    summary = [s if len(s) <= 80 else f"{s[:80]}..." for s in sentences[:4]]
    decisions = [s for s in sentences if re.search(r"采用|确定|决定|先做|必须|建议|目标|需要", s)][:5]
    risks = [s for s in sentences if re.search(r"风险|不稳定|失败|重试|敏感|隐私|重叠|问题", s)][:5]
    insights = {
        "summary": summary,
        "decisions": decisions,
        "risks": risks,
        "mindmap": [project.get("title") or "录音笔记", *keywords[:4]],
        "keywords": keywords,
        "generatedAt": now_iso(),
    }
    project["insights"] = insights
    merged_tags = set(project.get("tags") or [])
    for keyword in keywords:
        tag = normalize_tag(keyword)
        if tag:
            merged_tags.add(tag)
    project["tags"] = sorted(merged_tags)
    project["updatedAt"] = now_iso()
    return insights


def project_to_markdown(project: Dict[str, Any]) -> str:
    insights = project.get("insights") or {}
    lines = [f"# {project.get('title', '录音项目')}", ""]
    lines.append("## 摘要")
    for item in insights.get("summary") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## 决策 / 结论"])
    for item in insights.get("decisions") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## 风险提示"])
    for item in insights.get("risks") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## 待办"])
    for todo in project.get("todos") or []:
        checked = "x" if todo.get("done") else " "
        owner = todo.get("owner") or "未分配"
        lines.append(f"- [{checked}] {todo.get('title', '')} @{owner} - {todo.get('desc', '')}")
    lines.extend(["", "## 标签", " ".join(project.get("tags") or []), "", "## 转写"])
    for segment in project.get("segments") or []:
        lines.append(f"### {format_time(segment.get('start', 0))} Speaker {segment.get('speaker', '')} · {segment.get('name', '')}")
        lines.append(segment.get("textCorrected") or segment.get("textRaw") or "")
        lines.append("")
    return "\n".join(lines)


def format_time(seconds: Any) -> str:
    try:
        total = int(float(seconds))
    except Exception:
        total = 0
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


@dataclass
class ProjectStore:
    db_path: Path
    upload_dir: Path

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self._write({"projects": []})

    def _read(self) -> Dict[str, Any]:
        return json.loads(self.db_path.read_text(encoding="utf-8"))

    def _write(self, data: Dict[str, Any]) -> None:
        self.db_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_projects(self) -> List[Dict[str, Any]]:
        return self._read().get("projects", [])

    def get_project(self, project_id: str) -> Dict[str, Any]:
        for project in self.list_projects():
            if project.get("id") == project_id:
                return project
        raise KeyError(project_id)

    def save_project(self, project: Dict[str, Any]) -> Dict[str, Any]:
        data = self._read()
        projects = data.get("projects", [])
        project["updatedAt"] = now_iso()
        for index, existing in enumerate(projects):
            if existing.get("id") == project.get("id"):
                projects[index] = project
                self._write({"projects": projects})
                return project
        projects.insert(0, project)
        self._write({"projects": projects})
        return project

    def create_project(self, title: str, scene: str, glossary: Optional[List[str]] = None) -> Dict[str, Any]:
        project = default_project(title, scene, glossary)
        return self.save_project(project)

    def update_project(self, project_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        project = self.get_project(project_id)
        for key in ["title", "scene", "glossary", "segments", "tags", "todos", "insights", "duration"]:
            if key in patch:
                project[key] = patch[key]
        return self.save_project(project)

    def attach_upload(self, project_id: str, source_path: Path, filename: str, content_type: str = "") -> Dict[str, Any]:
        project = self.get_project(project_id)
        safe_name = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]", "_", filename) or "audio.bin"
        target = self.upload_dir / f"{project_id}_{safe_name}"
        shutil.copyfile(source_path, target)
        project["audio"] = {
            "name": filename,
            "path": str(target),
            "size": target.stat().st_size,
            "type": content_type,
            "uploadedAt": now_iso(),
        }
        return self.save_project(project)

    def set_segments(self, project_id: str, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        project = self.get_project(project_id)
        project["segments"] = segments
        project["tags"] = sorted(set(project.get("tags") or []) | {tag for seg in segments for tag in (seg.get("tags") or [])})
        return self.save_project(project)
