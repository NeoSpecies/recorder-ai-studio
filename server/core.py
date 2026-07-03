from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import Counter
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



_ASR_ENGINE: Optional["FunASREngine"] = None
_ASR_ENGINE_LOCK = threading.RLock()


def funasr_keepalive_seconds() -> int:
    try:
        return max(0, int(os.environ.get("FUNASR_KEEPALIVE_SECONDS", "600")))
    except ValueError:
        return 600


class FunASREngine:
    """Process-local FunASR engine with idle-time release.

    The model is reused within FUNASR_KEEPALIVE_SECONDS after the last task, and
    released once it has been idle for longer than that threshold. This avoids a
    permanently resident model while reducing repeated cold starts during active
    WorkBuddy/CLI/MCP sessions.
    """

    def __init__(self, keepalive_seconds: Optional[int] = None) -> None:
        self.keepalive_seconds = funasr_keepalive_seconds() if keepalive_seconds is None else max(0, keepalive_seconds)
        self.model = None
        self.model_signature: Optional[tuple[str, str, str, Optional[str]]] = None
        self.loaded_at: Optional[float] = None
        self.last_used_at: Optional[float] = None
        self.last_load_seconds: Optional[float] = None
        self.last_inference_seconds: Optional[float] = None
        self.load_count = 0
        self.release_count = 0
        self.lock = threading.RLock()

    def _signature(self) -> tuple[str, str, str, Optional[str]]:
        return (
            resolve_funasr_model(os.environ.get("FUNASR_MODEL", "SenseVoiceSmall")),
            os.environ.get("FUNASR_VAD_MODEL", "fsmn-vad"),
            os.environ.get("FUNASR_PUNC_MODEL", "ct-punc"),
            os.environ.get("FUNASR_SPK_MODEL", "") or None,
        )

    def _load_model(self, signature: tuple[str, str, str, Optional[str]]):
        try:
            from funasr import AutoModel
        except Exception as exc:
            raise RuntimeError(f"funasr is not installed: {exc}") from exc

        model_name, vad_model, punc_model, spk_model = signature
        kwargs: Dict[str, Any] = {"model": model_name, "disable_update": True}
        if vad_model:
            kwargs["vad_model"] = vad_model
        if punc_model:
            kwargs["punc_model"] = punc_model
        if spk_model:
            kwargs["spk_model"] = spk_model
        start = time.monotonic()
        model = AutoModel(**kwargs)
        self.last_load_seconds = round(time.monotonic() - start, 3)
        self.load_count += 1
        return model

    def release(self) -> None:
        with self.lock:
            if self.model is not None:
                self.model = None
                self.model_signature = None
                self.loaded_at = None
                self.release_count += 1

    def release_if_idle(self) -> bool:
        with self.lock:
            if self.model is None or self.last_used_at is None:
                return False
            if self.keepalive_seconds <= 0 or time.monotonic() - self.last_used_at > self.keepalive_seconds:
                self.release()
                return True
            return False

    def ensure_model(self):
        with self.lock:
            self.release_if_idle()
            signature = self._signature()
            if self.model is None or self.model_signature != signature:
                self.model = self._load_model(signature)
                self.model_signature = signature
                self.loaded_at = time.monotonic()
            return self.model

    def transcribe(self, audio_path: Path) -> List[Dict[str, Any]]:
        with self.lock:
            model = self.ensure_model()
            chunk_seconds = int(os.environ.get("FUNASR_CHUNK_SECONDS", "600"))
            batch_size_s = int(os.environ.get("FUNASR_BATCH_SIZE_S", "300"))
            segments: List[Dict[str, Any]] = []
            start = time.monotonic()
            try:
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
            finally:
                self.last_inference_seconds = round(time.monotonic() - start, 3)
                self.last_used_at = time.monotonic()

    def status(self) -> Dict[str, Any]:
        with self.lock:
            self.release_if_idle()
            now = time.monotonic()
            loaded = self.model is not None
            idle_seconds = None if self.last_used_at is None else round(now - self.last_used_at, 3)
            remaining = None
            if loaded and idle_seconds is not None:
                remaining = max(0, round(self.keepalive_seconds - idle_seconds, 3))
            return {
                "loaded": loaded,
                "keepaliveSeconds": self.keepalive_seconds,
                "idleSeconds": idle_seconds,
                "remainingKeepaliveSeconds": remaining,
                "lastLoadSeconds": self.last_load_seconds,
                "lastInferenceSeconds": self.last_inference_seconds,
                "loadCount": self.load_count,
                "releaseCount": self.release_count,
                "modelSignature": list(self.model_signature) if self.model_signature else None,
            }


def get_funasr_engine() -> FunASREngine:
    global _ASR_ENGINE
    with _ASR_ENGINE_LOCK:
        keepalive = funasr_keepalive_seconds()
        if _ASR_ENGINE is None or _ASR_ENGINE.keepalive_seconds != keepalive:
            _ASR_ENGINE = FunASREngine(keepalive_seconds=keepalive)
        return _ASR_ENGINE


def get_funasr_runtime_status() -> Dict[str, Any]:
    return get_funasr_engine().status()


def release_funasr_model() -> Dict[str, Any]:
    engine = get_funasr_engine()
    engine.release()
    return engine.status()


def local_funasr_transcript(audio_path: Path) -> List[Dict[str, Any]]:
    """Run local FunASR inference and normalize sentence segments.

    Long recordings are processed in deterministic chunks instead of converting
    the whole MP3 to one huge WAV. This keeps memory bounded and makes the same
    code path usable from the API and from end-to-end test scripts. It never
    fabricates transcript text; failures are surfaced to the API caller.
    """
    return get_funasr_engine().transcribe(audio_path)



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


def score_sentence(sentence: str, keywords: List[str]) -> int:
    score = min(len(sentence), 120) // 12
    score += sum(3 for keyword in keywords if keyword and keyword in sentence)
    score += 4 if re.search(r"必须|重点|关键|核心|优先|目标|结论|决定|风险|问题|建议|需要|推进|交付|验证|测试", sentence) else 0
    return score


def pick_ranked_sentences(sentences: List[str], keywords: List[str], pattern: str | None = None, limit: int = 6) -> List[str]:
    candidates = [sentence for sentence in sentences if not pattern or re.search(pattern, sentence)]
    ranked = sorted(enumerate(candidates), key=lambda pair: (-score_sentence(pair[1], keywords), pair[0]))
    picked: List[str] = []
    for _, sentence in ranked:
        cleaned = sentence.strip()
        if cleaned and cleaned not in picked:
            picked.append(cleaned if len(cleaned) <= 160 else f"{cleaned[:160]}...")
        if len(picked) >= limit:
            break
    return picked


def build_topic_cards(sentences: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    for keyword in keywords[:8]:
        evidence = [s for s in sentences if keyword in s][:4]
        if not evidence:
            continue
        cards.append({
            "title": keyword,
            "brief": evidence[0] if len(evidence[0]) <= 120 else f"{evidence[0][:120]}...",
            "details": evidence,
            "weight": len(evidence),
        })
    if cards:
        return cards
    for index, sentence in enumerate(sentences[:4], start=1):
        cards.append({"title": f"议题 {index}", "brief": sentence[:120], "details": [sentence], "weight": 1})
    return cards


def build_action_guidance(project: Dict[str, Any], insights: Dict[str, Any]) -> List[Dict[str, str]]:
    scene = project.get("scene") or "meeting"
    guidance = [
        {"step": "1", "title": "先校准专有名词", "desc": "结合术语表、参会人和业务上下文，优先修正公司名、项目名、芯片/模型/版本号等关键实体。"},
        {"step": "2", "title": "确认核心议题", "desc": "逐条检查“核心议题”和原文证据，删除误识别导致的伪议题，补充缺失背景。"},
        {"step": "3", "title": "沉淀行动项", "desc": "把“线索与追问”转成负责人、截止时间、验收标准明确的待办。"},
        {"step": "4", "title": "输出可复用纪要", "desc": "将重点摘要、决策结论、风险和下一步计划整理后同步给相关人员。"},
    ]
    if scene == "sales":
        guidance[2]["desc"] = "把客户线索转为商机阶段、关键人、痛点、预算、下一次跟进动作。"
    elif scene == "interview":
        guidance[2]["desc"] = "把候选人/受访者线索转为事实证据、追问问题和后续核验项。"
    return guidance


def generate_insights(project: Dict[str, Any]) -> Dict[str, Any]:
    segments = project.get("segments") or []
    sentences = [item.get("textCorrected") or item.get("textRaw") or "" for item in segments]
    sentences = [item.strip() for item in sentences if item.strip()]
    full_text = "\n".join(sentences)
    keywords = extract_keywords(full_text)
    if not sentences:
        insights = {
            "summary": [],
            "brief": "",
            "topics": [],
            "keyPoints": [],
            "details": [],
            "decisions": [],
            "risks": [],
            "clues": [],
            "questions": [],
            "actionGuidance": [],
            "mindmap": [],
            "keywords": [],
            "generatedAt": now_iso(),
        }
        project["insights"] = insights
        project["updatedAt"] = now_iso()
        return insights

    summary = pick_ranked_sentences(sentences, keywords, limit=5)
    brief = "；".join(summary[:3])
    if len(brief) > 240:
        brief = f"{brief[:240]}..."
    key_points = pick_ranked_sentences(sentences, keywords, r"重点|关键|核心|目标|必须|需要|建议|推进|交付|验证|测试|方案|优化", limit=8)
    details = pick_ranked_sentences(sentences, keywords, limit=10)
    decisions = pick_ranked_sentences(sentences, keywords, r"采用|确定|决定|先做|必须|建议|目标|需要|结论|同意", limit=6)
    risks = pick_ranked_sentences(sentences, keywords, r"风险|不稳定|失败|重试|敏感|隐私|重叠|问题|报错|不可用|瓶颈|慢", limit=6)
    clues = pick_ranked_sentences(sentences, keywords, r"线索|机会|客户|用户|反馈|需求|痛点|下一步|继续|跟进|验证|测试|观察", limit=8)
    questions = pick_ranked_sentences(sentences, keywords, r"什么|如何|是否|为什么|能不能|有没有|需要确认|待确认|疑问|问题", limit=6)
    topics = build_topic_cards(sentences, keywords)
    keyword_counts = Counter(keyword for keyword in keywords for sentence in sentences if keyword in sentence)
    insights = {
        "summary": summary,
        "brief": brief,
        "topics": topics,
        "keyPoints": key_points or summary,
        "details": details,
        "decisions": decisions,
        "risks": risks,
        "clues": clues,
        "questions": questions,
        "actionGuidance": [],
        "mindmap": [project.get("title") or "录音笔记", *[card["title"] for card in topics[:5]]],
        "keywords": keywords,
        "keywordStats": [{"keyword": keyword, "count": keyword_counts.get(keyword, 0)} for keyword in keywords[:12]],
        "generatedAt": now_iso(),
    }
    insights["actionGuidance"] = build_action_guidance(project, insights)
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
    if insights.get("brief"):
        lines.extend(["## 简述", insights["brief"], ""])
    lines.append("## 重点摘要")
    for item in insights.get("summary") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## 核心议题"])
    for topic in insights.get("topics") or []:
        lines.append(f"### {topic.get('title', '议题')}（出现 {topic.get('weight', 1)} 次）")
        lines.append(topic.get("brief", ""))
        for detail in topic.get("details") or []:
            lines.append(f"- {detail}")
    lines.extend(["", "## 关键重点"])
    for item in insights.get("keyPoints") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## 详情整理"])
    for item in insights.get("details") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## 决策 / 结论"])
    for item in insights.get("decisions") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## 风险提示"])
    for item in insights.get("risks") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## 线索与追问"])
    for item in insights.get("clues") or []:
        lines.append(f"- 线索：{item}")
    for item in insights.get("questions") or []:
        lines.append(f"- 待确认：{item}")
    lines.extend(["", "## 转写后工作指导"])
    for item in insights.get("actionGuidance") or []:
        lines.append(f"- {item.get('step')}. {item.get('title')}：{item.get('desc')}")
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


def project_to_html(project: Dict[str, Any]) -> str:
    insights = project.get("insights") or {}
    title = html.escape(project.get("title") or "录音项目")
    tags = project.get("tags") or []
    segments = project.get("segments") or []
    duration = max([float(seg.get("end") or 0) for seg in segments] or [0])

    def esc(value: Any) -> str:
        return html.escape(str(value or ""))

    def list_items(items: List[Any], empty: str = "暂无") -> str:
        if not items:
            return f'<p class="empty">{esc(empty)}</p>'
        return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in items) + "</ul>"

    def metric(label: str, value: Any) -> str:
        return f'<div class="metric"><span>{esc(label)}</span><strong>{esc(value)}</strong></div>'

    topic_cards = "".join(
        f'''<article class="topic-card">
          <div class="topic-head"><h3>{esc(topic.get('title'))}</h3><span>{esc(topic.get('weight', 1))} 条证据</span></div>
          <p>{esc(topic.get('brief'))}</p>
          {list_items(topic.get('details') or [], '暂无证据')}
        </article>'''
        for topic in insights.get("topics") or []
    ) or '<p class="empty">暂无核心议题</p>'

    guidance = "".join(
        f'''<div class="step"><b>{esc(item.get('step'))}</b><div><h3>{esc(item.get('title'))}</h3><p>{esc(item.get('desc'))}</p></div></div>'''
        for item in insights.get("actionGuidance") or []
    )

    transcript = "".join(
        f'''<article class="segment">
          <div><span class="time">{format_time(seg.get('start', 0))}</span><span class="speaker">Speaker {esc(seg.get('speaker'))}</span></div>
          <p>{esc(seg.get('textCorrected') or seg.get('textRaw'))}</p>
        </article>'''
        for seg in segments
    ) or '<p class="empty">暂无转写内容</p>'

    keyword_stats = insights.get("keywordStats") or []
    max_count = max([int(item.get("count") or 0) for item in keyword_stats] or [1])
    keyword_bars = "".join(
        f'''<div class="bar"><span>{esc(item.get('keyword'))}</span><i style="width:{max(8, int((int(item.get('count') or 0) / max_count) * 100))}%"></i><em>{esc(item.get('count'))}</em></div>'''
        for item in keyword_stats
    ) or '<p class="empty">暂无关键词统计</p>'

    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{title} · 录音智能报告</title>
<style>
:root {{ color-scheme: light dark; --bg:#f6f7fb; --card:#ffffff; --text:#172033; --muted:#637083; --line:#e6e9f0; --brand:#3b82f6; --brand2:#8b5cf6; --ok:#16a34a; --warn:#d97706; --danger:#dc2626; --chip:#eef4ff; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --card:#151b23; --text:#e6edf3; --muted:#9aa7b5; --line:#293241; --brand:#60a5fa; --brand2:#a78bfa; --chip:#18263a; }} }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:radial-gradient(circle at 10% -10%, rgba(59,130,246,.22), transparent 36%), var(--bg); color:var(--text); }}
.container {{ max-width:1180px; margin:0 auto; padding:32px 20px 56px; }}
.hero {{ display:grid; gap:22px; grid-template-columns:minmax(0,1.5fr) minmax(280px,.8fr); align-items:stretch; }}
.panel, .card, .topic-card, .segment {{ background:color-mix(in srgb, var(--card) 94%, transparent); border:1px solid var(--line); border-radius:22px; box-shadow:0 18px 48px rgba(15,23,42,.08); }}
.panel {{ padding:28px; }} h1 {{ margin:0 0 12px; font-size:34px; letter-spacing:-.03em; }} h2 {{ margin:0 0 16px; font-size:22px; }} h3 {{ margin:0; font-size:16px; }} p {{ line-height:1.7; }} .muted {{ color:var(--muted); }}
.metrics {{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }} .metric {{ padding:16px; border-radius:16px; background:var(--chip); }} .metric span {{ display:block; color:var(--muted); font-size:13px; }} .metric strong {{ display:block; margin-top:6px; font-size:22px; }}
.tags {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:18px; }} .tag {{ padding:6px 10px; border-radius:999px; background:var(--chip); color:var(--brand); font-size:13px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; margin-top:18px; }} .card {{ padding:22px; }} ul {{ padding-left:20px; margin:0; }} li {{ margin:9px 0; line-height:1.65; }}
.topic-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }} .topic-card {{ padding:18px; }} .topic-head {{ display:flex; justify-content:space-between; gap:10px; margin-bottom:10px; }} .topic-head span {{ color:var(--muted); font-size:13px; }}
.steps {{ display:grid; gap:12px; }} .step {{ display:flex; gap:14px; padding:14px; border:1px solid var(--line); border-radius:16px; }} .step b {{ width:30px; height:30px; border-radius:50%; display:grid; place-items:center; color:white; background:linear-gradient(135deg,var(--brand),var(--brand2)); flex:0 0 auto; }} .step p {{ margin:4px 0 0; color:var(--muted); }}
.bar {{ display:grid; grid-template-columns:90px 1fr 32px; gap:10px; align-items:center; margin:12px 0; }} .bar i {{ display:block; height:10px; border-radius:999px; background:linear-gradient(90deg,var(--brand),var(--brand2)); }} .bar em {{ color:var(--muted); font-style:normal; text-align:right; }}
.segment {{ padding:16px 18px; margin:12px 0; }} .segment div {{ display:flex; gap:10px; align-items:center; }} .time {{ color:var(--brand); font-weight:700; }} .speaker {{ color:var(--muted); font-size:13px; }} .segment p {{ margin:8px 0 0; }} .empty {{ color:var(--muted); }}
.section {{ margin-top:28px; }} .full {{ grid-column:1 / -1; }}
@media (max-width:860px) {{ .hero, .grid, .topic-grid {{ grid-template-columns:1fr; }} h1 {{ font-size:28px; }} }}
</style>
</head>
<body>
  <main class="container">
    <section class="hero">
      <div class="panel">
        <p class="muted">录音智能分析报告 · 本地 FunASR 转写，无 mock fallback</p>
        <h1>{title}</h1>
        <p>{esc(insights.get('brief') or '已完成真实录音转写，可继续进行人工校准、议题确认和行动项沉淀。')}</p>
        <div class="tags">{''.join(f'<span class="tag">{esc(tag)}</span>' for tag in tags[:20])}</div>
      </div>
      <div class="panel metrics">
        {metric('转写段落', len(segments))}
        {metric('音频时长', format_time(duration))}
        {metric('识别来源', project.get('transcriptionSource') or 'local_funasr')}
        {metric('核心议题', len(insights.get('topics') or []))}
      </div>
    </section>

    <section class="grid section">
      <div class="card"><h2>重点摘要</h2>{list_items(insights.get('summary') or [])}</div>
      <div class="card"><h2>关键重点</h2>{list_items(insights.get('keyPoints') or [])}</div>
      <div class="card"><h2>决策 / 结论</h2>{list_items(insights.get('decisions') or [], '暂无明确决策')}</div>
      <div class="card"><h2>风险提示</h2>{list_items(insights.get('risks') or [], '暂无明显风险')}</div>
    </section>

    <section class="section panel"><h2>核心议题与证据</h2><div class="topic-grid">{topic_cards}</div></section>

    <section class="grid section">
      <div class="card"><h2>详情整理</h2>{list_items(insights.get('details') or [])}</div>
      <div class="card"><h2>线索与追问</h2>{list_items((insights.get('clues') or []) + (insights.get('questions') or []), '暂无线索')}</div>
      <div class="card"><h2>关键词热度</h2>{keyword_bars}</div>
      <div class="card"><h2>转写后工作指导</h2><div class="steps">{guidance}</div></div>
    </section>

    <section class="section panel"><h2>完整转写</h2>{transcript}</section>
  </main>
</body>
</html>'''


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
