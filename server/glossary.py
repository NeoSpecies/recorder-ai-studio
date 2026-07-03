from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .core import now_iso

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOSSARY_DIR = ROOT / "data" / "glossary"


def default_glossary_dir() -> Path:
    configured = os.environ.get("RECORDER_AI_GLOSSARY_DIR")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_GLOSSARY_DIR

SCENE_CATEGORY_MAP: Dict[str, List[str]] = {
    "meeting": ["global", "meeting"],
    "technical_review": ["global", "meeting", "technology"],
    "research": ["global", "research", "technology", "business"],
    "research_interview": ["global", "research", "technology", "business"],
    "sales": ["global", "sales", "business"],
    "interview": ["global", "interview", "people"],
    "course": ["global", "course", "technology"],
}

CATEGORY_RULES: Dict[str, List[str]] = {
    "ai_chip": ["芯片", "微架构", "NPU", "GPU", "编译器", "工具链", "算力", "SoC", "端侧"],
    "ai_agent": ["Agent", "RAG", "MCP", "智能体", "工作流", "多智能体", "记忆", "召回"],
    "nas": ["NAS", "存储", "绿联", "极空间", "群晖", "威联通", "私有云"],
    "business": ["客户", "商机", "报价", "合同", "销售", "渠道", "预算", "采购"],
    "people": ["负责人", "同事", "团队", "老板", "客户", "候选人"],
    "product": ["产品", "版本", "需求", "功能", "体验", "交付", "验收"],
}

COMMON_STOPWORDS = {
    "我们", "这个", "那个", "就是", "然后", "因为", "所以", "但是", "还是", "进行", "一个", "一些", "现在", "可能", "需要", "没有", "可以", "比较", "如果", "时候", "问题",
    "会议", "内容", "整体", "情况", "方面", "相关", "后续", "当前", "里面", "出来", "一下", "他们", "这里", "今天", "刚才", "非常", "继续",
}


def normalize_term_candidate(value: str) -> str:
    term = str(value or "").strip(" ，,。；;：:、.\n\t")
    term = re.sub(r"^(和|与|及|或|对|把|将|在|从|到|的|了|是|为|做|讨论|关于|这次|本次)+", "", term)
    if "和" in term and not re.search(r"(?:联合|和谐|共和国)", term):
        term = term.split("和")[-1]
    if "与" in term:
        term = term.split("与")[-1]
    term = re.sub(r"(的|了|等|相关|方面)$", "", term)
    return term.strip(" ，,。；;：:、.\n\t")


def normalize_category(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", str(value or "").strip().lower())
    return value.strip("_") or "global"


def parse_category_list(value: Optional[str | Iterable[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [normalize_category(item) for item in re.split(r"[,，\s]+", value) if item.strip()]
    return [normalize_category(str(item)) for item in value if str(item).strip()]


def _read_category_file(category: str, glossary_dir: Path) -> Dict[str, Any]:
    path = glossary_dir / f"{normalize_category(category)}.json"
    if not path.exists():
        return {"category": normalize_category(category), "terms": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {"category": normalize_category(category), "terms": data}
    data.setdefault("category", normalize_category(category))
    data.setdefault("terms", [])
    return data


def _write_category_file(category: str, data: Dict[str, Any], glossary_dir: Path) -> Path:
    glossary_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalize_category(category)
    path = glossary_dir / f"{normalized}.json"
    data["category"] = normalized
    data["updatedAt"] = now_iso()
    data.setdefault("terms", [])
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def infer_categories(scene: str = "meeting", text: str = "", explicit_categories: Optional[Iterable[str]] = None) -> List[str]:
    categories: List[str] = []
    for category in SCENE_CATEGORY_MAP.get(scene or "meeting", ["global", scene or "meeting"]):
        normalized = normalize_category(category)
        if normalized not in categories:
            categories.append(normalized)
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword.lower() in text.lower() for keyword in keywords):
            normalized = normalize_category(category)
            if normalized not in categories:
                categories.append(normalized)
    for category in parse_category_list(explicit_categories):
        if category not in categories:
            categories.append(category)
    if "global" not in categories:
        categories.insert(0, "global")
    return categories[:8]


def load_glossary(categories: Iterable[str], glossary_dir: str | Path | None = None) -> Dict[str, Any]:
    root = Path(glossary_dir).expanduser().resolve() if glossary_dir else default_glossary_dir()
    loaded_categories: List[str] = []
    terms: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for category in categories:
        normalized = normalize_category(category)
        data = _read_category_file(normalized, root)
        loaded_categories.append(normalized)
        for item in data.get("terms") or []:
            if isinstance(item, str):
                item = {"term": item, "aliases": [], "category": normalized}
            term = str(item.get("term") or "").strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            record = dict(item)
            record.setdefault("aliases", [])
            record.setdefault("category", normalized)
            record.setdefault("frequency", 1)
            terms.append(record)
            seen.add(key)
    return {"glossaryDir": str(root), "categories": loaded_categories, "terms": terms}


def _term_texts(record: Dict[str, Any]) -> List[str]:
    values = [str(record.get("term") or "").strip()]
    values.extend(str(item).strip() for item in record.get("aliases") or [])
    return [item for item in values if item]


def match_glossary_terms(text: str, loaded: Dict[str, Any]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    lowered = text.lower()
    for record in loaded.get("terms") or []:
        matched_alias = ""
        for value in _term_texts(record):
            if value and value.lower() in lowered:
                matched_alias = value
                break
        if matched_alias:
            matches.append({
                "term": record.get("term"),
                "matched": matched_alias,
                "category": record.get("category"),
                "priority": record.get("priority", 5),
                "notes": record.get("notes", ""),
            })
    matches.sort(key=lambda item: (-int(item.get("priority") or 0), str(item.get("term") or "")))
    return matches[:80]


def extract_term_candidates(segments: List[Dict[str, Any]], loaded: Dict[str, Any], limit: int = 40) -> Dict[str, Any]:
    text = "\n".join(str(item.get("textCorrected") or item.get("textRaw") or "") for item in segments)
    matches = match_glossary_terms(text, loaded)
    known = {str(item.get("term") or "").lower() for item in loaded.get("terms") or []}
    known.update(alias.lower() for item in loaded.get("terms") or [] for alias in item.get("aliases") or [])

    candidates: Counter[str] = Counter()
    patterns = [
        r"[A-Za-z][A-Za-z0-9_+\-/]{2,}",
        r"[A-Z]{2,}(?:\s?[A-Z0-9]{1,})*",
        r"[\u4e00-\u9fffA-Za-z0-9]{2,14}(?:芯片|模型|工具链|编译器|平台|系统|项目|架构|方案|版本|接口|框架|产品)",
        r"(?:AI|Agent|RAG|MCP|NAS|NPU|GPU|CPU)[\u4e00-\u9fffA-Za-z0-9_\-]{0,12}",
    ]
    for pattern in patterns:
        for value in re.findall(pattern, text):
            term = normalize_term_candidate(value)
            if len(term) < 2 or term in COMMON_STOPWORDS or term.lower() in known:
                continue
            candidates[term] += 1

    # Chinese chunks that repeat several times are likely project/customer/product terms.
    for value in re.findall(r"[\u4e00-\u9fff]{2,8}", text):
        term = normalize_term_candidate(value)
        if term and term not in COMMON_STOPWORDS and term.lower() not in known:
            candidates[term] += 1

    candidate_items = [
        {
            "term": term,
            "count": count,
            "suggestedCategory": suggest_category(term),
            "reason": "高频或形态上疑似专名/术语，建议由智能体结合上下文确认后写入词库。",
            "status": "candidate",
        }
        for term, count in candidates.most_common(limit)
        if count >= 1
    ]
    low_confidence_terms = []
    for segment in segments:
        confidence = segment.get("confidence")
        if isinstance(confidence, (int, float)) and confidence < 85:
            low_confidence_terms.append({
                "segmentId": segment.get("id"),
                "start": segment.get("start"),
                "confidence": confidence,
                "text": segment.get("textCorrected") or segment.get("textRaw") or "",
                "reason": "ASR 置信度偏低，若包含专名应优先复核。",
            })
    return {"glossaryMatches": matches, "termCandidates": candidate_items, "lowConfidenceTerms": low_confidence_terms[:30]}


def suggest_category(term: str) -> str:
    lowered = str(term or "").lower()
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return category
    if re.search(r"[A-Za-z]", term):
        return "technology"
    return "meeting"


def select_project_glossary(project: Dict[str, Any], explicit_categories: Optional[Iterable[str]] = None, glossary_dir: str | Path | None = None) -> Dict[str, Any]:
    segments = project.get("segments") or []
    text = "\n".join(str(item.get("textCorrected") or item.get("textRaw") or "") for item in segments)
    text = f"{project.get('title') or ''}\n{project.get('scene') or ''}\n{text}"
    project_terms = project.get("glossary") or []
    categories = infer_categories(project.get("scene") or "meeting", text, explicit_categories)
    loaded = load_glossary(categories, glossary_dir=glossary_dir)
    if project_terms:
        loaded["terms"] = [
            {"term": term, "aliases": [], "category": "project", "priority": 10, "source": "project_glossary"}
            for term in project_terms
        ] + loaded.get("terms", [])
        if "project" not in loaded["categories"]:
            loaded["categories"].insert(0, "project")
    analysis = extract_term_candidates(segments, loaded)
    return {**loaded, **analysis}


def update_glossary_from_review(
    project: Dict[str, Any],
    review: Dict[str, Any],
    glossary_dir: str | Path | None = None,
    default_category: str = "meeting",
) -> Dict[str, Any]:
    root = Path(glossary_dir).expanduser().resolve() if glossary_dir else default_glossary_dir()
    raw_updates: List[Dict[str, Any]] = []
    for key in ("confirmedTerms", "glossaryUpdates", "termCorrections"):
        for item in review.get(key) or []:
            if isinstance(item, str):
                raw_updates.append({"term": item})
            elif isinstance(item, dict):
                raw_updates.append(dict(item))
    updated: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for item in raw_updates:
        term = str(item.get("term") or item.get("corrected") or item.get("canonical") or "").strip()
        if not term:
            continue
        confidence = item.get("confidence", 1)
        try:
            confidence_value = float(confidence)
        except Exception:
            confidence_value = 1.0
        if item.get("needHumanConfirm") is True or confidence_value < 0.72:
            skipped.append({"term": term, "reason": "needs_human_confirm_or_low_confidence"})
            continue
        category = normalize_category(item.get("category") or suggest_category(term) or default_category)
        data = _read_category_file(category, root)
        terms = data.setdefault("terms", [])
        existing = None
        for record in terms:
            if str(record.get("term") or "").lower() == term.lower():
                existing = record
                break
        aliases = [str(alias).strip() for alias in item.get("aliases") or [] if str(alias).strip()]
        raw = str(item.get("raw") or "").strip()
        if raw and raw.lower() != term.lower():
            aliases.append(raw)
        if existing:
            existing_aliases = set(existing.get("aliases") or [])
            existing_aliases.update(aliases)
            existing["aliases"] = sorted(existing_aliases)
            existing["frequency"] = int(existing.get("frequency") or 0) + 1
            existing["lastSeenAt"] = now_iso()
            existing["source"] = "workbuddy_review"
            existing["notes"] = item.get("notes") or item.get("reason") or existing.get("notes", "")
        else:
            terms.append({
                "term": term,
                "aliases": sorted(set(aliases)),
                "category": category,
                "priority": int(item.get("priority") or 5),
                "frequency": 1,
                "source": "workbuddy_review",
                "createdAt": now_iso(),
                "lastSeenAt": now_iso(),
                "scenes": sorted({project.get("scene") or default_category}),
                "notes": item.get("notes") or item.get("reason") or "由 WorkBuddy 智能体校准后确认写入。",
            })
        _write_category_file(category, data, root)
        updated.append({"term": term, "category": category, "path": str(root / f"{category}.json")})
    return {"glossaryDir": str(root), "updatedTerms": updated, "skippedTerms": skipped}
