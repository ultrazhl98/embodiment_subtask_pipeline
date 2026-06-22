"""Stage 3 — VLM 描述生成 + 一致性自检。

为每个 segment 生成 subtask 自然语言描述 (锚点 hard constraint, 1-C 文字 soft
constraint)，做物体指称对齐，再对整条序列做完整性 + 去重自检。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import Stage3Config
from ..data.types import AnchorObject, Episode, Segment
from ..llm.base import BaseClient, LLMResponseError
from ..llm.prompts import build_description_prompt, build_self_check_prompt


# ---------------------------------------------------------------------------
# Keyframe 提取
# ---------------------------------------------------------------------------


def segment_keyframes(seg: Segment, k: int) -> List[int]:
    """首帧 + 末帧 + 均匀采样的中间帧，共 k 帧。"""
    s, e = seg.start_frame, seg.end_frame
    if e <= s:
        return [s] * k
    idxs = np.linspace(s, e, k).astype(int)
    return sorted(set(int(i) for i in idxs)) or [s]


# ---------------------------------------------------------------------------
# 描述生成 + 校验
# ---------------------------------------------------------------------------


def _anchor_keywords(anchors: Sequence[AnchorObject]):
    words = set()
    for a in anchors:
        words.update(w.lower().strip(".,") for w in a.description.split() if len(w) > 2)
    return words


def _make_desc_validator(allowed_verbs, anchor_words, max_words: int):
    verbs = {v.lower() for v in allowed_verbs}

    def validate(parsed) -> Optional[str]:
        if not isinstance(parsed, dict):
            return "output must be JSON object"
        text = (parsed.get("subtask_text") or "").strip()
        if not text:
            return "empty subtask_text"
        if text.split()[0].lower() not in verbs:
            return f"must start with allowed verb, got '{text.split()[0]}'"
        if len(text.split()) > max_words:
            return f"subtask_text exceeds {max_words} words"
        if anchor_words and not (anchor_words & {w.lower().strip(".,") for w in text.split()}):
            return "must reference an anchor object"
        return None

    return validate


def _align_object_reference(text: str, anchors: Sequence[AnchorObject]) -> str:
    """字符串层面物体指称对齐: 若 VLM 用了同义词 (mug) 而非 anchor (cup)，替换之。"""
    synonyms = {"mug": "cup", "cup": "cup", "block": "block", "cube": "block",
                "container": "drawer", "box": "drawer"}
    for a in anchors:
        # anchor 关键词 = 颜色+类别，取描述中后两词作为类别近似
        cat_words = [w for w in a.description.split() if w.isalpha()]
        if not cat_words:
            continue
        category = cat_words[-1].lower()
        for syn, canon in synonyms.items():
            if canon == category and syn != category and re.search(rf"\b{syn}\b", text):
                text = re.sub(rf"\b{syn}\b", category, text)
    return text


def generate_description(episode: Episode, seg: Segment, anchors: Sequence[AnchorObject],
                         reference_text: str, client: BaseClient, cfg: Stage3Config,
                         allowed_verbs, extra_hint: str = "") -> tuple:
    """返回 (subtask_text, failed)。失败时回退 reference_text。"""
    kf = segment_keyframes(seg, cfg.keyframes_per_segment)
    images = [episode.image(i) for i in kf] if episode.has_images else None
    system, user = build_description_prompt(
        episode.task_instruction, anchors, reference_text,
        seg.start_frame, seg.end_frame, extra_hint)
    validator = _make_desc_validator(allowed_verbs, _anchor_keywords(anchors), cfg.max_desc_words)
    try:
        parsed = client.generate_json(system, user, images=images, validator=validator)
        text = _align_object_reference(parsed["subtask_text"].strip(), anchors)
        return text, False
    except LLMResponseError:
        return reference_text, True


# ---------------------------------------------------------------------------
# 自检: 完整性 + 去重
# ---------------------------------------------------------------------------


def _word_overlap(a: str, b: str) -> float:
    wa = {w.lower() for w in re.findall(r"[a-zA-Z]+", a)}
    wb = {w.lower() for w in re.findall(r"[a-zA-Z]+", b)}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def local_dedup_check(texts: Sequence[str], cfg: Stage3Config) -> List[tuple]:
    """词汇重叠率本地去重检查，返回疑似重复的相邻对 [(i, i+1, overlap)]。"""
    dup = []
    for i in range(len(texts) - 1):
        ov = _word_overlap(texts[i], texts[i + 1])
        if ov > cfg.dedup_word_overlap:
            dup.append((i, i + 1, round(ov, 3)))
    return dup


def self_check(episode: Episode, texts: Sequence[str], client: BaseClient) -> Dict:
    system, user = build_self_check_prompt(episode.task_instruction, texts)
    try:
        parsed = client.generate_json(system, user)
    except LLMResponseError:
        return {"overall_passed": True, "completeness_check": {"passed": True, "missing_steps": []},
                "redundancy_check": {"passed": True, "duplicate_pairs": []}, "_llm_failed": True}
    return parsed


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def run_stage3(episode: Episode, stage2_segments: List[Segment], anchors: Sequence[AnchorObject],
               reference_texts: Sequence[str], client: BaseClient, cfg: Stage3Config,
               allowed_verbs) -> Dict:
    segments: List[Segment] = []
    desc_failed_any = False

    for i, seg in enumerate(stage2_segments):
        ref = reference_texts[i] if i < len(reference_texts) else seg.subtask_text
        text, failed = generate_description(episode, seg, anchors, ref, client, cfg, allowed_verbs)
        desc_failed_any = desc_failed_any or failed
        seg.subtask_text = text
        seg.completion_frame = seg.end_frame  # sub-goal 完成帧
        segments.append(seg)

    # ---- 自检 + 重试 ----
    retries = 0
    passed = False
    while retries <= cfg.self_check_max_retries:
        texts = [s.subtask_text for s in segments]
        check = self_check(episode, texts, client)
        local_dups = local_dedup_check(texts, cfg)

        comp = check.get("completeness_check", {})
        red = check.get("redundancy_check", {})
        passed = bool(check.get("overall_passed", True)) and not local_dups
        if passed:
            break
        if retries >= cfg.self_check_max_retries:
            break
        retries += 1

        # 完整性失败: 把 missing_steps 作为 hint 重新生成对应/全部 segment
        missing = comp.get("missing_steps") or []
        if not comp.get("passed", True) and missing:
            hint = "Ensure these steps are covered: " + "; ".join(missing)
            for i, seg in enumerate(segments):
                ref = reference_texts[i] if i < len(reference_texts) else seg.subtask_text
                seg.subtask_text, _ = generate_description(
                    episode, seg, anchors, ref, client, cfg, allowed_verbs, extra_hint=hint)

        # 去重失败: 合并重复对中后一个到前一个，重新生成
        dup_pairs = red.get("duplicate_pairs") or [{"index_a": a, "index_b": b} for a, b, _ in local_dups]
        if dup_pairs:
            segments = _merge_duplicate_segments(episode, segments, anchors, reference_texts,
                                                 dup_pairs, client, cfg, allowed_verbs)

    result = {
        "segments": segments,
        "self_check_passed": passed,
        "self_check_retries": retries,
        "description_gen_failed": desc_failed_any,
    }
    episode.meta["stage3"] = {"self_check_passed": passed, "self_check_retries": retries,
                              "description_gen_failed": desc_failed_any}
    return result


def _merge_duplicate_segments(episode, segments, anchors, reference_texts, dup_pairs,
                              client, cfg, allowed_verbs):
    """合并第一个重复对，重新生成描述。一次只合并一对，避免下标错乱。"""
    pair = dup_pairs[0]
    a, b = pair.get("index_a"), pair.get("index_b")
    if a is None or b is None or b >= len(segments) or a < 0 or a == b:
        return segments
    a, b = min(a, b), max(a, b)
    merged = Segment(
        subtask_text=segments[a].subtask_text,
        start_frame=segments[a].start_frame, end_frame=segments[b].end_frame,
        keyframe=(segments[a].start_frame + segments[b].end_frame) // 2,
        completion_frame=segments[b].end_frame,
    )
    ref = reference_texts[a] if a < len(reference_texts) else merged.subtask_text
    merged.subtask_text, _ = generate_description(episode, merged, anchors, ref, client, cfg, allowed_verbs)
    return segments[:a] + [merged] + segments[b + 1:]
