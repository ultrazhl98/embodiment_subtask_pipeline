"""Stage 1-C — Per-segment 描述生成 (模板 + VLM 填槽)。

对 Stage 1-B 切好的每段, 选定 primitive 对应的描述模板, 让 VLM 从 global_summary
的物体列表里识别/grounding 出 object / target 槽位, 填入模板得到 subtask_text。
这是一个受约束的填槽任务 (而非开放生成), 稳定性高。并入了原 Stage 3 的描述职责:
不再做 LLM 自检, 完整性/去重交由 Stage 2 规则与上游分割保证。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import (PRIMITIVE_TEMPLATES, PRIMITIVE_TO_VERB, PRIMITIVES_REQUIRING_TARGET,
                      Stage1cConfig, starts_with_allowed_verb)
from ..data.types import Episode, Segment
from ..llm.base import BaseClient, LLMResponseError
from ..llm.prompts import build_segment_fill_prompt

_DEFAULT_SLOTS = {"object": "object", "target": "the target", "prep": "into", "direction": "forward"}


def _segment_keyframes(seg: Segment, k: int) -> List[int]:
    s, e = seg.start_frame, seg.end_frame
    if e <= s:
        return [s]
    idxs = np.linspace(s, e, k).astype(int)
    return sorted(set(int(i) for i in idxs)) or [s]


def _object_words(objects: Sequence[dict]) -> set:
    words = set()
    for o in objects:
        words.update(w.lower().strip(".,") for w in o.get("description", "").split() if len(w) > 2)
    return words


def fill_template(primitive: str, slots: Dict[str, str]) -> str:
    """用槽位填充 primitive 模板, 折叠多余空格。"""
    template = PRIMITIVE_TEMPLATES.get(primitive)
    if template is None:  # 未知 primitive: 退化为 verb + object
        verb = PRIMITIVE_TO_VERB.get(primitive, "move to")
        template = verb + " the {object}"
    merged = {**_DEFAULT_SLOTS, **{k: (v or _DEFAULT_SLOTS.get(k, "")).strip()
                                   for k, v in slots.items()}}
    text = template.format_map(merged)
    return " ".join(text.split())


def _make_fill_validator(primitive: str, object_words: set):
    def validate(parsed) -> Optional[str]:
        if not isinstance(parsed, dict):
            return "output must be a JSON object"
        obj = (parsed.get("object") or "").strip()
        if not obj:
            return "missing 'object'"
        if object_words and not (object_words & {w.lower().strip(".,") for w in obj.split()}):
            return "object must reference a known scene object"
        if primitive in PRIMITIVES_REQUIRING_TARGET and not (parsed.get("target") or "").strip():
            return f"primitive '{primitive}' requires a non-empty target"
        return None
    return validate


def _bootstrap_slots(global_summary: Optional[dict]) -> Dict[str, str]:
    """无 VLM 填槽时, 尽量用 global_summary 的 source/target 物体作槽位。"""
    slots = dict(_DEFAULT_SLOTS)
    if global_summary:
        for o in global_summary.get("objects", []):
            if o.get("role") == "source":
                slots["object"] = o.get("description", slots["object"])
            elif o.get("role") == "target":
                slots["target"] = o.get("description", slots["target"])
    return slots


def run_stage1c(episode: Episode, segments: List[Segment], client: BaseClient,
                cfg: Stage1cConfig, allowed_verbs: Sequence[str],
                enable_vlm: bool = True) -> List[Segment]:
    """逐段填充 subtask_text。enable_vlm=False 或无图像时走 bootstrap 模板。"""
    global_summary = episode.meta.get("global_summary")
    objects = global_summary.get("objects", []) if global_summary else []
    object_words = _object_words(objects)
    boot_slots = _bootstrap_slots(global_summary)

    use_vlm = enable_vlm and episode.has_images
    desc_failed_any = False

    for seg in segments:
        prim = seg.primitive_label or "unknown"
        template = PRIMITIVE_TEMPLATES.get(prim, "move to the {object}")
        slots = dict(boot_slots)
        if use_vlm:
            kf = _segment_keyframes(seg, cfg.keyframes_per_segment)
            images = [episode.image(i) for i in kf]
            system, user = build_segment_fill_prompt(
                episode.task_instruction, prim, template, objects,
                seg.start_frame, seg.end_frame)
            try:
                parsed = client.generate_json(system, user, images=images,
                                              validator=_make_fill_validator(prim, object_words))
                slots = {"object": parsed.get("object", ""), "target": parsed.get("target", ""),
                         "prep": parsed.get("prep", ""), "direction": parsed.get("direction", "")}
            except LLMResponseError:
                desc_failed_any = True  # 回退到 bootstrap 槽位

        text = fill_template(prim, slots)
        # 终检: 必须以允许动词开头且不超长, 否则回退到纯模板 (bootstrap 槽位)
        if not starts_with_allowed_verb(text, allowed_verbs) or len(text.split()) > cfg.max_desc_words:
            text = fill_template(prim, boot_slots)
            desc_failed_any = True

        seg.subtask_text = text
        seg.completion_frame = seg.end_frame

    episode.meta["stage1c"] = {"description_gen_failed": desc_failed_any,
                               "used_vlm": use_vlm}
    return segments
