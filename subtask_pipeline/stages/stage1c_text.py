"""Stage 1-C — 文本语义分解 (路径 C，参考 CycleVLA)。

用 LLM 对任务指令做纯文本原子 subtask 分解，注入锚点控制物体指称。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..data.types import AnchorObject, Episode
from ..llm.base import BaseClient
from ..llm.prompts import build_text_decomp_prompt


def _make_validator(allowed_verbs: Sequence[str], anchors: Sequence[AnchorObject]):
    verbs = {v.lower() for v in allowed_verbs}
    # 锚点关键词 (用于模糊匹配物体指称)
    anchor_words = set()
    for a in anchors:
        anchor_words.update(w.lower() for w in a.description.split() if len(w) > 2)

    def validate(parsed) -> Optional[str]:
        if not isinstance(parsed, dict):
            return "output must be a JSON object"
        texts = parsed.get("subtask_texts")
        if not isinstance(texts, list) or not texts:
            return "missing 'subtask_texts'"
        count = parsed.get("subtask_count", len(texts))
        if count != len(texts):
            return "subtask_count != len(subtask_texts)"
        if not (2 <= len(texts) <= 6):
            return "subtask count must be in 2..6"
        for t in texts:
            if not isinstance(t, str) or not t.strip():
                return "empty subtask text"
            first = t.strip().split()[0].lower()
            if first not in verbs:
                return f"subtask must start with allowed verb, got '{first}'"
            if anchor_words and not (anchor_words & {w.lower().strip('.,') for w in t.split()}):
                return f"subtask '{t}' must reference an anchor object"
        return None

    return validate


def run_stage1c(episode: Episode, anchors: Sequence[AnchorObject], client: BaseClient,
                allowed_verbs: Sequence[str]) -> List[str]:
    system, user = build_text_decomp_prompt(episode.task_instruction, anchors)
    validator = _make_validator(allowed_verbs, anchors)
    parsed = client.generate_json(system, user, validator=validator)
    texts = [t.strip() for t in parsed["subtask_texts"]]
    episode.meta["stage1c"] = {"N_text": len(texts), "subtask_texts": texts}
    return texts
