"""确定性 Mock 客户端。

不需要任何 API key，根据 prompt 内容用启发式规则产出 schema 合法的 JSON，
使整条产线 (含 VLM 分支) 可离线端到端跑通，用于开发与单元测试。

注意: mock 的语义质量很弱 (仅靠正则启发式)，仅用于打通流程，不可用于生产标注。
"""

from __future__ import annotations

import json
import re
from typing import List

import numpy as np

from .base import BaseClient


def _clean_np(phrase: str) -> str:
    words = [w for w in re.findall(r"[a-zA-Z]+", phrase) if w.lower() not in {"the", "a", "an"}]
    return " ".join(words).strip()


def _parse_objects(instruction: str):
    """从指令中粗略抽取 source / target 名词短语 (仅供 mock)。"""
    instr = instruction.strip().lower().rstrip(".")
    target = None

    # "open the X ... put the Y inside/in" 特例
    m = re.search(r"open the (.+?) and put (?:the )?(.+?)(?: inside| in| into|$)", instr)
    if m:
        return _clean_np(m.group(2)), _clean_np(m.group(1))

    # 普通 "... into/in/on/to/inside the TARGET"
    m = re.search(r"\b(?:into|in|inside|onto|on|to) (?:the )?(.+)$", instr)
    if m:
        target = _clean_np(m.group(1))
        instr = instr[:m.start()].strip()

    src = re.sub(r"^(put|pick up|pick|place|grasp|grab|move|push|pull|open|close|take|lift)\s+", "", instr)
    source = _clean_np(src) or "object"
    if target:
        target = re.sub(r"\b(it|down|inside)\b", "", target).strip() or None
    return source, target


class MockClient(BaseClient):
    """启发式 mock。按 system prompt 的特征分发到不同生成逻辑。"""

    def generate(self, system: str, user: str) -> str:
        if "manipulation analyst" in system:
            return self._global_summary(user)
        if "manipulation observer" in system:
            return self._segment_fill(user)
        return "{}"

    def generate_vlm(self, system: str, user: str, images: List[np.ndarray]) -> str:
        return self.generate(system, user)

    # -- Stage 0.5 全局理解 -------------------------------------------------
    def _global_summary(self, user: str) -> str:
        m = re.search(r'Task instruction: "(.+?)"', user)
        instruction = m.group(1) if m else ""
        source, target = _parse_objects(instruction)
        objects = [{"role": "source", "description": source}]
        if target:
            objects.append({"role": "target", "description": target})
        return json.dumps({
            "task_intent": instruction or "manipulate the object",
            "objects": objects,
            "key_events": [f"gripper interacts with the {source}"],
            "scene_context": "tabletop manipulation scene",
        })

    # -- Stage 1.C per-segment 填槽 ----------------------------------------
    def _segment_fill(self, user: str) -> str:
        objs = re.findall(r"^- (.+?): (.+)$", user, re.MULTILINE)
        roles = {role.strip().lower(): desc.strip() for role, desc in objs}
        obj = roles.get("source") or (objs[0][1].strip() if objs else "object")
        target = roles.get("target", "")
        prim_m = re.search(r"Physical primitive: (\S+)", user)
        prim = prim_m.group(1) if prim_m else ""
        prep = "into" if prim == "place" else ""
        direction = "forward" if prim in ("push", "pull") else ""
        return json.dumps({"object": obj, "target": target, "prep": prep, "direction": direction})
