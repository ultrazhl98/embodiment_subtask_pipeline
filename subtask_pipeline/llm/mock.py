"""确定性 Mock 客户端。

不需要任何 API key，根据 prompt 内容用启发式规则产出 schema 合法的 JSON，
使整条产线 (含 LLM/VLM 分支) 可离线端到端跑通，用于开发与单元测试。

注意: mock 的语义质量很弱 (仅靠正则启发式)，仅用于打通流程，不可用于生产标注。
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

import numpy as np

from .base import BaseClient

_STOPWORDS = {"the", "a", "an", "up", "it", "and", "then"}


def _clean_np(phrase: str) -> str:
    words = [w for w in re.findall(r"[a-zA-Z]+", phrase) if w.lower() not in {"the", "a", "an"}]
    return " ".join(words).strip()


def _parse_objects(instruction: str):
    """从指令中粗略抽取 source / target 名词短语 (仅供 mock)。"""
    instr = instruction.strip().lower().rstrip(".")
    target = None
    source = None

    # "open the X ... put the Y inside/in" 特例
    m = re.search(r"open the (.+?) and put (?:the )?(.+?)(?: inside| in| into|$)", instr)
    if m:
        target = _clean_np(m.group(1))
        source = _clean_np(m.group(2))
        return source, target

    # 普通 "... into/in/on/to/inside the TARGET"
    m = re.search(r"\b(?:into|in|inside|onto|on|to) (?:the )?(.+)$", instr)
    if m:
        target = _clean_np(m.group(1))
        instr = instr[:m.start()].strip()

    # source: 去掉前导动词
    src = re.sub(r"^(put|pick up|pick|place|grasp|grab|move|push|pull|open|close|take|lift)\s+", "", instr)
    source = _clean_np(src) or "object"
    # 修剪 target 里残留的连接词
    if target:
        target = re.sub(r"\b(it|down|inside)\b", "", target).strip() or None
    return source, target


class MockClient(BaseClient):
    """启发式 mock。按 system prompt 的特征分发到不同生成逻辑。"""

    def generate(self, system: str, user: str) -> str:
        if "manipulation analyst" in system:
            return self._anchor(user)
        if "task planner" in system:
            return self._text_decomp(user)
        if "trajectory analyst" in system:
            return self._negotiate(user)
        if "manipulation observer" in system:
            return self._fallback(user, None)
        if "description writer" in system:
            return self._description(user)
        if "quality control" in system:
            return self._self_check(user)
        return "{}"

    def generate_vlm(self, system: str, user: str, images: List[np.ndarray]) -> str:
        if "manipulation observer" in system:
            return self._fallback(user, images)
        return self.generate(system, user)

    # -- 各 prompt 的 mock 实现 --------------------------------------------
    def _anchor(self, user: str) -> str:
        m = re.search(r'Task instruction: "(.+?)"', user)
        instruction = m.group(1) if m else ""
        source, target = _parse_objects(instruction)
        objs = [{"role": "source", "description": source}]
        if target:
            objs.append({"role": "target", "description": target})
        return json.dumps({"anchor_objects": objs})

    def _read_anchors(self, user: str):
        src = re.search(r"- source(?: object)?: (.+)", user)
        tgt = re.search(r"- target(?: object)?: (.+)", user)
        return (src.group(1).strip() if src else "object",
                tgt.group(1).strip() if tgt else None)

    def _text_decomp(self, user: str) -> str:
        instr_m = re.search(r'Task instruction: "(.+?)"', user)
        instruction = (instr_m.group(1) if instr_m else "").lower()
        source, target = self._read_anchors(user)
        if target and ("open" in instruction and ("inside" in instruction or "put" in instruction)):
            texts = [f"reach toward the {target}", f"pull open the {target}",
                     f"grasp the {source}", f"place the {source} into the {target}"]
        elif target:
            texts = [f"reach toward the {source}", f"grasp the {source} and lift upward",
                     f"move the {source} to the {target} and release"]
        elif "push" in instruction:
            texts = [f"reach toward the {source}", f"push the {source}"]
        else:
            texts = [f"reach toward the {source}", f"grasp the {source} and lift upward"]
        return json.dumps({"subtask_count": len(texts), "subtask_texts": texts})

    def _negotiate(self, user: str) -> str:
        total = int(re.search(r"Total trajectory frames: (\d+)", user).group(1))
        subs = re.findall(r"^\d+: (.+)$", user, re.MULTILINE)
        n = len(subs)
        # 均匀切分作为 mock 的时间戳推断
        bounds = np.linspace(0, total, n + 1).astype(int)
        assignments = []
        for i in range(n):
            assignments.append({
                "subtask_index": i, "subtask_text": subs[i],
                "start_frame": int(bounds[i]),
                "end_frame": int(bounds[i + 1] - 1 if i < n - 1 else total - 1),
            })
        return json.dumps({"assignments": assignments})

    def _fallback(self, user: str, images) -> str:
        subs = re.findall(r"^\d+: (.+)$", user, re.MULTILINE)
        n = max(1, len(subs))
        # 用图像平均亮度做确定性标签，制造随帧变化的 selected_index
        if images:
            v = float(np.mean(images[0]))
            idx = int(v * 1000) % n
        else:
            idx = 0
        idx = int(np.clip(idx, 0, n - 1))
        return json.dumps({
            "selected_index": idx, "selected_subtask": subs[idx] if subs else "",
            "confidence": 0.7, "reasoning": "mock visual match",
        })

    def _description(self, user: str) -> str:
        ref = re.search(r'Reference subtask text \(.*?\):\s*"(.+?)"', user, re.DOTALL)
        text = ref.group(1).strip() if ref else "move the object"
        return json.dumps({"subtask_text": text})

    def _self_check(self, user: str) -> str:
        # mock 一律通过
        return json.dumps({
            "completeness_check": {"passed": True, "missing_steps": [], "verdict": "ok"},
            "redundancy_check": {"passed": True, "duplicate_pairs": [], "verdict": "ok"},
            "overall_passed": True,
        })
