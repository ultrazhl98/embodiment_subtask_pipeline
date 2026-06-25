"""Stage 2 — 规则质量过滤。

分割边界已由 Stage 1-B 用夹爪事件帧 (硬锚点) + RDP 拐点确定，Stage 2 不再做
计数比较式的三路由对齐，而是退化为一层轻量规则过滤 (无 LLM 调用):

- 逐段做 sanity check (原语↔夹爪状态一致性 / 时长合理性 / 文本动词一致性)
- 给整条记录一个置信度标记 Gold / Bronze / Flagged (而非"修复")

注: 旧版 align_by_event_frames(subtask_texts, ...) 假设"文本数驱动分割"，与新架构
"Stage 1-B 驱动分割"矛盾，故此处直接消费 Stage 1-B 的段, 不再重切边界。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..config import (PRIMITIVE_TO_VERB, PRIMITIVES_REQUIRING_TARGET, Stage2Config,
                      starts_with_allowed_verb)
from ..data.types import Episode, Segment

# 预期夹爪闭合 / 张开的原语 (transitional 的 pick_up/grasp/place 跨越切换帧, 跳过极性检查)
_EXPECT_CLOSED = {"transport", "hold", "push", "pull", "press", "wipe",
                  "open", "close", "rotate", "insert", "pour"}
_EXPECT_OPEN = {"reach", "retract"}


def rule_check(segment: Segment, stage1b: Dict, cfg: Stage2Config) -> Tuple[bool, str]:
    """三项规则检查, 全部通过才算 valid。gripper_binary 约定 1=open / 0=close。"""
    prim = segment.primitive_label or stage1b["primitive_by_frame"].get(segment.start_frame, "unknown")
    binary = stage1b["gripper_binary"]
    seg_grip = binary[segment.start_frame:segment.end_frame + 1]
    open_ratio = float(seg_grip.mean()) if len(seg_grip) else 1.0

    # 规则 1: primitive 与夹爪状态一致性
    if prim in _EXPECT_CLOSED and open_ratio > 0.7:
        return False, f"gripper mostly open but primitive '{prim}' expects closed"
    if prim in _EXPECT_OPEN and open_ratio < 0.3:
        return False, f"gripper mostly closed but primitive '{prim}' expects open"

    # 规则 2: 时长合理性
    duration = segment.end_frame - segment.start_frame + 1
    if duration < cfg.min_segment_frames:
        return False, f"segment too short: {duration} frames"
    if duration > cfg.max_segment_frames:
        return False, f"segment too long: {duration} frames"

    # 规则 3: subtask_text 动词与 primitive 一致性
    expected_verb = PRIMITIVE_TO_VERB.get(prim)
    if expected_verb and not segment.subtask_text.strip().lower().startswith(expected_verb):
        return False, f"verb mismatch: expected '{expected_verb}', got '{segment.subtask_text[:20]}'"

    return True, ""


def run_stage2(episode: Episode, stage1b: Dict, cfg: Stage2Config) -> Dict:
    """对 Stage 1-B 的段做规则过滤并定置信度。segments 直接来自 Stage 1-B/1-C。"""
    segments: List[Segment] = stage1b["segments"]

    rule_failures = []
    for i, seg in enumerate(segments):
        ok, reason = rule_check(seg, stage1b, cfg)
        if not ok:
            rule_failures.append({"index": i, "primitive": seg.primitive_label, "reason": reason})

    if rule_failures:
        confidence = "Flagged"
    elif stage1b.get("has_nonprehensile_fill"):
        confidence = "Bronze"  # 边界部分来自 RDP 填充 (非抓持轨迹), 建议抽检
    else:
        confidence = "Gold"    # 边界全部来自夹爪事件帧且规则全通过

    result = {
        "segments": segments,
        "confidence": confidence,
        "rule_failures": rule_failures,
    }
    episode.meta["stage2"] = {"confidence": confidence, "rule_failures": rule_failures}
    return result
