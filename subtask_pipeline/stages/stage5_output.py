"""Stage 5 — 质量分级输出。

整合各 Stage 产出为单条标准化记录, 按 confidence 写 loss_weight, 并补充 per-frame
进度信号 (progress) 用于 Steerable VLA 的 progress head 训练 (参考 TAPT)。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..data.types import Episode, Segment


def add_progress_signal(segments: List[Segment]) -> List[Segment]:
    """段内线性填充进度信号 0 -> 1。"""
    for seg in segments:
        n = seg.end_frame - seg.start_frame + 1
        seg.progress = [round(float(p), 4) for p in np.linspace(0.0, 1.0, max(n, 1))]
    return segments


def assemble_record(episode: Episode, segments: List[Segment], confidence: str,
                    loss_weights: Dict[str, float]) -> Dict:
    """产出单条最终记录。"""
    add_progress_signal(segments)
    stage1b = episode.meta.get("stage1b", {})
    stage1c = episode.meta.get("stage1c", {})
    stage2 = episode.meta.get("stage2", {})

    record = {
        "episode_id": episode.episode_id,
        "task_instruction": episode.task_instruction,
        "global_summary": episode.meta.get("global_summary"),
        "segments": [s.to_dict() for s in segments],
        "confidence": confidence,
        "loss_weight": loss_weights.get(confidence, 0.0),
        "gripper_quality_score": episode.meta.get("gripper_quality_score"),
        "annotation_meta": {
            "stage0": {
                "gripper_label": episode.meta.get("gripper_label"),
                "length_flag": episode.meta.get("length_flag"),
                "image_quality_flag": episode.meta.get("image_quality_flag"),
            },
            "stage1b": {
                "n_physical": stage1b.get("N_physical"),
                "event_frames": stage1b.get("event_frames"),
                "has_nonprehensile_fill": confidence == "Bronze",
            },
            "stage1c": {
                "description_gen_failed": stage1c.get("description_gen_failed"),
                "used_vlm": stage1c.get("used_vlm"),
            },
            "stage2": {
                "confidence": stage2.get("confidence"),
                "rule_failures": stage2.get("rule_failures", []),
            },
            "source": episode.meta.get("source"),
        },
    }
    return record
