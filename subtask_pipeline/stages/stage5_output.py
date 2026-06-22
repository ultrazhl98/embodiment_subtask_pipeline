"""Stage 5 — 质量分级输出。

整合各 Stage 产出为单条标准化记录，按 confidence 写 loss_weight，
与 LeRobot episode 兼容 (作为额外 metadata，不改原始 action/observation)。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..data.types import AnchorObject, Episode, Segment


def assemble_record(episode: Episode, anchors: Sequence[AnchorObject], segments: List[Segment],
                    confidence: str, branch: str, stage3_meta: Dict,
                    loss_weights: Dict[str, float]) -> Dict:
    """产出单条最终记录 (与 doc Stage 5 输出格式一致)。"""
    # Stage 3 自检失败 (重试用尽仍 fail) -> 降级 Bronze
    final_conf = confidence
    if not stage3_meta.get("self_check_passed", True) and \
            stage3_meta.get("self_check_retries", 0) >= 2:
        final_conf = "Bronze"

    record = {
        "episode_id": episode.episode_id,
        "task_instruction": episode.task_instruction,
        "anchor_objects": [a.to_dict() for a in anchors],
        "segments": [s.to_dict() for s in segments],
        "confidence": final_conf,
        "loss_weight": loss_weights.get(final_conf, 0.2),
        "branch": branch,
        "gripper_quality_score": episode.meta.get("gripper_quality_score"),
        "annotation_meta": {
            "stage0": {
                "gripper_label": episode.meta.get("gripper_label"),
                "length_flag": episode.meta.get("length_flag"),
                "image_quality_flag": episode.meta.get("image_quality_flag"),
            },
            "stage2_branch": branch,
            "stage2_delta": episode.meta.get("stage2", {}).get("delta"),
            "self_check_passed": stage3_meta.get("self_check_passed"),
            "self_check_retries": stage3_meta.get("self_check_retries"),
            "description_gen_failed": stage3_meta.get("description_gen_failed"),
            "source": episode.meta.get("source"),
        },
    }
    return record
