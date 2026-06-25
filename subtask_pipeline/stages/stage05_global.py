"""Stage 0.5 — 全局视频理解 (替代旧 Stage 1-A 首帧锚点)。

均匀采样覆盖全程的多帧 + 任务指令, 调 VLM 产出 global_summary:
task_intent / objects / key_events / scene_context, 写入 episode.meta["global_summary"]。
- objects 供 Stage 1-C 做 per-segment 物体指称约束 (替代首帧 anchor_objects)。
- 不依赖首帧, 解决机器人移动入场 / 视角变化场景下的锚点失效。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..config import Stage05Config
from ..data.types import Episode
from ..llm.base import BaseClient, LLMResponseError
from ..llm.prompts import build_global_summary_prompt


def sample_frame_indices(num_frames: int, count: int) -> List[int]:
    """均匀采样覆盖全程的帧索引。"""
    count = max(1, min(count, num_frames))
    return sorted(set(np.linspace(0, num_frames - 1, count).astype(int).tolist()))


def _make_validator(max_objects: int):
    def validate(parsed) -> Optional[str]:
        if not isinstance(parsed, dict):
            return "output must be a JSON object"
        if not (parsed.get("task_intent") or "").strip():
            return "missing 'task_intent'"
        objs = parsed.get("objects")
        if not isinstance(objs, list) or not objs:
            return "'objects' must be a non-empty array"
        if len(objs) > max_objects:
            return f"at most {max_objects} objects allowed"
        for o in objs:
            if not isinstance(o, dict) or not (o.get("description") or "").strip():
                return "each object needs a non-empty description"
        return None
    return validate


def run_stage05(episode: Episode, client: BaseClient, cfg: Stage05Config) -> Optional[Dict]:
    """产出全局摘要并写入 episode.meta。enable=False 或无图像时返回 None。"""
    if not cfg.enable or not episode.has_images:
        episode.meta["global_summary"] = None
        return None

    idxs = sample_frame_indices(episode.num_frames, cfg.sample_count)
    images = [episode.image(i) for i in idxs]
    system, user = build_global_summary_prompt(episode.task_instruction, cfg.max_objects)
    try:
        parsed = client.generate_json(system, user, images=images,
                                      validator=_make_validator(cfg.max_objects))
    except LLMResponseError:
        episode.meta["global_summary"] = None
        return None

    summary = {
        "task_intent": parsed["task_intent"].strip(),
        "objects": [{"role": o.get("role", "object"), "description": o["description"].strip()}
                    for o in parsed["objects"][:cfg.max_objects]],
        "key_events": [str(e).strip() for e in parsed.get("key_events", []) if str(e).strip()],
        "scene_context": (parsed.get("scene_context") or "").strip(),
        "sample_frames": idxs,
    }
    episode.meta["global_summary"] = summary
    return summary
