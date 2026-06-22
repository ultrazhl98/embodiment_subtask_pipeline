"""Stage 1-A — 语义锚点提取 (路径 A，参考 LaRA-VLA)。

从首帧 (跳过模糊/遮挡帧) + 任务指令调用 VLM，提取被操作物体锚点。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..config import Stage0Config
from ..data.types import AnchorObject, Episode
from ..llm.base import BaseClient
from ..llm.prompts import build_anchor_prompt


def select_first_clear_frame(episode: Episode, blur_threshold: float = 100.0,
                             max_skip: int = 20) -> int:
    """从第 0 帧起，取首个中心区域拉普拉斯方差达标的清晰帧。"""
    if not episode.has_images:
        return 0
    try:
        import cv2
    except Exception:
        cv2 = None
    n = min(episode.num_frames, max_skip)
    best_idx, best_var = 0, -1.0
    for i in range(n):
        img = episode.image(i)
        gray = img[..., :3].mean(axis=2) if img.ndim == 3 else img
        h, w = gray.shape
        center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
        if cv2 is not None:
            var = float(cv2.Laplacian(center, cv2.CV_64F).var())
        else:
            var = float(np.abs(np.diff(center, n=2, axis=0)).var())
        if var >= blur_threshold:
            return i
        if var > best_var:
            best_idx, best_var = i, var
    return best_idx


def _validate(parsed) -> Optional[str]:
    if not isinstance(parsed, dict) or "anchor_objects" not in parsed:
        return "missing 'anchor_objects'"
    objs = parsed["anchor_objects"]
    if not isinstance(objs, list) or not objs:
        return "'anchor_objects' must be a non-empty array"
    has_source = False
    for o in objs:
        if not isinstance(o, dict) or o.get("role") not in ("source", "target"):
            return "each object needs role source|target"
        desc = (o.get("description") or "").strip()
        if not desc or len(desc.split()) > 20:
            return "description empty or >20 words"
        if o["role"] == "source":
            has_source = True
    if not has_source:
        return "at least one source required"
    return None


def run_stage1a(episode: Episode, client: BaseClient,
                blur_threshold: float = 100.0) -> List[AnchorObject]:
    system, user = build_anchor_prompt(episode.task_instruction)
    frame_idx = select_first_clear_frame(episode, blur_threshold)
    img = episode.image(frame_idx)
    images = [img] if img is not None else None

    parsed = client.generate_json(system, user, images=images, validator=_validate)
    anchors = [AnchorObject(role=o["role"], description=o["description"].strip())
               for o in parsed["anchor_objects"]]
    episode.meta["anchor_frame_idx"] = frame_idx
    episode.meta["anchor_objects"] = [a.to_dict() for a in anchors]
    return anchors
