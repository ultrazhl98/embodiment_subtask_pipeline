"""产线核心数据结构。

这些类型把 pipeline 与具体数据集格式 (LeRobot v2/v2.1, RLDS, 合成数据) 解耦：
每个加载器只需产出 `Episode`，下游所有 Stage 都面向 `Episode` 编程。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 锚点 / 分段
# ---------------------------------------------------------------------------


@dataclass
class AnchorObject:
    """Stage 1-A 产出的语义锚点物体。"""

    role: str  # "source" | "target"
    description: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "description": self.description}


@dataclass
class Segment:
    """一个 subtask 分段。各 Stage 逐步填充字段。"""

    subtask_text: str
    start_frame: int
    end_frame: int
    keyframe: Optional[int] = None
    completion_frame: Optional[int] = None
    gripper_state: Optional[str] = None  # Stage 1-B 物理分段时的 "open"/"close"
    grounding: Optional[Any] = None  # Stage 4 产出

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "subtask_text": self.subtask_text,
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
        }
        if self.keyframe is not None:
            d["keyframe"] = int(self.keyframe)
        if self.completion_frame is not None:
            d["completion_frame"] = int(self.completion_frame)
        d["grounding"] = self.grounding
        return d


# ---------------------------------------------------------------------------
# 轨迹
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """单条机器人操控轨迹的标准化表示。

    属性
    ----
    episode_id : 唯一标识
    task_instruction : 全局任务指令文字
    num_frames : 帧数
    states : (N, D) 末端执行器本体感受序列。约定前 3 维为 xyz，
             之后为旋转 (rpy 或 quat)，最后若 `gripper` 未单独给出则取末维为 gripper。
    gripper : (N,) gripper 连续状态序列 (可为 None，则从 states 推断)
    image_fn : 惰性取帧函数 idx -> np.ndarray(H,W,3) uint8 (可为 None)
    meta : 透传的原始 metadata，以及各 Stage 写入的标记
    """

    episode_id: str
    task_instruction: str
    num_frames: int
    states: Optional[np.ndarray] = None
    gripper: Optional[np.ndarray] = None
    image_fn: Optional[Callable[[int], np.ndarray]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- 取帧 ---------------------------------------------------------------
    def image(self, idx: int) -> Optional[np.ndarray]:
        """返回第 idx 帧图像 (H,W,3 uint8)，无图像数据时返回 None。"""
        if self.image_fn is None:
            return None
        idx = int(np.clip(idx, 0, self.num_frames - 1))
        return self.image_fn(idx)

    @property
    def has_images(self) -> bool:
        return self.image_fn is not None

    # -- gripper 序列 -------------------------------------------------------
    def gripper_signal(self) -> np.ndarray:
        """返回 gripper 连续状态序列 (N,)。"""
        if self.gripper is not None:
            return np.asarray(self.gripper, dtype=np.float32).reshape(-1)
        if self.states is not None:
            return np.asarray(self.states, dtype=np.float32)[:, -1].reshape(-1)
        raise ValueError(f"episode {self.episode_id} 既无 gripper 也无 states，无法获取 gripper 信号")

    def eef_xyz(self) -> np.ndarray:
        """返回末端执行器 xyz 轨迹 (N,3)。"""
        if self.states is None:
            raise ValueError(f"episode {self.episode_id} 无 states，无法获取 EEF 轨迹")
        return np.asarray(self.states, dtype=np.float32)[:, :3]
