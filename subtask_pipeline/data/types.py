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
    """单条机器人操控轨迹的标准化表示 (所有 loader 须产出符合本契约的 Episode)。

    统一输入契约 (下游 Stage 均面向此编程)
    --------------------------------------
    episode_id       : 非空唯一字符串。
    task_instruction : L1 任务指令文本 (可空字符串，建议非空)。
    num_frames       : > 0，且与 states/gripper 第 0 维一致。
    states           : (N, D) float，**前 3 维为 EEF xyz**，线性尺度
                       (同一数据集内单位一致即可；阈值类参数按该尺度配置)。
    gripper          : (N,) float，**归一到 [0,1]，1=张开 / 0=闭合**。
                       loader 负责把原始量纲/极性转成此约定 (见 LeRobotLoader 的
                       gripper_min/gripper_max/gripper_open_is_high)。为 None 时
                       退回从 states 末维推断 (不推荐，仅兼容旧路径)。
    image_fn         : idx -> np.ndarray(H,W,3) uint8 RGB，或 None (无图走纯 proprio)。
    meta             : dict，至少含 'source'；其余透传，并由各 Stage 写入标记。
    """

    episode_id: str
    task_instruction: str
    num_frames: int
    states: Optional[np.ndarray] = None
    gripper: Optional[np.ndarray] = None
    image_fn: Optional[Callable[[int], np.ndarray]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- 契约校验 -----------------------------------------------------------
    def validate(self) -> None:
        """校验是否满足统一输入契约，违反则 raise ValueError。

        由 pipeline 逐条调用；单条失败收集到 failures，不崩溃整批。
        """
        if not self.episode_id:
            raise ValueError("episode_id 不能为空")
        if self.num_frames <= 0:
            raise ValueError(f"episode {self.episode_id} num_frames 必须 > 0")
        if self.states is None and self.gripper is None:
            raise ValueError(f"episode {self.episode_id} states 与 gripper 不能同时为 None")

        if self.states is not None:
            states = np.asarray(self.states)
            if states.ndim != 2 or states.shape[1] < 3:
                raise ValueError(
                    f"episode {self.episode_id} states 需为 (N, D>=3)，实际 {states.shape}")
            if states.shape[0] != self.num_frames:
                raise ValueError(
                    f"episode {self.episode_id} states 行数 {states.shape[0]} != num_frames {self.num_frames}")
            if np.isnan(states).all():
                raise ValueError(f"episode {self.episode_id} states 全为 NaN")

        if self.gripper is not None:
            gripper = np.asarray(self.gripper).reshape(-1)
            if gripper.shape[0] != self.num_frames:
                raise ValueError(
                    f"episode {self.episode_id} gripper 长度 {gripper.shape[0]} != num_frames {self.num_frames}")
            finite = gripper[np.isfinite(gripper)]
            if finite.size == 0:
                raise ValueError(f"episode {self.episode_id} gripper 全为 NaN")
            # 归一约定 [0,1]，留一点容差应对边界噪声
            if finite.min() < -0.05 or finite.max() > 1.05:
                raise ValueError(
                    f"episode {self.episode_id} gripper 超出 [0,1] 约定 "
                    f"(min={finite.min():.3f}, max={finite.max():.3f})；loader 应先归一/翻转极性")

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
