"""数据集 loader 统一接口与工厂。

所有数据集 loader 实现 `DatasetLoader` Protocol (产出符合 `Episode` 契约的轨迹)，
`build_loader` 按 `DatasetConfig.type` 分发。新增数据集 = 实现 Protocol + 在此注册一行，
CLI / pipeline 不需要改动。
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Protocol, runtime_checkable

from .types import Episode


@runtime_checkable
class DatasetLoader(Protocol):
    """统一 loader 接口。"""

    def __len__(self) -> int: ...

    def load_episode(self, i: int) -> Episode: ...

    def iter_episodes(self, indices: Optional[List[int]] = None) -> Iterator[Episode]: ...


def build_loader(spec) -> DatasetLoader:
    """根据 DatasetConfig 构建 loader (惰性 import 重依赖)。"""
    dtype = (spec.type or "synthetic").lower()
    if dtype == "synthetic":
        from .synthetic import SyntheticLoader
        return SyntheticLoader(n_episodes=spec.n or 5, with_images=spec.with_images)
    if dtype == "lerobot":
        if not spec.root:
            raise ValueError("dataset.type=lerobot 需指定 root")
        from .lerobot_loader import LeRobotLoader
        return LeRobotLoader(
            root=spec.root,
            state_key=spec.state_key,
            gripper_key=spec.gripper_key,
            gripper_dim=spec.gripper_dim,
            eef_xyz_dims=spec.eef_xyz_dims,
            image_camera=spec.image_camera,
            gripper_open_is_high=spec.gripper_open_is_high,
            gripper_min=spec.gripper_min,
            gripper_max=spec.gripper_max,
        )
    raise ValueError(f"未知 dataset type: {dtype}")
