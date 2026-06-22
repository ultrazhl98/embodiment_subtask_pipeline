"""LeRobot v2 / v2.1 数据集加载器。

LeRobot v2.1 数据集的典型磁盘布局::

    <root>/
        meta/
            info.json              # features schema, fps, total_episodes ...
            episodes.jsonl         # 每行: {"episode_index", "tasks", "length"}
            tasks.jsonl            # 每行: {"task_index", "task"}
        data/
            chunk-000/
                episode_000000.parquet   # 每行一帧，含 observation.state / action 等列
        videos/
            chunk-000/
                observation.images.<cam>/episode_000000.mp4

本加载器把每条 episode 转成统一的 `Episode`。图像帧从对应 mp4 惰性解码
(需 opencv)；若无视频则 image_fn 为 None，纯靠 proprio 走主链路。

注意: LeRobot 各数据集的列名/相机名差异较大，这里通过参数暴露可配置项，
并对常见命名做自动探测。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from .types import Episode


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _guess_chunk_dir(data_root: str, episode_index: int, chunk_size: int = 1000) -> str:
    return os.path.join(data_root, f"chunk-{episode_index // chunk_size:03d}")


class _VideoFrameReader:
    """惰性按帧解码 mp4，带最近一次解码缓存避免重复 seek。"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cap = None
        self._last_idx = -1

    def _ensure_open(self):
        if self._cap is None:
            import cv2
            self._cap = cv2.VideoCapture(self.video_path)

    def __call__(self, idx: int) -> Optional[np.ndarray]:
        import cv2
        self._ensure_open()
        if idx != self._last_idx + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        self._last_idx = idx
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class LeRobotLoader:
    """读取 LeRobot v2/v2.1 数据集。"""

    def __init__(
        self,
        root: str,
        state_key: str = "observation.state",
        gripper_key: Optional[str] = None,
        gripper_dim: int = -1,
        eef_xyz_dims: Optional[List[int]] = None,
        image_camera: Optional[str] = None,
        chunk_size: int = 1000,
    ):
        """参数
        ----
        state_key   : proprio 列名
        eef_xyz_dims: state 中末端 xyz 所在的 3 个维度下标 (如 austin_buds 为 [20,21,22])；
                      None 则默认取 state 前 3 维
        gripper_key : gripper 信号来源列 (可为 'action' 等其它列)；None 则取 state 的 gripper_dim
        gripper_dim : gripper 在该列向量中的下标 (如 action[6] 则为 6)
        """
        self.root = root
        self.state_key = state_key
        self.gripper_key = gripper_key
        self.gripper_dim = gripper_dim
        self.eef_xyz_dims = list(eef_xyz_dims) if eef_xyz_dims is not None else None
        self.image_camera = image_camera
        self.chunk_size = chunk_size

        meta_dir = os.path.join(root, "meta")
        with open(os.path.join(meta_dir, "info.json")) as f:
            self.info = json.load(f)
        self.episodes_meta = _read_jsonl(os.path.join(meta_dir, "episodes.jsonl"))
        tasks_path = os.path.join(meta_dir, "tasks.jsonl")
        self.tasks: Dict[int, str] = {}
        if os.path.exists(tasks_path):
            for row in _read_jsonl(tasks_path):
                self.tasks[row["task_index"]] = row["task"]

        self.chunk_size = self.info.get("chunks_size", chunk_size)
        # 自动探测相机
        if self.image_camera is None:
            self.image_camera = self._auto_detect_camera()

    def _auto_detect_camera(self) -> Optional[str]:
        feats = self.info.get("features", {})
        for name in feats:
            if name.startswith("observation.images"):
                return name.split("observation.images.")[-1]
        return None

    def __len__(self) -> int:
        return len(self.episodes_meta)

    def _resolve_task(self, ep_meta: dict) -> str:
        tasks = ep_meta.get("tasks")
        if isinstance(tasks, list) and tasks:
            return tasks[0]
        if "task_index" in ep_meta:
            return self.tasks.get(ep_meta["task_index"], "")
        return ep_meta.get("task", "")

    def load_episode(self, i: int) -> Episode:
        import pandas as pd  # 延迟导入

        ep_meta = self.episodes_meta[i]
        ep_idx = ep_meta.get("episode_index", i)
        chunk_dir = _guess_chunk_dir(os.path.join(self.root, "data"), ep_idx, self.chunk_size)
        parquet_path = os.path.join(chunk_dir, f"episode_{ep_idx:06d}.parquet")
        df = pd.read_parquet(parquet_path)

        full_state = np.stack(df[self.state_key].to_numpy()).astype(np.float32)
        n = len(full_state)
        # 末端 xyz: 指定维度则切片，否则取前 3 维 (Episode.eef_xyz 假设 xyz 在前 3 列)
        if self.eef_xyz_dims is not None:
            states = full_state[:, self.eef_xyz_dims]
        else:
            states = full_state

        # gripper: 可来自其它列 (如 action[6])
        if self.gripper_key and self.gripper_key in df.columns:
            col = np.stack(df[self.gripper_key].to_numpy()).astype(np.float32)
            gripper = (col[:, self.gripper_dim] if col.ndim == 2 else col).reshape(-1)
        else:
            gripper = full_state[:, self.gripper_dim].reshape(-1)

        image_fn = None
        if self.image_camera:
            video_path = os.path.join(
                self.root, "videos",
                f"chunk-{ep_idx // self.chunk_size:03d}",
                f"observation.images.{self.image_camera}",
                f"episode_{ep_idx:06d}.mp4",
            )
            if os.path.exists(video_path):
                image_fn = _VideoFrameReader(video_path)

        return Episode(
            episode_id=f"{os.path.basename(self.root.rstrip('/'))}_ep_{ep_idx:06d}",
            task_instruction=self._resolve_task(ep_meta),
            num_frames=n,
            states=states,
            gripper=gripper,
            image_fn=image_fn,
            meta={"source": "lerobot", "episode_index": ep_idx,
                  "fps": self.info.get("fps")},
        )

    def iter_episodes(self, indices: Optional[List[int]] = None):
        if indices is None:
            indices = range(len(self))
        for i in indices:
            yield self.load_episode(i)
