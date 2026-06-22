"""合成轨迹生成器。

用于在没有真实数据集 / API key 的情况下端到端验证产线。
生成的轨迹带有可控的 gripper 切换次数、EEF 运动方向与图像，
使 Stage 0/1-B/2 的确定性逻辑可被真实驱动。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .types import Episode


def _render_frame(t: float, eef: np.ndarray, h: int = 256, w: int = 256, noisy: bool = False) -> np.ndarray:
    """画一个简单场景：渐变背景 + 跟随 EEF 的方块，制造帧间像素差异。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = 60
    img[:, :, 1] = 80
    img[:, :, 2] = 100
    # 用 EEF 的 xy 映射到图像坐标驱动一个 "gripper" 方块
    cx = int((eef[0] + 1) / 2 * (w - 40)) + 20
    cy = int((eef[1] + 1) / 2 * (h - 40)) + 20
    cx = int(np.clip(cx, 10, w - 10))
    cy = int(np.clip(cy, 10, h - 10))
    img[cy - 8:cy + 8, cx - 8:cx + 8] = (200, 50, 50)
    # 固定的 target 容器
    img[h - 40:h - 10, w - 50:w - 20] = (50, 200, 80)
    if noisy:
        img = np.clip(img.astype(np.int16) + np.random.randint(-20, 20, img.shape), 0, 255).astype(np.uint8)
    return img


def make_synthetic_episode(
    episode_id: str,
    task_instruction: str = "put the red cup into the open drawer",
    n_subtasks: int = 3,
    frames_per_subtask: int = 50,
    gripper_noisy: bool = False,
    seed: int = 0,
    with_images: bool = True,
) -> Episode:
    """生成一条 n_subtasks 段的轨迹。

    每段给一个主运动方向并在段边界切换 gripper，使物理分段数 == n_subtasks。
    gripper_noisy=True 时在段内注入高频抖动，触发 Stage 0 的 noisy 标记。
    """
    rng = np.random.RandomState(seed)
    n = n_subtasks * frames_per_subtask

    # 每段一个目标方向，段内线性插值，制造可分辨的 movement primitive
    directions = [
        np.array([0.6, 0.2, -0.5]),   # reach forward-down
        np.array([0.0, 0.0, 0.7]),    # lift up
        np.array([0.5, -0.6, -0.3]),  # move right-down
        np.array([-0.4, 0.4, 0.2]),
        np.array([0.3, 0.3, 0.3]),
    ]
    xyz = np.zeros((n, 3), dtype=np.float32)
    pos = np.array([-0.5, -0.5, 0.0], dtype=np.float32)
    gripper = np.zeros(n, dtype=np.float32)
    state = 1.0  # 1=open, 0=close, 段边界交替

    for s in range(n_subtasks):
        d = directions[s % len(directions)] / max(1, frames_per_subtask)
        for i in range(frames_per_subtask):
            idx = s * frames_per_subtask + i
            pos = pos + d + rng.randn(3) * 0.005
            xyz[idx] = pos
            gripper[idx] = state
        state = 0.0 if state == 1.0 else 1.0  # 段边界翻转 gripper

    if gripper_noisy:
        # 注入高频抖动：随机若干帧翻转 gripper
        flip_idx = rng.choice(n, size=max(3, n // 15), replace=False)
        gripper[flip_idx] = 1.0 - gripper[flip_idx]

    # states: xyz + 3 维旋转(置0) + gripper
    states = np.concatenate([xyz, np.zeros((n, 3), dtype=np.float32), gripper[:, None]], axis=1)

    images: Optional[List[np.ndarray]] = None
    image_fn = None
    if with_images:
        images = [_render_frame(i / n, xyz[i], noisy=gripper_noisy) for i in range(n)]
        image_fn = lambda idx, _imgs=images: _imgs[idx]

    return Episode(
        episode_id=episode_id,
        task_instruction=task_instruction,
        num_frames=n,
        states=states,
        gripper=gripper,
        image_fn=image_fn,
        meta={"source": "synthetic", "true_n_subtasks": n_subtasks},
    )


def make_synthetic_dataset(n_episodes: int = 5, with_images: bool = True) -> List[Episode]:
    """生成一个小型合成数据集，混合 reliable / noisy 轨迹与不同段数。"""
    tasks = [
        ("put the red cup into the open drawer", 3),
        ("grasp the blue block", 2),
        ("pick up the apple and put it in the bowl", 3),
        ("open the cabinet door and put the bowl inside", 4),
        ("push the green button", 2),
    ]
    eps = []
    for i in range(n_episodes):
        task, n_sub = tasks[i % len(tasks)]
        eps.append(make_synthetic_episode(
            episode_id=f"synthetic_ep_{i:05d}",
            task_instruction=task,
            n_subtasks=n_sub,
            gripper_noisy=(i % 4 == 3),  # 每 4 条混入一条 noisy
            seed=i,
            with_images=with_images,
        ))
    return eps
