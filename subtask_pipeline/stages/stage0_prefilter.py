"""Stage 0 — 轨迹预过滤。

对每条轨迹打质量标记 (gripper_quality_score / gripper_label / length_flag /
image_quality_flag)，不做硬过滤。长度异常需要数据集级统计，故拆成两步:
逐条计算 gripper / 图像标记，数据集级回填 length_flag。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..config import Stage0Config
from ..data.types import Episode


# ---------------------------------------------------------------------------
# 1. Gripper 信号质量
# ---------------------------------------------------------------------------


def _binarize(signal: np.ndarray, threshold: float) -> np.ndarray:
    return (signal > threshold).astype(np.int8)


def _switch_points(binary: np.ndarray) -> np.ndarray:
    """返回状态切换发生的下标 (切换后第一帧)。"""
    if len(binary) < 2:
        return np.array([], dtype=int)
    return np.where(np.diff(binary) != 0)[0] + 1


def gripper_quality(signal: np.ndarray, cfg: Stage0Config) -> Dict:
    """计算 gripper 质量评分与标签。

    综合两个负面信号:
    - 高频抖动: 任意 `jitter_window` 帧窗口内切换次数 > `jitter_max_switches`
    - 无效切换: 切换后不足 `min_hold_frames` 即切回的跳变占比
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    n = len(signal)
    binary = _binarize(signal, cfg.gripper_binarize_threshold)
    switches = _switch_points(binary)
    n_switches = len(switches)

    # -- 高频抖动比例 --
    jitter_windows = 0
    total_windows = max(1, n - cfg.jitter_window + 1)
    if n >= cfg.jitter_window:
        diff = (np.diff(binary) != 0).astype(int)
        # 每个窗口内的切换数 = 窗口内 diff 之和
        csum = np.concatenate([[0], np.cumsum(diff)])
        for start in range(total_windows):
            end = start + cfg.jitter_window - 1
            sw = csum[min(end, len(csum) - 1)] - csum[start]
            if sw > cfg.jitter_max_switches:
                jitter_windows += 1
    jitter_ratio = jitter_windows / total_windows

    # -- 无效切换比例 (段长 < min_hold_frames) --
    seg_bounds = np.concatenate([[0], switches, [n]])
    seg_lengths = np.diff(seg_bounds)
    if len(seg_lengths) > 0:
        invalid = np.sum(seg_lengths < cfg.min_hold_frames)
        invalid_ratio = invalid / len(seg_lengths)
    else:
        invalid_ratio = 0.0

    score = float(np.clip(1.0 - 0.6 * jitter_ratio - 0.7 * invalid_ratio, 0.0, 1.0))
    label = "reliable" if score >= cfg.gripper_reliable_threshold else "noisy"
    return {
        "gripper_quality_score": round(score, 4),
        "gripper_label": label,
        "n_switches": int(n_switches),
        "jitter_ratio": round(float(jitter_ratio), 4),
        "invalid_switch_ratio": round(float(invalid_ratio), 4),
    }


# ---------------------------------------------------------------------------
# 2. 图像质量
# ---------------------------------------------------------------------------


def _laplacian_var(gray: np.ndarray) -> float:
    try:
        import cv2
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        # 退化实现: 二阶差分方差
        lap = np.abs(np.diff(gray.astype(np.float64), n=2, axis=0)).var()
        return float(lap)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return img[..., :3].mean(axis=2)
    return img.astype(np.float64)


def image_quality(episode: Episode, cfg: Stage0Config) -> Dict:
    """检测亮度 / 模糊 / frozen frame。无图像时返回 skipped。"""
    if not episode.has_images:
        return {"image_quality_flag": "skipped"}

    n = episode.num_frames
    idxs = sorted(set(np.linspace(0, n - 1, cfg.image_sample_count).astype(int)))
    grays = [_to_gray(episode.image(i)) for i in idxs]

    brightness = [g.mean() for g in grays]
    blur = [_laplacian_var(g) for g in grays]

    too_dark = any(b < cfg.brightness_threshold for b in brightness)
    blurry = any(v < cfg.blur_threshold for v in blur)

    # frozen frame: 连续采样帧像素差均值过小
    frozen_run = 0
    max_frozen_run = 0
    for a, b in zip(grays[:-1], grays[1:]):
        if np.abs(a - b).mean() < cfg.frozen_diff_threshold:
            frozen_run += 1
            max_frozen_run = max(max_frozen_run, frozen_run)
        else:
            frozen_run = 0
    frozen = max_frozen_run >= cfg.frozen_run_length

    flag = "pass"
    reasons = []
    if too_dark:
        reasons.append("dark")
    if blurry:
        reasons.append("blurry")
    if frozen:
        reasons.append("frozen")
    if reasons:
        flag = "fail"

    return {
        "image_quality_flag": flag,
        "image_quality_reasons": reasons,
        "min_brightness": round(float(min(brightness)), 2),
        "min_blur_var": round(float(min(blur)), 2),
    }


# ---------------------------------------------------------------------------
# 3. 长度异常 (数据集级)
# ---------------------------------------------------------------------------


def compute_length_flags(lengths_by_task: Dict[str, List[int]], cfg: Stage0Config) -> Dict[str, Dict]:
    """按任务分组用 median ± k*MAD 标记长度异常，返回每组的阈值信息。"""
    stats = {}
    for task, lengths in lengths_by_task.items():
        arr = np.asarray(lengths, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median))) or 1.0
        lo = median - cfg.length_mad_factor * mad
        hi = median + cfg.length_mad_factor * mad
        stats[task] = {"median": median, "mad": mad, "lo": lo, "hi": hi}
    return stats


def length_flag(n_frames: int, task: str, length_stats: Dict[str, Dict]) -> str:
    s = length_stats.get(task)
    if s is None:
        return "normal"
    if n_frames < s["lo"]:
        return "too_short"
    if n_frames > s["hi"]:
        return "too_long"
    return "normal"


# ---------------------------------------------------------------------------
# 主入口 (逐条标记，length_flag 由 pipeline 数据集级回填)
# ---------------------------------------------------------------------------


def run_stage0(episode: Episode, cfg: Stage0Config) -> Dict:
    """对单条轨迹计算 gripper 与图像标记，写入 episode.meta。"""
    result = {}
    result.update(gripper_quality(episode.gripper_signal(), cfg))
    result.update(image_quality(episode, cfg))
    result["length_flag"] = "normal"  # 占位，数据集级回填
    episode.meta.update(result)
    return result
