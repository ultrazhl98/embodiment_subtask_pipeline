"""Stage 1-B — 物理信号分割。

从 proprio (gripper + EEF) 提取候选 segment 边界与 movement primitive。
movement primitive 提取移植并扩展自 ECoT
`scripts/generate_embodied_data/primitive_movements.py`。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..config import Stage1bConfig
from ..data.types import Episode

# ---------------------------------------------------------------------------
# Movement primitive (移植自 ECoT primitive_movements.py)
# ---------------------------------------------------------------------------

_MOVE_NAMES = [
    {-1: "backward", 0: None, 1: "forward"},
    {-1: "right", 0: None, 1: "left"},
    {-1: "down", 0: None, 1: "up"},
]


def describe_move(move_vec: np.ndarray) -> str:
    """把 xyz 方向向量 (-1/0/1) 翻译为简短文字 (ECoT describe_move 的精简版)。"""
    parts = [_MOVE_NAMES[i][int(move_vec[i])] for i in range(3)]
    parts = [p for p in parts if p is not None]
    return ("move " + " ".join(parts)) if parts else "stop"


def classify_movement(window: np.ndarray, threshold: float = 0.03):
    """ECoT classify_movement: 取窗口首尾 xyz 位移，阈值化为方向向量。"""
    diff = window[-1] - window[0]
    s = np.sum(np.abs(diff[:3]))
    if s > 3 * threshold:
        diff[:3] = diff[:3] * (3 * threshold / s)
    move_vec = 1 * (diff[:3] > threshold) - 1 * (diff[:3] < -threshold)
    return describe_move(move_vec), move_vec


def per_frame_primitives(xyz: np.ndarray, cfg: Stage1bConfig) -> List[str]:
    """逐帧 movement primitive 文字 (供 Stage 2 协商路径)。"""
    n = len(xyz)
    if n < 2:
        return ["stop"] * n
    windows = [xyz[i:i + 4] for i in range(n - 1)]
    prims = [classify_movement(w, cfg.primitive_move_threshold)[0] for w in windows]
    prims.append(prims[-1])
    return prims


# ---------------------------------------------------------------------------
# Gripper 离散化与分段
# ---------------------------------------------------------------------------


def _median_filter(signal: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        return signal
    pad = window // 2
    padded = np.pad(signal, pad, mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(len(signal))])


def discretize_gripper(signal: np.ndarray, gripper_label: str, cfg: Stage1bConfig):
    """reliable 直接二值化; noisy 先中值滤波。返回 (binary, smoothing_seg_delta)。"""
    raw_bin = (signal > cfg.binarize_threshold).astype(np.int8)
    raw_segs = 1 + int(np.sum(np.diff(raw_bin) != 0)) if len(raw_bin) > 1 else 1
    if gripper_label == "noisy":
        smoothed = _median_filter(signal, cfg.median_filter_window)
        binary = (smoothed > cfg.binarize_threshold).astype(np.int8)
    else:
        binary = raw_bin
    smooth_segs = 1 + int(np.sum(np.diff(binary) != 0)) if len(binary) > 1 else 1
    return binary, raw_segs - smooth_segs


def _segments_from_binary(binary: np.ndarray):
    """由二值序列得到 [(start, end, state)] 闭区间分段。"""
    n = len(binary)
    if n == 0:
        return []
    bounds = np.where(np.diff(binary) != 0)[0] + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds - 1, [n - 1]])
    return [(int(s), int(e), "open" if binary[s] == 1 else "close")
            for s, e in zip(starts, ends)]


def _merge_short_segments(segments, min_len: int):
    """把长度 < min_len 的段并入较短的相邻段，保持覆盖连续。"""
    segs = [list(s) for s in segments]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (s, e, _state) in enumerate(segs):
            if e - s + 1 < min_len:
                # 选并入左还是右 (取相邻较短段方向，倾向并入左)
                if i == 0:
                    j = 1
                elif i == len(segs) - 1:
                    j = i - 1
                else:
                    left_len = segs[i - 1][1] - segs[i - 1][0]
                    right_len = segs[i + 1][1] - segs[i + 1][0]
                    j = i - 1 if left_len <= right_len else i + 1
                lo, hi = min(i, j), max(i, j)
                merged = [min(segs[lo][0], segs[hi][0]),
                          max(segs[lo][1], segs[hi][1]),
                          segs[lo][2]]  # 沿用靠前段的 gripper_state
                segs = segs[:lo] + [merged] + segs[hi + 1:]
                changed = True
                break
    return [tuple(s) for s in segs]


# ---------------------------------------------------------------------------
# Segment movement primitive (段级，≤6 词)
# ---------------------------------------------------------------------------


def _segment_primitive(xyz_seg: np.ndarray, cfg: Stage1bConfig, fast_speed: float) -> str:
    """段级主运动方向 + 速度修饰 (reach/place)，控制在 6 词内。"""
    if len(xyz_seg) < 2:
        return "hold position"
    disp = xyz_seg[-1] - xyz_seg[0]
    move_vec = 1 * (disp > cfg.primitive_move_threshold) - 1 * (disp < -cfg.primitive_move_threshold)
    dirs = [_MOVE_NAMES[i][int(move_vec[i])] for i in range(3)]
    dirs = [d for d in dirs if d is not None][:2]  # 最多两个方向词
    speed = float(np.linalg.norm(np.diff(xyz_seg[:, :3], axis=0), axis=1).mean())
    verb = "reach" if speed >= fast_speed else "move slowly"
    if not dirs:
        return "hold and grip"
    return f"{verb} " + " ".join(dirs)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def run_stage1b(episode: Episode, cfg: Stage1bConfig) -> Dict:
    signal = episode.gripper_signal()
    xyz = episode.eef_xyz()
    gripper_label = episode.meta.get("gripper_label", "reliable")

    binary, smooth_delta = discretize_gripper(signal, gripper_label, cfg)
    raw_segments = _segments_from_binary(binary)
    segments = _merge_short_segments(raw_segments, cfg.min_segment_frames)

    # 边界对齐: 首段从 0 开始，末段到最后一帧
    if segments:
        segments[0] = (0, segments[0][1], segments[0][2])
        segments[-1] = (segments[-1][0], episode.num_frames - 1, segments[-1][2])

    # 速度阈值用于 reach/place 区分
    frame_speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if len(xyz) > 1 else np.array([0.0])
    fast_speed = float(np.quantile(frame_speed, cfg.fast_speed_quantile)) if len(frame_speed) else 0.0

    seg_dicts = []
    primitives = []
    for s, e, state in segments:
        prim = _segment_primitive(xyz[s:e + 1], cfg, fast_speed)
        seg_dicts.append({"start_frame": s, "end_frame": e, "gripper_state": state})
        primitives.append(prim)

    result = {
        "N_physical": len(segments),
        "segments": seg_dicts,
        "primitives": primitives,
        "frame_primitives": per_frame_primitives(xyz, cfg),
        "gripper_smoothing_seg_delta": int(smooth_delta),
    }
    episode.meta["stage1b"] = {k: result[k] for k in ("N_physical", "segments", "primitives")}
    return result
