"""Stage 1-B — 事件帧提取 + 语义原语分类。

从 proprio (gripper + EEF) 切分轨迹并给每段一个**语义原语标签** (而非旧版的
纯运动学短语)。分割信号:
- 夹爪事件帧 (open<->close 切换): 抓持类任务的硬边界
- RDP 拐点 + 速度极小值: 夹爪全程闭合的非抓持任务 (push/wipe/...) 的补充边界

输出 (primitive_label, start_frame, end_frame) 列表 + 事件帧 + per-frame 信息，
供 Stage 1-C 填槽与 Stage 2 规则校验使用。
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..config import Stage1bConfig
from ..data.types import Episode, Segment


# ---------------------------------------------------------------------------
# Gripper 离散化
# ---------------------------------------------------------------------------


def _median_filter(signal: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        return signal
    pad = window // 2
    padded = np.pad(signal, pad, mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(len(signal))])


def discretize_gripper(signal: np.ndarray, gripper_label: str, cfg: Stage1bConfig) -> np.ndarray:
    """二值化夹爪信号 (1=open, 0=close)。noisy 轨迹先中值滤波。"""
    if gripper_label == "noisy":
        signal = _median_filter(signal, cfg.median_filter_window)
    return (signal > cfg.binarize_threshold).astype(np.int8)


def event_frames_from_binary(binary: np.ndarray) -> List[int]:
    """夹爪状态切换发生的帧索引 (切换后第一帧) = 抓持事件的硬锚点。"""
    if len(binary) < 2:
        return []
    return (np.where(np.diff(binary) != 0)[0] + 1).astype(int).tolist()


# ---------------------------------------------------------------------------
# RDP 拐点检测 (自包含实现, 不引入 rdp 依赖)
# ---------------------------------------------------------------------------


def _rdp_keep_mask(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker: 返回保留点的布尔掩码 (端点恒保留)。"""
    n = len(points)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    mask[0] = mask[-1] = True
    stack = [(0, n - 1)]
    while stack:
        s, e = stack.pop()
        if e <= s + 1:
            continue
        start, end = points[s], points[e]
        line = end - start
        line_len = float(np.linalg.norm(line))
        seg = points[s + 1:e] - start
        if line_len < 1e-12:
            dists = np.linalg.norm(seg, axis=1)
        else:
            dists = np.linalg.norm(np.cross(seg, line), axis=1) / line_len
        k = int(np.argmax(dists))
        if dists[k] > epsilon:
            keep = s + 1 + k
            mask[keep] = True
            stack.append((s, keep))
            stack.append((keep, e))
    return mask


def rdp_boundaries(xyz: np.ndarray, epsilon: float) -> List[int]:
    """RDP 保留的内部拐点索引 (运动方向突变点), 作为候选分割边界。"""
    if len(xyz) < 3:
        return []
    mask = _rdp_keep_mask(xyz, epsilon)
    mask[0] = mask[-1] = False  # 端点不作内部边界
    return sorted(np.where(mask)[0].tolist())


def _local_minima(speed: np.ndarray, window: int) -> List[int]:
    """速度局部极小值 (valley) -> EEF 在相位切换处减速的帧 (返回对应帧边界索引)。

    用 valley 判据 (严格下降后非上升) 避免在匀速平台上产生大量伪边界。
    """
    n = len(speed)
    out = []
    for i in range(1, n - 1):
        if speed[i] < speed[i - 1] and speed[i] <= speed[i + 1]:
            lo, hi = max(0, i - window), min(n, i + window + 1)
            if speed[i] <= speed[lo:hi].min() + 1e-12:
                out.append(i + 1)  # speed[i] 跨 frame i->i+1, 边界落在 i+1
    return out


def _enforce_min_gap(bounds: List[int], min_gap: int, n: int) -> List[int]:
    """过滤掉过近 (<min_gap) 或越界的边界。"""
    out: List[int] = []
    for b in sorted(set(bounds)):
        if b <= 0 or b >= n - 1:
            continue
        if out and b - out[-1] < min_gap:
            continue
        out.append(int(b))
    return out


def detect_nonprehensile_boundaries(xyz: np.ndarray, binary: np.ndarray,
                                    cfg: Stage1bConfig) -> List[int]:
    """对夹爪几乎全程闭合的非抓持轨迹, 用 RDP + 速度突变补充分割边界。"""
    if not cfg.use_rdp_for_nonprehensile or len(xyz) < 3:
        return []
    open_ratio = float(binary.mean()) if len(binary) else 1.0  # 1=open
    if open_ratio > cfg.nonprehensile_open_ratio_threshold:
        return []  # 含足够 open 帧 -> 抓持任务, 由夹爪事件分割即可
    bounds = set(rdp_boundaries(xyz, cfg.rdp_epsilon))
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    bounds |= set(_local_minima(speed, cfg.speed_minima_window))
    return _enforce_min_gap(sorted(bounds), cfg.min_segment_frames, len(xyz))


# ---------------------------------------------------------------------------
# 切分与短段合并
# ---------------------------------------------------------------------------


def _intervals_from_cuts(cuts: List[int], n: int):
    """由内部切点得到 [(start, end)] 闭区间, 覆盖 0..n-1。"""
    pts = [0] + [c for c in sorted(set(cuts)) if 0 < c < n] + [n]
    return [(pts[i], pts[i + 1] - 1) for i in range(len(pts) - 1)]


def _merge_short_segments(segments, binary, min_len: int):
    """长度 < min_len 的段并入较短相邻段, 保持覆盖连续。返回 [(s, e)]。"""
    segs = [list(s) for s in segments]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, (s, e) in enumerate(segs):
            if e - s + 1 < min_len:
                if i == 0:
                    j = 1
                elif i == len(segs) - 1:
                    j = i - 1
                else:
                    left_len = segs[i - 1][1] - segs[i - 1][0]
                    right_len = segs[i + 1][1] - segs[i + 1][0]
                    j = i - 1 if left_len <= right_len else i + 1
                lo, hi = min(i, j), max(i, j)
                segs = segs[:lo] + [[segs[lo][0], segs[hi][1]]] + segs[hi + 1:]
                changed = True
                break
    return [(int(s), int(e)) for s, e in segs]


# ---------------------------------------------------------------------------
# 语义原语推断
# ---------------------------------------------------------------------------


def infer_semantic_primitive(gripper_state: str, is_close_event: bool, is_open_event: bool,
                             disp_norm: float, disp_vertical: float, cfg: Stage1bConfig) -> str:
    """由夹爪状态/事件 + 段内位移推断语义原语标签。"""
    if is_close_event:
        return "pick_up" if disp_vertical > cfg.lift_threshold else "grasp"
    if is_open_event:
        return "place"
    if gripper_state == "close":
        return "transport" if disp_norm > cfg.transport_disp_threshold else "hold"
    if gripper_state == "open":
        return "reach"
    return "unknown"


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def run_stage1b(episode: Episode, cfg: Stage1bConfig) -> Dict:
    signal = episode.gripper_signal()
    xyz = episode.eef_xyz()
    n = episode.num_frames
    gripper_label = episode.meta.get("gripper_label", "reliable")

    binary = discretize_gripper(signal, gripper_label, cfg)
    event_frames = event_frames_from_binary(binary)
    nonpre_bounds = detect_nonprehensile_boundaries(xyz, binary, cfg)

    cuts = sorted(set(event_frames) | set(nonpre_bounds))
    intervals = _intervals_from_cuts(cuts, n)
    intervals = _merge_short_segments(intervals, binary, cfg.min_segment_frames)

    event_set = set(event_frames)
    segments: List[Segment] = []
    primitives: List[str] = []
    primitive_by_frame: Dict[int, str] = {}
    boundary_sources: List[str] = []

    for s, e in intervals:
        state = "open" if binary[s] == 1 else "close"
        is_close_event = s > 0 and binary[s] == 0 and binary[s - 1] == 1
        is_open_event = s > 0 and binary[s] == 1 and binary[s - 1] == 0
        seg_xyz = xyz[s:e + 1]
        disp = seg_xyz[-1] - seg_xyz[0] if len(seg_xyz) > 1 else np.zeros(3)
        disp_norm = float(np.linalg.norm(disp))
        disp_vertical = float(disp[2])
        prim = infer_semantic_primitive(state, is_close_event, is_open_event,
                                        disp_norm, disp_vertical, cfg)
        # start 边界来源: 起始帧 / 夹爪事件帧 -> "event" (硬锚点)，否则 RDP 补的边界
        source = "event" if (s == 0 or s in event_set) else "rdp"
        seg = Segment(subtask_text=prim, start_frame=s, end_frame=e,
                      keyframe=(s + e) // 2, gripper_state=state,
                      primitive_label=prim, boundary_source=source)
        segments.append(seg)
        primitives.append(prim)
        primitive_by_frame[s] = prim
        boundary_sources.append(source)

    result = {
        "segments": segments,
        "N_physical": len(segments),
        "primitives": primitives,
        "event_frames": event_frames,
        "gripper_binary": binary,
        "primitive_by_frame": primitive_by_frame,
        "has_nonprehensile_fill": any(src == "rdp" for src in boundary_sources),
    }
    episode.meta["stage1b"] = {
        "N_physical": len(segments),
        "primitives": primitives,
        "event_frames": event_frames,
        "segments": [{"start_frame": s.start_frame, "end_frame": s.end_frame,
                      "primitive_label": s.primitive_label} for s in segments],
    }
    return result
