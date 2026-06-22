"""Stage 2 — 一致性校验与对齐。

根据三路信号一致性路由到快速 (Gold) / 协商 (Silver) / 降级 (Bronze) 三条分支，
产出统一的 (subtask_text, start_frame, end_frame, keyframe) 列表。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import Stage2Config
from ..data.types import Episode, Segment
from ..llm.base import BaseClient, LLMResponseError
from ..llm.prompts import build_fallback_prompt, build_negotiate_prompt


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


def route(n_physical: int, n_text: int, gripper_label: str, cfg: Stage2Config) -> str:
    delta = abs(n_physical - n_text)
    if gripper_label == "noisy" and delta > 0:
        return "fallback"
    if delta == 0:
        return "fast"
    if delta <= cfg.delta_tolerance:
        return "negotiate"
    return "fallback"


def _keyframe(start: int, end: int) -> int:
    return (start + end) // 2


# ---------------------------------------------------------------------------
# 快速路径 (Gold)
# ---------------------------------------------------------------------------


def fast_path(subtask_texts: Sequence[str], phys_segments: Sequence[Dict]) -> List[Segment]:
    segs = []
    for text, ps in zip(subtask_texts, phys_segments):
        s, e = ps["start_frame"], ps["end_frame"]
        segs.append(Segment(subtask_text=text, start_frame=s, end_frame=e, keyframe=_keyframe(s, e)))
    return segs


# ---------------------------------------------------------------------------
# 协商路径 (Silver)
# ---------------------------------------------------------------------------


def _format_primitives(phys_segments: Sequence[Dict], primitives: Sequence[str],
                       max_tokens: int) -> str:
    lines = []
    for ps, prim in zip(phys_segments, primitives):
        lines.append(f"[frames {ps['start_frame']}-{ps['end_frame']}] {prim}")
    # 下采样到 max_tokens 行以内
    if len(lines) > max_tokens:
        idx = np.linspace(0, len(lines) - 1, max_tokens).astype(int)
        lines = [lines[i] for i in idx]
    return "\n".join(lines)


def _validate_assignments(total_frames: int, n: int):
    def validate(parsed) -> Optional[str]:
        a = parsed.get("assignments") if isinstance(parsed, dict) else None
        if not isinstance(a, list) or len(a) != n:
            return f"need exactly {n} assignments"
        try:
            a = sorted(a, key=lambda x: x["subtask_index"])
            if a[0]["start_frame"] != 0:
                return "first start_frame must be 0"
            if a[-1]["end_frame"] != total_frames - 1:
                return f"last end_frame must be {total_frames - 1}"
            for i in range(n):
                if a[i]["start_frame"] > a[i]["end_frame"]:
                    return "start_frame > end_frame"
                if i > 0 and a[i]["start_frame"] != a[i - 1]["end_frame"] + 1:
                    return "gap or overlap between assignments"
        except (KeyError, TypeError):
            return "assignment missing required int fields"
        return None
    return validate


def negotiate_path(episode: Episode, subtask_texts: Sequence[str], stage1b: Dict,
                   client: BaseClient, cfg: Stage2Config) -> Optional[List[Segment]]:
    total = episode.num_frames
    primitives_fmt = _format_primitives(stage1b["segments"], stage1b["primitives"],
                                        cfg.negotiate_max_primitive_tokens)
    fpp = max(1, total // max(1, len(stage1b["segments"])))
    system, user = build_negotiate_prompt(total, subtask_texts, primitives_fmt, fpp)
    try:
        parsed = client.generate_json(system, user,
                                       validator=_validate_assignments(total, len(subtask_texts)))
    except LLMResponseError:
        return None  # 校验失败 -> 由调用方降级 Bronze
    a = sorted(parsed["assignments"], key=lambda x: x["subtask_index"])
    return [Segment(subtask_text=subtask_texts[i], start_frame=int(x["start_frame"]),
                    end_frame=int(x["end_frame"]),
                    keyframe=_keyframe(int(x["start_frame"]), int(x["end_frame"])))
            for i, x in enumerate(a)]


# ---------------------------------------------------------------------------
# 降级路径 (Bronze)
# ---------------------------------------------------------------------------


def _merge_runs(frame_labels: List[int], frame_idxs: List[int]):
    """连续相同 label 合并为 segment (start,end,label)，覆盖到帧区间中点边界。"""
    runs = []
    cur_label = frame_labels[0]
    run_start_pos = 0
    for i in range(1, len(frame_labels) + 1):
        if i == len(frame_labels) or frame_labels[i] != cur_label:
            runs.append((run_start_pos, i - 1, cur_label))
            if i < len(frame_labels):
                cur_label = frame_labels[i]
                run_start_pos = i
    return runs


def fallback_path(episode: Episode, subtask_texts: Sequence[str], client: BaseClient,
                  cfg: Stage2Config) -> tuple:
    """返回 (segments, mean_confidence, low_visual_confidence)。"""
    n_text = len(subtask_texts)
    total = episode.num_frames
    n_samples = max(n_text, n_text * cfg.fallback_sample_factor)
    sample_idxs = sorted(set(np.linspace(0, total - 1, n_samples).astype(int)))

    labels: List[int] = []
    confs: List[float] = []
    if episode.has_images:
        system, user = build_fallback_prompt(episode.task_instruction, subtask_texts)
        for fi in sample_idxs:
            img = episode.image(fi)
            try:
                parsed = client.generate_json(system, user, images=[img])
                idx = int(parsed.get("selected_index", 0))
                conf = float(parsed.get("confidence", 0.0))
            except LLMResponseError:
                idx, conf = (labels[-1] if labels else 0), 0.0
            labels.append(int(np.clip(idx, 0, n_text - 1)))
            confs.append(conf)
        # 低置信度帧用相邻帧填充
        for i, c in enumerate(confs):
            if c < cfg.fallback_low_conf_threshold and i > 0:
                labels[i] = labels[i - 1]
    else:
        # 无图像: 退化为按候选数均匀切分
        for i, fi in enumerate(sample_idxs):
            labels.append(min(n_text - 1, i * n_text // len(sample_idxs)))
            confs.append(0.5)

    runs = _merge_runs(labels, sample_idxs)
    runs = _reconcile_run_count(runs, n_text, confs)

    # runs (按采样下标) -> 帧区间，边界取相邻采样点中点
    segs = []
    for k, (ps, pe, label) in enumerate(runs):
        start = 0 if k == 0 else (sample_idxs[ps] + sample_idxs[ps - 1]) // 2
        end = total - 1 if k == len(runs) - 1 else (sample_idxs[pe] + sample_idxs[pe + 1]) // 2 - 1
        text = subtask_texts[min(label, n_text - 1)]
        segs.append(Segment(subtask_text=text, start_frame=start, end_frame=max(start, end),
                            keyframe=_keyframe(start, max(start, end))))
    mean_conf = float(np.mean(confs)) if confs else 0.0
    return segs, mean_conf, mean_conf < cfg.fallback_mean_conf_threshold


def _reconcile_run_count(runs, n_text: int, confs):
    """把合并后的 run 数对齐到 n_text。"""
    runs = list(runs)
    # 段数 > n_text: 反复合并最短的相邻 run
    while len(runs) > n_text:
        lengths = [pe - ps for ps, pe, _ in runs]
        i = int(np.argmin(lengths))
        j = i + 1 if i + 1 < len(runs) else i - 1
        lo, hi = min(i, j), max(i, j)
        runs = runs[:lo] + [(runs[lo][0], runs[hi][1], runs[lo][2])] + runs[hi + 1:]
    # 段数 < n_text: 在最长的 run 处对半拆 (近似"最低置信度处插入边界")
    while len(runs) < n_text:
        lengths = [pe - ps for ps, pe, _ in runs]
        i = int(np.argmax(lengths))
        ps, pe, label = runs[i]
        mid = (ps + pe) // 2
        if mid <= ps:
            break
        runs = runs[:i] + [(ps, mid, label), (mid + 1, pe, label)] + runs[i + 1:]
    # 重新顺序赋 subtask label (确保各段对应递增的 subtask)
    if len(runs) == n_text:
        runs = [(ps, pe, k) for k, (ps, pe, _) in enumerate(runs)]
    return runs


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

_CONF = {"fast": "Gold", "negotiate": "Silver", "fallback": "Bronze"}


def run_stage2(episode: Episode, subtask_texts: Sequence[str], stage1b: Dict,
               client: BaseClient, cfg: Stage2Config) -> Dict:
    gripper_label = episode.meta.get("gripper_label", "reliable")
    n_physical = stage1b["N_physical"]
    n_text = len(subtask_texts)
    branch = route(n_physical, n_text, gripper_label, cfg)

    extra = {}
    if branch == "fast":
        segments = fast_path(subtask_texts, stage1b["segments"])
    elif branch == "negotiate":
        segments = negotiate_path(episode, subtask_texts, stage1b, client, cfg)
        if segments is None:  # 协商校验失败 -> 降级
            branch = "fallback"
            segments, mean_conf, low = fallback_path(episode, subtask_texts, client, cfg)
            extra = {"fallback_mean_conf": round(mean_conf, 3), "low_visual_confidence": low,
                     "downgraded_from": "negotiate"}
    else:  # fallback
        segments, mean_conf, low = fallback_path(episode, subtask_texts, client, cfg)
        extra = {"fallback_mean_conf": round(mean_conf, 3), "low_visual_confidence": low}

    result = {
        "segments": segments,
        "confidence": _CONF[branch],
        "branch": branch,
        "delta": abs(n_physical - n_text),
        **extra,
    }
    episode.meta["stage2"] = {"branch": branch, "confidence": _CONF[branch],
                              "delta": result["delta"], **extra}
    return result
