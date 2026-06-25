"""端到端编排器。

把 Stage 0 ~ Stage 5 串成一条产线:
    Stage 0    轨迹预过滤
    Stage 0.5  全局视频理解 (写 episode.meta["global_summary"])
    Stage 1-B  事件帧提取 + 语义原语分割 (驱动分割边界)
    Stage 1-C  per-segment 描述填槽
    Stage 2    规则质量过滤 (Gold / Bronze / Flagged)
    Stage 4    grounding (可选)
    Stage 5    质量分级输出 (+ progress)
逐条轨迹容错: 单条失败不影响其余, 错误收集到 failures。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from .config import PipelineConfig
from .data.types import Episode
from .llm import build_client
from .llm.base import BaseClient
from .stages import (
    stage0_prefilter as s0,
    stage05_global as s05,
    stage1b_physical as s1b,
    stage1c_text as s1c,
    stage2_align as s2,
    stage4_grounding as s4,
    stage5_output as s5,
)

logger = logging.getLogger("subtask_pipeline")


class PipelineRunner:
    def __init__(self, config: Optional[PipelineConfig] = None, client: Optional[BaseClient] = None):
        self.cfg = config or PipelineConfig()
        self.client = client or build_client(self.cfg.llm)
        self._grounder = None
        if self.cfg.stage4.enable_grounding:
            self._grounder = s4.Grounder(self.cfg.stage4)

    # ----------------------------------------------------------------------
    def run(self, episodes: Sequence[Episode]) -> Dict:
        episodes = list(episodes)
        failures: List[Dict] = []

        # Pass 0: 契约校验，违约者计入 failures 不参与后续
        valid: List[Episode] = []
        for ep in episodes:
            try:
                ep.validate()
                valid.append(ep)
            except Exception as e:  # noqa: BLE001
                logger.warning("episode %s 不满足输入契约: %s", ep.episode_id, e)
                failures.append({"episode_id": ep.episode_id, "error": repr(e)})
        episodes = valid

        # Pass 1: Stage 0 逐条 + 数据集级长度异常
        lengths_by_task: Dict[str, List[int]] = {}
        for ep in episodes:
            try:
                s0.run_stage0(ep, self.cfg.stage0)
            except Exception as e:  # noqa: BLE001
                logger.warning("stage0 failed for %s: %s", ep.episode_id, e)
            lengths_by_task.setdefault(ep.task_instruction, []).append(ep.num_frames)
        length_stats = s0.compute_length_flags(lengths_by_task, self.cfg.stage0)
        for ep in episodes:
            ep.meta["length_flag"] = s0.length_flag(ep.num_frames, ep.task_instruction, length_stats)

        # Pass 2: Stage 0.5 ~ 5 逐条
        records = []
        for ep in episodes:
            try:
                records.append(self._run_episode(ep))
            except Exception as e:  # noqa: BLE001
                logger.exception("episode %s failed", ep.episode_id)
                failures.append({"episode_id": ep.episode_id, "error": repr(e)})

        from .stats import build_report
        report = build_report(records, failures, length_stats)
        return {"records": records, "failures": failures, "report": report}

    # ----------------------------------------------------------------------
    def _run_episode(self, ep: Episode) -> Dict:
        # Stage 0.5 全局视频理解
        summary = s05.run_stage05(ep, self.client, self.cfg.stage05)

        # Stage 1-B 物理分割 (驱动分割边界 + 语义原语)
        stage1b = s1b.run_stage1b(ep, self.cfg.stage1b)

        # Stage 1-C per-segment 描述填槽
        s1c.run_stage1c(ep, stage1b["segments"], self.client, self.cfg.stage1c,
                        self.cfg.allowed_verbs, enable_vlm=self.cfg.enable_text_decomposition)

        # Stage 2 规则质量过滤
        stage2 = s2.run_stage2(ep, stage1b, self.cfg.stage2)

        # Stage 4 grounding (可选, 使用 global_summary 的物体列表)
        scene_objects = summary.get("objects", []) if summary else []
        segments = s4.run_stage4(ep, stage2["segments"], scene_objects, self.cfg.stage4, self._grounder)

        # Stage 5 输出装配
        return s5.assemble_record(ep, segments, stage2["confidence"], self.cfg.loss_weights)


def run_pipeline(episodes: Sequence[Episode], config: Optional[PipelineConfig] = None) -> Dict:
    """便捷函数。"""
    return PipelineRunner(config).run(episodes)
