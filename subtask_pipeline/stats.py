"""数据集级别统计报告 (Stage 5 #2)。"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np


def _dist(values: List[float]) -> Dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(len(arr)),
        "mean": round(float(arr.mean()), 3),
        "std": round(float(arr.std()), 3),
        "min": round(float(arr.min()), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "max": round(float(arr.max()), 3),
    }


def build_report(records: List[Dict], failures: List[Dict], length_stats: Dict) -> Dict:
    n = len(records)
    conf_counter = Counter(r["confidence"] for r in records)
    branch_counter = Counter(r["branch"] for r in records)

    n_subtasks = [len(r["segments"]) for r in records]
    seg_lengths = [s["end_frame"] - s["start_frame"] + 1 for r in records for s in r["segments"]]
    gripper_scores = [r["gripper_quality_score"] for r in records
                      if r.get("gripper_quality_score") is not None]
    retries = [r["annotation_meta"].get("self_check_retries", 0) or 0 for r in records]
    self_check_fail = sum(1 for r in records if r["annotation_meta"].get("self_check_passed") is False)
    desc_failed = sum(1 for r in records if r["annotation_meta"].get("description_gen_failed"))

    def ratio(c: Counter) -> Dict:
        return {k: {"count": v, "ratio": round(v / n, 3) if n else 0.0} for k, v in c.items()}

    return {
        "total_records": n,
        "total_failures": len(failures),
        "confidence_distribution": ratio(conf_counter),
        "branch_hit_rate": ratio(branch_counter),
        "subtask_count_distribution": _dist(n_subtasks),
        "segment_length_distribution": _dist(seg_lengths),
        "gripper_quality_distribution": _dist(gripper_scores),
        "self_check_retry_rate": round(sum(1 for r in retries if r > 0) / n, 3) if n else 0.0,
        "self_check_fail_count": self_check_fail,
        "description_gen_failed_count": desc_failed,
        "n_task_groups": len(length_stats),
    }


def filter_by_confidence(records: List[Dict], levels) -> List[Dict]:
    """训练集分层导出: 按 confidence 级别过滤 (如 ['Gold','Silver'])。"""
    levels = set(levels)
    return [r for r in records if r["confidence"] in levels]
