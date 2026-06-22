"""产线冒烟测试 + 关键确定性逻辑单测。

运行: /path/to/python -m pytest tests/ -q   或直接 python tests/test_pipeline.py
"""

import numpy as np

from subtask_pipeline.config import PipelineConfig, Stage0Config, Stage1bConfig, Stage2Config
from subtask_pipeline.data.synthetic import make_synthetic_dataset, make_synthetic_episode
from subtask_pipeline.pipeline import PipelineRunner
from subtask_pipeline.stages import stage0_prefilter as s0
from subtask_pipeline.stages import stage1b_physical as s1b
from subtask_pipeline.stages import stage2_align as s2


def test_stage0_gripper_clean_vs_noisy():
    cfg = Stage0Config()
    clean = np.array([0] * 30 + [1] * 30, dtype=np.float32)
    noisy = clean.copy()
    rng = np.random.RandomState(0)
    flip = rng.choice(60, 12, replace=False)
    noisy[flip] = 1 - noisy[flip]
    q_clean = s0.gripper_quality(clean, cfg)
    q_noisy = s0.gripper_quality(noisy, cfg)
    assert q_clean["gripper_label"] == "reliable"
    assert q_clean["gripper_quality_score"] > q_noisy["gripper_quality_score"]
    assert q_noisy["gripper_label"] == "noisy"


def test_stage1b_segment_count_matches_subtasks():
    cfg = Stage1bConfig()
    ep = make_synthetic_episode("t", n_subtasks=3, gripper_noisy=False, seed=1)
    s0.run_stage0(ep, Stage0Config())
    out = s1b.run_stage1b(ep, cfg)
    # 3 段轨迹在每段边界翻转 gripper -> 物理分段应为 3
    assert out["N_physical"] == 3
    assert len(out["primitives"]) == 3
    # 边界对齐: 首段从 0, 末段到末帧, 连续覆盖
    segs = out["segments"]
    assert segs[0]["start_frame"] == 0
    assert segs[-1]["end_frame"] == ep.num_frames - 1
    for a, b in zip(segs[:-1], segs[1:]):
        assert b["start_frame"] == a["end_frame"] + 1


def test_stage2_routing():
    cfg = Stage2Config(delta_tolerance=1)
    assert s2.route(3, 3, "reliable", cfg) == "fast"
    assert s2.route(3, 4, "reliable", cfg) == "negotiate"
    assert s2.route(3, 6, "reliable", cfg) == "fallback"
    assert s2.route(3, 4, "noisy", cfg) == "fallback"   # noisy + 分歧 -> 降级


def test_negotiate_coverage_valid():
    """协商路径输出必须连续覆盖全程无 gap。"""
    cfg = PipelineConfig()
    cfg.stage2.delta_tolerance = 2
    runner = PipelineRunner(cfg)
    ep = make_synthetic_episode("neg", task_instruction="put the red cup into the open drawer",
                                n_subtasks=3, seed=2)
    out = runner.run([ep])
    rec = out["records"][0]
    segs = rec["segments"]
    assert segs[0]["start_frame"] == 0
    assert segs[-1]["end_frame"] == ep.num_frames - 1
    for a, b in zip(segs[:-1], segs[1:]):
        assert b["start_frame"] == a["end_frame"] + 1


def test_fallback_path_produces_n_text_segments():
    cfg = Stage2Config()
    ep = make_synthetic_episode("fb", n_subtasks=3, seed=3)
    s0.run_stage0(ep, Stage0Config())
    from subtask_pipeline.llm.mock import MockClient
    segs, mean_conf, low = s2.fallback_path(ep, ["reach the cup", "grasp the cup", "place the cup"],
                                            MockClient(), cfg)
    assert len(segs) == 3
    assert segs[0].start_frame == 0
    assert segs[-1].end_frame == ep.num_frames - 1


def test_end_to_end_full_and_bootstrap():
    eps = make_synthetic_dataset(n_episodes=5)
    # 完整链路
    full = PipelineRunner(PipelineConfig()).run(eps)
    assert len(full["records"]) == 5
    assert full["failures"] == []
    assert set(r["confidence"] for r in full["records"]) <= {"Gold", "Silver", "Bronze"}
    # 主链路 bootstrap (无锚点/文本分解 -> 纯物理, 应全 fast/Gold)
    cfg = PipelineConfig()
    cfg.enable_anchor_extraction = False
    cfg.enable_text_decomposition = False
    boot = PipelineRunner(cfg).run(make_synthetic_dataset(n_episodes=5))
    assert all(r["branch"] == "fast" for r in boot["records"])
    for r in boot["records"]:
        assert r["loss_weight"] == 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")
