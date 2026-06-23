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


def test_episode_validate():
    from subtask_pipeline.data.types import Episode
    # 合成 Episode 满足契约
    make_synthetic_episode("ok", n_subtasks=2, seed=0).validate()
    # gripper 超出 [0,1] 约定 -> raise
    bad = make_synthetic_episode("bad_grip", n_subtasks=2, seed=0)
    bad.gripper = bad.gripper + 5.0
    try:
        bad.validate()
        assert False, "应因 gripper 越界报错"
    except ValueError:
        pass
    # states 行数与 num_frames 不一致 -> raise
    bad2 = make_synthetic_episode("bad_shape", n_subtasks=2, seed=0)
    bad2.num_frames = bad2.num_frames + 7
    try:
        bad2.validate()
        assert False, "应因 shape 不一致报错"
    except ValueError:
        pass


def test_build_loader_synthetic_runs():
    from subtask_pipeline.config import DatasetConfig
    from subtask_pipeline.data import build_loader
    loader = build_loader(DatasetConfig(type="synthetic", n=3, with_images=False))
    assert len(loader) == 3
    out = PipelineRunner(PipelineConfig()).run(list(loader.iter_episodes()))
    assert len(out["records"]) == 3
    assert out["failures"] == []


def test_gripper_polarity_normalization():
    from subtask_pipeline.data.lerobot_loader import LeRobotLoader
    # 不经磁盘构造, 只测归一逻辑
    loader = object.__new__(LeRobotLoader)
    raw = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)  # 原始: 1=闭合
    # 正极性: 原样
    loader.gripper_open_is_high, loader.gripper_min, loader.gripper_max = True, None, None
    assert np.allclose(loader._normalize_gripper(raw), raw)
    # 反极性: 翻转 -> 1=张开
    loader.gripper_open_is_high = False
    assert np.allclose(loader._normalize_gripper(raw), 1.0 - raw)
    # min-max 归一
    loader.gripper_open_is_high, loader.gripper_min, loader.gripper_max = True, 0.0, 2.0
    assert np.allclose(loader._normalize_gripper(np.array([0.0, 1.0, 2.0])), [0.0, 0.5, 1.0])


def test_invalid_episode_collected_as_failure():
    from subtask_pipeline.data.types import Episode
    bad = make_synthetic_episode("e2e_bad", n_subtasks=2, seed=0)
    bad.gripper = bad.gripper + 5.0
    good = make_synthetic_episode("e2e_ok", n_subtasks=2, seed=1)
    out = PipelineRunner(PipelineConfig()).run([bad, good])
    assert len(out["records"]) == 1
    assert len(out["failures"]) == 1
    assert out["failures"][0]["episode_id"] == "e2e_bad"


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
