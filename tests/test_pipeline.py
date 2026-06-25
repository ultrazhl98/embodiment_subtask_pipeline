"""产线冒烟测试 + 关键确定性逻辑单测。

运行: /path/to/python -m pytest tests/ -q   或直接 python tests/test_pipeline.py
"""

import numpy as np

from subtask_pipeline.config import (PipelineConfig, Stage0Config, Stage1bConfig,
                                     Stage1cConfig, Stage2Config)
from subtask_pipeline.data.synthetic import make_synthetic_dataset, make_synthetic_episode
from subtask_pipeline.data.types import Episode, Segment
from subtask_pipeline.llm.mock import MockClient
from subtask_pipeline.pipeline import PipelineRunner
from subtask_pipeline.stages import stage0_prefilter as s0
from subtask_pipeline.stages import stage05_global as s05
from subtask_pipeline.stages import stage1b_physical as s1b
from subtask_pipeline.stages import stage1c_text as s1c
from subtask_pipeline.stages import stage2_align as s2


# ---------------------------------------------------------------------------
# Stage 0
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stage 1-B: 语义原语 + 事件帧 + RDP
# ---------------------------------------------------------------------------


def test_stage1b_semantic_primitives_and_events():
    cfg = Stage1bConfig()
    ep = make_synthetic_episode("t", n_subtasks=3, gripper_noisy=False, seed=1)
    s0.run_stage0(ep, Stage0Config())
    out = s1b.run_stage1b(ep, cfg)
    # 3 段轨迹在每段边界翻转 gripper -> 物理分段应为 3
    assert out["N_physical"] == 3
    prims = out["primitives"]
    # 语义标签: 首段 reach(open) -> pick_up(close 事件 + 抬升) -> place(open 事件)
    assert prims[0] == "reach"
    assert prims[1] in ("pick_up", "grasp")
    assert prims[2] == "place"
    # 夹爪事件帧在两个段边界处
    assert out["event_frames"] == [50, 100]
    segs = out["segments"]
    assert isinstance(segs[0], Segment)
    assert segs[0].start_frame == 0
    assert segs[-1].end_frame == ep.num_frames - 1
    for a, b in zip(segs[:-1], segs[1:]):
        assert b.start_frame == a.end_frame + 1


def test_infer_semantic_primitive():
    cfg = Stage1bConfig()
    f = s1b.infer_semantic_primitive
    assert f("close", True, False, 0.6, 0.3, cfg) == "pick_up"   # close 事件 + 抬升
    assert f("close", True, False, 0.02, 0.0, cfg) == "grasp"    # close 事件 + 无抬升
    assert f("open", False, True, 0.5, 0.0, cfg) == "place"      # open 事件
    assert f("close", False, False, 0.5, 0.0, cfg) == "transport"  # 闭合 + 大位移
    assert f("close", False, False, 0.01, 0.0, cfg) == "hold"    # 闭合 + 小位移
    assert f("open", False, False, 0.5, 0.0, cfg) == "reach"


def _nonprehensile_episode() -> Episode:
    """夹爪全程闭合的 L 形轨迹 (push 类): 前段 +x, 后段 +y, 一个明显拐点。"""
    rng = np.random.RandomState(0)
    n = 60
    xyz = np.zeros((n, 3), dtype=np.float32)
    pos = np.zeros(3, dtype=np.float32)
    for i in range(n):
        step = np.array([0.02, 0.0, 0.0]) if i < 30 else np.array([0.0, 0.02, 0.0])
        pos = pos + step + rng.randn(3) * 0.001
        xyz[i] = pos
    gripper = np.zeros(n, dtype=np.float32)  # 全程闭合
    states = np.concatenate([xyz, np.zeros((n, 3), dtype=np.float32), gripper[:, None]], axis=1)
    return Episode(episode_id="nonpre", task_instruction="push the box around the corner",
                   num_frames=n, states=states, gripper=gripper, meta={"source": "synthetic"})


def test_stage1b_rdp_nonprehensile_split():
    cfg = Stage1bConfig()
    ep = _nonprehensile_episode()
    s0.run_stage0(ep, Stage0Config())
    out = s1b.run_stage1b(ep, cfg)
    # 夹爪无切换 -> 仅靠 RDP 拐点把 L 形拆成 >=2 段
    assert out["event_frames"] == []
    assert out["N_physical"] >= 2
    assert out["has_nonprehensile_fill"] is True


def test_rdp_boundaries_detects_corner():
    xyz = np.concatenate([
        np.stack([np.linspace(0, 1, 30), np.zeros(30), np.zeros(30)], axis=1),
        np.stack([np.ones(30), np.linspace(0, 1, 30), np.zeros(30)], axis=1),
    ]).astype(np.float32)
    bounds = s1b.rdp_boundaries(xyz, epsilon=0.05)
    assert any(25 <= b <= 35 for b in bounds)  # 拐点在 ~30


# ---------------------------------------------------------------------------
# Stage 0.5 全局理解
# ---------------------------------------------------------------------------


def test_stage05_global_summary():
    from subtask_pipeline.config import Stage05Config
    ep = make_synthetic_episode("g", n_subtasks=3, seed=1)
    summary = s05.run_stage05(ep, MockClient(), Stage05Config())
    assert summary is not None
    assert summary["task_intent"]
    assert summary["objects"] and all(o["description"] for o in summary["objects"])
    assert ep.meta["global_summary"] is summary


def test_stage05_disabled_returns_none():
    from subtask_pipeline.config import Stage05Config
    ep = make_synthetic_episode("g2", n_subtasks=2, seed=1)
    assert s05.run_stage05(ep, MockClient(), Stage05Config(enable=False)) is None


# ---------------------------------------------------------------------------
# Stage 1-C 填槽
# ---------------------------------------------------------------------------


def test_stage1c_fills_template_with_object():
    ep = make_synthetic_episode("c", n_subtasks=3, seed=1)
    s0.run_stage0(ep, Stage0Config())
    from subtask_pipeline.config import Stage05Config
    s05.run_stage05(ep, MockClient(), Stage05Config())
    stage1b = s1b.run_stage1b(ep, Stage1bConfig())
    segs = s1c.run_stage1c(ep, stage1b["segments"], MockClient(), Stage1cConfig(),
                           PipelineConfig().allowed_verbs)
    # 每段文本以允许动词开头, 且引用了场景物体 (cup)
    for seg in segs:
        assert seg.subtask_text
        assert seg.completion_frame == seg.end_frame
    joined = " ".join(s.subtask_text for s in segs)
    assert "cup" in joined


def test_stage1c_bootstrap_no_vlm():
    ep = make_synthetic_episode("cb", n_subtasks=2, seed=1, with_images=False)
    s0.run_stage0(ep, Stage0Config())
    stage1b = s1b.run_stage1b(ep, Stage1bConfig())
    segs = s1c.run_stage1c(ep, stage1b["segments"], MockClient(), Stage1cConfig(),
                           PipelineConfig().allowed_verbs, enable_vlm=False)
    # bootstrap 仍以允许动词开头 (来自模板)
    from subtask_pipeline.config import starts_with_allowed_verb
    for seg in segs:
        assert starts_with_allowed_verb(seg.subtask_text, PipelineConfig().allowed_verbs)


# ---------------------------------------------------------------------------
# Stage 2 规则过滤
# ---------------------------------------------------------------------------


def test_stage2_rule_check_pass_and_fail():
    cfg = Stage2Config()
    stage1b = {
        "primitive_by_frame": {0: "reach"},
        "gripper_binary": np.ones(60, dtype=np.int8),  # 全 open
    }
    good = Segment(subtask_text="reach for the cup", start_frame=0, end_frame=40,
                   primitive_label="reach")
    ok, _ = s2.rule_check(good, stage1b, cfg)
    assert ok
    # 动词不匹配
    bad_verb = Segment(subtask_text="push the cup", start_frame=0, end_frame=40,
                       primitive_label="reach")
    ok, reason = s2.rule_check(bad_verb, stage1b, cfg)
    assert not ok and "verb" in reason
    # 时长过短
    short = Segment(subtask_text="reach for the cup", start_frame=0, end_frame=3,
                    primitive_label="reach")
    ok, reason = s2.rule_check(short, stage1b, cfg)
    assert not ok and "short" in reason


def test_stage2_gripper_polarity_mismatch():
    cfg = Stage2Config()
    stage1b = {"primitive_by_frame": {}, "gripper_binary": np.ones(60, dtype=np.int8)}  # 全 open
    # transport 期望闭合, 但全 open -> 失败
    seg = Segment(subtask_text="transport the cup to the drawer", start_frame=0, end_frame=40,
                  primitive_label="transport")
    ok, reason = s2.rule_check(seg, stage1b, cfg)
    assert not ok and "expects closed" in reason


# ---------------------------------------------------------------------------
# 契约 / loader
# ---------------------------------------------------------------------------


def test_episode_validate():
    make_synthetic_episode("ok", n_subtasks=2, seed=0).validate()
    bad = make_synthetic_episode("bad_grip", n_subtasks=2, seed=0)
    bad.gripper = bad.gripper + 5.0
    try:
        bad.validate()
        assert False, "应因 gripper 越界报错"
    except ValueError:
        pass
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
    loader = object.__new__(LeRobotLoader)
    raw = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    loader.gripper_open_is_high, loader.gripper_min, loader.gripper_max = True, None, None
    assert np.allclose(loader._normalize_gripper(raw), raw)
    loader.gripper_open_is_high = False
    assert np.allclose(loader._normalize_gripper(raw), 1.0 - raw)
    loader.gripper_open_is_high, loader.gripper_min, loader.gripper_max = True, 0.0, 2.0
    assert np.allclose(loader._normalize_gripper(np.array([0.0, 1.0, 2.0])), [0.0, 0.5, 1.0])


def test_invalid_episode_collected_as_failure():
    bad = make_synthetic_episode("e2e_bad", n_subtasks=2, seed=0)
    bad.gripper = bad.gripper + 5.0
    good = make_synthetic_episode("e2e_ok", n_subtasks=2, seed=1)
    out = PipelineRunner(PipelineConfig()).run([bad, good])
    assert len(out["records"]) == 1
    assert len(out["failures"]) == 1
    assert out["failures"][0]["episode_id"] == "e2e_bad"


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------


def test_end_to_end_full_and_bootstrap():
    eps = make_synthetic_dataset(n_episodes=5)
    full = PipelineRunner(PipelineConfig()).run(eps)
    assert len(full["records"]) == 5
    assert full["failures"] == []
    assert set(r["confidence"] for r in full["records"]) <= {"Gold", "Bronze", "Flagged"}
    # 输出含 primitive_label / progress
    rec = full["records"][0]
    seg0 = rec["segments"][0]
    assert "primitive_label" in seg0
    assert isinstance(seg0["progress"], list) and seg0["progress"][0] == 0.0 and seg0["progress"][-1] == 1.0
    assert rec["global_summary"] is not None

    # bootstrap: 无全局理解 + 无 VLM 填槽 -> 纯物理分割, 抓持轨迹应全 Gold
    cfg = PipelineConfig()
    cfg.stage05.enable = False
    cfg.enable_text_decomposition = False
    boot = PipelineRunner(cfg).run(make_synthetic_dataset(n_episodes=5))
    for r in boot["records"]:
        assert r["confidence"] == "Gold"
        assert r["loss_weight"] == 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")
