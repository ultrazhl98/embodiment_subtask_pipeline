# Subtask 标注数据生产产线

综合 **LaRA-VLA** 与 **CycleVLA** 设计、参考 **ECoT (embodied-CoT)** 特征提取代码的
L1→L2 指令分解数据标注产线。把开源机器人操控轨迹 (LeRobot v2/v2.1) 标注为带置信度
分级的 subtask 序列 `(subtask_text, start_frame, end_frame, keyframe, ...)`。

完整设计见 [`doc/pipeline_overview.md`](doc/pipeline_overview.md)，prompt 见 `doc/prompt_*.md`。

## 快速开始

无需 GPU / API key，用合成数据 + Mock LLM 离线跑通整条产线：

```bash
pip install -e ".[images,lerobot]"          # 或: pip install -r requirements.txt
python -m subtask_pipeline.cli run --synthetic --n 5 --out out.jsonl --report report.json
```

真实数据 + 真实 VLM (OpenAI 兼容端点，如本地 vLLM 部署的 Qwen2.5-VL)：

```bash
export OPENAI_API_KEY=...
python -m subtask_pipeline.cli run \
    --lerobot /path/to/lerobot_dataset \
    --config configs/default.yaml --backend openai \
    --out out.jsonl --report report.json
```

主链路 bootstrap（先验证端到端，跳过锚点/文本分解，走纯物理分割 → 全 Gold）：

```bash
python -m subtask_pipeline.cli run --synthetic --no-anchor --out out.jsonl
```

## 产线结构

| Stage | 模块 | 说明 | ECoT 参照 |
|-------|------|------|-----------|
| 0  | `stages/stage0_prefilter.py` | gripper 质量 / 长度异常 / 图像质量打标 | — |
| 1-A| `stages/stage1a_anchors.py`  | 首帧 VLM 提取语义锚点 (LaRA-VLA) | — |
| 1-B| `stages/stage1b_physical.py` | gripper 离散化 + 分段 + movement primitive | `primitive_movements.py` |
| 1-C| `stages/stage1c_text.py`     | LLM 文本语义分解 (CycleVLA) | — |
| 2  | `stages/stage2_align.py`     | 一致性路由: fast/negotiate/fallback | — |
| 3  | `stages/stage3_describe.py`  | VLM 描述生成 + 完整性/去重自检 | `full_reasonings.py` |
| 4  | `stages/stage4_grounding.py` | GroundingDINO/OWL-ViT + SAM bbox (可选) | `gripper_positions.py` |
| 5  | `stages/stage5_output.py`    | 质量分级输出 + `stats.py` 统计报告 | — |

数据抽象层 `data/`：`Episode` 把数据集格式与产线解耦；提供 `LeRobotLoader`
(v2/v2.1 parquet + mp4) 与 `make_synthetic_dataset` (离线测试)。

LLM 层 `llm/`：`BaseClient` 统一 JSON 解析 + 校验重试 (参考 ECoT Gemini 封装)；
`MockClient` 离线确定性桩、`OpenAICompatClient` 实后端。切换由 `LLMConfig.backend` 控制。

## 置信度分级

| branch | confidence | loss_weight | 触发条件 |
|--------|-----------|-------------|---------|
| fast      | Gold   | 1.0 | `N_physical == N_text` |
| negotiate | Silver | 0.5 | `0 < delta <= delta_tolerance` 且 gripper reliable |
| fallback  | Bronze | 0.2 | gripper noisy & 分歧，或 delta 过大，或上游降级 |

按级别分层导出见 `stats.filter_by_confidence` (预训练用全量，SFT 只用 Gold+Silver)。

## 配置

所有可调参数集中在 `subtask_pipeline/config.py`，默认值取自 doc 的"关键参数"。
用 `configs/default.yaml` 覆盖，按数据集调整 (如 DROID 设 `stage2.delta_tolerance: 2`)。

## 测试

```bash
PYTHONPATH=. python tests/test_pipeline.py        # 或 pytest tests/ -q
```

## 当前实现说明

- 确定性环节 (Stage 0 / 1-B / 2 路由 / 5 / 统计) 为完整可用实现。
- VLM/LLM 环节默认走 `MockClient` 打通流程；接真实模型只需设 `--backend openai`
  并配置 `base_url` 指向 Qwen2.5-VL 等服务，prompt 已按 doc 移植到 `llm/prompts.py`。
- Stage 4 grounding 为可选重依赖模块，默认关闭。
