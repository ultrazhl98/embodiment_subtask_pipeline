# Subtask 标注产线开发文档

综合 LaRA-VLA 和 CycleVLA 设计的 L1→L2 指令分解数据标注产线。

---

## 产线总览

```
轨迹数据 (LeRobot v2/v2.1)
        │
        ▼
  Stage 0: 轨迹预过滤
        │
        ▼
  Stage 1: 三路并行提取
  ┌──────┴──────────────────┐──────────────────┐
  │路径A: 语义锚点提取       │路径B: 物理信号分割 │路径C: 文本语义分解
  └──────┬──────────────────┘──────────────────┘
        │
        ▼
  Stage 2: 一致性校验与对齐
  ┌──────┴──────────────┬──────────────────────┐
  │快速路径 (Gold)       │协商路径 (Silver)      │降级路径 (Bronze)
  └──────┬──────────────┘──────────────────────┘
        │
        ▼
  Stage 3: VLM 描述生成 + 一致性自检
        │
        ▼
  Stage 4: Grounding 扩展（可选）
        │
        ▼
  Stage 5: 质量分级输出
```

---

## Prompt 文件索引

| Prompt | 用途 | 所在阶段 |
|--------|------|---------|
| [prompt_1a_anchor_extraction.md](prompts/prompt_1a_anchor_extraction.md) | 从首帧图像和任务指令中提取语义锚点物体 | Stage 1-A |
| [prompt_1c_text_decomposition.md](prompts/prompt_1c_text_decomposition.md) | LLM 对任务指令做原子 subtask 文本分解 | Stage 1-C |
| [prompt_2_negotiate.md](prompts/prompt_2_negotiate.md) | 协商路径：结合 movement primitive 重新推断 subtask 时间戳 | Stage 2 |
| [prompt_2_fallback.md](prompts/prompt_2_fallback.md) | 降级路径：视觉关键帧的检索式 subtask 标签匹配 | Stage 2 |
| [prompt_3_description_gen.md](prompts/prompt_3_description_gen.md) | 为每个 segment 生成 subtask 自然语言描述 | Stage 3 |
| [prompt_3_self_check.md](prompts/prompt_3_self_check.md) | 对生成的 subtask 序列做完整性和去重自检 | Stage 3 |

---

## Stage 0 — 轨迹预过滤

### 目标

在进入标注流程之前，对每条轨迹打质量标记，决定后续走哪条分支，避免下游对噪声数据做无效计算。

### 输入

- 轨迹的 gripper 状态序列（连续值或离散 open/close）
- 轨迹的图像帧序列
- 轨迹元数据（帧数、采集时间戳）

### 输出

每条轨迹附加以下 metadata 字段：

```json
{
  "gripper_quality_score": 0.85,
  "gripper_label": "reliable",
  "length_flag": "normal",
  "image_quality_flag": "pass"
}
```

### 开发内容

#### 1. Gripper Signal 质量检测器

- **高频抖动检测**：计算单位时间内 gripper 状态切换次数，超过阈值（建议每 10 帧超过 2 次切换）标记为抖动
- **无效切换过滤**：切换后不足最小持续帧数（建议 5 帧）即切回的视为噪声跳变，统计无效切换比例
- **整体置信度评分**：综合抖动频率和无效切换比例，输出 `gripper_quality_score ∈ [0, 1]`
- **标记规则**：score ≥ 0.7 → `reliable`；score < 0.7 → `noisy`

#### 2. 轨迹长度异常检测

- 按任务类型分组，计算各组轨迹长度的中位数和 MAD（中位数绝对偏差）
- 长度超出 `median ± 3×MAD` 的轨迹标记为异常
- 分别标记 `too_short`（可能采集中断）和 `too_long`（可能 demo 失败后继续录制）

#### 3. 图像质量检测

- **亮度检测**：对首帧、末帧及均匀采样的中间帧计算灰度均值，低于阈值（建议 30/255）标记为过暗
- **模糊检测**：计算图像拉普拉斯方差，低于阈值（建议 100）标记为模糊
- **Frozen frame 检测**：计算连续帧像素差的均值，连续多帧差异极小（建议均值 < 1.0）标记为信号丢失

#### 4. 过滤结果写入

将以上四项结果写入每条轨迹的 metadata，供后续所有 Stage 读取，不做硬过滤（保留所有轨迹，只打标记）。

### 关键参数（建议值，需按数据集调整）

| 参数 | 建议值 |
|------|--------|
| 抖动检测窗口 | 10 帧 |
| 无效切换最小持续帧 | 5 帧 |
| gripper reliable 阈值 | score ≥ 0.7 |
| 长度异常 MAD 倍数 | 3× |
| 图像过暗阈值 | 灰度均值 < 30 |
| 模糊检测阈值 | 拉普拉斯方差 < 100 |

---

## Stage 1 — 三路并行提取

三条路径可以并行运行，互相独立，结果在 Stage 2 汇合。

---

### 路径 A：语义锚点提取

**参考来源：** LaRA-VLA

#### 目标

从首帧图像和任务指令中识别本次任务涉及的被操作物体，产出语义锚点，用于后续 Stage 1-C 的 LLM 分解和 Stage 3 的描述生成中控制物体指称一致性。

#### 输入

- 任务全局指令文字（如 `"put the red cup into the drawer"`）
- 轨迹首帧图像（若首帧被机械臂遮挡，跳过前若干帧取第一个清晰帧）

#### 输出

```json
{
  "anchor_objects": [
    { "role": "source", "description": "the red cup on the left side of the table" },
    { "role": "target", "description": "the open drawer below the counter" }
  ]
}
```

#### 开发内容

- **首帧选择逻辑**：从第 0 帧开始，计算图像中心区域的拉普拉斯方差，跳过模糊或遮挡帧（方差低于阈值），取第一个清晰帧作为锚点提取的输入
- **VLM 调用**：调用 VLM（建议 Qwen2.5-VL 或同等能力模型），输入首帧图像和任务指令，使用 [prompt_1a_anchor_extraction.md](prompts/prompt_1a_anchor_extraction.md)
- **输出解析**：解析 JSON 输出，校验 `role` 字段是否为 `source` 或 `target`，物体描述是否非空
- **后处理**：统一描述格式为"颜色 + 类别 + 位置"，去除冗余修饰词

#### 注意点

- 对于单物体任务（如 `"grasp the mug"`），只有 `source`，没有 `target`
- 对于多步骤任务（如 `"pick A, then pick B, put both in C"`），需要提取全部涉及的物体
- 锚点描述应足够具体（含颜色/形状/位置），以便 GroundingDINO 在 Stage 4 做 grounding

**→ Prompt 参见：** [prompt_1a_anchor_extraction.md](prompts/prompt_1a_anchor_extraction.md)

---

### 路径 B：物理信号分割

**参考来源：** CycleVLA + LaRA-VLA

#### 目标

从 robot proprio 数据中提取 segment 边界和 movement primitive 描述，产出候选分割结果。

#### 输入

- Gripper 状态序列（来自 proprio 数据）
- 末端执行器位置序列（xyz + 旋转）
- Stage 0 的 `gripper_label`

#### 输出

```json
{
  "N_physical": 3,
  "segments": [
    { "start_frame": 0, "end_frame": 87, "gripper_state": "open" },
    { "start_frame": 88, "end_frame": 143, "gripper_state": "close" },
    { "start_frame": 144, "end_frame": 201, "gripper_state": "open" }
  ],
  "primitives": [
    "reach forward and down toward target",
    "grasp and lift upward",
    "move right and release"
  ]
}
```

#### 开发内容

**Gripper State 离散化**

- `reliable` 轨迹：对连续 gripper 值用固定阈值（如 0.5）做二值化
- `noisy` 轨迹：先做滑动窗口中值滤波（窗口大小 5-10 帧），再做二值化，记录平滑前后段数差异作为置信度降级依据

**Segment 边界确定**

- 检测 open→close 和 close→open 的切换点，作为候选边界
- 合并过短 segment（长度 < 10 帧）到相邻段
- 对首尾做边界对齐：第一个 segment 从第 0 帧开始，最后一个 segment 到最后一帧结束

**Movement Primitive 提取**

- 对每个 segment 内的 EEF 轨迹计算主运动方向（前/后/左/右/上/下，取位移向量最大分量）
- 计算 segment 内平均速度，区分快速移动（reach）和缓慢精细（place）
- 将运动特征转换为简短文字描述（6 词以内），作为 Stage 2 协商路径的物理证据

---

### 路径 C：文本语义分解

**参考来源：** CycleVLA

#### 目标

用 LLM 对任务指令做纯文本分解，注入语义锚点以控制物体指称，产出原子 subtask 序列。

#### 输入

- 任务全局指令文字
- Stage 1-A 产出的 `anchor_objects`

#### 输出

```json
{
  "N_text": 3,
  "subtask_texts": [
    "reach toward the red cup on the left",
    "grasp the red cup and lift upward",
    "move to the open drawer and release"
  ]
}
```

#### 开发内容

- **LLM 调用**：使用 [prompt_1c_text_decomposition.md](prompts/prompt_1c_text_decomposition.md)，将任务指令和锚点物体列表一起输入
- **合法性校验**：
  - subtask 数量是否在合理范围（2-6 个）
  - 动词是否在约束词汇表内（reach / grasp / move / place / release / push / pull / open / close）
  - 物体指称是否与锚点一致（字符串匹配或简单语义检查）
- **格式标准化**：统一为现在进行时（"grasping"）或动词原形（"grasp"），保持全批数据一致

**→ Prompt 参见：** [prompt_1c_text_decomposition.md](prompts/prompt_1c_text_decomposition.md)

---

## Stage 2 — 一致性校验与对齐

### 目标

根据三路信号的一致性决定走哪条分支，产出最终的 `(subtask_text, start_frame, end_frame)` 三元组列表，并标记置信度等级。

### 输入

- Stage 1-A：`anchor_objects`
- Stage 1-B：`N_physical`、`segments`、`primitives`
- Stage 1-C：`N_text`、`subtask_texts`
- Stage 0：`gripper_label`

### 输出（三条分支统一格式）

```json
{
  "segments": [
    {
      "subtask_text": "reach toward the red cup",
      "start_frame": 0,
      "end_frame": 87,
      "keyframe": 44
    }
  ],
  "confidence": "Gold",
  "branch": "fast"
}
```

### 开发内容

#### 1. 一致性判断器

```
delta = |N_physical - N_text|

if gripper_label == "noisy" and delta > 0:
    → 降级路径

elif delta == 0:
    → 快速路径

elif delta <= 2:
    → 协商路径

else:
    → 降级路径
```

`delta` 容忍阈值建议作为可配置参数，不同数据集特性差异大（LIBERO 建议 delta ≤ 1，DROID 建议 delta ≤ 2）。

#### 2. 快速路径（Gold）

- 将 `subtask_texts` 列表与 `segments` 列表按顺序逐一配对
- 每个 segment 取中间帧作为 `keyframe`
- 输出 confidence = `Gold`，branch = `fast`

#### 3. 协商路径（Silver）

- 将下采样后的 `primitives` 序列（最多保留 20 个 token）与当前 `subtask_texts` 一起构建 prompt
- 调用 LLM 重新推断每个 subtask 的时间戳区间，要求连续覆盖、无 gap、无重叠
- 对 LLM 输出做合法性校验：时间戳是否在轨迹范围内（0 到总帧数）、是否覆盖全程
- 校验失败则降级为 Bronze
- 输出 confidence = `Silver`，branch = `negotiate`

**→ Prompt 参见：** [prompt_2_negotiate.md](prompts/prompt_2_negotiate.md)

#### 4. 降级路径（Bronze）

- 从轨迹中均匀采样 `N_text × 3` 帧作为候选关键帧
- 将候选帧图像送入 VLM，对照预定义的 subtask 词汇表做检索式标签匹配
- 连续相同标签的帧合并为一个 segment
- 合并后段数与 `N_text` 仍不一致时，保留物理上最自然的分割（取段数等于 `N_text` 的合并方案）
- 输出 confidence = `Bronze`，branch = `fallback`

**→ Prompt 参见：** [prompt_2_fallback.md](prompts/prompt_2_fallback.md)

---

## Stage 3 — VLM 描述生成

### 目标

在 Stage 2 确定 segment 边界后，为每个 segment 生成高质量的 subtask 自然语言描述，并做一致性自检。

### 输入

- Stage 2 产出的 `segments`（含时间戳）
- Stage 1-A 的 `anchor_objects`（hard constraint）
- Stage 1-C 的 `subtask_texts`（soft constraint，参考用）
- 轨迹图像帧

### 输出

```json
{
  "segments": [
    {
      "subtask_text": "reach toward the red cup on the left side of the table",
      "start_frame": 0,
      "end_frame": 87,
      "keyframe": 44,
      "completion_frame": 87
    }
  ],
  "self_check_passed": true,
  "self_check_retries": 0
}
```

### 开发内容

#### 1. Keyframe 提取

- 对每个 segment，提取首帧、末帧和均匀采样的 2 帧中间帧，共 4 帧作为 VLM 视觉输入
- `completion_frame` 单独记录为 segment 末帧（sub-goal 完成时的状态），用于有 Imagination Head 的模型训练

#### 2. 描述生成

- 构建包含四部分的 prompt：视觉输入（4 帧 keyframe）、全局任务指令、语义锚点（hard constraint）、Stage 1-C 的参考文字（soft constraint）
- 输出格式约束为 JSON，字段为 `subtask_text`（15 词以内）
- 对输出做格式标准化：统一动词时态、确保物体指称与锚点一致（做一次字符串替换对齐）

**→ Prompt 参见：** [prompt_3_description_gen.md](prompts/prompt_3_description_gen.md)

#### 3. 一致性自检

对完整的 subtask 序列做两项自检：

**完整性检查：** 构建 prompt 让 VLM 判断 subtask 序列是否完整覆盖了全局任务指令的所有步骤，输出 `pass` 或 `fail` 并给出失败原因。

**去重检查：** 计算相邻 subtask 文字的词汇重叠率，重叠率 > 0.6 则标记为疑似重复（可选：用 embedding 余弦相似度替代词汇重叠，阈值 > 0.85）。

不通过则对失败的 segment 重新调用描述生成（最多 2 次重试）。两次后仍不通过，将该条轨迹的 confidence 降级为 Bronze，并记录 `self_check_retries: 2`。

**→ Prompt 参见：** [prompt_3_self_check.md](prompts/prompt_3_self_check.md)

---

## Stage 4 — Grounding 扩展（可选模块）

### 目标

为每个 segment 的关键帧产出被操作物体的 bounding box，作为空间定位监督信号。

### 输入

- Stage 3 产出的 `segments`（含 keyframe 索引）
- Stage 1-A 的 `anchor_objects`（作为 GroundingDINO 的文字 query）
- 轨迹图像帧

### 输出

```json
{
  "grounding": [
    {
      "frame_idx": 44,
      "objects": [
        { "object_id": "source_0", "bbox": [120, 85, 64, 48], "confidence": 0.91 },
        { "object_id": "target_0", "bbox": [310, 210, 80, 60], "confidence": 0.87 }
      ]
    }
  ]
}
```

### 开发内容

#### 1. GroundingDINO 推理

- 输入：关键帧图像 + `anchor_objects` 的描述文字
- 对置信度低于阈值（建议 0.3）的 box 做过滤
- 对同一帧内多个候选 box 按置信度排序，取最高分的作为结果

#### 2. SAM 精化

- 用 GroundingDINO 的 box 作为 prompt 输入 SAM，得到像素级 mask
- 从 mask 重新计算紧致 bounding box（比 GroundingDINO 的 box 更准确）

#### 3. 时序一致性追踪

- 对同一物体在不同关键帧的 box 做 IoU-based 追踪，确保 `object_id` 在时序上一致
- 对短暂消失（如被遮挡）的帧做 box carry-over（沿用上一帧的 box），标记 `occluded: true`

#### 4. 可选性设计

通过配置开关 `enable_grounding: true/false` 控制是否运行此 Stage。跳过时，Stage 5 的输出中 `grounding` 字段为 `null`，不影响其他字段。

---

## Stage 5 — 质量分级输出

### 目标

整合所有 Stage 结果，产出标准化数据集格式，并按置信度分级供训练框架使用。

### 输入

- Stage 0-4 的全部产出

### 输出格式（单条记录）

```json
{
  "episode_id": "bridge_v2_ep_00142",
  "task_instruction": "put the red cup into the drawer",
  "anchor_objects": [
    { "role": "source", "description": "the red cup on the left side of the table" },
    { "role": "target", "description": "the open drawer below the counter" }
  ],
  "segments": [
    {
      "subtask_text": "reach toward the red cup on the left side of the table",
      "start_frame": 0,
      "end_frame": 87,
      "keyframe": 44,
      "completion_frame": 87,
      "grounding": null
    },
    {
      "subtask_text": "grasp the red cup and lift upward",
      "start_frame": 88,
      "end_frame": 143,
      "keyframe": 115,
      "completion_frame": 143,
      "grounding": null
    },
    {
      "subtask_text": "move the red cup to the open drawer and release",
      "start_frame": 144,
      "end_frame": 201,
      "keyframe": 172,
      "completion_frame": 201,
      "grounding": null
    }
  ],
  "confidence": "Gold",
  "loss_weight": 1.0,
  "branch": "fast",
  "gripper_quality_score": 0.92,
  "annotation_meta": {
    "stage2_branch": "fast",
    "self_check_passed": true,
    "self_check_retries": 0
  }
}
```

### 开发内容

#### 1. 数据整合与字段合并

- 合并各 Stage 产出到单条 JSON 记录
- 按 confidence 写入 `loss_weight`：Gold → 1.0，Silver → 0.5，Bronze → 0.2
- 保持与 LeRobot v2/v2.1 episode 格式兼容：subtask 标注作为额外 metadata 字段附加，不修改原有 action/observation 数据

#### 2. 数据集级别统计报告

每批数据处理完成后，自动生成统计报告，包含：

- Gold / Silver / Bronze 占比（总体及按数据集分项）
- Stage 2 各分支命中率（fast / negotiate / fallback）
- 平均 subtask 数量及分布
- 平均 segment 长度及分布
- Stage 3 自检重试率
- Gripper quality score 分布

这些统计用于发现数据集特定问题（如某数据集的 gripper signal 系统性偏差）。

#### 3. 训练集拆分

- 提供按 confidence 级别的分层导出接口
- 提供 per-dataset 的 loss weight 配置，允许在数据集级别叠加调整（如 Bronze + DROID 数据额外降权）
- 支持按 confidence 级别过滤，方便分阶段使用（预训练用全量，SFT 只用 Gold+Silver）

---

## 各阶段依赖关系与开发优先级

| 阶段 | 强依赖 | 优先级 | 建议开发顺序 |
|------|--------|--------|------------|
| Stage 0 | 无 | 最高 | 第 1 批 |
| Stage 1-B | Stage 0 | 高 | 第 1 批 |
| Stage 1-A | 无 | 高 | 第 1 批 |
| Stage 1-C | Stage 1-A | 高 | 第 1 批 |
| Stage 2 快速路径 | Stage 1 全部 | 高 | 第 2 批 |
| Stage 3 | Stage 2 | 高 | 第 2 批 |
| Stage 5 | Stage 3 | 中 | 第 2 批 |
| Stage 2 协商路径 | Stage 1 全部 | 中 | 第 3 批 |
| Stage 2 降级路径 | Stage 1 全部 | 中 | 第 3 批 |
| Stage 4 | Stage 3 | 低 | 按需开启 |

**建议开发策略：** 先跑通 Stage 0 → Stage 1-B → Stage 2 快速路径 → Stage 3 → Stage 5 的主链路，验证端到端可以产出可用数据后，再补充 Stage 1-A、Stage 1-C 的锚点注入、Stage 2 的协商和降级分支。Stage 4 作为独立模块，在主产线稳定后按需开启。
