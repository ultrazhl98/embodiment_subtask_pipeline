# 子任务分割 Pipeline 改进计划

> 基于现有 `embodiment_subtask_pipeline` 代码仓的问题分析，结合 TAPT 论文（arXiv 2605.13119）的设计思路，提出以下改进方案。

---

## 一、背景与现状

### 1.1 现有架构

```
Stage 0   轨迹预过滤
Stage 1A  视觉锚点提取（首帧）
Stage 1B  物理信号分割（纯夹爪状态）
Stage 1C  文本语义分解（LLM 生成自然语言）
Stage 2   一致性校验与对齐（Gold/Silver/Bronze 三路由）
Stage 3   VLM 描述生成 + 自检
Stage 4   Grounding（可选）
Stage 5   质量分级输出
```

### 1.2 核心问题

**P0 · Stage 1B 物理分割失效**
- `_segment_primitive` 输出纯运动学描述（`"reach forward up"`），不携带操作语义
- 非抓持操作（push/wipe/pour/rotate）夹爪全程不变，Stage 1B 无法产生有效分割边界
- 指南文档中设计了 RDP 拐点检测，但代码中完全未实现

**P0 · Stage 2 路由逻辑根本性错误**
- 路由判据 `|N_physical - N_text|` 比较的是两个不可比的量：gripper segment 数 vs 语义 subtask 数
- 标准 pick-and-place 任务（3 个语义步骤）的 gripper 切换只产生 2 段，delta 永远 ≥ 1，Gold 路径几乎不触发
- Silver 路径的 negotiate 逻辑把无语义的运动学 primitive 喂给 LLM 推断边界，实质失效
- Bronze 路径的 `_reconcile_run_count` 用"在最长 run 处对半拆"做边界插入，是纯几何逻辑，语义错误

**P1 · 动词表三处不一致**
- `config.py` 的 `ALLOWED_VERBS`（12 个）缺少 `pick up`、`transport`、`wipe`、`pour`
- `stage1b_physical.py` 的 primitive 描述是运动学短语，与语义动词体系完全脱离
- `prompts.py` 里动词列表硬编码，与 `config.py` 不同步
- `Stage1C` 的动词校验器对多词动词（`pick up`、`move to`）的前缀匹配逻辑有 bug

**P1 · Stage 1A 首帧锚点的根本局限**
- 假设"首帧包含所有会被操作的物体"，在机器人需要移动到操作区域、多物体顺序操作、视角变化等场景下失效
- 锚点物体描述不能可靠地对应到每个 subtask 真实操作的物体

---

## 二、改进目标

1. 与论文 TAPT pipeline 的结构对齐：全局理解 → 物理分割 → VLM 标注 → 质量过滤
2. 引入封闭的核心原语集（primitive vocabulary），保证跨数据集标注一致性
3. 用事件帧（event frame）替代计数比较作为对齐依据，消除 Stage 2 的根本性错误
4. 将 Stage 2 退化为轻量规则过滤层，去掉 LLM 调用
5. 补齐全局视频理解，解决首帧锚点的局限

---

## 三、新架构

```
Stage 0    轨迹预过滤（基本保留）
Stage 0.5  全局视频理解（新增，替代 Stage 1A）
Stage 1B   事件帧提取 + 原语分类（重构）
Stage 1C   Per-segment 描述生成（重构，原 Stage 3 降级）
Stage 2    规则质量过滤（大幅简化）
Stage 4    Grounding（可选，保留）
Stage 5    输出（补充 progress 字段）
```

**移除：** Stage 1A（首帧锚点）、Stage 3（VLM 描述生成+自检，职责并入 Stage 1C）

---

## 四、各 Stage 改进细节

### 4.1 Stage 0：轨迹预过滤（基本保留）

现有实现基本合理，保留以下内容：
- Gripper 信号质量检测（抖动检测 + 无效切换过滤）
- 图像质量检测（亮度 / 模糊 / frozen frame）
- 轨迹长度异常检测（MAD 方法）

**小修改：** `gripper_quality_score` 的计算权重（当前 `0.6 * jitter_ratio + 0.7 * invalid_ratio`）在某些数据集上过于激进，建议改为可配置参数。

---

### 4.2 Stage 0.5：全局视频理解（新增）

**替代 Stage 1A**，在 Stage 0 之后、Stage 1B 之前执行。

**输入**
- 完整轨迹视频（均匀采样 8-12 帧覆盖全程，而非只取首帧）
- 任务指令文本

**输出（写入 `episode.meta["global_summary"]`）**
```json
{
  "task_intent": "机器人将红色杯子从桌面放入抽屉",
  "objects": [
    { "role": "source", "description": "red cup on the left side of table" },
    { "role": "target", "description": "open drawer below the counter" }
  ],
  "key_events": ["gripper closes around cup", "cup lifted", "drawer opened", "cup placed inside"],
  "scene_context": "kitchen countertop with drawer on the right"
}
```

**设计要点**

- 采样帧覆盖全程，不依赖首帧，解决机器人移动和视角变化问题
- `objects` 字段替代 Stage 1A 的 anchor_objects，供 Stage 1C 做物体指称约束
- `key_events` 为 Stage 1B 提供语义参考，辅助非抓持操作的边界判断
- VLM prompt 需要明确要求输出 JSON 格式，校验逻辑参考现有 Stage 1A 的 `_validate`

**新增配置（`Stage05Config`）**
```python
@dataclass
class Stage05Config:
    sample_count: int = 10          # 全局理解采样帧数
    max_objects: int = 4            # 最多识别物体数
    enable: bool = True             # 可关闭，退化为无全局理解模式
```

---

### 4.3 Stage 1B：事件帧提取 + 原语分类（重构）

**目标：** 输出 `(primitive_label, start_frame, end_frame)` 列表，primitive_label 是语义标签而非运动学描述。

#### 4.3.1 封闭原语集

从现有动词体系中提取核心原语集，作为跨数据集标准：

```python
PHYSICAL_PRIMITIVES = {
    # 无接触
    "reach":     {"gripper": "open",  "motion": "approach"},
    "retract":   {"gripper": "open",  "motion": "retreat"},
    # 抓持
    "pick_up":   {"gripper": "close_event + lift"},
    "transport": {"gripper": "close", "motion": "lateral"},
    "place":     {"gripper": "close_event -> open_event"},
    # 非抓持接触
    "push":      {"gripper": "close", "motion": "forward_force"},
    "pull":      {"gripper": "close", "motion": "backward_force"},
    "press":     {"gripper": "close", "motion": "downward"},
    "wipe":      {"gripper": "close", "motion": "surface_sweep"},
    # 铰接
    "open":      {"gripper": "close", "motion": "articulated"},
    "close":     {"gripper": "close", "motion": "articulated"},
    "rotate":    {"gripper": "close", "motion": "rotational"},
    "insert":    {"gripper": "close", "motion": "insertion_axis"},
    "pour":      {"gripper": "close", "motion": "tilt"},
}
```

每个原语有明确的 gripper 状态约束，可用于 Stage 2 的 sanity check。

#### 4.3.2 语义 primitive 推断（替换 `_segment_primitive`）

```python
def infer_semantic_primitive(
    gripper_state: str,       # "open" | "close"
    is_close_event: bool,     # 本段起始帧发生夹爪关闭
    is_open_event: bool,      # 本段起始帧发生夹爪打开
    disp_norm: float,         # 段内总位移量
    disp_vertical: float,     # 垂直方向位移（用于区分 lift vs transport）
    cfg: Stage1bConfig,
) -> str:
    if is_close_event:
        if disp_vertical > cfg.lift_threshold:
            return "pick_up"
        return "grasp"          # 原地抓取（后续接 open 操作等）
    if is_open_event:
        return "place"
    if gripper_state == "close":
        if disp_norm > cfg.transport_disp_threshold:
            return "transport"
        return "hold"           # 短距离调整，通常合并入相邻段
    if gripper_state == "open":
        return "reach"
    return "unknown"
```

#### 4.3.3 非抓持操作分割（补实现 RDP）

对夹爪全程 close 的轨迹段，使用 RDP 拐点检测作为额外分割信号：

```python
def rdp_boundaries(xyz: np.ndarray, epsilon: float) -> list[int]:
    """返回 RDP 简化后被移除的原始点索引（即运动方向拐点）。"""
    from rdp import rdp
    mask = rdp(xyz, epsilon=epsilon, return_mask=True)
    return sorted(np.where(~mask)[0].tolist())

def detect_nonprehensile_boundaries(
    xyz: np.ndarray,
    binary: np.ndarray,
    cfg: Stage1bConfig,
) -> list[int]:
    """对夹爪全程 close 的段，用 RDP + 速度突变检测分割边界。"""
    # 仅在夹爪极少 open 的轨迹上激活（非抓持操作特征）
    open_ratio = binary.mean()
    if open_ratio > cfg.nonprehensile_open_ratio_threshold:
        return []
    rdp_bounds = rdp_boundaries(xyz, cfg.rdp_epsilon)
    # 速度突变点（EEF 速度的局部极小值）
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    speed_minima = _local_minima(speed, cfg.speed_minima_window)
    all_bounds = sorted(set(rdp_bounds) | set(speed_minima))
    return _enforce_min_gap(all_bounds, cfg.min_segment_frames)
```

**新增配置参数**
```python
# Stage1bConfig 新增
lift_threshold: float = 0.05              # 垂直位移阈值，区分 pick_up vs grasp
transport_disp_threshold: float = 0.08    # 水平位移阈值，区分 transport vs hold
use_rdp_for_nonprehensile: bool = True    # 非抓持轨迹启用 RDP
rdp_epsilon: float = 0.02                 # RDP 简化精度
nonprehensile_open_ratio_threshold: float = 0.1  # 判断为非抓持轨迹的 open 帧占比上限
speed_minima_window: int = 5              # 速度极小值检测窗口
```

---

### 4.4 Stage 1C：Per-segment 描述生成（重构）

**目标：** 对 Stage 1B 切好的每个片段，结合全局摘要和关键帧图像，生成包含具体物体指称的自然语言描述。职责合并原 Stage 3 的描述生成功能。

**输入（每个 segment）**
- `primitive_label`（来自 Stage 1B）
- 3-4 帧关键帧图像（首帧 + 末帧 + 均匀中间帧）
- `global_summary`（来自 Stage 0.5，包含物体列表和场景上下文）
- 任务指令

**处理流程**

```
primitive_label → 模板选择 → VLM 填槽（object / target）→ subtask_text
```

**模板体系（替代现有的开放生成）**

```python
PRIMITIVE_TEMPLATES = {
    "reach":     "reach for the {object}",
    "retract":   "retract the arm from {object}",
    "pick_up":   "pick up the {object}",
    "transport": "transport the {object} to {target}",
    "place":     "place the {object} {prep} {target}",  # prep: on/into/onto
    "push":      "push the {object} {direction}",
    "pull":      "pull the {object} toward {direction}",
    "press":     "press the {object}",
    "wipe":      "wipe the {target} with the {object}",
    "open":      "open the {object}",
    "close":     "close the {object}",
    "rotate":    "rotate the {object}",
    "insert":    "insert the {object} into {target}",
    "pour":      "pour {object} into {target}",
}
```

VLM 的任务从"生成描述"变为"识别当前片段的 object 和 target，从 global_summary.objects 中选择或描述"，是一个更受约束的分类/填槽任务，稳定性更高。

**校验规则**

- object 描述必须与 `global_summary.objects` 中的某个条目有词汇重叠（防止幻觉）
- 对 `place`、`insert`、`pour` 等需要 target 的动词，target 字段必须非空
- 描述长度不超过 15 词

**移除**
- 原 Stage 3 的 `self_check`（LLM 自检）：改为规则校验，不再需要 LLM 做完整性检查
- 原 Stage 3 的 `_merge_duplicate_segments`：由 Stage 2 的规则过滤在上游处理
- `_align_object_reference` 的硬编码同义词表：由 global_summary 约束替代

---

### 4.5 Stage 2：规则质量过滤（大幅简化）

**目标：** 从"对齐"改为"过滤"，不做 LLM 调用，纯规则检查，结果是置信度标记而不是修复。

**移除**
- Gold/Silver/Bronze 三路由逻辑（`route` 函数）
- `negotiate_path`（LLM 推断边界）
- `fallback_path`（逐帧 VLM 视觉检索）
- `_reconcile_run_count`（几何 reconcile）

**保留并重构：事件帧对齐**

Stage 2 唯一的对齐逻辑是：用事件帧确定最终边界，然后做 sanity check。

```python
def align_by_event_frames(
    subtask_texts: list[str],
    stage1b: dict,
    total_frames: int,
) -> tuple[list[Segment], str]:
    """
    以夹爪事件帧为 hard anchor 确定 segment 边界。
    返回 (segments, confidence)，confidence ∈ {"Gold", "Bronze"}。
    """
    event_frames = extract_event_frames(stage1b)  # grasp/release 帧索引
    n = len(subtask_texts)
    need = n - 1

    if len(event_frames) >= need:
        # 事件帧足够：直接取前 need 个作边界
        boundaries = [0] + sorted(event_frames)[:need] + [total_frames - 1]
        confidence = "Gold"
    else:
        # 事件帧不足（非抓持操作为主）：以已有事件帧为 anchor，
        # 剩余边界按帧数均匀填充（Bronze，建议人工抽检）
        boundaries = fill_remaining_boundaries(event_frames, need, total_frames)
        confidence = "Bronze"

    segments = build_segments(subtask_texts, boundaries)
    return segments, confidence
```

**规则质量检查**

```python
def rule_check(segment: Segment, stage1b: dict, cfg: Stage2Config) -> tuple[bool, str]:
    """
    三项规则检查，全部通过才认为 valid。
    """
    prim = stage1b["primitive_by_frame"].get(segment.start_frame, "unknown")
    gripper_in_seg = stage1b["gripper_binary"][segment.start_frame:segment.end_frame+1]

    # 规则 1：primitive 与 gripper 状态一致性
    if prim in ("pick_up", "transport", "place", "push", "pull", "press", "wipe",
                "open", "close", "rotate", "insert", "pour"):
        if gripper_in_seg.mean() < 0.3:   # 预期 close 但实际 open 占多数
            return False, f"gripper state mismatch for primitive '{prim}'"
    if prim in ("reach", "retract"):
        if gripper_in_seg.mean() > 0.7:   # 预期 open 但实际 close 占多数
            return False, f"gripper state mismatch for primitive '{prim}'"

    # 规则 2：时长合理性
    duration = segment.end_frame - segment.start_frame
    if duration < cfg.min_segment_frames:
        return False, f"segment too short: {duration} frames"
    if duration > cfg.max_segment_frames:
        return False, f"segment too long: {duration} frames"

    # 规则 3：subtask_text 与 primitive 动词一致性
    expected_verb = PRIMITIVE_TO_VERB.get(prim)
    if expected_verb and not segment.subtask_text.lower().startswith(expected_verb):
        return False, f"verb mismatch: expected '{expected_verb}', got '{segment.subtask_text[:20]}'"

    return True, ""
```

**新增配置（替换旧 Stage2Config）**
```python
@dataclass
class Stage2Config:
    min_segment_frames: int = 10
    max_segment_frames: int = 400
    # 旧参数全部移除：delta_tolerance, fallback_sample_factor,
    # fallback_low_conf_threshold, negotiate_max_primitive_tokens
```

**置信度语义重新定义**

| 置信度 | 含义 | loss_weight |
|--------|------|-------------|
| Gold   | 边界全部来自夹爪事件帧，规则检查全部通过 | 1.0 |
| Bronze | 边界部分或全部由均匀填充补齐（非抓持轨迹为主） | 0.3 |
| Flagged | 规则检查失败，建议人工复查，不参与训练 | 0.0 |

Silver 置信度移除（原 Silver 路径的中间状态无明确语义）。

---

### 4.6 Stage 5：输出补充 progress 字段

在每个 `Segment` 的输出里补充 per-frame 进度信号，用于支持 Steerable VLA 的 progress head 训练（参考 TAPT 论文设计）：

```python
def add_progress_signal(segments: list[Segment]) -> list[Segment]:
    """段内线性填充进度信号 0 → 1。"""
    for seg in segments:
        n = seg.end_frame - seg.start_frame + 1
        seg.progress = np.linspace(0.0, 1.0, n).tolist()
    return segments
```

`Segment.to_dict()` 补充 `progress` 字段输出。

---

## 五、动词体系统一

### 5.1 统一 source of truth

`config.py` 里声明两套词汇，其他所有地方动态引用，不允许硬编码：

```python
# 物理信号层（Stage 1B 内部使用，不对外暴露）
PHYSICAL_PRIMITIVES: list[str] = [
    "reach", "retract", "pick_up", "transport", "place",
    "push", "pull", "press", "wipe",
    "open", "close", "rotate", "insert", "pour",
    "hold", "unknown",
]

# 语义标注层（Stage 1C 描述生成的动词约束，与指南对齐）
ALLOWED_VERBS: list[str] = [
    "reach", "retract", "move to",
    "pick up", "place", "transport", "hand over",
    "push", "pull", "press", "wipe",
    "open", "close", "rotate", "insert", "pour",
]

# primitive → 自然语言动词的映射（用于 Stage 2 sanity check）
PRIMITIVE_TO_VERB: dict[str, str] = {
    "reach":     "reach",
    "retract":   "retract",
    "pick_up":   "pick up",
    "transport": "transport",
    "place":     "place",
    "push":      "push",
    "pull":      "pull",
    "press":     "press",
    "wipe":      "wipe",
    "open":      "open",
    "close":     "close",
    "rotate":    "rotate",
    "insert":    "insert",
    "pour":      "pour",
}
```

### 5.2 修复多词动词校验 bug

`stage1c_text.py` 的 `_make_validator` 和 `stage3_describe.py` 的 `_make_desc_validator` 中，动词前缀匹配改为支持多词：

```python
def _starts_with_allowed_verb(text: str, verbs: set[str]) -> bool:
    """支持多词动词（pick up、move to、hand over）的前缀匹配。"""
    words = text.strip().lower().split()
    for n in (2, 1):  # 优先匹配两词动词
        prefix = " ".join(words[:n])
        if prefix in verbs:
            return True
    return False
```

### 5.3 prompts.py 去硬编码

`TEXT_DECOMP_SYSTEM`、`DESC_SYSTEM` 里硬编码的动词列表改为从配置动态注入：

```python
def build_text_decomp_prompt(task_instruction: str, anchors, allowed_verbs: list[str]):
    vocab_str = ", ".join(allowed_verbs)  # 从配置传入
    ...
```

---

## 六、数据集配置补充

现有 `configs/datasets/` 只有 `austin_buds.yaml`，需要补充目标数据集的配置文件。

每个数据集 profile 需要确认以下字段（以 BridgeData V2 为例）：

```yaml
# configs/datasets/bridge_v2.yaml
type: lerobot
state_key: observation.state
eef_xyz_dims: [0, 1, 2]        # 需要按实际 state 格式确认
gripper_key: action
gripper_dim: 6                  # 需要确认
gripper_open_is_high: true      # 需要确认极性
gripper_min: 0.0
gripper_max: 1.0
image_camera: exterior_image_1_left  # 需要确认相机名

stage1b:
  rdp_epsilon: 0.015            # 按 BridgeData 的动作尺度调整
  transport_disp_threshold: 0.06

stage2:
  min_segment_frames: 8
  max_segment_frames: 350
```

待补充的数据集：`bridge_v2`、`droid`、`libero`、`agibot`、`calvin`、`robotwin`。

---

## 七、实施顺序

### Phase 1：基础修复（不改接口，优先解决 P0）

1. `stage1b_physical.py`：实现 `infer_semantic_primitive`，输出语义标签
2. `stage1b_physical.py`：实现 `rdp_boundaries` 和 `detect_nonprehensile_boundaries`
3. `config.py`：统一动词表，修复多词动词校验 bug
4. `prompts.py`：去掉硬编码动词列表，改为从配置注入

### Phase 2：Stage 2 重构

5. `stage2_align.py`：用 `align_by_event_frames` 替换三路由逻辑
6. `stage2_align.py`：实现 `rule_check`，替换原有 `validate_segment`
7. 更新 `Stage2Config`，移除废弃参数
8. 更新 `pipeline.py` 编排逻辑

### Phase 3：新增 Stage 0.5

9. 新增 `stage05_global.py`，实现全局视频理解
10. `prompts.py`：新增全局视频理解 prompt
11. `pipeline.py`：插入 Stage 0.5 调用，`episode.meta` 写入 `global_summary`

### Phase 4：Stage 1C 重构

12. `stage1c_text.py`：重构为 per-segment 描述生成，引入模板体系
13. `prompts.py`：新增 per-segment 描述填槽 prompt
14. 原 `stage3_describe.py` 职责合并入新 Stage 1C，旧文件可保留兼容层或删除

### Phase 5：补全与收尾

15. `stage5_output.py`：补充 progress 字段
16. `data/types.py`：`Segment` 新增 `progress` 字段
17. 补充各数据集 YAML 配置
18. 更新 `tests/` 覆盖新逻辑

---

## 八、改动后的接口契约

### Episode 输入契约（不变）

现有 `Episode.validate()` 契约不变，下游所有 Stage 继续面向 `Episode` 编程。

### 新增 `episode.meta` 字段

| 字段 | 写入 Stage | 说明 |
|------|-----------|------|
| `global_summary` | Stage 0.5 | 全局视频理解结果 |
| `stage1b.primitives` | Stage 1B | 语义原语标签列表（原为运动学描述） |
| `stage1b.event_frames` | Stage 1B | 夹爪事件帧索引列表 |
| `stage2.confidence` | Stage 2 | `"Gold"` \| `"Bronze"` \| `"Flagged"` |
| `stage2.rule_failures` | Stage 2 | 未通过规则校验的 segment 信息 |

### Segment 输出格式（新增 progress）

```json
{
  "subtask_text": "pick up the red cup",
  "primitive_label": "pick_up",
  "start_frame": 45,
  "end_frame": 112,
  "keyframe": 78,
  "completion_frame": 112,
  "progress": [0.0, 0.015, 0.030, "...", 1.0],
  "grounding": null
}
```

---

## 九、与 TAPT 论文的对比

| 维度 | TAPT 论文 | 改进后 pipeline |
|------|-----------|----------------|
| 全局视频理解 | 有 | 有（Stage 0.5，新增） |
| 分割信号 | 夹爪 + EEF 速度/角速度 | 夹爪 + RDP 拐点（补实现） |
| 标签体系 | 封闭工具族标签 `g` | 封闭原语集 + 自然语言描述 |
| 物体识别 | 标注时同步识别 | Per-segment VLM 填槽 |
| 进度信号 | 有 | 有（Stage 5，补充） |
| 一致性验证 | 单次决策 + 置信度降级 | 规则校验 + 置信度标记 |
| 质量分级 | 无 | Gold / Bronze / Flagged |
| Grounding | 无 | 有（Stage 4，可选） |
| LLM 调用次数 | 每段 1 次 | Stage 0.5 × 1 + Stage 1C × N_seg |
