# L2 子任务指令标注指南

> 面向开源操作轨迹数据集（BridgeData V2 / DROID / AgiBot / RoboMIND 等）的  
> 子任务切割与语言标注系统方案

---

## 目录

1. [切割依据](#1-切割依据)
2. [动词体系](#2-动词体系)
3. [模板格式](#3-模板格式)
4. [自动化 Pipeline](#4-自动化-pipeline)

---

## 1. 切割依据

### 1.1 核心原则

子任务边界对应机器人与环境的**接触状态变化**（contact state transition），而非任意时间分段。一段连续轨迹中，每当接触关系发生本质改变（从自由运动到抓取、从抓取到释放等），即产生一个子任务边界。

> **最小化分解原则**：专注于完成任务的必要且充分的动作，不要过度分解。偏向原子但必要的步骤，单纯的重新定位或空接近运动不被视为独立子任务，而是合并到相邻 clip 中。（RoboInter, CycleVLA）

---

### 1.2 边界信号（优先级从高到低）

| 优先级 | 信号类型 | 具体表现 | 说明 |
|--------|----------|----------|------|
| ① | **Gripper close 事件** | `gripper_state` open → close | 抓取物体，最强边界信号 |
| ② | **Gripper open 事件** | `gripper_state` close → open | 释放物体，最强边界信号 |
| ③ | **EEF 轨迹几何拐点** | RDP 算法检测偏离简化路径超过阈值 ε 的点 | 运动方向突变，如提起→水平移动 |
| ④ | **EEF 加速度突变** | `‖ä(t)‖₂ > α`（阈值需按数据集调整） | 速度骤降通常对应对齐/放置时刻 |
| ⑤ | **VLM 语义验证** | 相邻候选帧 VLM 输出标签相同则合并 | 去除运动学假阳性 |

---

### 1.3 典型 Pick-and-Place 轨迹的边界分布

```
时间轴 →

[gripper: open ────────][ close ──────][close(carry)────────────][ open ──][ open ]
   ↑                        ↑                                       ↑
 开始接近目标         抓取（close 事件）                       释放（open 事件）
   │                        │                                       │
   ▼                        ▼                                       ▼
┌────────────┐     ┌──────────────┐     ┌─────────────────────┐  ┌───────────┐
│   reach    │     │   pick up    │     │     transport        │  │   place   │
│（可合并）   │     │              │     │                     │  │           │
└────────────┘     └──────────────┘     └─────────────────────┘  └───────────┘
```

---

### 1.4 关于 reach / retract 的设计选择

存在两种流派，根据数据集情况选择：

**合并派**（π0.5、AtomVLA、GigaBrain）  
- 将 reach 并入 pick，retract 并入 place  
- 每次 pick-and-place 产生 **2–3 个**子任务标签  
- 适用于**短轨迹**（< 5s）、单次操作

**分离派**（RoboSubtaskNet、RoboInter）  
- 序列：`reach → pick → transport → place → retract`  
- 每次 pick-and-place 产生 **4–5 个**子任务标签  
- 适用于**长轨迹**（> 10s）、多物体操作，边界信息更丰富

> **实践建议**：对 BridgeData V2、DROID 等轨迹长度差异大的数据集，建议按阈值分类处理：短轨迹（< 5s）用合并派，长轨迹（> 10s）用分离派。关键原则是保证每个子任务段内 gripper 状态单调（不发生内部翻转）。

---

### 1.5 gripper 状态的鲁棒检测

原始 gripper 信号存在遥操作噪声和短暂抖动，需做滤波处理：

```python
# 多阈值投票方案（来自 CycleVLA）
THRESHOLDS = [0.028, 0.03, 0.032]

def detect_gripper_state(gripper_signal: np.ndarray) -> np.ndarray:
    """
    Returns per-frame state: -1 (close), 0 (idle), +1 (open)
    """
    votes = []
    for thr in THRESHOLDS:
        state = np.where(gripper_signal > thr, 1, -1)
        votes.append(state)
    averaged = np.mean(votes, axis=0)
    return np.round(averaged).astype(int)

def find_gripper_transitions(states: np.ndarray, min_gap: int = 30) -> list[int]:
    """
    Find frame indices where gripper state changes.
    min_gap: minimum frames between consecutive boundaries (default 30).
    """
    transitions = []
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            if not transitions or (i - transitions[-1]) >= min_gap:
                transitions.append(i)
    return transitions
```

---

## 2. 动词体系

### 2.1 分类框架

动词按**末端执行器与物体的接触类型**分为四大类，共约 15–20 个核心 primitive：

---

### 2.2 核心原语（Core Primitives）

**类别 A：无接触自由运动**（gripper 不接触物体，EEF 在自由空间移动）

| 动词 | Gripper 状态 | 典型时长 | 说明 |
|------|-------------|----------|------|
| `reach` | open | 1–3s | 接近目标物体，为抓取做准备 |
| `retract` | open/close | 0.5–2s | 收回手臂，撤离操作区域 |
| `move to` | open | 1–4s | 空载移动到某位置（无特定目标物） |

**类别 B：抓持操作（Prehensile）**（gripper 夹持物体，物体随手臂运动）

| 动词 | Gripper 状态 | 典型时长 | 说明 |
|------|-------------|----------|------|
| `pick up` | open → close → lift | 1–4s | 包含：接近 + 闭夹 + 提起 |
| `place` / `put down` | closed → lower → open | 1–3s | 包含：下降 + 放置 + 松开 |
| `transport` | closed（持物） | 2–8s | 持物水平移动，连接 pick 和 place 的过渡段 |
| `hand over` | closed → open | 1–3s | 将物体转交给另一只手或另一个机器人 |

> `transport` 在简短轨迹中通常不单独标注，直接并入 `pick up` 或 `place` 的语义中。

**类别 C：接触非抓持（Non-Prehensile Contact）**（接触物体但不夹持，靠力/摩擦作用）

| 动词 | Gripper 状态 | 典型时长 | 说明 |
|------|-------------|----------|------|
| `push` | closed/open | 0.5–3s | 单向推动物体改变位置 |
| `pull` | closed | 0.5–3s | 向自身方向拉动物体 |
| `press` / `tap` | closed | 0.2–1s | 按压（按钮、开关） |
| `wipe` / `sweep` | closed | 2–5s | 在表面上擦拭或扫动 |

**类别 D：铰接/约束运动（Articulated）**（操作对象有约束自由度，如门轴、抽屉导轨）

| 动词 | Gripper 状态 | 典型时长 | 说明 |
|------|-------------|----------|------|
| `open` | closed | 1–4s | 打开门/抽屉/容器盖/瓶盖 |
| `close` | closed | 1–3s | 关闭门/抽屉/盖子 |
| `rotate` / `turn` | closed | 1–4s | 旋拧、扭转（旋钮/瓶盖/开关） |
| `insert` | closed | 1–3s | 插入/装配（插座/插销/组装件） |
| `pour` | closed | 2–5s | 倾倒液体或小件物体 |
| `scoop` | closed | 1–3s | 舀取 |

---

### 2.3 扩展集（Extended Primitives）

出现在部分长时域/家务类数据集中，频率较低：

| 动词 | 典型场景 |
|------|----------|
| `fold` | 折叠衣物/布料 |
| `shake` | 摇动容器（混合液体） |
| `cut` | 使用刀具切割（工具使用） |
| `stir` | 使用器具搅拌 |
| `hang` | 将物体挂置（如衣架、挂钩） |
| `stack` | 将物体堆叠放置 |
| `peel` | 揭开/剥开 |
| `squeeze` | 挤压（软性容器） |

---

### 2.4 各数据集动词覆盖对比

| 数据集 / 论文 | 动词集大小 | 核心词汇 |
|--------------|-----------|----------|
| RoboInter（ICLR 2026） | 15 | pick, place, push, pull, press, wipe, open, close, rotate, insert, pour, reach, retract, transport, fold |
| RH20T-P | 10+ | move, pick, place, pull, rotate, press + gripper-based |
| RoboSubtaskNet | 7 | reach, pick, move, pour, give, place, wipe, retract |
| AtomVLA | 5（粗） | pick up, place, move arm to, open/close, push |
| GigaBrain | ~8 | pick, place, open, push + 场景特定 |
| π0.5 | 开放词汇 | 以 pick/place/rearrange 为主，自然语言描述 |

---

## 3. 模板格式

### 3.1 统一模板结构

```
[动词]  +  [物体]  +  [介词短语（可选）]
```

| 字段 | 规范 | 示例 |
|------|------|------|
| **动词** | 从词汇表中严格选取，不自由生成 | `pick up` |
| **物体** | 具体可辨认的名词，含颜色/形状等区分属性 | `the red cup` / `the white mug` |
| **介词短语** | `into [container]` / `on [surface]` / `to [location]` | `into the basket` / `on the plate` |

---

### 3.2 各动词的标准模板示例

**无接触运动**

```
reach for the [object]
reach toward the [location]
retract the arm
move to the [location]
```

**抓持操作**

```
pick up the [object]
pick up the [color] [object]
place the [object] on the [target_surface]
place the [object] into the [container]
put the [object] down
transport the [object] to the [location]
```

**接触非抓持**

```
push the [object] to the [direction/location]
pull the [object] toward [direction]
press the [button/switch]
wipe the [surface] with the [tool]
sweep the [surface]
```

**铰接/约束**

```
open the [door/drawer/lid/bottle]
close the [door/drawer/lid]
rotate the [knob/cap/handle]
insert the [object] into the [slot/socket]
pour the [liquid/contents] into the [container]
```

---

### 3.3 规范性要求

**应该做：**
- 物体名称具体化，带有区分性属性（`white mug` 而非 `cup`）
- 容器/目标位置在 place 类动词中必须标注（`place ... into the basket`）
- 介词选择精确：`into`（容器内部）/ `on`（表面）/ `onto`（放上去的动作感）
- 使用第一人称视角描述目标位置方向（`in front of the robot`），避免相对左右（`move left toward`）

**不应该做：**
- ❌ 自由生成未在词汇表中的动词（如 `grasp`——统一用 `pick up`）
- ❌ 在子任务描述中包含动作路径细节（路径信息属于 L3/L4 层）
- ❌ 一个子任务描述中包含两个语义动作（`pick up the cup and place it`——这是两个子任务）
- ❌ 过度抽象（`manipulate the object`——无法指导执行）

---

### 3.4 边界情况处理

| 情况 | 处理方式 |
|------|----------|
| open/close drawer 后有 pick 动作 | 分两段：`open the drawer` → `pick up the [object]` |
| 双臂协作 | 以主操作臂为主体，另一臂描述为辅助（`hold the bowl while ...`） |
| 工具使用 | 动词描述工具施加的动作（`wipe the table with the sponge`） |
| 失败重试 | 过滤掉失败段，或单独标注为 `retry [verb]` |
| 轨迹开头/结尾空载 | 过滤掉静止段（速度近零超过 N 帧），不纳入子任务标注 |

---

## 4. 自动化 Pipeline

### 4.1 总体架构

```
原始轨迹（action + observation）
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: 运动学自动切割      │  ← 信号驱动，无需视觉
│  Gripper 状态检测 +          │
│  EEF 轨迹 RDP 拐点提取       │
└────────────┬────────────────┘
             │ 候选边界帧集合
             ▼
┌─────────────────────────────┐
│  Stage 2: VLM 语义标注       │  ← 视觉+语言驱动
│  关键帧截图 → VLM 生成标签   │
│  相邻相同标签合并             │
└────────────┬────────────────┘
             │ 带标签的子任务段
             ▼
┌─────────────────────────────┐
│  Stage 3: 质检与修正（可选）  │  ← 人工/规则
│  人工抽样 QA                 │
│  规则约束过滤                 │
└─────────────────────────────┘
```

---

### 4.2 Stage 1：运动学自动切割

**Step 1.1 — 静止段过滤**

```python
def filter_idle_frames(actions: np.ndarray, vel_threshold: float = 0.005) -> tuple[int, int]:
    """
    Find the first and last frames where EEF is actually moving.
    Returns (start_frame, end_frame) of the active region.
    """
    eef_vel = np.linalg.norm(np.diff(actions[:, :3], axis=0), axis=1)
    active = np.where(eef_vel > vel_threshold)[0]
    if len(active) == 0:
        return 0, len(actions) - 1
    return int(active[0]), int(active[-1]) + 1
```

**Step 1.2 — Gripper 状态跳变检测**

```python
import numpy as np

GRIPPER_THRESHOLDS = [0.028, 0.03, 0.032]
MIN_GAP_FRAMES = 30  # 相邻边界最小间距

def detect_gripper_boundaries(gripper_signal: np.ndarray) -> list[int]:
    """
    Detect gripper state change frames using multi-threshold voting.
    gripper_signal: 1D array, values in [0, 1], smaller = more closed.
    """
    votes = []
    for thr in GRIPPER_THRESHOLDS:
        state = (gripper_signal > thr).astype(int)  # 1=open, 0=closed
        votes.append(state)
    
    avg_state = np.mean(votes, axis=0)
    binary_state = (avg_state > 0.5).astype(int)
    
    # Find transitions
    transitions = []
    for i in range(1, len(binary_state)):
        if binary_state[i] != binary_state[i - 1]:
            transitions.append(i)
    
    # Enforce minimum gap
    filtered = []
    for t in transitions:
        if not filtered or (t - filtered[-1]) >= MIN_GAP_FRAMES:
            filtered.append(t)
    
    return filtered
```

**Step 1.3 — EEF 轨迹几何拐点（RDP）**

```python
def rdp_keyframes(eef_xyz: np.ndarray, epsilon: float = 0.02) -> list[int]:
    """
    Ramer-Douglas-Peucker algorithm to find geometric waypoints.
    eef_xyz: (T, 3) array of end-effector positions.
    epsilon: distance threshold in meters.
    Returns list of keyframe indices.
    """
    def rdp_recursive(points, indices, start, end, eps):
        if end - start < 2:
            return []
        
        line_start = points[start]
        line_end = points[end]
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-8:
            dists = np.linalg.norm(points[start:end+1] - line_start, axis=1)
        else:
            line_unit = line_vec / line_len
            vecs = points[start:end+1] - line_start
            proj = np.dot(vecs, line_unit)[:, None] * line_unit
            dists = np.linalg.norm(vecs - proj, axis=1)
        
        max_idx = np.argmax(dists) + start
        max_dist = dists[max_idx - start]
        
        if max_dist > eps:
            left = rdp_recursive(points, indices, start, max_idx, eps)
            right = rdp_recursive(points, indices, max_idx, end, eps)
            return left + [max_idx] + right
        return []
    
    T = len(eef_xyz)
    inner = rdp_recursive(eef_xyz, list(range(T)), 0, T - 1, epsilon)
    return sorted(set([0] + inner + [T - 1]))


def merge_boundaries(gripper_boundaries: list[int],
                     rdp_boundaries: list[int],
                     min_gap: int = 30) -> list[int]:
    """
    Merge gripper and geometric boundaries, enforce min_gap.
    Gripper boundaries have priority.
    """
    all_boundaries = sorted(set(gripper_boundaries + rdp_boundaries))
    
    filtered = []
    for b in all_boundaries:
        if not filtered or (b - filtered[-1]) >= min_gap:
            filtered.append(b)
    
    return filtered
```

---

### 4.3 Stage 2：VLM 语义标注

**Step 2.1 — 关键帧截图提取**

对每个候选段，取**首帧 + 中间帧 + 末帧**作为 VLM 输入：

```python
def extract_segment_frames(observations: np.ndarray,
                           boundaries: list[int]) -> list[dict]:
    """
    For each segment defined by consecutive boundary pairs,
    extract start / mid / end frames.
    observations: (T, H, W, 3) uint8 image array.
    """
    segments = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        mid = (start + end) // 2
        segments.append({
            "segment_id": i,
            "start_frame": start,
            "end_frame": end,
            "frames": {
                "start": observations[start],
                "mid": observations[mid],
                "end": observations[end - 1],
            }
        })
    return segments
```

**Step 2.2 — VLM 标注 Prompt**

```python
PRIMITIVE_VOCAB = [
    "reach", "retract", "move to",
    "pick up", "place", "transport", "hand over",
    "push", "pull", "press", "wipe",
    "open", "close", "rotate", "insert", "pour",
]

ANNOTATION_PROMPT = """
You are annotating a robot manipulation trajectory segment.

Overall task: {task_instruction}
This is segment {segment_id} of the trajectory (frames {start}–{end}).
You are given 3 images: the start frame, middle frame, and end frame of this segment.

Instructions:
1. Select exactly ONE primitive skill from the vocabulary below:
   {vocab_list}

2. Fill in the template:
   [verb] the [object] [into/on/onto target (if applicable)]

Rules:
- Object names must be specific (e.g., "red cup", "white plate"), not generic ("object", "thing")
- For place/insert actions, the target container or surface is required
- Do NOT include path or trajectory details in the description
- Do NOT combine two actions in one label (one segment = one primitive)
- Focus on what the robot IS DOING in this segment, not what it will do next

Output format (JSON only, no other text):
{{
  "primitive": "<verb from vocabulary>",
  "label": "<complete label string>",
  "confidence": <0.0–1.0>
}}
"""

def build_prompt(task_instruction: str, segment: dict) -> str:
    return ANNOTATION_PROMPT.format(
        task_instruction=task_instruction,
        segment_id=segment["segment_id"],
        start=segment["start_frame"],
        end=segment["end_frame"],
        vocab_list="\n   ".join(PRIMITIVE_VOCAB),
    )
```

**Step 2.3 — 相邻相同标签合并**

```python
def merge_same_label_segments(segments: list[dict]) -> list[dict]:
    """
    Merge consecutive segments with identical primitive labels.
    """
    if not segments:
        return segments
    
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        if seg["primitive"] == merged[-1]["primitive"]:
            # Extend the last merged segment
            merged[-1]["end_frame"] = seg["end_frame"]
        else:
            merged.append(seg.copy())
    
    return merged
```

---

### 4.4 Stage 3：质检规则

**规则过滤（自动）**

```python
def validate_segment(segment: dict,
                     actions: np.ndarray,
                     min_frames: int = 10,
                     max_frames: int = 500) -> tuple[bool, str]:
    """
    Validate a segment against consistency rules.
    Returns (is_valid, reason_if_invalid).
    """
    start, end = segment["start_frame"], segment["end_frame"]
    duration = end - start
    primitive = segment["primitive"]
    
    # Rule 1: Minimum duration
    if duration < min_frames:
        return False, f"Segment too short: {duration} frames"
    
    # Rule 2: Maximum duration
    if duration > max_frames:
        return False, f"Segment too long: {duration} frames, consider re-splitting"
    
    # Rule 3: Gripper consistency for prehensile actions
    gripper = actions[start:end, -1]  # assuming last dim is gripper
    gripper_range = gripper.max() - gripper.min()
    
    if primitive == "transport" and gripper_range > 0.05:
        return False, "transport segment has gripper state change — likely mis-segmented"
    
    if primitive in ("reach", "retract") and gripper_range > 0.1:
        return False, f"{primitive} segment has unexpected gripper activity"
    
    # Rule 4: Low confidence flag
    if segment.get("confidence", 1.0) < 0.6:
        return False, f"Low VLM confidence: {segment['confidence']:.2f}"
    
    return True, ""
```

**人工 QA 抽样建议**

| 检查项 | 建议抽样比例 | 重点关注 |
|--------|------------|----------|
| pick/place 边界对齐 | 10% | gripper 状态转变是否与标注边界一致 |
| 物体名称准确性 | 5% | VLM 是否幻觉出不存在的物体 |
| 多物体场景 | 20% | 标注是否指向正确的被操作物体 |
| 失败/异常轨迹 | 100% | 确认过滤后无污染数据残留 |

---

### 4.5 完整 Pipeline 调用示例

```python
def annotate_trajectory(
    actions: np.ndarray,           # (T, action_dim)，最后一维为 gripper
    observations: np.ndarray,      # (T, H, W, 3)
    task_instruction: str,
    vlm_client,                    # 兼容 OpenAI/Qwen API 的客户端
    gripper_dim: int = -1,
    rdp_epsilon: float = 0.02,
    min_gap: int = 30,
) -> list[dict]:
    
    # Stage 1: 运动学切割
    active_start, active_end = filter_idle_frames(actions)
    active_actions = actions[active_start:active_end]
    active_obs = observations[active_start:active_end]
    
    gripper_signal = active_actions[:, gripper_dim]
    eef_xyz = active_actions[:, :3]
    
    gripper_bounds = detect_gripper_boundaries(gripper_signal)
    rdp_bounds = rdp_keyframes(eef_xyz, epsilon=rdp_epsilon)
    
    all_bounds = merge_boundaries(gripper_bounds, rdp_bounds, min_gap=min_gap)
    # Always include start and end
    all_bounds = sorted(set([0] + all_bounds + [len(active_actions) - 1]))
    
    # Stage 2: VLM 语义标注
    segments = extract_segment_frames(active_obs, all_bounds)
    
    annotated = []
    for seg in segments:
        prompt = build_prompt(task_instruction, seg)
        response = vlm_client.annotate(prompt, seg["frames"])  # 接入 Qwen-VL / GPT-4o
        
        seg["primitive"] = response.get("primitive", "unknown")
        seg["label"] = response.get("label", "")
        seg["confidence"] = response.get("confidence", 0.0)
        
        # Offset back to original frame indices
        seg["start_frame"] += active_start
        seg["end_frame"] += active_start
        
        annotated.append(seg)
    
    # Merge same-label consecutive segments
    merged = merge_same_label_segments(annotated)
    
    # Stage 3: 规则过滤
    valid_segments = []
    for seg in merged:
        ok, reason = validate_segment(seg, actions)
        if ok:
            valid_segments.append(seg)
        else:
            seg["_filtered_reason"] = reason  # 保留供审计
    
    return valid_segments
```

---

### 4.6 输出数据结构（LeRobot v2.1 兼容）

```json
{
  "episode_id": "episode_0042",
  "task_instruction": "pick up the red cup and place it on the plate",
  "subtasks": [
    {
      "segment_id": 0,
      "start_frame": 0,
      "end_frame": 45,
      "primitive": "reach",
      "label": "reach for the red cup",
      "confidence": 0.92
    },
    {
      "segment_id": 1,
      "start_frame": 45,
      "end_frame": 112,
      "primitive": "pick up",
      "label": "pick up the red cup",
      "confidence": 0.97
    },
    {
      "segment_id": 2,
      "start_frame": 112,
      "end_frame": 198,
      "primitive": "place",
      "label": "place the red cup on the plate",
      "confidence": 0.95
    }
  ]
}
```

---

## 参考文献

| 论文 / 数据集 | 主要贡献 |
|--------------|----------|
| RoboInter (Li et al., ICLR 2026) | 15 primitive skills 定义；半自动标注工具 RoboInter-Tool |
| RH20T-P (Chen et al., 2024) | motion-based + gripper-based 两类原语；38k 标注 clips |
| CycleVLA (2025) | 多阈值 gripper 状态投票；子任务自动切割 prompt 设计 |
| Tri-System VLA (2025) | RDP + gripper 联合边界检测；VLM 语义合并 pipeline |
| AtomVLA / AtomicVLA (2025) | 粗粒度标准词汇表（5类），适合大规模自动标注 |
| FastUMI-100K (2024) | 双层标注（subtask + motion）；第一视角坐标系约定 |
| GigaBrain-0 (2025) | gripper 状态切割 + Qwen-VL 批量语言标注实践 |
| Latent Reasoning VLA (2025) | anchor-first, generate-later 标注范式 |
| π0.5 (Physical Intelligence, 2025) | 开放词汇语义子任务；人类实时干预接口 |
| RT-H (Google DeepMind, 2024) | language motion 层级（L3）；双层语言指令体系 |
| RoboSubtaskNet (2025) | 7词最小动词集实验验证 |
| StreamVLA (2025) | VLM 预标注 + 人工 web 端验证的半自动 pipeline |
