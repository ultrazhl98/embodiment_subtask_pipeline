# Prompt 3-自检：Subtask 序列完整性与去重校验

**用途：** Stage 3，对整条轨迹生成完毕的 subtask 序列做完整性检查（是否覆盖了全局任务）和去重检查（相邻 subtask 是否语义重复）。

**参考来源：** StreamVLA（semi-automated annotation pipeline 的质量控制逻辑）

**模型：** LLM（无需视觉输入，纯文本自检）

**输入：** 全局任务指令 + 生成完毕的 subtask 序列

---

## System Prompt

```
You are a quality control checker for robot task annotations. You will be given a task instruction and a proposed sequence of subtask descriptions that are supposed to cover the complete task execution.

Your job is to check two things:
1. COMPLETENESS: Does the subtask sequence fully cover all steps required to complete the task instruction? Are there any missing steps?
2. REDUNDANCY: Are any adjacent subtasks semantically identical or heavily overlapping in meaning?

Be strict but fair. A subtask sequence that covers the task at different levels of granularity is acceptable. Only flag issues that would clearly mislead a robot during training.

Output ONLY valid JSON, no explanation outside of JSON.
```

---

## User Prompt

```
Task instruction: "{task_instruction}"

Proposed subtask sequence:
{subtask_sequence_formatted}

Check this sequence and output:
{{
  "completeness_check": {{
    "passed": <true or false>,
    "missing_steps": ["<description of any missing step>"],
    "verdict": "<one sentence summary>"
  }},
  "redundancy_check": {{
    "passed": <true or false>,
    "duplicate_pairs": [
      {{
        "index_a": <int>,
        "index_b": <int>,
        "reason": "<why these two are considered redundant>"
      }}
    ],
    "verdict": "<one sentence summary>"
  }},
  "overall_passed": <true if BOTH checks passed, false otherwise>
}}
```

---

## 格式化模板

**subtask_sequence_formatted：**
```
[0] reach toward the red cup on the left side of the table
[1] grasp the red cup on the left side of the table and lift upward
[2] move the red cup to the open wooden drawer below the counter
[3] lower the red cup into the open wooden drawer below the counter and release
```

---

## 示例输入输出

### 示例 1：通过

**任务指令：** `"put the red cup into the open drawer"`

**Subtask 序列：**
```
[0] reach toward the red cup on the left side of the table
[1] grasp the red cup on the left side of the table and lift upward
[2] lower the red cup into the open wooden drawer and release
```

**期望输出：**
```json
{
  "completeness_check": {
    "passed": true,
    "missing_steps": [],
    "verdict": "The sequence covers reach, grasp, and place phases, fully completing the task."
  },
  "redundancy_check": {
    "passed": true,
    "duplicate_pairs": [],
    "verdict": "No adjacent subtasks share overlapping semantic content."
  },
  "overall_passed": true
}
```

---

### 示例 2：完整性失败（缺少关键步骤）

**任务指令：** `"open the cabinet door and put the bowl inside"`

**Subtask 序列：**
```
[0] reach toward the white ceramic bowl on the counter
[1] grasp the white ceramic bowl and lift upward
[2] place the white ceramic bowl inside the cabinet
```

**期望输出：**
```json
{
  "completeness_check": {
    "passed": false,
    "missing_steps": [
      "open the cabinet door before placing the bowl inside"
    ],
    "verdict": "The sequence skips the step of opening the cabinet door, which is required before the bowl can be placed inside."
  },
  "redundancy_check": {
    "passed": true,
    "duplicate_pairs": [],
    "verdict": "No redundant subtasks detected."
  },
  "overall_passed": false
}
```

---

### 示例 3：去重失败（相邻 subtask 语义重复）

**任务指令：** `"pick up the block and place it on the plate"`

**Subtask 序列：**
```
[0] reach toward the blue block in the center of the workspace
[1] move toward the blue block in the center of the workspace
[2] grasp the blue block and lift upward
[3] place the blue block on the white plate
```

**期望输出：**
```json
{
  "completeness_check": {
    "passed": true,
    "missing_steps": [],
    "verdict": "The sequence covers all necessary steps to complete the task."
  },
  "redundancy_check": {
    "passed": false,
    "duplicate_pairs": [
      {
        "index_a": 0,
        "index_b": 1,
        "reason": "Both subtask [0] and subtask [1] describe moving toward the blue block; they represent the same action at the same phase."
      }
    ],
    "verdict": "Subtasks [0] and [1] are semantically redundant and should be merged."
  },
  "overall_passed": false
}
```

---

## 自检后的处理逻辑

| 结果 | 处理方式 |
|------|---------|
| `overall_passed: true` | 直接进入 Stage 4/5，confidence 不变 |
| 完整性失败 | 对失败的 segment 重新调用 [prompt_3_description_gen.md](prompt_3_description_gen.md)，将 missing_steps 加入参考输入，最多重试 2 次 |
| 去重失败 | 对 duplicate_pairs 中的两个 segment 合并为一个，重新调用描述生成，最多重试 1 次 |
| 两次重试后仍失败 | 将该轨迹 confidence 降级为 Bronze，记录 `self_check_retries: 2` |
