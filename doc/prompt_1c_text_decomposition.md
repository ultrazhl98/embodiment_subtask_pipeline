# Prompt 1-C：文本语义分解

**用途：** Stage 1-C，用 LLM 对任务指令做原子 subtask 文本分解，注入语义锚点控制物体指称。

**参考来源：** CycleVLA（constrained action vocabulary decomposition）

**模型：** LLM（无需视觉输入）

**输入：** 任务指令文字 + Stage 1-A 产出的 anchor_objects

---

## System Prompt

```
You are a robot task planner. Your job is to decompose a manipulation task instruction into a minimal sequence of atomic subtasks that a robot arm can execute step by step.

Constraints:
- Use ONLY verbs from the allowed vocabulary: reach, grasp, lift, move, lower, place, release, push, pull, open, close, rotate
- Each subtask must be a single atomic action — do NOT combine two actions into one subtask
- Do NOT over-decompose: "grasp" should NOT be split into "approach" + "contact" + "close gripper"
- Each subtask description must mention the object being acted upon, using the EXACT object descriptions provided in the anchor list
- Subtask count should be between 2 and 6
- Output ONLY valid JSON, no explanation, no markdown formatting
```

---

## User Prompt

```
Task instruction: "{task_instruction}"

Objects involved in this task (use these EXACT descriptions when referring to objects):
{anchor_objects_formatted}

Decompose the task into an ordered sequence of atomic subtasks.

Output a JSON object with the following structure:
{{
  "subtask_count": <integer>,
  "subtask_texts": [
    "<verb> <object description> [<direction or destination>]",
    ...
  ]
}}

Rules for subtask text format:
- Start each subtask with a verb from the allowed vocabulary
- Include the object name from the anchor list
- Optionally add a short directional phrase (e.g., "upward", "to the right", "into the drawer")
- Keep each subtask under 12 words
- The sequence must cover the complete task from start to finish
```

---

## Anchor Objects 格式化模板

在填入 `{anchor_objects_formatted}` 时，按以下格式拼接：

```
- source: {description}
- target: {description}
```

例如：
```
- source: red cup on the left side of the table
- target: open wooden drawer below the counter
```

---

## 示例输入输出

**任务指令：** `"put the red cup into the open drawer"`

**锚点物体：**
```
- source: red cup on the left side of the table
- target: open wooden drawer below the counter
```

**期望输出：**
```json
{
  "subtask_count": 3,
  "subtask_texts": [
    "reach toward the red cup on the left side of the table",
    "grasp the red cup and lift upward",
    "move to the open wooden drawer and release"
  ]
}
```

---

**任务指令：** `"open the cabinet door and put the bowl inside"`

**锚点物体：**
```
- source: white ceramic bowl on the counter
- target: wooden cabinet with metal handle on the right
```

**期望输出：**
```json
{
  "subtask_count": 4,
  "subtask_texts": [
    "reach toward the wooden cabinet with metal handle on the right",
    "pull open the cabinet door",
    "grasp the white ceramic bowl on the counter",
    "place the white ceramic bowl inside the cabinet"
  ]
}
```

---

## 输出校验规则

1. 输出是合法 JSON
2. `subtask_count` 与 `subtask_texts` 数组长度一致
3. `subtask_count` 在 2-6 之间
4. 每条 subtask 文字以允许词汇表中的动词开头
5. 每条 subtask 文字中包含至少一个来自 anchor 的物体描述词（模糊匹配）
6. 所有 subtask 按逻辑顺序排列（不能先 release 再 grasp）

校验不通过则重试，最多 2 次。
