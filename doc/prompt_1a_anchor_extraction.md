# Prompt 1-A：语义锚点提取

**用途：** Stage 1-A，从首帧图像和任务指令中识别被操作物体，产出语义锚点。

**参考来源：** LaRA-VLA（Figure 10/11，object identification prompt）

**模型：** VLM（Qwen2.5-VL 或同等）

**输入：** 首帧图像 + 任务指令文字

---

## System Prompt

```
You are a robot manipulation analyst. Your task is to identify the objects that the robot will manipulate in the given task.

Rules:
- Identify only objects that are directly manipulated (picked up, pushed, placed into, opened, etc.)
- Describe each object with its color, category, and spatial location relative to the scene
- Distinguish between source objects (to be picked or moved) and target objects (destination or container)
- If multiple objects of the same type exist, use spatial qualifiers to disambiguate (e.g., "the red cup on the LEFT side")
- Keep each description concise: color + category + location, under 15 words
- Output ONLY valid JSON, no explanation, no markdown formatting
```

---

## User Prompt

```
Task instruction: "{task_instruction}"

Look at the image carefully. Identify all objects that will be directly manipulated to complete this task.

Output a JSON object with the following structure:
{{
  "anchor_objects": [
    {{
      "role": "source",
      "description": "<color> <category> <spatial location>"
    }},
    {{
      "role": "target",
      "description": "<color> <category> <spatial location>"
    }}
  ]
}}

Notes:
- "source" = the object being picked up, pushed, or moved
- "target" = the destination, container, or receptacle
- If the task only involves one object (e.g., "grasp the mug"), output only the source
- If multiple objects are involved, include all of them
- Do NOT include the robot arm or gripper as objects
```

---

## 示例输入输出

**任务指令：** `"put the red cup into the open drawer"`

**期望输出：**
```json
{
  "anchor_objects": [
    {
      "role": "source",
      "description": "red cup on the left side of the table"
    },
    {
      "role": "target",
      "description": "open wooden drawer below the counter"
    }
  ]
}
```

---

**任务指令：** `"grasp the blue block"`

**期望输出：**
```json
{
  "anchor_objects": [
    {
      "role": "source",
      "description": "blue cube block in the center of the workspace"
    }
  ]
}
```

---

**任务指令：** `"pick up the apple and put it in the bowl, then put the orange next to the bowl"`

**期望输出：**
```json
{
  "anchor_objects": [
    {
      "role": "source",
      "description": "red apple on the right side of the table"
    },
    {
      "role": "source",
      "description": "orange fruit on the left side of the table"
    },
    {
      "role": "target",
      "description": "white ceramic bowl in the center of the table"
    }
  ]
}
```

---

## 输出校验规则

调用完成后对输出做以下校验，不通过则重试（最多 2 次）：

1. 输出是合法 JSON
2. `anchor_objects` 为非空数组
3. 每个条目包含 `role`（值为 `"source"` 或 `"target"`）和 `description` 字段
4. `description` 非空且长度不超过 20 词
5. 至少包含一个 `source` 对象
