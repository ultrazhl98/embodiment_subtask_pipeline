# Prompt 3-描述生成：Segment Subtask 自然语言描述

**用途：** Stage 3，在已确定 segment 边界后，以视觉 keyframe + 锚点约束 + 参考文字为输入，为每个 segment 生成高质量的 subtask 自然语言描述。

**参考来源：** LaRA-VLA（Figure 11/12，subtask description generation prompt）+ CycleVLA（constrained vocabulary）

**模型：** VLM（需要视觉输入，输入多帧 keyframe）

**输入（每个 segment 独立调用）：** 4 帧 keyframe 图像 + 任务指令 + anchor_objects + 参考 subtask 文字

---

## System Prompt

```
You are a robot manipulation description writer. You will be shown a sequence of images capturing one phase of a robot manipulation task.

Your job is to write a precise, concise natural language description of what the robot is doing in these images.

Rules:
- Use ONLY verbs from this allowed vocabulary: reach, grasp, lift, move, lower, place, release, push, pull, open, close, rotate
- The description MUST refer to the manipulated object using the EXACT phrasing provided in the anchor object list
- Keep the description under 15 words
- Write in simple present tense (e.g., "grasp the red cup and lift upward")
- Do NOT describe the robot's joints, motors, or internal state — only observable actions and objects
- Do NOT add qualifiers like "carefully", "slowly", "successfully" — focus on the action itself
- Output ONLY valid JSON, no explanation, no markdown formatting
```

---

## User Prompt

```
Task instruction: "{task_instruction}"

Object anchor list (you MUST use these exact descriptions when referring to objects):
{anchor_objects_formatted}

Reference subtask text (use as a guide, but refine based on what you actually see in the images):
"{reference_subtask_text}"

The images show frames {start_frame} to {end_frame} of the robot trajectory. Look at all images to understand what action is being performed in this segment.

Describe what the robot is doing in this segment:

{{
  "subtask_text": "<verb> <object from anchor list> [<short directional phrase>]"
}}
```

---

## 格式化模板

**anchor_objects_formatted：**
```
- source object: red cup on the left side of the table
- target object: open wooden drawer below the counter
```

---

## 示例输入输出

### 示例 1：Reach 阶段

**参考文字：** `"reach toward the red cup"`

**图像描述（推断）：** 机械臂从初始位置向左前方伸展，末端执行器逐渐靠近桌面上的红色杯子，gripper 为开启状态。

**期望输出：**
```json
{
  "subtask_text": "reach toward the red cup on the left side of the table"
}
```

---

### 示例 2：Grasp and Lift 阶段

**参考文字：** `"grasp the red cup"`

**图像描述（推断）：** Gripper 闭合包裹红色杯子，杯子已离开桌面，机械臂向上抬升。

**期望输出：**
```json
{
  "subtask_text": "grasp the red cup on the left side of the table and lift upward"
}
```

---

### 示例 3：Move and Place 阶段

**参考文字：** `"move to the drawer and release"`

**图像描述（推断）：** 机械臂持杯移动到右侧抽屉上方，gripper 开始张开，杯子下降至抽屉内。

**期望输出：**
```json
{
  "subtask_text": "lower the red cup into the open wooden drawer below the counter and release"
}
```

---

## 输出校验规则

1. 输出是合法 JSON
2. `subtask_text` 字段非空
3. `subtask_text` 以允许词汇表中的动词开头
4. `subtask_text` 长度不超过 15 词
5. `subtask_text` 中包含至少一个 anchor 物体描述词（模糊匹配，关键词重叠即可）

校验不通过则重试，最多 2 次。两次后仍不通过，该 segment 保留 Stage 1-C 的参考文字作为最终 subtask_text，并在 annotation_meta 中记录 `description_gen_failed: true`。

---

## 物体指称对齐后处理

描述生成完成后，做一次字符串层面的物体指称对齐：

- 检查 `subtask_text` 中是否包含 anchor 物体的关键词（颜色 + 类别，如 "red cup"）
- 若出现了非 anchor 中的物体指称（如 VLM 输出了 "mug" 而 anchor 是 "cup"），用 anchor 描述替换
- 对齐只做关键词替换，不重写整句

这一步在代码层面完成，不依赖 VLM。
