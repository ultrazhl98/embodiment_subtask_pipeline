# Prompt 2-协商路径：Movement Primitive 时间戳推断

**用途：** Stage 2 协商路径，当 N_physical ≠ N_text 但差值在容忍范围内时，结合 movement primitive 序列重新推断每个 subtask 的时间戳边界。

**参考来源：** CycleVLA（movement primitive alignment）

**模型：** LLM（无需视觉输入）

**输入：** subtask_texts + movement primitive 序列 + 轨迹总帧数

---

## System Prompt

```
You are a robot trajectory analyst. You will be given:
1. A list of subtask descriptions for a robot manipulation task
2. A time-ordered sequence of movement primitives extracted from the robot's trajectory
3. The total number of frames in the trajectory

Your job is to assign precise timestamp boundaries to each subtask by reasoning about which movement primitives correspond to which subtask.

Rules:
- Every frame must be assigned to exactly one subtask — no gaps, no overlaps
- The subtask sequence must be in the given order (do not reorder)
- The first subtask always starts at frame 0
- The last subtask always ends at the final frame
- Use the movement primitives as evidence for where each subtask begins and ends
- A subtask boundary typically coincides with a gripper state change or a major direction change in motion
- Output ONLY valid JSON, no explanation, no markdown formatting
```

---

## User Prompt

```
Total trajectory frames: {total_frames}

Subtasks to assign (in order):
{subtask_texts_formatted}

Movement primitive sequence (each entry covers approximately {frames_per_primitive} frames):
{primitives_formatted}

Assign a start and end frame to each subtask. The boundaries must:
- Cover all frames from 0 to {total_frames_minus_1} with no gaps
- Follow the natural transitions in the movement primitive sequence

Output a JSON object with the following structure:
{{
  "assignments": [
    {{
      "subtask_index": 0,
      "subtask_text": "<copied from input>",
      "start_frame": <int>,
      "end_frame": <int>
    }},
    ...
  ]
}}
```

---

## 格式化模板

**subtask_texts_formatted：**
```
0: reach toward the red cup on the left side of the table
1: grasp the red cup and lift upward
2: move the red cup to the open drawer and release
```

**primitives_formatted：**
```
[frames 0-30]   move forward and down at medium speed
[frames 31-45]  slow approach, minimal lateral movement
[frames 46-55]  gripper closing, stationary
[frames 56-90]  lift upward at low speed
[frames 91-130] move right and forward at medium speed
[frames 131-145] lower downward slowly
[frames 146-155] gripper opening, stationary
```

---

## 示例输入输出

**输入：**
- 总帧数：156
- Subtasks：
  ```
  0: reach toward the red cup on the left side of the table
  1: grasp the red cup and lift upward
  2: move the red cup to the open drawer and release
  ```
- Movement primitives：
  ```
  [frames 0-45]   move forward and down, slow approach
  [frames 46-55]  gripper closing, stationary
  [frames 56-90]  lift upward
  [frames 91-145] move right and lower into drawer
  [frames 146-155] gripper opening
  ```

**期望输出：**
```json
{
  "assignments": [
    {
      "subtask_index": 0,
      "subtask_text": "reach toward the red cup on the left side of the table",
      "start_frame": 0,
      "end_frame": 45
    },
    {
      "subtask_index": 1,
      "subtask_text": "grasp the red cup and lift upward",
      "start_frame": 46,
      "end_frame": 90
    },
    {
      "subtask_index": 2,
      "subtask_text": "move the red cup to the open drawer and release",
      "start_frame": 91,
      "end_frame": 155
    }
  ]
}
```

---

## 输出校验规则

1. 输出是合法 JSON
2. `assignments` 数组长度等于输入 subtask 数量
3. 第一个 assignment 的 `start_frame` == 0
4. 最后一个 assignment 的 `end_frame` == `total_frames - 1`
5. 相邻 assignment 之间无 gap：`assignments[i+1].start_frame == assignments[i].end_frame + 1`
6. 所有 `start_frame < end_frame`
7. `subtask_text` 与输入一致（不允许修改）

校验不通过（如存在 gap 或越界），该条轨迹降级为 Bronze 路径处理。
