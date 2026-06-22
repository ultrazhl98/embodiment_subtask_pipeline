"""Prompt 模板。

直接移植自 doc/prompt_*.md 的 System / User Prompt，并提供格式化辅助函数。
每个 build_* 返回 (system, user) 二元组，供 BaseClient.generate_json 使用。
"""

from __future__ import annotations

from typing import List, Sequence

from ..data.types import AnchorObject, Segment

# ---------------------------------------------------------------------------
# 格式化辅助
# ---------------------------------------------------------------------------


def format_anchors_inline(anchors: Sequence[AnchorObject]) -> str:
    """- source: ... / - target: ... 形式 (Stage 1-C)。"""
    return "\n".join(f"- {a.role}: {a.description}" for a in anchors)


def format_anchors_labeled(anchors: Sequence[AnchorObject]) -> str:
    """- source object: ... / - target object: ... 形式 (Stage 3)。"""
    return "\n".join(f"- {a.role} object: {a.description}" for a in anchors)


def format_indexed(texts: Sequence[str]) -> str:
    """0: xxx\n1: yyy ..."""
    return "\n".join(f"{i}: {t}" for i, t in enumerate(texts))


def format_bracketed(texts: Sequence[str]) -> str:
    """[0] xxx\n[1] yyy ..."""
    return "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))


# ---------------------------------------------------------------------------
# Prompt 1-A: 语义锚点提取 (VLM)
# ---------------------------------------------------------------------------

ANCHOR_SYSTEM = """You are a robot manipulation analyst. Your task is to identify the objects that the robot will manipulate in the given task.

Rules:
- Identify only objects that are directly manipulated (picked up, pushed, placed into, opened, etc.)
- Describe each object with its color, category, and spatial location relative to the scene
- Distinguish between source objects (to be picked or moved) and target objects (destination or container)
- If multiple objects of the same type exist, use spatial qualifiers to disambiguate (e.g., "the red cup on the LEFT side")
- Keep each description concise: color + category + location, under 15 words
- Output ONLY valid JSON, no explanation, no markdown formatting"""


def build_anchor_prompt(task_instruction: str):
    user = f"""Task instruction: "{task_instruction}"

Look at the image carefully. Identify all objects that will be directly manipulated to complete this task.

Output a JSON object with the following structure:
{{
  "anchor_objects": [
    {{ "role": "source", "description": "<color> <category> <spatial location>" }},
    {{ "role": "target", "description": "<color> <category> <spatial location>" }}
  ]
}}

Notes:
- "source" = the object being picked up, pushed, or moved
- "target" = the destination, container, or receptacle
- If the task only involves one object (e.g., "grasp the mug"), output only the source
- If multiple objects are involved, include all of them
- Do NOT include the robot arm or gripper as objects"""
    return ANCHOR_SYSTEM, user


# ---------------------------------------------------------------------------
# Prompt 1-C: 文本语义分解 (LLM)
# ---------------------------------------------------------------------------

TEXT_DECOMP_SYSTEM = """You are a robot task planner. Your job is to decompose a manipulation task instruction into a minimal sequence of atomic subtasks that a robot arm can execute step by step.

Constraints:
- Use ONLY verbs from the allowed vocabulary: reach, grasp, lift, move, lower, place, release, push, pull, open, close, rotate
- Each subtask must be a single atomic action — do NOT combine two actions into one subtask
- Do NOT over-decompose: "grasp" should NOT be split into "approach" + "contact" + "close gripper"
- Each subtask description must mention the object being acted upon, using the EXACT object descriptions provided in the anchor list
- Subtask count should be between 2 and 6
- Output ONLY valid JSON, no explanation, no markdown formatting"""


def build_text_decomp_prompt(task_instruction: str, anchors: Sequence[AnchorObject]):
    user = f"""Task instruction: "{task_instruction}"

Objects involved in this task (use these EXACT descriptions when referring to objects):
{format_anchors_inline(anchors)}

Decompose the task into an ordered sequence of atomic subtasks.

Output a JSON object with the following structure:
{{
  "subtask_count": <integer>,
  "subtask_texts": [ "<verb> <object description> [<direction or destination>]", ... ]
}}

Rules for subtask text format:
- Start each subtask with a verb from the allowed vocabulary
- Include the object name from the anchor list
- Optionally add a short directional phrase (e.g., "upward", "to the right", "into the drawer")
- Keep each subtask under 12 words
- The sequence must cover the complete task from start to finish"""
    return TEXT_DECOMP_SYSTEM, user


# ---------------------------------------------------------------------------
# Prompt 2-negotiate: movement primitive 时间戳推断 (LLM)
# ---------------------------------------------------------------------------

NEGOTIATE_SYSTEM = """You are a robot trajectory analyst. You will be given:
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
- Output ONLY valid JSON, no explanation, no markdown formatting"""


def build_negotiate_prompt(total_frames: int, subtask_texts: Sequence[str],
                           primitives_formatted: str, frames_per_primitive: int):
    user = f"""Total trajectory frames: {total_frames}

Subtasks to assign (in order):
{format_indexed(subtask_texts)}

Movement primitive sequence (each entry covers approximately {frames_per_primitive} frames):
{primitives_formatted}

Assign a start and end frame to each subtask. The boundaries must:
- Cover all frames from 0 to {total_frames - 1} with no gaps
- Follow the natural transitions in the movement primitive sequence

Output a JSON object:
{{
  "assignments": [
    {{ "subtask_index": 0, "subtask_text": "<copied from input>", "start_frame": <int>, "end_frame": <int> }}
  ]
}}"""
    return NEGOTIATE_SYSTEM, user


# ---------------------------------------------------------------------------
# Prompt 2-fallback: 视觉关键帧检索式标签匹配 (VLM, 单帧)
# ---------------------------------------------------------------------------

FALLBACK_SYSTEM = """You are a robot manipulation observer. You will be shown a single image captured during a robot manipulation task, and a list of candidate subtask descriptions.

Your job is to select the ONE subtask description that best matches what the robot is currently doing in the image.

Rules:
- Select EXACTLY one subtask from the provided candidate list
- Do NOT invent new descriptions — only choose from the given candidates
- Base your decision on the robot's gripper state, arm position, and the objects' current state visible in the image
- If the image shows a transition between two subtasks, choose the subtask that is MORE complete at this frame
- Output ONLY valid JSON, no explanation, no markdown formatting"""


def build_fallback_prompt(task_instruction: str, candidate_subtasks: Sequence[str]):
    user = f"""Task instruction: "{task_instruction}"

Candidate subtask descriptions (choose exactly one):
{format_indexed(candidate_subtasks)}

Look at the image. Based on the robot's current state and the objects in the scene, which subtask is the robot currently performing?

Output a JSON object:
{{
  "selected_index": <integer, 0-based index from the candidate list>,
  "selected_subtask": "<copied exactly from the candidate list>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explaining the visual evidence>"
}}"""
    return FALLBACK_SYSTEM, user


# ---------------------------------------------------------------------------
# Prompt 3-description: segment 描述生成 (VLM, 多帧)
# ---------------------------------------------------------------------------

DESC_SYSTEM = """You are a robot manipulation description writer. You will be shown a sequence of images capturing one phase of a robot manipulation task.

Your job is to write a precise, concise natural language description of what the robot is doing in these images.

Rules:
- Use ONLY verbs from this allowed vocabulary: reach, grasp, lift, move, lower, place, release, push, pull, open, close, rotate
- The description MUST refer to the manipulated object using the EXACT phrasing provided in the anchor object list
- Keep the description under 15 words
- Write in simple present tense (e.g., "grasp the red cup and lift upward")
- Do NOT describe the robot's joints, motors, or internal state — only observable actions and objects
- Do NOT add qualifiers like "carefully", "slowly", "successfully" — focus on the action itself
- Output ONLY valid JSON, no explanation, no markdown formatting"""


def build_description_prompt(task_instruction: str, anchors: Sequence[AnchorObject],
                             reference_subtask_text: str, start_frame: int, end_frame: int,
                             extra_hint: str = ""):
    hint = f"\n\nAdditional guidance: {extra_hint}" if extra_hint else ""
    user = f"""Task instruction: "{task_instruction}"

Object anchor list (you MUST use these exact descriptions when referring to objects):
{format_anchors_labeled(anchors)}

Reference subtask text (use as a guide, but refine based on what you actually see in the images):
"{reference_subtask_text}"

The images show frames {start_frame} to {end_frame} of the robot trajectory. Look at all images to understand what action is being performed in this segment.{hint}

Describe what the robot is doing in this segment:
{{ "subtask_text": "<verb> <object from anchor list> [<short directional phrase>]" }}"""
    return DESC_SYSTEM, user


# ---------------------------------------------------------------------------
# Prompt 3-self check: 序列完整性 + 去重 (LLM)
# ---------------------------------------------------------------------------

SELF_CHECK_SYSTEM = """You are a quality control checker for robot task annotations. You will be given a task instruction and a proposed sequence of subtask descriptions that are supposed to cover the complete task execution.

Your job is to check two things:
1. COMPLETENESS: Does the subtask sequence fully cover all steps required to complete the task instruction? Are there any missing steps?
2. REDUNDANCY: Are any adjacent subtasks semantically identical or heavily overlapping in meaning?

Be strict but fair. A subtask sequence that covers the task at different levels of granularity is acceptable. Only flag issues that would clearly mislead a robot during training.

Output ONLY valid JSON, no explanation outside of JSON."""


def build_self_check_prompt(task_instruction: str, subtask_texts: Sequence[str]):
    user = f"""Task instruction: "{task_instruction}"

Proposed subtask sequence:
{format_bracketed(subtask_texts)}

Check this sequence and output:
{{
  "completeness_check": {{ "passed": <bool>, "missing_steps": ["..."], "verdict": "<one sentence>" }},
  "redundancy_check": {{ "passed": <bool>, "duplicate_pairs": [{{ "index_a": <int>, "index_b": <int>, "reason": "..." }}], "verdict": "<one sentence>" }},
  "overall_passed": <true if BOTH checks passed, false otherwise>
}}"""
    return SELF_CHECK_SYSTEM, user
