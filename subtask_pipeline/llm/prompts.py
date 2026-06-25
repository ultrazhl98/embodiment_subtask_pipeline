"""Prompt 模板。

每个 build_* 返回 (system, user) 二元组, 供 BaseClient.generate_json 使用。
动词词表统一从 config 注入 (build_* 接 allowed_verbs 参数), 不在此硬编码。
"""

from __future__ import annotations

from typing import List, Sequence


# ---------------------------------------------------------------------------
# 格式化辅助
# ---------------------------------------------------------------------------


def format_objects(objects: Sequence[dict]) -> str:
    """global_summary.objects -> "- role: description" 列表。"""
    return "\n".join(f"- {o.get('role', 'object')}: {o.get('description', '')}" for o in objects)


# ---------------------------------------------------------------------------
# Prompt 0.5: 全局视频理解 (VLM, 多帧覆盖全程)
# ---------------------------------------------------------------------------

GLOBAL_SUMMARY_SYSTEM = """You are a robot manipulation analyst. You will be shown a sequence of frames sampled uniformly across an entire robot manipulation trajectory, plus the task instruction.

Your job is to produce a global understanding of the whole episode:
- the overall task intent
- the objects that get manipulated, each with a role (source / target) and a concise color+category+location description
- the key physical events in temporal order
- a short scene context

Rules:
- Look across ALL frames, not just the first one — the robot may move into the work area or the viewpoint may change.
- Describe each object concisely (color + category + location), under 15 words.
- Output ONLY valid JSON, no explanation, no markdown formatting."""


def build_global_summary_prompt(task_instruction: str, max_objects: int):
    user = f"""Task instruction: "{task_instruction}"

The images are sampled uniformly from start to end of the trajectory. Study the full sequence.

Output a JSON object with exactly this structure:
{{
  "task_intent": "<one sentence describing what the robot accomplishes>",
  "objects": [
    {{ "role": "source", "description": "<color> <category> <location>" }},
    {{ "role": "target", "description": "<color> <category> <location>" }}
  ],
  "key_events": ["<event 1>", "<event 2>", "..."],
  "scene_context": "<short description of the scene>"
}}

Notes:
- "source" = object being picked/pushed/moved; "target" = destination/container/receptacle.
- Include at most {max_objects} objects. Omit the robot arm/gripper.
- If the task involves a single object, output only the source."""
    return GLOBAL_SUMMARY_SYSTEM, user


# ---------------------------------------------------------------------------
# Prompt 1-C: per-segment 物体填槽 (VLM, 多帧关键帧)
# ---------------------------------------------------------------------------

SEGMENT_FILL_SYSTEM = """You are a robot manipulation observer. You will be shown a few keyframes from ONE segment of a robot manipulation trajectory, the physical primitive the robot is performing in this segment, and a list of known objects in the scene.

Your job is NOT to write a free-form description. It is to identify which object(s) this segment acts on, by selecting from or grounding to the known object list.

Rules:
- "object" = the thing being acted upon in THIS segment; it MUST reuse wording from the known object list.
- "target" = the destination/receptacle, required for placing/inserting/transporting/pouring/wiping segments; otherwise it may be an empty string.
- Do NOT invent objects that are not in the known object list.
- Output ONLY valid JSON, no explanation, no markdown formatting."""


def build_segment_fill_prompt(task_instruction: str, primitive_label: str, template: str,
                              objects: Sequence[dict], start_frame: int, end_frame: int):
    objects_block = format_objects(objects) if objects else "(none provided)"
    user = f"""Task instruction: "{task_instruction}"

This segment covers frames {start_frame} to {end_frame}.
Physical primitive: {primitive_label}
Description template to fill: "{template}"

Known objects in the scene (reuse these descriptions):
{objects_block}

Look at the keyframes and fill the template slots.

Output a JSON object:
{{
  "object": "<object from the known list being acted on>",
  "target": "<destination object, or empty string if not applicable>",
  "prep": "<one of: into, onto, on; only for place segments, else empty>",
  "direction": "<short directional phrase, or empty string>"
}}"""
    return SEGMENT_FILL_SYSTEM, user
