"""VLM / LLM 客户端抽象。

参考 ECoT `full_reasonings.py` 的 Gemini 封装 (safe_call 重试 + 结构化解析)，
但抽象成统一接口，便于切换 mock / OpenAI 兼容 / Gemini 后端。

约定:
- LLM 调用: `generate(system, user) -> str`
- VLM 调用: `generate_vlm(system, user, images) -> str`
- `generate_json` 在上面包一层 JSON 解析 + 校验 + 重试。
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class LLMResponseError(RuntimeError):
    """LLM 输出无法解析或多次校验失败。"""


def extract_json(text: str) -> Any:
    """从模型输出中鲁棒地抽取 JSON。

    处理常见情形: ```json fenced block、前后多余文字、单引号等。
    """
    if text is None:
        raise LLMResponseError("空响应")
    # 去掉 markdown code fence
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # 截取第一个 { 到最后一个 } / 第一个 [ 到最后一个 ]
    candidates = []
    for open_c, close_c in (("{", "}"), ("[", "]")):
        s, e = text.find(open_c), text.rfind(close_c)
        if s != -1 and e != -1 and e > s:
            candidates.append(text[s:e + 1])
    candidates.append(text.strip())
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            try:
                return json.loads(cand.replace("'", '"'))
            except json.JSONDecodeError:
                continue
    raise LLMResponseError(f"无法解析 JSON: {text[:200]!r}")


class BaseClient(ABC):
    """所有后端的公共逻辑: JSON 解析 + 校验重试。"""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    # -- 子类实现纯文本 / 多模态生成 ---------------------------------------
    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        ...

    def generate_vlm(self, system: str, user: str, images: List[np.ndarray]) -> str:
        """默认回退到纯文本 (mock / 纯 LLM 后端)。支持视觉的后端应覆盖。"""
        return self.generate(system, user)

    # -- 带 JSON 解析与校验的高层调用 --------------------------------------
    def generate_json(
        self,
        system: str,
        user: str,
        images: Optional[List[np.ndarray]] = None,
        validator: Optional[Callable[[Any], Optional[str]]] = None,
    ) -> Dict[str, Any]:
        """调用模型并解析为 JSON。

        validator(parsed) 返回 None 表示通过，返回错误字符串表示失败 -> 触发重试。
        重试时把上一次的错误反馈拼到 user prompt 末尾 (类似 ECoT 的 "please continue")。
        """
        last_err = ""
        cur_user = user
        for attempt in range(self.max_retries + 1):
            raw = (self.generate_vlm(system, cur_user, images)
                   if images is not None else self.generate(system, cur_user))
            try:
                parsed = extract_json(raw)
            except LLMResponseError as e:
                last_err = str(e)
                cur_user = user + f"\n\nYour previous output was not valid JSON ({last_err}). Output ONLY valid JSON."
                continue
            if validator is not None:
                err = validator(parsed)
                if err:
                    last_err = err
                    cur_user = user + f"\n\nYour previous output failed validation: {err}. Fix it and output ONLY valid JSON."
                    continue
            return parsed
        raise LLMResponseError(f"{self.max_retries + 1} 次尝试后仍失败: {last_err}")
