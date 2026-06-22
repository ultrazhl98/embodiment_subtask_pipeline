"""OpenAI 兼容后端 (可对接 OpenAI / vLLM / Qwen2.5-VL 的 OpenAI 兼容服务)。

需要 `openai` 包，并设置 api_key / base_url。VLM 调用把图像编码为 base64 data URL。
"""

from __future__ import annotations

import base64
import io
from typing import List

import numpy as np

from .base import BaseClient


def _encode_image(img: np.ndarray) -> str:
    from PIL import Image
    pil = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


class OpenAICompatClient(BaseClient):
    def __init__(self, llm_model: str, vlm_model: str, api_key=None, base_url=None,
                 temperature: float = 0.0, max_tokens: int = 1024, max_retries: int = 2,
                 timeout: float = 60.0):
        super().__init__(max_retries=max_retries)
        from openai import OpenAI  # 延迟导入
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.llm_model = llm_model
        self.vlm_model = vlm_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content

    def generate_vlm(self, system: str, user: str, images: List[np.ndarray]) -> str:
        content = [{"type": "text", "text": user}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": _encode_image(img)}})
        resp = self.client.chat.completions.create(
            model=self.vlm_model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            temperature=self.temperature, max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content
