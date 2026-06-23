"""vLLM 后端客户端 (面向 vLLM 部署的 Qwen-VL 等大模型)。

vLLM 以 OpenAI 兼容接口对外服务，端点为 http://<ip>:<port>/v1。
本客户端只用标准库 urllib (零额外依赖) 直接打 HTTP，使用上只需配置服务 IP:

    from subtask_pipeline.config import LLMConfig
    from subtask_pipeline.llm.vllm_client import VLLMClient
    client = VLLMClient(host="10.0.0.5", port=8000)   # 其余自动探测

特性:
- 只给 host 即可: base_url 自动拼成 http://host:port/v1
- model 不填时自动调用 /v1/models 发现已部署模型名
- 文本与图文(多模态)统一走 chat/completions, 图像编码为 base64 data URL
  (Qwen2.5-VL 经 vLLM OpenAI server 支持 image_url 输入)
- 网络层瞬时错误自动重试 (参考 ECoT safe_call 思路)
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
from typing import List, Optional

import numpy as np

from .base import BaseClient


def _encode_image(img: np.ndarray) -> str:
    """np.ndarray(H,W,3) -> data:image/png;base64,... """
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("图像编码需要 Pillow: pip install Pillow") from e
    pil = Image.fromarray(np.asarray(img).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class VLLMClient(BaseClient):
    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 8000,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 2,
        timeout: float = 60.0,
        net_retries: int = 4,
        net_backoff: float = 3.0,
    ):
        super().__init__(max_retries=max_retries)
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif host:
            self.base_url = f"http://{host}:{port}/v1"
        else:
            raise ValueError("VLLMClient 需要 host 或 base_url 之一")
        self.api_key = api_key or "EMPTY"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.net_retries = net_retries
        self.net_backoff = net_backoff
        self._model = model  # None -> 首次使用时惰性发现 (避免构造阶段因不可达而抛错)

    # -- HTTP --------------------------------------------------------------
    def _request(self, path: str, payload: Optional[dict] = None, method: str = "POST") -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        last_err: Optional[Exception] = None
        for attempt in range(self.net_retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="ignore")[:300]
                # 5xx / 429 可重试，其它直接抛
                if e.code in (429, 500, 502, 503, 504) and attempt < self.net_retries - 1:
                    last_err = RuntimeError(f"HTTP {e.code}: {body}")
                    time.sleep(self.net_backoff * (attempt + 1))
                    continue
                raise RuntimeError(f"vLLM HTTP {e.code} at {url}: {body}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(self.net_backoff * (attempt + 1))
        raise RuntimeError(f"vLLM 请求失败 ({self.net_retries} 次): {last_err}")

    @property
    def model(self) -> str:
        if self._model is None:
            self._model = self._discover_model()
        return self._model

    def _discover_model(self) -> str:
        try:
            data = self._request("/models", method="GET")
            models = [m["id"] for m in data.get("data", []) if "id" in m]
            if models:
                return models[0]
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            f"无法从 {self.base_url}/models 自动发现模型，请显式传入 model 名 "
            "(vLLM 启动时的 --served-model-name)")

    # -- chat completions --------------------------------------------------
    def _chat(self, messages: list) -> str:
        payload = {
            "model": self.model, "messages": messages,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        }
        resp = self._request("/chat/completions", payload)
        return resp["choices"][0]["message"]["content"]

    def generate(self, system: str, user: str) -> str:
        return self._chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

    def generate_vlm(self, system: str, user: str, images: List[np.ndarray]) -> str:
        content = [{"type": "text", "text": user}]
        for img in images or []:
            content.append({"type": "image_url", "image_url": {"url": _encode_image(img)}})
        return self._chat([
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ])

    # -- 连通性自检 --------------------------------------------------------
    def ping(self) -> dict:
        """返回 {base_url, model, reachable}，用于部署后快速验证。"""
        try:
            data = self._request("/models", method="GET")
            models = [m.get("id") for m in data.get("data", [])]
            model = self._model or (models[0] if models else None)
            return {"base_url": self.base_url, "model": model,
                    "available_models": models, "reachable": True}
        except Exception as e:  # noqa: BLE001
            return {"base_url": self.base_url, "model": self._model,
                    "reachable": False, "error": repr(e)}
