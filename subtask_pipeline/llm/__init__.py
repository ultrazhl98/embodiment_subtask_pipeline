"""LLM / VLM 客户端层。"""

from .base import BaseClient, LLMResponseError, extract_json
from .mock import MockClient


def build_client(llm_cfg) -> BaseClient:
    """根据 LLMConfig.backend 构建客户端。"""
    backend = (llm_cfg.backend or "mock").lower()
    if backend == "mock":
        return MockClient(max_retries=llm_cfg.max_retries)
    if backend == "vllm":
        from .vllm_client import VLLMClient
        return VLLMClient(
            host=llm_cfg.host, port=llm_cfg.port, base_url=llm_cfg.base_url,
            model=llm_cfg.model,  # None -> 调 /v1/models 自动发现
            api_key=llm_cfg.api_key or "EMPTY",
            temperature=llm_cfg.temperature, max_tokens=llm_cfg.max_tokens,
            max_retries=llm_cfg.max_retries, timeout=llm_cfg.request_timeout,
        )
    if backend in ("openai", "qwen"):
        from .openai_compat import OpenAICompatClient
        return OpenAICompatClient(
            llm_model=llm_cfg.llm_model, vlm_model=llm_cfg.vlm_model,
            api_key=llm_cfg.api_key, base_url=llm_cfg.base_url,
            temperature=llm_cfg.temperature, max_tokens=llm_cfg.max_tokens,
            max_retries=llm_cfg.max_retries, timeout=llm_cfg.request_timeout,
        )
    raise ValueError(f"未知 LLM backend: {backend}")


__all__ = ["BaseClient", "LLMResponseError", "extract_json", "MockClient",
           "build_client", "VLLMClient"]


def __getattr__(name):
    # 惰性导出 VLLMClient, 避免无谓导入
    if name == "VLLMClient":
        from .vllm_client import VLLMClient
        return VLLMClient
    raise AttributeError(name)
