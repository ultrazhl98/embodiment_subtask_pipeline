"""VLLMClient 离线集成测试。

起一个 stdlib http.server 模拟 vLLM 的 OpenAI 兼容接口 (/v1/models,
/v1/chat/completions)，验证: 自动发现模型、文本/图文调用、JSON 解析链路、
连通性 ping。无需 GPU 或真实服务。
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from subtask_pipeline.llm.vllm_client import VLLMClient

FAKE_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"
_last_request = {}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静音
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._send({"data": [{"id": FAKE_MODEL}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n).decode())
        _last_request.clear()
        _last_request.update(payload)
        # 回显: 把收到的多模态/文本内容数量塞进 JSON 字符串里返回
        content = payload["messages"][-1]["content"]
        n_images = sum(1 for c in content if isinstance(c, dict) and c.get("type") == "image_url") \
            if isinstance(content, list) else 0
        reply = json.dumps({"ok": True, "model": payload["model"], "n_images": n_images})
        self._send({"choices": [{"message": {"content": reply}}]})


def _start_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_vllm_autodiscover_and_text():
    srv, port = _start_server()
    try:
        client = VLLMClient(host="127.0.0.1", port=port)  # model 不填 -> 自动发现
        assert client.model == FAKE_MODEL
        parsed = client.generate_json("You are a task planner.", "decompose this")
        assert parsed["ok"] is True
        assert parsed["model"] == FAKE_MODEL
        assert parsed["n_images"] == 0
    finally:
        srv.shutdown()


def test_vllm_multimodal():
    srv, port = _start_server()
    try:
        client = VLLMClient(host="127.0.0.1", port=port, model=FAKE_MODEL)
        imgs = [np.zeros((8, 8, 3), dtype=np.uint8), np.ones((8, 8, 3), dtype=np.uint8) * 255]
        parsed = client.generate_json("You are a description writer.", "describe", images=imgs)
        assert parsed["n_images"] == 2
        # 校验请求体确实带了 image_url
        content = _last_request["messages"][-1]["content"]
        urls = [c for c in content if c.get("type") == "image_url"]
        assert len(urls) == 2
        assert urls[0]["image_url"]["url"].startswith("data:image/png;base64,")
    finally:
        srv.shutdown()


def test_vllm_ping_reachable_and_unreachable():
    srv, port = _start_server()
    try:
        ok = VLLMClient(host="127.0.0.1", port=port).ping()
        assert ok["reachable"] is True
        assert FAKE_MODEL in ok["available_models"]
    finally:
        srv.shutdown()
    # 不可达端口: ping 优雅返回 False, 不抛异常
    bad = VLLMClient(host="127.0.0.1", port=1, net_retries=1, net_backoff=0.01).ping()
    assert bad["reachable"] is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} vLLM tests passed.")
