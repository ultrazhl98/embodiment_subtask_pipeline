"""命令行入口。

示例:
    # 合成数据离线跑通主链路 (mock LLM, 无需 API key)
    python -m subtask_pipeline.cli run --synthetic --n 5 --out out.jsonl

    # 真实 LeRobot 数据集 + OpenAI 兼容 VLM
    python -m subtask_pipeline.cli run --lerobot /path/to/dataset \
        --config configs/default.yaml --out out.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import List

from .config import DatasetConfig, PipelineConfig
from .data import build_loader
from .data.types import Episode
from .pipeline import PipelineRunner


def _resolve_dataset(cfg: PipelineConfig, args) -> DatasetConfig:
    """决定数据集 spec: CLI 快捷方式 (--synthetic/--lerobot) 优先，否则用 profile。"""
    if args.synthetic:
        return DatasetConfig(type="synthetic", n=args.n, with_images=not args.no_images)
    if args.lerobot:
        eef_dims = [int(x) for x in args.eef_xyz_dims.split(",")] if args.eef_xyz_dims else None
        return DatasetConfig(
            type="lerobot", root=args.lerobot, n=args.n, image_camera=args.camera,
            gripper_key=args.gripper_key, gripper_dim=args.gripper_dim, eef_xyz_dims=eef_dims)
    if cfg.dataset is not None:
        spec = cfg.dataset
        if args.n is not None:  # CLI --n 覆盖 profile
            spec.n = args.n
        return spec
    raise SystemExit("需指定 --synthetic / --lerobot <root>，或在 --config 里配置 dataset:")


def _load_episodes(cfg: PipelineConfig, args) -> List[Episode]:
    spec = _resolve_dataset(cfg, args)
    loader = build_loader(spec)
    n = spec.n
    indices = range(min(n, len(loader))) if n else None
    return list(loader.iter_episodes(indices))


def _apply_llm_overrides(cfg, args):
    """把 CLI 上的 LLM/vLLM 覆盖项写入配置。"""
    if args.backend:
        cfg.llm.backend = args.backend
    if getattr(args, "vllm_host", None):
        cfg.llm.backend = "vllm"
        cfg.llm.host = args.vllm_host
    if getattr(args, "vllm_port", None):
        cfg.llm.port = args.vllm_port
    if getattr(args, "vllm_model", None):
        cfg.llm.model = args.vllm_model
    if getattr(args, "base_url", None):
        cfg.llm.backend = cfg.llm.backend if cfg.llm.backend != "mock" else "vllm"
        cfg.llm.base_url = args.base_url


def cmd_run(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    _apply_llm_overrides(cfg, args)
    if args.no_anchor:
        cfg.enable_anchor_extraction = False
        cfg.enable_text_decomposition = False
    if args.grounding:
        cfg.stage4.enable_grounding = True

    episodes = _load_episodes(cfg, args)
    runner = PipelineRunner(cfg)
    out = runner.run(episodes)

    if args.out:
        with open(args.out, "w") as f:
            for rec in out["records"]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"wrote {len(out['records'])} records -> {args.out}")
    if args.report:
        with open(args.report, "w") as f:
            json.dump(out["report"], f, ensure_ascii=False, indent=2)

    print(json.dumps(out["report"], ensure_ascii=False, indent=2))
    if out["failures"]:
        print(f"\n{len(out['failures'])} failures", file=sys.stderr)


def cmd_ping(args):
    from .llm.vllm_client import VLLMClient
    client = VLLMClient(host=args.vllm_host, port=args.vllm_port, model=args.vllm_model,
                        net_retries=1, net_backoff=0.5, timeout=10.0)
    print(json.dumps(client.ping(), ensure_ascii=False, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="subtask_pipeline", description="Subtask 标注产线")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="运行产线")
    src = r.add_argument_group("data source")
    src.add_argument("--synthetic", action="store_true", help="使用合成数据")
    src.add_argument("--lerobot", metavar="ROOT", help="LeRobot v2/v2.1 数据集根目录")
    src.add_argument("--camera", default=None, help="LeRobot 相机名 (默认自动探测)")
    src.add_argument("--gripper-key", default=None, help="gripper 来源列 (如 action)")
    src.add_argument("--gripper-dim", type=int, default=-1, help="gripper 在该列向量中的下标 (如 action[6] -> 6)")
    src.add_argument("--eef-xyz-dims", default=None, help="state 中末端 xyz 维度, 逗号分隔 (如 austin_buds: 20,21,22)")
    src.add_argument("--n", type=int, default=None, help="处理的轨迹条数 (合成默认5, 真实数据默认全部)")
    src.add_argument("--no-images", action="store_true", help="合成数据不生成图像")

    r.add_argument("--config", help="YAML 配置文件")
    r.add_argument("--backend", choices=["mock", "vllm", "openai", "gemini"], help="覆盖 LLM 后端")
    llm = r.add_argument_group("vLLM 后端 (只需配 --vllm-host 即可启用)")
    llm.add_argument("--vllm-host", default=None, help="vLLM 服务 IP, 设置即自动切到 vllm 后端")
    llm.add_argument("--vllm-port", type=int, default=None, help="vLLM 端口 (默认 8000)")
    llm.add_argument("--vllm-model", default=None, help="served model 名 (默认自动发现)")
    llm.add_argument("--base-url", default=None, help="直接指定 OpenAI 兼容 base_url (覆盖 host/port)")
    r.add_argument("--no-anchor", action="store_true", help="主链路 bootstrap: 跳过锚点/文本分解")
    r.add_argument("--grounding", action="store_true", help="开启 Stage 4 grounding")
    r.add_argument("--out", help="输出 jsonl 路径")
    r.add_argument("--report", help="统计报告 json 路径")
    r.set_defaults(func=cmd_run)

    # ping 子命令: 部署后快速验证连通性
    pg = sub.add_parser("ping", help="检查 vLLM 服务连通性")
    pg.add_argument("--vllm-host", required=True, help="vLLM 服务 IP")
    pg.add_argument("--vllm-port", type=int, default=8000, help="vLLM 端口 (默认 8000)")
    pg.add_argument("--vllm-model", default=None, help="served model 名 (默认自动发现)")
    pg.set_defaults(func=cmd_ping)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
