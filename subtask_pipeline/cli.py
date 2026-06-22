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

from .config import PipelineConfig
from .data.synthetic import make_synthetic_dataset
from .data.types import Episode
from .pipeline import PipelineRunner


def _load_episodes(args) -> List[Episode]:
    if args.synthetic:
        return make_synthetic_dataset(n_episodes=args.n, with_images=not args.no_images)
    if args.lerobot:
        from .data.lerobot_loader import LeRobotLoader
        eef_dims = [int(x) for x in args.eef_xyz_dims.split(",")] if args.eef_xyz_dims else None
        loader = LeRobotLoader(
            args.lerobot, image_camera=args.camera,
            gripper_key=args.gripper_key, gripper_dim=args.gripper_dim,
            eef_xyz_dims=eef_dims)
        indices = range(min(args.n, len(loader))) if args.n else None
        return list(loader.iter_episodes(indices))
    raise SystemExit("需指定 --synthetic 或 --lerobot <root>")


def cmd_run(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    if args.backend:
        cfg.llm.backend = args.backend
    if args.no_anchor:
        cfg.enable_anchor_extraction = False
        cfg.enable_text_decomposition = False
    if args.grounding:
        cfg.stage4.enable_grounding = True

    episodes = _load_episodes(args)
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
    src.add_argument("--n", type=int, default=5, help="处理的轨迹条数")
    src.add_argument("--no-images", action="store_true", help="合成数据不生成图像")

    r.add_argument("--config", help="YAML 配置文件")
    r.add_argument("--backend", choices=["mock", "openai", "gemini"], help="覆盖 LLM 后端")
    r.add_argument("--no-anchor", action="store_true", help="主链路 bootstrap: 跳过锚点/文本分解")
    r.add_argument("--grounding", action="store_true", help="开启 Stage 4 grounding")
    r.add_argument("--out", help="输出 jsonl 路径")
    r.add_argument("--report", help="统计报告 json 路径")
    r.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
