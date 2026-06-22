"""Subtask 标注产线 (L1 -> L2 指令分解数据标注产线)。

综合 LaRA-VLA 与 CycleVLA 的设计，参考 ECoT (embodied-CoT) 的特征提取代码，
对开源机器人操控轨迹做 subtask 级别的任务规划标注。

模块布局:
    data    - 轨迹数据抽象层 (Episode) + LeRobot 加载器 + 合成数据生成器
    llm     - VLM / LLM 客户端抽象、prompt 模板、Mock 与实现
    stages  - Stage 0 ~ Stage 5 各阶段实现
    pipeline- 端到端编排器
    stats   - 数据集级别统计报告
"""

from .config import PipelineConfig

__all__ = ["PipelineConfig"]
__version__ = "0.1.0"
