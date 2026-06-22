"""产线配置。

所有 Stage 的可调参数集中在此，按 doc/pipeline_overview.md 的"关键参数"小节给默认值。
支持从 YAML 加载并覆盖默认值。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional


@dataclass
class Stage0Config:
    """Stage 0 轨迹预过滤。"""

    jitter_window: int = 10            # 抖动检测窗口 (帧)
    jitter_max_switches: int = 2       # 窗口内超过该切换次数判为抖动
    min_hold_frames: int = 5           # 无效切换最小持续帧
    gripper_reliable_threshold: float = 0.7
    length_mad_factor: float = 3.0     # 长度异常 MAD 倍数
    brightness_threshold: float = 30.0  # 灰度均值过暗阈值
    blur_threshold: float = 100.0      # 拉普拉斯方差模糊阈值
    frozen_diff_threshold: float = 1.0  # 连续帧像素差均值低于此判为 frozen
    frozen_run_length: int = 3         # 连续多少帧 frozen 才标记
    image_sample_count: int = 5        # 图像质量均匀采样帧数
    gripper_binarize_threshold: float = 0.5


@dataclass
class Stage1bConfig:
    """Stage 1-B 物理信号分割。"""

    binarize_threshold: float = 0.5
    median_filter_window: int = 7      # noisy 轨迹中值滤波窗口
    min_segment_frames: int = 10       # 过短 segment 合并阈值
    primitive_move_threshold: float = 0.03  # 主运动方向阈值 (沿用 ECoT classify_movement)
    fast_speed_quantile: float = 0.6   # 区分 reach(快)/place(慢) 的速度分位


@dataclass
class Stage2Config:
    """Stage 2 一致性校验与对齐。"""

    delta_tolerance: int = 1           # |N_physical - N_text| 容忍阈值 (LIBERO=1, DROID=2)
    fallback_sample_factor: int = 3    # 降级路径采样 N_text * factor 帧
    fallback_low_conf_threshold: float = 0.4
    fallback_mean_conf_threshold: float = 0.5
    negotiate_max_primitive_tokens: int = 20


@dataclass
class Stage3Config:
    """Stage 3 描述生成 + 自检。"""

    keyframes_per_segment: int = 4     # 每段送入 VLM 的关键帧数 (首+末+2中间)
    max_desc_words: int = 15
    max_desc_retries: int = 2
    dedup_word_overlap: float = 0.6    # 相邻 subtask 词汇重叠率阈值
    dedup_use_embedding: bool = False
    dedup_embedding_threshold: float = 0.85
    self_check_max_retries: int = 2


@dataclass
class Stage4Config:
    """Stage 4 grounding 扩展 (可选)。"""

    enable_grounding: bool = False
    box_confidence_threshold: float = 0.3
    use_sam_refine: bool = True
    track_iou_threshold: float = 0.5
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    sam_model: str = "facebook/sam-vit-base"


@dataclass
class LLMConfig:
    """VLM / LLM 客户端配置。"""

    backend: str = "mock"              # "mock" | "openai" | "gemini"
    llm_model: str = "gpt-4o-mini"
    vlm_model: str = "qwen2.5-vl-7b-instruct"
    api_key: Optional[str] = None
    base_url: Optional[str] = None     # OpenAI 兼容端点
    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 2               # JSON 解析/校验失败重试次数
    request_timeout: float = 60.0


# 置信度 -> loss_weight 映射 (Stage 5)
DEFAULT_LOSS_WEIGHTS = {"Gold": 1.0, "Silver": 0.5, "Bronze": 0.2}

# Stage 1-C / Stage 3 允许的动词词汇表
ALLOWED_VERBS = [
    "reach", "grasp", "lift", "move", "lower", "place",
    "release", "push", "pull", "open", "close", "rotate",
]


@dataclass
class PipelineConfig:
    """顶层配置。"""

    stage0: Stage0Config = field(default_factory=Stage0Config)
    stage1b: Stage1bConfig = field(default_factory=Stage1bConfig)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)
    stage4: Stage4Config = field(default_factory=Stage4Config)
    llm: LLMConfig = field(default_factory=LLMConfig)

    loss_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_WEIGHTS))
    allowed_verbs: List[str] = field(default_factory=lambda: list(ALLOWED_VERBS))

    # 主链路开关：关闭锚点注入时跳过 Stage 1-A / 1-C，走纯物理分割 (用于先跑通主链路)
    enable_anchor_extraction: bool = True
    enable_text_decomposition: bool = True

    # ----------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        d = dict(d or {})
        sub_map = {
            "stage0": Stage0Config, "stage1b": Stage1bConfig, "stage2": Stage2Config,
            "stage3": Stage3Config, "stage4": Stage4Config, "llm": LLMConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, sub_cls in sub_map.items():
            if key in d and d[key] is not None:
                valid = {f.name for f in fields(sub_cls)}
                kwargs[key] = sub_cls(**{k: v for k, v in d[key].items() if k in valid})
        for scalar in ("loss_weights", "allowed_verbs",
                       "enable_anchor_extraction", "enable_text_decomposition"):
            if scalar in d and d[scalar] is not None:
                kwargs[scalar] = d[scalar]
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        import yaml
        with open(path, "r") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
