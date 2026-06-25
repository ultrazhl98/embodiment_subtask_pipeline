"""产线配置。

所有 Stage 的可调参数集中在此。支持从 YAML 加载并覆盖默认值。

动词/原语体系的 single source of truth 也在此声明 (PHYSICAL_PRIMITIVES /
ALLOWED_VERBS / PRIMITIVE_TO_VERB / PRIMITIVE_TEMPLATES)，其余模块一律动态引用，
不允许硬编码 (见 doc/pipeline_improvement_plan.md 第五节)。
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
    gripper_jitter_weight: float = 0.6   # quality_score 中抖动项权重 (可按数据集调)
    gripper_invalid_weight: float = 0.7  # quality_score 中无效切换项权重
    length_mad_factor: float = 3.0     # 长度异常 MAD 倍数
    brightness_threshold: float = 30.0  # 灰度均值过暗阈值
    blur_threshold: float = 100.0      # 拉普拉斯方差模糊阈值
    frozen_diff_threshold: float = 1.0  # 连续帧像素差均值低于此判为 frozen
    frozen_run_length: int = 3         # 连续多少帧 frozen 才标记
    image_sample_count: int = 5        # 图像质量均匀采样帧数
    gripper_binarize_threshold: float = 0.5


@dataclass
class Stage05Config:
    """Stage 0.5 全局视频理解 (替代旧 Stage 1-A 首帧锚点)。"""

    enable: bool = True                # 关闭则退化为无全局理解模式
    sample_count: int = 10             # 全局理解均匀采样帧数 (覆盖全程)
    max_objects: int = 4               # 最多识别物体数


@dataclass
class Stage1bConfig:
    """Stage 1-B 事件帧提取 + 语义原语分类。"""

    binarize_threshold: float = 0.5
    median_filter_window: int = 7      # noisy 轨迹中值滤波窗口
    min_segment_frames: int = 10       # 过短 segment 合并阈值
    lift_threshold: float = 0.05       # 垂直位移阈值, 区分 pick_up vs grasp
    transport_disp_threshold: float = 0.08  # 水平位移阈值, 区分 transport vs hold
    use_rdp_for_nonprehensile: bool = True  # 非抓持轨迹启用 RDP 拐点分割
    rdp_epsilon: float = 0.02          # RDP 简化精度 (按动作尺度调)
    nonprehensile_open_ratio_threshold: float = 0.1  # 判为非抓持轨迹的 open 帧占比上限
    speed_minima_window: int = 5       # 速度极小值检测窗口


@dataclass
class Stage1cConfig:
    """Stage 1-C Per-segment 描述生成 (模板 + VLM 填槽, 并入原 Stage 3 职责)。"""

    keyframes_per_segment: int = 4     # 每段送入 VLM 的关键帧数 (首+末+2中间)
    max_desc_words: int = 15


@dataclass
class Stage2Config:
    """Stage 2 规则质量过滤 (从"对齐"退化为"过滤")。"""

    min_segment_frames: int = 10
    max_segment_frames: int = 400


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
class DatasetConfig:
    """数据集 profile：把一个数据集的来源 + 加载约定收拢到一处。

    新增一个 LeRobot 数据集只需在 YAML 的 `dataset:` 块里填这些字段，无需改代码。
    """

    type: str = "synthetic"            # "lerobot" | "synthetic"
    # --- 通用 ---
    n: Optional[int] = None            # 处理的轨迹条数 (None=全部)
    # --- LeRobot ---
    root: Optional[str] = None         # 数据集根目录
    state_key: str = "observation.state"
    eef_xyz_dims: Optional[List[int]] = None  # state 中末端 xyz 的 3 个维度下标
    gripper_key: Optional[str] = None  # gripper 来源列 (如 'action')；None 则取 state
    gripper_dim: int = -1              # gripper 在该列向量中的下标
    gripper_open_is_high: bool = True  # True: 值大=张开；False: 翻转极性
    gripper_min: Optional[float] = None  # 给定 min/max 则线性归一到 [0,1]
    gripper_max: Optional[float] = None
    image_camera: Optional[str] = None  # 相机名 (None=自动探测)
    # --- synthetic ---
    with_images: bool = True


@dataclass
class LLMConfig:
    """VLM / LLM 客户端配置。"""

    backend: str = "mock"              # "mock" | "vllm" | "openai" | "gemini"
    llm_model: str = "gpt-4o-mini"
    vlm_model: str = "qwen2.5-vl-7b-instruct"
    api_key: Optional[str] = None
    base_url: Optional[str] = None     # OpenAI 兼容端点 (显式指定时优先)
    host: Optional[str] = None         # vLLM 服务 IP (backend=vllm 时只需配这个)
    port: int = 8000                   # vLLM 服务端口
    model: Optional[str] = None        # vLLM served model 名; None 则自动发现
    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 2               # JSON 解析/校验失败重试次数
    request_timeout: float = 60.0


# ---------------------------------------------------------------------------
# 动词 / 原语体系 (single source of truth)
# ---------------------------------------------------------------------------

# 物理信号层 (Stage 1-B 内部使用的封闭原语集)
PHYSICAL_PRIMITIVES: List[str] = [
    "reach", "retract", "pick_up", "grasp", "transport", "place",
    "push", "pull", "press", "wipe",
    "open", "close", "rotate", "insert", "pour",
    "hold", "unknown",
]

# 语义标注层 (Stage 1-C 描述生成的动词约束，与标注指南对齐)
ALLOWED_VERBS: List[str] = [
    "reach", "retract", "move to",
    "pick up", "place", "transport", "hand over",
    "push", "pull", "press", "wipe",
    "open", "close", "rotate", "insert", "pour",
]

# primitive -> 自然语言动词 (Stage 2 sanity check 与 Stage 1-C bootstrap 文本)
PRIMITIVE_TO_VERB: Dict[str, str] = {
    "reach": "reach",
    "retract": "retract",
    "pick_up": "pick up",
    "grasp": "pick up",
    "transport": "transport",
    "place": "place",
    "push": "push",
    "pull": "pull",
    "press": "press",
    "wipe": "wipe",
    "open": "open",
    "close": "close",
    "rotate": "rotate",
    "insert": "insert",
    "pour": "pour",
    "hold": "transport",
}

# primitive -> 描述模板 (Stage 1-C 填槽, 替代旧的开放生成)
PRIMITIVE_TEMPLATES: Dict[str, str] = {
    "reach": "reach for the {object}",
    "retract": "retract the arm from {object}",
    "pick_up": "pick up the {object}",
    "grasp": "pick up the {object}",
    "transport": "transport the {object} to {target}",
    "place": "place the {object} {prep} {target}",
    "push": "push the {object} {direction}",
    "pull": "pull the {object} toward {direction}",
    "press": "press the {object}",
    "wipe": "wipe the {target} with the {object}",
    "open": "open the {object}",
    "close": "close the {object}",
    "rotate": "rotate the {object}",
    "insert": "insert the {object} into {target}",
    "pour": "pour {object} into {target}",
    "hold": "transport the {object} to {target}",
}

# 需要 target 槽位的原语 (Stage 1-C 校验: target 必须非空)
PRIMITIVES_REQUIRING_TARGET = {"transport", "place", "insert", "pour", "wipe", "hold"}

# 置信度 -> loss_weight 映射 (Stage 5)
DEFAULT_LOSS_WEIGHTS = {"Gold": 1.0, "Bronze": 0.3, "Flagged": 0.0}


def starts_with_allowed_verb(text: str, verbs) -> bool:
    """支持多词动词 (pick up / move to / hand over) 的前缀匹配。"""
    vset = {v.lower() for v in verbs}
    words = text.strip().lower().split()
    for n in (2, 1):  # 优先匹配两词动词
        if " ".join(words[:n]) in vset:
            return True
    return False


@dataclass
class PipelineConfig:
    """顶层配置。"""

    stage0: Stage0Config = field(default_factory=Stage0Config)
    stage05: Stage05Config = field(default_factory=Stage05Config)
    stage1b: Stage1bConfig = field(default_factory=Stage1bConfig)
    stage1c: Stage1cConfig = field(default_factory=Stage1cConfig)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage4: Stage4Config = field(default_factory=Stage4Config)
    llm: LLMConfig = field(default_factory=LLMConfig)
    dataset: Optional[DatasetConfig] = None  # 数据集 profile (CLI 快捷方式也会填充它)

    loss_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LOSS_WEIGHTS))
    allowed_verbs: List[str] = field(default_factory=lambda: list(ALLOWED_VERBS))

    # 主链路开关：关闭文本描述时跳过 Stage 1-C 的 VLM 填槽，
    # 直接用 primitive 对应的模板/动词作 subtask_text (用于无 VLM 离线跑通主链路)。
    enable_text_decomposition: bool = True

    # ----------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        d = dict(d or {})
        sub_map = {
            "stage0": Stage0Config, "stage05": Stage05Config, "stage1b": Stage1bConfig,
            "stage1c": Stage1cConfig, "stage2": Stage2Config, "stage4": Stage4Config,
            "llm": LLMConfig, "dataset": DatasetConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, sub_cls in sub_map.items():
            if key in d and d[key] is not None:
                valid = {f.name for f in fields(sub_cls)}
                kwargs[key] = sub_cls(**{k: v for k, v in d[key].items() if k in valid})
        for scalar in ("loss_weights", "allowed_verbs", "enable_text_decomposition"):
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
