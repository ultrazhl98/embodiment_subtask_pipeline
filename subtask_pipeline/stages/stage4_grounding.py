"""Stage 4 — Grounding 扩展 (可选模块)。

为每个 segment 关键帧产出被操作物体的 bounding box。检测 + SAM 精化的实现
思路参考 ECoT `scripts/generate_embodied_data/gripper_positions.py`
(OWL-ViT 零样本检测 + SAM mask -> 紧致 box)。

torch / transformers 为重依赖，全部惰性加载；未安装或 enable_grounding=False
时本 Stage 直接跳过，segments 的 grounding 字段保持 None。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import Stage4Config
from ..data.types import Episode, Segment


def _iou(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[0] + b1[2], b2[0] + b2[2]), min(b1[1] + b1[3], b2[1] + b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1, a2 = b1[2] * b1[3], b2[2] * b2[3]
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class Grounder:
    """封装检测器 + SAM。延迟初始化，仅在首次使用时加载模型。"""

    def __init__(self, cfg: Stage4Config):
        self.cfg = cfg
        self._detector = None
        self._sam_model = None
        self._sam_processor = None

    def _ensure_loaded(self):
        if self._detector is not None:
            return
        from transformers import pipeline  # 惰性
        self._detector = pipeline(model="google/owlvit-base-patch16",
                                  task="zero-shot-object-detection")
        if self.cfg.use_sam_refine:
            from transformers import SamModel, SamProcessor
            self._sam_model = SamModel.from_pretrained(self.cfg.sam_model)
            self._sam_processor = SamProcessor.from_pretrained(self.cfg.sam_model)

    def detect(self, img: np.ndarray, query: str) -> Optional[Dict]:
        from PIL import Image
        self._ensure_loaded()
        pil = Image.fromarray(img.astype(np.uint8))
        preds = self._detector(pil, candidate_labels=[query],
                               threshold=self.cfg.box_confidence_threshold)
        if not preds:
            return None
        best = max(preds, key=lambda p: p["score"])
        box = best["box"]
        xywh = [box["xmin"], box["ymin"], box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]]
        if self.cfg.use_sam_refine:
            xywh = self._sam_refine(pil, box) or xywh
        return {"bbox": [int(v) for v in xywh], "confidence": round(float(best["score"]), 3)}

    def _sam_refine(self, pil, box) -> Optional[list]:
        import torch
        inputs = self._sam_processor(
            pil, input_boxes=[[[box["xmin"], box["ymin"], box["xmax"], box["ymax"]]]],
            return_tensors="pt")
        with torch.no_grad():
            outputs = self._sam_model(**inputs)
        mask = self._sam_processor.image_processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        )[0][0][0].numpy()
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())]


def run_stage4(episode: Episode, segments: List[Segment], scene_objects: Sequence[dict],
               cfg: Stage4Config, grounder: Optional[Grounder] = None) -> List[Segment]:
    if not cfg.enable_grounding or not episode.has_images:
        for seg in segments:
            seg.grounding = None
        return segments

    if grounder is None:
        grounder = Grounder(cfg)

    # 为每个 global_summary 物体分配稳定 object_id
    object_ids = [f"{o.get('role', 'object')}_{i}" for i, o in enumerate(scene_objects)]
    prev_boxes: Dict[str, list] = {}

    for seg in segments:
        frame_idx = seg.keyframe if seg.keyframe is not None else (seg.start_frame + seg.end_frame) // 2
        img = episode.image(frame_idx)
        objects = []
        for oid, obj in zip(object_ids, scene_objects):
            try:
                det = grounder.detect(img, obj.get("description", ""))
            except Exception:
                det = None
            if det is None:
                # 时序 carry-over: 沿用上一关键帧的 box
                if oid in prev_boxes:
                    objects.append({"object_id": oid, "bbox": prev_boxes[oid],
                                    "confidence": 0.0, "occluded": True})
                continue
            # IoU 一致性 (仅记录, object_id 已由 anchor 固定)
            det["object_id"] = oid
            det["occluded"] = False
            prev_boxes[oid] = det["bbox"]
            objects.append(det)
        seg.grounding = {"frame_idx": int(frame_idx), "objects": objects} if objects else None

    return segments
