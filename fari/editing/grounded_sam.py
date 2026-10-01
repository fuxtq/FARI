from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw


def _ensure_grounded_sam_paths(repo_root: str) -> None:
    root = os.path.abspath(str(repo_root))
    candidates = (
        root,
        os.path.join(root, "GroundingDINO"),
        os.path.join(root, "segment_anything"),
    )
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)


def _load_grounding_modules(repo_root: str):
    _ensure_grounded_sam_paths(repo_root)
    import GroundingDINO.groundingdino.datasets.transforms as T
    from GroundingDINO.groundingdino.models import build_model
    from GroundingDINO.groundingdino.util.slconfig import SLConfig
    from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

    return T, build_model, SLConfig, clean_state_dict, get_phrases_from_posmap


def _load_segment_anything(repo_root: str):
    _ensure_grounded_sam_paths(repo_root)
    from segment_anything import SamPredictor, sam_model_registry

    return SamPredictor, sam_model_registry


def preprocess_grounding_caption(text: str) -> str:
    caption = str(text).strip().lower()
    if not caption:
        raise ValueError("Grounding caption must be non-empty.")
    return caption if caption.endswith(".") else f"{caption}."


def load_groundingdino_model(*, repo_root: str, config_path: str, checkpoint_path: str, device: str):
    T, build_model, SLConfig, clean_state_dict, _ = _load_grounding_modules(repo_root)
    _ = T
    args = SLConfig.fromfile(str(config_path))
    args.device = str(device)
    model = build_model(args)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def load_sam_predictor(*, repo_root: str, sam_checkpoint: str, sam_arch: str, device: str):
    SamPredictor, sam_model_registry = _load_segment_anything(repo_root)
    sam_model = sam_model_registry[str(sam_arch)](checkpoint=str(sam_checkpoint))
    sam_model.to(str(device))
    return SamPredictor(sam_model)


def load_grounding_image_tensor(*, repo_root: str, image_path: str) -> tuple[np.ndarray, torch.Tensor]:
    T, _, _, _, _ = _load_grounding_modules(repo_root)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image_pil, dtype=np.uint8)
    image_tensor, _ = transform(image_pil, None)
    return image_np, image_tensor


def _cxcywh_to_xyxy(box_cxcywh: Sequence[float], *, width: int, height: int) -> List[float]:
    cx, cy, w, h = [float(x) for x in list(box_cxcywh)[:4]]
    x0 = (cx - (w / 2.0)) * float(width)
    y0 = (cy - (h / 2.0)) * float(height)
    x1 = (cx + (w / 2.0)) * float(width)
    y1 = (cy + (h / 2.0)) * float(height)
    return [float(x0), float(y0), float(x1), float(y1)]


def run_grounding_dino(
    *,
    repo_root: str,
    model,
    image_path: str,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str,
) -> List[dict]:
    _, _, _, _, get_phrases_from_posmap = _load_grounding_modules(repo_root)
    image_np, image_tensor = load_grounding_image_tensor(repo_root=repo_root, image_path=image_path)
    normalized_caption = preprocess_grounding_caption(caption)
    model = model.to(str(device))
    image_tensor = image_tensor.to(str(device))
    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[normalized_caption])

    pred_logits = outputs["pred_logits"].cpu().sigmoid()[0]
    pred_boxes = outputs["pred_boxes"].cpu()[0]
    keep = pred_logits.max(dim=1)[0] > float(box_threshold)
    pred_logits = pred_logits[keep]
    pred_boxes = pred_boxes[keep]
    tokenizer = model.tokenizer
    tokenized = tokenizer(normalized_caption)
    height, width = int(image_np.shape[0]), int(image_np.shape[1])

    candidates: List[dict] = []
    for idx, (logit, box) in enumerate(zip(pred_logits, pred_boxes)):
        phrase = get_phrases_from_posmap(logit > float(text_threshold), tokenized, tokenizer).replace(".", "").strip()
        confidence = float(logit.max().item())
        box_xyxy = _cxcywh_to_xyxy(box.tolist(), width=width, height=height)
        candidates.append(
            {
                "rank_index": int(idx),
                "phrase": phrase,
                "confidence": confidence,
                "box_xyxy": [float(x) for x in box_xyxy],
                "box_cxcywh": [float(x) for x in box.tolist()],
            }
        )
    return candidates


def select_best_grounding_candidate(candidates: Sequence[dict]) -> dict:
    if not candidates:
        raise ValueError("Grounding candidate list is empty.")

    def _score(item: dict) -> tuple[float, float, float]:
        box = [float(x) for x in list(item.get("box_xyxy", []))[:4]]
        if len(box) != 4:
            return (float("-inf"), float("-inf"), float("-inf"))
        width = max(0.0, box[2] - box[0])
        height = max(0.0, box[3] - box[1])
        area = width * height
        return (float(item.get("confidence", 0.0)), float(area), -float(item.get("rank_index", 0)))

    return max((dict(candidate) for candidate in candidates), key=_score)


def run_sam_box_segmentation(
    *,
    predictor,
    image_np: np.ndarray,
    box_xyxy: Sequence[float],
    device: str,
) -> np.ndarray:
    predictor.set_image(image_np)
    box_tensor = torch.tensor([list(box_xyxy)], dtype=torch.float32, device=str(device))
    transformed_boxes = predictor.transform.apply_boxes_torch(box_tensor, image_np.shape[:2]).to(str(device))
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )
    mask = masks[0, 0].detach().cpu().numpy().astype(np.uint8)
    if mask.ndim != 2 or int(mask.sum()) <= 0:
        raise RuntimeError("SAM returned an empty mask.")
    return mask


def build_mask_overlay(
    image: Image.Image,
    mask_2d: np.ndarray,
    *,
    alpha: float = 0.42,
    invert: bool = False,
) -> Image.Image:
    mask = np.clip(mask_2d.astype(np.float32, copy=False), 0.0, 1.0)
    if invert:
        mask = 1.0 - mask
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[..., 1] = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    rgb[..., 0] = np.clip(mask * 40.0, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(mask * 70.0, 0, 255).astype(np.uint8)
    mask_img = Image.fromarray(rgb, mode="RGB").resize(image.size, Image.BILINEAR)
    return Image.blend(image.convert("RGB"), mask_img, float(alpha))


def build_box_overlay(image: Image.Image, box_xyxy: Sequence[float], *, label: str | None = None) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    box = [float(x) for x in list(box_xyxy)[:4]]
    if len(box) != 4:
        raise ValueError("box_xyxy must contain exactly 4 values.")
    color = ImageColor.getrgb("#3ddc97")
    draw.rectangle(box, outline=color, width=4)
    if label:
        draw.text((box[0] + 4.0, max(0.0, box[1] - 18.0)), str(label), fill=color)
    return output
