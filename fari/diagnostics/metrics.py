from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


def _distribution(values, *, epsilon: float = 1e-8) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = np.maximum(array, 0.0) + float(epsilon)
    return array / array.sum()


def attention_kl(source_attention, target_attention, *, epsilon: float = 1e-8) -> float:
    """Directional KL used by the paper: D_KL(H_source || H_target)."""
    source = _distribution(source_attention, epsilon=epsilon)
    target = _distribution(target_attention, epsilon=epsilon)
    if source.shape != target.shape:
        raise ValueError(f"Attention shapes differ: {source.shape} vs {target.shape}")
    return float(np.sum(source * np.log(source / target)))


def edit_region_mask(source_image: Image.Image, edited_image: Image.Image, *, threshold: float = 0.12) -> np.ndarray:
    source = np.asarray(source_image.convert("RGB"), dtype=np.float32) / 255.0
    edited = np.asarray(edited_image.convert("RGB"), dtype=np.float32) / 255.0
    if source.shape != edited.shape:
        raise ValueError(f"Image shapes differ: {source.shape} vs {edited.shape}")
    difference = np.abs(edited - source).mean(axis=-1)
    mask = difference >= float(threshold)
    if not np.any(mask):
        adaptive = min(float(difference.mean() + difference.std()), float(threshold) / 2.0)
        mask = difference >= adaptive
    return mask.astype(bool)


def attention_region(attention) -> np.ndarray:
    values = np.asarray(attention, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D attention grid, got {values.shape}")
    minimum, maximum = float(values.min()), float(values.max())
    normalized = (values - minimum) / (maximum - minimum) if maximum > minimum else np.zeros_like(values)
    return normalized >= float(normalized.mean())


@dataclass(frozen=True)
class LocalizationResult:
    iou: float
    precision: float
    recall: float
    attention_region: np.ndarray
    resized_edit_region: np.ndarray


def spatial_localization(attention, final_edit_region) -> LocalizationResult:
    attention_array = np.asarray(attention, dtype=np.float32)
    if attention_array.ndim != 2:
        raise ValueError("attention must be a 2D grid")
    edit = np.asarray(final_edit_region, dtype=bool)
    resized = Image.fromarray(edit.astype(np.uint8) * 255, mode="L").resize(
        (attention_array.shape[1], attention_array.shape[0]),
        Image.Resampling.NEAREST,
    )
    edit_small = np.asarray(resized, dtype=np.uint8) >= 128
    region = attention_region(attention_array)
    intersection = int(np.logical_and(region, edit_small).sum())
    union = int(np.logical_or(region, edit_small).sum())
    predicted = int(region.sum())
    actual = int(edit_small.sum())
    return LocalizationResult(
        iou=float(intersection / union) if union else 0.0,
        precision=float(intersection / predicted) if predicted else 0.0,
        recall=float(intersection / actual) if actual else 0.0,
        attention_region=region,
        resized_edit_region=edit_small,
    )
