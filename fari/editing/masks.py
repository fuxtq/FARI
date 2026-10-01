from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from PIL import Image


def infer_token_grid(image_token_count: int, *, height: int, width: int) -> tuple[int, int]:
    if image_token_count <= 0:
        raise ValueError("image_token_count must be positive")
    aspect = float(width) / float(height)
    cols = max(1, int(round(math.sqrt(image_token_count * aspect))))
    while cols > 1 and image_token_count % cols:
        cols -= 1
    rows = image_token_count // cols
    if rows * cols != image_token_count:
        raise ValueError(f"Cannot infer a token grid for {image_token_count} tokens")
    return rows, cols


def preserved_token_indices(
    preserve_mask: Image.Image | str,
    *,
    image_token_count: int,
    height: int,
    width: int,
    threshold: int = 128,
) -> list[int]:
    mask = Image.open(preserve_mask) if isinstance(preserve_mask, str) else preserve_mask
    rows, cols = infer_token_grid(image_token_count, height=height, width=width)
    resized = mask.convert("L").resize((cols, rows), resample=Image.Resampling.NEAREST)
    values = np.asarray(resized, dtype=np.uint8).reshape(-1)
    return [int(i) for i, value in enumerate(values) if int(value) >= int(threshold)]


def invert_mask(mask: Image.Image) -> Image.Image:
    values = 255 - np.asarray(mask.convert("L"), dtype=np.uint8)
    return Image.fromarray(values, mode="L")


def mask_from_indices(indices: Sequence[int], *, image_token_count: int, height: int, width: int) -> Image.Image:
    rows, cols = infer_token_grid(image_token_count, height=height, width=width)
    values = np.zeros(image_token_count, dtype=np.uint8)
    values[[int(i) for i in indices if 0 <= int(i) < image_token_count]] = 255
    return Image.fromarray(values.reshape(rows, cols), mode="L").resize((width, height), Image.Resampling.NEAREST)
