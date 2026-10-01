from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from fari.diagnostics.metrics import edit_region_mask


def _rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def evaluate_image_pair(
    source: Image.Image,
    edited: Image.Image,
    *,
    preserve_mask: Image.Image | None = None,
    source_prompt: str = "",
    target_prompt: str = "",
    clip_scorer=None,
    dino_scorer=None,
    lpips_scorer=None,
) -> dict[str, float]:
    source_array, edited_array = _rgb(source), _rgb(edited)
    if source_array.shape != edited_array.shape:
        raise ValueError("Source and edited image sizes differ")
    difference = source_array - edited_array
    mse = float(np.mean(difference**2))
    metrics = {
        "global_l1": float(np.abs(difference).mean()),
        "global_mse": mse,
        "global_psnr": float(-10.0 * math.log10(max(mse, 1e-12))),
        "edit_area_ratio": float(edit_region_mask(source, edited).mean()),
    }
    if preserve_mask is not None:
        keep = np.asarray(preserve_mask.convert("L").resize(source.size, Image.Resampling.NEAREST)) >= 128
        if np.any(keep):
            region_difference = difference[keep]
            region_mse = float(np.mean(region_difference**2))
            metrics.update(
                {
                    "preserved_l1": float(np.abs(region_difference).mean()),
                    "preserved_psnr": float(-10.0 * math.log10(max(region_mse, 1e-12))),
                }
            )
    try:
        from skimage.metrics import structural_similarity

        metrics["global_ssim"] = float(
            structural_similarity(source_array, edited_array, channel_axis=2, data_range=1.0)
        )
    except ImportError:
        pass
    if clip_scorer is not None:
        if not source_prompt or not target_prompt:
            raise ValueError("source_prompt and target_prompt are required for CLIP metrics")
        metrics.update(clip_scorer.score(source, edited, source_prompt, target_prompt))
    if dino_scorer is not None:
        metrics["global_dino_similarity"] = dino_scorer.similarity(source, edited)
    if lpips_scorer is not None:
        metrics["global_lpips"] = lpips_scorer.distance(source, edited)
    return metrics


def evaluate_manifest(
    manifest_jsonl: str,
    *,
    output_jsonl: str,
    clip_scorer=None,
    dino_scorer=None,
    lpips_scorer=None,
) -> list[dict]:
    rows: list[dict] = []
    root = Path(manifest_jsonl).resolve().parent
    for line in Path(manifest_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)

        def resolve(key: str) -> Path:
            path = Path(row[key])
            return path if path.is_absolute() else root / path

        source = Image.open(resolve("source_image_path"))
        edited = Image.open(resolve("edited_image_path"))
        mask = Image.open(resolve("preserve_mask_path")) if row.get("preserve_mask_path") else None
        rows.append(
            {
                **row,
                **evaluate_image_pair(
                    source,
                    edited,
                    preserve_mask=mask,
                    source_prompt=str(row.get("source_prompt", "")),
                    target_prompt=str(row.get("target_prompt", "")),
                    clip_scorer=clip_scorer,
                    dino_scorer=dino_scorer,
                    lpips_scorer=lpips_scorer,
                ),
            }
        )
    destination = Path(output_jsonl)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows
