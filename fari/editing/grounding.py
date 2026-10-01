from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass

import numpy as np
from PIL import Image

from fari.editing.grounded_sam import (
    load_grounding_image_tensor,
    load_groundingdino_model,
    load_sam_predictor,
    run_grounding_dino,
    run_sam_box_segmentation,
    select_best_grounding_candidate,
)
from fari.editing.masks import invert_mask


GROUNDING_SYSTEM_PROMPT = """You are helping construct editable region masks for a scientific image-editing benchmark.
Given a source prompt, target prompt, and edit family, identify the concrete source object or region that should be edited.
Return strict JSON with one key named grounding_query_source. Use a short noun phrase visible in the source image.
For object addition, return the source object or surface around which the new object should be placed."""


@dataclass(frozen=True)
class GroundingConfig:
    repo_root: str
    grounding_config: str
    grounding_checkpoint: str
    sam_checkpoint: str
    sam_arch: str = "vit_h"
    box_threshold: float = 0.3
    text_threshold: float = 0.25
    device: str = "cuda"


def infer_grounding_query(
    *,
    source_prompt: str,
    target_prompt: str,
    edit_family: str,
    api_key: str | None = None,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    timeout: int = 60,
) -> str:
    key = str(api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
    if not key:
        raise ValueError("Set DEEPSEEK_API_KEY or provide a grounding query explicitly.")
    user_prompt = json.dumps(
        {"source_prompt": source_prompt, "target_prompt": target_prompt, "edit_family": edit_family},
        ensure_ascii=False,
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": 128,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=int(timeout)) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    query = str(parsed.get("grounding_query_source", "")).strip()
    if not query:
        raise ValueError("The LLM returned an empty grounding query.")
    return query


def build_preserve_mask(
    source_image: Image.Image,
    *,
    grounding_query: str,
    config: GroundingConfig,
) -> tuple[Image.Image, dict]:
    """Run external GroundingDINO + SAM and return the complement mask."""
    with tempfile.TemporaryDirectory(prefix="fari-grounding-") as directory:
        image_path = os.path.join(directory, "source.png")
        source_image.convert("RGB").save(image_path)
        grounding_model = load_groundingdino_model(
            repo_root=config.repo_root,
            config_path=config.grounding_config,
            checkpoint_path=config.grounding_checkpoint,
            device=config.device,
        )
        predictor = load_sam_predictor(
            repo_root=config.repo_root,
            sam_checkpoint=config.sam_checkpoint,
            sam_arch=config.sam_arch,
            device=config.device,
        )
        candidates = run_grounding_dino(
            repo_root=config.repo_root,
            model=grounding_model,
            image_path=image_path,
            caption=grounding_query,
            box_threshold=config.box_threshold,
            text_threshold=config.text_threshold,
            device=config.device,
        )
        selected = select_best_grounding_candidate(candidates)
        image_np, _ = load_grounding_image_tensor(repo_root=config.repo_root, image_path=image_path)
        edit_mask_np = run_sam_box_segmentation(
            predictor=predictor,
            image_np=image_np,
            box_xyxy=selected["box_xyxy"],
            device=config.device,
        )
    edit_mask = Image.fromarray((np.asarray(edit_mask_np) > 0).astype(np.uint8) * 255, mode="L")
    edit_mask = edit_mask.resize(source_image.size, Image.Resampling.NEAREST)
    return invert_mask(edit_mask), {"grounding_query": grounding_query, "candidate": selected}
