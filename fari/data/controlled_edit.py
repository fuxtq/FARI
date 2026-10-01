from __future__ import annotations

import json
import os
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path


CATEGORY_LEXICON = {
    "animal": ("dog", "cat", "horse", "bird", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"),
    "person": ("man", "woman", "boy", "girl", "child", "person"),
    "vehicle": ("car", "truck", "bus", "train", "bicycle", "motorcycle", "boat", "airplane"),
    "furniture": ("chair", "couch", "bench", "bed", "table", "desk"),
    "container": ("bottle", "cup", "bowl", "vase", "sink"),
    "food": ("apple", "banana", "orange", "pizza", "cake", "sandwich", "donut"),
    "handheld_item": ("book", "umbrella", "bag", "suitcase", "frisbee", "kite"),
    "device": ("laptop", "clock", "phone", "remote", "keyboard", "mouse"),
    "indoor_scene": ("kitchen", "bathroom", "bedroom", "room"),
    "outdoor_scene": ("beach", "street", "park", "field", "road", "sidewalk"),
}

EDIT_FAMILIES = ("add_object", "remove_object", "change_pose", "change_object", "change_color")
EXCLUDED_WORDS = {"and", "while", "together", "group", "groups", "crowd", "two", "three", "four", "five"}

ROUTER_SYSTEM_PROMPT = """You are constructing a scientific validation dataset for text-guided image editing.
Use the COCO caption only as a semantic anchor. Select only genuinely compatible edit families and create short, concrete source and target prompts that differ by exactly one controlled visual change.
Return strict JSON with one top-level key named candidates. Every candidate must contain family, category, source_prompt, target_prompt, changed_span_source, changed_span_target, and invariants. It is acceptable to return fewer candidates than requested."""


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def extract_coco_seeds(
    captions_json: str,
    *,
    per_category: int = 25,
    seed: int = 0,
) -> list[dict]:
    """Apply the filtering protocol from Supplementary Sec. S2.1."""
    payload = json.loads(Path(captions_json).read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for annotation in payload.get("annotations", []):
        caption = " ".join(str(annotation.get("caption", "")).strip().split())
        tokens = _words(caption)
        if not 4 <= len(tokens) <= 12 or EXCLUDED_WORDS.intersection(tokens):
            continue
        matches = []
        for category, keywords in CATEGORY_LEXICON.items():
            for keyword in keywords:
                if tokens.count(keyword) == 1:
                    matches.append((category, keyword))
        if len(matches) != 1 or caption.lower() in seen:
            continue
        category, keyword = matches[0]
        seen.add(caption.lower())
        grouped[category].append(
            {
                "coco_caption_id": int(annotation["id"]),
                "coco_image_id": int(annotation["image_id"]),
                "seed_prompt": caption,
                "category": category,
                "main_keyword": keyword,
            }
        )
    rng = random.Random(int(seed))
    selected: list[dict] = []
    for category in CATEGORY_LEXICON:
        rows = grouped.get(category, [])
        rng.shuffle(rows)
        selected.extend(rows[: int(per_category)])
    for index, row in enumerate(selected):
        row["seed_id"] = f"seed_{index:04d}"
    return selected


def _deepseek_json(*, api_key: str, system: str, user: str, model: str, temperature: float) -> dict:
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"},
                "temperature": float(temperature),
                "max_tokens": 1500,
            }
        ).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return json.loads(payload["choices"][0]["message"]["content"])


def generate_controlled_pairs(
    seeds: list[dict],
    *,
    output_jsonl: str,
    api_key: str | None = None,
    per_family: int = 20,
    model: str = "deepseek-chat",
    temperature: float = 0.3,
) -> list[dict]:
    key = str(api_key or os.environ.get("DEEPSEEK_API_KEY", "")).strip()
    if not key:
        raise ValueError("DEEPSEEK_API_KEY is required to generate controlled pairs.")
    counts = {family: 0 for family in EDIT_FAMILIES}
    output: list[dict] = []
    for seed_row in seeds:
        remaining = [family for family, count in counts.items() if count < int(per_family)]
        if not remaining:
            break
        user = json.dumps(
            {
                "coco_seed_caption": seed_row["seed_prompt"],
                "seed_category": seed_row["category"],
                "main_keyword": seed_row["main_keyword"],
                "candidate_families": remaining,
            },
            ensure_ascii=False,
        )
        response = _deepseek_json(
            api_key=key,
            system=ROUTER_SYSTEM_PROMPT,
            user=user,
            model=model,
            temperature=temperature,
        )
        for candidate in response.get("candidates", []):
            family = str(candidate.get("family", "")).strip()
            if family not in remaining or counts[family] >= int(per_family):
                continue
            if not str(candidate.get("source_prompt", "")).strip() or not str(candidate.get("target_prompt", "")).strip():
                continue
            row = {**seed_row, **candidate, "family_index": counts[family]}
            row["id"] = f"{family}_{counts[family]:04d}"
            output.append(row)
            counts[family] += 1
    path = Path(output_jsonl)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    return output
