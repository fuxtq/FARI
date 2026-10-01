from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _csv_ints(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _dtype(name: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)


def _load_editor(args):
    from fari import FARIEditor

    return FARIEditor.from_pretrained(
        args.model,
        device=args.device,
        dtype=_dtype(args.dtype),
        token=os.environ.get("HF_TOKEN"),
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        cpu_offload=args.cpu_offload,
    )


def _run_edit(args) -> None:
    from fari.editing.grounding import GroundingConfig

    editor = _load_editor(args)
    grounding = None
    if not args.preserve_mask:
        required = (args.grounded_sam_root, args.grounding_config, args.grounding_checkpoint, args.sam_checkpoint)
        if not all(required):
            raise ValueError("Provide --preserve-mask or all Grounded-SAM path arguments.")
        grounding = GroundingConfig(
            repo_root=args.grounded_sam_root,
            grounding_config=args.grounding_config,
            grounding_checkpoint=args.grounding_checkpoint,
            sam_checkpoint=args.sam_checkpoint,
            sam_arch=args.sam_arch,
            device=args.device,
        )
    result = editor.edit(
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        family=args.family,
        seed=args.seed,
        preserve_mask=args.preserve_mask,
        grounding_query=args.grounding_query,
        grounding_config=grounding,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
    )
    result.save(args.output)
    Path(args.output, "metadata.json").write_text(
        json.dumps(result.metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_diagnose(args) -> None:
    import numpy as np
    import torch
    from PIL import Image

    from fari.diagnostics import (
        attention_kl,
        edit_region_mask,
        hidden_state_separability,
        run_diagnostic_capture,
        spatial_localization,
    )
    from fari.editing.masks import infer_token_grid

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.probe_artifacts:
        paths = [Path(item.strip()) for item in args.probe_artifacts.split(",") if item.strip()]
        if len(paths) < 2:
            raise ValueError("Hidden-state separability needs at least two paired diagnostic artifacts.")
        grouped: dict[tuple, dict[str, list]] = {}
        for path in paths:
            payload = torch.load(path, map_location="cpu")
            for key, features in payload.get("hidden", {}).items():
                group = grouped.setdefault(tuple(key), {"source": [], "target": [], "groups": []})
                group["source"].append(features["source"].numpy())
                group["target"].append(features["target"].numpy())
                group["groups"].append(path.stem)
        probe_rows = []
        for (stream, block, step), features in sorted(grouped.items()):
            scores = hidden_state_separability(
                np.stack(features["source"]),
                np.stack(features["target"]),
                groups=features["groups"],
                cv_folds=min(5, len(features["groups"])),
            )
            probe_rows.append({"stream": stream, "block_index": block, "step_index": step, **scores})
        (output / "hidden_probe_summary.json").write_text(
            json.dumps(probe_rows, indent=2) + "\n",
            encoding="utf-8",
        )
        return

    if not all((args.source_prompt, args.target_prompt, args.source_changed_tokens, args.target_changed_tokens)):
        raise ValueError("Capture mode requires both prompts and both changed-token index lists.")
    editor = _load_editor(args)
    run = run_diagnostic_capture(
        editor,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        source_changed_token_indices=_csv_ints(args.source_changed_tokens) or [],
        target_changed_token_indices=_csv_ints(args.target_changed_tokens) or [],
        seed=args.seed,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        capture_steps=_csv_ints(args.capture_steps),
        double_blocks=_csv_ints(args.double_blocks),
        single_blocks=_csv_ints(args.single_blocks),
    )
    run.source_image.save(output / "source.png")
    run.target_image.save(output / "target.png")
    torch.save(run.artifacts, output / "diagnostics.pt")
    final_region = edit_region_mask(run.source_image, run.target_image)
    summary = []
    for (stream, block, step), maps in sorted(run.artifacts["attention"].items()):
        source = maps["source"].numpy()
        target = maps["target"].numpy()
        rows, cols = infer_token_grid(len(target), height=args.height, width=args.width)
        localization = spatial_localization(target.reshape(rows, cols), final_region)
        summary.append(
            {
                "stream": stream,
                "block_index": block,
                "step_index": step,
                "attention_kl_source_target": attention_kl(source, target),
                "spatial_localization_iou": localization.iou,
                "spatial_localization_precision": localization.precision,
                "spatial_localization_recall": localization.recall,
            }
        )
    (output / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    Image.fromarray(final_region.astype(np.uint8) * 255, mode="L").save(output / "final_edit_region.png")


def _run_build_data(args) -> None:
    from fari.data import extract_coco_seeds, generate_controlled_pairs

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = extract_coco_seeds(args.coco_captions, per_category=args.per_category, seed=args.seed)
    (output / "coco_seeds.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in seeds),
        encoding="utf-8",
    )
    if not args.seeds_only:
        generate_controlled_pairs(
            seeds,
            output_jsonl=str(output / "controlled_pairs.jsonl"),
            per_family=args.per_family,
            model=args.llm_model,
            temperature=args.temperature,
        )


def _run_evaluate(args) -> None:
    from fari.evaluation import CLIPScorer, DINOScorer, LPIPSScorer, evaluate_manifest

    clip = CLIPScorer(args.clip_model, device=args.device) if args.clip_model else None
    dino = DINOScorer(args.dino_model, device=args.device) if args.dino_model else None
    lpips = LPIPSScorer(net=args.lpips_net, device=args.device) if args.lpips else None
    rows = evaluate_manifest(
        args.manifest,
        output_jsonl=args.output,
        clip_scorer=clip,
        dino_scorer=dino,
        lpips_scorer=lpips,
    )
    print(f"evaluated {len(rows)} image pairs -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fari", description="Function-Aware Regional Injection")
    commands = parser.add_subparsers(dest="command", required=True)

    edit = commands.add_parser("edit", help="Run one shared-noise FARI edit")
    _add_model_args(edit)
    edit.add_argument("--source-prompt", required=True)
    edit.add_argument("--target-prompt", required=True)
    edit.add_argument(
        "--family",
        required=True,
        choices=("add_object", "remove_object", "change_pose", "change_color", "change_object"),
    )
    edit.add_argument("--preserve-mask")
    edit.add_argument("--grounding-query")
    edit.add_argument("--grounded-sam-root")
    edit.add_argument("--grounding-config")
    edit.add_argument("--grounding-checkpoint")
    edit.add_argument("--sam-checkpoint")
    edit.add_argument("--sam-arch", default="vit_h")
    edit.add_argument("--output", required=True)
    edit.set_defaults(func=_run_edit)

    diagnose = commands.add_parser("diagnose", help="Capture the three mechanism diagnostics")
    _add_model_args(diagnose)
    diagnose.add_argument("--source-prompt")
    diagnose.add_argument("--target-prompt")
    diagnose.add_argument("--source-changed-tokens", help="Comma-separated tokenizer indices")
    diagnose.add_argument("--target-changed-tokens", help="Comma-separated tokenizer indices")
    diagnose.add_argument("--probe-artifacts", help="Comma-separated diagnostics.pt files for linear probing")
    diagnose.add_argument("--capture-steps", help="Comma-separated denoising step indices; default: all")
    diagnose.add_argument("--double-blocks", help="Comma-separated block indices; default: all")
    diagnose.add_argument("--single-blocks", help="Comma-separated block indices; default: all")
    diagnose.add_argument("--output", required=True)
    diagnose.set_defaults(func=_run_diagnose)

    data = commands.add_parser("build-data", help="Recreate the Controlled Edit construction protocol")
    data.add_argument("--coco-captions", required=True)
    data.add_argument("--output", required=True)
    data.add_argument("--seed", type=int, default=0)
    data.add_argument("--per-category", type=int, default=25)
    data.add_argument("--per-family", type=int, default=20)
    data.add_argument("--llm-model", default="deepseek-chat")
    data.add_argument("--temperature", type=float, default=0.3)
    data.add_argument("--seeds-only", action="store_true")
    data.set_defaults(func=_run_build_data)

    evaluate = commands.add_parser("evaluate", help="Evaluate a JSONL generation manifest")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--clip-model", help="Optional CLIP model path or Hugging Face id")
    evaluate.add_argument("--dino-model", help="Optional DINOv2 model path or Hugging Face id")
    evaluate.add_argument("--lpips", action="store_true")
    evaluate.add_argument("--lpips-net", default="alex")
    evaluate.set_defaults(func=_run_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
