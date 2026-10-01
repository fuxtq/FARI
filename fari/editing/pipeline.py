from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image

from fari.editing.controller import FARIController
from fari.editing.grounding import GroundingConfig, build_preserve_mask, infer_grounding_query
from fari.editing.masks import preserved_token_indices
from fari.editing.policies import EditFamily, FARIPolicy, policy_for_family
from fari.models.flux import load_flux_pipeline


@dataclass
class EditResult:
    source_image: Image.Image
    edited_image: Image.Image
    preserve_mask: Image.Image
    policy: FARIPolicy
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.source_image.save(os.path.join(output_dir, "source.png"))
        self.edited_image.save(os.path.join(output_dir, "edited.png"))
        self.preserve_mask.save(os.path.join(output_dir, "preserve_mask.png"))


def build_shared_noise_latents(*, pipe, seed: int, device: str, height: int, width: int) -> torch.Tensor:
    transformer = pipe.transformer
    in_channels = int(getattr(transformer.config, "in_channels", 64))
    vae_scale_factor = int(getattr(pipe, "vae_scale_factor", 8))
    if getattr(pipe, "vae", None) is not None:
        channels = getattr(pipe.vae.config, "block_out_channels", [])
        vae_scale_factor = 2 ** (len(channels) or 3)
    packed_height = 2 * (int(height) // vae_scale_factor)
    packed_width = 2 * (int(width) // vae_scale_factor)
    image_token_count = (packed_height // 2) * (packed_width // 2)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(
        (image_token_count, in_channels),
        generator=generator,
        device=device,
        dtype=pipe.transformer.dtype,
    )


class FARIEditor:
    def __init__(self, pipe, *, device: str = "cuda") -> None:
        self.pipe = pipe
        self.device = str(device)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str = "black-forest-labs/FLUX.1-dev",
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        token: str | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        cpu_offload: bool = False,
    ) -> "FARIEditor":
        pipe = load_flux_pipeline(
            model_path,
            torch_dtype=dtype,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        return cls(pipe, device=device)

    def _generate(
        self,
        prompts: list[str],
        *,
        latent: torch.Tensor,
        seed: int,
        height: int,
        width: int,
        steps: int,
        guidance_scale: float,
        max_sequence_length: int,
        controller=None,
    ) -> list[Image.Image]:
        batch_latents = latent.unsqueeze(0).repeat(len(prompts), 1, 1)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        output = self.pipe(
            prompts,
            height=int(height),
            width=int(width),
            guidance_scale=[float(guidance_scale)] * len(prompts),
            output_type="pil",
            num_inference_steps=int(steps),
            max_sequence_length=int(max_sequence_length),
            latents=batch_latents,
            generator=generator,
            edit_controller=controller,
        )
        return list(output.images)

    def edit(
        self,
        *,
        source_prompt: str,
        target_prompt: str,
        family: str | EditFamily,
        seed: int = 42,
        preserve_mask: Image.Image | str | None = None,
        grounding_query: str | None = None,
        grounding_config: GroundingConfig | None = None,
        height: int = 1024,
        width: int = 1024,
        steps: int = 50,
        guidance_scale: float = 3.5,
        max_sequence_length: int = 512,
    ) -> EditResult:
        resolved_family = EditFamily.parse(family)
        policy = policy_for_family(resolved_family)
        latent = build_shared_noise_latents(
            pipe=self.pipe,
            seed=seed,
            device=self.device,
            height=height,
            width=width,
        )
        grounding_metadata: dict[str, Any] = {}
        if preserve_mask is None:
            if grounding_config is None:
                raise ValueError("Provide preserve_mask or grounding_config.")
            preview = self._generate(
                [source_prompt],
                latent=latent,
                seed=seed,
                height=height,
                width=width,
                steps=steps,
                guidance_scale=guidance_scale,
                max_sequence_length=max_sequence_length,
            )[0]
            query = grounding_query or infer_grounding_query(
                source_prompt=source_prompt,
                target_prompt=target_prompt,
                edit_family=resolved_family.value,
            )
            preserve_mask, grounding_metadata = build_preserve_mask(
                preview,
                grounding_query=query,
                config=grounding_config,
            )
        mask_image = Image.open(preserve_mask).convert("L") if isinstance(preserve_mask, str) else preserve_mask.convert("L")
        token_indices = preserved_token_indices(
            mask_image,
            image_token_count=int(latent.shape[0]),
            height=height,
            width=width,
        )
        if not token_indices:
            raise ValueError("The preserve mask contains no preserved image tokens.")
        controller = FARIController(policy=policy, preserved_token_indices=token_indices)
        images = self._generate(
            [source_prompt, target_prompt],
            latent=latent,
            seed=seed,
            height=height,
            width=width,
            steps=steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            controller=controller,
        )
        return EditResult(
            source_image=images[0],
            edited_image=images[1],
            preserve_mask=mask_image,
            policy=policy,
            metadata={
                "family": resolved_family.value,
                "seed": int(seed),
                "policy": policy.name,
                "preserved_token_count": len(token_indices),
                **grounding_metadata,
            },
        )
