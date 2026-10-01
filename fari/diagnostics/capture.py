from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from fari.editing.pipeline import FARIEditor, build_shared_noise_latents
from fari.models.hooks import FluxEditController


@dataclass
class DiagnosticRun:
    source_image: object
    target_image: object
    artifacts: dict


class DiagnosticController(FluxEditController):
    """Capture attention maps and changed-token hidden states without altering generation."""

    enable_modify_multimodal_attention_logits = True
    enable_modify_single_attention_logits = True

    def __init__(
        self,
        *,
        source_changed_token_indices: Sequence[int],
        target_changed_token_indices: Sequence[int],
        steps: Sequence[int] | None = None,
        double_blocks: Sequence[int] | None = None,
        single_blocks: Sequence[int] | None = None,
    ) -> None:
        self.source_indices = tuple(int(i) for i in source_changed_token_indices)
        self.target_indices = tuple(int(i) for i in target_changed_token_indices)
        self.steps = None if steps is None else {int(i) for i in steps}
        self.double_blocks = None if double_blocks is None else {int(i) for i in double_blocks}
        self.single_blocks = None if single_blocks is None else {int(i) for i in single_blocks}
        self.attention: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}
        self.hidden: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}

    def _selected(self, stream: str, block: int | None, step: int | None) -> bool:
        if block is None or step is None or (self.steps is not None and int(step) not in self.steps):
            return False
        blocks = self.double_blocks if stream == "double" else self.single_blocks
        return blocks is None or int(block) in blocks

    def _capture_attention(self, *, logits, stream, block_index, step_index, metadata):
        if not self._selected(stream, block_index, step_index) or logits.shape[0] < 2:
            return logits
        text_count = int(metadata.get("text_token_count", 0) or 0)
        image_count = int(metadata.get("image_token_count", 0) or 0)
        if text_count <= 0 or image_count <= 0:
            return logits
        probabilities = logits.float().softmax(dim=-1)
        image_slice = slice(text_count, text_count + image_count)

        def aggregate(branch: int, indices: tuple[int, ...]):
            valid = [i for i in indices if 0 <= i < text_count]
            if not valid:
                return torch.zeros(image_count, dtype=torch.float32)
            return probabilities[branch, :, valid, image_slice].mean(dim=(0, 1)).detach().cpu()

        self.attention[(stream, int(block_index), int(step_index))] = {
            "source": aggregate(0, self.source_indices),
            "target": aggregate(1, self.target_indices),
        }
        return logits

    def modify_multimodal_attention_logits(self, *, attention_logits, block_index, step_index, metadata):
        return self._capture_attention(
            logits=attention_logits,
            stream="double",
            block_index=block_index,
            step_index=step_index,
            metadata=metadata,
        )

    def modify_single_attention_logits(self, *, attention_logits, block_index, step_index, metadata):
        return self._capture_attention(
            logits=attention_logits,
            stream="single",
            block_index=block_index,
            step_index=step_index,
            metadata=metadata,
        )

    def _capture_hidden(self, *, states, stream, block_index, step_index):
        if not self._selected(stream, block_index, step_index) or states.shape[0] < 2:
            return

        def aggregate(branch: int, indices: tuple[int, ...]):
            valid = [i for i in indices if 0 <= i < states.shape[1]]
            if not valid:
                return torch.zeros(states.shape[-1], dtype=torch.float32)
            return states[branch, valid].float().mean(dim=0).detach().cpu()

        self.hidden[(stream, int(block_index), int(step_index))] = {
            "source": aggregate(0, self.source_indices),
            "target": aggregate(1, self.target_indices),
        }

    def on_double_block_end(self, *, block_index, step_index, encoder_hidden_states, hidden_states, metadata):
        del metadata
        self._capture_hidden(
            states=encoder_hidden_states,
            stream="double",
            block_index=block_index,
            step_index=step_index,
        )
        return encoder_hidden_states, hidden_states

    def on_single_block_end(self, *, block_index, step_index, hidden_states, metadata):
        del metadata
        self._capture_hidden(
            states=hidden_states,
            stream="single",
            block_index=block_index,
            step_index=step_index,
        )
        return hidden_states

    def export(self) -> dict:
        return {"attention": self.attention, "hidden": self.hidden}


def run_diagnostic_capture(
    editor: FARIEditor,
    *,
    source_prompt: str,
    target_prompt: str,
    source_changed_token_indices: Sequence[int],
    target_changed_token_indices: Sequence[int],
    seed: int = 42,
    height: int = 1024,
    width: int = 1024,
    steps: int = 50,
    guidance_scale: float = 3.5,
    max_sequence_length: int = 512,
    capture_steps: Sequence[int] | None = None,
    double_blocks: Sequence[int] | None = None,
    single_blocks: Sequence[int] | None = None,
) -> DiagnosticRun:
    latent = build_shared_noise_latents(
        pipe=editor.pipe,
        seed=seed,
        device=editor.device,
        height=height,
        width=width,
    )
    controller = DiagnosticController(
        source_changed_token_indices=source_changed_token_indices,
        target_changed_token_indices=target_changed_token_indices,
        steps=capture_steps,
        double_blocks=double_blocks,
        single_blocks=single_blocks,
    )
    images = editor._generate(
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
    return DiagnosticRun(source_image=images[0], target_image=images[1], artifacts=controller.export())
