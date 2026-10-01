from __future__ import annotations

from collections.abc import Sequence

import torch

from fari.editing.policies import FARIPolicy
from fari.models.hooks import FluxEditController


class FARIController(FluxEditController):
    """Copy source K/V features into the target branch by functional zone."""

    def __init__(self, *, policy: FARIPolicy, preserved_token_indices: Sequence[int]) -> None:
        self.policy = policy
        self.preserved_token_indices = tuple(sorted({int(i) for i in preserved_token_indices if int(i) >= 0}))

    def _region(self, *, stream: str, block_index: int | None, image_token_count: int) -> tuple[int, ...] | None:
        if block_index is None:
            return None
        block = int(block_index)
        if stream == "double":
            if block in self.policy.anchoring_blocks:
                return tuple(i for i in self.preserved_token_indices if i < image_token_count)
            if block in self.policy.preactivation_blocks:
                return tuple(range(image_token_count))
        elif stream == "single":
            if block in self.policy.semantic_blocks:
                return tuple(range(image_token_count))
            if block in self.policy.refinement_blocks:
                return tuple(i for i in self.preserved_token_indices if i < image_token_count)
        return None

    @staticmethod
    def _copy_source_kv(
        key: torch.Tensor,
        value: torch.Tensor,
        token_indices: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key.shape[0] < 2 or not token_indices:
            return key, value
        index = torch.as_tensor(tuple(token_indices), dtype=torch.long, device=key.device)
        key_out = key.clone()
        value_out = value.clone()
        key_out[1:, index] = key[:1, index]
        value_out[1:, index] = value[:1, index]
        return key_out, value_out

    def transform_multimodal_attention(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        encoder_query: torch.Tensor,
        encoder_key: torch.Tensor,
        encoder_value: torch.Tensor,
        block_index: int | None,
        step_index: int | None,
        metadata: dict,
    ):
        del step_index, metadata
        region = self._region(stream="double", block_index=block_index, image_token_count=int(key.shape[1]))
        if region is not None:
            key, value = self._copy_source_kv(key, value, region)
        return query, key, value, encoder_query, encoder_key, encoder_value

    def transform_single_attention(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_index: int | None,
        step_index: int | None,
        metadata: dict,
    ):
        del step_index
        text_count = int(metadata.get("text_token_count", 0) or 0)
        image_count = int(metadata.get("image_token_count", 0) or max(0, key.shape[1] - text_count))
        region = self._region(stream="single", block_index=block_index, image_token_count=image_count)
        if region is not None:
            key, value = self._copy_source_kv(key, value, tuple(text_count + i for i in region))
        return query, key, value
