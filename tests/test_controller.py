import torch

from fari.editing.controller import FARIController
from fari.editing.policies import REFINE_POLICY, REWRITE_POLICY


def _double(controller, block):
    key = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)
    value = key + 100
    zeros = torch.zeros_like(key)
    result = controller.transform_multimodal_attention(
        query=zeros,
        key=key,
        value=value,
        encoder_query=zeros,
        encoder_key=zeros,
        encoder_value=zeros,
        block_index=block,
        step_index=3,
        metadata={},
    )
    return key, value, result[1], result[2]


def test_anchor_copies_only_preserved_image_tokens():
    controller = FARIController(policy=REWRITE_POLICY, preserved_token_indices=[0, 2])
    original_key, _, key, _ = _double(controller, 2)
    assert torch.equal(key[1, [0, 2]], original_key[0, [0, 2]])
    assert torch.equal(key[1, [1, 3]], original_key[1, [1, 3]])


def test_preactivation_copies_full_image():
    controller = FARIController(policy=REWRITE_POLICY, preserved_token_indices=[0])
    original_key, _, key, _ = _double(controller, 17)
    assert torch.equal(key[1], original_key[0])


def test_semantic_copies_only_image_portion_in_single_stream():
    controller = FARIController(policy=REWRITE_POLICY, preserved_token_indices=[0])
    key = torch.arange(2 * 7, dtype=torch.float32).reshape(2, 7, 1)
    _, copied, _ = controller.transform_single_attention(
        query=key,
        key=key,
        value=key,
        block_index=6,
        step_index=0,
        metadata={"text_token_count": 3, "image_token_count": 4},
    )
    assert torch.equal(copied[1, :3], key[1, :3])
    assert torch.equal(copied[1, 3:], key[0, 3:])


def test_late_refinement_is_family_policy_dependent():
    refine = FARIController(policy=REFINE_POLICY, preserved_token_indices=[1])
    rewrite = FARIController(policy=REWRITE_POLICY, preserved_token_indices=[1])
    key = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6, 1)
    kwargs = dict(query=key, key=key, value=key, block_index=35, step_index=0, metadata={"text_token_count": 2, "image_token_count": 4})
    _, refine_key, _ = refine.transform_single_attention(**kwargs)
    _, rewrite_key, _ = rewrite.transform_single_attention(**kwargs)
    assert refine_key[1, 3] == key[0, 3]
    assert torch.equal(rewrite_key, key)
