import numpy as np
from PIL import Image

from fari.editing.masks import infer_token_grid, invert_mask, preserved_token_indices


def test_grid_respects_aspect_ratio():
    assert infer_token_grid(8, height=32, width=64) == (2, 4)


def test_mask_to_tokens_and_inversion():
    mask = Image.fromarray(np.array([[255, 0], [0, 255]], dtype=np.uint8), mode="L")
    assert preserved_token_indices(mask, image_token_count=4, height=2, width=2) == [0, 3]
    assert preserved_token_indices(invert_mask(mask), image_token_count=4, height=2, width=2) == [1, 2]
