import numpy as np
from PIL import Image

from fari.diagnostics.metrics import attention_kl, edit_region_mask, spatial_localization
from fari.diagnostics.probe import hidden_state_separability


def test_attention_kl_zero_for_equal_maps():
    values = np.array([0.1, 0.2, 0.7])
    assert attention_kl(values, values) == 0.0


def test_attention_kl_is_directional():
    source = np.array([0.8, 0.1, 0.1])
    target = np.array([0.4, 0.3, 0.3])
    assert attention_kl(source, target) != attention_kl(target, source)


def test_spatial_localization_perfect_overlap():
    attention = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    edit = np.array([[True, False], [False, False]])
    result = spatial_localization(attention, edit)
    assert result.iou == 1.0


def test_edit_region_uses_rgb_mean_threshold():
    source = Image.new("RGB", (2, 2), "black")
    edited = Image.new("RGB", (2, 2), "black")
    edited.putpixel((0, 0), (255, 255, 255))
    mask = edit_region_mask(source, edited)
    assert mask.sum() == 1


def test_linear_probe_detects_separable_vectors():
    rng = np.random.default_rng(0)
    source = rng.normal(-2, 0.1, size=(12, 4))
    target = rng.normal(2, 0.1, size=(12, 4))
    result = hidden_state_separability(source, target, cv_folds=3)
    assert result["accuracy"] > 0.95
