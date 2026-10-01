import pytest

from fari.editing.policies import EditFamily, policy_for_family


@pytest.mark.parametrize("family", ["add_object", "remove_object", "change_pose", "pose_geometry"])
def test_rewrite_families(family):
    policy = policy_for_family(family)
    assert policy.name == "rewrite"
    assert policy.refinement_blocks == ()


@pytest.mark.parametrize("family", ["change_color", "change_object", "color", "object_identity"])
def test_refine_families(family):
    policy = policy_for_family(family)
    assert policy.name == "refine"
    assert policy.refinement_blocks == (35, 36, 37)


def test_unknown_family_is_rejected():
    with pytest.raises(ValueError):
        EditFamily.parse("background_replacement")
