from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EditFamily(str, Enum):
    ADD_OBJECT = "add_object"
    REMOVE_OBJECT = "remove_object"
    CHANGE_POSE = "change_pose"
    CHANGE_COLOR = "change_color"
    CHANGE_OBJECT = "change_object"

    @classmethod
    def parse(cls, value: str | "EditFamily") -> "EditFamily":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "delete_object": cls.REMOVE_OBJECT,
            "pose_geometry": cls.CHANGE_POSE,
            "change_attribute_pose": cls.CHANGE_POSE,
            "color": cls.CHANGE_COLOR,
            "change_attribute_color": cls.CHANGE_COLOR,
            "object_identity": cls.CHANGE_OBJECT,
        }
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)


@dataclass(frozen=True)
class FARIPolicy:
    name: str
    anchoring_blocks: tuple[int, ...] = (2, 4, 8)
    preactivation_blocks: tuple[int, ...] = (17,)
    semantic_blocks: tuple[int, ...] = (6, 9)
    refinement_blocks: tuple[int, ...] = ()

    @property
    def uses_late_refinement(self) -> bool:
        return bool(self.refinement_blocks)


REWRITE_POLICY = FARIPolicy(name="rewrite")
REFINE_POLICY = FARIPolicy(name="refine", refinement_blocks=(35, 36, 37))


def policy_for_family(family: str | EditFamily) -> FARIPolicy:
    resolved = EditFamily.parse(family)
    if resolved in {EditFamily.CHANGE_COLOR, EditFamily.CHANGE_OBJECT}:
        return REFINE_POLICY
    return REWRITE_POLICY
