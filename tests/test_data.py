import json

from fari.data.controlled_edit import extract_coco_seeds


def test_coco_seed_filtering(tmp_path):
    captions = {
        "annotations": [
            {"id": 1, "image_id": 10, "caption": "A dog happily runs near grass"},
            {"id": 2, "image_id": 11, "caption": "A dog and cat sit together"},
            {"id": 3, "image_id": 12, "caption": "A cup sits on a table"},
        ]
    }
    path = tmp_path / "captions.json"
    path.write_text(json.dumps(captions), encoding="utf-8")
    rows = extract_coco_seeds(str(path), per_category=25)
    assert len(rows) == 1
    assert rows[0]["main_keyword"] == "dog"
    assert rows[0]["category"] == "animal"
