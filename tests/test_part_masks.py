"""Part masks of the front view must be complete and anatomically placed.

Run: python -m tests.test_part_masks   (needs views/front.png, keypoints.json, palette.json)
"""
import json

import numpy as np
from PIL import Image

from src import config
from src.onboarding import part_masks

CHAR = config.CHARACTERS / "protagonist"


def centre(lab, name):
    ys, xs = np.where(lab == part_masks.PART_ID[name])
    return xs.mean(), ys.mean()


def test_front_parts():
    assert part_masks.run("protagonist") == []
    lab = np.asarray(Image.open(CHAR / "parts/front/labels.png"))
    alpha = np.asarray(Image.open(CHAR / "views/front.png").convert("RGBA"))[..., 3] > 0
    kp = json.loads((CHAR / "keypoints.json").read_text())["front"]["joints"]

    # every visible pixel (incl. the soft edge) belongs to exactly one part
    assert ((lab > 0) == alpha).all()

    c = {n: centre(lab, n) for n, _ in part_masks.PARTS}
    # top → bottom
    assert c["head_hair"][1] < c["head"][1] < c["neck"][1] < c["torso"][1] < c["pelvis"][1]
    assert c["pelvis"][1] < c["leg_l"][1] < c["foot_l"][1]
    # her right = viewer's left
    for a, b in (("hand_r", "hand_l"), ("leg_r", "leg_l"), ("foot_r", "foot_l"), ("ponytail", "strand_r")):
        assert c[a][0] < c[b][0], f"{a} should be left of {b}"
    # arm chain runs shoulder → elbow → wrist
    for s in "rl":
        assert c[f"upper_arm_{s}"][1] < c[f"forearm_{s}"][1] < c[f"hand_{s}"][1]
        wx, wy = kp[f"{s}_wrist"][:2]
        assert np.hypot(c[f"hand_{s}"][0] - wx, c[f"hand_{s}"][1] - wy) < 120
    print("part masks: all checks passed")


if __name__ == "__main__":
    test_front_parts()
