"""Part masks of the front view must be complete and anatomically placed.

Run: python -m tests.test_part_masks   (needs views/front.png, keypoints.json, palette.json)
"""
import json

import numpy as np
from scipy import ndimage as ndi
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


def test_side_parts():
    from src.onboarding import part_masks_side as pms
    assert pms.run("protagonist") == []
    lab = np.asarray(Image.open(CHAR / "parts/side_left/labels.png"))
    alpha = np.asarray(Image.open(CHAR / "views/side_left.png").convert("RGBA"))[..., 3] > 0
    assert ((lab > 0) == alpha).all()

    def c(n):
        ys, xs = np.where(lab == pms.PART_ID[n])
        return xs.mean(), ys.mean()
    assert c("head")[1] < c("neck")[1] < c("torso")[1] < c("pelvis")[1] < c("thigh_l")[1] \
        < c("shin_l")[1] < c("foot_l")[1]
    assert c("upper_arm_l")[1] < c("forearm_l")[1] < c("hand_l")[1]
    assert c("ponytail")[0] > c("head")[0], "she faces left: ponytail hangs behind (right)"
    for limb in pms.LIMBS:                         # far limbs: darker copies of the near ones
        far = np.asarray(Image.open(CHAR / f"parts/side_left/{limb}_r.png"))
        near = np.asarray(Image.open(CHAR / f"parts/side_left/{limb}_l.png"))
        region = ndi.binary_dilation((lab == pms.PART_ID[f"{limb}_l"]) | (near[..., 3] > 0), iterations=1)
        on = far[..., 3] > 0
        assert not (on & ~region).any(), "far copy must stay inside the near limb's region"
        assert on.sum() > 0.7 * (near[..., 3] > 0).sum(), "far copy too thin"
        assert far[on][:, :3].mean() < near[on & (near[..., 3] > 0)][:, :3].mean() + 1
    print("side part masks: all checks passed")


if __name__ == "__main__":
    test_front_parts()
    test_side_parts()
