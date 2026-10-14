"""Hidden-area fill must be invisible at rest and only ever add pixels.

Run: python -m tests.test_hidden_fill   (needs parts/front from part_masks)
"""
import numpy as np
from PIL import Image

from src import config
from src.animation import rigid_pose
from src.onboarding import hidden_fill

CHAR = config.CHARACTERS / "protagonist"


def test_fill():
    hidden_fill.run("protagonist")
    meta, filled = rigid_pose.load_rig(CHAR, "front", "filled")
    _, raw = rigid_pose.load_rig(CHAR, "front")

    # 1. rest pose with filled parts == original art (fills are hidden behind front parts)
    art = np.asarray(Image.open(CHAR / "views/front.png").convert("RGBA")).astype(float)
    a = art[..., 3:] / 255
    ref = art[..., :3] * a + 255 * (1 - a)
    diff = np.abs(rigid_pose.render(meta, filled, {}).astype(float) - ref).max(-1)
    assert diff.max() <= 2, f"rest pose changed by fill: max diff {diff.max():.0f}"

    # 2. a filled part keeps every original pixel unchanged
    for n in meta["layer_order"]:
        keep = raw[n][..., 3] > 0
        assert (filled[n][keep] == raw[n][keep]).all(), f"{n}: original pixels altered"

    # 3. the parts that open gaps when posed (elbow, torso side, neck) really got filled
    for n in ("forearm_l", "torso", "neck"):
        assert (filled[n][..., 3] > 0).sum() > (raw[n][..., 3] > 0).sum(), f"{n}: nothing filled"
    print("hidden fill: all checks passed")


if __name__ == "__main__":
    test_fill()
