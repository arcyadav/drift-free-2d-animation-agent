"""Hidden-area fill must be invisible at rest and only ever add pixels.

Run: python -m tests.test_hidden_fill   (needs parts/front from part_masks)
"""
import numpy as np
from PIL import Image

from src import config
from src.animation import rigid_pose
from src.onboarding import hidden_fill

CHAR = config.CHARACTERS / "protagonist"


def test_fill(view="front", must_fill=("forearm_l", "torso", "neck")):
    hidden_fill.run("protagonist", view)
    meta, filled = rigid_pose.load_rig(CHAR, view, "filled")
    _, raw = rigid_pose.load_rig(CHAR, view)

    # 1. rest pose: filled parts render exactly like the plain cut-outs (fills stay hidden)
    #    and like the original art away from the edges (edges lose background halo on purpose)
    filled_img = rigid_pose.render(meta, filled, {}).astype(float)
    raw_img = rigid_pose.render(meta, raw, {}).astype(float)
    diff = np.abs(filled_img - raw_img).max(-1)
    assert diff.max() <= 2, f"rest pose changed by fill: max diff {diff.max():.0f}"
    art = np.asarray(Image.open(CHAR / f"views/{view}.png").convert("RGBA")).astype(float)
    a = art[..., 3:] / 255
    ref = art[..., :3] * a + 255 * (1 - a)
    from scipy import ndimage as ndi
    core = ndi.binary_erosion(art[..., 3] == 255, iterations=6)
    d_art = np.abs(raw_img - ref).max(-1)
    assert d_art[core].max() <= 2, f"cut-outs differ from the art inside the body: {d_art[core].max():.0f}"

    # 2. a filled part keeps every original pixel unchanged
    for n in meta["layer_order"]:
        if meta["parts"][n].get("copy_of"):      # far limbs are rebuilt from the filled near limb
            continue
        keep = raw[n][..., 3] > 0
        assert (filled[n][keep] == raw[n][keep]).all(), f"{n}: original pixels altered"

    # 3. the parts that open gaps when posed (elbow, torso side, neck) really got filled
    for n in must_fill:
        assert (filled[n][..., 3] > 0).sum() > (raw[n][..., 3] > 0).sum(), f"{n}: nothing filled"
    print(f"hidden fill ({view}): all checks passed")


if __name__ == "__main__":
    test_fill()
    test_fill("side_left", ("torso", "pelvis", "thigh_l", "shin_l"))
