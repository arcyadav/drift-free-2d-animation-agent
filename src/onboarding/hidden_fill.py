"""Rigger step C — paint what is hidden behind other parts (spec §7 R2).

Classical, no diffusion model: the art is flat-coloured, so a hidden area is filled with
the part's own local colour (median of its clean pixels nearby, outlines and the cover's
shadow excluded).
Two kinds of fill, both invisible in the rest pose:
  * underlap  — a part continues under the part drawn in front of it
                (torso under the arms and ponytail, neck under the head, legs under
                the pelvis, ...). Its new silhouette edge gets the part's outline colour,
                so when the front part moves away the edge looks drawn, not cut.
  * joint cap — a disc around the joint, added to the part drawn on top
                (shoulder, elbow, wrist): rotating about the disc centre keeps the joint
                closed at any angle. The disc copies the original art under it, so it is
                pixel-identical in the rest pose.
A part may only extend under parts drawn after it (layer order) — otherwise the fill
would be visible at rest. Checked, not trusted.
SD inpainting on Kaggle stays a fallback for textured art; flat colours do not need it.

Outputs  characters/<c>/parts/<view>/filled/<part>.png + fill_report.json
Usage: python -m src.onboarding.hidden_fill --character protagonist [--view side_left]
"""
import argparse
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab

from src import config
from src.utils.io import save_image, save_text

# per view. UNDERLAP: (part, extends under these parts, band px, outline on new edge, clip_rows)
#   clip_rows: stay within the part's own top..bottom rows (tee must not grow below its hem)
# CAPS: (part on top, pivot of, disc reaches into these parts, radius = half limb width × factor)
RULES = {
    "front": {
        "underlap": [
            ("torso", ["upper_arm_l", "upper_arm_r", "forearm_l", "forearm_r", "hand_l", "hand_r"], 14, True, False),
            ("torso", ["ponytail", "head_hair"], 40, False, False),
            ("neck", ["head", "head_hair", "ponytail", "strand_r"], 30, False, False),
            ("head", ["head_hair", "ponytail", "strand_r"], 18, False, False),
            ("head_hair", ["ponytail"], 25, False, False),
            ("pelvis", ["torso"], 25, False, False),
            ("leg_l", ["pelvis"], 30, False, False),
            ("leg_r", ["pelvis", "leg_l"], 30, False, False),
            ("foot_l", ["leg_l"], 25, False, False),
            ("foot_r", ["leg_r"], 25, False, False),
        ],
        "caps": [
            ("upper_arm_l", "upper_arm_l", ["torso"], 1.6),
            ("upper_arm_r", "upper_arm_r", ["torso"], 1.6),
            ("forearm_l", "forearm_l", ["upper_arm_l"], 1.0),
            ("forearm_r", "forearm_r", ["upper_arm_r"], 1.0),
            ("upper_arm_l", "forearm_l", ["forearm_l"], 1.0),      # upper arm's elbow end, under the forearm
            ("upper_arm_r", "forearm_r", ["forearm_r"], 1.0),
            ("hand_l", "hand_l", ["forearm_l"], 1.0),
            ("hand_r", "hand_r", ["forearm_r"], 1.0),
            ("forearm_l", "hand_l", ["hand_l"], 1.0),
            ("forearm_r", "hand_r", ["hand_r"], 1.0),
        ],
    },
    # side: the near arm swings over the torso and pelvis, the near leg over the pelvis
    "side_left": {
        "underlap": [
            ("torso", ["upper_arm_l", "forearm_l"], 140, True, True),
            ("torso", ["ponytail", "head_hair"], 15, False, False),
            ("pelvis", ["forearm_l", "hand_l", "torso"], 110, False, True),
            ("pelvis", ["thigh_l"], 45, False, False),
            ("thigh_l", ["hand_l", "forearm_l"], 80, False, True),
            ("shin_l", ["thigh_l"], 35, False, False),
            ("foot_l", ["shin_l"], 30, False, False),
            ("neck", ["head", "head_hair", "ponytail"], 30, False, False),
            ("head", ["head_hair", "ponytail"], 18, False, False),
            ("head_hair", ["ponytail"], 25, False, False),
        ],
        "caps": [
            ("upper_arm_l", "upper_arm_l", ["torso"], 1.0),
            ("forearm_l", "forearm_l", ["upper_arm_l"], 1.0),
            ("upper_arm_l", "forearm_l", ["forearm_l"], 1.0),
            ("hand_l", "hand_l", ["forearm_l"], 1.0),
            ("forearm_l", "hand_l", ["hand_l"], 1.0),
            ("thigh_l", "thigh_l", ["pelvis"], 0.8),
            ("thigh_l", "shin_l", ["shin_l"], 0.8),               # knee: thigh end over the shin
        ],
    },
}
OUTLINE_PX = 2
OUTLINE_DL = 15        # outline pixels: L* at least this much darker than the part's median
MIN_CAP_R = 10
BORDER_SKIP = 5        # ignore the part's pixels this close to the covering part (outline, shadow)
RING_PX = 40           # colour of a hidden patch = median of clean pixels within this distance


def load(char_dir, view):
    d = char_dir / "parts" / view
    meta = json.loads((d / "parts.json").read_text())
    labels = np.asarray(Image.open(d / "labels.png"))
    art = np.asarray(Image.open(char_dir / "views" / f"{view}.png").convert("RGBA"))
    return d, meta, labels, art


def own_colours(art, own, L):
    """flat-fill source: the part's pixels minus its dark outline."""
    Lm = np.median(L[own])
    src = own & (L > Lm - OUTLINE_DL)
    return src if src.sum() > 50 else own


def outline_colour(art, own, L):
    edge = own & ~ndi.binary_erosion(own, iterations=2)
    dark = edge & (L < np.median(L[own]) - OUTLINE_DL)
    px = art[dark][:, :3] if dark.sum() > 20 else art[edge][:, :3]
    return np.median(px, 0)


def extend(rgba, own_src, region, cover):
    """fill each hidden patch with the median colour of the part's clean pixels around it.
    Copying the nearest pixel instead would stretch the cover's outline and shadow (they sit
    right at the border) into streaks — a 'ghost arm' once the arm moves away."""
    src = own_src & ~ndi.binary_dilation(cover, iterations=BORDER_SKIP)
    if src.sum() < 50:
        src = own_src
    pl, n = ndi.label(region)
    for i in range(1, n + 1):
        patch = pl == i
        ring = ndi.binary_dilation(patch, iterations=RING_PX) & src
        pool = rgba[ring if ring.sum() >= 30 else src][:, :3]
        rgba[patch, :3] = np.median(pool, 0)
        rgba[patch, 3] = 255


def run(character, view="front"):
    char_dir = config.CHARACTERS / character
    d, meta, labels, art = load(char_dir, view)
    rules = RULES[view]
    L = rgb2lab(art[..., :3] / 255.0)[..., 0]
    ids = {n: p["id"] for n, p in meta["parts"].items()}
    masks = {n: labels == i for n, i in ids.items()}
    fg = labels > 0
    solid = ndi.binary_erosion(art[..., 3] == 255, iterations=1)   # fills never reach the soft rim
    copies = {n: p for n, p in meta["parts"].items() if p.get("copy_of")}   # far limbs (side view)
    parts = {n: np.asarray(Image.open(d / f"{n}.png").convert("RGBA")).copy()   # cut-outs (clean rims)
             for n in ids if n not in copies}
    added = {n: 0 for n in parts}
    yy, xx = np.mgrid[:labels.shape[0], :labels.shape[1]]

    order = meta["layer_order"]
    for name, under, band, outline, clip_rows in rules["underlap"]:
        behind = [u for u in under if order.index(u) < order.index(name)]
        if behind:
            raise ValueError(f"{name} cannot extend under {behind}: they are drawn behind it")
        own = masks[name]
        cover = np.any([masks[u] for u in under], 0)
        near = ndi.distance_transform_edt(~own) <= band
        region = near & cover & (parts[name][..., 3] == 0) & solid
        if clip_rows:
            rows = np.where(own.any(1))[0]
            region &= (yy >= rows.min()) & (yy <= rows.max())
        if not region.any():
            continue
        extend(parts[name], own_colours(art, own, L), region, cover)
        if outline:
            ext = parts[name][..., 3] > 0
            rim = ext & ~ndi.binary_erosion(ext, iterations=OUTLINE_PX) & region
            parts[name][rim, :3] = outline_colour(art, own, L)
        added[name] += int(region.sum())

    dist_fg = ndi.distance_transform_edt(fg)
    for name, pivot_of, into, factor in rules["caps"]:
        px, py = meta["parts"][pivot_of]["pivot"]
        half_w = dist_fg[int(round(py)), int(round(px))]
        r = max(MIN_CAP_R, factor * half_w)
        disc = (xx - px) ** 2 + (yy - py) ** 2 <= r * r
        region = disc & np.any([masks[i] for i in into], 0) & (parts[name][..., 3] == 0) & solid
        if not region.any():
            continue
        parts[name][region] = art[region]                       # identical at rest
        added[name] += int(region.sum())

    from src.onboarding.part_masks_side import far_copy
    for n, p in copies.items():                                   # far limbs: shaded filled copies
        parts[n] = far_copy(parts[p["copy_of"]], p["shade"])

    out = d / "filled"
    out.mkdir(exist_ok=True)
    for n, im in parts.items():
        save_image(im, out / f"{n}.png")
    report = {"view": view, "added_px": added, "underlap": [u[:3] for u in rules["underlap"]],
              "caps": [c[:2] for c in rules["caps"]]}
    save_text(json.dumps(report, indent=2), out / "fill_report.json")
    for n, a in added.items():
        if a:
            print(f"  {n:12s} +{a}px")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="protagonist")
    ap.add_argument("--view", default="front", choices=list(RULES))
    a = ap.parse_args()
    run(a.character, a.view)
