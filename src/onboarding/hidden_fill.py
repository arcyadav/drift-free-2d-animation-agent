"""Rigger step C — paint what is hidden behind other parts (spec §7 R2).

Classical, no diffusion model: the art is flat-coloured, so a hidden area is filled by
extending the part's own colours outward (nearest non-outline pixel of the same part).
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

Outputs  characters/<c>/parts/front/filled/<part>.png + fill_report.json
Usage: python -m src.onboarding.hidden_fill --character protagonist
"""
import argparse
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab

from src import config
from src.utils.io import save_image, save_text

# (part, extends under these parts, band px, draw outline on the new edge)
UNDERLAP = [
    ("torso", ["upper_arm_l", "upper_arm_r", "forearm_l", "forearm_r", "hand_l", "hand_r"], 14, True),
    ("torso", ["ponytail", "head_hair"], 40, False),
    ("neck", ["head", "head_hair", "ponytail", "strand_r"], 30, False),
    ("head", ["head_hair", "ponytail", "strand_r"], 18, False),
    ("head_hair", ["ponytail"], 25, False),
    ("pelvis", ["torso"], 25, False),
    ("leg_l", ["pelvis"], 30, False),
    ("leg_r", ["pelvis", "leg_l"], 30, False),
    ("foot_l", ["leg_l"], 25, False),
    ("foot_r", ["leg_r"], 25, False),
]
# (part on top, pivot of, disc reaches into these parts, radius = half limb width × factor)
CAPS = [
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
]
OUTLINE_PX = 2
OUTLINE_DL = 15        # outline pixels: L* at least this much darker than the part's median
MIN_CAP_R = 10


def load(char_dir):
    d = char_dir / "parts" / "front"
    meta = json.loads((d / "parts.json").read_text())
    labels = np.asarray(Image.open(d / "labels.png"))
    art = np.asarray(Image.open(char_dir / "views" / "front.png").convert("RGBA"))
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


def extend(rgba, own_src, region):
    """fill `region` with the colour of the nearest pixel in own_src."""
    _, (iy, ix) = ndi.distance_transform_edt(~own_src, return_indices=True)
    rgba[region, :3] = rgba[iy[region], ix[region], :3]
    rgba[region, 3] = 255


def run(character):
    char_dir = config.CHARACTERS / character
    d, meta, labels, art = load(char_dir)
    L = rgb2lab(art[..., :3] / 255.0)[..., 0]
    ids = {n: p["id"] for n, p in meta["parts"].items()}
    masks = {n: labels == i for n, i in ids.items()}
    fg = labels > 0
    solid = ndi.binary_erosion(art[..., 3] == 255, iterations=1)   # fills never reach the soft rim
    parts = {n: np.where(masks[n][..., None], art, 0).astype(np.uint8) for n in ids}
    added = {n: 0 for n in ids}

    order = meta["layer_order"]
    for name, under, band, outline in UNDERLAP:
        behind = [u for u in under if order.index(u) < order.index(name)]
        if behind:
            raise ValueError(f"{name} cannot extend under {behind}: they are drawn behind it")
        own = masks[name]
        cover = np.any([masks[u] for u in under], 0)
        near = ndi.distance_transform_edt(~own) <= band
        region = near & cover & (parts[name][..., 3] == 0) & solid
        if not region.any():
            continue
        extend(parts[name], own_colours(art, own, L), region)
        if outline:
            ext = parts[name][..., 3] > 0
            rim = ext & ~ndi.binary_erosion(ext, iterations=OUTLINE_PX) & region
            parts[name][rim, :3] = outline_colour(art, own, L)
        added[name] += int(region.sum())

    dist_fg = ndi.distance_transform_edt(fg)
    for name, pivot_of, into, factor in CAPS:
        px, py = meta["parts"][pivot_of]["pivot"]
        half_w = dist_fg[int(round(py)), int(round(px))]
        r = max(MIN_CAP_R, factor * half_w)
        yy, xx = np.ogrid[:labels.shape[0], :labels.shape[1]]
        disc = (xx - px) ** 2 + (yy - py) ** 2 <= r * r
        region = disc & np.any([masks[i] for i in into], 0) & (parts[name][..., 3] == 0) & solid
        if not region.any():
            continue
        parts[name][region] = art[region]                       # identical at rest
        added[name] += int(region.sum())

    out = d / "filled"
    out.mkdir(exist_ok=True)
    for n, im in parts.items():
        save_image(im, out / f"{n}.png")
    report = {"added_px": added, "underlap": [u[:3] for u in UNDERLAP], "caps": [c[:2] for c in CAPS]}
    save_text(json.dumps(report, indent=2), out / "fill_report.json")
    for n, a in added.items():
        if a:
            print(f"  {n:12s} +{a}px")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="protagonist")
    run(ap.parse_args().character)
