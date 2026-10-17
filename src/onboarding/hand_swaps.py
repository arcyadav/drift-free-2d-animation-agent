"""Hand swap images (spec §7): finger poses the rig cannot bend (open palm for the wave).

The hand images are full-body edits of the front view with only one hand changed, so
  1. clean them with the front view's transform (same scale / placement as the rig)
  2. cut the hand: past the wrist (plane across the forearm), near the wrist, foreground
  3. check alignment: outside the hand the edit must match the front view
Output: characters/<c>/hands/front/<pose>_<side>.png (full-canvas RGBA, pivot = wrist)

Usage: python -m src.onboarding.hand_swaps --character protagonist
"""
import argparse
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab

from src import config
from src.onboarding.sheet_cleaner import clean_image
from src.utils.io import save_image, save_text

SWAPS = {"open_front": ("open", "l"), "fist_front": ("fist", "l")}   # raw file → (pose, side)
HAND_REACH = 175        # px from the wrist that can still be hand
DENIM_B = -6.0         # b* below this (bluish) and not too light = jeans, not hand
ALIGN_MAX = 6.0         # mean abs diff (0-255) allowed between edit and view, away from the hand


def run(character):
    char = config.CHARACTERS / character
    meta = json.loads((char / "parts" / "front" / "parts.json").read_text())
    _, t_front, _, size = clean_image(config.ASSETS_RAW / "turnaround" / "front.png")
    view = np.asarray(Image.open(char / "views" / "front.png").convert("RGBA")).astype(float)
    out = char / "hands" / "front"
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for raw, (pose, s) in SWAPS.items():
        p = config.ASSETS_RAW / "hands" / f"{raw}.png"
        if not p.exists():
            continue
        im, _, _, _ = clean_image(p, transform=t_front, ref_size=size)
        a = np.asarray(im).copy()
        wr = np.array(meta["parts"][f"hand_{s}"]["pivot"])
        el = np.array(meta["parts"][f"forearm_{s}"]["pivot"])
        d = (wr - el) / np.linalg.norm(wr - el)
        yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
        P = np.stack([xx, yy], -1).astype(float)
        past = ((P - wr) @ d) > -4                                  # a little overlap into the wrist
        near = np.linalg.norm(P - wr, axis=-1) < HAND_REACH
        # not denim: the jeans next to the hand are past the wrist too (gave a blue triangle)
        lab = rgb2lab(a[..., :3] / 255.0)
        denim = (lab[..., 2] < DENIM_B) & (lab[..., 0] < 75)
        cand = past & near & (a[..., 3] > 0) & ~denim
        # the hand = the connected piece that starts at the wrist
        cl, _ = ndi.label(ndi.binary_closing(cand, iterations=2) & cand)
        seed = np.round(wr + d * 25).astype(int)
        lid = cl[seed[1], seed[0]]
        if lid == 0:
            dd, (iy, ix) = ndi.distance_transform_edt(cl == 0, return_indices=True)
            lid = cl[iy[seed[1], seed[0]], ix[seed[1], seed[0]]]
        hand = ndi.binary_dilation(cl == lid, iterations=1) & (a[..., 3] > 0) & past & near
        cut = a.copy()
        cut[~hand, 3] = 0
        save_image(cut, out / f"{pose}_{s}.png")
        # alignment: away from the hand the edit must equal the front view
        far = ~(np.linalg.norm(P - wr, axis=-1) < HAND_REACH + 40) & (view[..., 3] > 200) & (a[..., 3] > 200)
        diff = float(np.abs(a[..., :3].astype(float) - view[..., :3])[far].mean())
        report[f"{pose}_{s}"] = {"px": int(hand.sum()), "align_mean_diff": round(diff, 2),
                                 "aligned": diff <= ALIGN_MAX, "pivot": wr.round(1).tolist()}
        print(f"{pose}_{s}: {hand.sum()} px, alignment diff {diff:.2f} ({'ok' if diff <= ALIGN_MAX else 'CHECK'})")
    save_text(json.dumps(report, indent=2), out / "hands.json")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="protagonist")
    run(ap.parse_args().character)
