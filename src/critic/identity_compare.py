"""Identity comparison: the same measurements on an AI-video clip, our clip and the drawing.

Per frame (character = pixels that differ from the flat backdrop):
  * body proportions: silhouette width in the torso band (25-40 % of body height from the top) and
    at the hips (48-55 %), divided by body height — compared with the original side-view drawing
  * size stability: body height in px over the clip (an AI video may rescale the character)
  * matched pose: the AI clip's first frame (standing, side view, like the drawing) vs the drawing
Our clip's silhouette comes from the critic's green-key render (exact); the AI clip's from its flat
backdrop. (A colour test like the critic's was tried and dropped here: it called both clips 0 %
off-model, so it cannot separate them.)

Usage: python -m src.critic.identity_compare --video assets/baseline/<kling>.mp4
"""
import argparse
import json

import imageio.v2 as iio
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from src import config
from src.animation.animator import Walker
from src.animation.seal import KEY, silhouette
from src.utils.io import save_text

BANDS = {"head": (0.02, 0.10), "torso": (0.25, 0.40)}   # rows (share of body height from the top)
BG_DIST = 60                 # |rgb - backdrop| summed: above = character


def character_mask(img, bg=None):
    """biggest blob that differs from the backdrop colour (estimated from the top corners)."""
    if bg is None:
        bg = np.median(np.concatenate([img[:40, :40].reshape(-1, 3), img[:40, -40:].reshape(-1, 3)]), 0)
    m = np.abs(img.astype(int) - bg).sum(-1) > BG_DIST
    m = ndi.binary_opening(m, iterations=2)
    lab, n = ndi.label(m)
    if not n:
        return m
    return ndi.binary_fill_holes(lab == np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)


def proportions(m):
    ys = np.where(m.any(1))[0]
    top, bot = ys.min(), ys.max()
    h = bot - top + 1
    out = {"height_px": int(h)}
    for name, (a, b) in BANDS.items():
        rows = m[top + int(a * h): top + int(b * h)]
        out[name] = float(np.median(rows.sum(1)) / h)
    return out


def measure_video(path, every=4):
    return [proportions(character_mask(img)) for k, img in enumerate(iio.get_reader(path)) if k % every == 0]


def measure_ours(character="protagonist", action="walk", seconds=5.0):
    """our walk, silhouettes from the green-key render (exact)."""
    return [proportions(silhouette(img)) for img in Walker(character, action).keyed(seconds, KEY, step=4)]


def measure_drawing(character="protagonist"):
    a = np.asarray(Image.open(config.CHARACTERS / character / "views" / "side_left.png").convert("RGBA"))
    return proportions(a[..., 3] > 128)


def summary(rows, ref):
    hs = np.array([r["height_px"] for r in rows])
    out = {"first_frame_vs_drawing_pct": {n: round(100 * (rows[0][n] / ref[n] - 1), 1) for n in BANDS},
           "height_sd_pct": round(100 * hs.std() / hs.mean(), 1)}
    for name in BANDS:
        v = np.array([r[name] for r in rows])
        out[name] = {"vs_drawing_median_pct": round(100 * (np.median(v) / ref[name] - 1), 1),
                     "vs_drawing_range_pct": [round(100 * (v.min() / ref[name] - 1), 1),
                                              round(100 * (v.max() / ref[name] - 1), 1)],
                     "frame_to_frame_sd_pct": round(100 * v.std() / v.mean(), 1)}
    out["frames_measured"] = len(rows)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append", required=True)
    ap.add_argument("--character", default="protagonist")
    a = ap.parse_args()
    ref = measure_drawing(a.character)
    res = {"drawing": ref, "ours (walk, rig)": summary(measure_ours(a.character), ref)}
    for v in a.video:
        res[v] = summary(measure_video(v), ref)
    save_text(json.dumps(res, indent=2), config.ROOT / "docs" / "identity_compare.json")
    print(json.dumps(res, indent=2))
