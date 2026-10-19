"""Critic (spec §12) — scores a planned clip before it is written, labels the failure type.

Checks (CPU, a few seconds per clip):
  cracks      the poses are re-rendered on a flat key colour: background showing through *inside*
              the silhouette (enclosed holes ≥ CRACK_MIN_BLOB px) = a gap between parts
              (e.g. a sleeve swinging away from the torso, A-pose armpit gaps)
  colours     share of character pixels whose colour does not occur in the character's own
              images (identity: a rig only moves her pixels, so this should stay ~0)
  foot_slide  planted-foot movement per frame while the foot lock holds it (walks)
  plan        did the clip deliver the plan: requested expression available (else neutral),
              open hand in a wave, blinks in clips ≥ 3 s

Fix table (the orchestrator applies it): cracks → seal (thin holes filled from the colours around
them, see animation/seal.py), then re-check; missing expression / hand image → cannot be fixed by
re-rendering: reported, naming the image the user should add.
(Tried first and rejected: moving less. Cracks are rig defects, not motion — his walk still
cracked at half the arm swing, and react cracked *more* with the arms half raised.)

Usage: python -m src.critic.critic --character male --action walk
"""
import argparse
import functools
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab

from src import config
from src.animation.animator import FPS, ON_TWOS, SCALE, Walker, shot_camera
from src.animation.front_clip import NO_BLINK, FrontClip
from src.animation.seal import KEY, holes, silhouette

CRACK_MAX = 0.002            # worst frame's crack area / silhouette area (her walk 0.1%, his 1.2%)
LAB_BIN = 3.0                # colour lookup grid (Lab units)
COLOUR_MAX = 0.02            # share of off-model character pixels allowed
STEP = 2 * ON_TWOS           # check every 2nd distinct pose (cracks last several poses)
FOOT_MAX = 1.0               # px per frame while planted
BLINK_FROM_S = 3.0           # clips at least this long should blink


def cracks(img, zoom, ids=None, drawn=None):
    """background showing through between parts, where the drawing had none"""
    return holes(img, zoom=zoom, ids=ids, drawn=drawn), int(silhouette(img).sum())


@functools.lru_cache(maxsize=8)
def palette(character, view):
    return Palette(character, view)


class Palette:
    """every colour in the character's own images, on a coarse Lab grid (± one bin)"""

    def __init__(self, character, view):
        d = config.CHARACTERS / character
        files = list((d / "parts" / view / "filled").glob("*.png")) + list((d / "faces" / view).rglob("*.png")) \
            + list((d / "hands" / view).glob("*.png"))
        px = []
        for f in files:
            a = np.asarray(Image.open(f).convert("RGBA"))
            px.append(a[a[..., 3] > 200][:, :3])
        lab = rgb2lab(np.concatenate(px)[None] / 255.0)[0]
        self.lo = np.array([0.0, -128.0, -128.0])
        self.grid = np.zeros((int(101 / LAB_BIN) + 2, int(256 / LAB_BIN) + 2, int(256 / LAB_BIN) + 2), bool)
        self.grid[tuple(self.idx(lab).T)] = True
        self.grid = ndi.binary_dilation(self.grid, np.ones((3, 3, 3), bool))

    def idx(self, lab):
        return np.clip(((lab - self.lo) / LAB_BIN).astype(int), 0, np.array(self.grid.shape) - 1)

    def off_model(self, img):
        m = ndi.binary_erosion(silhouette(img), iterations=2)        # skip edges blended with the key
        if not m.any():
            return 0.0
        lab = rgb2lab(img[m][::4][None] / 255.0)[0]                  # every 4th pixel is plenty
        return float(1.0 - self.grid[tuple(self.idx(lab).T)].mean())


def check(value, limit, ok, note=""):
    return {"value": value, "limit": limit, "pass": bool(ok), "note": note}


def critique(plan, character="protagonist", seal=False):
    """score the clip the plan describes (rendered off-screen on the key colour)."""
    side = plan["view"] == "side_left"
    seconds = plan.get("seconds") or 5.0
    checks, stats = {}, {}
    if side:
        clip = Walker(character, plan["action"], speed=plan["speed"], mirror=plan["direction"] == "right",
                      shot=plan["shot"], expression=plan.get("side_expression"), seal=seal)
        frames, slide = clip.poses(seconds)
        pairs = list(clip.images(frames, key=KEY, step=STEP, with_ids=True))
        locked = slide["locked"]
        mx = round(max(locked), 2) if locked else 0.0
        checks["foot_slide"] = check(mx, FOOT_MAX, mx <= FOOT_MAX, "px/frame while planted")
        face = clip.face
        want = face["requested"] if face else plan.get("side_expression")
        got = face["expression"] if face else None
        n_blink = len(clip.blink_frames(len(frames)))
        n_frames = len(frames)
    else:
        clip = FrontClip(character, plan["action"], plan["expression"], plan["speed"], plan.get("seconds"),
                         plan["shot"], seal=seal)
        frames = clip.frames()
        pairs = list(clip.images(frames, key=KEY, step=STEP, stats=stats, with_ids=True))
        want = plan["expression"]
        shown = {s for _, s in stats["shown"]}
        got = want if (want in (None, "neutral") or want in shown or plan["action"] in ("look_down",)) else "neutral"
        n_blink = stats["blinks"]
        n_frames = len(frames)
        if clip.m.get("hand"):
            checks["open_hand"] = check(stats["opened"], 1, stats["opened"] > 0,
                                        "" if stats["opened"] else f"no open-hand image: add hands/front/open_*.png")

    # ---- cracks ----
    zoom = shot_camera(plan["shot"], 0.0, clip.ground_y)[1] / SCALE
    worst, worst_k, area = 0.0, 0, 1
    imgs = [p[0] for p in pairs]
    for k, (img, ids, drawn) in enumerate(pairs):
        c, a = cracks(img, zoom, ids, drawn)
        r = c.sum() / max(a, 1)
        if r > worst:
            worst, worst_k, area = r, k, a
    checks["cracks"] = check(round(worst, 4), CRACK_MAX, worst <= CRACK_MAX,
                             f"worst at frame {worst_k * STEP} ({worst * area:.0f} px)")
    # ---- identity (colours) ----
    pal = palette(character, "side_left" if side else "front")
    off = max(pal.off_model(imgs[k]) for k in range(0, len(imgs), max(len(imgs) // 8, 1)))
    checks["colours"] = check(round(off, 4), COLOUR_MAX, off <= COLOUR_MAX, "share of off-model pixels")
    # ---- plan delivered ----
    if want and want != "neutral":
        checks["expression"] = check(got, want, got == want,
                                     "" if got == want else f"no '{want}' face for this view: add faces/"
                                     f"{'side_left' if side else 'front'}/{want}.png")
    if n_frames / FPS >= BLINK_FROM_S and plan["action"] not in NO_BLINK:
        checks["blinks"] = check(int(n_blink), 1, n_blink > 0, "")

    failed = [k for k, v in checks.items() if not v["pass"]]
    return {"pass": not failed, "failed": failed, "checks": checks, "seal": seal}


FIXES = {"cracks": "seal"}   # failures a re-render can fix; the others need new images from the user


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="protagonist")
    ap.add_argument("--action", default="walk")
    ap.add_argument("--expression", default=None)
    a = ap.parse_args()
    view = "side_left" if a.action.startswith("walk") else "front"
    plan = {"action": a.action, "view": view, "expression": a.expression, "side_expression": a.expression,
            "speed": 1.0, "direction": "left", "shot": "full", "seconds": 4.0 if view == "side_left" else None}
    print(json.dumps(critique(plan, a.character), indent=2))
