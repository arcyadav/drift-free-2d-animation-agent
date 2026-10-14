"""Rigid layered posing: each part rotates about its pivot, children follow parents,
parts are stacked in layer order. Used to preview test poses (Rig Tester) before the
mesh rig exists — it shows every gap that opens at a joint.

Angles are degrees, positive = counter-clockwise on screen, relative to the parent.

Usage: python -m src.animation.rigid_pose --character protagonist [--parts-dir filled]
"""
import argparse
import json

import cv2
import numpy as np
from PIL import Image

from src import config
from src.utils.io import save_image

# test poses per view — enough to open every joint the actions use
# (side_left faces the viewer's left: negative = limb swings forward)
TEST_POSES = {"front": {
    "rest": {},
    "wave": {"upper_arm_l": 70, "forearm_l": 85, "hand_l": 10},
    "arm_out": {"upper_arm_l": 70, "upper_arm_r": -70},
    "elbow_bend": {"forearm_l": 90, "forearm_r": -90, "hand_l": 20, "hand_r": -20},
    "head_tilt": {"head": 12, "ponytail": -15, "strand_r": 10},
    "look_down": {"head": -6, "ponytail": 8, "strand_r": -8},
}, "side_left": {
    "rest": {},
    "stride": {"thigh_l": -25, "shin_l": 10, "thigh_r": 20, "shin_r": 25,
               "upper_arm_l": 20, "forearm_l": -10, "upper_arm_r": -20, "forearm_r": -15},
    "passing": {"thigh_l": -30, "shin_l": 60, "foot_l": -10, "thigh_r": 5},
    "arm_swing": {"upper_arm_l": -30, "forearm_l": -25, "upper_arm_r": 25},
    "head_nod": {"head": 10, "ponytail": -12},
    "look_down": {"head": 8, "ponytail": -6},
}}
CROPS = {"front": (180, 60, 860, 1000), "side_left": (300, 100, 760, 1500)}


def load_rig(char_dir, view="front", sub=""):
    d = char_dir / "parts" / view
    meta = json.loads((d / "parts.json").read_text())
    src = d / sub if sub else d
    imgs = {n: np.asarray(Image.open(src / f"{n}.png").convert("RGBA")) for n in meta["layer_order"]}
    return meta, imgs


def world_transforms(meta, angles):
    parts = meta["parts"]
    cache = {}

    def T(name):
        if name in cache:
            return cache[name]
        p = parts[name]
        R = np.eye(3)
        R[:2] = cv2.getRotationMatrix2D(tuple(p["pivot"]), angles.get(name, 0.0), 1.0)
        cache[name] = (T(p["parent"]) if p["parent"] else np.eye(3)) @ R
        return cache[name]

    return {n: T(n) for n in parts}


def _sprite(img):
    """part → (premultiplied float crop of its non-transparent box, x0, y0); cached."""
    key = id(img)
    hit = _SPRITES.get(key)
    if hit is None or hit[0] is not img:          # ids get reused once an image is freed
        ys, xs = np.where(img[..., 3] > 0)
        if len(ys) == 0:
            sp = None
        else:
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = img[y0:y1, x0:x1].astype(np.float32)
            crop[..., :3] *= crop[..., 3:4] / 255      # premultiplied: no dark fringes when rotated
            sp = (crop, int(x0), int(y0))
        _SPRITES[key] = hit = (img, sp)           # keeping img alive pins its id
    return hit[1]


_SPRITES = {}


def render(meta, imgs, angles, bg=(255, 255, 255)):
    """warp each part only inside its own bounding box (≈30x faster than full-canvas warps)."""
    W, H = meta["canvas"]
    Ts = world_transforms(meta, angles)
    out = np.empty((H, W, 3), np.float32)
    out[:] = bg
    for n in meta["layer_order"]:
        sp = _sprite(imgs[n])
        if sp is None:
            continue
        crop, x0, y0 = sp
        h, w = crop.shape[:2]
        M = Ts[n] @ np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], float)
        corners = M[:2] @ np.array([[0, w, 0, w], [0, 0, h, h], [1, 1, 1, 1]], float)
        bx0, by0 = np.floor(corners.min(1)).astype(int) - 1
        bx1, by1 = np.ceil(corners.max(1)).astype(int) + 2
        bx0, by0, bx1, by1 = max(bx0, 0), max(by0, 0), min(bx1, W), min(by1, H)
        if bx1 <= bx0 or by1 <= by0:
            continue
        Mb = np.array([[1, 0, -bx0], [0, 1, -by0], [0, 0, 1]], float) @ M
        warped = cv2.warpAffine(crop, Mb[:2], (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = warped[..., 3:4] / 255
        roi = out[by0:by1, bx0:bx1]
        roi *= 1 - a
        roi += warped[..., :3]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def contact(char_dir, sub="", view="front"):
    meta, imgs = load_rig(char_dir, view, sub)
    tiles = []
    for name, ang in TEST_POSES[view].items():
        im = Image.fromarray(render(meta, imgs, ang)).crop(CROPS[view])
        tiles.append((name, im))
    w, h = tiles[0][1].size
    sheet = Image.new("RGB", (w * 3, h * 2), "white")
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(tiles):
        x, y = (i % 3) * w, (i // 3) * h
        sheet.paste(im, (x, y))
        d.text((x + 10, y + 10), name, fill="black")
    out = char_dir / "work" / f"test_poses_{view}{'_' + sub if sub else ''}.png"
    save_image(sheet, out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="protagonist")
    ap.add_argument("--parts-dir", default="", help="sub-folder of parts/<view> (e.g. filled)")
    ap.add_argument("--view", default="front", choices=list(TEST_POSES))
    a = ap.parse_args()
    print(contact(config.CHARACTERS / a.character, a.parts_dir, a.view))
