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

# test poses — enough to open every joint the front-view actions use
TEST_POSES = {
    "rest": {},
    "wave": {"upper_arm_l": 70, "forearm_l": 85, "hand_l": 10},
    "arm_out": {"upper_arm_l": 70, "upper_arm_r": -70},
    "elbow_bend": {"forearm_l": 90, "forearm_r": -90, "hand_l": 20, "hand_r": -20},
    "head_tilt": {"head": 12, "ponytail": -15, "strand_r": 10},
    "look_down": {"head": -6, "ponytail": 8, "strand_r": -8},
}


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


def render(meta, imgs, angles, bg=(255, 255, 255)):
    W, H = meta["canvas"]
    Ts = world_transforms(meta, angles)
    out = np.zeros((H, W, 4), np.float32)
    out[..., :3] = bg
    out[..., 3] = 1
    for n in meta["layer_order"]:
        # premultiplied alpha: interpolating straight RGBA drags edge colours toward black
        im = imgs[n].astype(np.float32)
        im[..., :3] *= im[..., 3:4] / 255
        warped = cv2.warpAffine(im, Ts[n][:2], (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        a = warped[..., 3:4] / 255
        out[..., :3] = out[..., :3] * (1 - a) + warped[..., :3]
    return np.clip(out[..., :3] + 0.5, 0, 255).astype(np.uint8)


def contact(char_dir, sub="", crop=(180, 60, 860, 1000)):
    meta, imgs = load_rig(char_dir, "front", sub)
    tiles = []
    for name, ang in TEST_POSES.items():
        im = Image.fromarray(render(meta, imgs, ang)).crop(crop)
        tiles.append((name, im))
    w, h = tiles[0][1].size
    sheet = Image.new("RGB", (w * 3, h * 2), "white")
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(tiles):
        x, y = (i % 3) * w, (i // 3) * h
        sheet.paste(im, (x, y))
        d.text((x + 10, y + 10), name, fill="black")
    out = char_dir / "work" / f"test_poses{'_' + sub if sub else ''}.png"
    save_image(sheet, out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="protagonist")
    ap.add_argument("--parts-dir", default="", help="sub-folder of parts/front (e.g. filled)")
    a = ap.parse_args()
    print(contact(config.CHARACTERS / a.character, a.parts_dir))
