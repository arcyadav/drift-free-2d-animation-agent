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


def world_transforms(meta, angles, root=None, offsets=None):
    """root: optional 3x3 applied to every root part (moves the whole body in the world)."""
    parts = meta["parts"]
    cache = {}
    base = np.eye(3) if root is None else root

    def T(name):
        if name in cache:
            return cache[name]
        p = parts[name]
        R = np.eye(3)
        R[:2] = cv2.getRotationMatrix2D(tuple(p["pivot"]), angles.get(name, 0.0), 1.0)
        if offsets and name in offsets:                        # translate (children follow)
            dx, dy = offsets[name]
            R = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], float) @ R
        cache[name] = (T(p["parent"]) if p["parent"] else base) @ R
        return cache[name]

    return {n: T(n) for n in parts}


def _sprite(img, scale=1.0):
    """part → (premultiplied float crop of its non-transparent box, x0, y0, scale); cached.
    When the camera shrinks the character, the crop is pre-shrunk with area averaging —
    a bilinear warp alone would skip pixels and stair-step every edge."""
    scale = round(float(scale), 3)
    key = (id(img), scale if scale < 0.95 else 1.0)
    hit = _SPRITES.get(key)
    if hit is None or hit[0] is not img:          # ids get reused once an image is freed
        ys, xs = np.where(img[..., 3] > 0)
        if len(ys) == 0:
            sp = None
        else:
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = img[y0:y1, x0:x1].astype(np.float32)
            crop[..., :3] *= crop[..., 3:4] / 255      # premultiplied: no dark fringes when rotated
            s = key[1]
            if s < 1.0:
                h, w = crop.shape[:2]
                nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
                crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
                s = (nw / w + nh / h) / 2
            sp = (crop, int(x0), int(y0), s)
        _SPRITES[key] = hit = (img, sp)           # keeping img alive pins its id
    return hit[1]


_SPRITES = {}


def _merged(meta, imgs, a, b, scale):
    """parent a + child b as one premultiplied sprite (layer order kept), pre-scaled."""
    key = ("merge", id(imgs[a]), id(imgs[b]), round(float(scale), 3))
    hit = _SPRITES.get(key)
    if hit is not None and hit[0] is imgs[a] and hit[1] is imgs[b]:
        return hit[2]
    order = meta["layer_order"]
    first, second = (a, b) if order.index(a) < order.index(b) else (b, a)
    alpha = (imgs[a][..., 3] > 0) | (imgs[b][..., 3] > 0)
    ys, xs = np.where(alpha)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    out = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
    for n in (first, second):
        im = imgs[n][y0:y1, x0:x1].astype(np.float32)
        al = im[..., 3:4] / 255
        out[..., :3] = out[..., :3] * (1 - al) + im[..., :3] * al
        out[..., 3:4] = out[..., 3:4] * (1 - al) + im[..., 3:4]
    s = 1.0
    if scale < 0.95:
        h, w = out.shape[:2]
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)
        s = (nw / w + nh / h) / 2
    sp = (out, int(x0), int(y0), s)
    _SPRITES[key] = (imgs[a], imgs[b], sp)
    return sp


def _draw_bent(out, meta, imgs, a, b, band, Ta, Tb, cam, cam_scale, W, H):
    """Soft joint: one image for parent a + child b; each source pixel moves with a blend
    of both bones, weight rising smoothly across `band` px around the joint — jeans curve
    at the knee instead of folding like two stiff pieces (linear blend skinning)."""
    crop, x0, y0, s = _merged(meta, imgs, a, b, cam_scale)
    h, w = crop.shape[:2]
    P = {n: np.array(meta["parts"][n]["pivot"], float) for n in (a, b)}
    axis = P[b] - P[a]
    L = np.linalg.norm(axis)
    u = axis / L
    Ma, Mb = cam @ Ta, cam @ Tb
    src_corners = np.array([[x0, x0 + w / s, x0, x0 + w / s], [y0, y0, y0 + h / s, y0 + h / s], [1] * 4])
    pts = np.concatenate([(Ma @ src_corners)[:2], (Mb @ src_corners)[:2]], 1)
    bx0, by0 = np.maximum(np.floor(pts.min(1)).astype(int) - 2, 0)
    bx1, by1 = np.minimum(np.ceil(pts.max(1)).astype(int) + 3, [W, H])
    if bx1 <= bx0 or by1 <= by0:
        return
    gy, gx = np.mgrid[by0:by1, bx0:bx1].astype(np.float64)
    Y = np.stack([gx, gy], -1)

    def weight(X):
        t = (X - P[a]) @ u
        return np.clip((t - (L - band / 2)) / band, 0, 1) ** 2 * (3 - 2 * np.clip((t - (L - band / 2)) / band, 0, 1))

    def inv(M, Yp):
        A, c = M[:2, :2], M[:2, 2]
        return (Yp - c) @ np.linalg.inv(A).T

    Xa, Xb = inv(Ma, Y), inv(Mb, Y)
    X = np.where((weight(Xa) < 0.5)[..., None], Xa, Xb)
    for _ in range(4):                                  # fixed point on the blended transform
        wgt = weight(X)[..., None, None]
        A = (1 - wgt) * Ma[:2, :2] + wgt * Mb[:2, :2]
        c = (1 - wgt[..., 0]) * Ma[:2, 2] + wgt[..., 0] * Mb[:2, 2]
        det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
        r = Y - c
        X = np.stack([(A[..., 1, 1] * r[..., 0] - A[..., 0, 1] * r[..., 1]) / det,
                      (-A[..., 1, 0] * r[..., 0] + A[..., 0, 0] * r[..., 1]) / det], -1)
    mx = ((X[..., 0] - x0) * s).astype(np.float32)
    my = ((X[..., 1] - y0) * s).astype(np.float32)
    warped = cv2.remap(crop, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0, 0))
    al = warped[..., 3:4] / 255
    roi = out[by0:by1, bx0:bx1]
    roi *= 1 - al
    roi += warped[..., :3]


def render(meta, imgs, angles, bg=(255, 255, 255), root=None, camera=None, size=None, canvas=None,
           bends=(), offsets=None):
    """warp each part only inside its own bounding box (≈30x faster than full-canvas warps).
    root: body placement in the world; camera: world → screen 3x3 (scale + pan);
    size: (W, H) of the output; canvas: background image to draw onto;
    bends: [(parent, child, band_px)] drawn as one softly bending piece."""
    W, H = size or meta["canvas"]
    Ts = world_transforms(meta, angles, root, offsets)
    cam = np.eye(3) if camera is None else camera
    cam_scale = float(np.sqrt(abs(np.linalg.det(cam[:2, :2]))))
    if canvas is not None:
        out = canvas.astype(np.float32).copy()
    else:
        out = np.empty((H, W, 3), np.float32)
        out[:] = bg
    order = meta["layer_order"]
    bent = {}
    for pa, ch, band in bends:
        later = pa if order.index(pa) > order.index(ch) else ch
        bent[later] = (pa, ch, band)
        bent.setdefault(pa if later == ch else ch, None)          # earlier one: skipped
    for n in order:
        if n in bent:
            if bent[n] is not None:
                pa, ch, band = bent[n]
                _draw_bent(out, meta, imgs, pa, ch, band, Ts[pa], Ts[ch], cam, cam_scale, W, H)
            continue
        sp = _sprite(imgs[n], cam_scale)
        if sp is None:
            continue
        crop, x0, y0, s = sp
        h, w = crop.shape[:2]
        M = cam @ Ts[n] @ np.array([[1 / s, 0, x0], [0, 1 / s, y0], [0, 0, 1]], float)
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
