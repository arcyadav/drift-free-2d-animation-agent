"""Rigger step A — body keypoints per view (spec §7 R1).

Detector: RTMPose (COCO-17) via rtmlib on CPU. Its output is never trusted blindly:
  * every joint must lie on/near the character's mask
  * left/right are fixed by view (front: her left = viewer's right)
  * rule checks: shoulders/hips level, limb lengths symmetric
Views share height + baseline (Sheet Cleaner), so the front view's joint *heights*
are reused as a prior for views where detectors struggle (back, side).

Usage: python -m src.onboarding.keypoints --character protagonist --views front 3q_left
"""
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from src import config

COCO = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
        "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
        "l_knee", "r_knee", "l_ankle", "r_ankle"]
BONES = [("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
         ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
         ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
         ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"),
         ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
         ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
         ("neck", "head_top"), ("neck", "pelvis")]
MIN_SCORE = 0.3
ON_MASK_PX = 12          # joint may sit this far outside the silhouette
LEVEL_TOL = 0.03         # shoulders / hips level within 3% of body height (front)
SYM_TOL = 0.15           # left vs right limb length within 15%

_model = None


def detector():
    global _model
    if _model is None:
        from rtmlib import Body
        _model = Body(mode="balanced", backend="onnxruntime", device="cpu")
    return _model


def load_view(char_dir, view):
    rgba = Image.open(char_dir / "views" / f"{view}.png").convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    white.alpha_composite(rgba)
    mask = np.asarray(rgba)[..., 3] > 128
    return np.asarray(white.convert("RGB")), mask


def detect(rgb):
    bgr = rgb[..., ::-1].copy()
    kps, scores = detector()(bgr)
    if len(kps) == 0:
        raise RuntimeError("no person detected")
    i = int(np.argmax(scores.mean(1)))          # most confident person
    return {n: (float(kps[i][j][0]), float(kps[i][j][1]), float(scores[i][j]))
            for j, n in enumerate(COCO)}


def add_derived(kp, mask):
    """neck, pelvis, head_top from detected joints + silhouette."""
    def mid(a, b):
        return ((kp[a][0] + kp[b][0]) / 2, (kp[a][1] + kp[b][1]) / 2,
                min(kp[a][2], kp[b][2]))
    kp["neck"] = mid("l_shoulder", "r_shoulder")
    kp["pelvis"] = mid("l_hip", "r_hip")
    ys = np.where(mask.any(1))[0]
    x = int(round(kp["nose"][0]))
    col = np.where(mask[:, max(x - 3, 0):x + 4].any(1))[0]
    kp["head_top"] = (float(x), float(col.min() if len(col) else ys.min()), 1.0)
    return kp


def validate(kp, mask, view):
    issues = []
    near = ndimage.binary_dilation(mask, iterations=ON_MASK_PX)
    H = config.TARGET_HEIGHT
    for n, (x, y, s) in kp.items():
        if s < MIN_SCORE:
            issues.append(f"low confidence {n} ({s:.2f})")
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1]) or not near[yi, xi]:
            issues.append(f"{n} off the body at ({xi},{yi})")
    if view in ("front", "back"):
        for a, b in (("l_shoulder", "r_shoulder"), ("l_hip", "r_hip")):
            if abs(kp[a][1] - kp[b][1]) / H > LEVEL_TOL:
                issues.append(f"{a}/{b} not level")
        # front: her left is on the viewer's right
        want_left_right = view == "front"
        if (kp["l_shoulder"][0] > kp["r_shoulder"][0]) != want_left_right:
            issues.append("left/right swapped")

        def length(a, b):
            return np.hypot(kp[a][0] - kp[b][0], kp[a][1] - kp[b][1])
        for side_pairs in ((("l_shoulder", "l_elbow"), ("r_shoulder", "r_elbow")),
                           (("l_elbow", "l_wrist"), ("r_elbow", "r_wrist")),
                           (("l_hip", "l_knee"), ("r_hip", "r_knee")),
                           (("l_knee", "l_ankle"), ("r_knee", "r_ankle"))):
            la, lb = length(*side_pairs[0]), length(*side_pairs[1])
            if abs(la - lb) / max(la, lb) > SYM_TOL:
                issues.append(f"asymmetric {side_pairs[0][1][2:]}: {la:.0f} vs {lb:.0f}px")
    return issues


def draw(rgb, kp, out_path):
    im = Image.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(im)
    for a, b in BONES:
        if a in kp and b in kp:
            col = (230, 60, 60) if a.startswith("l_") else (60, 120, 230) if a.startswith("r_") else (40, 180, 60)
            d.line([kp[a][:2], kp[b][:2]], fill=col, width=5)
    for n, (x, y, s) in kp.items():
        r = 8
        d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 220, 0) if s >= MIN_SCORE else (255, 0, 0),
                  outline="black")
        d.text((x + 10, y - 6), n, fill="black")
    im.save(out_path)


def run(character, views):
    char_dir = config.CHARACTERS / character
    work = char_dir / "work"
    work.mkdir(exist_ok=True)
    out = {}
    for v in views:
        rgb, mask = load_view(char_dir, v)
        kp = add_derived(detect(rgb), mask)
        issues = validate(kp, mask, v)
        out[v] = {"joints": {n: [round(x, 1), round(y, 1), round(s, 3)] for n, (x, y, s) in kp.items()},
                  "issues": issues}
        draw(rgb, kp, work / f"keypoints_{v}.png")
        print(f"{v}: {len(issues)} issue(s)")
        for i in issues:
            print("   ", i)
    (char_dir / "keypoints.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="protagonist")
    ap.add_argument("--views", nargs="+", default=["front", "3q_left"])
    a = ap.parse_args()
    run(a.character, a.views)
