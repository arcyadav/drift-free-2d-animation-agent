"""Rigger step B — cut the front view into body parts (spec §7 R1).

No SAM: the art is flat-coloured, so palette regions (Sheet Cleaner / View Checker)
crossed with skeleton cuts (keypoints.json) are enough and run on CPU in a second.
  1. head zone   = everything above the tee/arms → hair vs face by darkness + thickness
  2. hair        = thick dark blobs (head hair, ponytail) + thin curls hanging free
                   → head_hair / ponytail / strand_r by nearest blob + position
  3. face        = filled skin region → head (above jaw line) / neck (below)
  4. body        = palette colour per pixel, then cut by lines through the joints:
                   shoulder→hip line separates arms from torso, perpendicular cuts at
                   elbow / wrist, crotch + inseam split pelvis / legs / feet
  5. outlines    = dark line pixels join the nearest part
Each part is checked by rules (present, sane size, one main piece) — the Rig Tester
later catches what rules miss.

Outputs  characters/<c>/parts/front/<part>.png   (full-canvas RGBA cut-outs)
         characters/<c>/parts/front/labels.png   (index map, value = part id)
         characters/<c>/parts/front/parts.json   (ids, layer order, bbox, pivot joint)
         characters/<c>/work/parts_front.png     (colour overlay for checking)

Usage: python -m src.onboarding.part_masks --character protagonist
"""
import argparse
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab
from skimage.morphology import disk

from src import config
from src.utils.io import save_image, save_text

# back → front (draw order); pivot = joint the part rotates about
PARTS = [
    ("foot_r", "r_ankle"), ("foot_l", "l_ankle"), ("leg_r", "r_hip"), ("leg_l", "l_hip"),
    ("pelvis", "pelvis"), ("neck", "neck"), ("torso", "pelvis"),
    ("head", "neck_top"), ("head_hair", "neck_top"), ("strand_r", "l_ear"), ("ponytail", "scrunchie"),
    ("upper_arm_r", "r_shoulder"), ("forearm_r", "r_elbow"), ("hand_r", "r_wrist"),
    ("upper_arm_l", "l_shoulder"), ("forearm_l", "l_elbow"), ("hand_l", "l_wrist"),
]
PARENT = {"torso": None, "pelvis": None, "neck": "torso", "head": "neck",
          "head_hair": "head", "strand_r": "head", "ponytail": "head",
          **{f"{a}_{s}": p for s in "rl" for a, p in
             (("upper_arm", "torso"), ("forearm", f"upper_arm_{s}"), ("hand", f"forearm_{s}"),
              ("leg", "pelvis"), ("foot", f"leg_{s}"))}}
PART_ID = {n: i + 1 for i, (n, _) in enumerate(PARTS)}
COLOURS = {  # overlay colours
    "leg_r": (70, 110, 200), "leg_l": (120, 160, 240), "foot_r": (200, 200, 60), "foot_l": (240, 240, 120),
    "pelvis": (40, 60, 140), "neck": (240, 170, 120), "torso": (230, 230, 230),
    "head": (250, 200, 160), "head_hair": (120, 60, 30), "strand_r": (230, 120, 30),
    "ponytail": (200, 40, 160),
    "upper_arm_r": (40, 180, 80), "forearm_r": (100, 220, 120), "hand_r": (180, 250, 180),
    "upper_arm_l": (220, 50, 50), "forearm_l": (250, 120, 120), "hand_l": (250, 190, 190),
}

DARK_L = 35            # Lab L* below this = hair or outline
HAIR_OPEN = 3          # opening radius: hair blobs survive, outlines / brows do not
HAIR_MIN_PX = 1500     # a hair blob is at least this big (pupils ~300)
FAR_FROM_BLOB = 8      # hair pixels farther than this from every blob hang free
EAR_MIN_PX = 40        # skin island inside the hair this big = ear
TIE_A = 12             # scrunchie: dark but red (a* ≈ 17; hair a* ≈ 3)
CROTCH_FRAC = 0.10     # crotch below hips, as fraction of hip→knee
MIN_PART_PX = 150
MIN_GAP_PX = 2         # background run this wide = arm separated from body
JAW_L = 50             # chin outline is mid-dark (L* ≈ 32), neck shadow ≈ 60
STRAY_FRAC = 0.10      # small pieces (< this share of a part) ...
STRAY_GAP = 6          # ... farther than this from its main piece go to their neighbour
MAIN_PIECE_FRAC = 0.85  # largest connected piece should hold ≥85% of a part


def load(char_dir):
    rgba = np.asarray(Image.open(char_dir / "views" / "front.png").convert("RGBA"))
    joints = json.loads((char_dir / "keypoints.json").read_text())["front"]["joints"]
    pal = json.loads((char_dir / "palette.json").read_text())
    kp = {n: np.array(v[:2], float) for n, v in joints.items()}
    return rgba, kp, pal


def classify(lab, fg, pal):
    """nearest palette colour (tee / jeans / sneakers / skin); dark pixels → -1."""
    names = ["tee", "jeans", "sneakers", "skin"]
    ref = np.array([pal[n]["lab"] for n in names])
    d = ((lab[..., None, :] - ref[None, None]) ** 2).sum(-1)
    cls = np.argmin(d, -1)
    cls[lab[..., 0] < DARK_L] = -1
    cls[~fg] = -2
    return cls, names


def side(p, a, b):
    """signed side of points p (…,2 as x,y) relative to line a→b."""
    return (b[0] - a[0]) * (p[..., 1] - a[1]) - (b[1] - a[1]) * (p[..., 0] - a[0])


def beyond(p, joint, d_in, d_out):
    """True where p lies past the cut plane at `joint` (normal = mean bone direction)."""
    n = d_in / np.linalg.norm(d_in) + d_out / np.linalg.norm(d_out)
    return ((p - joint) @ n) > 0


def split_parts(rgba, kp, pal):
    H, W = rgba.shape[:2]
    fg = rgba[..., 3] > 128
    lab = rgb2lab(rgba[..., :3] / 255.0)
    cls, names = classify(lab, fg, pal)
    ci = {n: i for i, n in enumerate(names)}
    yy, xx = np.mgrid[:H, :W]
    P = np.stack([xx, yy], -1).astype(float)
    lab_map = np.zeros((H, W), np.uint8)

    # ---------- body zone: the tee body defines where the head zone ends ----------
    tee = cls == ci["tee"]
    tl, n = ndi.label(tee)
    sizes = ndi.sum(tee, tl, range(1, n + 1))
    tee_body = tl == (np.argmax(sizes) + 1)
    collar_y = np.where(tee_body.any(1))[0].min()               # top of the shoulders

    # ---------- hair ----------
    dark = fg & (cls == -1)
    hl, n = ndi.label(ndi.binary_opening(dark, disk(HAIR_OPEN)))
    sizes = ndi.sum(hl > 0, hl, range(1, n + 1))
    blobs = [i + 1 for i, s in enumerate(sizes) if s >= HAIR_MIN_PX]
    blobs.sort(key=lambda i: -sizes[i - 1])
    head_blob, tail_blob = (hl == blobs[0]), (hl == blobs[1])   # biggest = head hair
    if np.where(tail_blob)[1].mean() > kp["nose"][0]:
        raise RuntimeError("ponytail blob not on her right (viewer's left)")

    # face = skin of the head zone, holes (eyes, brows, mouth) filled, outline included
    head_zone = fg & (yy < collar_y + 40) & ~ndi.binary_dilation(tee_body, iterations=2)
    skin = head_zone & (cls != -1) & ~ndi.binary_dilation(head_blob | tail_blob, iterations=2)
    sl, n = ndi.label(skin)
    sizes = ndi.sum(skin, sl, range(1, n + 1))
    face_core = sl == (np.argmax(sizes) + 1)                    # face + ears + neck
    face = ndi.binary_fill_holes(ndi.binary_closing(face_core, disk(4)))
    face = ndi.binary_dilation(face, disk(3)) & head_zone & ~head_blob & ~tail_blob

    hair = head_zone & ~face & (yy < collar_y + 40)
    hair &= ~(cls == ci["tee"])                                  # stray tee bits at shoulders
    # ears: skin islands cut off from the face by their own outline → head, not hair
    el, n = ndi.label(hair & (cls == ci["skin"]))
    sizes = ndi.sum(el > 0, el, range(1, n + 1))
    ears = np.isin(el, np.where(sizes >= EAR_MIN_PX)[0] + 1)
    face |= ears
    hair &= ~ears
    d_head = ndi.distance_transform_edt(~head_blob)
    d_tail = ndi.distance_transform_edt(~tail_blob)
    to_tail = d_tail < d_head
    free = np.minimum(d_head, d_tail) > FAR_FROM_BLOB
    face_right = np.where(face.any(0))[0].max()
    ear_y = (kp["l_ear"][1] + kp["r_ear"][1]) / 2
    strand_r = hair & free & (xx > kp["nose"][0]) & (yy > ear_y)
    # scrunchie (dark red tie) = ponytail root / pivot
    tie = hair & (lab[..., 0] < DARK_L) & (lab[..., 1] > TIE_A) & (lab[..., 1] > lab[..., 2])         & (xx < kp["nose"][0]) & (yy > ear_y)
    tie_xy = [round(float(xx[tie].mean()), 1), round(float(yy[tie].mean()), 1)] if tie.sum() > 20 else None
    lab_map[hair & to_tail] = PART_ID["ponytail"]
    lab_map[hair & ~to_tail] = PART_ID["head_hair"]
    lab_map[strand_r] = PART_ID["strand_r"]

    # head vs neck: the jaw line is a mid-dark outline crossing the neck column
    neck_row = face[collar_y - 8]
    neck_cols = np.where(ndi.binary_erosion(neck_row, iterations=3))[0]
    jaw_y = jaw_line(lab[..., 0], face, neck_cols, kp["nose"][1] + 40, collar_y)
    jaw_row = np.full(W, jaw_y.max(), float)                     # below the chin = neck
    jaw_row[neck_cols] = jaw_y
    head = face & (yy < jaw_row[None, :])
    lab_map[head] = PART_ID["head"]
    lab_map[face & ~head] = PART_ID["neck"]

    # ---------- body ----------
    body = fg & (lab_map == 0) & ~(head_zone & (yy < collar_y))
    # jeans zone: stitching, pockets and fly lines inside the denim count as jeans
    jeans = body & (cls == ci["jeans"])
    jeans = ndi.binary_fill_holes(ndi.binary_closing(jeans, disk(5))) & body
    ankle_y = min(kp["l_ankle"][1], kp["r_ankle"][1])
    shoes = body & ~jeans & (yy > ankle_y)                       # white sneakers ≈ tee colour
    upper = body & ~jeans & ~shoes                               # tee, skin, outlines

    for s in ("r", "l"):
        sh, el, wr, hip = (kp[f"{s}_{j}"] for j in ("shoulder", "elbow", "wrist", "hip"))
        out_sign = np.sign(side(np.array([kp["neck"]]), sh, hip))[0] * -1
        arm = upper & (np.sign(side(P, sh, hip)) == out_sign)
        step = 1 if el[0] > sh[0] else -1                        # outward = towards the elbow
        arm &= ~body_before_gap(fg, sh, hip, step, int(sh[1]), int(wr[1]))
        fore = arm & beyond(P, el, el - sh, wr - el)
        hand = fore & beyond(P, wr, wr - el, wr - el)
        lab_map[arm & ~fore] = PART_ID[f"upper_arm_{s}"]
        lab_map[fore & ~hand] = PART_ID[f"forearm_{s}"]
        lab_map[hand] = PART_ID[f"hand_{s}"]
        upper &= ~arm
    lab_map[upper & (yy < kp["pelvis"][1] + 40)] = PART_ID["torso"]

    crotch_y = kp["pelvis"][1] + CROTCH_FRAC * (kp["l_knee"][1] - kp["l_hip"][1])
    feet_mid = (kp["l_ankle"] + kp["r_ankle"]) / 2
    crotch = np.array([kp["pelvis"][0], crotch_y])
    leg_side = np.sign(side(P, crotch, feet_mid))
    l_sign = np.sign(side(np.array([kp["l_knee"]]), crotch, feet_mid))[0]
    lab_map[jeans & (yy < crotch_y)] = PART_ID["pelvis"]
    legs = jeans & (yy >= crotch_y)
    lab_map[legs & (leg_side == l_sign)] = PART_ID["leg_l"]
    lab_map[legs & (leg_side != l_sign)] = PART_ID["leg_r"]
    lab_map[shoes & (leg_side == l_sign)] = PART_ID["foot_l"]
    lab_map[shoes & (leg_side != l_sign)] = PART_ID["foot_r"]

    # ---------- outlines, leftovers and stray pieces → nearest part ----------
    lab_map[~fg] = 0
    for _ in range(2):
        fill_nearest(lab_map, fg)
        for name, _ in PARTS:
            m = lab_map == PART_ID[name]
            cl, n = ndi.label(m)
            if n < 2:
                continue
            sizes = ndi.sum(m, cl, range(1, n + 1))
            near_main = ndi.binary_dilation(cl == np.argmax(sizes) + 1, iterations=STRAY_GAP)
            for i in np.where(sizes < STRAY_FRAC * sizes.sum())[0] + 1:
                piece = cl == i
                if not (piece & near_main).any():                # far from the part → stray
                    lab_map[piece] = 0
    fill_nearest(lab_map, fg)
    fill_nearest(lab_map, rgba[..., 3] > 0)                      # soft anti-aliased rim too
    info = {"collar_y": int(collar_y), "crotch_y": round(float(crotch_y), 1),
            "jaw_y": [int(jaw_y.min()), int(jaw_y.max())], "face_right": int(face_right),
            "scrunchie": tie_xy, "neck_top": [round(float(neck_cols.mean()), 1), round(float(np.median(jaw_y)), 1)]}
    return lab_map, info


def body_before_gap(fg, a, b, step, y0, y1):
    """Below the armpit the tee body continues outside the shoulder→hip line.
    Per row: walk outward from the line; if a background gap separates what follows
    (the arm), the pixels before the gap are body, not arm."""
    body = np.zeros_like(fg)
    W = fg.shape[1]
    for y in range(max(y0, 0), min(y1, fg.shape[0])):
        x_line = a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1])
        xs = np.arange(int(round(x_line)), -1 if step < 0 else W, step)
        row = fg[y, xs]
        gap = np.where(~row)[0]
        if len(gap) == 0:
            continue
        g0 = gap[0]
        after = np.where(row[g0:])[0]
        if len(after) and g0 > 0 and after[0] >= MIN_GAP_PX:
            body[y, xs[:g0]] = True
    return body


def fill_nearest(lab_map, fg):
    unl = fg & (lab_map == 0)
    if unl.any():
        _, (iy, ix) = ndi.distance_transform_edt(lab_map == 0, return_indices=True)
        lab_map[unl] = lab_map[iy[unl], ix[unl]]


def jaw_line(L, face, cols, y0, collar_y):
    """per neck column: first outline (L* < JAW_L) below the mouth = chin / jaw line."""
    ys = np.arange(int(y0), int(collar_y))
    out = []
    for x in cols:
        hits = ys[(L[ys, x] < JAW_L) & face[ys, x]]
        out.append(hits.min() + 3 if len(hits) else np.nan)      # +3: line stays with the head
    out = np.array(out, float)
    out[np.isnan(out)] = np.nanmedian(out)
    return ndi.median_filter(out, 9)


def validate(lab_map):
    issues, stats = [], {}
    for name, _ in PARTS:
        m = lab_map == PART_ID[name]
        px = int(m.sum())
        if px < MIN_PART_PX:
            issues.append(f"{name}: missing / too small ({px}px)")
            stats[name] = {"px": px}
            continue
        cl, n = ndi.label(m)
        main = ndi.sum(m, cl, range(1, n + 1)).max() / px
        if main < MAIN_PIECE_FRAC:
            issues.append(f"{name}: broken into pieces (main piece {main:.0%})")
        ys, xs = np.where(m)
        stats[name] = {"px": px, "pieces": int(n), "main_piece": round(float(main), 3),
                       "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}
    return issues, stats


def save(char_dir, rgba, lab_map, kp, info, issues, stats):
    out = char_dir / "parts" / "front"
    out.mkdir(parents=True, exist_ok=True)
    save_image(lab_map, out / "labels.png")
    for name, _ in PARTS:
        cut = rgba.copy()
        cut[..., 3] = np.where(lab_map == PART_ID[name], rgba[..., 3], 0)
        save_image(cut, out / f"{name}.png")
    pivots = dict(kp)
    pivots["neck_top"] = np.array(info["neck_top"])              # head turns here, not at the shoulders
    tail = lab_map == PART_ID["ponytail"]
    if info["scrunchie"]:
        pivots["scrunchie"] = np.array(info["scrunchie"])
    elif tail.any():                     # no tie found: ponytail root = its highest point
        ys, xs = np.where(tail)
        top = ys < ys.min() + 15
        pivots["scrunchie"] = np.array([xs[top].mean(), ys[top].mean()])
    meta = {"view": "front", "canvas": list(rgba.shape[1::-1]), "info": info, "issues": issues,
            "layer_order": [n for n, _ in PARTS],
            "parts": {n: {"id": PART_ID[n], "parent": PARENT[n], "pivot_joint": j,
                          "pivot": [round(float(v), 1) for v in pivots[j]], **stats.get(n, {})}
                      for n, j in PARTS}}
    save_text(json.dumps(meta, indent=2), out / "parts.json")

    # overlay: part colours at 55% over the art
    over = rgba[..., :3].astype(float)
    colour = np.zeros_like(over)
    for name, c in COLOURS.items():
        colour[lab_map == PART_ID[name]] = c
    m = (lab_map > 0)[..., None]
    over = np.where(m, 0.45 * over + 0.55 * colour, 255)
    save_image(over.astype(np.uint8), char_dir / "work" / "parts_front.png")


def run(character):
    char_dir = config.CHARACTERS / character
    rgba, kp, pal = load(char_dir)
    lab_map, info = split_parts(rgba, kp, pal)
    issues, stats = validate(lab_map)
    save(char_dir, rgba, lab_map, kp, info, issues, stats)
    print("info:", info)
    for n, _ in PARTS:
        s = stats[n]
        print(f"  {n:12s} {s['px']:7d}px  pieces={s.get('pieces', 0)}  main={s.get('main_piece', 0):.0%}")
    print(f"{len(issues)} issue(s)")
    for i in issues:
        print("   ", i)
    return issues


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="protagonist")
    run(ap.parse_args().character)
