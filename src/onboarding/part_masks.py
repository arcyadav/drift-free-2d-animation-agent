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
OUTLINE_DE = 15        # outline pixel: colour this far (ΔE) from its part's median, and darker
OUTLINE_REACH = 2      # ... and within this many px of a part drawn in front → goes to that part
BLEND_DE = 6           # anti-aliased blend next to a front part: smaller colour difference ...
BLEND_REACH = 2        # ... within this many px
MOVERS = [f"{a}_{s}" for s in "lr" for a in ("upper_arm", "forearm", "hand")]
UNMIX_REACH = 22       # arm pixels this close to a part behind may carry its colour (finger gaps)
UNMIX_MIN = 0.12       # ignore tiny mixes (clean skin / outline)
SOLID_MAX_B = 0.6      # solid copy: pixels less than 60% denim count as arm
UNMIX_CLEAN_GAP = 4    # px from the arm before denim counts as clean
DEFAULT_LINE = (78, 52, 40)   # warm dark brown, if no clean arm outline is found
UNMIX_CONTRAST = 40    # RGB distance arm ↔ part behind below which there is nothing to unmix
HALO_SOLID_A = 200     # dehalo: pixels at least this opaque count as the part's true colour
HALO_DL = 20           # edge pixel this much lighter than the part = background halo
HALO_RAMP = 25         # ... fully transparent once this much lighter again
HALO_BAND = 5          # outer band (px) of a part that may hold background halo (seen: 3-4 px)


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
    tie_xy = scrunchie(hair, lab, (xx < kp["nose"][0]) & (yy > ear_y))
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
    jeans = ndi.binary_fill_holes(ndi.binary_closing(jeans, disk(5))) & body & (cls != ci["skin"])
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

    cleanup(lab_map, fg, rgba[..., 3] > 0, PART_ID.values())
    outlines_to_front(lab_map, lab, [PART_ID[n] for n, _ in PARTS])
    info = {"collar_y": int(collar_y), "crotch_y": round(float(crotch_y), 1),
            "jaw_y": [int(jaw_y.min()), int(jaw_y.max())], "face_right": int(face_right),
            "scrunchie": tie_xy, "neck_top": [round(float(neck_cols.mean()), 1), round(float(np.median(jaw_y)), 1)]}
    return lab_map, info


def scrunchie(hair, lab, where):
    """dark red hair tie = ponytail root / pivot (None if not found)."""
    tie = hair & where & (lab[..., 0] < DARK_L) & (lab[..., 1] > TIE_A) & (lab[..., 1] > lab[..., 2])
    if tie.sum() <= 20:
        return None
    ys, xs = np.where(tie)
    return [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]


def outlines_to_front(lab_map, lab, order_ids):
    """dark outline pixels (ΔE > 15, darker, within 2 px), then the 1-px anti-aliased blend
    (ΔE > 6, lighter or darker: a skin edge over denim blends to tan, not dark)."""
    n = _to_front(lab_map, lab, order_ids, OUTLINE_DE, OUTLINE_REACH, darker=True)
    return n + _to_front(lab_map, lab, order_ids, BLEND_DE, BLEND_REACH, darker=False)


def _to_front(lab_map, lab, order_ids, min_de, reach, darker):
    """An outline between two parts belongs to the part drawn in front (the one it outlines):
    otherwise the arm's outline stays painted on the jeans when the arm swings away."""
    rank = np.zeros(256, np.int32)
    for r, pid in enumerate(order_ids, 1):
        rank[pid] = r
    rmap = rank[lab_map]
    med = np.zeros((256, 3))
    for pid in order_ids:
        m = lab_map == pid
        if m.any():
            med[pid] = np.median(lab[m], 0)
    # outline = far from its part's colour (brown arm outline on blue denim differs in hue,
    # not much in lightness) and darker than it
    de = np.linalg.norm(lab - med[lab_map], axis=-1)
    line = (lab_map > 0) & (de > min_de)
    if darker:
        line &= lab[..., 0] < med[lab_map][..., 0]
    front_r = ndi.maximum_filter(rmap, size=2 * reach + 1)
    move = line & (front_r > rmap)
    id_of_rank = np.zeros(len(order_ids) + 1, np.int32)
    id_of_rank[1:] = order_ids
    lab_map[move] = id_of_rank[front_r[move]]
    return int(move.sum())


def cleanup(lab_map, fg, soft, ids):
    """outlines, leftovers and stray pieces → nearest part; then the soft rim."""
    lab_map[~fg] = 0
    for _ in range(2):
        fill_nearest(lab_map, fg)
        for pid in ids:
            m = lab_map == pid
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
    fill_nearest(lab_map, soft)                                  # soft anti-aliased rim too


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


def validate(lab_map, parts=PARTS, part_id=PART_ID):
    issues, stats = [], {}
    for name, _ in parts:
        m = lab_map == part_id[name]
        px = int(m.sum())
        if px < MIN_PART_PX:
            issues.append(f"{name}: missing / too small ({px}px)")
            stats[name] = {"px": px}
            continue
        cl, n = ndi.label(m)
        # pieces split only by a part drawn in front of it (an arm across the hip) are fine
        names = [p for p, _ in parts]
        front = np.isin(lab_map, [part_id[p] for p in names[names.index(name) + 1:]])
        jl, _ = ndi.label(m | front)
        main = max(m[jl == k].sum() for k in np.unique(jl[m])) / px
        if main < MAIN_PIECE_FRAC:
            issues.append(f"{name}: broken into pieces (main piece {main:.0%})")
        ys, xs = np.where(m)
        stats[name] = {"px": px, "pieces": int(n), "main_piece": round(float(main), 3),
                       "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}
    return issues, stats


def unmix_movers(cuts, rgba, lab_map, parts, part_id, movers, solid=None):
    """The arm is drawn over the jeans, so its edge pixels are a mix of arm and denim.
    Per pixel near a part behind it: p = (1-b)*A + b*D with D = denim just outside and A =
    the arm's own colour (skin or its outline, whichever explains p best). The arm keeps
    (p - b*D)/(1-b) at opacity (1-b); the part behind gets D at that pixel. Over the jeans
    the sum is exactly the original pixel; once the arm swings away its edge is clean —
    no blue fringe, no denim between the fingers.
    solid (dict, optional) receives a full-width version of each moving part: mixed edge
    pixels painted in the arm's own colour instead of made see-through (source for the
    far-arm copies, whose fingers would otherwise look thin)."""
    names = [n for n, _ in parts]
    mov = [m for m in movers if m in part_id and (lab_map == part_id[m]).any()]
    if not mov:
        return 0
    first = min(names.index(m) for m in mov)
    behind_ids = [part_id[n] for n in names[:first]]
    M = np.isin(lab_map, [part_id[m] for m in mov])
    O = np.isin(lab_map, behind_ids)
    if not O.any():
        return 0
    rgb = rgba[..., :3].astype(np.float32)
    lab = rgb2lab(rgba[..., :3] / 255.0)
    dO = ndi.distance_transform_edt(~O)
    R = M & (dO <= UNMIX_REACH)
    # the denim colour comes from CLEAN denim a few px away from the arm: the touching
    # pixels still hold bits of the arm's outline (would paint a brown trace on the jeans)
    O_clean = O & (ndi.distance_transform_edt(~M) > UNMIX_CLEAN_GAP)
    _, (oy, ox) = ndi.distance_transform_edt(~O_clean, return_indices=True)
    name_of = {v: k for k, v in part_id.items()}
    # the arm's own outline colour, measured where the arm is NOT next to the jeans (its
    # edge against the background / tee) — the hand alone is all edge near the denim
    skin_L = np.median(lab[M][:, 0])
    clean_line = M & (lab[..., 0] < skin_L - 15) & (dO > UNMIX_REACH)
    line = np.median(rgb[clean_line], 0) if clean_line.sum() > 20 else np.array(DEFAULT_LINE, np.float32)
    changed = 0
    for m in mov:
        own = lab_map == part_id[m]
        reg = R & own
        if not reg.any():
            continue
        core = ndi.binary_erosion(own, iterations=3)
        src = core if core.sum() > 30 else own
        skin = np.median(rgb[src & (lab[..., 0] > skin_L - 15)], 0) if (src & (lab[..., 0] > skin_L - 15)).sum() > 30             else np.median(rgb[src], 0)
        p = rgb[reg]
        D = rgb[oy[reg], ox[reg]]
        best = None
        for A in (skin, line):
            d = D - A
            bta = np.clip(((p - A) * d).sum(1) / np.maximum((d * d).sum(1), 1.0), 0, 1)
            res = np.linalg.norm(p - (A + bta[:, None] * d), axis=1)
            Ab = np.repeat(A[None], len(p), 0)
            if best is None:
                best = [bta, res, Ab]
            else:
                pick = res < best[1]
                best[0][pick], best[1][pick], best[2][pick] = bta[pick], res[pick], Ab[pick]
        bta, _, Ab = best
        # cap the denim share so the arm's unmixed colour stays in 0..255: then
        # arm over jeans adds up to the original pixel exactly
        with np.errstate(divide="ignore", invalid="ignore"):
            lo = np.where(D > 0, p / D, np.inf)
            hi = np.where(D < 255, (255 - p) / (255 - D), np.inf)
        bta = np.minimum(bta, np.minimum(lo, hi).min(1))
        # only where arm and the part behind clearly differ (skin vs denim); tee sleeve over
        # tee body is the same colour — nothing to unmix, and the mix can't be measured
        contrast = np.linalg.norm(D - Ab, axis=1)
        act = (bta > UNMIX_MIN) & (contrast > UNMIX_CONTRAST)
        if not act.any():
            continue
        ys, xs = np.where(reg)
        ys, xs, bt, P_, D_, A_ = ys[act], xs[act], bta[act], p[act], D[act], Ab[act]
        if solid is not None:
            sc = cuts[m].copy()
            mostly_arm = bt < SOLID_MAX_B
            sc[ys[mostly_arm], xs[mostly_arm], :3] = np.clip(A_[mostly_arm], 0, 255).astype(np.uint8)
            sc[ys[~mostly_arm], xs[~mostly_arm], 3] = 0           # pure denim (finger gaps): not arm
            solid[m] = sc
        a8 = np.round(255 * (1 - bt))                             # 8-bit opacity the arm keeps
        al = (a8 / 255)[:, None]
        # colour from the rounded opacity → arm over D reproduces p exactly
        col = np.where(al > 0, (P_ - (1 - al) * D_) / np.maximum(al, 1e-6), 0)
        cut = cuts[m]
        cut[ys, xs, :3] = np.clip(np.round(col), 0, 255).astype(np.uint8)
        cut[ys, xs, 3] = a8.astype(np.uint8)
        under = np.where(al > 0, D_, P_)                          # fully given away → exact pixel
        owner = lab_map[oy[ys, xs], ox[ys, xs]]                  # the part behind gets the denim share
        for oid in np.unique(owner):
            sel = owner == oid
            nb = cuts[name_of[int(oid)]]
            nb[ys[sel], xs[sel], :3] = np.clip(np.round(under[sel]), 0, 255).astype(np.uint8)
            nb[ys[sel], xs[sel], 3] = 255
        changed += int(act.sum())
    return changed


def silhouette_band(rgba):
    """pixels within HALO_BAND px of the drawing's true background."""
    return ndi.distance_transform_edt(rgba[..., 3] > 0) <= HALO_BAND


def dehalo(cut, silhouette_band):
    """Edge pixels much lighter than the part — cream background left around dark denim /
    hair by the sheet cleaner — fade to transparent (the lighter, the more). Over the cream
    background nothing changes visibly; over another part (near leg over the darker far
    leg) the thin white line disappears and the artist's dark outline becomes the edge.
    Only next to the drawing's true background, never at boundaries between parts (eye
    whites next to the hair are not halo); light parts (tee, sneakers) are never affected."""
    a = cut[..., 3]
    if not (a > 0).any():
        return cut
    L = rgb2lab(cut[..., :3] / 255.0)[..., 0]
    interior = ndi.binary_erosion(a >= HALO_SOLID_A, iterations=HALO_BAND)
    if interior.sum() < 20:
        return cut
    med = np.median(L[interior])
    w = np.clip((L - (med + HALO_DL)) / HALO_RAMP, 0, 1) * (silhouette_band & ~interior)
    cut[..., 3] = (a * (1 - w)).astype(np.uint8)
    return cut


def head_pivots(kp, info, lab_map, tail_id):
    pivots = dict(kp)
    pivots["neck_top"] = np.array(info["neck_top"])              # head turns here, not at the shoulders
    tail = lab_map == tail_id
    if info["scrunchie"]:
        pivots["scrunchie"] = np.array(info["scrunchie"])
    elif tail.any():                     # no tie found: ponytail root = its highest point
        ys, xs = np.where(tail)
        top = ys < ys.min() + 15
        pivots["scrunchie"] = np.array([xs[top].mean(), ys[top].mean()])
    return pivots


def save(char_dir, view, rgba, lab_map, pivots, info, issues, stats, parts, parent, part_id, colours,
         derived=None):
    """derived: optional {part: (source part, fn)} for parts made from another part's final
    cut-out (side view: far limbs = darkened copies of the cleaned near ones)."""
    out = char_dir / "parts" / view
    out.mkdir(parents=True, exist_ok=True)
    band = silhouette_band(rgba)
    save_image(lab_map, out / "labels.png")
    cuts = {}
    for name, _ in parts:
        cut = rgba.copy()
        cut[..., 3] = np.where(lab_map == part_id[name], rgba[..., 3], 0)
        cuts[name] = cut
    solid = {}
    info["unmixed_px"] = unmix_movers(cuts, rgba, lab_map, parts, part_id, MOVERS, solid)
    final = {}
    for name, _ in parts:
        if not (derived and name in derived):
            final[name] = dehalo(cuts[name].copy(), band)
    for name, (src, fn) in (derived or {}).items():
        base = dehalo(solid[src].copy(), band) if src in solid else final[src]
        final[name] = fn(base)
    for name, _ in parts:
        save_image(final[name], out / f"{name}.png")
    meta = {"view": view, "canvas": list(rgba.shape[1::-1]), "info": info, "issues": issues,
            "layer_order": [n for n, _ in parts],
            "parts": {n: {"id": part_id[n], "parent": parent[n], "pivot_joint": j,
                          "pivot": [round(float(v), 1) for v in pivots[j]], **stats.get(n, {})}
                      for n, j in parts}}
    save_text(json.dumps(meta, indent=2), out / "parts.json")

    # overlay: part colours at 55% over the art
    over = rgba[..., :3].astype(float)
    colour = np.zeros_like(over)
    for name, c in colours.items():
        colour[lab_map == part_id[name]] = c
    m = (lab_map > 0)[..., None]
    over = np.where(m, 0.45 * over + 0.55 * colour, 255)
    save_image(over.astype(np.uint8), char_dir / "work" / f"parts_{view}.png")


def run(character):
    char_dir = config.CHARACTERS / character
    rgba, kp, pal = load(char_dir)
    lab_map, info = split_parts(rgba, kp, pal)
    issues, stats = validate(lab_map)
    save(char_dir, "front", rgba, lab_map, head_pivots(kp, info, lab_map, PART_ID["ponytail"]),
         info, issues, stats, PARTS, PARENT, PART_ID, COLOURS)
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
