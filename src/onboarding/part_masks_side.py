"""Rigger step B, side view — cut side_left into parts for the walk (spec §7 R1, §10.3).

She faces the viewer's left, so we see her LEFT side: the left limbs are the near ones.
The right arm and leg are hidden behind the body; they become darkened copies of the
near limbs drawn behind everything ("far" limbs), as planned for the side-view walk.

Same tools as the front view (palette regions × skeleton cuts), plus two side-only cues:
  * sleeve: in profile the sleeve lies on the same-coloured tee; its front edge is a
    faint seam line. One straight line is fitted through the darkest pixel of each row
    (robust to rows where the seam is invisible).
  * legs: one visible leg column → thigh / shin split at the knee, sneaker = foot.

Outputs  characters/<c>/parts/side_left/…   (same layout as the front view)
Usage: python -m src.onboarding.part_masks_side --character protagonist
"""
import argparse
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab
from skimage.morphology import disk

from src import config
from src.onboarding.part_masks import (DARK_L, HAIR_MIN_PX, HAIR_OPEN, EAR_MIN_PX, CROTCH_FRAC,
                                       beyond, classify, cleanup, head_pivots, jaw_line,
                                       outlines_to_front, save, scrunchie, side, validate)

VIEW = "side_left"
NEAR, FAR = "l", "r"
LIMBS = ["upper_arm", "forearm", "hand", "thigh", "shin", "foot"]
# back → front
PARTS = [
    ("foot_r", "r_ankle"), ("shin_r", "r_knee"), ("thigh_r", "r_hip"),
    ("upper_arm_r", "r_shoulder"), ("forearm_r", "r_elbow"), ("hand_r", "r_wrist"),
    ("pelvis", "pelvis"), ("foot_l", "l_ankle"), ("shin_l", "l_knee"), ("thigh_l", "l_hip"),
    ("neck", "neck"), ("torso", "pelvis"), ("head", "neck_top"), ("head_hair", "neck_top"),
    ("upper_arm_l", "l_shoulder"), ("forearm_l", "l_elbow"), ("hand_l", "l_wrist"),
    ("ponytail", "scrunchie"),
]
PARENT = {"torso": None, "pelvis": None, "neck": "torso", "head": "neck",
          "head_hair": "head", "ponytail": "head",
          **{f"{a}_{s}": p for s in "lr" for a, p in
             (("upper_arm", "torso"), ("forearm", f"upper_arm_{s}"), ("hand", f"forearm_{s}"),
              ("thigh", "pelvis"), ("shin", f"thigh_{s}"), ("foot", f"shin_{s}"))}}
PART_ID = {n: i + 1 for i, (n, _) in enumerate(PARTS)}
COLOURS = {
    "pelvis": (40, 60, 140), "foot_l": (240, 240, 120), "shin_l": (120, 160, 240), "thigh_l": (70, 110, 200),
    "neck": (240, 170, 120), "torso": (230, 230, 230), "head": (250, 200, 160),
    "head_hair": (120, 60, 30), "upper_arm_l": (220, 50, 50), "forearm_l": (250, 120, 120),
    "hand_l": (250, 190, 190), "ponytail": (200, 40, 160),
}
FAR_SHADE = 0.78       # far limbs: same art, darker (they are in shadow behind the body)
FAR_SHRINK = 2         # px the far copy is shrunk so it never peeks through the near edge
SEAM_DL = 4            # sleeve seam: L* at least this much below the row's tee median
SEAM_WIN = (-55, 5)    # search window for the seam, x relative to the shoulder joint
SEAM_TOL = 6           # px from the fitted line to count as an inlier
TIP_L = 62             # pixel in the denim this light (denim L* ~40) near the hand = fingertip
TIP_REACH = 14
CHIN_AHEAD = 25        # chin rows: face front at least this far ahead of the neck front


def load(char_dir):
    rgba = np.asarray(Image.open(char_dir / "views" / f"{VIEW}.png").convert("RGBA"))
    joints = json.loads((char_dir / "keypoints.json").read_text())[VIEW]["joints"]
    pal = json.loads((char_dir / "palette.json").read_text())
    kp = {n: np.array(v[:2], float) for n, v in joints.items()}
    # in profile left/right joints overlap: use their mean for both
    for j in ("eye", "ear", "shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
        m = (kp[f"l_{j}"] + kp[f"r_{j}"]) / 2
        kp[f"l_{j}"], kp[f"r_{j}"] = m, m.copy()
    kp["neck"] = kp["l_shoulder"].copy()
    kp["pelvis"] = kp["l_hip"].copy()
    return rgba, kp, pal


def sleeve_line(L, tee, sh, y0, y1):
    """fit x = a*y + b through the faint seam at the sleeve's front edge."""
    ys, xs = [], []
    for y in range(int(y0), int(y1)):
        x_lo, x_hi = int(sh[0] + SEAM_WIN[0]), int(sh[0] + SEAM_WIN[1])
        row = tee[y, x_lo:x_hi]
        if row.sum() < 10:
            continue
        vals = np.where(row, L[y, x_lo:x_hi], 999)
        i = int(np.argmin(vals))
        if np.median(L[y, x_lo:x_hi][row]) - vals[i] >= SEAM_DL:
            ys.append(y)
            xs.append(x_lo + i)
    ys, xs = np.array(ys, float), np.array(xs, float)
    if len(ys) < 10:
        raise RuntimeError("sleeve seam not found")
    keep = np.abs(xs - np.median(xs)) < 3 * SEAM_TOL
    for _ in range(3):                                   # refit without outliers
        a, b = np.polyfit(ys[keep], xs[keep], 1)
        keep = np.abs(xs - (a * ys + b)) < SEAM_TOL
    return a, b, int(keep.sum()), len(ys)


def split_parts(rgba, kp, pal):
    H, W = rgba.shape[:2]
    fg = rgba[..., 3] > 128
    lab = rgb2lab(rgba[..., :3] / 255.0)
    L = lab[..., 0]
    cls, names = classify(lab, fg, pal)
    ci = {n: i for i, n in enumerate(names)}
    yy, xx = np.mgrid[:H, :W]
    P = np.stack([xx, yy], -1).astype(float)
    lab_map = np.zeros((H, W), np.uint8)

    tee = cls == ci["tee"]
    tl, n = ndi.label(tee)
    tee_body = tl == (np.argmax(ndi.sum(tee, tl, range(1, n + 1))) + 1)
    collar_y = np.where(tee_body.any(1))[0].min()

    # ---------- hair: thick dark blobs + their curls ----------
    dark = fg & (cls == -1)
    hl, n = ndi.label(ndi.binary_opening(dark, disk(HAIR_OPEN)))
    sizes = ndi.sum(hl > 0, hl, range(1, n + 1))
    blobs = np.isin(hl, np.where(sizes >= HAIR_MIN_PX)[0] + 1)
    hair = fg & ndi.binary_dilation(blobs, iterations=4) & ~tee_body & ((cls == -1) | ~(cls == ci["skin"]))
    hair |= blobs

    # face = skin near the head, eyes / brows / mouth filled, ears = skin islands in the hair
    head_zone = fg & (yy < collar_y + 40) & ~ndi.binary_dilation(tee_body, iterations=2)
    skin = head_zone & (cls == ci["skin"]) & ~ndi.binary_dilation(blobs, iterations=2)
    sl, n = ndi.label(skin)
    face_core = sl == (np.argmax(ndi.sum(skin, sl, range(1, n + 1))) + 1)
    face = ndi.binary_fill_holes(ndi.binary_closing(face_core, disk(4)))
    face = ndi.binary_dilation(face, disk(3)) & head_zone & ~blobs
    el, n = ndi.label(hair & head_zone & (cls == ci["skin"]))
    ears = np.isin(el, np.where(ndi.sum(el > 0, el, range(1, n + 1)) >= EAR_MIN_PX)[0] + 1)
    face |= ears
    hair &= ~face

    # ponytail: hair beyond a cut through the scrunchie, across the ponytail's direction
    ear_y = kp["l_ear"][1]
    tie = scrunchie(hair, lab, (xx > kp["l_ear"][0]) & (yy > ear_y - 30))
    if tie is None:
        raise RuntimeError("scrunchie not found in side view")
    tie_p = np.array(tie)
    seed = hair & (yy > tie_p[1]) & (xx > tie_p[0] - 10)
    d = np.array([xx[seed].mean(), yy[seed].mean()]) - tie_p
    d /= np.linalg.norm(d)
    tail = hair & (((P - tie_p) @ d) > 0) & (xx > tie_p[0] - 25)
    lab_map[hair & ~tail] = PART_ID["head_hair"]
    lab_map[tail] = PART_ID["ponytail"]

    # head vs neck in profile: a line from the throat corner back to the jaw under the ear.
    # chin bottom = lowest row where the face still sticks out well in front of the neck;
    # throat = deepest concave point of the outline between chin bottom and collar.
    ys = np.arange(int(kp["nose"][1]) + 10, collar_y - 4)
    front = np.array([np.where(face[y])[0].min() if face[y].any() else 9999 for y in ys], float)
    neck_front = front[-1]
    c = int(np.where(front < neck_front - CHIN_AHEAD)[0].max())
    p0, p1 = np.array([front[c], ys[c]]), np.array([front[-1], ys[-1]])
    seg = np.stack([front[c:], ys[c:]], 1)
    t = c + int(np.argmax(side(seg, p0, p1) * np.sign(side(np.array([[p1[0] + 50, p0[1]]]), p0, p1))[0]))
    throat = np.array([front[t], ys[t]], float)
    neck_back = np.where(face[collar_y - 8])[0].max()
    back_cols = np.arange(int((throat[0] + neck_back) / 2), neck_back - 2)
    back_y = float(np.median(jaw_line(L, face, back_cols, kp["nose"][1] + 40, collar_y)))
    back = np.array([neck_back, back_y])
    below = side(P, throat, back) > 0 if back[0] > throat[0] else None
    head = face & ~(below & (xx >= throat[0] - 2))
    lab_map[head & (lab_map == 0)] = PART_ID["head"]
    lab_map[face & ~head & (lab_map == 0)] = PART_ID["neck"]
    neck_top = (throat + back) / 2

    # ---------- body ----------
    body = fg & (lab_map == 0) & ~(head_zone & (yy < collar_y))
    jeans = body & (cls == ci["jeans"])
    jeans = ndi.binary_fill_holes(ndi.binary_closing(jeans, disk(5))) & body & (cls != ci["skin"])
    shoes = body & ~jeans & (yy > kp["l_ankle"][1])
    upper = body & ~jeans & ~shoes

    sh, el_, wr = kp["l_shoulder"], kp["l_elbow"], kp["l_wrist"]
    # arm skin = the skin component reaching the wrist
    # arm skin = all skin below the collar (in profile only the near arm shows skin there;
    # fingertips over the denim are separate islands and must come along)
    arm_skin = upper & (cls == ci["skin"]) & (yy > collar_y + 60)
    hem_y = int(np.where(arm_skin.any(1))[0].min())               # sleeve hem = top of arm skin
    a, b, inl, rows = sleeve_line(L, tee_body, sh, sh[1] + 25, hem_y - 8)
    sleeve = upper & ~arm_skin & (xx > a * yy + b) & (yy < hem_y + 2) & (yy > collar_y - 5)
    arm = sleeve | ndi.binary_dilation(arm_skin, iterations=2) & upper   # skin + its outline
    fore = arm & beyond(P, el_, el_ - sh, wr - el_)
    hand = fore & beyond(P, wr, wr - el_, wr - el_)
    lab_map[arm & ~fore] = PART_ID["upper_arm_l"]
    lab_map[fore & ~hand] = PART_ID["forearm_l"]
    lab_map[hand] = PART_ID["hand_l"]
    lab_map[upper & ~arm] = PART_ID["torso"]

    hip, knee, ankle = kp["l_hip"], kp["l_knee"], kp["l_ankle"]
    crotch_y = hip[1] + CROTCH_FRAC * (knee[1] - hip[1])
    lab_map[jeans & (yy < crotch_y)] = PART_ID["pelvis"]
    leg = jeans & (yy >= crotch_y)
    shin = leg & beyond(P, knee, knee - hip, ankle - knee)
    lab_map[leg & ~shin] = PART_ID["thigh_l"]
    lab_map[shin] = PART_ID["shin_l"]
    lab_map[shoes] = PART_ID["foot_l"]

    near_ids = [PART_ID[n] for n, _ in PARTS if not n.endswith("_r")]
    cleanup(lab_map, fg, rgba[..., 3] > 0, near_ids)
    # fingertips / nail highlights over the denim are light, not denim: they belong to the hand
    hand = lab_map == PART_ID["hand_l"]
    denim = np.isin(lab_map, [PART_ID["pelvis"], PART_ID["thigh_l"]])
    tips = denim & (L > TIP_L) & (ndi.distance_transform_edt(~hand) <= TIP_REACH)
    lab_map[tips] = PART_ID["hand_l"]
    outlines_to_front(lab_map, lab, near_ids)
    info = {"collar_y": int(collar_y), "crotch_y": round(float(crotch_y), 1), "hem_y": hem_y,
            "sleeve_line": [round(float(a), 4), round(float(b), 1)], "sleeve_seam_rows": [inl, rows],
            "throat": throat.round(1).tolist(), "jaw_back": back.round(1).tolist(), "scrunchie": tie,
            "neck_top": neck_top.round(1).tolist()}
    return lab_map, info


def far_copy(near, shade=FAR_SHADE):
    """far limb = near limb, darker, shrunk 2 px: at rest it sits exactly under the near
    limb, and its edge would otherwise show through the near limb's soft outline."""
    cut = near.copy()
    # blur the edge first, then cut further inside: smooth contour, ~FAR_SHRINK px smaller
    a = ndi.gaussian_filter(cut[..., 3].astype(np.float32) / 255, FAR_SHRINK)
    cut[..., 3] = (np.clip((a - 0.78) / 0.12, 0, 1) * 255).astype(np.uint8)
    cut[..., :3] = (cut[..., :3] * shade).astype(np.uint8)
    return cut


def run(character):
    char_dir = config.CHARACTERS / character
    rgba, kp, pal = load(char_dir)
    lab_map, info = split_parts(rgba, kp, pal)
    near_parts = [(n, j) for n, j in PARTS if not n.endswith("_r")]
    issues, stats = validate(lab_map, near_parts, PART_ID)
    for limb in LIMBS:                                    # far limbs share the near limb's stats
        stats[f"{limb}_{FAR}"] = {**stats[f"{limb}_{NEAR}"], "copy_of": f"{limb}_{NEAR}", "shade": FAR_SHADE}
    save(char_dir, VIEW, rgba, lab_map, head_pivots(kp, info, lab_map, PART_ID["ponytail"]),
         info, issues, stats, PARTS, PARENT, PART_ID, COLOURS,
         derived={f"{limb}_{FAR}": (f"{limb}_{NEAR}", far_copy) for limb in LIMBS})
    print("info:", info)
    for n, _ in near_parts:
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
