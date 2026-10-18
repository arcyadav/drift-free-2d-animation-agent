"""View Checker — onboarding step 2 (spec §7, §11).

Runs on the Sheet Cleaner's normalised views (same height / baseline / centre):
  1. builds the character palette from the front view (k-means in Lab) and
     names the clusters: hair, skin, jeans, light (tee + sneakers)
  2. colour check: mean colour of each named region per view vs front (ΔE)
  3. proportion check: where the tee ends / jeans start and end, per view
  4. creates mirrored right-facing views (3q_right, side_right)
Writes palette.json and view_check.json. No ML model; the VLM questions and
DINOv2 similarity are added later as extra votes (spec §11.2).

Usage: python -m src.onboarding.view_checker --character protagonist
"""
import argparse
import json

import numpy as np
from PIL import Image, ImageOps
from scipy.cluster.vq import kmeans2
from skimage.color import rgb2lab, deltaE_ciede2000

from src import config

K = 8
DE_WARN, DE_FAIL = 6.0, 12.0        # ΔE2000: <6 same colour to the eye, >12 clearly different
PROP_WARN, PROP_FAIL = 0.02, 0.04   # landmark height difference as a fraction of body height
COVER_FAIL = 0.6   # garment fills <60% of its front-view share of its body band → recoloured/missing
GARMENTS = ["tee", "jeans"]
MIRRORS = {"3q_left": "3q_right", "side_left": "side_right"}
# landmarks compared across views; back view has no visible face so no face landmarks
LANDMARKS = ["tee_bottom", "jeans_top", "jeans_bottom"]


def load_view(char_dir, view):
    im = np.asarray(Image.open(char_dir / "views" / f"{view}.png").convert("RGBA"))
    return im[..., :3], im[..., 3] > 200


def build_palette(rgb, mask):
    lab = rgb2lab(rgb[mask].astype(np.float64) / 255)
    rng = np.random.default_rng(0)
    sample = lab[rng.choice(len(lab), min(40000, len(lab)), replace=False)]
    centres, _ = kmeans2(sample, K, seed=0, minit="++")
    names = name_clusters(centres, rgb, mask)
    return centres, names


def name_clusters(centres, rgb, mask):
    """Name clusters: colour rules first (light tee, blue jeans), and if the outfit does not fit
    them, purely by where each colour sits on the body (any outfit)."""
    try:
        names = name_by_colour(centres, rgb, mask)
        if plausible(names, centres, rgb, mask):
            return names
    except ValueError:
        pass
    return name_by_position(centres, rgb, mask)


def _shares(centres, rgb, mask):
    lab = rgb2lab(rgb.astype(np.float64) / 255)
    nearest = np.linalg.norm(lab[..., None, :] - centres[None, None], axis=-1).argmin(-1)
    rows = np.where(mask.any(1))[0]
    top, bot = rows.min(), rows.max()

    def share(i, f0, f1):            # fraction of the cluster's pixels in a body-height band
        m = mask & (nearest == i)
        y0, y1 = int(top + f0 * (bot - top)), int(top + f1 * (bot - top))
        return m[y0:y1].sum() / max(m.sum(), 1)
    return share


def plausible(names, centres, rgb, mask):
    """colour-named garments must also sit where that garment belongs."""
    share = _shares(centres, rgb, mask)
    return (share(names["tee"], 0.15, 0.50) > 0.5 and share(names["jeans"], 0.40, 0.98) > 0.5
            and share(names["hair"], 0.0, 0.35) > 0.4)


def name_by_position(centres, rgb, mask):
    """garment = the colour that fills that body band (head / torso / legs / feet); skin = the
    warm colour left over. Works for any outfit colours."""
    share = _shares(centres, rgb, mask)
    L, a, b = centres[:, 0], centres[:, 1], centres[:, 2]
    idx = list(range(len(centres)))
    warm = lambda i: a[i] + b[i] - abs(L[i] - 65)
    names = {}
    skin = max(idx, key=warm)
    others = [i for i in idx if i != skin and warm(i) < warm(skin) - 15]   # not a skin shade
    # hair and top may share a colour (dark hair + dark tee): each picked independently
    names["hair"] = max(others, key=lambda i: share(i, 0.0, 0.10) - 0.01 * L[i])
    names["tee"] = max(others, key=lambda i: share(i, 0.20, 0.45))
    left = [i for i in others if i not in (names["hair"], names["tee"])]
    names["jeans"] = max(left, key=lambda i: share(i, 0.55, 0.88))
    left.remove(names["jeans"])
    if left:
        names["sneakers"] = max(left, key=lambda i: share(i, 0.92, 1.0))
    names["skin"] = skin
    return names


def name_by_colour(centres, rgb, mask):
    """Name clusters by colour *and* where they sit on the body.
    Unnamed clusters are outlines/shadows."""
    L, a, b = centres[:, 0], centres[:, 1], centres[:, 2]
    lab = rgb2lab(rgb.astype(np.float64) / 255)
    nearest = np.linalg.norm(lab[..., None, :] - centres[None, None], axis=-1).argmin(-1)
    H = config.CANVAS_H
    def share(i, y0, y1):  # fraction of the cluster's pixels inside a height band
        m = mask & (nearest == i)
        return m[int(y0 * H):int(y1 * H)].sum() / max(m.sum(), 1)

    names = {"hair": int(np.argmin(L)),
             "jeans": int(np.argmin(b - 0.2 * L))}               # most blue
    light = [i for i in range(len(centres)) if L[i] > 85 and i not in names.values()]
    names["tee"] = max(light, key=lambda i: share(i, 0.2, 0.45))   # torso band
    shoes = [i for i in light if i != names["tee"]]
    if shoes:
        names["sneakers"] = max(shoes, key=lambda i: share(i, 0.88, 1.0))
    rest = [i for i in range(len(centres)) if i not in names.values()]
    names["skin"] = int(max(rest, key=lambda i: a[i] + b[i] - abs(L[i] - 65)))
    return names


def region_masks(rgb, mask, centres, names, max_dist=18.0):
    lab = rgb2lab(rgb.astype(np.float64) / 255)
    d = np.linalg.norm(lab[..., None, :] - centres[None, None], axis=-1)
    nearest = d.argmin(-1)
    ok = mask & (d.min(-1) < max_dist)
    return {n: ok & (nearest == i) for n, i in names.items()}, lab


def landmarks(regions):
    """Row positions (as fraction of canvas height) of outfit boundaries."""
    def rows(m, min_px=15):
        r = np.where(m.sum(1) >= min_px)[0]
        return r
    H = config.CANVAS_H
    jeans = rows(regions["jeans"])
    out = {}
    if len(jeans):
        out["jeans_top"] = float(np.percentile(jeans, 1)) / H
        out["jeans_bottom"] = float(np.percentile(jeans, 99)) / H
        tee = regions["tee"].copy()
        tee[int(out["jeans_top"] * H) + int(0.03 * H):] = False   # tee only above the waist
        tee_rows = rows(tee)
        if len(tee_rows):
            out["tee_bottom"] = float(np.percentile(tee_rows, 99)) / H
    return out


def garment_bands(regions):
    """Rows where each garment is solidly present in the front view."""
    bands = {}
    for g in GARMENTS:
        r = np.where(regions[g].sum(1) >= 15)[0]
        if len(r):
            lo, hi = np.percentile(r, [15, 85]).astype(int)
            bands[g] = (lo, hi)
    return bands


def coverage(regions, mask, bands):
    """Share of body pixels inside each garment's band that have the garment colour.
    Catches recoloured/missing clothes even when the new colour matches another
    cluster (e.g. black jeans get classed as hair)."""
    return {g: float(regions[g][lo:hi].sum() / max(mask[lo:hi].sum(), 1))
            for g, (lo, hi) in bands.items()}


def run(character):
    char_dir = config.CHARACTERS / character
    views = [v for v in config.VIEWS if (char_dir / "views" / f"{v}.png").exists()]
    rgb_f, mask_f = load_view(char_dir, "front")
    centres, names = build_palette(rgb_f, mask_f)

    body_h = config.TARGET_HEIGHT / config.CANVAS_H
    report = {"views": {}, "summary": {}}
    ref = {}
    for v in views:
        rgb, mask = load_view(char_dir, v)
        regions, lab = region_masks(rgb, mask, centres, names)
        colours = {n: lab[m].mean(0) for n, m in regions.items() if m.sum() > 200}
        lm = landmarks(regions)
        if v == "front":
            bands = garment_bands(regions)
            ref = {"colours": colours, "landmarks": lm, "coverage": coverage(regions, mask, bands)}
        entry = {"colour_dE": {}, "landmark_diff": {}, "coverage_vs_front": {}, "issues": []}
        for n, c in colours.items():
            if n in ref["colours"]:
                de = float(deltaE_ciede2000(ref["colours"][n][None], c[None])[0])
                entry["colour_dE"][n] = round(de, 2)
                if de > DE_FAIL:
                    entry["issues"].append(f"FAIL {n} colour differs from front (dE {de:.1f})")
                elif de > DE_WARN:
                    entry["issues"].append(f"warn {n} colour slightly off (dE {de:.1f})")
        for g, c in coverage(regions, mask, bands).items():
            rel = c / max(ref["coverage"][g], 1e-6)
            entry["coverage_vs_front"][g] = round(rel, 3)
            if rel < COVER_FAIL:
                entry["issues"].append(f"FAIL {g} mostly missing or recoloured ({rel:.0%} of front)")
        for k in LANDMARKS:
            if k in lm and k in ref["landmarks"]:
                diff = (lm[k] - ref["landmarks"][k]) / body_h
                entry["landmark_diff"][k] = round(diff, 4)
                if abs(diff) > PROP_FAIL:
                    entry["issues"].append(f"FAIL {k} {diff:+.1%} of body height vs front")
                elif abs(diff) > PROP_WARN:
                    entry["issues"].append(f"warn {k} {diff:+.1%} of body height vs front")
        entry["status"] = ("fail" if any(i.startswith("FAIL") for i in entry["issues"])
                           else "warn" if entry["issues"] else "ok")
        entry["region_px"] = {n: int(m.sum()) for n, m in regions.items()}
        report["views"][v] = entry
        report["summary"][v] = entry["status"]

    # mirrored right-facing views (spec §4)
    for src, dst in MIRRORS.items():
        p = char_dir / "views" / f"{src}.png"
        if p.exists():
            ImageOps.mirror(Image.open(p)).save(char_dir / "views" / f"{dst}.png")
            report["summary"][dst] = f"mirrored from {src}"

    palette = {n: {"lab": [round(float(x), 2) for x in centres[i]],
                   "rgb": [int(round(x)) for x in
                           (np.clip(lab2rgb_one(centres[i]), 0, 1) * 255)]}
               for n, i in names.items()}
    (char_dir / "palette.json").write_text(json.dumps(palette, indent=2))
    (char_dir / "view_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    for v, e in report["views"].items():
        print(f"\n{v}: dE={e['colour_dE']}  landmarks={e['landmark_diff']}  "
              f"coverage={e['coverage_vs_front']}")
        for i in e["issues"]:
            print("   ", i)
    print("\npalette:", {n: p["rgb"] for n, p in palette.items()})
    return report


def lab2rgb_one(lab):
    from skimage.color import lab2rgb
    return lab2rgb(np.asarray(lab, dtype=np.float64)[None, None])[0, 0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="protagonist")
    run(ap.parse_args().character)
