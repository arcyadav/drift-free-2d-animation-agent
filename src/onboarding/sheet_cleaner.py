"""Sheet Cleaner — onboarding step 1 (spec §7, §11).

For every raw image of a character:
  1. estimate the flat background colour from the image border
  2. remove background by flood-filling inward from the border; the drawn
     outline stops the fill, so the cream tee/sneakers (only ~10 levels away
     from the cream background) are kept
  3. also remove enclosed background pockets (e.g. between ponytail and neck)
  4. drop specks, watermarks and text labels (small components away from the body)
  5. crop + normalise scale so every view has the same height, baseline and centre
  6. face images reuse the transform of their view, so they stay pixel-aligned

No ML model needed; runs on CPU in seconds.

Usage: python -m src.onboarding.sheet_cleaner --character protagonist
"""
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from src import config

BG_TOL = 9           # max channel diff from bg colour that the border fill may cross
POCKET_TOL = 5.5     # max *mean* diff for an enclosed region to count as trapped background
MIN_POCKET_PX = 30   # enclosed regions smaller than this are left alone
# note: eye whites sit ~7 from the bg colour, so POCKET_TOL must stay below that
FRINGE_HI = 30       # edge pixels below this diff (outside the outline) are bg/fg blends
FRINGE_BAND = 4      # how far in from the mask edge we look for blended pixels
MIN_SOLID_DIFF = 40  # only unmix next to colours clearly unlike bg (not tee/sneakers)
KEEP_FRAC = 0.002    # keep extra components >= 0.2% of the body's area if near the body
NEAR_PX = 25         # ...and within this distance of the body
LOW_RES_SCALE = 1.25 # warn when a source must be upscaled more than this


# ---------- background removal ----------

def estimate_bg(rgb):
    b = 8
    border = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                             rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    return np.median(border, axis=0)


def foreground_mask(rgb):
    """Return (fg_mask bool HxW, dist HxW, bg rgb, info dict)."""
    bg = estimate_bg(rgb)
    smooth = ndimage.uniform_filter(rgb.astype(np.float32), size=(3, 3, 1))  # kill JPEG grain
    dist = np.abs(smooth - bg).max(axis=2)

    near_bg = dist <= BG_TOL
    labels, n = ndimage.label(near_bg)
    edge_labels = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    bg_mask = np.isin(labels, edge_labels[edge_labels > 0])

    # enclosed pockets (e.g. gap between ponytail and neck): judged by the *mean*
    # distance of the whole region, so a sneaker highlight (~7) survives while a
    # trapped patch of background (~4-5) is removed
    pocket_ids = [i for i in range(1, n + 1) if i not in set(edge_labels)]
    pocket_mask = np.zeros_like(near_bg)
    if pocket_ids:
        sizes = ndimage.sum(near_bg, labels, pocket_ids)
        means = ndimage.mean(dist, labels, pocket_ids)
        chosen = [i for i, s, m in zip(pocket_ids, sizes, means)
                  if s >= MIN_POCKET_PX and m <= POCKET_TOL]
        pocket_mask = np.isin(labels, chosen)
        bg_mask |= pocket_mask
    pockets = int(ndimage.label(pocket_mask)[1])

    fg = ~bg_mask
    fg = ndimage.binary_opening(fg, iterations=1)

    # keep the body + attached-ish bits; drop specks, watermarks, labels
    flabels, fn = ndimage.label(fg)
    if fn == 0:
        raise ValueError("no foreground found")
    sizes = ndimage.sum(fg, flabels, range(1, fn + 1))
    main = int(np.argmax(sizes)) + 1
    near_main = ndimage.binary_dilation(flabels == main, iterations=NEAR_PX)
    keep = [main]
    for i, s in enumerate(sizes, start=1):
        if i != main and s >= KEEP_FRAC * sizes[main - 1] and near_main[flabels == i].any():
            keep.append(i)
    fg = np.isin(flabels, keep)
    fg = ndimage.binary_fill_holes(fg) & ~pocket_mask  # fill pin-holes, keep real pockets open
    removed = fn - len(keep)

    info = {"bg_rgb": [round(float(c), 1) for c in bg], "pockets_removed": pockets,
            "specks_removed": int(removed)}
    return fg, dist, bg, info


def unmix_edges(rgb, fg, dist, bg):
    """Remove the cream halo: treat edge pixels as blends of bg and the nearest
    solid character colour, recover alpha, and strip the bg tint from colour.

    Only pixels reachable from the outside without crossing the drawn outline
    (dist > FRINGE_HI) count as fringe, so light areas inside the outline
    (cream tee, white sneakers) are never made transparent.
    """
    band = fg & ~ndimage.binary_erosion(fg, iterations=FRINGE_BAND)
    soft = band & (dist < FRINGE_HI)
    lab, _ = ndimage.label(soft | ~fg)
    outside_ids = np.unique(lab[~fg])
    fringe = soft & np.isin(lab, outside_ids[outside_ids > 0])

    solid = fg & ~fringe
    _, (iy, ix) = ndimage.distance_transform_edt(~solid, return_indices=True)
    img = rgb.astype(np.float32)
    F = img[iy, ix]                       # nearest solid colour
    fb, ib = F - bg, img - bg
    # next to cream tee / white sneakers there is no outline to unmix against
    fringe &= np.abs(fb).max(axis=2) >= MIN_SOLID_DIFF
    a = (ib * fb).sum(2) / np.maximum((fb * fb).sum(2), 1.0)  # projection onto bg→F
    a = np.clip(a, 0, 1)

    alpha = fg.astype(np.float32)
    alpha[fringe] = a[fringe]
    alpha[alpha < 0.08] = 0
    out = img.copy()
    m = fringe & (alpha > 0)
    out[m] = np.clip((img[m] - (1 - alpha[m, None]) * bg) / alpha[m, None], 0, 255)
    return out.astype(np.uint8), (alpha * 255).astype(np.uint8), int(fringe.sum())


# ---------- normalisation ----------

def compute_transform(fg):
    ys, xs = np.where(fg)
    top, bottom = ys.min(), ys.max()
    height = bottom - top + 1
    legs = fg[top + int(0.6 * height):bottom + 1]
    leg_x = np.where(legs)[1].mean()
    s = config.TARGET_HEIGHT / height
    return {"scale": s, "anchor_x": float(leg_x), "anchor_y": float(bottom),
            "src_bbox": [int(xs.min()), int(top), int(xs.max()), int(bottom)]}


def apply_transform(rgba, t):
    """Scale about (anchor_x, anchor_y) and place it at (CENTER_X, FEET_Y)."""
    s = t["scale"]
    w, h = rgba.size
    scaled = rgba.resize((round(w * s), round(h * s)), Image.LANCZOS)
    canvas = Image.new("RGBA", (config.CANVAS_W, config.CANVAS_H), (0, 0, 0, 0))
    ox = round(config.CENTER_X - t["anchor_x"] * s)
    oy = round(config.FEET_Y - t["anchor_y"] * s)
    canvas.paste(scaled, (ox, oy), scaled)  # paste clips negative offsets
    return canvas


# ---------- per-image ----------

def clean_image(path, transform=None, ref_size=None):
    img = Image.open(path).convert("RGB")
    if ref_size and img.size != ref_size:  # face edits saved at another resolution
        img = img.resize(ref_size, Image.LANCZOS)
    rgb = np.asarray(img)
    fg, dist, bg, info = foreground_mask(rgb)
    rgb_clean, alpha, info["fringe_px_unmixed"] = unmix_edges(rgb, fg, dist, bg)
    rgba = Image.fromarray(np.dstack([rgb_clean, alpha]), "RGBA")
    t = transform or compute_transform(fg)
    info.update({"src": str(path.relative_to(config.ROOT)).replace("\\", "/"),
                 "src_size": list(img.size), "scale": round(t["scale"], 4)})
    if t["scale"] > LOW_RES_SCALE:
        info["warning"] = f"low-res source: upscaled x{t['scale']:.2f}"
    return apply_transform(rgba, t), t, info, img.size


def contact_sheet(items, out_path, cols=4, thumb_h=600):
    """Each cleaned image on a checkerboard with baseline/height guides."""
    tw = round(thumb_h * config.CANVAS_W / config.CANVAS_H)
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (tw * cols, (thumb_h + 24) * rows), "white")
    d = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(items):
        x, y = (i % cols) * tw, (i // cols) * (thumb_h + 24)
        checker = Image.new("RGB", (config.CANVAS_W, config.CANVAS_H), (255, 0, 255))
        cd = ImageDraw.Draw(checker)
        for cy in range(0, config.CANVAS_H, 64):
            for cx in range(0, config.CANVAS_W, 64):
                if (cx // 64 + cy // 64) % 2:
                    cd.rectangle([cx, cy, cx + 63, cy + 63], fill=(200, 0, 200))
        checker.paste(im, (0, 0), im)
        top = config.FEET_Y - config.TARGET_HEIGHT
        for gy in (top, config.FEET_Y):
            cd.line([(0, gy), (config.CANVAS_W, gy)], fill=(0, 255, 0), width=3)
        cd.line([(config.CENTER_X, 0), (config.CENTER_X, config.CANVAS_H)], fill=(0, 255, 0), width=2)
        sheet.paste(checker.resize((tw, thumb_h)), (x, y))
        d.text((x + 6, y + thumb_h + 5), name, fill="black")
    sheet.save(out_path)


def run(character):
    raw = config.ASSETS_RAW
    out = config.CHARACTERS / character
    work = out / "work"
    for sub in ("views", "apose", "work/faces_clean"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    report, sheet_items, transforms = {"views": {}, "apose": {}, "faces": {}}, [], {}

    for view in config.VIEWS:
        p = raw / "turnaround" / f"{view}.png"
        if not p.exists():
            report["views"][view] = {"missing": True}
            continue
        im, t, info, size = clean_image(p)
        transforms[view] = (t, size)
        im.save(out / "views" / f"{view}.png")
        report["views"][view] = info
        sheet_items.append((f"view/{view}", im))

    for p in sorted((raw / "apose").glob("*.png")):
        im, _, info, _ = clean_image(p)
        im.save(out / "apose" / p.name)
        report["apose"][p.stem] = info
        sheet_items.append((f"apose/{p.stem}", im))

    for view_dir in sorted((raw / "faces").iterdir()):
        if not view_dir.is_dir() or view_dir.name not in transforms:
            continue
        t, size = transforms[view_dir.name]
        dst = work / "faces_clean" / view_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(view_dir.glob("*.png")):
            im, _, info, _ = clean_image(p, transform=t, ref_size=size)
            im.save(dst / p.name)
            report["faces"][f"{view_dir.name}/{p.stem}"] = info
            sheet_items.append((f"face/{view_dir.name}/{p.stem}", im))

    (out / "cleaner_report.json").write_text(json.dumps(report, indent=2))
    contact_sheet(sheet_items, work / "cleaner_contact.png")
    print(json.dumps(report, indent=2))
    print(f"\ncontact sheet: {work / 'cleaner_contact.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="protagonist")
    run(ap.parse_args().character)
