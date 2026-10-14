"""Check an expression image against its view's neutral image.

Reports size match, global shift (should be ~0), where pixels changed
(should be only eyes/brows/mouth), and writes a visual report:
neutral crop | candidate crop | diff heatmap.

Usage: python -m src.utils.check_face assets/raw/faces/front/eyes_half.png [out.png]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageFilter

DIFF_THRESHOLD = 40  # per-pixel max channel difference counted as "changed"


def load(path):
    return Image.open(path).convert("RGB")


def global_shift(a, b):
    """Phase correlation on grayscale → (dx, dy) shift of b relative to a."""
    ga = np.asarray(a.convert("L"), dtype=np.float32)
    gb = np.asarray(b.convert("L"), dtype=np.float32)
    fa, fb = np.fft.fft2(ga), np.fft.fft2(gb)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-8
    corr = np.abs(np.fft.ifft2(r))
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = ga.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return -int(dx), -int(dy)


def check(candidate_path, out_path=None):
    candidate_path = Path(candidate_path)
    neutral_path = candidate_path.parent / "neutral.png"
    neutral, cand = load(neutral_path), load(candidate_path)

    print(f"neutral:   {neutral_path.name} {neutral.size}")
    print(f"candidate: {candidate_path.name} {cand.size}")
    if cand.size != neutral.size:
        print("!! size differs -> resizing candidate for comparison (regenerate if framing changed)")
        cand = cand.resize(neutral.size, Image.LANCZOS)

    dx, dy = global_shift(neutral, cand)
    print(f"global shift: dx={dx}px dy={dy}px  (want 0,0)")

    diff = ImageChops.difference(neutral, cand)
    d = np.asarray(diff).max(axis=2)
    changed = d > DIFF_THRESHOLD
    # drop isolated noise pixels (JPEG/regeneration grain)
    changed = np.asarray(
        Image.fromarray(changed.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))
    ) > 0
    h, w = changed.shape
    print(f"changed pixels: {changed.mean() * 100:.2f}% of image")

    if changed.any():
        ys, xs = np.where(changed)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        print(f"changed region bbox: x {x0}-{x1}, y {y0}-{y1} "
              f"({(x1 - x0) / w * 100:.0f}% wide, {(y1 - y0) / h * 100:.0f}% tall)")
        # rows split into bands to show *where* change happens
        bands = [("head (top 25%)", 0, 0.25), ("torso", 0.25, 0.55), ("legs/feet", 0.55, 1.0)]
        for name, a, b in bands:
            band = changed[int(a * h):int(b * h)]
            print(f"  {name:15s} {band.sum():7d} px changed")
    else:
        print("no significant change detected (expression may not have been applied)")

    # visual report: crop around head (top 30%) — neutral | candidate | heatmap
    if out_path:
        box = (int(w * 0.25), 0, int(w * 0.75), int(h * 0.3))
        heat = Image.fromarray((np.clip(d * 3, 0, 255)).astype(np.uint8)).convert("RGB")
        crops = [im.crop(box) for im in (neutral, cand, heat)]
        cw, ch = crops[0].size
        sheet = Image.new("RGB", (cw * 3, ch), "white")
        for i, c in enumerate(crops):
            sheet.paste(c, (i * cw, 0))
        sheet.save(out_path)
        print(f"visual report: {out_path}")


if __name__ == "__main__":
    check(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
