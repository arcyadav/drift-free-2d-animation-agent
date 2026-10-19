"""Crack sealing — the critic's fix for gaps between parts.

A frame rendered on a flat key colour shows exactly where background shows through *inside*
the silhouette. Thin, sliver-shaped enclosed holes are cracks between parts (a sleeve swinging off the torso,
an A-pose armpit gap) and are filled from the colours around them (OpenCV inpainting). Thick or
roundish enclosed openings are real (the triangle between the legs closed by the feet, hands on
hips) and are left alone.
Sizes are for the full shot; closer shots scale them by the zoom.

A crack is a gap *between* parts that the drawing did not have:
  * the same frame rendered with every part painted a flat ID colour tells which parts surround
    each hole — a hole inside one part is drawn that way;
  * holes already present in the rest pose (gaps between curls), carried into the frame by its
    camera, are drawn that way too.
Neither is ever touched.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi

from src.animation import rigid_pose

KEY = (0, 255, 0)            # pure green: no character colour is near it (magenta hit skin / sneakers)
KEY_DIST = 150               # |rgb - key| summed: above = character pixel
CRACK_MIN_BLOB = 25          # px: smaller enclosed holes are gaps between fingers etc.
CRACK_MAX_R = 8              # px half-width: thicker holes are openings (legs + feet close a triangle)
CRACK_SHAPE = 8.0            # area / half-width²: cracks are slivers (9-20), openings are fat (~5)


ID_STEP = 5                  # part i is painted (i * ID_STEP, 0, 128)
RING_SHARE = 0.15            # a part must own this share of a hole's rim to count as a neighbour
DRAWN_SLACK = 12             # rig px: rest-pose holes are widened by this (small head sways)
DRAWN_SHARE = 0.5            # a hole this much inside a rest-pose hole was drawn
_IDS, _ID_CACHE = {}, {}


def id_layers(imgs):
    """the same layers painted one flat colour per part (face layers belong to the head)"""
    out = {}
    for n, a in imgs.items():
        hit = _ID_CACHE.get((n, id(a)))
        if hit is None or hit[0] is not a:
            g = "head" if n.startswith("face_") else n
            i = _IDS.setdefault(g, len(_IDS) + 1)
            b = a.copy()
            b[..., 0], b[..., 1], b[..., 2] = i * ID_STEP, 0, 128
            _ID_CACHE[(n, id(a))] = hit = (a, b)              # keeping a alive pins its id
        out[n] = hit[1]
    return out


def rim_parts(ids, hole):
    """part ids owning the rim of one hole"""
    rim = ndi.binary_dilation(hole, iterations=2) & ~hole
    px = ids[rim].astype(int)
    px = px[(px[:, 1] < 40) & (np.abs(px[:, 2] - 128) < 40)]
    if not len(px):
        return set()
    v, c = np.unique(np.round(px[:, 0] / ID_STEP).astype(int), return_counts=True)
    return set(v[c >= RING_SHARE * len(px)].tolist())


def drawn_holes(meta, imgs, bends=()):
    """holes of the rest pose, in rig px (the art's own gaps), slightly widened"""
    W, H = meta["canvas"]
    keyed = rigid_pose.render(meta, imgs, {}, canvas=np.full((H, W, 3), KEY, np.uint8), bends=bends)
    m = silhouette(keyed)
    return ndi.binary_dilation(ndi.binary_fill_holes(m) & ~m, iterations=DRAWN_SLACK)


def to_screen(mask, M, size):
    """rig-px mask → screen through the 3x3 camera(·root) matrix"""
    return cv2.warpAffine(mask.astype(np.uint8), M[:2].astype(np.float64), size, flags=cv2.INTER_NEAREST) > 0


def silhouette(keyed):
    return np.abs(keyed.astype(int) - KEY).sum(-1) > KEY_DIST


def holes(keyed, zoom=1.0, ids=None, drawn=None):
    """cracks: enclosed background slivers ≥ CRACK_MIN_BLOB px; zoom = camera scale / full-shot
    scale; ids = the part-ID render: keep only holes between parts; drawn = rest-pose holes on
    screen: drop holes the art already had"""
    m = silhouette(keyed)
    out = np.zeros_like(m)
    ys, xs = np.where(m.any(1))[0], np.where(m.any(0))[0]
    if not len(ys):
        return out
    box = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))   # work on the character only
    mb = m[box]
    h = ndi.binary_fill_holes(mb) & ~mb
    lab, n = ndi.label(h)
    if not n:
        return out
    idx = np.arange(1, n + 1)
    area = ndi.sum(h, lab, idx)
    r = np.maximum(ndi.maximum(ndi.distance_transform_edt(h), lab, idx), 1.0)
    keep = (area >= CRACK_MIN_BLOB * zoom ** 2) & (r <= CRACK_MAX_R * zoom) & (area >= CRACK_SHAPE * r ** 2)
    if drawn is not None and keep.any():
        inside = ndi.sum(drawn[box], lab, idx) / np.maximum(ndi.sum(h, lab, idx), 1)
        keep &= inside < DRAWN_SHARE
    if ids is not None:
        ib = ids[box]
        for j in np.where(keep)[0]:
            keep[j] = len(rim_parts(ib, lab == idx[j])) >= 2
    out[box] = np.isin(lab, idx[keep])
    return out


def seal(img, keyed, zoom=1.0, ids=None, drawn=None):
    """fill thin cracks in img (the same frame rendered on the key colour shows where they are)."""
    h = holes(keyed, zoom, ids, drawn)
    if not h.any():
        return img
    m = ndi.binary_dilation(h, iterations=1).astype(np.uint8) * 255
    return cv2.inpaint(np.ascontiguousarray(img), m, 3, cv2.INPAINT_TELEA)
