"""Face Cutter (spec §7, §9.2): expression images → eyes / brows / mouth layers.

The face images are full-body edits of the front view with only the face changed, so
  1. align them with the front view's cleaner transform (as for the hands)
  2. zones from the neutral face: pupils = the two darkest blobs in the head, brows = the
     dark strokes above them, mouth = where the expressions change the lower face
  3. each image → soft-edged layers inside the zones only (the image tools also redraw
     a few pixels along the face outline / hairline — outside the zones, ignored)
  4. recipes (expressions/recipes.json) combine layers: sad = brows + mouth of sad.png,
     eyes of eyes_down.png; a blink swaps only the eyes (eyes_half → eyes_closed)
Check: neutral layers composited on the head reproduce the drawing.

Outputs  characters/<c>/faces/front/<layer>/<image>.png (full-canvas RGBA, soft alpha)
         characters/<c>/faces/front/face.json, expressions/recipes.json
         characters/<c>/work/face_layers_front.png (contact sheet of all recipes)
Usage: python -m src.onboarding.face_cutter --character protagonist
"""
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.color import rgb2lab

from src import config
from src.onboarding.sheet_cleaner import apply_transform, clean_image
from src.utils.io import save_image, save_text

LAYERS = ("eyes", "brows", "mouth")          # drawn in this order over the head
RECIPES = {
    "neutral": {"eyes": "neutral", "brows": "neutral", "mouth": "neutral"},
    "smile": {"eyes": "smile", "brows": "smile", "mouth": "smile"},
    "sad": {"eyes": "eyes_down", "brows": "sad", "mouth": "sad"},
    "worried": {"eyes": "worried", "brows": "worried", "mouth": "worried"},
    "shocked": {"eyes": "shocked", "brows": "shocked", "mouth": "shocked"},
    "look_down": {"eyes": "eyes_down", "brows": "neutral", "mouth": "neutral"},
}
BLINK = ["eyes_half", "eyes_closed", "eyes_half"]   # eye images of a blink (between open frames)
PUPIL_L = 30            # pupils / iris: darkest blobs in the face
EYE_RX, EYE_RY = 1.25, 1.05      # eye zone radii as multiples of the eye spacing / 2.5 ...
BROW_UP = 0.62          # brow centre above the pupil (x eye spacing)
FEATHER = 3.5           # px of soft edge on every layer
SYNTH_BLINK = True      # blink frames drawn from the neutral eyes instead of the soft edits
LASH_PX = 4.0           # lash line thickness at the lid
COVER_RING = 3          # px around the opening also repainted (its outline / lash)
COLOR_SIGMA = 14        # smoothness (px) of the skin-tone correction field
SOFT_RATIO = 0.6        # image sharper than this share of the neutral counts as fine ...
SHARPEN = 0.8           # ... else unsharp mask with this amount


def ellipse(shape, c, rx, ry):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return ((xx - c[0]) / rx) ** 2 + ((yy - c[1]) / ry) ** 2 <= 1


def soft(mask):
    """feathered alpha 0..1 from a hard zone mask (inside stays 1)."""
    d_in = ndi.distance_transform_edt(mask)
    return np.clip(d_in / FEATHER, 0, 1)


def find_zones(neutral, head, expressions):
    lab = rgb2lab(neutral[..., :3] / 255.0)
    dark = head & (lab[..., 0] < PUPIL_L)
    cl, n = ndi.label(dark)
    sizes = ndi.sum(dark, cl, range(1, n + 1))
    two = np.argsort(sizes)[-2:] + 1
    eyes = sorted([np.array(ndi.center_of_mass(cl == i))[::-1] for i in two], key=lambda c: c[0])
    gap = np.linalg.norm(eyes[1] - eyes[0])
    rx, ry = EYE_RX * gap / 2.5, EYE_RY * gap / 2.5
    eye_c = [e + np.array([0, -0.08 * gap]) for e in eyes]          # lids sit above the pupil
    brow_c = [e + np.array([0, -BROW_UP * gap]) for e in eyes]
    # mouth: where the expressions change the face below the eyes and above the chin
    below = np.zeros_like(head)
    ys = np.where(head.any(1))[0]
    yy = np.mgrid[:head.shape[0], :head.shape[1]][0]
    below = head & (yy > eyes[0][1] + 0.55 * gap)
    changed = np.zeros_like(head)
    for e in expressions.values():
        d = np.linalg.norm(rgb2lab(e[..., :3] / 255.0) - lab, axis=-1)
        changed |= below & (d > 10)
    changed = ndi.binary_opening(changed, iterations=1)
    cy, cx = ndi.center_of_mass(changed)
    my = np.where(changed)[0]
    mouth_c = np.array([cx, cy])
    m_rx, m_ry = 0.62 * gap, 0.5 * (np.percentile(my, 97) - np.percentile(my, 3)) + 4
    zones = {
        "eyes": np.any([ellipse(head.shape, c, rx, ry) for c in eye_c], 0),
        "brows": np.any([ellipse(head.shape, c, 0.85 * rx, 0.36 * ry) for c in brow_c], 0),
        "mouth": ellipse(head.shape, mouth_c, m_rx, m_ry),
    }
    zones["eyes"] &= ~zones["brows"]                                  # brows win over the eye zone
    info = {"pupils": [c.round(1).tolist() for c in eyes], "eye_spacing": round(float(gap), 1),
            "eye_radii": [round(float(rx), 1), round(float(ry), 1)],
            "brow_centres": [c.round(1).tolist() for c in brow_c], "mouth_centre": mouth_c.round(1).tolist(),
            "mouth_radii": [round(float(m_rx), 1), round(float(m_ry), 1)]}
    return zones, info


def match_to_neutral(img, neutral, zone):
    """Edits come back with slightly different skin tone (dE up to ~6) and sometimes soft
    (Gemini). Correct the colour with a smooth field measured on skin around the features,
    and sharpen soft images, so a pasted layer has no visible patch edge."""
    lab_e = rgb2lab(img[..., :3] / 255.0)
    lab_n = rgb2lab(neutral[..., :3] / 255.0)
    skin = zone & (lab_n[..., 0] > 55) & (lab_n[..., 1] > 8) & (lab_e[..., 0] > 55) & (lab_e[..., 1] > 8)
    skin &= np.linalg.norm(lab_e - lab_n, axis=-1) < 12                # same skin, not a changed feature
    out = img.astype(np.float32).copy()
    fix = {"skin_px": int(skin.sum())}
    if skin.sum() > 200:
        d = np.zeros_like(lab_e)
        d[skin] = (lab_n - lab_e)[skin]
        w = ndi.gaussian_filter(skin.astype(float), COLOR_SIGMA)
        field = np.stack([ndi.gaussian_filter(d[..., c], COLOR_SIGMA) for c in range(3)], -1) / np.maximum(w, 1e-3)[..., None]
        field[w < 0.02] = (lab_n - lab_e)[skin].mean(0)
        from skimage.color import lab2rgb
        out[..., :3] = np.clip(lab2rgb(lab_e + field) * 255, 0, 255)
        fix["mean_shift_dE"] = round(float(np.linalg.norm((lab_n - lab_e)[skin].mean(0))), 2)
    sharp_n = ndi.laplace(neutral[..., :3].mean(-1).astype(float))[zone].var()
    sharp_e = ndi.laplace(out[..., :3].mean(-1))[zone].var()
    if sharp_e < SOFT_RATIO * sharp_n:                                  # soft image: unsharp mask
        blur = np.stack([ndi.gaussian_filter(out[..., c], 1.2) for c in range(3)], -1)
        out[..., :3] = np.clip(out[..., :3] + SHARPEN * (out[..., :3] - blur), 0, 255)
        fix["sharpened"] = True
    return out.astype(np.uint8), fix


def eye_opening(lab, pupil, gap):
    """the visible eye (white + iris + upper lash) around a pupil, from the neutral face."""
    h, w = lab.shape[:2]
    zone = ellipse((h, w), pupil + np.array([0, -0.03 * gap]), 0.36 * gap, 0.26 * gap)
    feat = zone & ((lab[..., 0] > 84) | (lab[..., 0] < 40))           # white or dark (iris, lash)
    cl, _ = ndi.label(feat)
    keep = cl == cl[int(pupil[1]), int(pupil[0])]
    return ndi.binary_fill_holes(ndi.binary_closing(keep, iterations=2)) & zone


def synth_blinks(neutral, pupils, gap):
    """Blink frames drawn from the neutral eyes (the generated blink images were soft and
    off-tone): skin painted over the opening (inpainting from the surrounding skin), and
    the upper lash line redrawn at the lid: halfway down (half), at the bottom (closed)."""
    import cv2
    rgb = neutral[..., :3].copy()
    lab = rgb2lab(rgb / 255.0)
    lash_px = []
    masks = []
    for p in pupils:
        m = eye_opening(lab, np.array(p), gap)
        masks.append(m)
        lash_px.append(rgb[m & (lab[..., 0] < 30)])
    lash = np.median(np.concatenate(lash_px), 0)
    out = {}
    for name, depth in (("eyes_half", 0.5), ("eyes_closed", 1.0)):
        img = rgb.astype(np.float32).copy()
        lines = []
        for m in masks:
            cols = np.where(m.any(0))[0]
            top = np.array([np.where(m[:, c])[0].min() for c in cols], float)
            bot = np.array([np.where(m[:, c])[0].max() for c in cols], float)
            lid = top + depth * (bot - top)
            # smooth lid arc (a quadratic through the lid points, ends trimmed)
            k = slice(int(0.08 * len(cols)), int(0.92 * len(cols)) + 1)
            fit = np.polyfit(cols[k], lid[k], 2)
            xs = np.arange(cols[k][0], cols[k][-1] + 1)
            arc = np.polyval(fit, xs) - (1.0 if depth == 1.0 else 0.0)
            # cover: the opening (and its lash / outline ring) above the lid
            yy = np.arange(rgb.shape[0])[:, None]
            lid_full = np.full(rgb.shape[1], -1.0)
            lid_full[cols] = np.polyval(fit, cols)
            ring_zone = ndi.binary_dilation(m, iterations=COVER_RING)
            cols_r = np.where(ring_zone.any(0))[0]
            lid_full[cols_r] = np.polyval(fit, np.clip(cols_r, cols[0], cols[-1]))
            cover = ring_zone & (yy <= lid_full[None, :] + (2 if depth == 1.0 else 0))
            cover &= (lab[..., 0] < 88) | m                                # not past the eye (skin only)
            # flat fill with her own skin colour from a clean ring around the eye
            skin_ring = ndi.binary_dilation(m, iterations=COVER_RING + 8) & ~ring_zone                 & (lab[..., 0] > 60) & (lab[..., 0] < 82) & (lab[..., 1] > 8)
            skin = np.median(rgb[skin_ring], 0)
            a = ndi.gaussian_filter(cover.astype(np.float32), 1.0)[..., None]
            img = img * (1 - a) + skin * a
            lines.append(np.stack([xs, arc], 1))
        img = np.clip(img, 0, 255).astype(np.uint8)
        big = 4                                                     # anti-aliased lash line
        canvas = cv2.resize(img, None, fx=big, fy=big, interpolation=cv2.INTER_LINEAR)
        for ln in lines:
            pts = (ln * big).astype(np.int32).reshape(-1, 1, 2)
            thick = int(LASH_PX * big * (1.0 if depth < 1 else 0.75))
            cv2.polylines(canvas, [pts], False, tuple(int(v) for v in lash), thick, cv2.LINE_AA)
        img = cv2.resize(canvas, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        res = neutral.copy()
        res[..., :3] = img
        out[name] = res
    return out


def run(character, view="front"):
    char = config.CHARACTERS / character
    meta = json.loads((char / "parts" / view / "parts.json").read_text())
    labels = np.asarray(Image.open(char / "parts" / view / "labels.png"))
    head = labels == meta["parts"]["head"]["id"]
    _, t, _, size = clean_image(config.ASSETS_RAW / "turnaround" / f"{view}.png")
    imgs = {}
    for p in sorted((config.ASSETS_RAW / "faces" / view).glob("*.png")):
        # plain aligned image: no background removal (it punched out the bright eye whites)
        raw = Image.open(p).convert("RGBA")
        if raw.size != size:
            raw = raw.resize(size, Image.LANCZOS)
        imgs[p.stem] = np.asarray(apply_transform(raw, t))
    neutral = np.asarray(Image.open(char / "views" / f"{view}.png").convert("RGBA"))
    imgs["neutral"] = neutral                                        # exactly the rig's drawing
    zones, info = find_zones(neutral, head, {k: v for k, v in imgs.items() if k != "neutral"})
    alpha = {k: soft(z & head) for k, z in zones.items()}
    face_zone = ndi.binary_dilation(np.any(list(zones.values()), 0), iterations=12) & head
    if SYNTH_BLINK:                                                  # drawn from the neutral eyes
        imgs.update(synth_blinks(neutral, info["pupils"], info["eye_spacing"]))
    fixes = {}
    for k in list(imgs):
        if k == "neutral" or (SYNTH_BLINK and k in ("eyes_half", "eyes_closed")):
            continue
        imgs[k], fixes[k] = match_to_neutral(imgs[k], neutral, face_zone)

    out = char / "faces" / view
    for layer in LAYERS:
        (out / layer).mkdir(parents=True, exist_ok=True)
        for name, im in imgs.items():
            cut = im.copy()
            cut[..., 3] = (alpha[layer] * (im[..., 3] > 0) * 255).astype(np.uint8)
            save_image(cut, out / layer / f"{name}.png")

    # check: the neutral layers over the head reproduce the drawing
    head_rgb = neutral[..., :3].astype(float)
    comp = head_rgb.copy()
    for layer in LAYERS:
        a = alpha[layer][..., None]
        comp = comp * (1 - a) + neutral[..., :3] * a
    identity = float(np.abs(comp - head_rgb)[head].max())
    # how much each expression differs from neutral outside the zones (redraw noise ignored)
    outside = head & ~np.any(list(zones.values()), 0)
    noise = {k: round(float(np.linalg.norm(rgb2lab(v[..., :3] / 255.0) - rgb2lab(neutral[..., :3] / 255.0),
                                           axis=-1)[outside].mean()), 2) for k, v in imgs.items() if k != "neutral"}
    info.update({"images": sorted(imgs), "identity_max_diff": round(identity, 2), "fixes": fixes,
                 "outside_zone_mean_dE": noise, "layers": list(LAYERS), "blink": BLINK})
    save_text(json.dumps(info, indent=2), out / "face.json")
    save_text(json.dumps({"recipes": RECIPES, "blink": BLINK, "layers": list(LAYERS)}, indent=2),
              config.ROOT / "expressions" / "recipes.json")
    sheet(char, neutral, head, out, view)
    print(json.dumps({k: info[k] for k in ("pupils", "eye_spacing", "mouth_centre", "identity_max_diff")}))
    return info


def compose(neutral, out, recipe, eyes_override=None):
    img = neutral.astype(float).copy()
    for layer in LAYERS:
        src = eyes_override if (layer == "eyes" and eyes_override) else recipe[layer]
        L = np.asarray(Image.open(out / layer / f"{src}.png")).astype(float)
        a = L[..., 3:4] / 255
        img[..., :3] = img[..., :3] * (1 - a) + L[..., :3] * a
    return img.astype(np.uint8)


def sheet(char, neutral, head, out, view):
    ys, xs = np.where(head)
    box = (xs.min() - 30, ys.min() - 40, xs.max() + 30, ys.max() + 25)
    tiles = [(n, compose(neutral, out, r)) for n, r in RECIPES.items()]
    tiles += [(f"blink:{e}", compose(neutral, out, RECIPES["neutral"], e)) for e in ("eyes_half", "eyes_closed")]
    w, h = box[2] - box[0], box[3] - box[1]
    s = 2
    o = Image.new("RGB", (w * s * 4, (h * s + 24) * 2), "white")
    d = ImageDraw.Draw(o)
    for i, (n, im) in enumerate(tiles):
        bg = Image.new("RGBA", im.shape[1::-1], (255, 255, 255, 255))
        bg.alpha_composite(Image.fromarray(im))
        x, y = (i % 4) * w * s, (i // 4) * (h * s + 24)
        o.paste(bg.crop(box).convert("RGB").resize((w * s, h * s)), (x, y + 22))
        d.text((x + 6, y + 4), n, fill="black")
    save_image(o, char / "work" / f"face_layers_{view}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="protagonist")
    run(ap.parse_args().character)
