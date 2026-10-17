"""Face Cutter: neutral layers reproduce the drawing, expressions change only the face
zones, blink frames are drawn from the neutral eyes (sharp, her own skin and lashes).

Run: python -m tests.test_face_cutter
"""
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.color import rgb2lab

from src import config
from src.onboarding import face_cutter

CHAR = config.CHARACTERS / "protagonist"


def test_faces():
    info = face_cutter.run("protagonist")
    assert info["identity_max_diff"] <= 1, "neutral layers must reproduce the drawing"
    (lx, ly), (rx, ry) = info["pupils"]
    assert abs(ly - ry) < 6 and 40 < rx - lx < 120, "two pupils, level, sensible spacing"
    assert info["mouth_centre"][1] > ly + 30, "mouth below the eyes"

    out = CHAR / "faces" / "front"
    neutral = np.asarray(Image.open(CHAR / "views/front.png").convert("RGB")).astype(float)
    recipes = json.loads((config.ROOT / "expressions/recipes.json").read_text())
    for name, r in recipes["recipes"].items():
        img = face_cutter.compose(np.asarray(Image.open(CHAR / "views/front.png").convert("RGBA")), out, r)
        d = np.abs(img[..., :3].astype(float) - neutral).max(-1)
        if name != "neutral":
            assert (d > 20).sum() > 300, f"{name}: the expression must change the face"
    # layers are soft-edged and stay inside the head
    meta = json.loads((CHAR / "parts/front/parts.json").read_text())
    head = np.asarray(Image.open(CHAR / "parts/front/labels.png")) == meta["parts"]["head"]["id"]
    for layer in face_cutter.LAYERS:
        a = np.asarray(Image.open(out / layer / "smile.png"))[..., 3]
        assert not ((a > 0) & ~head).any(), f"{layer} layer leaks outside the head"
        assert ((a > 0) & (a < 255)).sum() > 50, f"{layer} layer has a soft edge"
    # blink frames: eyes covered (no eye white left), sharp like the neutral
    lab_n = rgb2lab(neutral / 255.0)
    for e in ("eyes_half", "eyes_closed"):
        im = np.asarray(Image.open(out / "eyes" / f"{e}.png").convert("RGB")).astype(float)
        a = np.asarray(Image.open(out / "eyes" / f"{e}.png"))[..., 3] > 0
        white_n = a & (lab_n[..., 0] > 88)
        white_e = a & (rgb2lab(im / 255.0)[..., 0] > 88)
        frac = white_e.sum() / max(white_n.sum(), 1)
        assert frac < (0.75 if e == "eyes_half" else 0.1), f"{e}: eye whites covered ({frac:.0%} left)"
        # crisp lash line: the strongest edge in the layer is as sharp as the neutral eye's
        # (a covered eye has less detail overall, so total sharpness is the wrong measure)
        g = lambda x: np.hypot(ndi.sobel(x.mean(-1), 0), ndi.sobel(x.mean(-1), 1))
        sharp = np.percentile(g(im)[a], 99.5) / np.percentile(g(neutral)[a], 99.5)
        assert sharp > 0.6, f"{e}: blink frame too soft ({sharp:.2f})"
        assert (a & (rgb2lab(im / 255.0)[..., 0] < 35)).sum() > 40, f"{e}: lash line drawn"
    print("face cutter: all checks passed")


if __name__ == "__main__":
    test_faces()
