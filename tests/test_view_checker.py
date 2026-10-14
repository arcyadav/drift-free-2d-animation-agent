"""View Checker must pass the real views and catch deliberately broken ones.

Run: python -m tests.test_view_checker   (needs the Sheet Cleaner output for protagonist)
"""
import shutil

import numpy as np
from PIL import Image

from src import config
from src.onboarding import view_checker

SRC = config.CHARACTERS / "protagonist"
TST = config.CHARACTERS / "_selftest"


def make_broken_copy():
    shutil.rmtree(TST, ignore_errors=True)
    (TST / "views").mkdir(parents=True)
    for v in config.VIEWS:
        shutil.copy(SRC / "views" / f"{v}.png", TST / "views" / f"{v}.png")

    # 3q_left: jeans recoloured to near-black
    im = np.asarray(Image.open(TST / "views/3q_left.png")).copy()
    rgb = im[..., :3].astype(int)
    blue = (rgb[..., 2] > rgb[..., 0] + 15) & (im[..., 3] > 0)
    im[blue, :3] = (rgb[blue] * 0.35).astype(np.uint8)
    Image.fromarray(im).save(TST / "views/3q_left.png")

    # back: upper body stretched → tee hem / waist too low
    b = Image.open(TST / "views/back.png")
    w, _ = b.size
    top = b.crop((0, 0, w, 700)).resize((w, 800))
    b.paste((0, 0, 0, 0), (0, 0, w, 800))
    b.alpha_composite(top, (0, 0))
    b.save(TST / "views/back.png")


def test_real_views_pass():
    report = view_checker.run("protagonist")
    for v in config.VIEWS:
        assert report["summary"][v] == "ok", (v, report["views"][v]["issues"])


def test_broken_views_fail():
    make_broken_copy()
    try:
        report = view_checker.run("_selftest")
    finally:
        shutil.rmtree(TST, ignore_errors=True)
    assert report["summary"]["front"] == "ok"
    assert report["summary"]["side_left"] == "ok"
    assert report["summary"]["3q_left"] == "fail", report["views"]["3q_left"]["issues"]
    assert report["summary"]["back"] == "fail", report["views"]["back"]["issues"]


if __name__ == "__main__":
    test_real_views_pass()
    test_broken_views_fail()
    print("\nALL VIEW CHECKER TESTS PASSED")
