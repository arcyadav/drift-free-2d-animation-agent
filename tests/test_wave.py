"""Wave from the phone video: motion only, a real wave, clean open-hand swap.

Run: python -m tests.test_wave   (the video itself is local only; the test checks the
stored motion and the hand cut-out, and re-imports only if the video is present)
"""
import json

import numpy as np
from PIL import Image
from skimage.color import rgb2lab

from src import config
from src.onboarding import hand_swaps

ACT = config.ROOT / "actions" / "wave" / "motion.json"
CHAR = config.CHARACTERS / "protagonist"


def test_wave():
    m = json.loads(ACT.read_text())
    assert set(m) >= {"fps", "frames", "deltas", "hand"} and "joints" not in m, "motion only"
    fore = np.array(m["deltas"]["forearm_l"])
    up = np.array(m["deltas"]["upper_arm_l"])
    assert fore.max() > 90, "forearm comes up (elbow-down wave)"
    assert np.abs(up).max() < 30, "upper arm stays relaxed"
    raised = fore[fore > 90]
    swings = np.sum(np.diff(np.sign(np.diff(raised))) != 0)
    assert swings >= 4, f"waves side to side ({swings} turns)"
    hand = m["hand"]["hand_l"]
    assert 0.3 < hand.count("open") / len(hand) < 0.95, "open hand while waving, relaxed at rest"
    assert hand[0] == "relaxed" and hand[-1] == "relaxed"

    hand_swaps.run("protagonist")
    im = np.asarray(Image.open(CHAR / "hands/front/open_l.png").convert("RGBA"))
    a = im[..., 3] > 0
    lab = rgb2lab(im[..., :3] / 255.0)
    assert (a & (lab[..., 2] < -8) & (lab[..., 0] < 60)).sum() == 0, "no denim in the open hand"
    rep = json.loads((CHAR / "hands/front/hands.json").read_text())
    assert rep["open_l"]["aligned"], "open-hand edit lines up with the front view"
    # a hand does not grow when it opens: open hand as long as the relaxed one (+-10%)
    meta = json.loads((CHAR / "parts/front/parts.json").read_text())
    wr = np.array(meta["parts"]["hand_l"]["pivot"])
    el = np.array(meta["parts"]["forearm_l"]["pivot"])
    d = (wr - el) / np.linalg.norm(wr - el)
    def length(m):
        ys, xs = np.where(m)
        return np.percentile((np.stack([xs, ys], 1) - wr) @ d, 99)
    relaxed = np.asarray(Image.open(CHAR / "parts/front/hand_l.png"))[..., 3] > 128
    ratio = length(im[..., 3] > 128) / length(relaxed)
    assert 0.9 < ratio < 1.1, f"open hand size matches the relaxed hand ({ratio:.2f})"

    # the wrist moves during the wave (follow-through), not only the elbow
    from src.animation.front_clip import FrontClip
    clip = FrontClip("protagonist", "wave")
    rel = []
    for local, hand, _, _ in clip.frames():
        if hand.get("hand_l") == "open":
            rel.append(local.get("hand_l", 0.0))
    assert np.ptp(rel) > 10, f"wrist bends while waving ({np.ptp(rel):.0f} deg range)"
    print("wave: all checks passed")


if __name__ == "__main__":
    test_wave()
