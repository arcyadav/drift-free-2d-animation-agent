"""Keyframed walk + animator: loopable cycle, a foot always down, planted foot never slides.

Run: python -m tests.test_walk
"""
import numpy as np

from src.animation import keyframe_walk
from src.animation.animator import Walker


def test_walk():
    act = keyframe_walk.run("walk")
    l, r = np.array(act["contacts"]["l"]), np.array(act["contacts"]["r"])
    assert (l | r).all(), "a foot is always on the ground"
    ang = {k: np.array(v) for k, v in act["angles"].items()}
    seam = np.mean([abs(v[0] - v[-1]) for v in ang.values()])
    step = np.mean([np.abs(np.diff(v)).mean() for v in ang.values()])
    assert seam < 2.5 * step, "cycle must loop smoothly"
    knee = ang["l_hip>l_knee"] - ang["l_knee>l_ankle"]          # knee flexion
    stance = knee[l][: int(0.75 * l.sum())]                  # before heel-off (knee bends then)
    assert stance.max() < 20, f"planted knee nearly straight (soft), got {stance.max():.0f} deg"
    assert knee.max() < 70, "swing knee bends less than 70 degrees"

    _, slide = Walker("protagonist").poses(3.0)
    assert np.mean(slide["locked"]) < 0.5 and np.max(slide["locked"]) < 2.0, slide["locked"]
    print("walk: all checks passed")


if __name__ == "__main__":
    test_walk()
