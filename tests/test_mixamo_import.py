"""Mixamo walk import must give a clean, loopable, walk-like side-view cycle.

Run: python -m tests.test_mixamo_import   (needs assets/motion/mixamo/Walking.zip)
"""
import numpy as np

from src import config
from src.animation import mixamo_import

SRC = config.ROOT / "assets/motion/mixamo/Walking.zip"


def test_walk():
    act = mixamo_import.run(SRC, "walk_mixamo")
    F, st = act["frames"], act["stats"]
    assert F == len(act["root"]) == len(act["contacts"]["l"])
    assert st["loop_seam_ratio"] < 2, "loop seam jumps more than two normal frames"
    assert 1.0 < st["stride_per_cycle"] < 3.0, "stride should be 1-3 leg lengths per cycle"

    l, r = np.array(act["contacts"]["l"]), np.array(act["contacts"]["r"])
    assert (l | r).all(), "walking: a foot is always on the ground"
    assert 0.3 < l.mean() < 0.7 and 0.3 < r.mean() < 0.7, "each foot planted roughly half the cycle"

    root = np.array(act["root"])
    assert (np.diff(root[:, 0]) > 0).all(), "pelvis moves forward every frame"
    assert np.mean(act["lateral"]["l_hip"]) > np.mean(act["lateral"]["r_hip"]), "+lateral = her left"
    print("mixamo import: all checks passed")


if __name__ == "__main__":
    test_walk()
