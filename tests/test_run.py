"""Run: loopable cycle with flight phases, planted foot never slides, no torn knees.

Run: python -m tests.test_run
"""
import numpy as np

from src.animation import keyframe_run
from src.animation.animator import Walker


def test_run():
    act = keyframe_run.run("run")
    assert act["frames"] % 4 == 0, "two equal steps on twos"
    l, r = np.array(act["contacts"]["l"]), np.array(act["contacts"]["r"])
    assert (~(l | r)).sum() >= 2, "a run has flight phases (both feet off the ground)"
    assert not (l & r).any(), "a run never has both feet down"
    a = {k: np.array(v) for k, v in act["angles"].items()}
    knee = a["l_hip>l_knee"] - a["l_knee>l_ankle"]
    assert knee.max() <= 85, "heel kick limited: deeper knee folds tear the wide jeans"
    _, slide = Walker("protagonist", action="run").poses(2.0)
    assert np.max(slide["locked"]) < 2.0, "planted foot does not slide"
    print("run: all checks passed")


if __name__ == "__main__":
    test_run()
