"""Front code-keyframe actions: hands follow their forearms (no 'broken wrist'), the shock
reaction starts neutral and flinches fast.

Run: python -m tests.test_front_actions
"""
import json

import numpy as np

from src import config
from src.animation import keyframe_front

WRIST_MAX = 30.0          # deg a hand may bend away from its forearm (world angles)


def test_actions():
    keyframe_front.run()
    for name in keyframe_front.ACTIONS:
        m = json.loads((config.ROOT / "actions" / name / "motion.json").read_text())
        d = {k: np.array(v) for k, v in m["deltas"].items()}
        for s in "lr":
            fore = d.get(f"forearm_{s}", np.zeros(m["frames"]))
            hand = d.get(f"hand_{s}", fore)          # absent = follows the forearm
            bend = np.abs(hand - fore).max()
            assert bend <= WRIST_MAX, f"{name}: hand_{s} bends {bend:.0f} deg away from the forearm"
    r = json.loads((config.ROOT / "actions" / "react" / "motion.json").read_text())
    fore = np.abs(np.array(r["deltas"]["forearm_l"]))
    at = int(r["react_at"] * r["fps"])
    assert fore[:at].max() < 1, "calm before the reaction"
    assert fore[at + int(0.25 * r["fps"])] > 0.8 * fore.max(), "the flinch is fast (< 0.25 s)"
    print("front actions: all checks passed")


if __name__ == "__main__":
    test_actions()
