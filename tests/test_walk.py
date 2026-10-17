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
    st = knee[l]
    n = len(st)
    # natural walk, not a lunge: the support leg stays nearly straight, the body vaults over it
    assert st[: int(0.45 * n)].max() <= 25, f"support knee lunges: {st[: int(0.45 * n)].max():.0f} deg"
    assert st[int(0.35 * n): int(0.6 * n)].max() <= 15, "support leg nearly straight in mid-stance"
    assert knee.max() < 70, "swing knee bends less than 70 degrees"
    sw = knee[~l]
    assert sw[-2:].max() <= 8, "knee straight before the heel lands (leg reaches forward)"
    assert int(sw.argmax()) < len(sw) * 0.6, "swing knee bends most early in the swing"
    H = np.array(act["root"])[:, 1]
    assert np.abs(np.diff(np.r_[H, H[0]])).max() <= 0.016, "no sudden drops / hops in body height"
    assert 0.03 <= H.max() - H.min() <= 0.07, f"gentle bounce, got {H.max() - H.min():.3f}"
    heel = np.array(act["joints"]["l_ankle"])[:, 1]
    assert heel[l][-3:].max() > heel[l][:3].min() + 0.03, "heel lifts before toe-off (push-off)"
    assert act.get("torso_lean", 0) >= 3, "upper body leans forward into the step"
    J = {k: np.array(v) for k, v in act["joints"].items()}
    th = np.concatenate([np.degrees(np.arctan2((J[f"{x}_knee"] - J[f"{x}_hip"])[:, 0],
                                               -(J[f"{x}_knee"] - J[f"{x}_hip"])[:, 1])) for x in "lr"])
    assert np.percentile(th, 98) <= 26, "thigh does not kick up toward the body"
    assert -np.percentile(th, 2) >= 10, "leg trails far enough behind (thigh back >= 10 deg)"
    assert act["frames"] % 4 == 0, "cycle = two equal steps on twos (odd cycles limp)"

    frames, slide = Walker("protagonist").poses(3.0)
    y = np.array([off[1] for _, off in frames])
    per_step = act["frames"] // 2 * 24 // act["fps"]
    a, b = y[:per_step], y[per_step:2 * per_step]
    assert np.abs(a - b).max() < 0.5, "left and right steps identical (no limp)"
    assert np.mean(slide["locked"]) < 0.5 and np.max(slide["locked"]) < 2.0, slide["locked"]
    print("walk: all checks passed")


if __name__ == "__main__":
    test_walk()
