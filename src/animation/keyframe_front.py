"""Front-view code-keyframe actions → action library (spec §9.1 'code keyframes').

Same motion.json format as video_import (world rotation deltas per part + hand poses), so
front_clip renders them unchanged.
  idle      : weight shifts, a slow head sway, arms swaying slightly (loops)
  look_down : the head tilts and stays; neutral face with the eyes of the sad image
  react     : a shock flinch — hands fly up to the chest, head jerks up, face switches
              (set by the action, see front_clip.ACTION_EYES)

Usage: python -m src.animation.keyframe_front
"""
import json

import numpy as np

from src import config
from src.utils.io import save_text

FPS = 24


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def idle(seconds=4.0):
    t = np.arange(int(seconds * FPS)) / FPS
    w = 2 * np.pi / seconds                                        # one slow cycle per clip: loops
    return {
        "head": 1.6 * np.sin(w * t + 0.6),
        "upper_arm_l": 1.2 * np.sin(w * t),
        "upper_arm_r": -1.2 * np.sin(w * t + 0.3),
        "forearm_l": 0.8 * np.sin(w * t - 0.4),
        "forearm_r": -0.8 * np.sin(w * t - 0.1),
    }, True


def look_down(seconds=4.0):
    t = np.arange(int(seconds * FPS)) / FPS
    down = smooth((t - 0.4) / 0.8)                                 # lowers over 0.8 s, then holds
    return {
        "head": 5.0 * down + 0.6 * np.sin(2 * np.pi * t / seconds) * down,   # tilts and stays
        "upper_arm_l": -1.0 * down,
        "upper_arm_r": 1.0 * down,
    }, False


def react(seconds=3.0, at=0.5):
    """shock: a sudden flinch at `at` s — head jerks up, both hands fly up toward the chest
    (fast in, overshoot, settle), then a small tremble; the face switches at the flinch."""
    t = np.arange(int(seconds * FPS)) / FPS
    u = np.clip((t - at) / 0.18, 0, 1)                             # 0.18 s: a fast flinch
    hit = 1 - (1 - u) ** 3                                         # ease out
    over = 0.12 * np.sin(np.pi * np.clip((t - at - 0.18) / 0.3, 0, 1))   # overshoot, settles
    pose = np.clip(hit + over, 0, None)
    tremble = 0.8 * np.sin(2 * np.pi * 7 * t) * (t > at + 0.5) * np.exp(-(t - at - 0.5).clip(0) * 1.2)
    return {
        "head_drop": -5.0 * pose,                                  # head jerks up a little
        "head": 2.0 * pose + tremble * 0.5,
        "upper_arm_l": 22.0 * pose + tremble, "upper_arm_r": -22.0 * pose - tremble,
        "forearm_l": -150.0 * pose, "forearm_r": 150.0 * pose,      # hands up toward the chest
        "hand_l": 15.0 * pose, "hand_r": -15.0 * pose,
    }, False, {"react_at": at, "default_expression": "shocked"}


ACTIONS = {"idle": idle, "look_down": look_down, "react": react}


def run():
    for name, fn in ACTIONS.items():
        res = fn()
        deltas, loop, extra = (res + ({},))[:3]
        n = len(next(iter(deltas.values())))
        act = {"fps": FPS, "frames": n, "view": "front", "loop": loop,
               "units": "world rotation (deg, screen, counter-clockwise +) relative to the rest pose",
               "deltas": {k: np.round(v, 2).tolist() for k, v in deltas.items()},
               "hand": {}, **extra}
        out = config.ROOT / "actions" / name
        out.mkdir(parents=True, exist_ok=True)
        save_text(json.dumps(act), out / "motion.json")
        save_text(json.dumps({"action": name, "view": "front", "fps": FPS, "frames": n, "loop": loop,
                              "source": "code keyframes"}, indent=2), out / "meta.json")
        save_text(f"# {name}\n\n- Source: code keyframes (`src/animation/keyframe_front.py`).\n",
                  out / "sources.md")
        print(name, n, "frames")


if __name__ == "__main__":
    run()
