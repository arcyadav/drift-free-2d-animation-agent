"""Front-view code-keyframe actions → action library (spec §9.1 'code keyframes').

Same motion.json format as video_import (world rotation deltas per part + hand poses), so
front_clip renders them unchanged.
  idle      : weight shifts, a slow head sway, arms swaying slightly (loops)
  look_down : head lowers / tilts a little and stays there; the face uses the eyes_down layer
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
        "head": 4.0 * down + 0.6 * np.sin(2 * np.pi * t / seconds) * down,   # slight tilt, alive
        "upper_arm_l": -1.0 * down,
        "upper_arm_r": 1.0 * down,
    }, False


ACTIONS = {"idle": idle, "look_down": look_down}


def run():
    for name, fn in ACTIONS.items():
        deltas, loop = fn()
        n = len(next(iter(deltas.values())))
        act = {"fps": FPS, "frames": n, "view": "front", "loop": loop,
               "units": "world rotation (deg, screen, counter-clockwise +) relative to the rest pose",
               "deltas": {k: np.round(v, 2).tolist() for k, v in deltas.items()},
               "hand": {}}
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
