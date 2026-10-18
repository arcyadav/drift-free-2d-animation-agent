"""Hand-designed side-view RUN cycle → action library (same motion.json format as the walk).

How a run differs from the walk:
  * flight: each foot is down only ~38% of the cycle → both feet are off the ground twice per
    cycle; the body follows a small ballistic arc between take-off and landing
  * stance: the knee absorbs the landing (bends most at mid-stance), the foot stays planted
    (it moves back under the body at running speed = still in the world → foot lock)
  * swing keyed in joint angles: the heel kicks up behind (knee ~115°), the thigh drives
    forward high, the shin reaches out just before landing
  * arms pump with ~90° elbows, opposite to the legs; strong forward lean, head kept up

Usage: python -m src.animation.keyframe_run
"""
import argparse
import json

import numpy as np

from src import config
from src.animation.keyframe_walk import FPS, SHIN, THIGH, ang, ik, leg_angles, smooth
from src.utils.io import save_text

STYLE = {
    "cycle_s": 0.67,       # ~3 steps / s (snapped to a multiple of 4 frames)
    "stance": 0.32,        # share of the cycle a foot is down (< 0.5 → flight phases)
    "step": 0.62,          # ankle travel under the body during stance (leg lengths)
    "ankle_h": 0.20,
    "land_flex": 14.0,     # knee at touch-down / take-off (deg)
    "absorb": 32.0,        # extra knee bend at mid-stance (landing absorbed)
    "kick": 75.0,          # swing knee peak (heel up behind); >~80 tears the wide jeans at the knee
    "kick_at": 0.38,       # ... at this share of the swing
    "drive": 36.0,         # thigh forward drive (deg from vertical) ...
    "drive_at": 0.72,      # ... reached here, then the leg reaches out to land
    "reach_knee": 18.0,    # knee just before landing
    "flight_rise": 0.07,   # apex of the flight arc above take-off height (leg lengths)
    "arm_swing": 32.0,     # deg each way, pumping
    "elbow": (80.0, 100.0),
    "lean": 10.0,
    "head_pitch": -6.0,    # head kept up against the lean
    "heel_strike": 4.0,
    "toe_off": -35.0,
}


def build(st=STYLE):
    F = int(4 * round(st["cycle_s"] * FPS / 4))
    cycle = F / FPS
    S = st["stance"]
    speed = st["step"] / (S * cycle)
    phase = {"l": 0.0, "r": 0.5}

    def stance_q(p):
        return p / S if p < S else None

    def flex(q):                         # stance knee: land soft, absorb, push off
        return st["land_flex"] + st["absorb"] * np.sin(np.pi * q)

    def stance_height(q):
        x = st["step"] / 2 - st["step"] * q
        d = (THIGH + SHIN) * np.cos(np.radians(flex(q)) / 2)
        return st["ankle_h"] + np.sqrt(max(d ** 2 - x ** 2, 0.05)), x

    # body height: from the planted leg in stance, a ballistic arc in flight
    H = np.zeros(F)
    h_edge = stance_height(0.0)[0]                     # take-off == landing height
    for f in range(F):
        hs = [stance_height(stance_q((f / F + phase[s]) % 1))[0]
              for s in "lr" if stance_q((f / F + phase[s]) % 1) is not None]
        if hs:
            H[f] = min(hs)
        else:                                          # flight: parabola between the stances
            p = (f / F) % 0.5                          # share of the half cycle
            u = (p - S) / (0.5 - S)
            H[f] = h_edge + st["flight_rise"] * 4 * u * (1 - u)

    def stance_pose(s, f):
        q = stance_q((f / F + phase[s]) % 1)
        h, x = stance_height(q)
        root = np.array([0.0, H[f]])
        knee, ank = ik(root, np.array([x, st["ankle_h"]]))
        return leg_angles(root, knee, ank)

    def swing_angles(q, start, end):
        (th0, k0), (th1, k1) = start, end
        if q < st["drive_at"]:
            th = th0 + (st["drive"] - th0) * smooth(q / st["drive_at"])
        else:
            th = st["drive"] + (th1 - st["drive"]) * smooth((q - st["drive_at"]) / (1 - st["drive_at"]))
        if q < st["kick_at"]:
            k = k0 + (st["kick"] - k0) * smooth(q / st["kick_at"])
        else:
            k = st["kick"] + (k1 - st["kick"]) * smooth((q - st["kick_at"]) / (0.92 - st["kick_at"]))
        return th, max(k, k1)

    ends = {}
    for s in "lr":
        p0 = (1 - phase[s]) % 1
        first = int(round(p0 * F)) % F
        last = (first + int(round(S * F)) - 1) % F
        ends[s] = (stance_pose(s, last), stance_pose(s, first))

    names = ("pelvis", "chest", "neck", "head_top", "l_hip", "r_hip", "l_knee", "r_knee",
             "l_ankle", "r_ankle", "l_toe", "r_toe", "l_shoulder", "r_shoulder",
             "l_elbow", "r_elbow", "l_wrist", "r_wrist")
    J = {k: np.zeros((F, 2)) for k in names}
    contacts = {"l": [], "r": []}
    for f in range(F):
        root = np.array([speed * f / FPS, H[f]])
        J["pelvis"][f] = root
        lean = np.radians(st["lean"])
        J["chest"][f] = root + 0.55 * np.array([np.sin(lean), np.cos(lean)])
        J["neck"][f] = root + 0.75 * np.array([np.sin(lean), np.cos(lean)])
        J["head_top"][f] = J["neck"][f] + 0.35 * np.array([np.sin(0.057), np.cos(0.057)])
        for s in "lr":
            p = (f / F + phase[s]) % 1
            q = stance_q(p)
            if q is not None:
                _, x = stance_height(q)
                knee, ank = ik(root, root * [1, 0] + np.array([x, st["ankle_h"]]))
                pitch = st["heel_strike"] * (1 - smooth(q / 0.15)) + st["toe_off"] * smooth((q - 0.7) / 0.3)
            else:
                qs = (p - S) / (1 - S)
                th, k = swing_angles(qs, *ends[s])
                knee = root + THIGH * np.array([np.sin(np.radians(th)), -np.cos(np.radians(th))])
                ank = knee + SHIN * np.array([np.sin(np.radians(th - k)), -np.cos(np.radians(th - k))])
                pitch = st["toe_off"] + (st["heel_strike"] - st["toe_off"]) * smooth(qs)
            J[f"{s}_hip"][f], J[f"{s}_knee"][f], J[f"{s}_ankle"][f] = root, knee, ank
            fa = np.radians(-30 + pitch)
            J[f"{s}_toe"][f] = ank + 0.25 * np.array([np.cos(fa), np.sin(fa)])
            contacts[s].append(q is not None)
            # arms pump opposite to the same-side leg, elbows bent ~90°
            sw = np.radians(-st["arm_swing"] * np.cos(2 * np.pi * p))
            sh = root + np.array([0.0, 0.62])
            el = sh + 0.36 * np.array([np.sin(sw), -np.cos(sw)])
            fwd = (np.sin(sw) / np.sin(np.radians(st["arm_swing"])) + 1) / 2
            bend = np.radians(st["elbow"][0] + (st["elbow"][1] - st["elbow"][0]) * fwd)
            J[f"{s}_shoulder"][f], J[f"{s}_elbow"][f] = sh, el
            J[f"{s}_wrist"][f] = el + 0.34 * np.array([np.sin(sw + bend), -np.cos(sw + bend)])
    bones = [("pelvis", "chest"), ("chest", "neck"), ("neck", "head_top"),
             ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"),
             ("r_elbow", "r_wrist"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
             ("l_ankle", "l_toe"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_toe")]
    angles = {f"{a}>{b}": [round(ang(J[b][f] - J[a][f]), 2) for f in range(F)] for a, b in bones}
    r4 = lambda arr: np.round(arr, 4).tolist()
    flight = int(sum(not (a or b) for a, b in zip(contacts["l"], contacts["r"])))
    return {
        "fps": FPS, "frames": F, "view": "side", "loop": True,
        "units": "leg length (hip→ankle); x = forward, y = up from ground",
        "skeleton": list(J), "bones": [list(b) for b in bones],
        "joints": {k: r4(v) for k, v in J.items()},
        "lateral": {f"{s}_{j}": [0.1 if s == "l" else -0.1] * F
                    for s in "lr" for j in ("shoulder", "hip", "knee", "ankle", "wrist")},
        "angles": angles, "root": r4(J["pelvis"]), "contacts": contacts,
        "torso_lean": st["lean"], "head_pitch": st["head_pitch"],
        "stats": {"stride_per_cycle": round(speed * cycle, 3), "cycle_frames": F, "flight_frames": flight,
                  "bounce": round(float(H.max() - H.min()), 3), "style": st},
    }


def run(action="run"):
    act = build()
    out = config.ROOT / "actions" / action
    out.mkdir(parents=True, exist_ok=True)
    save_text(json.dumps(act), out / "motion.json")
    save_text(json.dumps({"action": action, "view": "side", "fps": FPS, "frames": act["frames"],
                          "loop": True, "source": "code keyframes",
                          **{k: v for k, v in act["stats"].items() if k != "style"}}, indent=2),
              out / "meta.json")
    save_text(f"# {action}\n\n- Source: hand-designed run cycle (`src/animation/keyframe_run.py`): "
              "flight phases, absorbing stance, heel kick, thigh drive, pumping arms.\n", out / "sources.md")
    print(json.dumps({k: v for k, v in act["stats"].items() if k != "style"}))
    return act


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="run")
    run(ap.parse_args().action)
