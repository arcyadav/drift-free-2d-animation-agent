"""Hand-designed side-view walk cycle → action library (spec §9.1 'code keyframes').

Why not only Mixamo: its walk is an adult's long, purposeful stride with bent arms — on a
small cartoon girl in wide jeans it reads stiff (deep knee folds, claw-like hands). This is
the classic 2D walk (contact → down → passing → up) with style parameters, written in the
same motion.json format, so the Retargeter / foot lock / renderer are unchanged.

Built from the feet up (all lengths in leg lengths, x = forward, y = up):
  * stance (60% of the cycle): the ankle slides back under the body at walking speed —
    i.e. it stands still on the ground in the world; the heel lifts in the last part
  * swing: the ankle arcs forward (smoothstep), lifted LIFT at mid-swing
  * hip height = what the stance leg(s) can reach with a slightly soft knee → natural
    down / up bounce; thigh + shin angles come from 2-bone IK (knee forward)
  * arms swing opposite the legs, elbows almost straight (more bend when forward)

Usage: python -m src.animation.keyframe_walk --action walk
"""
import argparse
import json

import numpy as np

from src import config
from src.utils.io import save_text

STYLE = {
    "cycle_s": 1.0,        # two steps per second (normal cadence)
    "step": 0.62,          # ankle travel under the body during stance (leg lengths)
    "stance": 0.6,         # share of the cycle a foot is on the ground
    "lift": 0.09,          # swing-foot lift at mid-swing
    "reach": 0.995,        # stance leg never fully straight: soft knee (~11 deg)
    "ankle_h": 0.20,       # ankle above the ground (her sneaker is tall)
    "arm_swing": 13.0,     # deg each way
    "elbow": (6.0, 16.0),  # elbow bend (back, forward)
    "lean": 2.0,           # deg forward
    "heel_strike": 12.0,   # toe up at contact, deg
    "push_off": 22.0,      # toe down at the end of stance, deg
}
FPS = 24
THIGH = SHIN = 0.5


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def ankle_body(p, st):
    """ankle position relative to the hip's ground point, phase p (0 = heel strike)."""
    s, sf = st["step"], st["stance"]
    if p < sf:
        q = p / sf
        x = s / 2 - s * q
        heel = smooth((q - 0.75) / 0.25) * 0.05                # heel rises before toe-off
        return np.array([x, st["ankle_h"] + heel]), True
    q = (p - sf) / (1 - sf)
    x = -s / 2 + s * smooth(q)
    y = st["ankle_h"] + 0.05 * (1 - q) + st["lift"] * np.sin(np.pi * q)
    return np.array([x, y]), False


def ik(hip, ankle):
    d_vec = ankle - hip
    d = np.clip(np.linalg.norm(d_vec), 1e-3, THIGH + SHIN - 1e-4)
    u = d_vec / np.linalg.norm(d_vec)
    a = np.arccos(np.clip((THIGH ** 2 + d ** 2 - SHIN ** 2) / (2 * THIGH * d), -1, 1))
    best = None
    for sg in (1, -1):
        c, s = np.cos(sg * a), np.sin(sg * a)
        k = hip + THIGH * np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])
        if best is None or k[0] > best[0]:                      # knee forward (+x)
            best = (k[0], k)
    knee = best[1]
    ank = knee + SHIN * (ankle - knee) / np.linalg.norm(ankle - knee)
    return knee, ank


def ang(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def build(st=STYLE):
    F = int(round(st["cycle_s"] * FPS))
    speed = st["step"] / (st["stance"] * st["cycle_s"])          # stance foot stands still
    stride = speed * st["cycle_s"]
    phase = {"l": 0.0, "r": 0.5}
    # hip height: lowest of what each planted leg can reach with a soft knee
    H = np.zeros(F)
    for f in range(F):
        h = []
        for s in "lr":
            a, planted = ankle_body((f / F + phase[s]) % 1, st)
            if planted:
                h.append(a[1] + np.sqrt(max(st["reach"] ** 2 - a[0] ** 2, 0.3)))
        H[f] = min(h)
    # (no smoothing: averaging lowers the peak and bends the planted knee a lot)

    J = {k: np.zeros((F, 2)) for k in ("pelvis", "chest", "neck", "head_top", "l_hip", "r_hip",
                                       "l_knee", "r_knee", "l_ankle", "r_ankle", "l_toe", "r_toe",
                                       "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
                                       "l_wrist", "r_wrist")}
    contacts = {"l": [], "r": []}
    angles = {}
    for f in range(F):
        t = f / FPS
        root = np.array([speed * t, H[f]])
        J["pelvis"][f] = root
        lean = np.radians(st["lean"])
        J["chest"][f] = root + 0.55 * np.array([np.sin(lean), np.cos(lean)])
        J["neck"][f] = root + 0.75 * np.array([np.sin(lean), np.cos(lean)])
        J["head_top"][f] = J["neck"][f] + np.array([0.02, 0.35])
        for s in "lr":
            p = (f / F + phase[s]) % 1
            a, planted = ankle_body(p, st)
            hip = root                                           # H is the hip height
            knee, ank = ik(hip, root * [1, 0] + a)
            J[f"{s}_hip"][f], J[f"{s}_knee"][f], J[f"{s}_ankle"][f] = hip, knee, ank
            # foot pitch: toe up at heel strike, flat, toe down at push-off, swing in between
            if p < 0.08:
                pitch = st["heel_strike"] * (1 - smooth(p / 0.08))
            elif p < st["stance"] * 0.75:
                pitch = 0.0
            elif p < st["stance"]:
                pitch = -st["push_off"] * smooth((p - st["stance"] * 0.75) / (st["stance"] * 0.25))
            else:
                q = (p - st["stance"]) / (1 - st["stance"])
                pitch = -st["push_off"] + (st["push_off"] + st["heel_strike"]) * smooth(q)
            fa = np.radians(-30 + pitch)                       # -30 = flat foot (ankle → toe base)
            J[f"{s}_toe"][f] = ank + 0.25 * np.array([np.cos(fa), np.sin(fa)])
            contacts[s].append(bool(planted))
            # arms: opposite to the same-side leg
            sw = np.radians(-st["arm_swing"] * np.cos(2 * np.pi * p))   # + = forward
            sh = root + np.array([0.0, 0.62])
            J[f"{s}_shoulder"][f] = sh
            up = 0.36 * np.array([np.sin(sw), -np.cos(sw)])
            el = sh + up
            fwd = (np.sin(sw) / np.sin(np.radians(st["arm_swing"])) + 1) / 2
            bend = np.radians(st["elbow"][0] + (st["elbow"][1] - st["elbow"][0]) * fwd)
            fo = 0.34 * np.array([np.sin(sw + bend), -np.cos(sw + bend)])
            J[f"{s}_elbow"][f], J[f"{s}_wrist"][f] = el, el + fo
    bones = [("pelvis", "chest"), ("chest", "neck"), ("neck", "head_top"),
             ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"),
             ("r_elbow", "r_wrist"), ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
             ("l_ankle", "l_toe"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_toe")]
    for a, b in bones:
        angles[f"{a}>{b}"] = [round(ang(J[b][f] - J[a][f]), 2) for f in range(F)]
    r4 = lambda arr: np.round(arr, 4).tolist()
    return {
        "fps": FPS, "frames": F, "view": "side", "loop": True,
        "units": "leg length (hip→ankle); x = forward, y = up from ground",
        "skeleton": list(J), "bones": [list(b) for b in bones],
        "joints": {k: r4(v) for k, v in J.items()},
        "lateral": {f"{s}_{j}": [0.1 if s == "l" else -0.1] * F
                    for s in "lr" for j in ("shoulder", "hip", "knee", "ankle", "wrist")},
        "angles": angles, "root": r4(J["pelvis"]), "contacts": contacts,
        "stats": {"stride_per_cycle": round(stride, 3), "cycle_frames": F,
                  "bounce": round(float(H.max() - H.min()), 3),
                  "contact_frames": {s: int(sum(c)) for s, c in contacts.items()},
                  "style": st},
    }


def run(action="walk"):
    act = build()
    out = config.ROOT / "actions" / action
    out.mkdir(parents=True, exist_ok=True)
    save_text(json.dumps(act), out / "motion.json")
    save_text(json.dumps({"action": action, "view": "side", "fps": FPS, "frames": act["frames"],
                          "loop": True, "source": "code keyframes", **{k: v for k, v in act["stats"].items()
                                                                      if k != "style"}}, indent=2),
              out / "meta.json")
    save_text(f"# {action}\n\n- Source: hand-designed 2D walk cycle (`src/animation/keyframe_walk.py`), "
              "classic contact / down / passing / up poses built from foot paths + 2-bone IK.\n"
              f"- Style: {json.dumps(STYLE)}\n- Alternative: `walk_mixamo` (Mixamo 'Walking').\n",
              out / "sources.md")
    print(json.dumps(act["stats"], indent=2))
    return act


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", default="walk")
    run(ap.parse_args().action)
