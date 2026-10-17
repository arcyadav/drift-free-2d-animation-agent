"""Hand-designed side-view walk cycle → action library (spec §9.1 'code keyframes').

Why not only Mixamo: its walk is an adult's long, purposeful stride with bent arms — on a
small cartoon girl in wide jeans it reads stiff (deep knee folds, claw-like hands). This is
the classic 2D walk (contact → down → passing → up) with style parameters, written in the
same motion.json format, so the Retargeter / foot lock / renderer are unchanged.

Built from the feet up (all lengths in leg lengths, x = forward, y = up):
  * body height is keyed per step like a 2D animator does, with heights that a real
    walk allows: contact (front leg reaching, low) -> down (foot settles, slight dip) ->
    passing (support leg nearly straight, HIGHEST) -> up (heel lifts as the foot is
    behind the body: push-off) -> next contact; shape-preserving curve between keys
  * stance (60% of the cycle): the ankle slides back under the body at walking speed —
    i.e. it stands still on the ground in the world. Where the body is higher than a
    flat foot can reach (up), the heel lifts by exactly what is needed: the push-off
  * swing: the ankle arcs forward (smoothstep), lifted LIFT at mid-swing
  * thigh + shin from 2-bone IK (knee forward): the knee gives at 'down' by itself
  * upper body follows: chest tips against the hips, head nods slightly after each down
  * arms swing opposite the legs, elbows almost straight (more bend when forward)

Usage: python -m src.animation.keyframe_walk --action walk
"""
import argparse
import json

import numpy as np

from src import config
from src.utils.io import save_text

STYLE = {
    "cycle_s": 1.5,        # ~1.33 steps / s: relaxed pace measured from a reference AI walk
                           # video (2 steps / s looked hurried); snapped to a multiple of 4 frames
    "step": 0.66,          # ankle travel under the body during stance (leg lengths)
    "stance": 0.6,         # share of the cycle a foot is on the ground
    "lift": 0.03,          # swing-foot lift at mid-swing: low, relaxed (higher kicked the thigh up)
    "swing_lag": 1.8,      # (foot-path swing, unused: the swing is keyed in joint angles)
    "swing_knee": 50.0,    # deg: swing knee bends most early in the swing (foot clears) ...
    "knee_peak_at": 0.35,  # ... at this share of the swing
    "knee_straight_at": 0.85,  # ... and is fully straight from here until the heel lands
    "thigh_reach_at": 0.8,     # thigh has swung forward to its heel-strike angle by here
    "foot_back": 0.03,     # footprints sit this much behind the hips so that at heel strike the
                           # hips are midway between the feet: legs open into a symmetric V
                           # (= step/2 - step/2.2; the stance foot keeps moving back during the
                           # 10% double support). Reach is guaranteed by the animator's pelvis drop.
    "heel_from": 0.75,     # share of stance after which the back heel lifts (back leg stays
                           # straight until just before toe-off: the V at heel strike)
    "knee_soft": 3.0,      # deg: stance knee almost straight (it vaults the body)
    # Real walking: the support leg stays nearly straight (a stilt) and the body vaults over
    # it, falling forward into the next step. Bending the support knee instead looks like a
    # lunge / penguin. So: hip height = what the support leg reaches with a soft knee,
    # minus a slight dip after heel strike ('down'); the back heel lifts as the body moves
    # ahead of it ('up', push-off) — no extra bounce.
    "settle": 0.016,       # leg lengths: body holds low while the front foot takes the weight
    "load_peak": 0.15,     # ... deepest at this share of the stance
    "descent": 0.012,      # leg lengths per frame: the body comes down steadily into each
                           # heel strike (a controlled fall onto the front foot, no drop)
    "heel_min": 0.08,      # heel lift at toe-off (leg lengths)
    "ankle_h": 0.20,       # ankle above the ground (her sneaker is tall)
    "arm_swing": 7.0,      # deg each way: relaxed arms (13 swung like marching)
    "elbow": (4.0, 10.0),  # elbow bend (back, forward)
    "lean": 3.0,           # deg: upper body tips forward from the hips (falling into the step)
    "chest_counter": 0.4,  # chest tips opposite to the hips (share of hip_tilt)
    "head_nod": 0.0,       # deg: head dip after 'down' — any visible nod reads as a chicken
    "hip_tilt": 3.0,       # deg: pelvis tips forward/back once per step (peaks at push-off)
    "heel_strike": 12.0,   # toe up at contact, deg
}
FPS = 24
THIGH = SHIN = 0.5


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


BALL = 0.22                # ankle -> ball of the foot (the foot pivots here at push-off)


def ankle_body(p, st):
    """ankle position relative to the hip's ground point, phase p (0 = heel strike)."""
    s, sf = st["step"], st["stance"]
    if p < sf:
        q = p / sf
        x = s / 2 - s * q - st["foot_back"]
        return np.array([x, st["ankle_h"]]), True              # heel lift added in build()
    q = (p - sf) / (1 - sf)
    # the foot travels forward late in the swing (after the knee has started to straighten):
    # swinging it forward early with a bent knee kicked the thigh up toward the body
    x = -s / 2 + s * smooth(q) ** st["swing_lag"] - st["foot_back"]
    y = st["ankle_h"] + st["heel_min"] * (1 - q) + st["lift"] * np.sin(np.pi * q)
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


def leg_angles(hip, knee, ank):
    """thigh angle from vertical (+ = forward) and knee flexion (+ = shin behind), deg."""
    v, w = knee - hip, ank - knee
    th = np.degrees(np.arctan2(v[0], -v[1]))
    return th, th - np.degrees(np.arctan2(w[0], -w[1]))


def swing_pose(q, start, end, st):
    """swing keyed like an animator: thigh swings forward and arrives by thigh_reach_at;
    knee bends to swing_knee early (foot clears the ground), then straightens fully by
    knee_straight_at and stays straight until the heel lands."""
    (th0, k0), (th1, k1) = start, end
    th = th0 + (th1 - th0) * smooth(q / st["thigh_reach_at"])
    pk, sk, peak = st["knee_peak_at"], st["knee_straight_at"], st["swing_knee"]
    if q < pk:
        k = k0 + (peak - k0) * smooth(q / pk)
    elif q < sk:
        k = peak + (k1 - peak) * smooth((q - pk) / (sk - pk))
    else:
        k = k1
    return th, k


def ang(v):
    return float(np.degrees(np.arctan2(v[1], v[0])))


def build(st=STYLE):
    # a multiple of 4 frames: two equal steps, each a whole number of 'twos' — an odd cycle
    # (35) made the steps 18 / 17 frames and sampled them differently: a limp
    F = int(4 * round(st["cycle_s"] * FPS / 4))
    cycle = F / FPS
    speed = st["step"] / (st["stance"] * cycle)                  # stance foot stands still
    stride = speed * cycle
    phase = {"l": 0.0, "r": 0.5}
    phase_frames = {s: (1 - ph) % 1 * F for s, ph in phase.items()}    # frame of heel strike
    # body height: the support leg(s) vault the body with a soft knee; a leg late in its
    # stance may lift its heel (that extends its reach), so the body is held up by the leg
    # it is moving over, not dragged down by the one it is leaving
    d_soft = (THIGH + SHIN) * np.cos(np.radians(st["knee_soft"]) / 2)

    def heel_ramp(q):                     # push-off heel lift late in stance (also in lift below)
        return st["heel_min"] * smooth((q - st["heel_from"]) / (1 - st["heel_from"]))

    H = np.zeros(F)
    for f in range(F):
        h = []
        for s_ in "lr":
            p = (f / F + phase[s_]) % 1
            a, planted = ankle_body(p, st)
            if planted:
                q = p / st["stance"]
                h.append(st["ankle_h"] + heel_ramp(q) + np.sqrt(max(d_soft ** 2 - a[0] ** 2, 0.0)))
        H[f] = min(h)
        for s_ in "lr":                                       # 'down': slight dip after heel strike
            q = ((f / F + phase[s_]) % 1) / st["stance"]
            if q < 0.45:
                bump = smooth(q / st["load_peak"]) if q < st["load_peak"] \
                    else 1 - smooth((q - st["load_peak"]) / (0.45 - st["load_peak"]))
                H[f] -= st["settle"] * bump
    # anticipation: descend steadily into each heel strike instead of dropping at contact
    Hc = st["ankle_h"] + np.sqrt(d_soft ** 2 - (st["step"] / 2 - st["foot_back"]) ** 2)
    strikes = sorted(int(round((1 - ph) % 1 * F)) % F for ph in phase.values())
    for f in range(F):
        until = min((c - f) % F for c in strikes)
        H[f] = min(H[f], Hc + st["descent"] * until)

    # heel lift per stance frame: what the body height needs (push-off) and at least a ramp
    # to heel_min by toe-off; once lifted it only rises until the toe leaves the ground
    lift = {s: np.zeros(F) for s in "lr"}
    n_st = int(round(st["stance"] * F))
    for s in "lr":
        start = int(round(phase_frames[s]))
        run_max = 0.0
        for k in range(n_st + 1):
            f = (start + k) % F
            q = k / (st["stance"] * F)
            x = st["step"] / 2 - st["step"] * q - st["foot_back"]
            flat = st["ankle_h"] + np.sqrt(max(d_soft ** 2 - x ** 2, 0.0))
            need = max(0.0, H[f] - flat, heel_ramp(q))
            run_max = max(run_max, need) if q > 0.4 else need
            lift[s][f] = run_max

    def stance_pose(s, f):
        p = (f / F + phase[s]) % 1
        a, _ = ankle_body(p, st)
        root = np.array([0.0, H[f]])
        knee, ank = ik(root, root * [1, 0] + a + np.array([0.0, lift[s][f]]))
        return leg_angles(root, knee, ank)

    swing_ends = {}
    for s in "lr":
        planted = [ankle_body((f / F + phase[s]) % 1, st)[1] for f in range(F)]
        last = next(f for f in range(F) if planted[f] and not planted[(f + 1) % F])   # toe-off
        first = next(f for f in range(F) if planted[f] and not planted[(f - 1) % F])  # heel strike
        swing_ends[s] = (stance_pose(s, last), stance_pose(s, first))

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
        tilt_f = st["hip_tilt"] * np.cos(2 * np.pi * 2 * (f / F - st["stance"] * 0.9))
        lean = np.radians(st["lean"] - st["chest_counter"] * tilt_f)       # chest counters the hips
        J["chest"][f] = root + 0.55 * np.array([np.sin(lean), np.cos(lean)])
        J["neck"][f] = root + 0.75 * np.array([np.sin(lean), np.cos(lean)])
        # head nod: dips forward just after each 'down' (lags the body by ~0.05 of a cycle)
        nod = np.radians(st["head_nod"] * max(0.0, np.cos(2 * np.pi * 2 * (f / F - st["load_peak"] * st["stance"] - 0.05))))
        J["head_top"][f] = J["neck"][f] + 0.35 * np.array([np.sin(nod + 0.057), np.cos(nod + 0.057)])
        for s in "lr":
            p = (f / F + phase[s]) % 1
            a, planted = ankle_body(p, st)
            lf = lift[s][f] if planted else 0.0
            hip = root                                           # H is the hip height
            if planted:
                a = a + np.array([0.0, lf])                     # push-off: heel up
                knee, ank = ik(hip, root * [1, 0] + a)
            else:
                q = (p - st["stance"]) / (1 - st["stance"])
                th, k = swing_pose(q, swing_ends[s][0], swing_ends[s][1], st)
                knee = hip + THIGH * np.array([np.sin(np.radians(th)), -np.cos(np.radians(th))])
                ank = knee + SHIN * np.array([np.sin(np.radians(th - k)), -np.cos(np.radians(th - k))])
            J[f"{s}_hip"][f], J[f"{s}_knee"][f], J[f"{s}_ankle"][f] = hip, knee, ank
            # foot pitch: toe up at heel strike, flat, then the heel lift turns the foot about
            # the ball (push-off), swing carries it round to the next heel strike
            toe_off = -np.degrees(np.arcsin(min(st["heel_min"] / BALL, 0.95)))
            if planted and p < 0.08:
                pitch = st["heel_strike"] * (1 - smooth(p / 0.08))
            elif planted:
                pitch = -np.degrees(np.arcsin(min(lf / BALL, 0.95)))
            else:
                q = (p - st["stance"]) / (1 - st["stance"])
                pitch = toe_off + (st["heel_strike"] - toe_off) * smooth(q)
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
    # pelvis pitch (+ = top tips forward): twice per cycle, peak as each leg pushes off
    push = st["stance"] * 0.9
    angles["hip_tilt"] = [round(st["hip_tilt"] * float(np.cos(2 * np.pi * 2 * (f / F - push))), 2)
                          for f in range(F)]
    r4 = lambda arr: np.round(arr, 4).tolist()
    return {
        "fps": FPS, "frames": F, "view": "side", "loop": True,
        "units": "leg length (hip→ankle); x = forward, y = up from ground",
        "skeleton": list(J), "bones": [list(b) for b in bones],
        "joints": {k: r4(v) for k, v in J.items()},
        "lateral": {f"{s}_{j}": [0.1 if s == "l" else -0.1] * F
                    for s in "lr" for j in ("shoulder", "hip", "knee", "ankle", "wrist")},
        "angles": angles, "root": r4(J["pelvis"]), "contacts": contacts,
        "torso_lean": st["lean"],
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
