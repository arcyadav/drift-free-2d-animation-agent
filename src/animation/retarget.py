"""Retargeter — action-library motion → this character's rig angles (spec §8.2, §11).

Motion is stored as bone angles in a side plane (x = forward, y = up) in leg-length units.
For a rig view we
  * limbs (thigh, shin, upper arm, forearm): copy the motion's absolute bone direction —
    a standing human and her drawing hang the limbs the same way
  * foot, torso, head: copy only the change around a reference (foot: its angle while
    planted; torso/head: cycle mean) — Mixamo's 'flat foot' and slight lean differ from
    the drawing, and absolute copies would tilt her sneaker or bend her over
  * root: pelvis travel × her leg length (px); bob relative to the highest pelvis point
    (= leg straight under the body, as in the drawing)
Angles are world deltas in screen degrees (positive = counter-clockwise as seen), then
turned into parent-relative angles for the renderer.
"""
import json

import numpy as np

from src import config

HEAD_KEEP_LEAN = 0.3   # share of the torso lean the head keeps (eyes stay on the road)

# rig part → (motion bone, mode)
BONE_MAP_SIDE = {
    "thigh": ("{s}_hip>{s}_knee", "abs"), "shin": ("{s}_knee>{s}_ankle", "abs"),
    "foot": ("{s}_ankle>{s}_toe", "rel_contact"),
    "upper_arm": ("{s}_shoulder>{s}_elbow", "abs"), "forearm": ("{s}_elbow>{s}_wrist", "abs"),
}
# rig rest bone of a part = its pivot → its child's pivot
REST_CHILD = {"thigh": "shin", "shin": "foot", "upper_arm": "forearm", "forearm": "hand"}


def load_motion(action):
    return json.loads((config.ROOT / "actions" / action / "motion.json").read_text())


def vis(v):
    """screen vector (x right, y down) → visual angle in degrees (counter-clockwise +)."""
    return np.degrees(np.arctan2(-v[..., 1], v[..., 0]))


def motion_to_screen(deg, facing):
    """motion bone angle (0 = forward, 90 = up) → visual screen angle for a facing."""
    return 180 - deg if facing == "left" else deg


def wrap(a):
    return (a + 180) % 360 - 180


class Retargeter:
    def __init__(self, meta, motion, facing="left"):
        self.meta, self.m, self.facing = meta, motion, facing
        P = {n: np.array(p["pivot"]) for n, p in meta["parts"].items()}
        self.pivot = P
        self.rest = {}
        for s in "lr":
            for limb, child in REST_CHILD.items():
                self.rest[f"{limb}_{s}"] = vis(P[f"{child}_{s}"] - P[f"{limb}_{s}"])
        self.thigh_len = float(np.linalg.norm(P["shin_l"] - P["thigh_l"]))
        self.shin_len = float(np.linalg.norm(P["foot_l"] - P["shin_l"]))
        self.leg_px = self.thigh_len + self.shin_len
        A = {k: np.unwrap(np.radians(np.array(v))) for k, v in motion["angles"].items()}
        self.ang = {k: np.degrees(v) for k, v in A.items()}
        self.F = motion["frames"]
        c = {s: np.array(motion["contacts"][s]) for s in "lr"}
        self.contacts = c
        self.foot_ref = {s: np.median(self.ang[f"{s}_ankle>{s}_toe"][c[s]]) for s in "lr"}
        self.root = np.array(motion["root"])
        self.stride = motion.get("stats", {}).get("stride_per_cycle",
                                                  float(self.root[-1, 0] - self.root[0, 0]))
        self.up_ref = self.root[:, 1].max()

    # ---- sampling (looping, linear interpolation between motion frames) ----
    def _sample(self, arr, f):
        i0 = int(np.floor(f)) % self.F
        i1 = (i0 + 1) % self.F
        w = f - np.floor(f)
        a0, a1 = arr[i0], arr[i1]
        if arr.ndim == 1:
            return a0 + w * wrap(a1 - a0)
        return a0 + w * (a1 - a0)

    def frame_at(self, t):
        return t * self.m["fps"]

    def contact(self, side, t):
        return bool(self.contacts[side][int(round(self.frame_at(t))) % self.F])

    def world_deltas(self, t):
        """part → world rotation (visual deg) relative to the drawing, at time t (s)."""
        f = self.frame_at(t)
        out = {}
        for s in "lr":
            for limb, (bone, mode) in BONE_MAP_SIDE.items():
                a = self._sample(self.ang[bone.format(s=s)], f)
                if mode == "abs":
                    out[f"{limb}_{s}"] = wrap(motion_to_screen(a, self.facing) - self.rest[f"{limb}_{s}"])
                else:                                            # foot: change from planted angle
                    d = a - self.foot_ref[s]
                    out[f"{limb}_{s}"] = -d if self.facing == "left" else d
        spine = self._sample(self.ang["pelvis>chest"], f) - self.ang["pelvis>chest"].mean()
        head = self._sample(self.ang["neck>head_top"], f) - self.ang["neck>head_top"].mean()
        sgn = -1 if self.facing == "left" else 1
        out["torso"] = sgn * spine
        if "hip_tilt" in self.ang:                              # pelvis pitch, + = tips forward
            out["pelvis"] = -sgn * self._sample(self.ang["hip_tilt"], f)
        out["head"] = out["torso"] + sgn * head * 0.5          # head steadier than the spine
        lean = self.m.get("torso_lean", 0.0)                    # constant forward lean (deg)
        if lean:
            out["torso"] += -sgn * lean                          # + forward = toward the facing
            out["head"] += -sgn * lean * (1 - HEAD_KEEP_LEAN)    # head stays mostly level
        pitch = self.m.get("head_pitch", 0.0)                   # mood: head down (+) / up (-)
        if pitch:
            out["head"] += -sgn * pitch
        return out

    def root_offset(self, t):
        """pelvis displacement in px (screen) at time t, including whole cycles walked."""
        f = self.frame_at(t)
        cycles = np.floor(f / self.F)
        i0 = int(np.floor(f)) % self.F
        w = f - np.floor(f)
        nxt = self.root[(i0 + 1) % self.F].copy()
        if i0 == self.F - 1:
            nxt[0] += self.stride                  # next cycle starts one stride further on
        fwd, up = self.root[i0] + w * (nxt - self.root[i0])
        fwd += cycles * self.stride - self.root[0, 0]
        dx = -fwd * self.leg_px if self.facing == "left" else fwd * self.leg_px
        dy = -(up - self.up_ref) * self.leg_px
        return np.array([dx, dy])


def to_local(meta, world):
    """world deltas → parent-relative angles (what rigid_pose.render expects)."""
    local = {}
    for n, p in meta["parts"].items():
        par = p["parent"]
        pw = 0.0
        while par is not None:          # nearest ancestor that has a world delta
            if par in world:
                pw = world[par]
                break
            par = meta["parts"][par]["parent"]
        if n in world:
            local[n] = wrap(world[n] - pw)
    return local
