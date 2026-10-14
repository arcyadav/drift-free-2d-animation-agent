"""Mixamo (Collada .dae) → action library entry (spec §8.2, §9.1).

Mixamo gives a 3D skeleton: per joint, a local 4x4 matrix per frame. We
  1. run forward kinematics → 3D joint positions per frame
  2. keep the 2D plane the action is seen in (side view: forward + up, drop left/right)
     but keep left/right as `lateral` → which limb is nearer the camera (layer order)
  3. normalise by leg length (hip → ankle), so any character can reuse it
  4. store bone angles (what the Retargeter applies), root motion and foot contacts
     (planted foot = low + not moving forward → foot lock in the Animator)
Only motion is stored, never images.

Usage: python -m src.animation.mixamo_import assets/motion/mixamo/Walking.zip --action walk
"""
import argparse
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from src import config
from src.utils.io import save_text

# our standard skeleton ← Mixamo joint names (fingers dropped)
JOINTS = {
    "pelvis": "Hips", "spine": "Spine", "chest": "Spine2", "neck": "Neck",
    "head": "Head", "head_top": "HeadTop_End",
    "l_shoulder": "LeftArm", "l_elbow": "LeftForeArm", "l_wrist": "LeftHand",
    "r_shoulder": "RightArm", "r_elbow": "RightForeArm", "r_wrist": "RightHand",
    "l_hip": "LeftUpLeg", "l_knee": "LeftLeg", "l_ankle": "LeftFoot", "l_toe": "LeftToeBase",
    "l_toe_end": "LeftToe_End",
    "r_hip": "RightUpLeg", "r_knee": "RightLeg", "r_ankle": "RightFoot", "r_toe": "RightToeBase",
    "r_toe_end": "RightToe_End",
}
BONES = [  # (parent joint, child joint) — angle of each is stored per frame
    ("pelvis", "chest"), ("chest", "neck"), ("neck", "head_top"),
    ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
    ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
    ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("l_ankle", "l_toe"),
    ("r_hip", "r_knee"), ("r_knee", "r_ankle"), ("r_ankle", "r_toe"),
]
CONTACT_HEIGHT = 0.04   # foot within 4% of leg length of the ground ...
CONTACT_SPEED = 0.012   # ... and moving < 1.2% of leg length per frame (world space)
DUP_TOL = 1e-3          # last frame == first frame (Mixamo cycles repeat frame 0)


def _floats(el):
    return np.array(el.text.split(), float)


def load_dae(path):
    """→ joint name → (F,4,4) local matrices, parent map, fps."""
    p = Path(path)
    if p.suffix == ".zip":
        z = zipfile.ZipFile(p)
        dae = next(n for n in z.namelist() if n.lower().endswith(".dae"))
        root = ET.fromstring(z.read(dae))
    else:
        root = ET.parse(p).getroot()
    ns = {"c": root.tag.split("}")[0].strip("{")}
    if root.find(".//c:asset/c:up_axis", ns).text != "Y_UP":
        raise ValueError("expected Y_UP Collada from Mixamo")

    rest, parent = {}, {}

    def walk(node, par):
        name = node.get("id", "").replace("mixamorig_", "")
        rest[name] = _floats(node.find("c:matrix", ns)).reshape(4, 4)
        parent[name] = par
        for ch in node.findall("c:node", ns):
            walk(ch, name)

    for top in root.find(".//c:library_visual_scenes/c:visual_scene", ns).findall("c:node", ns):
        walk(top, None)

    anim, times = {}, None
    for a in root.findall(".//c:library_animations/c:animation", ns):
        srcs = a.findall("c:source", ns)
        t = _floats(next(s for s in srcs if s.get("id").endswith("input")).find("c:float_array", ns))
        m = _floats(next(s for s in srcs if "output" in s.get("id")).find("c:float_array", ns))
        target = a.find("c:channel", ns).get("target").split("/")[0].replace("mixamorig_", "")
        anim[target] = m.reshape(-1, 4, 4)
        times = t
    F = len(times)
    local = {j: anim.get(j, np.repeat(rest[j][None], F, 0)) for j in rest}
    fps = round(1 / np.median(np.diff(times)))
    return local, parent, F, fps


def world_positions(local, parent, F):
    world = {}

    def W(j):
        if j not in world:
            world[j] = local[j] if parent[j] is None else W(parent[j]) @ local[j]
        return world[j]

    return {j: W(j)[:, :3, 3] for j in local}          # (F,3) x=lateral, y=up, z=forward


def to_action(pos, F, fps, view="side"):
    P = {k: pos[v] for k, v in JOINTS.items()}
    leg = np.mean([np.linalg.norm(P[f"{s}_hip"][0] - P[f"{s}_knee"][0]) +
                   np.linalg.norm(P[f"{s}_knee"][0] - P[f"{s}_ankle"][0]) for s in "lr"])
    ground = min(P[f"{s}_{j}"][:, 1].min() for s in "lr" for j in ("ankle", "toe", "toe_end"))
    origin = np.array([P["pelvis"][0, 0], ground, P["pelvis"][0, 2]])

    # side view plane: (forward, up); lateral kept for near/far
    def plane(p):
        q = (p - origin) / leg
        return np.stack([q[:, 2], q[:, 1]], 1), q[:, 0]

    j2d, lateral = {}, {}
    for k, p in P.items():
        j2d[k], lateral[k] = plane(p)

    # Mixamo repeats frame 0 at the end of a cycle → drop it, or every loop stutters
    rel = lambda f: np.concatenate([j2d[k][f] - j2d["pelvis"][f] for k in j2d])
    stride = float(j2d["pelvis"][-1, 0] - j2d["pelvis"][0, 0])  # forward travel per cycle
    if np.abs(rel(0) - rel(F - 1)).max() < DUP_TOL:
        F -= 1
        j2d = {k: v[:F] for k, v in j2d.items()}
        lateral = {k: v[:F] for k, v in lateral.items()}
    else:
        stride *= F / (F - 1)

    angles = {}
    for a, b in BONES:
        d = j2d[b] - j2d[a]
        angles[f"{a}>{b}"] = np.degrees(np.arctan2(d[:, 1], d[:, 0]))   # 0 = forward, 90 = up

    contacts = {}
    for s in "lr":
        low = np.minimum(j2d[f"{s}_ankle"][:, 1], j2d[f"{s}_toe"][:, 1])
        vel = np.abs(np.gradient(j2d[f"{s}_toe"][:, 0]))
        contacts[s] = ((low < CONTACT_HEIGHT) & (vel < CONTACT_SPEED)).tolist()

    root = j2d["pelvis"]
    # loop seam: last → first frame should be no bigger a jump than a normal frame step
    step = lambda a, b: np.mean([abs(((angles[k][b] - angles[k][a] + 180) % 360) - 180) for k in angles])
    loop_err = float(step(F - 1, 0) / max(np.mean([step(i, i + 1) for i in range(F - 1)]), 1e-6))
    r = lambda arr: np.round(arr, 4).tolist()
    return {
        "fps": fps, "frames": F, "view": view, "loop": True,
        "units": "leg length (hip→ankle); x = forward, y = up from ground",
        "skeleton": list(JOINTS), "bones": [list(b) for b in BONES],
        "joints": {k: r(v) for k, v in j2d.items()},
        "lateral": {k: r(v) for k, v in lateral.items() if k.endswith(("shoulder", "hip", "knee", "ankle", "wrist"))},
        "angles": {k: np.round(v, 2).tolist() for k, v in angles.items()},
        "root": r(root),
        "contacts": contacts,
        "stats": {"leg_cm": round(float(leg), 1), "stride_per_cycle": round(stride, 3), "cycle_frames": F,
                  "loop_seam_ratio": round(loop_err, 2),
                  "contact_frames": {s: int(sum(c)) for s, c in contacts.items()}},
    }


def run(src, action, view="side"):
    local, parent, F, fps = load_dae(src)
    act = to_action(world_positions(local, parent, F), F, fps, view)
    out = config.ROOT / "actions" / action
    out.mkdir(parents=True, exist_ok=True)
    save_text(json.dumps(act), out / "motion.json")
    save_text(json.dumps({"action": action, "view": view, "fps": fps, "frames": act["frames"], "loop": True,
                          "source": "mixamo", "file": Path(src).name, **act["stats"]}, indent=2),
              out / "meta.json")
    save_text(f"# {action}\n\n- Source: Mixamo (Adobe), `{Path(src).name}` — 'In Place' off, "
              f"without skin, {fps} fps.\n- Licence: Mixamo animations are royalty-free for "
              "personal, commercial and non-profit projects (Adobe Mixamo FAQ).\n"
              "- Converted by `src/animation/mixamo_import.py` (3D → side-plane 2D, leg-length units).\n",
              out / "sources.md")
    print(json.dumps(act["stats"], indent=2))
    return act


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--action", required=True)
    ap.add_argument("--view", default="side")
    a = ap.parse_args()
    run(a.src, a.action, a.view)
