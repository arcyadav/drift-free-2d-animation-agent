"""Animator — retargeted motion → rendered clip (spec §10.3, §11).

  * timing: 24 fps animated on twos (a new pose every 2nd frame), like hand-drawn 2D
  * foot lock: from heel-strike to heel-off the ankle is pinned where it landed; a 2-bone
    IK re-bends hip + knee to reach it (knee bends forward). Released when the heel lifts
    (push-off rolls over the toe, as in a real step).
  * secondary motion: ponytail (+ optional front strand) on damped springs driven by
    the head's up/down motion,
    trailing slightly while walking
  * camera follows her; ground marks scroll so the walk reads as travel; soft shadow
  * report: foot slide (px) before / after the lock → used by the critic

Usage: python -m src.animation.animator --action walk --seconds 4
"""
import argparse
import json

import cv2
import imageio.v2 as imageio
import numpy as np

from src import config
from src.animation import rigid_pose
from src.animation.retarget import Retargeter, load_motion, to_local, vis, wrap
from src.utils.io import save_text

FPS = 24
ON_TWOS = 2
OUT_W, OUT_H = 1280, 720
SCALE = 0.42                 # character px → screen px (she is 1344 px tall on the rig canvas)
GROUND_SCREEN_Y = 650
BG = (246, 241, 232)         # warm cream, close to the source art's background
GROUND = (222, 214, 200)
# hair springs: stiffness k, damping d (per s²), gain (deg per px/s² of head vertical
# acceleration), trail (deg it hangs back while walking). The thin front strand reacts
# faster and less than the heavy ponytail.
SPRINGS = {"ponytail": dict(k=120.0, d=9.0, gain=0.06, trail=4.0),
           "strand_l": dict(k=260.0, d=14.0, gain=0.035, trail=2.5)}
BENDS = [("thigh_l", "shin_l", 110), ("thigh_r", "shin_r", 110)]   # soft knees (band px)
REACH_MARGIN = 0.998         # a locked leg may stretch to this share of its length
HEEL_LIFT_PX = 8             # planted ankle rising more than this = heel-off → release the lock


def two_bone_ik(hip, target, l1, l2, forward_x):
    """knee position for hip → knee → ankle reaching target; knee on the forward side."""
    d_vec = target - hip
    d = np.linalg.norm(d_vec)
    d = np.clip(d, abs(l1 - l2) + 1e-3, l1 + l2 - 1e-3)
    u = d_vec / max(np.linalg.norm(d_vec), 1e-6)
    a = np.arccos(np.clip((l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d), -1, 1))
    best = None
    for sgn in (1, -1):
        c, s = np.cos(sgn * a), np.sin(sgn * a)
        dirk = np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])
        knee = hip + l1 * dirk
        score = knee[0] * forward_x
        if best is None or score > best[0]:
            best = (score, knee)
    knee = best[1]
    ankle = knee + l2 * (target - knee) / max(np.linalg.norm(target - knee), 1e-6)
    return knee, ankle


class Walker:
    def __init__(self, character, action="walk", view="side_left", strands=False, speed=1.0,
                 mirror=False):
        self.speed, self.mirror = speed, mirror          # mirror: walk to the right (view graph)
        self.char_dir = config.CHARACTERS / character
        self.meta, self.imgs = rigid_pose.load_rig(self.char_dir, view, "filled")
        self.rt = Retargeter(self.meta, load_motion(action), facing="left")
        self.P = self.rt.pivot
        self.ground_y = max(p["bbox"][3] for n, p in self.meta["parts"].items() if n.startswith("foot"))
        self.forward_x = -1.0
        self.springs = {n: c for n, c in SPRINGS.items()
                        if n in self.meta["parts"] and (strands or not n.startswith("strand"))}

    def chain_world(self, world, side, offset):
        """FK of one leg → hip, knee, ankle world positions (rig px + root offset)."""
        P, rt = self.P, self.rt
        hip = P[f"thigh_{side}"] + offset
        a1 = np.radians(rt.rest[f"thigh_{side}"] + world.get(f"thigh_{side}", 0))
        knee = hip + rt.thigh_len * np.array([np.cos(a1), -np.sin(a1)])
        a2 = np.radians(rt.rest[f"shin_{side}"] + world.get(f"shin_{side}", 0))
        ankle = knee + rt.shin_len * np.array([np.cos(a2), -np.sin(a2)])
        return hip, knee, ankle

    def poses(self, seconds):
        """per output frame: (local angles, root offset, info). Poses held on twos."""
        n_frames = int(round(seconds * FPS))
        anchors = {"l": None, "r": None}
        released = {"l": False, "r": False}
        slide = {"raw": [], "locked": []}
        prev_ankle = {"l": None, "r": None, "raw_l": None, "raw_r": None}
        spring = {n: [c["trail"], 0.0] for n, c in self.springs.items()}
        head_y = []
        out = []
        dt = ON_TWOS / FPS
        for k in range(0, n_frames, ON_TWOS):
            t = k / FPS * self.speed
            world = self.rt.world_deltas(t)
            off = self.rt.root_offset(t)
            # ---- foot lock ----
            fk, locked = {}, []
            for s in "lr":
                hip, _, ankle = self.chain_world(world, s, off)
                fk[s] = ankle
                planted = self.rt.contact(s, t)
                if planted and anchors[s] is None and not released[s]:
                    # land where the motion puts the ankle, never below the ground
                    anchors[s] = np.array([ankle[0], min(ankle[1], self.P[f"foot_{s}"][1])])
                if planted and anchors[s] is not None and ankle[1] < anchors[s][1] - HEEL_LIFT_PX:
                    anchors[s], released[s] = None, True        # heel lifts: push off on the toe
                if not planted:
                    anchors[s], released[s] = None, False
                if planted and anchors[s] is not None:
                    locked.append(s)
            # pelvis drop: if a locked foot is just out of the leg's reach, lower the body
            # a little (as game engines do) instead of letting the foot slip
            reach = (self.rt.thigh_len + self.rt.shin_len) * REACH_MARGIN
            drop = 0.0
            for s in locked:
                hip = self.P[f"thigh_{s}"] + off
                dx = anchors[s][0] - hip[0]
                need = np.sqrt(max(reach ** 2 - dx ** 2, 0.0))
                drop = max(drop, (anchors[s][1] - hip[1]) - need)
            off = off + np.array([0.0, max(drop, 0.0)])
            for s in "lr":
                if s in locked:
                    hip = self.P[f"thigh_{s}"] + off
                    if prev_ankle[f"raw_{s}"] is not None:
                        slide["raw"].append(float(np.linalg.norm(fk[s] - prev_ankle[f"raw_{s}"])))
                    knee, ank = two_bone_ik(hip, anchors[s], self.rt.thigh_len, self.rt.shin_len,
                                            self.forward_x)
                    world[f"thigh_{s}"] = wrap(vis(knee - hip) - self.rt.rest[f"thigh_{s}"])
                    world[f"shin_{s}"] = wrap(vis(ank - knee) - self.rt.rest[f"shin_{s}"])
                    if prev_ankle[s] is not None:
                        slide["locked"].append(float(np.linalg.norm(ank - prev_ankle[s])))
                    prev_ankle[s] = ank
                    prev_ankle[f"raw_{s}"] = fk[s]
                else:
                    prev_ankle[s] = prev_ankle[f"raw_{s}"] = None
            # ---- hair springs (driven by vertical acceleration of the head) ----
            head_y.append(off[1])
            acc = 0.0
            if len(head_y) >= 3:
                acc = (head_y[-1] - 2 * head_y[-2] + head_y[-3]) / dt ** 2
            for n, c in self.springs.items():
                st = spring[n]
                for _ in range(4):                               # sub-steps for stability
                    h = dt / 4
                    a = -c["k"] * (st[0] - c["trail"]) - c["d"] * st[1] + c["gain"] * acc
                    st[1] += a * h
                    st[0] += st[1] * h
                world[n] = world.get("head", 0) + st[0]
            local = to_local(self.meta, world)
            for _ in range(ON_TWOS):
                if len(out) < n_frames:
                    out.append((local, off.copy()))
        return out, slide

    def render(self, seconds, out_path):
        frames, slide = self.poses(seconds)
        start = np.array(self.P["pelvis"])
        writer = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8,
                                    macro_block_size=8, ffmpeg_log_level="error")
        for local, off in frames:
            body_x = start[0] + off[0]
            cam_x = OUT_W * 0.5 - body_x * SCALE              # camera keeps her centred
            cam_y = GROUND_SCREEN_Y - self.ground_y * SCALE
            cam = np.array([[SCALE, 0, cam_x], [0, SCALE, cam_y], [0, 0, 1]])
            canvas = self.background(cam_x, body_x)
            root = np.array([[1, 0, off[0]], [0, 1, off[1]], [0, 0, 1]])
            img = rigid_pose.render(self.meta, self.imgs, local, root=root, camera=cam,
                                    size=(OUT_W, OUT_H), canvas=canvas, bends=BENDS)
            writer.append_data(img[:, ::-1] if self.mirror else img)
        writer.close()
        rep = {"frames": len(frames), "fps": FPS, "on_twos": True, "seconds": seconds,
               "foot_slide_px": {k: {"mean": round(float(np.mean(v)), 2) if v else 0.0,
                                     "max": round(float(np.max(v)), 2) if v else 0.0}
                                 for k, v in slide.items()}}
        return rep

    def background(self, cam_x, body_x):
        return background(cam_x, body_x)


def background(cam_x, body_x, ground_marks=True):
    """cream backdrop, ground strip, scrolling ground marks, soft contact shadow."""
    img = np.empty((OUT_H, OUT_W, 3), np.uint8)
    img[:] = BG
    cv2.rectangle(img, (0, GROUND_SCREEN_Y), (OUT_W, OUT_H), GROUND, -1)
    if ground_marks:                      # every 160 world px, scrolling with the camera
        step = 160
        x0 = int(np.floor((-cam_x) / (step * SCALE))) - 1
        for i in range(x0, x0 + int(OUT_W / (step * SCALE)) + 3):
            sx = int(i * step * SCALE + cam_x)
            cv2.line(img, (sx, GROUND_SCREEN_Y + 18), (sx + 26, GROUND_SCREEN_Y + 18), (205, 196, 180), 3)
    sx = int(body_x * SCALE + cam_x)
    sh = np.zeros_like(img, np.float32)
    cv2.ellipse(sh, (sx, GROUND_SCREEN_Y), (int(110 * SCALE * 2.2), 12), 0, 0, 360, (1, 1, 1), -1)
    sh = cv2.GaussianBlur(sh, (0, 0), 6)
    return (img * (1 - 0.18 * sh)).astype(np.uint8)


def run(character="protagonist", action="walk", seconds=4.0, out=None, strands=False):
    out = out or config.ROOT / "outputs" / f"{action}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = Walker(character, action, strands=strands).render(seconds, str(out))
    rep["strands"] = strands
    rep.update({"character": character, "action": action, "file": str(out.name)})
    save_text(json.dumps(rep, indent=2), out.with_suffix(".json"))
    print(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="protagonist")
    ap.add_argument("--action", default="walk")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--strands", action="store_true", help="front hair strand sways on a spring (off by default)")
    a = ap.parse_args()
    run(a.character, a.action, a.seconds, strands=a.strands)
