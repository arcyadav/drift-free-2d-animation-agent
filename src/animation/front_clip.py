"""Front-view clips from library motion (wave, …) — spec §9.1, §10.

  * motion = world rotation deltas per part (from video_import), resampled to 24 fps on twos
  * hand swap: the open-hand image replaces the relaxed hand while the motion says 'open'
  * face: eyes / brows / mouth layers (Face Cutter) ride on the head; expression from a
    simple timeline (smile while the hand is up, else neutral) + automatic blinks
    (half → closed → half, one pose per 'two', every ~3 s with a little randomness)
  * life: ponytail + front strand on damped springs driven by the head; a slight head tilt
    toward the waving hand while it is up (people tilt into a wave)
  * camera: full body, static, same backdrop as the walk

Usage: python -m src.animation.front_clip --action wave
"""
import argparse
import json

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from src import config
from src.animation import rigid_pose
from src.animation.animator import FPS, ON_TWOS, OUT_H, OUT_W, SCALE, GROUND_SCREEN_Y, background
from src.animation.retarget import to_local
from src.utils.io import save_text

FACE_LAYERS = ("eyes", "brows", "mouth")
BLINK_EVERY = (2.6, 3.8)    # s between blinks (random in this range, fixed seed)
WRIST_LAG = 0.08            # s: the hand trails the forearm by this much ...
WRIST_GAIN = 0.9            # ... turning by this share of the forearm's recent swing ...
WRIST_MAX = 18.0            # ... at most this many degrees
# actions with their own face (user's choice for look_down: neutral face, eyes of the sad image)
ACTION_FACE = {"look_down": {"eyes": "sad", "brows": "neutral", "mouth": "neutral"}}
NO_BLINK = {"look_down"}
GAZE_FROM = 2.0                             # deg of head tilt after which the eyes look down                    # blinking from open eyes looks wrong while looking down
HEAD_TILT = -3.0            # deg toward her left (viewer's right) while the hand is up
SPRINGS = {"ponytail": dict(k=110.0, d=8.0, gain=0.9), "strand_r": dict(k=220.0, d=12.0, gain=0.6)}


class FrontClip:
    def __init__(self, character, action, expression=None, speed=1.0, seconds=None):
        self.action, self.expression, self.speed, self.seconds = action, expression, speed, seconds
        self.char = config.CHARACTERS / character
        self.meta, self.imgs = rigid_pose.load_rig(self.char, "front", "filled")
        self.m = json.loads((config.ROOT / "actions" / action / "motion.json").read_text())
        self.swaps = {}
        for part, poses in self.m.get("hand", {}).items():
            side = part.split("_")[-1]
            p = self.char / "hands" / "front" / f"open_{side}.png"
            if p.exists():
                self.swaps[part] = np.asarray(Image.open(p).convert("RGBA"))
        self.ground_y = max(p["bbox"][3] for n, p in self.meta["parts"].items() if n.startswith("foot"))
        self.recipes = json.loads((config.ROOT / "expressions" / "recipes.json").read_text())
        face_dir = self.char / "faces" / "front"
        self.face = {}
        if face_dir.exists():
            for layer in FACE_LAYERS:
                for p in (face_dir / layer).glob("*.png"):
                    self.face[(layer, p.stem)] = np.asarray(Image.open(p).convert("RGBA"))
            # face layers ride on the head, drawn right above it (below the hair)
            order = list(self.meta["layer_order"])
            i = order.index("head") + 1
            head = self.meta["parts"]["head"]
            for layer in FACE_LAYERS:
                name = f"face_{layer}"
                order.insert(i, name); i += 1
                self.meta["parts"][name] = {"id": 0, "parent": "head", "pivot_joint": head["pivot_joint"],
                                            "pivot": head["pivot"]}
            self.meta["layer_order"] = order

    def sample(self, t):
        f = t * self.speed * self.m["fps"]
        f = f % self.m["frames"] if self.m.get("loop") else min(f, self.m["frames"] - 1)
        i0 = int(np.floor(f)); i1 = min(i0 + 1, self.m["frames"] - 1); w = f - i0
        world = {k: (1 - w) * v[i0] + w * v[i1] for k, v in self.m["deltas"].items()}
        hand = {k: v[int(round(f))] for k, v in self.m.get("hand", {}).items()}
        return world, hand

    def blinks(self, n):
        """frame index → eye image of a blink (or None)."""
        rng = np.random.default_rng(7)
        out = [None] * n
        t = rng.uniform(0.8, 1.6)
        seq = self.recipes.get("blink", ["eyes_half", "eyes_closed", "eyes_half"])
        while t * FPS < n:
            k0 = int(t * FPS) // ON_TWOS * ON_TWOS
            for j, e in enumerate(seq):
                for d in range(ON_TWOS):
                    if k0 + j * ON_TWOS + d < n:
                        out[k0 + j * ON_TWOS + d] = e
            t += rng.uniform(*BLINK_EVERY)
        return out

    def frames(self):
        dur = self.seconds or self.m["frames"] / self.m["fps"] / self.speed
        n = int(np.ceil(dur * FPS))
        blink = self.blinks(n) if self.action not in NO_BLINK else [None] * n
        dt = ON_TWOS / FPS
        spring = {k: [0.0, 0.0] for k in SPRINGS if k in self.meta["parts"]}
        tilt, prev_head = 0.0, [0.0, 0.0]
        out = []
        for k in range(0, n, ON_TWOS):
            world, hand = self.sample(k / FPS)
            up = any(v == "open" for v in hand.values())
            # wrist follow-through: the open hand trails the forearm's swing and flicks at
            # each turn (hand angle ~ minus the forearm's angular velocity), only while waving
            for part, pose in hand.items():
                fore = part.replace("hand", "forearm")
                if pose == "open" and fore in world:
                    prev, _ = self.sample(max(k / FPS - WRIST_LAG, 0))
                    vel = world[fore] - prev.get(fore, world[fore])
                    world[part] = world[fore] + float(np.clip(-WRIST_GAIN * vel, -WRIST_MAX, WRIST_MAX))
            tilt += (HEAD_TILT * up - tilt) * 0.35                  # ease into / out of the tilt
            tilt_head = tilt + world.get("head", 0.0)                # action's own head motion
            world["head"] = tilt_head
            acc = (tilt_head - 2 * prev_head[-1] + prev_head[-2]) / dt ** 2
            prev_head = [prev_head[-1], tilt_head]
            for n_, c in SPRINGS.items():
                if n_ not in spring:
                    continue
                st = spring[n_]
                for _ in range(4):
                    h = dt / 4
                    a = -c["k"] * st[0] - c["d"] * st[1] - c["gain"] * acc
                    st[1] += a * h; st[0] += st[1] * h
                world[n_] = tilt_head + st[0]
            drop = float(world.pop("head_drop", 0.0))                # optional head dip (px)
            tilted = abs(world.get("head", 0.0) - tilt) >= GAZE_FROM
            local = to_local(self.meta, world)
            # a smiling wave smiles while the hand is up; otherwise the planned expression
            if self.expression in (None, "smile") and self.m.get("hand"):
                expr = "smile" if up else "neutral"
            else:
                expr = self.expression or "neutral"
            for _ in range(ON_TWOS):
                if len(out) < n:
                    out.append((local, hand, expr, blink[len(out)], (drop, tilted)))
        return out

    def render(self, out_path):
        cam_x = OUT_W * 0.5 - config.CENTER_X * SCALE
        cam_y = GROUND_SCREEN_Y - self.ground_y * SCALE
        cam = np.array([[SCALE, 0, cam_x], [0, SCALE, cam_y], [0, 0, 1]])
        bg = background(cam_x, config.CENTER_X, ground_marks=False)
        writer = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8,
                                    macro_block_size=8, ffmpeg_log_level="error")
        frames = self.frames()
        opened = 0
        blinks = 0
        for local, hand, expr, blink, (drop, tilted) in frames:
            imgs = dict(self.imgs)
            for part, pose in hand.items():
                if pose == "open" and part in self.swaps:
                    imgs[part] = self.swaps[part]
                    opened += 1
            recipe = self.recipes["recipes"].get(expr, self.recipes["recipes"]["neutral"])
            for layer in FACE_LAYERS:
                src = blink if (layer == "eyes" and blink) else recipe[layer]
                if self.action in ACTION_FACE and not (layer == "eyes" and blink):
                    face = ACTION_FACE[self.action]
                    if layer != "eyes" or tilted:
                        src = face[layer]
                if (layer, src) in self.face:
                    imgs[f"face_{layer}"] = self.face[(layer, src)]
            blinks += blink == "eyes_closed"
            writer.append_data(rigid_pose.render(self.meta, imgs, local, camera=cam,
                                                 offsets={"head": (0.0, drop)} if drop else None,
                                                 size=(OUT_W, OUT_H), canvas=bg))
        writer.close()
        return {"frames": len(frames), "fps": FPS, "on_twos": True, "open_hand_frames": opened,
                "blink_closed_frames": int(blinks)}


def run(character="protagonist", action="wave", expression=None, speed=1.0, seconds=None, out=None):
    out = out or config.ROOT / "outputs" / f"{action}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = FrontClip(character, action, expression, speed, seconds).render(str(out))
    rep.update({"character": character, "action": action})
    save_text(json.dumps(rep, indent=2), out.with_suffix(".json"))
    print(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", default="protagonist")
    ap.add_argument("--action", default="wave")
    a = ap.parse_args()
    run(a.character, a.action)
