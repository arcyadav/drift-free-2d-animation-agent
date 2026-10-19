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
from src.animation.animator import (FPS, ON_TWOS, OUT_H, OUT_W, SCALE, GROUND_SCREEN_Y, background,
                                    plain_background, shot_camera)
from src.animation.retarget import to_local
from src.animation.seal import KEY, drawn_holes, id_layers, seal as seal_cracks, to_screen
from src.utils.io import save_text

FACE_LAYERS = ("eyes", "brows", "mouth")
BLINK_EVERY = (2.6, 3.8)    # s between blinks (random in this range, fixed seed)
WRIST_LAG = 0.08            # s: the hand trails the forearm by this much ...
WRIST_GAIN = 0.9            # ... turning by this share of the forearm's recent swing ...
WRIST_MAX = 18.0            # ... at most this many degrees
# actions with their own face (user's choice for look_down: neutral face, eyes of the sad image)
# (user's reads: sad eyes + tilt = quietly looking down; eyes_down + tilt = thinking "hmm")
ACTION_FACE = {"look_down": {"eyes": "sad", "brows": "neutral", "mouth": "neutral"}}
ACTION_FACE_BY_EXPR = {("look_down", "thoughtful"): {"eyes": "eyes_down", "brows": "neutral", "mouth": "neutral"},
                       # a sad request shows the full sad face (brows + downturned mouth)
                       ("look_down", "sad"): {"eyes": "sad", "brows": "sad", "mouth": "sad"},
                       ("look_down", "worried"): {"eyes": "worried", "brows": "worried", "mouth": "worried"}}
NO_BLINK = {"look_down"}
GAZE_FROM = 2.0                             # deg of head tilt after which the eyes look down                    # blinking from open eyes looks wrong while looking down
HEAD_TILT = -3.0            # deg toward her left (viewer's right) while the hand is up
SPRINGS = {"ponytail": dict(k=110.0, d=8.0, gain=0.9), "strand_r": dict(k=220.0, d=12.0, gain=0.6)}


class FrontClip:
    def __init__(self, character, action, expression=None, speed=1.0, seconds=None, shot="full",
                 amplitude=1.0, seal=False):
        self.action, self.expression, self.speed, self.seconds = action, expression, speed, seconds
        self.amplitude = amplitude                   # motion scale
        self.seal = seal                             # fill cracks between parts (the critic's fix)
        self.shot = shot
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
        own = self.char / "faces" / "front" / "face.json"
        if own.exists() and "recipes" in json.loads(own.read_text()):     # per-character availability
            self.recipes["recipes"] = json.loads(own.read_text())["recipes"]
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
        world = {k: self.amplitude * ((1 - w) * v[i0] + w * v[i1]) for k, v in self.m["deltas"].items()}
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
                expr = self.expression or self.m.get("default_expression") or "neutral"
                if "react_at" in self.m and k / FPS * self.speed < self.m["react_at"]:
                    expr = "neutral"                                  # before the reaction
            for _ in range(ON_TWOS):
                if len(out) < n:
                    out.append((local, hand, expr, blink[len(out)], (drop, tilted)))
        return out

    def images(self, frames, key=None, step=1, stats=None, with_ids=False):
        """rendered frames; key = a flat colour instead of the backdrop (the critic's crack check);
        with_ids: yield (frame, part-ID frame, rest-pose holes on screen)"""
        cam, _, cam_x = shot_camera(self.shot, config.CENTER_X, self.ground_y)
        drawn = None
        if self.seal or with_ids:
            rest_imgs = dict(self.imgs, **{f"face_{L}": self.face[(L, "neutral")] for L in FACE_LAYERS
                                           if (L, "neutral") in self.face})
            drawn = to_screen(drawn_holes(self.meta, rest_imgs), cam, (OUT_W, OUT_H))
        if key is not None:
            bg = np.full((OUT_H, OUT_W, 3), key, np.uint8)
        elif self.shot == "full":
            bg = background(cam_x, config.CENTER_X, ground_marks=False)
        else:
            bg = plain_background()
        stats = stats if stats is not None else {}
        stats.update(opened=0, blinks=0, shown=set())
        for local, hand, expr, blink, (drop, tilted) in frames[::step]:
            imgs = dict(self.imgs)
            for part, pose in hand.items():
                if pose == "open" and part in self.swaps:
                    imgs[part] = self.swaps[part]
                    stats["opened"] += 1
            recipe = self.recipes["recipes"].get(expr, self.recipes["recipes"]["neutral"])
            for layer in FACE_LAYERS:
                src = blink if (layer == "eyes" and blink) else recipe[layer]
                if self.action in ACTION_FACE and not (layer == "eyes" and blink):
                    face = ACTION_FACE_BY_EXPR.get((self.action, self.expression), ACTION_FACE[self.action])
                    src = face[layer] if tilted else "neutral"      # face changes as the head tilts
                if (layer, src) not in self.face:                    # this character lacks it
                    src = "neutral"
                stats["shown"].add((layer, src))
                imgs[f"face_{layer}"] = self.face[(layer, src)]
            stats["blinks"] += blink == "eyes_closed"
            def draw(cv, layers=None):
                return rigid_pose.render(self.meta, layers or imgs, local, camera=cam,
                                         offsets={"head": (0.0, drop)} if drop else None,
                                         size=(OUT_W, OUT_H), canvas=cv)
            img = draw(bg)
            ids = None
            if self.seal or with_ids:
                keyed = img if key == KEY else draw(np.full((OUT_H, OUT_W, 3), KEY, np.uint8))
                ids = draw(np.full((OUT_H, OUT_W, 3), KEY, np.uint8), id_layers(imgs))
                if self.seal:
                    img = seal_cracks(img, keyed, cam[0, 0] / SCALE, ids, drawn)
            yield (img, ids, drawn) if with_ids else img

    def keyed(self, key, step=ON_TWOS):
        return list(self.images(self.frames(), key=key, step=step))

    def render(self, out_path):
        writer = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=8,
                                    macro_block_size=8, ffmpeg_log_level="error")
        frames = self.frames()
        st = {}
        for img in self.images(frames, stats=st):
            writer.append_data(img)
        writer.close()
        return {"frames": len(frames), "fps": FPS, "on_twos": True, "open_hand_frames": st["opened"],
                "blink_closed_frames": int(st["blinks"]),
                "face_layers_shown": sorted(f"{l}:{s}" for l, s in st["shown"])}


def run(character="protagonist", action="wave", expression=None, speed=1.0, seconds=None, out=None,
        shot="full"):
    out = out or config.ROOT / "outputs" / f"{action}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = FrontClip(character, action, expression, speed, seconds, shot).render(str(out))
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
