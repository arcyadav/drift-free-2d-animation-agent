"""Front-view clips from library motion (wave, …) — spec §9.1, §10.

  * motion = world rotation deltas per part (from video_import), resampled to 24 fps on twos
  * hand swap: the open-hand image replaces the relaxed hand while the motion says 'open'
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

HEAD_TILT = -3.0            # deg toward her left (viewer's right) while the hand is up
SPRINGS = {"ponytail": dict(k=110.0, d=8.0, gain=0.9), "strand_r": dict(k=220.0, d=12.0, gain=0.6)}


class FrontClip:
    def __init__(self, character, action):
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

    def sample(self, t):
        f = min(t * self.m["fps"], self.m["frames"] - 1)
        i0 = int(np.floor(f)); i1 = min(i0 + 1, self.m["frames"] - 1); w = f - i0
        world = {k: (1 - w) * v[i0] + w * v[i1] for k, v in self.m["deltas"].items()}
        hand = {k: v[int(round(f))] for k, v in self.m.get("hand", {}).items()}
        return world, hand

    def frames(self):
        n = int(np.ceil(self.m["frames"] / self.m["fps"] * FPS))
        dt = ON_TWOS / FPS
        spring = {k: [0.0, 0.0] for k in SPRINGS if k in self.meta["parts"]}
        tilt, prev_head = 0.0, [0.0, 0.0]
        out = []
        for k in range(0, n, ON_TWOS):
            world, hand = self.sample(k / FPS)
            up = any(v == "open" for v in hand.values())
            tilt += (HEAD_TILT * up - tilt) * 0.35                  # ease into / out of the tilt
            world["head"] = tilt
            acc = (tilt - 2 * prev_head[-1] + prev_head[-2]) / dt ** 2
            prev_head = [prev_head[-1], tilt]
            for n_, c in SPRINGS.items():
                if n_ not in spring:
                    continue
                st = spring[n_]
                for _ in range(4):
                    h = dt / 4
                    a = -c["k"] * st[0] - c["d"] * st[1] - c["gain"] * acc
                    st[1] += a * h; st[0] += st[1] * h
                world[n_] = tilt + st[0]
            local = to_local(self.meta, world)
            for _ in range(ON_TWOS):
                if len(out) < n:
                    out.append((local, hand))
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
        for local, hand in frames:
            imgs = dict(self.imgs)
            for part, pose in hand.items():
                if pose == "open" and part in self.swaps:
                    imgs[part] = self.swaps[part]
                    opened += 1
            writer.append_data(rigid_pose.render(self.meta, imgs, local, camera=cam,
                                                 size=(OUT_W, OUT_H), canvas=bg))
        writer.close()
        return {"frames": len(frames), "fps": FPS, "on_twos": True, "open_hand_frames": opened}


def run(character="protagonist", action="wave"):
    out = config.ROOT / "outputs" / f"{action}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = FrontClip(character, action).render(str(out))
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
