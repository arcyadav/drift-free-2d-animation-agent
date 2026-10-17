"""Phone video → action library entry (spec §9.1 'own video'; §8.2: motion only).

  1. RTMPose (rtmlib, CPU) on every frame → 2D joints; most confident person
  2. smooth: low-confidence joints interpolated, then a Savitzky-Golay filter (no jitter)
  3. bone angles in screen space for the chosen arm (upper arm, forearm), minus the
     shoulder line's own rotation (body sway is not arm motion)
  4. stored RELATIVE to the person's own rest pose (the still start of the clip):
     the character's arm rests differently, and the deltas transfer anyway
  5. hand pose per frame: 'open' while the forearm points up (the hand swap image)
No frames, crops or images of the video are written anywhere — only angles.

Usage: python -m src.animation.video_import assets/motion/video/wave.mp4 --action wave --side l
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

from src import config
from src.onboarding.keypoints import COCO, MIN_SCORE, detector
from src.utils.io import save_text

REST_S = 0.8            # the first 0.8 s are the rest pose (person stands still)
MOVE_THRESH = 0.03      # wrist speed (shoulder widths / frame) that counts as moving
SMOOTH_WIN = 9          # Savitzky-Golay window (frames) ...
SMOOTH_ORDER = 2        # ... and polynomial order
OPEN_FOREARM = 15.0     # forearm at least this many degrees above horizontal → open hand
PAD_S = 0.4             # rest kept before / after the motion


def vis(v):
    return np.degrees(np.arctan2(-v[..., 1], v[..., 0]))


def wrap(a):
    return (a + 180) % 360 - 180


def track(video):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    K, S = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kps, scores = detector()(frame)
        if len(kps) == 0:
            K.append(np.full((17, 2), np.nan)); S.append(np.zeros(17))
            continue
        i = int(np.argmax(scores.mean(1)))
        K.append(kps[i]); S.append(scores[i])
    cap.release()
    return np.array(K, float), np.array(S, float), float(fps)


def clean(K, S):
    K = K.copy()
    K[S < MIN_SCORE] = np.nan
    t = np.arange(len(K))
    for j in range(K.shape[1]):
        for c in range(2):
            v = K[:, j, c]
            ok = ~np.isnan(v)
            if ok.sum() >= 2:
                K[:, j, c] = np.interp(t, t[ok], v[ok])
            else:                                   # never seen (e.g. legs out of frame): unused
                K[:, j, c] = 0.0
    win = min(SMOOTH_WIN, len(K) - (1 - len(K) % 2))
    return savgol_filter(K, win, SMOOTH_ORDER, axis=0)


def run(video, action, side="l"):
    cache = config.ROOT / "outputs" / ".cache" / f"{Path(video).stem}_joints.npz"   # numbers only
    if cache.exists() and cache.stat().st_mtime > Path(video).stat().st_mtime:
        z = np.load(cache)
        K, S, fps = z["K"], z["S"], float(z["fps"])
    else:
        K, S, fps = track(video)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, K=K, S=S, fps=fps)
    K = clean(K, S)
    J = {n: K[:, i] for i, n in enumerate(COCO)}
    # which arm actually moves (a mirrored phone video swaps left / right)
    moved = {s: np.ptp(J[f"{s}_wrist"][:, 1]) for s in "lr"}
    used = max(moved, key=moved.get)
    mirrored = used != side
    sw = np.linalg.norm(J["l_shoulder"] - J["r_shoulder"], axis=1).mean()
    sh, el, wr = J[f"{used}_shoulder"], J[f"{used}_elbow"], J[f"{used}_wrist"]
    body = vis(J["l_shoulder"] - J["r_shoulder"])                   # shoulder line roll
    upper = vis(el - sh) - (body - body[0])
    fore = vis(wr - el) - (body - body[0])
    if mirrored:                                                   # mirror back to `side`
        upper, fore = 180 - upper, 180 - fore

    n_rest = max(3, int(REST_S * fps))
    ref_u, ref_f = np.median(upper[:n_rest]), np.median(fore[:n_rest])
    d_up = wrap(upper - ref_u)
    d_fo = wrap(fore - ref_f)

    speed = np.r_[0, np.linalg.norm(np.diff(wr, axis=0), axis=1)] / sw
    moving = np.where(speed > MOVE_THRESH)[0]
    pad = int(PAD_S * fps)
    f0 = max(0, moving.min() - pad) if len(moving) else 0
    f1 = min(len(K), moving.max() + pad) if len(moving) else len(K)
    open_hand = (fore > OPEN_FOREARM) & (fore < 180 - OPEN_FOREARM) if not mirrored else \
        (180 - fore > OPEN_FOREARM) & (180 - fore < 180 - OPEN_FOREARM)

    r2 = lambda a: np.round(a[f0:f1], 2).tolist()
    act = {
        "fps": fps, "frames": int(f1 - f0), "view": "front", "loop": False,
        "units": "world rotation (deg, screen, counter-clockwise +) relative to the rest pose",
        "deltas": {f"upper_arm_{side}": r2(d_up), f"forearm_{side}": r2(d_fo)},
        "hand": {f"hand_{side}": ["open" if o else "relaxed" for o in open_hand[f0:f1]]},
        "stats": {"source_frames": len(K), "used": [int(f0), int(f1)], "mirrored_video": bool(mirrored),
                  "upper_range_deg": [round(float(d_up[f0:f1].min()), 1), round(float(d_up[f0:f1].max()), 1)],
                  "forearm_range_deg": [round(float(d_fo[f0:f1].min()), 1), round(float(d_fo[f0:f1].max()), 1)],
                  "open_frames": int(open_hand[f0:f1].sum()),
                  "mean_score": round(float(S.mean()), 3)},
    }
    out = config.ROOT / "actions" / action
    out.mkdir(parents=True, exist_ok=True)
    save_text(json.dumps(act), out / "motion.json")
    save_text(json.dumps({"action": action, "view": "front", "fps": fps, "frames": act["frames"],
                          "loop": False, "source": "own video (motion only)", **act["stats"]}, indent=2),
              out / "meta.json")
    save_text(f"# {action}\n\n- Source: own phone video `{Path(video).name}` (kept local, not in git).\n"
              "- Only joint angles are stored (RTMPose via rtmlib, smoothed); no frames or images.\n"
              f"- Arm: {side} (video mirrored: {mirrored}). Converted by `src/animation/video_import.py`.\n",
              out / "sources.md")
    print(json.dumps(act["stats"], indent=2))
    return act


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--action", required=True)
    ap.add_argument("--side", default="l", choices=["l", "r"])
    a = ap.parse_args()
    run(a.video, a.action, a.side)
