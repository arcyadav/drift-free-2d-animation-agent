"""Walk-style metrics — compare any walk video (e.g. an AI-video baseline) with ours.

Measured from 2D joints (pose detector on a video, or our motion.json):
  steps per second, foot spread (step length), knee-bend range, body bob, arm swing.
Used to tune the hand-designed walk to a natural pace and for the evaluation table.

Usage: python -m src.critic.gait_metrics --video assets/baseline/<clip>.mp4 --action walk
"""
import argparse
import json

import imageio.v2 as iio
import numpy as np
from scipy.signal import savgol_filter

from src import config
from src.onboarding.keypoints import COCO, MIN_SCORE, detector


def joints_from_video(path):
    K, S = [], []
    reader = iio.get_reader(path)
    fps = reader.get_meta_data().get("fps", 24.0)
    for f in reader:
        kp, sc = detector()(f[..., ::-1].copy())
        if len(kp) == 0:
            K.append(np.full((17, 2), np.nan)); S.append(np.zeros(17)); continue
        i = int(np.argmax(sc.mean(1))); K.append(kp[i]); S.append(sc[i])
    K, S = np.array(K, float), np.array(S)
    K[S < MIN_SCORE] = np.nan
    J = {n: K[:, i].copy() for i, n in enumerate(COCO)}
    for n in J:
        for c in range(2):
            v = J[n][:, c]; ok = ~np.isnan(v)
            if ok.sum() > 10:
                v = np.interp(np.arange(len(v)), np.where(ok)[0], v[ok])
                J[n][:, c] = savgol_filter(v, 7, 2)
    return J, fps


def joints_from_motion(action, cycles=4):
    m = json.loads((config.ROOT / "actions" / action / "motion.json").read_text())
    return {k: np.tile(np.array(v) * [1, -1], (cycles, 1)) for k, v in m["joints"].items()}, m["fps"]


def _bend(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return 180 - np.degrees(np.arccos(np.clip(cos, -1, 1)))


def metrics(J, fps):
    hip = (J["l_hip"] + J["r_hip"]) / 2
    leg = np.nanmedian(np.linalg.norm(J["l_hip"] - J["l_knee"], axis=1)
                       + np.linalg.norm(J["l_knee"] - J["l_ankle"], axis=1))
    gap = J["l_ankle"][:, 0] - J["r_ankle"][:, 0]
    g = gap - np.nanmean(gap)
    crossings = np.where(np.diff(np.sign(g)) != 0)[0]
    flex = np.r_[_bend(J["l_hip"], J["l_knee"], J["l_ankle"]), _bend(J["r_hip"], J["r_knee"], J["r_ankle"])]
    arm = [np.degrees(np.arctan2(J[f"{s}_wrist"][:, 0] - J[f"{s}_shoulder"][:, 0],
                                 np.abs(J[f"{s}_wrist"][:, 1] - J[f"{s}_shoulder"][:, 1]))) for s in "lr"]
    return {
        "steps_per_s": round(len(crossings) / (len(g) / fps), 2),
        "foot_spread_legs": round(float(np.nanpercentile(np.abs(gap), 95) / leg), 2),
        "knee_bend_deg": [round(float(np.nanpercentile(flex, 5))), round(float(np.nanpercentile(flex, 95)))],
        "bob_legs": round(float((np.nanpercentile(hip[:, 1], 95) - np.nanpercentile(hip[:, 1], 5)) / leg), 3),
        "arm_swing_deg": round(float(np.nanmean([np.nanpercentile(a, 95) - np.nanpercentile(a, 5) for a in arm])), 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--action", default="walk")
    a = ap.parse_args()
    rows = {}
    if a.video:
        rows["video"] = metrics(*joints_from_video(a.video))
    rows[a.action] = metrics(*joints_from_motion(a.action))
    print(json.dumps(rows, indent=2))
