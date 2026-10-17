"""Orchestrator (spec §7): request → plan (Planner) → clip (Animator) → report.

The critic + fix loop will plug in after rendering (spec §12).
Usage: python -m src.agent.orchestrator "she walks slowly to the right"
"""
import json
import re
import sys
import time

from src import config
from src.agent.planner import plan_request
from src.animation import front_clip
from src.animation.animator import Walker
from src.utils.io import save_text


def slug(text, n=40):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:n] or "clip"


def make_clip(text, character="protagonist", use_llm=True, out_dir=None):
    t0 = time.time()
    plan = plan_request(text, use_llm=use_llm)
    t_plan = time.time() - t0
    out_dir = out_dir or config.ROOT / "outputs" / "requests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{slug(text)}.mp4"
    t1 = time.time()
    if plan["view"] == "side_left":
        w = Walker(character, plan["action"], speed=plan["speed"], mirror=plan["direction"] == "right")
        rep = w.render(plan["seconds"] or 5.0, str(out))
    else:
        rep = front_clip.FrontClip(character, plan["action"], plan["expression"], plan["speed"],
                                   plan["seconds"]).render(str(out))
    report = {"request": text, "plan": plan, "render": rep, "file": str(out.relative_to(config.ROOT)),
              "timing_s": {"plan": round(t_plan, 2), "render": round(time.time() - t1, 2)}}
    save_text(json.dumps(report, indent=2), out.with_suffix(".json"))
    return report


if __name__ == "__main__":
    r = make_clip(" ".join(sys.argv[1:]) or "make her wave happily")
    print(json.dumps({k: r[k] for k in ("file", "timing_s")}, indent=2))
    print(json.dumps({k: r["plan"][k] for k in ("action", "expression", "view", "speed", "direction", "notes")}))
