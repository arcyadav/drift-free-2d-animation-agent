"""Orchestrator (spec §7): request → plan (Planner) → clip (Animator) → report.

Critic + fix loop (spec §12): the plan is scored off-screen before the clip is written; a fixable
failure (cracks) re-runs the render with the fix (seal) and is re-checked (max 3 tries, every
decision logged); failures that need new images are reported, naming the image to add.
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
from src.critic.critic import FIXES, critique
from src.utils.io import save_text


def slug(text, n=40):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:n] or "clip"


MAX_TRIES = 3


def criticise(plan, character):
    """critic → fix → re-check; returns the render options to use and the decision log."""
    opts, log = {}, []
    for attempt in range(1, MAX_TRIES + 1):
        r = critique(plan, character, **opts)
        fixes = [FIXES[f] for f in r["failed"] if f in FIXES and not opts.get(FIXES[f])]
        log.append({"try": attempt, "options": dict(opts), "pass": r["pass"], "failed": r["failed"],
                    "checks": r["checks"], "fix_next": fixes})
        if not fixes:
            break
        opts.update({f: True for f in fixes})
    return opts, log


def make_clip(text, character="protagonist", use_llm=True, out_dir=None, critic=True):
    t0 = time.time()
    plan = plan_request(text, use_llm=use_llm)
    t_plan = time.time() - t0
    out_dir = out_dir or config.ROOT / "outputs" / "requests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{slug(text)}.mp4"
    t1 = time.time()
    if plan["view"] == "side_left" and not plan.get("seconds"):
        plan["seconds"] = 5.0
    opts, log = criticise(plan, character) if critic else ({}, [])
    t_critic = time.time() - t1
    t1 = time.time()
    if plan["view"] == "side_left":
        w = Walker(character, plan["action"], speed=plan["speed"], mirror=plan["direction"] == "right",
                   shot=plan["shot"], expression=plan.get("side_expression"), **opts)
        rep = w.render(plan["seconds"], str(out))
    else:
        rep = front_clip.FrontClip(character, plan["action"], plan["expression"], plan["speed"],
                                   plan["seconds"], plan["shot"], **opts).render(str(out))
    final = log[-1] if log else None
    report = {"request": text, "plan": plan, "render": rep, "file": str(out.relative_to(config.ROOT)),
              "critic": {"pass": final["pass"] if final else None, "fixes_applied": sorted(opts),
                         "unfixed": final["failed"] if final else [], "tries": log},
              "timing_s": {"plan": round(t_plan, 2), "critic": round(t_critic, 2),
                           "render": round(time.time() - t1, 2)}}
    save_text(json.dumps(report, indent=2), out.with_suffix(".json"))
    return report


if __name__ == "__main__":
    r = make_clip(" ".join(sys.argv[1:]) or "make her wave happily")
    print(json.dumps({k: r[k] for k in ("file", "timing_s")}, indent=2))
    print("critic:", json.dumps({k: v for k, v in r["critic"].items() if k != "tries"}))
    print(json.dumps({k: r["plan"][k] for k in ("action", "expression", "view", "speed", "direction", "shot", "notes")}))
