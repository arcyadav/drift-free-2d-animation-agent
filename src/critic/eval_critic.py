"""Critic evaluation: every showcase request (protagonist) + the reuse-test character's clips,
first check vs after the fix loop → outputs/critic_eval.json + a markdown table.

Usage: python -m src.critic.eval_critic
"""
import json
import time

from src import config
from src.agent.orchestrator import criticise
from src.agent.planner import plan_request
from src.utils.io import save_text

from deploy.build_static import REQUESTS

MALE = ["make him wave happily", "he just heard terrible news", "he looks down sadly", "he stands and waits",
        "he walks to the left", "walking cheerfully, hoping", "he walks sadly"]


def run():
    rows = []
    for character, reqs in (("protagonist", REQUESTS), ("male", MALE)):
        for q in reqs:
            plan = plan_request(q, use_llm=True)
            if plan["view"] == "side_left" and not plan.get("seconds"):
                plan["seconds"] = 5.0
            t = time.time()
            opts, log = criticise(plan, character)
            first, last = log[0], log[-1]
            rows.append({"character": character, "request": q, "action": plan["action"],
                         "first_failed": first["failed"], "fixes": sorted(opts), "final_failed": last["failed"],
                         "cracks_px_first": first["checks"]["cracks"]["note"],
                         "off_model_colours": last["checks"]["colours"]["value"],
                         "foot_slide": last["checks"].get("foot_slide", {}).get("value"),
                         "notes": [c["note"] for c in last["checks"].values() if not c["pass"] and c["note"]],
                         "seconds": round(time.time() - t, 1)})
            r = rows[-1]
            print(f"{character:11s} {q[:42]:42s} first {r['first_failed'] or 'pass'} -> fix {r['fixes'] or '-'} -> "
                  f"{r['final_failed'] or 'pass'}  ({r['seconds']} s)")
    n = len(rows)
    summary = {"clips": n,
               "pass_first": sum(not r["first_failed"] for r in rows),
               "fixed_by_seal": sum(bool(r["first_failed"]) and not r["final_failed"] for r in rows),
               "unfixed_need_images": sum(bool(r["final_failed"]) for r in rows),
               "max_off_model_colours": max(r["off_model_colours"] for r in rows),
               "mean_critic_seconds": round(sum(r["seconds"] for r in rows) / n, 1)}
    save_text(json.dumps({"summary": summary, "rows": rows}, indent=2), config.ROOT / "outputs" / "critic_eval.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
