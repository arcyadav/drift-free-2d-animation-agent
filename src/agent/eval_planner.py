"""Planner evaluation (spec §11.2 item 8): accuracy on the 40-request test set for
rules only, and rules + local LLM. Only the fields given in a case are scored.

Usage: python -m src.agent.eval_planner
"""
import json
import time

from src import config
from src.agent.planner import plan_request
from src.utils.io import save_text

CASES = config.ROOT / "tests" / "planner_cases.json"
FIELDS = ("action", "expression", "speed", "direction", "seconds")


def score(use_llm):
    cases = json.loads(CASES.read_text())
    right = {f: 0 for f in FIELDS}
    total = {f: 0 for f in FIELDS}
    exact, wrong, t0, llm_calls = 0, [], time.time(), 0
    for c in cases:
        p = plan_request(c["q"], use_llm=use_llm)
        llm_calls += "llm" in p["source"].values()
        ok_all = True
        for f in FIELDS:
            if f in c:
                total[f] += 1
                ok = p[f] == c[f] or (f == "action" and str(p[f]).startswith(c[f] + "_"))
                right[f] += ok
                if not ok:
                    ok_all = False
                    wrong.append(f'{c["q"]!r}: {f} = {p[f]!r}, expected {c[f]!r}')
        exact += ok_all
    acc = {f: round(right[f] / total[f], 3) for f in FIELDS if total[f]}
    return {"cases": len(cases), "requests_fully_right": round(exact / len(cases), 3),
            "field_accuracy": acc, "llm_used_in": llm_calls,
            "seconds_total": round(time.time() - t0, 1), "errors": wrong}


if __name__ == "__main__":
    res = {"rules_only": score(False), "rules_plus_llm": score(True)}
    save_text(json.dumps(res, indent=2), config.ROOT / "outputs" / "planner_eval.json")
    for k, r in res.items():
        print(f"{k:15s} fully right {r['requests_fully_right']:.0%}  fields {r['field_accuracy']}  "
              f"llm used {r['llm_used_in']}x  {r['seconds_total']} s")
    print("\nerrors (rules + llm):")
    for e in res["rules_plus_llm"]["errors"]:
        print("  ", e)
