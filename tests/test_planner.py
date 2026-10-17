"""Planner: plans are always valid library choices; rules alone get the explicit requests;
with the local LLM (cached answers) the test set reaches >= 90%.

Run: python -m tests.test_planner
"""
from src.agent import llm
from src.agent.eval_planner import score
from src.agent.planner import ACTION_VIEW, library, plan_request


def test_planner():
    lib = library()
    for q in ("", "dance on the moon", "she walks sadly", "make her fly"):
        p = plan_request(q, use_llm=False)
        assert p["action"] in lib["actions"] and p["expression"] in lib["expressions"]
        assert p["view"] == ACTION_VIEW[p["action"]] and 0.5 <= p["speed"] <= 1.5
    p = plan_request("she walks sadly", use_llm=False)
    assert p["expression"] == "neutral" and p["notes"], "side view cannot show sad: repaired + logged"

    rules = score(use_llm=False)
    assert rules["requests_fully_right"] >= 0.8, rules["errors"]
    if llm.available():
        both = score(use_llm=True)
        assert both["requests_fully_right"] >= 0.9, both["errors"]
        print(f"planner: rules {rules['requests_fully_right']:.0%}, rules+llm {both['requests_fully_right']:.0%}")
    print("planner: all checks passed")


if __name__ == "__main__":
    test_planner()
