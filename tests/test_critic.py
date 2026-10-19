"""Critic: passes a good clip, catches the cracks the eye caught (male walk, react armpits),
the seal fix removes them, and a missing expression image is named.

Run: python -m tests.test_critic
"""
from src.agent.orchestrator import criticise
from src.critic.critic import critique


def plan(action, expression=None, view="front"):
    return {"action": action, "view": view, "expression": expression, "side_expression": expression,
            "speed": 1.0, "direction": "left", "shot": "full", "seconds": 4.0 if view == "side_left" else None}


def test_critic():
    good = critique(plan("walk", view="side_left"))
    assert good["pass"], good["failed"]
    assert good["checks"]["foot_slide"]["value"] == 0.0
    assert good["checks"]["colours"]["value"] < 0.01, "a rig only moves her own pixels"

    broken = critique(plan("walk", view="side_left"), "male")
    assert broken["failed"] == ["cracks"], broken["failed"]           # the back crack seen by eye
    assert not critique(plan("react", "shocked"))["checks"]["cracks"]["pass"]   # armpit gaps

    opts, log = criticise(plan("walk", view="side_left"), "male")
    assert opts == {"seal": True} and log[-1]["pass"], log[-1]["failed"]

    sad = critique(plan("walk_sad", "sad", view="side_left"), "male")
    assert "expression" in sad["failed"] and "side_left/sad.png" in sad["checks"]["expression"]["note"]
    print("critic: all checks passed")


if __name__ == "__main__":
    test_critic()
