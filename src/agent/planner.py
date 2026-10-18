"""Planner (spec §7, §11.2): plain-language request → validated plan JSON.

  {"action", "expression", "view", "speed", "direction", "seconds"}

Designed for a small local model:
  1. rules first: a keyword / synonym parser fills every slot it can (most requests never
     reach the LLM)
  2. the LLM (qwen2.5:3b via Ollama) only fills slots the rules left empty, and only by
     CHOOSING from enums built from the libraries (constrained JSON)
  3. validate → repair → fall back: values outside the library, or an expression the view
     cannot show, are corrected by rule and logged in plan["notes"]

Usage: python -m src.agent.planner "make her wave happily"
"""
import json
import re
import sys

from src import config
from src.agent import llm

# which view each action is drawn in (library actions only)
ACTION_VIEW = {"run": "side_left", "react": "front", "walk": "side_left", "walk_sad": "side_left", "walk_happy": "side_left",
               "walk_worried": "side_left", "wave": "front", "idle": "front", "look_down": "front"}
DEFAULT_SECONDS = {"run": 4.0, "react": 3.0, "walk_sad": 5.0, "walk_happy": 5.0, "walk_worried": 5.0, "walk": 5.0, "wave": None, "idle": 4.0, "look_down": 4.0}
# expressions each view can show (front: full face layers; side: none cut yet)
VIEW_EXPRESSIONS = {"front": None, "side_left": ["neutral"]}

ACTION_WORDS = {
    "run": ["run", "runs", "running", "ran", "sprint", "jog", "dash", "races", "racing", "chases"],
    "walk": ["walk", "walks", "walking", "stroll", "strolls", "go to", "goes", "heads", "head to",
             "leave", "leaves", "cross", "crosses", "move along", "pace"],
    "wave": ["wave", "waves", "waving", "say hi", "says hi", "hello", "bye", "goodbye", "greet",
             "greets", "hi to"],
    "react": ["oh no", "fell", "falls", "fall in", "dropped", "drops", "lost her", "loses her", "shock",
              "gasp", "startl", "jump scare", "scared", "frighten", "can't believe", "cannot believe", "spill"],
    "look_down": ["look down", "looks down", "looking down", "stares at the floor", "at her feet",
                  "lowers her eyes", "glance down"],
    "idle": ["stand", "stands", "standing", "idle", "wait", "waits", "waiting", "stay", "breathe"],
}
EXPRESSION_WORDS = {
    "smile": ["happ", "smil", "cheer", "joy", "glad", "excit", "friendly", "warm", "hop"],
    "sad": ["sad", "unhapp", "down cast", "gloom", "upset", "cry", "tear", "sorrow", "heartbroken",
            "griev", "grief", "mourn", "depress", "lonely", "terrible", "bad news", "awful", "failed"],
    "worried": ["worr", "anxious", "nervous", "uneasy", "tense", "concern"],
    "shocked": ["shock", "surpris", "startl", "gasp", "stun", "amazed"],
    "thoughtful": ["think", "thought", "wonder", "ponder", "hmm", "curious", "interest", "consider",
                   "puzzl", "reflect"],
    "neutral": ["neutral", "calm", "blank", "no expression", "normally"],
}
SPEED_WORDS = [(0.6, ["very slow", "really slow", "sluggish"]), (0.75, ["slow", "slowly", "gently", "lazily", "tired"]),
               (1.3, ["quick", "fast", "hurr", "rush", "brisk"]), (1.0, ["normal"])]
MOOD_ACTION = {"sad": "look_down", "thoughtful": "look_down", "worried": "look_down",
               "shocked": "react", "smile": "wave"}
SHOT_WORDS = {"close": ["close-up", "close up", "closeup", "zoom", "her face", "on her face", "face only"],
              "medium": ["medium", "waist up", "upper body", "closer", "half body", "shoulders"],
              "full": ["full body", "full shot", "wide", "whole body"]}
DIRECTION_WORDS = {"right": ["right", "to the right"], "left": ["left", "to the left"]}


def library():
    """enums from the libraries actually on disk."""
    acts = sorted(p.name for p in (config.ROOT / "actions").iterdir()
                  if (p / "motion.json").exists() and p.name in ACTION_VIEW)
    rec = json.loads((config.ROOT / "expressions" / "recipes.json").read_text())
    return {"actions": acts, "expressions": sorted(rec["recipes"])}


def _has(text, words):
    return any(re.search(r"(?<![a-z])" + re.escape(w), text) for w in words)


def parse_rules(text):
    t = " " + text.lower() + " "
    plan = {}
    for a, words in ACTION_WORDS.items():
        if _has(t, words):
            plan["action"] = a
            break
    for e, words in EXPRESSION_WORDS.items():
        if _has(t, words):
            plan["expression"] = e
            break
    for v, words in SPEED_WORDS:
        if _has(t, words):
            plan["speed"] = v
            break
    for sh, words in SHOT_WORDS.items():
        if _has(t, words):
            plan["shot"] = sh
            break
    for d, words in DIRECTION_WORDS.items():
        if _has(t, words):
            plan["direction"] = d
            break
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b", t)
    if m:
        plan["seconds"] = float(m.group(1))
    return plan


FEW_SHOT = [
    ("she says goodbye to her mom", {"action": "wave", "expression": "smile"}),
    ("she just heard terrible news", {"action": "look_down", "expression": "sad"}),
    ("she heads to college", {"action": "walk", "expression": "neutral"}),
    ("she realises she lost her phone", {"action": "react", "expression": "shocked"}),
    ("she wonders about the strange letter", {"action": "look_down", "expression": "thoughtful"}),
]


def ask_llm(text, missing, lib):
    """fill only the missing slots, by choosing from enums."""
    props = {}
    base = [a for a in lib["actions"] if not a.startswith("walk_")]   # mood walks come from walk + mood
    if "action" in missing:
        props["action"] = {"type": "string", "enum": base}
    if "expression" in missing:
        props["expression"] = {"type": "string", "enum": lib["expressions"]}
    if not props:
        return {}
    schema = {"type": "object", "properties": props, "required": list(props)}
    shots = "\n".join(f'Request: "{q}" -> ' + json.dumps({k: v for k, v in a.items() if k in props})
                      for q, a in FEW_SHOT)
    msgs = [{"role": "system", "content":
             "You map an animation request for a cartoon girl to library choices. "
             f"Actions: {base}. Expressions: {lib['expressions']}. "
             "Answer only with JSON."},
            {"role": "user", "content": f"{shots}\nRequest: \"{text}\" ->"}]
    out = llm.chat_json(msgs, schema)
    return {k: v for k, v in (out or {}).items() if k in props}


def validate(plan, lib, notes):
    """repair anything outside the library; returns the final plan."""
    if plan.get("action") not in lib["actions"]:
        notes.append(f"action {plan.get('action')!r} not in library {lib['actions']} -> 'wave'"
                     if plan.get("action") else "no action found -> 'wave'")
        plan["action"] = "wave" if "wave" in lib["actions"] else lib["actions"][0]
    # a walk with a mood → the body-language walk for that mood (side view has no face moods)
    mood = {"sad": "walk_sad", "smile": "walk_happy", "worried": "walk_worried", "thoughtful": "walk_sad"}
    if plan["action"] == "walk" and mood.get(plan.get("expression")) in lib["actions"]:
        notes.append(f"walk + {plan['expression']} -> {mood[plan['expression']]} (body language)")
        plan["action"] = mood[plan["expression"]]
    view = ACTION_VIEW[plan["action"]]
    plan["view"] = view
    expr = plan.get("expression") or {"wave": "smile", "react": "shocked"}.get(plan["action"], "neutral")
    if expr not in lib["expressions"]:
        notes.append(f"expression {expr!r} unknown -> neutral")
        expr = "neutral"
    allowed = VIEW_EXPRESSIONS[view]
    if allowed is not None and expr not in allowed:
        notes.append(f"expression {expr!r} not available in {view} (face layers: {allowed}) -> neutral")
        expr = "neutral"
    plan["expression"] = expr
    plan["speed"] = float(min(max(plan.get("speed", 1.0), 0.5), 1.5))
    plan["direction"] = plan.get("direction", "left")
    if plan["direction"] not in ("left", "right"):
        plan["direction"] = "left"
    plan["shot"] = plan.get("shot", "full")
    sec = plan.get("seconds") or DEFAULT_SECONDS.get(plan["action"])
    plan["seconds"] = float(min(max(sec, 1.0), 15.0)) if sec else None
    return plan


def plan_request(text, use_llm=True):
    lib = library()
    plan = parse_rules(text)
    notes, source = [], {k: "rules" for k in plan}
    # a mood with no action: the action that shows that mood best (deterministic, no LLM)
    if "action" not in plan and plan.get("expression") in MOOD_ACTION:
        plan["action"] = MOOD_ACTION[plan["expression"]]
        source["action"] = "rules"
    missing = [k for k in ("action", "expression") if k not in plan]
    if missing and use_llm and llm.available():
        got = ask_llm(text, missing, lib)
        plan.update(got)
        source.update({k: "llm" for k in got})
    elif missing and use_llm:
        notes.append("local LLM not available -> rule defaults")
    plan = validate(plan, lib, notes)
    plan["source"] = source
    plan["notes"] = notes
    plan["request"] = text
    return plan


if __name__ == "__main__":
    print(json.dumps(plan_request(" ".join(sys.argv[1:]) or "make her wave happily"), indent=2))
