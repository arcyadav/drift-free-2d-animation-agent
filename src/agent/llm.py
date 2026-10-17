"""One interface for all model calls (spec §11.1): local Ollama, temperature 0, JSON-schema
constrained output, on-disk cache → reproducible, offline, free.

Usage:  from src.agent.llm import chat_json
        chat_json(messages, schema)  → dict, or None if the model is unavailable
"""
import hashlib
import json
import os
import urllib.error
import urllib.request

from src import config

MODEL = os.environ.get("ANIM_LLM", "qwen2.5:3b")
HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TIMEOUT_S = 120
CACHE = config.ROOT / "outputs" / ".cache" / "llm"


def available():
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=3) as r:
            tags = json.loads(r.read())
        return any(m["name"].startswith(MODEL) for m in tags.get("models", []))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def chat_json(messages, schema, model=MODEL):
    """messages: [{role, content}]; schema: JSON schema the answer must follow."""
    body = {"model": model, "messages": messages, "stream": False, "format": schema,
            "options": {"temperature": 0, "seed": 0}}
    key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]
    CACHE.mkdir(parents=True, exist_ok=True)
    hit = CACHE / f"{key}.json"
    if hit.exists():
        return json.loads(hit.read_text(encoding="utf-8"))
    req = urllib.request.Request(f"{HOST}/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            content = json.loads(r.read())["message"]["content"]
        out = json.loads(content)
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return None
    hit.write_text(json.dumps(out), encoding="utf-8")
    return out
