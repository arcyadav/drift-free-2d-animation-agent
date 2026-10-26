"""Claude Code session log → one standalone transcript.html (viva requirement):
Turn 1, 2, … = each request of mine, shown in full; the response (text, tool calls, results)
collapsed under it.

The raw session log (.jsonl) is used rather than /export alone, because the session was compacted
(summarised) twice; the log still holds every turn. Resumes and compaction write some messages
more than once under the same id; each is shown once, in the order it was written. Hidden reasoning and images are left out; long tool outputs are shortened.

Usage: python deploy/transcript_html.py --log <session>.jsonl [--out outputs/transcript/transcript.html]
"""
import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
MD = MarkdownIt("commonmark", {"html": False}).enable("table").enable("strikethrough")
CHAT = {"user", "assistant", "system"}
TOOL_IN_MAX = 1500          # chars of a tool call's input shown
TOOL_OUT_MAX = 1200         # chars of a tool result shown
SECRET = re.compile(r"(hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_-]{20,}")
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
PASTE = re.compile(r'<pasted_content id="[^"]*">\s*(.*?)\s*</pasted_content>', re.S)
NOT_A_TURN = ("<task-notification>", "<local-command", "<command-", "Your claude.ai usage limit",
              "[Request interrupted", "Caveat:")

CSS = """
:root{--bg:#f7f5f0;--card:#fffdf9;--ink:#26221e;--muted:#6f655c;--line:#e4dccf;--me:#eaf1fb;--me-line:#c4d6ee;
--accent:#3d5a80;--code:#f1ece3;--tool:#f6f2ea}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1b1916;--card:#24211d;--ink:#ece6dd;--muted:#a99f93;
--line:#39332c;--me:#1f2a38;--me-line:#33465f;--accent:#8fb3e0;--code:#2d2924;--tool:#29251f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 16px 80px}h1{font-size:1.7rem;margin:0 0 4px}
.lead{color:var(--muted)}.meta{color:var(--muted);font-size:.86rem}
.toc{display:flex;flex-wrap:wrap;gap:4px;margin:14px 0 24px}.toc a{font-size:.78rem;border:1px solid var(--line);border-radius:6px;
padding:1px 6px;text-decoration:none;color:var(--accent)}
section.turn{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:14px 0;overflow:hidden}
section.turn h2{font-size:.95rem;margin:0;padding:8px 14px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}
section.turn h2 span{color:var(--muted);font-weight:400;font-size:.8rem}
.me{background:var(--me);border-bottom:1px solid var(--me-line);padding:10px 14px;white-space:pre-wrap;overflow-wrap:anywhere}
details.resp>summary{cursor:pointer;padding:8px 14px;color:var(--accent);font-size:.88rem}
.respbody{padding:0 14px 12px;overflow-wrap:anywhere}.respbody p{margin:.5em 0}
details.tool{background:var(--tool);border:1px solid var(--line);border-radius:8px;margin:6px 0;font-size:.82rem}
details.tool>summary{cursor:pointer;padding:4px 10px;color:var(--muted)}
pre{background:var(--code);padding:8px 10px;border-radius:6px;overflow:auto;font-size:.78rem;white-space:pre-wrap;margin:4px 0}
code{background:var(--code);padding:0 3px;border-radius:3px;font-size:.88em}pre code{background:none;padding:0}
table{border-collapse:collapse;margin:8px 0;font-size:.86rem;display:block;overflow-x:auto}td,th{border:1px solid var(--line);padding:4px 8px;text-align:left}
.note{color:var(--muted);font-size:.82rem;font-style:italic;margin:6px 0}
.marker{text-align:center;color:var(--muted);font-size:.85rem;margin:22px 0}
a{color:var(--accent)}
"""


def clean(text):
    text = REMINDER.sub("", text)
    text = PASTE.sub(lambda m: "[pasted text]\n" + m.group(1), text)
    return SECRET.sub("[redacted]", text).strip()


def short(text, n):
    text = clean(text)
    return text if len(text) <= n else text[:n] + f"\n… [{len(text) - n} more characters]"


def blocks(content):
    return [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])


def load_path(log):
    """every message once, in the order it was written. Resumes and compaction re-write messages
    under the same id (the copies are dropped); compaction summaries stay as markers."""
    out, seen = [], set()
    for line in open(log, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        queued = r.get("type") == "attachment" and (r.get("attachment") or {}).get("type") == "queued_command"
        if (r.get("type") in CHAT or queued) and r.get("uuid") and not r.get("isSidechain") and r["uuid"] not in seen:
            seen.add(r["uuid"])
            out.append(r)
    return out


def result_text(b):
    c = b.get("content")
    if isinstance(c, list):
        return "\n".join(x.get("text", "[image]") if x.get("type") == "text" else "[image]" for x in c)
    return str(c or "")


def tool_label(b):
    inp = b.get("input", {})
    desc = inp.get("description") or inp.get("file_path") or inp.get("pattern") or inp.get("prompt", "")[:80]
    return f"{b.get('name')}: {desc}" if desc else b.get("name", "tool")


def build(log, out):
    turns, cur = [], None
    for r in load_path(log):
        msg = r.get("message", {})
        ts = r.get("timestamp", "")
        if r["type"] == "attachment":                      # a message I sent while Claude was working
            text = clean(str(r["attachment"].get("prompt", "")))
            if text:
                cur = {"me": text, "ts": ts, "parts": [], "tools": 0, "mid": True}
                turns.append(cur)
            continue
        if r["type"] == "system":
            if r.get("subtype") == "compact_boundary":
                turns.append({"marker": "The conversation was summarised here to free up context (it continued in the same session)."})
            continue
        if r["type"] == "assistant":
            if cur is None:
                continue
            for b in blocks(msg.get("content")):
                if b.get("type") == "text" and b.get("text", "").strip():
                    cur["parts"].append(("md", clean(b["text"])))
                elif b.get("type") == "tool_use":
                    cur["parts"].append(("tool", tool_label(b), short(json.dumps(b.get("input", {}), indent=1,
                                                                              ensure_ascii=False), TOOL_IN_MAX)))
                    cur["tools"] += 1
            continue
        # user record
        if r.get("isCompactSummary"):
            turns.append({"marker": "Context summary inserted by Claude Code after compaction (the turns before it are above)."})
            continue
        if r.get("isMeta"):
            continue
        texts, images = [], 0
        for b in blocks(msg.get("content")):
            if b.get("type") == "tool_result" and cur is not None:
                cur["parts"].append(("result", short(result_text(b), TOOL_OUT_MAX)))
            elif b.get("type") == "text":
                texts.append(b["text"])
            elif b.get("type") == "image":
                images += 1
        text = clean("\n".join(texts))
        if not text and not images:
            continue
        if text.startswith(NOT_A_TURN) or "<task-notification>" in text[:200]:
            if cur is not None and "<task-notification>" in text:
                cur["parts"].append(("note", "A background task finished."))
            elif cur is not None and text.startswith("[Request interrupted"):
                cur["parts"].append(("note", "(interrupted by me)"))
            continue
        if images:
            text += f"\n[{images} image{'s' if images > 1 else ''} attached]"
        cur = {"me": text, "ts": ts, "parts": [], "tools": 0}
        turns.append(cur)

    n = sum("me" in t for t in turns)
    stamps = [t["ts"] for t in turns if t.get("ts")]
    day = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).strftime("%d %b %Y")
    toc, body, k = [], [], 0
    for t in turns:
        if "marker" in t:
            body.append(f'<div class="marker">— {html.escape(t["marker"])} —</div>')
            continue
        k += 1
        toc.append(f'<a href="#t{k}">{k}</a>')
        parts = []
        for p in t["parts"]:
            if p[0] == "md":
                parts.append(MD.render(p[1]))
            elif p[0] == "tool":
                parts.append(f'<details class="tool"><summary>🔧 {html.escape(p[1][:160])}</summary>'
                             f'<pre>{html.escape(p[2])}</pre></details>')
            elif p[0] == "result":
                parts.append(f'<details class="tool"><summary>↳ result</summary><pre>{html.escape(p[1])}</pre></details>')
            else:
                parts.append(f'<div class="note">{html.escape(p[1])}</div>')
        when = datetime.fromisoformat(t["ts"].replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M UTC") if t["ts"] else ""
        mid = " · sent while Claude was working" if t.get("mid") else ""
        body.append(f'<section class="turn" id="t{k}"><h2>Turn {k} <span>{when}{mid}</span></h2>'
                    f'<div class="me">{html.escape(t["me"])}</div>'
                    f'<details class="resp"><summary>Response ({t["tools"]} tool calls)</summary>'
                    f'<div class="respbody">{"".join(parts) or "<p class=note>(no text response)</p>"}</div></details></section>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Project 3 Transcript</title><style>{CSS}</style></head><body><main>
<h1>Drift-Free 2D Character Animation Agent — Claude Code transcript</h1>
<p class="lead">UMC301 Project 3 · Archana Yadav · {n} turns · {day(stamps[0])} – {day(stamps[-1])}</p>
<p class="meta">Each turn shows my request; click <b>Response</b> to expand Claude's answer, tool calls and shortened tool outputs.
Generated from the full Claude Code session log by <code>deploy/transcript_html.py</code> (the session was summarised twice to
free up context; the log keeps every turn). Hidden reasoning and images are not included.
Showcase: <a href="https://huggingface.co/spaces/Arcyadav/drift-free-2d-animation-agent">huggingface.co/spaces/Arcyadav/drift-free-2d-animation-agent</a></p>
<nav class="toc">{''.join(toc)}</nav>
{''.join(body)}
</main></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"{out}  turns={n}  size={out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "transcript" / "transcript.html"))
    a = ap.parse_args()
    build(a.log, Path(a.out))
