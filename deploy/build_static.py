"""Build the static showcase site (free static Hugging Face Space): pre-rendered clips for a set of
plain-language requests, each with the plan the agent made, plus how it works and the evaluation.

Usage: python deploy/build_static.py [--upload] [--space Arcyadav/drift-free-2d-animation-agent]
"""
import argparse
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from src.agent.orchestrator import make_clip, slug  # noqa: E402

SITE = ROOT / "outputs" / ".static"
REQUESTS = [
    "make her wave happily",
    "close-up of her face, she smiles and waves",
    "she says goodbye to her mom",
    "she just heard terrible news",
    "her phone fell in the drain, she is shocked!",
    "close-up, she is worried",
    "she looks down, wondering hmm",
    "medium shot, she looks down sadly",
    "she stands and waits",
    "she walks to the left",
    "she walks slowly to the right",
    "she is walking lost in thoughts and grieved",
    "walking cheerfully, hoping",
    "medium shot, she walks worried",
]

CSS = """
:root{--bg:#f6f1e8;--card:#fffdf9;--ink:#2b2420;--muted:#6f655c;--line:#e6ddd0;--accent:#3d5a80;--chip:#efe7da}
@media (prefers-color-scheme:dark){:root{--bg:#1d1a17;--card:#26221e;--ink:#efe8df;--muted:#b3a99d;--line:#3a342d;--accent:#8fb3e0;--chip:#332d27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px 64px}h1{font-size:2rem;margin:0 0 6px}h2{margin:44px 0 12px;font-size:1.35rem}
.lead{color:var(--muted);max-width:760px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card video{width:100%;display:block;background:#000}.card .body{padding:12px 14px}
.req{font-weight:600}.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.chip{background:var(--chip);border-radius:999px;padding:2px 10px;font-size:.82rem}
.note{color:var(--muted);font-size:.85rem}details summary{cursor:pointer;color:var(--accent);font-size:.85rem}
pre{background:var(--chip);padding:8px;border-radius:8px;overflow:auto;font-size:.78rem}
table{border-collapse:collapse;width:100%;background:var(--card);border-radius:12px;overflow:hidden}
td,th{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;font-size:.92rem}
ol li,ul li{margin:4px 0}.pill{display:inline-block;border:1px solid var(--line);border-radius:8px;padding:2px 8px;margin:2px;font-size:.85rem}
a{color:var(--accent)}
"""


def build(render=True):
    clips = SITE / "clips"
    if render:
        shutil.rmtree(SITE, ignore_errors=True)
    clips.mkdir(parents=True, exist_ok=True)
    cards = []
    for q in REQUESTS:
        f = clips / f"{slug(q)}.mp4"
        meta = f.with_suffix(".json")
        if render or not meta.exists():
            rep = make_clip(q, use_llm=True, out_dir=clips)
        else:
            rep = json.loads(meta.read_text())
        p = rep["plan"]
        chips = "".join(f'<span class="chip">{html.escape(str(v))}</span>' for v in
                        (p["action"], p["expression"], p["view"], f"speed {p['speed']}", p["direction"], f"shot {p['shot']}"))
        src = ", ".join(f"{k}: {v}" for k, v in p["source"].items()) or "defaults"
        notes = "".join(f'<div class="note">⚠ {html.escape(n)}</div>' for n in p["notes"])
        plan = {k: p[k] for k in ("action", "expression", "view", "speed", "direction", "seconds", "shot", "source", "notes")}
        cards.append(f"""<div class="card"><video src="clips/{f.name}" controls loop muted playsinline preload="metadata"></video>
<div class="body"><div class="req">“{html.escape(q)}”</div><div class="chips">{chips}</div>
<div class="note">decided by {html.escape(src)} · render {rep['timing_s']['render']} s on a laptop CPU</div>{notes}
<details><summary>plan JSON</summary><pre>{html.escape(json.dumps(plan, indent=2))}</pre></details></div></div>""")
    male_cards = ""
    male = ROOT / "outputs" / "male"
    # front clips only: his side-view walk is not presentable (on his dark tee the torso / sleeve
    # pieces read as cut apart from the trousers while walking — documented in docs/reuse_test.md)
    for name, label in (("wave", "wave (smile, open hand)"), ("look_down", "look down (sad)"),
                        ("idle", "idle (blinks)")):
        f = male / f"{name}.mp4"
        if f.exists():
            shutil.copy2(f, clips / f"male_{name}.mp4")
            male_cards += (f'<div class="card"><video src="clips/male_{name}.mp4" controls loop muted playsinline '
                           f'preload="metadata"></video><div class="body"><div class="req">{label}</div>'
                           f'<div class="note">same motion as the protagonist · checked by the critic</div></div></div>')
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drift-Free Animation Agent</title><style>{CSS}</style></head><body><main>
<h1>Drift-Free 2D Character Animation Agent</h1>
<p class="lead">Type what a character should do in plain language — the agent plans it and animates a rigged
2D character on a laptop CPU. No video-generation model: every frame is the same drawing, moved, so the
character <b>never changes</b> between frames or clips. UMC301 Project 3.</p>
<p><span class="pill">CPU only</span><span class="pill">free per clip</span><span class="pill">zero identity drift</span><span class="pill">local open-source LLM</span>
· <a href="https://github.com/arcyadav/drift-free-2d-animation-agent">code (GitHub)</a></p>
<h2>Requests → clips (pre-rendered by the agent)</h2>
<p class="note">A live version (type any request) runs from the author's laptop during the demo — the same agent, including the local LLM.</p>
<div class="grid">{''.join(cards)}</div>
<h2>Second character — reuse test</h2>
<p class="lead">A different character, onboarded automatically from 8 images, performing the <b>same
motion files</b> — zero new animation work. His side-view walk works mechanically but is not shown: on his
dark tee the torso and sleeve pieces read as cut apart from the trousers (see limits).</p>
<div class="grid">{male_cards}</div>
<h2>How it works</h2>
<ol>
<li><b>Onboarding (once per character):</b> a few AI-generated images (turnaround, expressions, hands) are cleaned,
checked for consistency, cut into parts (palette regions × skeleton cuts — no segmentation model), hidden areas filled,
and face layers (eyes / brows / mouth) cut and aligned.</li>
<li><b>Action library (motion only):</b> hand-designed walk cycles (natural gait, 4 moods), a wave captured from a phone
video (pose detector → angles), idle and look-down as code keyframes. Any character reuses them.</li>
<li><b>Planner:</b> rules first; a small local LLM (qwen2.5:3b) only chooses from the library's options for what the rules
could not decide; every plan is validated and repaired (e.g. a sad face in side view → body-language sad walk).</li>
<li><b>Critic:</b> every clip is checked before it is written: cracks between parts (a key-colour + part-ID render;
only gaps the drawing did not have), off-model colours, foot slide, and whether the plan was delivered. Cracks are sealed
and re-checked; a missing image is named for the user.</li>
<li><b>Animator:</b> foot lock with 2-bone IK (planted foot slides 0.0 px), soft knees, hair springs, blinks, face layers,
24 fps animated on twos, camera shots.</li>
</ol>
<h2>Measured</h2>
<table><tr><th>Check</th><th>Result</th></tr>
<tr><td>Planner, 45 requests (rules only / rules + LLM)</td><td>89% / 98% fully right</td></tr>
<tr><td>Planted-foot slide (no lock → foot lock)</td><td>4.7 → 0.0 px per frame</td></tr>
<tr><td>Walk vs. an AI-video reference (steps/s, step length, bob)</td><td>1.33 / 0.60 / 0.045 vs. 1.39 / 0.59 / 0.063</td></tr>
<tr><td>Identity: commercial AI video of the same character</td><td>body widened 8–13% vs. the drawing; ours: pixel-exact</td></tr>
<tr><td>Critic on 21 clips (both characters): pass first / cracks fixed by seal / needs a new image</td><td>16 / 4 / 1 · off-model colours ≤ 1.2%</td></tr>
<tr><td>Render time</td><td>~3–17 s per clip on a laptop CPU, no GPU</td></tr>
</table>
<h2>Honest limits</h2>
<ul><li>Cut-out (puppet) look: rigid sleeves, no cloth simulation — AI video has richer frames, but redraws the character.</li>
<li>Side view: smile / sad / worried / shocked faces from profile images; other moods fall back to neutral.</li>
<li>Onboarding rules assume tee / jeans / sneakers; new outfit types need new garment rules.</li>
<li>Second character, side view: with no visible sleeve seam on his dark tee, the sleeve piece is cut too large, so
while walking his torso reads as separate from his trousers. Layered art (each part drawn whole) would remove this.</li></ul>
</main></body></html>"""
    (SITE / "index.html").write_text(page, encoding="utf-8")
    meta = ["---", "title: Drift-Free 2D Animation Agent", "colorFrom: yellow", "colorTo: blue",
            "sdk: static", "pinned: false", "---", ""]
    (SITE / "README.md").write_text("\n".join(meta), encoding="utf-8")
    print("site:", SITE / "index.html")


def upload(space):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=str(SITE), repo_id=space, repo_type="space",
                      commit_message="Static showcase of the drift-free animation agent")
    print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--space", default="Arcyadav/drift-free-2d-animation-agent")
    a = ap.parse_args()
    build(render=not a.no_render)
    if a.upload:
        upload(a.space)
