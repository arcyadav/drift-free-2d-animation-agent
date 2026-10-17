"""Drift-Free 2D Character Animation Agent — live web demo (Gradio, spec §15.1).

Run from the laptop (the full agent incl. the local LLM):
    python app.py --share     public link for the viva (laptop must stay on)
    python app.py --lan       intranet: http://<this-laptop-ip>:7860 from other devices
A static showcase (pre-rendered clips) is published separately: deploy/build_static.py.

Type a request in plain language → the Planner turns it into a plan (rules first, the local LLM
only if available) → the Animator renders the clip on CPU from the rigged character.
No image or video generation model is used: every frame is the same drawing, moved.
"""
import json
import time

import gradio as gr

from src import config
from src.agent import llm
from src.agent.orchestrator import make_clip

EXAMPLES = [
    "make her wave happily",
    "close-up of her face, she smiles and waves",
    "she just heard terrible news",
    "she looks down, wondering hmm",
    "she walks slowly to the right",
    "she is walking lost in thoughts and grieved",
    "walking cheerfully, hoping",
    "medium shot, she stands and waits",
]
LLM_ON = llm.available()

ABOUT = f"""
**What this is.** A character drawn once (a few AI-generated images) is cut into parts, rigged and
animated **on a CPU** — no video generation, so she never changes: no identity drift, exact control,
zero cost per clip. UMC301 Project 3.

**Can do:** walk (also sad / happy / worried body language), wave, look down, idle ·
expressions: neutral, smile, sad, worried, shocked, thoughtful (+ blinks) · speed, left / right,
duration, shot (full / medium / close-up).

**Planner:** rules first; local LLM (`{llm.MODEL}`) {'**on**' if LLM_ON else '**off here** (rules only — the LLM runs on the local machine)'}.
"""


def run(request):
    if not request or not request.strip():
        return None, "Type a request first.", ""
    t0 = time.time()
    rep = make_clip(request.strip(), use_llm=LLM_ON, out_dir=config.ROOT / "outputs" / "app")
    plan = rep["plan"]
    summary = (f"**Plan:** {plan['action']} · {plan['expression']} · view {plan['view']} · "
               f"speed {plan['speed']} · {plan['direction']} · shot {plan['shot']}  \n"
               f"**Decided by:** {', '.join(f'{k}: {v}' for k, v in plan['source'].items()) or 'defaults'}  \n"
               f"**Time:** plan {rep['timing_s']['plan']} s, render {rep['timing_s']['render']} s "
               f"(total {time.time() - t0:.1f} s, CPU)")
    if plan["notes"]:
        summary += "  \n**Notes:** " + "; ".join(plan["notes"])
    details = json.dumps({k: plan[k] for k in ("request", "action", "expression", "view", "speed",
                                               "direction", "seconds", "shot", "source", "notes")}, indent=2)
    return str(config.ROOT / rep["file"]), summary, details


with gr.Blocks(title="Drift-Free 2D Animation Agent") as demo:
    gr.Markdown("# Drift-Free 2D Character Animation Agent")
    gr.Markdown(ABOUT)
    with gr.Row():
        with gr.Column(scale=2):
            req = gr.Textbox(label="What should she do?", placeholder="e.g. she waves goodbye happily",
                             lines=2)
            go = gr.Button("Animate", variant="primary")
            gr.Examples(EXAMPLES, inputs=req)
        with gr.Column(scale=3):
            video = gr.Video(label="Clip", autoplay=True)
            summary = gr.Markdown()
            with gr.Accordion("Plan JSON", open=False):
                details = gr.Code(language="json")
    go.click(run, inputs=req, outputs=[video, summary, details], api_name="animate")
    req.submit(run, inputs=req, outputs=[video, summary, details])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="live demo: local, --lan (intranet) or --share (internet link)")
    ap.add_argument("--share", action="store_true", help="public *.gradio.live link (valid 72 h)")
    ap.add_argument("--lan", action="store_true", help="reachable from other devices on this network")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.queue(max_size=8).launch(share=a.share, server_name="0.0.0.0" if a.lan else "127.0.0.1",
                                  server_port=a.port, show_api=False, show_error=True)
