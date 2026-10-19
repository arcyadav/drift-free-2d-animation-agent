# Drift-Free 2D Character Animation Agent (UMC301 Project 3)

Type what a character should do in plain language. The agent plans it, animates a rigged 2D character on a
laptop CPU, checks the clip and fixes what it can. Every frame is the character's own drawing, moved, so she
never changes between frames or clips.

- **Showcase (public):** https://huggingface.co/spaces/Arcyadav/drift-free-2d-animation-agent
- **Spec** (plan + status as built, §0): `project-3-animation-agent-spec.md`
- **Evaluation:** `docs/evaluation.md` · reuse test: `docs/reuse_test.md`

## Run
```powershell
C:\Users\ysarc\.venvs\aids_p3\Scripts\Activate.ps1      # Python 3.11 venv, kept outside OneDrive
ollama serve                                            # local LLM (qwen2.5:3b) for the planner
python -m src.agent.orchestrator "she walks slowly to the right"   # request -> plan -> critic -> clip
python app.py --share        # live demo (public gradio.live link, 72 h); --lan for the intranet
python -m tests.test_critic  # tests: python -m tests.<name>
```

## Layout
```
assets/raw/    input images per character (protagonist at the top level, others in assets/raw/<name>/)
characters/    character library built by onboarding: views, parts, filled parts, face layers, hands
actions/       action library (motion only): motion.json, meta.json, sources.md per action
expressions/   shared expression recipes
src/
  onboarding/  sheet_cleaner, view_checker, keypoints, part_masks(_side), hidden_fill, face_cutter, hand_swaps
  animation/   retarget, rigid_pose (renderer), animator (walks), front_clip, keyframe_*, video_import, seal
  agent/       planner (rules + local LLM), llm, orchestrator (plan -> critic + fix -> render), eval_planner
  critic/      critic (cracks, colours, foot slide, plan delivered), eval_critic, identity_compare, gait_metrics
deploy/        build_static.py (showcase site + upload)
docs/          evaluation, reuse test, measured results (JSON)
tests/         regression tests for every stage
outputs/       generated clips and reports (not in git)
```

Everything runs on CPU. No GPU, no paid API. Personal recordings (`assets/motion/video/`) are not in git.
