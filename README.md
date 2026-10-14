# Drift-Free 2D Character Animation Agent (UMC301 Project 3)

Spec: `project-3-animation-agent-spec.md`

## Layout
```
assets/        raw inputs you provide (see assets/README.md)
characters/    character library (built by onboarding)
actions/       action library: motion.json, meta.json, sources.md per action
expressions/   shared expression recipes
src/
  onboarding/  Sheet Cleaner, View Checker, Rigger, Face Cutter, Rig Tester
  agent/       Orchestrator, Planner, Fix Selector
  animation/   Retargeter, Animator, Renderer, Stylizer
  critic/      metrics + VLM critic
  utils/       shared helpers
notebooks/     Kaggle/Colab notebooks for GPU steps (SAM, inpaint, stylize, baselines)
outputs/       generated clips, frames, report.json
tests/
```

## Compute
No local GPU → GPU steps run on Kaggle (T4/P100, ~30 h/week free) or Colab (T4).
Rig building and animation run on CPU and can run locally.

## Local environment
Python 3.11 venv kept **outside OneDrive** (so it isn't synced): `C:\Users\ysarc\.venvs\aids_p3`

```powershell
# activate (PowerShell), from the project folder
C:\Users\ysarc\.venvs\aids_p3\Scripts\Activate.ps1
python -m src.utils.check_face assets/raw/faces/front/smile.png

# add a package: put it in requirements.txt, then
uv pip install -r requirements.txt
```
Don't `pip install` into Anaconda base for this project.
