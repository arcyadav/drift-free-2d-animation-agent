"""Stage the files the web app needs and upload them to a Hugging Face Space.

Needs a one-time `huggingface-cli login` with a WRITE token.
Usage: python deploy/deploy_space.py [--space Arcyadav/drift-free-2d-animation-agent] [--dry-run]
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "outputs" / ".space"
INCLUDE = ["app.py", "src", "actions", "expressions", "characters/protagonist"]
SKIP = {"work", "apose", "__pycache__", ".cache"}


def stage():
    shutil.rmtree(STAGE, ignore_errors=True)
    for rel in INCLUDE:
        src = ROOT / rel
        dst = STAGE / rel
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            shutil.copytree(src, dst, ignore=lambda d, names: [n for n in names if n in SKIP or n.endswith(".pyc")])
    shutil.copy2(ROOT / "deploy" / "space_requirements.txt", STAGE / "requirements.txt")
    shutil.copy2(ROOT / "deploy" / "space_README.md", STAGE / "README.md")
    size = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file()) / 1e6
    print(f"staged {sum(1 for p in STAGE.rglob('*') if p.is_file())} files, {size:.1f} MB -> {STAGE}")


def upload(space):
    from huggingface_hub import HfApi
    api = HfApi()
    print("logged in as", api.whoami()["name"])
    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=space, repo_type="space",
                      commit_message="Deploy drift-free animation agent")
    print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="Arcyadav/drift-free-2d-animation-agent")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    stage()
    if not a.dry_run:
        upload(a.space)
