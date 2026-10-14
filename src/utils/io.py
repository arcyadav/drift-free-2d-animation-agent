"""File writes that survive OneDrive.

The project lives in OneDrive, which briefly locks freshly written files while it
uploads them; rewriting one then fails with OSError (Errno 22 / 13). Write to a temp
file next to the target and swap it in, retrying a few times.
"""
import os
import time
from pathlib import Path

from PIL import Image

RETRIES = 8
WAIT_S = 0.5


def _atomic(path, write):
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    for i in range(RETRIES):
        try:
            write(tmp)
            os.replace(tmp, path)
            return path
        except OSError:
            if i == RETRIES - 1:
                raise
            time.sleep(WAIT_S * (i + 1))
    return path


def save_image(img, path):
    """img: PIL image or numpy array."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return _atomic(path, lambda p: img.save(p, format="PNG"))


def save_text(text, path):
    return _atomic(path, lambda p: Path(p).write_text(text, encoding="utf-8"))
