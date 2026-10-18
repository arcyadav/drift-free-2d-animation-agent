"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS_RAW = ROOT / "assets" / "raw"
CHARACTERS = ROOT / "characters"
# raw character sheet per character (the protagonist's lives directly in assets/raw)
RAW_DIRS = {"protagonist": ASSETS_RAW, "male": ASSETS_RAW / "male"}


def raw_dir(character):
    return RAW_DIRS.get(character, ASSETS_RAW / character)

# Normalised canvas every view is placed on (rig works at this resolution;
# final render is downscaled to 512x768).
CANVAS_W, CANVAS_H = 1024, 1536
TARGET_HEIGHT = 1344      # top of hair → bottom of shoes, in px
FEET_Y = 1472             # baseline: bottom of shoes sits on this row
CENTER_X = CANVAS_W // 2  # legs centroid is placed on this column

VIEWS = ["front", "3q_left", "side_left", "back"]
