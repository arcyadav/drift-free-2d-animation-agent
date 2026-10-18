# Reuse test — second character (male, front view only), 2026-10-18

Input: one AI-generated front image (`assets/raw/male/turnaround/front.png`): dark tee, olive cargo
pants, short black hair, sneakers — deliberately unlike the protagonist (white tee, blue jeans,
curly ponytail).

| Stage | Out of the box | Change needed (general, not per-character) |
|---|---|---|
| Sheet Cleaner | ✅ | — |
| Keypoints (RTMPose) | ✅ 0 issues | — |
| Palette / View Checker | ❌ assumed light tee + blue jeans | garments named by body position when colour rules do not fit; hair and top may share a colour |
| Part splitter | ❌ dark = hair; ponytail / strand / scrunchie required | 'dark = hair' only in the head zone when the top is dark; ponytail and strands optional |
| Hidden fill | ⚠ seam streaks on a dark tee | generic seam bleed (all characters); deeper fill under sleeves |
| Face Cutter | ⚠ needed expression images | per-character recipes; missing expressions fall back to neutral; blinks drawn from neutral eyes |
| **Actions (wave, idle, look_down, react)** | ✅ **zero new animation work** | — |

Result: after making 4 rules general (no character-specific tuning), the second character was onboarded
from a single image and performs the protagonist's front actions unchanged. The protagonist's outputs
and tests are unchanged.

Remaining limits seen: extreme arm-raising poses (react) expose the background gap between arm and body
of the A-pose drawing as light patches at the armpits — very visible on dark tops. Expressions need the
character's own face images; the side view (walk) needs a side image.
