# Reuse test — second character (male), 2026-10-18

Input: AI-generated images of a deliberately different character — dark tee, olive cargo pants,
short black hair, no ponytail (protagonist: white tee, blue jeans, curly ponytail):
front view, side view, front smile / sad / shocked, side smile + eyes_closed, open hand (8 images).

| Stage | Out of the box | Change needed (general rules, no per-character tuning) |
|---|---|---|
| Sheet Cleaner | ✅ | — |
| Keypoints (RTMPose), front + side | ✅ 0 issues | — |
| Palette / View Checker | ❌ assumed a light tee + blue jeans | garments named by body position when colour rules do not fit; hair and top may share a colour |
| Part splitter (front + side) | ❌ dark = hair; ponytail / strand / scrunchie required | 'dark = hair' only in the head zone when the top is dark; ponytail and strands optional |
| Hidden fill | ⚠ seam streaks on a dark tee | seam bleed under later parts; deeper fill under sleeves (both characters) |
| Face Cutter | ⚠ profile eye found on the dark collar; thick brows taken as eyes | eye search ignores the brow zone; profile eye guided by the pose detector's eye keypoint; per-character recipes, missing expressions → neutral |
| Hand swap (open hand) | ✅ | — |
| **Actions: wave, idle, look_down, react** (front) | ✅ **zero new animation work** | — |
| walk, walk_happy (side) | ⚠ moves correctly, looks cut apart on the dark tee | sleeve cut needs a visible seam / layered art |

Result: after making 6 rules general, the second character was onboarded automatically and performs
all of the protagonist's actions with the same motion files (planted-foot slide 0.0 px in his walk).
The protagonist's outputs are unchanged (her face layers pixel-identical; all tests pass).

Limits seen:
- **His side-view walk is not presentable**, so it's left out of the showcase. It works mechanically (foot slide 0.0 px),
  but on his dark tee the sleeve seam is invisible, so the sleeve piece was cut far too large (half his back). While walking,
  his torso reads as cut apart from his trousers. Found and fixed during the test: cracks between parts (→ critic + seal)
  and 4 feet (both shoes drawn in his side view → shoe split). The remaining fix needs layered art or a
  manual correction of the sleeve cut.
- **His shocked face does not composite well:** the edit drew the eyes ~10 px lower and much taller and
  raised the brows into the hairline — features overlap the layer zones, so eyes / brows / mouth layers
  cannot reassemble it cleanly (falls back to a mixed look). A whole-face swap would be the fix.
- Extreme arm-raising poses (react) show the A-pose gap between arm and body as light patches at the
  armpits — very visible on dark tops.
- Expressions and the walk need the character's own face and side images (8 images here).
