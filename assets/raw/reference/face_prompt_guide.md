# Face image prompt guide (ChatGPT image editing)

Tool depends on which tool made the view (edits keep the style of their input's family):
- **Front view → ChatGPT** — keeps the original eye style (big round eyes, thick dark upper lash line); Gemini redraws them smaller.
- **3/4 + side views → Gemini** — those views were made in Gemini, so Gemini edits match them.
- **Hands → Gemini** — simple shapes, style not critical.

ChatGPT has a daily image limit → priority order (held expressions first):
1. today (ChatGPT): front `smile`, `sad`, `eyes_down`
2. tomorrow (ChatGPT): front `shocked`
3. today (Gemini): 3/4 `eyes_closed`, 3/4 `sad`, side `eyes_closed`, 4 hands
4. optional (ChatGPT, spare credits): front `eyes_half`, `eyes_closed` — blink frames last ~0.08 s, Gemini versions are fine

## Rules for every image
- Upload the **original view image** every time (not an edited result) → no drift build-up.
- One expression per request; start a **new chat** if results start drifting.
- Output must be **portrait 1024×1536** (front) / same framing as the input, full body visible.
- Save as `assets/raw/faces/<view>/<name>.png`, then check:
  `python -m src.utils.check_face assets/raw/faces/<view>/<name>.png`
  → want shift 0,0 and no changes in torso/legs.

## Base prompt (paste first, then the expression line)
> Edit this image. Keep everything exactly the same: full body, pose, clothes, hair, curls, ponytail, head position, head size, head angle, framing, image size, art style, line style, colours and background. Keep the same large round eyes with the thick dark upper lash line and the same eyebrow style. Do not move, tilt or resize the head. Output the full image at the same size. Change ONLY the facial expression to:

## Front — upload `turnaround/front.png` (ChatGPT)

| Save as | Expression line | Status |
|---|---|---|
| `neutral.png` | — (copy of front.png) | ✅ |
| `worried.png` | inner eyebrow ends raised and slightly drawn together, eyes open a little wider looking ahead, small mouth with corners slightly down; anxious, not angry | ✅ ChatGPT |
| `smile.png` | **a clear, warm, happy smile: mouth corners lifted noticeably, lips closed or slightly parted, cheeks lifted, lower eyelids slightly raised so the eyes curve happily** | ✅ ChatGPT (clear smile) |
| `eyes_half.png` | eyelids half closed, as in the middle of a blink; eyebrows and mouth unchanged (neutral) | optional redo — Gemini version OK |
| `eyes_closed.png` | eyes fully closed and relaxed, as if blinking, lashes as a soft curved line; eyebrows and mouth unchanged (neutral) | optional redo — Gemini version OK |
| `eyes_down.png` | eyes looking down toward the floor, upper eyelids lowered to follow the gaze; head stays facing forward, not tilted; eyebrows and mouth neutral | ✅ ChatGPT |
| `sad.png` | sad: inner ends of the eyebrows raised, eyes slightly lowered, small downturned mouth | ✅ ChatGPT (eyes from eyes_down in recipe) |
| `shocked.png` | shocked: eyes wide open, eyebrows raised high, mouth open in a small "O" shape | ✅ ChatGPT |

For `eyes_half`, `eyes_closed`, `eyes_down`: eyebrows and mouth must stay **exactly neutral** — these images only supply the eye layer.

## 3/4 left — upload `turnaround/3q_left.png` (Gemini)
Add to the base prompt: *"Keep the three-quarter head angle exactly as it is."* — and **remove** the "large round eyes / thick lash line" sentence (that describes the front view); use *"Keep the same eye shape and style"* instead.

| Save as | Expression line |
|---|---|
| `neutral.png` | — (copy of 3q_left.png) ✅ |
| `eyes_closed.png` | eyes fully closed and relaxed, as if blinking; eyebrows and mouth unchanged ✅ |
| `sad.png` | sad: inner ends of the eyebrows raised, eyes slightly lowered, small downturned mouth ✅ |

## Side left — upload `turnaround/side_left.png` (Gemini)
Add to the base prompt: *"Keep the exact side profile; only the one visible eye changes."* — and replace the eye-style sentence with *"Keep the same eye shape and style"* (as for 3/4).

| Save as | Expression line |
|---|---|
| `neutral.png` | — (copy of side_left.png) ✅ |
| `eyes_closed.png` | the visible eye fully closed and relaxed, as if blinking; eyebrow and mouth unchanged ✅ |

## If something goes wrong
- **Head moved / framing changed** → regenerate; add "same image size and framing as the input".
- **Looks angry instead of worried/sad** → add "Do NOT lower or furrow the inner eyebrows; no frown lines, no narrowed eyes."
- **Eye style changed** → repeat "Keep the same large round eyes with the thick dark upper lash line".
- **Hair/curls changed** → add "Do not redraw the hair".

## Hands (Gemini) — swap images for finger poses the rig cannot bend
The relaxed hands already exist in the view images → only the open hand is needed now.

| Save as (`assets/raw/hands/`) | Upload | Hand | Used in | Status |
|---|---|---|---|---|
| `relaxed_front.png` | — | already in front.png | idle, look_down | ✅ (no image needed) |
| `relaxed_side.png` | — | already in side_left.png | walk | ✅ (no image needed) |
| `open_front.png` | front.png | open palm facing viewer, fingers straight + slightly spread, **still pointing down, arm not moved** | wave | ✅ (her LEFT hand, viewer's right) |
| `fist_front.png` | front.png | closed fist, arm not moved | search_pockets (stretch) | ✅ (her LEFT hand) |

Prompt: *Edit this image. Keep everything exactly the same: full body, pose, arm positions, clothes, hair, head, framing, image size, art style, colours and background. Change ONLY her right hand (the hand on the viewer's left): make it an open hand with the palm facing the viewer, fingers straight and slightly spread, still pointing downward along the arm, the same size and in the same position. Do not raise or move the arm.*

Why pointing down: the rig raises the arm and rotates the hand with it during the wave; the swap image must sit exactly where the relaxed hand is.

## Status 2026-10-14
All face images done. Gemini outputs (3/4 eyes_closed + sad, side eyes_closed) are ~3x softer than
neutral and 3/4 ones are 847 px wide (resampled, 0.1%) — colour shift dE 1-2. Accepted: only the eye /
brow / mouth layers are cut from them, Face Cutter sharpens the cut layers, final render is half size.
