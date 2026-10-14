# Assets — where to put your images

Use these **exact file names** (PNG). The onboarding code looks for them by name.
Rules for every image: plain cream background, no text labels, flat colours, same character scale.

## raw/turnaround/   (full body, same scale)
- `front.png`        ✅ already have
- `3q_left.png`      character turned 3/4 toward the viewer's left
- `side_left.png`    full profile facing left
- `back.png`

## raw/apose/   (full body, arms ~25° out from the body)
- `front.png`        ✅ already have
- `side.png`

## raw/faces/  — full-body edits of the view image, ONLY the face changed (head must not move)

## raw/faces/front/   (edit of turnaround/front.png)
- `neutral.png` ✅ (copy of front view), `eyes_half.png`, `eyes_closed.png`, `eyes_down.png`
- `smile.png`, `sad.png`, `shocked.png`, `worried.png`

## raw/faces/3q_left/   (edit of turnaround/3q_left.png)
- `neutral.png` ✅, `eyes_closed.png`, `sad.png`

## raw/faces/side_left/   (edit of turnaround/side_left.png)
- `neutral.png` ✅, `eyes_closed.png`

## raw/hands/
- `open_front.png`, `relaxed_front.png`, `fist_front.png`, `relaxed_side.png`

## raw/reference/
- Anything else useful: original generation prompt (`prompt.txt`), colour notes, rejected attempts.

## motion/
- `mixamo/`  → Mixamo downloads (Walking, Sitting) as Collada .dae zips (In Place off, Without Skin, 30 fps)
- `video/`   → your phone clips, e.g. `wave.mp4` (3–5 s, full body, plain wall, still camera)
