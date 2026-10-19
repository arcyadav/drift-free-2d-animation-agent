# Evaluation — Drift-Free 2D Character Animation Agent

All numbers below were measured on the author's laptop (CPU only, no GPU) in October 2026. Each
section names the script and output file that produced them, so every number can be reproduced.

## Summary

| Question | Result |
|---|---|
| Does the character stay the same? | **Yes, by construction**: every frame is the drawing's own pixels, moved. Head width across our walk: −0.6% vs the drawing (range −2.1…0%). A commercial AI video (Kling) of the same character drew her head/hair **~11% wider in every frame** (10.1…12.7%). |
| Does the agent understand requests? | 46 test requests: **rules only 97.8%**, **rules + local LLM 100%** fully right. (The test set was written by the developer; see the caveat in §1.) |
| Does the motion look physically right? | Planted-foot slide **3.38 px → 0.0 px** with the foot lock. Walk pace and step length match the AI-video reference: 1.33 vs 1.39 steps/s, 0.60 vs 0.59 leg-lengths. |
| Does it catch its own mistakes? | Critic on 21 clips: **16 pass first time, 4 had cracks that were found and sealed, 1 needs an image the user must add** (named by the critic). |
| Does it work for another character? | A second, very different character was onboarded from 8 images; his front actions transfer with **zero new animation work**. His side walk is not presentable (§5). |
| Cost | Free per clip, laptop CPU: median render **12.8 s** (front) / **18.6 s** (walk), critic **15 s**. |

---

## 1. Planner (request → plan)
Script `src/agent/eval_planner.py` → `outputs/planner_eval.json`; test set `tests/planner_cases.json` (46 requests).

| System | Fully right | Action | Expression | Speed | Direction | Seconds | Shot | LLM calls | Time |
|---|---|---|---|---|---|---|---|---|---|
| Rules only | **97.8%** (45/46) | 97.4% | 100% | 100% | 100% | 100% | 100% | 0 | 0.1 s total |
| Rules + `qwen2.5:3b` | **100%** (46/46) | 100% | 100% | 100% | 100% | 100% | 100% | 22 of 46 | ~5 s per LLM call (CPU) |

- The one rules-only error: *"she hurries to catch the bus"* → `wave` instead of `walk`. The LLM fixes it.
- The LLM only fills slots the rules leave open, and only with values from the library (constrained JSON).
  Every plan is then validated and repaired (e.g. a facial expression asked for in side view becomes a body-language walk).
- Earlier version (40 requests): rules 88%, rules + LLM 98%. Rules were added as failures were found.
- **Caveat:** the developer wrote this test set while building the rules, so these numbers are optimistic. A
  held-out set written by someone else would be the fair test. It wasn't collected in time.

## 2. Motion quality
### 2.1 Foot lock
Walker report → `outputs/walk.json` (5 s walk, 120 frames).

| | mean slide | max slide |
|---|---|---|
| Planted foot, motion only (no lock) | 0.46 px/frame | 3.38 px/frame |
| **With foot lock (2-bone IK + pelvis drop)** | **0.0** | **0.0** |

### 2.2 Gait vs. an AI-video walk of the same character
Script `src/critic/gait_metrics.py` (pose detector on the Kling clip; joint paths of our motion).

| | steps / s | step length (leg lengths) | knee bend (°) | body bob (leg lengths) | arm swing (°) |
|---|---|---|---|---|---|
| Kling (AI video) | 1.39 | 0.59 | 1 – 53 | 0.063 | 16 |
| **Ours** | 1.33 | 0.60 | 3 – 49 | 0.045 | 25 |

Pace, stride and knee range match closely. Ours bobs a little less and swings the arms more. These were the
walk's tuning targets (the keyed walk was built from foot paths, then compared with this reference).

## 3. Identity (drift)
Script `src/critic/identity_compare.py` → `docs/identity_compare.json`. Silhouette widths ÷ body height,
compared with the original side-view drawing. Ours: 30 frames of the walk (exact silhouettes from the critic's
green-key render). Kling: 31 frames of its 5 s clip (flat backdrop).

| Width vs. the drawing | **Ours (rig)** | Kling (AI video) |
|---|---|---|
| Head / hair band (2–10% of height) | **−0.6%** (−2.1 … 0.0) | **+11.1%** (+10.1 … +12.7), already +10.4% in frame 1 |
| Torso band (25–40%) | +2.2% (−3.4 … +8.6) | +1.8% (−4.1 … +16.2) |
| Body height wobble (sd) | 0.8% | 0.5% |

- The torso band changes in both clips because the swinging arm enters it, so it says nothing about identity.
- The head band is where the difference shows. By eye: **Kling made her hair fuller and rounder on top and her
  face slightly rounder**, while keeping the outfit and pose. It is consistent within the clip (not flicker) but
  it is a different-looking character from the drawing. Ours stays within 2% (head motion only).
- **Correction:** earlier project notes said Kling "widened her body by 8–13%". Re-measured with this script,
  the change is in the **head/hair (~11%)**; the torso matches on average.
- A colour test (share of off-model pixel colours) was tried for this comparison and dropped: on compressed
  video it called both clips 0% off-model, so it can't separate them.
- Not done: DINOv2/CLIP embedding similarity (no GPU in the setup; the spec planned it for a stylize pass that
  wasn't built).

## 4. Critic (self-check + fix)
Script `src/critic/eval_critic.py` → `docs/critic_eval.json`. 21 clips = the 14 showcase requests (protagonist)
+ 7 requests for the second character.

| Outcome | Clips |
|---|---|
| Pass first time | 16 |
| Cracks found → sealed → pass on re-check | 4 (her shocked reaction, her cheerful walk, his two walks) |
| Unfixable by re-rendering | 1 (his side "sad" face is missing; the critic names `faces/side_left/sad.png`) |
| Off-model colours (critic's identity check, raw frames) | ≤ 1.2% in every clip |
| Critic time per clip | median 15 s (max 43 s) |

How the crack check was made trustworthy (each rule came from a false alarm, checked by eye):
1. Magenta key colour → matched skin and sneakers → **pure green key**.
2. Gaps between curls in close-ups were flagged → **holes already in the rest pose are art**, not cracks.
3. The triangle between the legs closed by the feet was flagged → **cracks are thin slivers** (≤ 8 px half-width,
   area ≥ 8 × width²); fat openings are real.
4. Holes are only cracks if they lie **between two parts** (a part-ID render shows who surrounds them).

**A fix that didn't work:** moving less. His walk still cracked at half the arm swing, and the shocked
reaction cracked *more* with arms half raised. Cracks come from how the drawing was cut, not from the motion.
So the fix is the seal (inpaint only those slivers), plus better cutting at onboarding.

**Blind spot:** light pixels cut *into* a part at onboarding (his A-pose armpit area) aren't background, so
the crack check can't see them.

## 5. Reuse test (second character)
Details: `docs/reuse_test.md`.
- Input: 8 AI-generated images of a deliberately different character (dark tee, olive cargo pants, short hair).
- Onboarding needed **6 rules generalised** (garments by body position, dark top, optional ponytail and strands,
  seam bleed, brow-free eye search, profile eye guided by the pose keypoint, per-character expression recipes).
  The protagonist's outputs were unchanged (her face layers pixel-identical; all tests pass).
- **Front actions (wave, idle, look down, react): transfer with zero new animation work.**
- **Side walk:** moves correctly (foot slide 0.0 px), but it isn't shown. On his dark tee the sleeve seam is
  invisible, so the sleeve piece was cut far too large and his torso reads as cut apart from his trousers. Two bugs
  this test found were fixed (cracks → critic + seal; both shoes drawn → "4 feet" → shoe split). The rest
  needs layered art or a manual sleeve correction.
- His shocked face doesn't assemble cleanly from eye/brow/mouth layers (the generated image moved and
  enlarged the eyes), so it falls back to a mixed look.

## 6. Cost and time
Clip timings from the 14 showcase clips (`outputs/.static/clips/*.json`), laptop CPU:

| Step | Median |
|---|---|
| Plan (rules; the LLM result is cached after the first call) | ≈ 0 s (an uncached LLM call: ~5 s) |
| Critic | 15 s |
| Render, front clip (3–4 s) | 12.8 s |
| Render, walk (5 s) | 18.6 s |

- Cost per clip: **free**, with no GPU, no API and no credits. The commercial AI video used as the baseline is
  credit-based and runs on cloud GPUs.
- One showcase render recorded an outlier of several hours of wall time. It ran while the laptop was busy with
  other jobs (or asleep), so it's excluded from the medians.

## 7. Planned in the spec but not done (honest scope)
| Planned | Status |
|---|---|
| Blind human ratings vs AI video | not done (time) |
| VLM "looks odd?" critic vote, LPIPS flicker, DINOv2 identity | not done: no GPU in the setup, no stylize pass to guard |
| Stylize (AI redraw) pass | not built; moved to future work (Level 3) |
| Animated Drawings baseline | not run |
| Held-out planner test set by another person | not collected |
| 95% confidence intervals | not computed (one clip per condition) |

## 8. Main limits
- Puppet (cut-out) look: rigid sleeves and trousers, no cloth. AI video has richer frames but redraws the character.
- Onboarding cuts **flat** drawings, so hidden areas are guessed. Dark clothes expose the guesses (seams,
  armpits). Professional rigs avoid this with **layered art**, where each part is drawn whole. That is the most
  valuable next step.
- Side-view expressions need the character's own profile images; missing ones fall back to neutral (and the critic says so).
