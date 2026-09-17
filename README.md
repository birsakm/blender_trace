# BlenderTrace

Mining verified Blender tutorial "edit traces" — narration + before/after
viewport frames + reconstructed `bpy`/`bmesh` code — as training data for
LLM/VLM Blender modelers.

## Motivation

Existing code-generation approaches to 3D modeling (e.g. LL3M, LAM) top out
in quality because Blender artists don't actually write `bpy` code — they
work click-by-click in the GUI. There's no natural corpus of high-quality
modeling code to train or fine-tune on.

There is, however, a large corpus of high-quality Blender tutorials on
YouTube, where artists narrate what they're doing in detail while working.
The bet here: mine that narration (and, where narration alone is ambiguous,
paired before/after viewport frames + F9 redo-panel OCR) to reconstruct the
`bpy`/`bmesh` operations an artist actually performed, verify the
reconstruction by rendering and comparing against the real frame, and keep
only verified segments as training examples. High-level `bpy.ops` calls map
to discrete GUI actions; low-level `bmesh` access (vertex/edge/face data)
covers the operations with no direct GUI equivalent — both are likely
necessary.

## Status

Early scaffolding. What exists right now is the **data-mining side**: live
video discovery, download, transcription, keyframe/panel extraction, and
manifest building. It produces `manifest.json` per video — a list of
candidate `{narration, frame_before, frame_after}` segments — which is *not*
yet a training dataset. The **verification loop** (agent reconstructs
`bpy`/`bmesh` code per segment → render in headless Blender → compare against
`frame_after` → keep only matches) is the next stage and is not implemented
yet.

This machine has real internet access, 4x A100 GPUs, and several `bpy_*`
conda environments (4.0 through 5.0.1) already available — the verification
loop's headless-render step can run directly here.

### First experiment: findings (Josh Gambrell, "This Shape Is Easy!")

Ran the full pipeline against one real video end to end. Fixed three real
bugs surfaced by doing that (not just theoretical edge cases):

- `download.py` defaulted to `yt-dlp`'s "best" format, which is often
  AV1-only; this OpenCV build has no AV1 decoder, so `cv2.VideoCapture.read()`
  silently failed on every frame. Now prefers avc1/h264 explicitly.
- `keyframes.py` and `transcribe.py`'s Whisper fallback resolved the video
  file via `glob("video.*")`, which also matches the sibling `.srt`/`.vtt`
  caption files `download.py` writes — glob order isn't alphabetical, so it
  could silently open a subtitle file as if it were the video. Now points at
  `video.mp4` directly (which is what `download.py` always merges to).
- `transcribe.py`'s SRT parser joined raw cue text naively. YouTube's
  auto-captions use a rolling multi-line window (each cue repeats the
  previous line and appends a new one), so naive joining duplicated most
  words up to 4x. Fixed by deduping on each cue's last line.

Two non-bug findings that matter more for the research direction:

- **Narration quality is excellent** — confirms the core hypothesis. This
  artist narrates specific key commands ("F key", "shift R", "ctrl 2",
  "shift G, co-planar") in a way that's plausibly reconstructable into
  discrete `bpy`/`bmesh` calls.
- **Viewport pixel-diff is an unreliable segmentation signal.** The original
  boundary detector compared each sampled frame only to the *previous
  sample*, not the last saved keyframe, so a long run of small cumulative
  edits (~10 distinct operations over 71s in one case) never crossed the
  diff threshold and collapsed into one unusable segment. Fixed to diff
  against the last keyframe instead, which helped, but a deeper problem
  remains even after that fix and after lowering the threshold: **this
  channel orbits/zooms the camera constantly while narrating**, and camera
  movement alone produces viewport pixel-diffs just as large as an actual
  mesh edit — the detector cannot tell them apart. Confirmed by inspecting
  frame pairs directly: a segment labeled "delete out this corner" showed a
  completely different camera angle/zoom between `frame_before` and
  `frame_after`, not a legible before/after of the described edit.

**Implication for next steps:** pixel-diff-based visual segmentation should
not be the primary segmentation signal. Narration-driven segmentation
(splitting on sentence/clause boundaries that name a specific action or
tool) is likely a better primary signal, with frames pulled at those
boundaries for verification rather than for detecting the boundaries
themselves. This is a design change to `keyframes.py`/`manifest.py`, not yet
implemented.

Separately: the F9 redo-panel and the persistent Properties-editor panel
(which shows live modifier values) occupy *different* screen regions and
neither is reliably present on its own — `keyframes.py` only supports one
crop box per run. See `configs/panel_boxes.yaml` for the calibration used
here and the tradeoff it makes.

### Second and third videos: camera movement is video-style-dependent, not universal

Ran two more videos to check whether the camera-movement problem generalizes:

- **Josh Gambrell, "Easy PLUGS in Blender"** (same channel, different video):
  segments here were clean. Static camera during editing, and the frame
  pairs directly show the named operation — e.g. a segment narrated "give
  the cylinder a couple more segments, 64" pairs a frame mid-`Add > Mesh >
  Cylinder` with the resulting cylinder and its "Add Cylinder" redo panel;
  another narrated "select one, shift G, co-planar" pairs cleanly with a
  frame showing the "Select Similar → Coplanar" menu literally open. No
  camera-movement contamination in the segments checked.
- **Grant Abbitt, "Make a Low Poly Hot Rod"** (different channel): a
  different problem entirely — a talking-head webcam overlay (bottom-right
  quadrant) plus a semi-transparent reference blueprint image in the
  viewport. The talking head overlaps the region the keyframe diff
  computation treats as "viewport" (left 70% of frame), so it's a source of
  constant, edit-irrelevant pixel motion that needs excluding, similar to
  how the panel-box already gets excluded from the diff.

Net conclusion: **video "style" varies enormously even within one creator's
back-catalog**, and different styles break the current pixel-diff detector
in different ways (orbit-heavy explainers vs. talking-head overlays vs.
clean static-camera step-through). A production version of this pipeline
likely needs either (a) a per-video/per-channel style classifier that picks
detector settings (or an exclusion mask) accordingly, or (b) to drop
pixel-diff as the primary boundary signal in favor of narration-driven
segmentation, which is unaffected by any of this.

### Narration-driven segmentation (implemented)

Acted on the conclusion above: `segment.py` now decides segment boundaries
from transcript content instead of pixel-diffing. It looks for cues that
name a specific hotkey or tool/operator (`ACTION_CUES` -- e.g. "shift R",
"bevel", "grid fill", "unwrap") and starts a new segment at the first cue
that mentions one, provided enough time (`--min-segment-s`, default 1.5s)
has passed since the last split. `manifest.py`'s narration method (now the
default for `blender-trace manifest`/`pipeline`) uses this directly:
`keyframes.py`'s pixel-diff detector (the `keyframes` command, `--method
visual`) is no longer required and is kept only for comparison.

This is a heuristic keyword list, not a parser, and it will always be
incomplete for a domain with this many named operators -- expect to keep
extending `ACTION_CUES` as more videos get processed. Even so, re-running it
against all three pilot videos fixed the specific failure found earlier:
the 71.5-second, ~10-operation blob in video 1 became 9 segments of 1.6-11.5s
each, and spot-checking one of them ("delete out this corner, fill these
faces in with the F key") showed the `frame_after` with the exact "New
Edge/Face from Vertices (F)" menu entry open -- a literal, verifiable match
to the narration, and unaffected by camera movement since nothing here
looks at pixel differences at all.

## Setup

```bash
conda activate blender_trace
pip install -e ".[dev]"
# ffmpeg must be on PATH (already present on this machine: /usr/bin/ffmpeg)
```

## Legal / ethical note (read before running at scale)

- This is meant for small-scale research analysis, not redistribution. Keep
  raw video files local and private.
- Prefer downloading only what you need: for many videos, official/auto
  captions are enough for the narration side without ever pulling the full
  video; only download video for the tutorials you actually plan to run the
  visual pipeline on.
- If a derived dataset is eventually published, publish transcripts / frame
  embeddings / mined code snippets, not raw video, and credit the source
  channels.
- Respect per-video restrictions (some creators disable download / age-restrict
  content) — `yt-dlp` will simply fail on those; don't route around it.

## Pipeline

1. **`discover-channel` / `discover-search`** — live YouTube discovery via
   `yt-dlp`. Never hand-pick a video link and assume it's still live —
   channels get renamed, videos get taken down or region-locked. Always
   discover live or run `discover-check` on a specific URL first.
2. **`download`** — downloads video (capped at 1080p — enough to read the F9
   redo-panel, no more) + subtitles if available, into `data/<video_id>/`.
3. **`transcribe`** — if captions exist, normalizes them; otherwise runs
   local Whisper (`faster-whisper`) on the extracted audio.
4. **`manifest`** — decides segment boundaries and builds `manifest.json`: a
   list of `{t_start, t_end, narration, frame_before, frame_after,
   panel_before, panel_after}` segments — the unit of work for the
   (not-yet-built) VLM reconstruction step. Two methods (`--method`):
   - `narration` (default) — segments on transcript content (see
     `segment.py`), sampling frames directly at the resulting boundaries.
     Only needs `transcribe` to have run first.
   - `visual` — the original approach: pairs up consecutive keyframes from
     a prior `keyframes` run (frame-differencing in the viewport region).
     Kept for comparison; the pilot run (see "Status" below) found it
     unreliable across real videos, so it's no longer the default.

```bash
# 1. Find what's currently live on a seed channel (see configs/channels.yaml)
blender-trace discover-channel "https://www.youtube.com/@JoshGambrell/videos" \
    --keyword "hard surface" --out discovered.json

# 2. Sanity-check one specific URL before committing to it
blender-trace discover-check "https://www.youtube.com/watch?v=<id>"

# 3. Run the full pipeline for one video (recommended for the first 2-3 videos
#    per channel, while calibrating --panel-box)
blender-trace pipeline "https://www.youtube.com/watch?v=<id>" \
    --panel-box 0.55 0.0 1.0 0.06

# Or run stages individually once settings are calibrated for a channel:
blender-trace download "<url>"
blender-trace transcribe data/<video_id>
blender-trace manifest data/<video_id> --panel-box 0.55 0.0 1.0 0.06
```

`manifest.json` is the artifact to actually inspect: for each segment, look
at `frame_before`/`frame_after` side by side with the narration text and ask
whether a person could plausibly write the edit from that information alone
— that's the go/no-go signal before touching an agent or fine-tuning at all.

## Calibrating the panel crop

The F9 redo-panel / header text location differs per channel (screen
resolution, custom UI layout, side-panel width, etc.). `--panel-box` takes
fractional coordinates (0-1) of the frame; open one exported keyframe in an
image viewer, eyeball the box around the header/redo panel text, and set it
once per channel in `configs/panel_boxes.yaml` — most channels keep a
consistent layout across their videos, so this is a one-time cost per
source, not per video.

## Seed channels

See `configs/channels.yaml` — live-verified via `yt-dlp` as of the date
noted in that file. Re-verify with `discover-check` before a large batch run
rather than trusting it blindly; channels get renamed or merged (e.g. Josh
Gambrell and Ponte Ryuurui both currently publish under the "Blender Bros"
team brand rather than their individual old handles).

## Tests

```bash
pytest
```

Only the manifest-building logic is unit-testable without network/Blender;
the discovery/download/transcribe/keyframe stages need real network or media
files and are meant to be run and eyeballed manually per the pipeline above.
