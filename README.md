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
4. **`keyframes`** — detects edit boundaries via frame-differencing in the
   viewport region, saves keyframe PNGs, and crops+saves the header/status
   bar / F9 redo-panel region for later OCR.
5. **`manifest`** — stitches keyframes + transcript segments into
   `manifest.json`: a list of `{t_start, t_end, narration, frame_before,
   frame_after, panel_before, panel_after}` segments — the unit of work for
   the (not-yet-built) VLM reconstruction step.

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
blender-trace keyframes data/<video_id> --panel-box 0.55 0.0 1.0 0.06
blender-trace manifest data/<video_id>
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
