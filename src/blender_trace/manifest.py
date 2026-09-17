"""
Build manifest.json: the actual unit of work for the downstream VLM
code-reconstruction agent. Each entry is one candidate "edit segment" with
the narration that overlaps it and a before/after frame pair.

Output: <video_dir>/manifest.json
    [{
        "t_start": 12.4, "t_end": 15.9,
        "narration": "now I'll add a bevel to soften this edge",
        "frame_before": "frames/frame_12.40.png",
        "frame_after": "frames/frame_15.90.png",
        "panel_before": "panels/panel_12.40.png",
        "panel_after": "panels/panel_15.90.png"
    }, ...]

Two ways to decide segment boundaries:

- "narration" (recommended, default): split on narration content -- see
  segment.py. Only needs transcript.json + video.mp4; samples frames
  directly at the resulting boundaries.
- "visual": the original approach -- boundaries come from keyframes.json
  (viewport pixel-diffing, see keyframes.py). Kept for comparison; the pilot
  run (see README) found it unreliable across real videos -- pixel-diff
  can't distinguish camera movement from an actual edit, and gets
  contaminated by talking-head overlays. Requires running the `keyframes`
  command first.

This is intentionally *not* the finished dataset -- it's the raw candidate
segment list. The next stage (not implemented yet -- needs an actual Blender +
VLM agent loop) is: for each segment, try to synthesize bmesh code that
transforms frame_before's mesh state into frame_after's, verified by
rendering and comparing. Only segments where that verification succeeds
become actual training examples.
"""
import json
from pathlib import Path

from . import keyframes as keyframes_mod
from . import segment as segment_mod


def overlapping_narration(transcript, t_start, t_end):
    return segment_mod.overlapping_narration(transcript, t_start, t_end)


def build_manifest(transcript: list[dict], keyframes: list[dict]) -> list[dict]:
    """Visual method: pair up consecutive pixel-diff keyframes."""
    manifest = []
    for kf_before, kf_after in zip(keyframes[:-1], keyframes[1:]):
        narration = overlapping_narration(transcript, kf_before["t"], kf_after["t"])
        manifest.append({
            "t_start": kf_before["t"],
            "t_end": kf_after["t"],
            "narration": narration,
            "frame_before": kf_before["frame_path"],
            "frame_after": kf_after["frame_path"],
            "panel_before": kf_before["panel_path"],
            "panel_after": kf_after["panel_path"],
        })
    return manifest


def build_narration_manifest(
    video_dir: Path,
    panel_box: tuple[float, float, float, float],
    min_segment_s: float = 1.5,
) -> list[dict]:
    """Narration method: segment.py decides boundaries from transcript
    content, then frames are sampled directly at those boundaries -- no
    pixel-diffing, so no dependency on keyframes.json."""
    transcript = json.loads((video_dir / "transcript.json").read_text())
    segments = segment_mod.segment_transcript(transcript, min_segment_s)

    timestamps = sorted({t for seg in segments for t in (seg["t_start"], seg["t_end"])})
    frame_map = keyframes_mod.extract_frames_at(video_dir, timestamps, panel_box)

    manifest = []
    for seg in segments:
        before = frame_map.get(seg["t_start"])
        after = frame_map.get(seg["t_end"])
        if before is None or after is None:
            continue  # ran past end of video (e.g. transcript outlasts the file)
        manifest.append({
            "t_start": seg["t_start"],
            "t_end": seg["t_end"],
            "narration": seg["narration"],
            "frame_before": before["frame_path"],
            "frame_after": after["frame_path"],
            "panel_before": before["panel_path"],
            "panel_after": after["panel_path"],
        })
    return manifest


def main(
    video_dir: Path,
    method: str = "narration",
    panel_box: tuple[float, float, float, float] | None = None,
    min_segment_s: float = 1.5,
):
    if method == "narration":
        manifest = build_narration_manifest(video_dir, panel_box, min_segment_s)
    elif method == "visual":
        transcript = json.loads((video_dir / "transcript.json").read_text())
        keyframes = json.loads((video_dir / "keyframes.json").read_text())
        if len(keyframes) < 2:
            print("Fewer than 2 keyframes found -- check keyframes.py settings.")
            return
        manifest = build_manifest(transcript, keyframes)
    else:
        raise ValueError(f"unknown method: {method!r} (expected 'narration' or 'visual')")

    out_path = video_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    empty_narration = sum(1 for m in manifest if not m["narration"])
    print(f"Wrote {len(manifest)} segments to {out_path} "
          f"({empty_narration} with no overlapping narration -- expect to discard or "
          f"merge these with a neighboring segment)")
