"""
Combine transcript.json + keyframes.json into manifest.json: the actual unit
of work for the downstream VLM code-reconstruction agent. Each entry is one
candidate "edit segment" bounded by two consecutive keyframes, with the
narration that overlaps that time window.

Output: <video_dir>/manifest.json
    [{
        "t_start": 12.4, "t_end": 15.9,
        "narration": "now I'll add a bevel to soften this edge",
        "frame_before": "frames/frame_12.40.png",
        "frame_after": "frames/frame_15.90.png",
        "panel_before": "panels/panel_12.40.png",
        "panel_after": "panels/panel_15.90.png"
    }, ...]

This is intentionally *not* the finished dataset -- it's the raw candidate
segment list. The next stage (not implemented yet -- needs an actual Blender +
VLM agent loop) is: for each segment, try to synthesize bmesh code that
transforms frame_before's mesh state into frame_after's, verified by
rendering and comparing. Only segments where that verification succeeds
become actual training examples.
"""
import json
from pathlib import Path


def overlapping_narration(transcript, t_start, t_end):
    texts = [
        seg["text"] for seg in transcript
        if seg["t_end"] > t_start and seg["t_start"] < t_end
    ]
    return " ".join(texts).strip()


def build_manifest(transcript: list[dict], keyframes: list[dict]) -> list[dict]:
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


def main(video_dir: Path):
    transcript = json.loads((video_dir / "transcript.json").read_text())
    keyframes = json.loads((video_dir / "keyframes.json").read_text())

    if len(keyframes) < 2:
        print("Fewer than 2 keyframes found -- check keyframes.py settings.")
        return

    manifest = build_manifest(transcript, keyframes)

    out_path = video_dir / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    empty_narration = sum(1 for m in manifest if not m["narration"])
    print(f"Wrote {len(manifest)} segments to {out_path} "
          f"({empty_narration} with no overlapping narration -- expect to discard or "
          f"merge these with a neighboring segment)")
