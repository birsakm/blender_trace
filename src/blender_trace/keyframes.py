"""
Extract keyframes at likely "edit boundaries" from a downloaded tutorial video,
plus a cropped region of the header/F9-redo-panel for later OCR.

--panel-box is fractional (x0 y0 x1 y1), 0-1 of the frame. Calibrate this once
per channel by eyeballing a saved frame (see README).

Why frame-differencing on the *viewport* region and not the whole frame: talking
-head overlays, timeline scrubbers, and cursor blink cause constant low-level
pixel change outside the modeling viewport that would otherwise swamp real
signal. Restrict the diff to the 3D viewport region (roughly the left/center
~70% of the frame, excluding the panel-box you're also extracting) to reduce
false positives.

Output:
    <video_dir>/frames/frame_<t>.png    (full frame at each boundary)
    <video_dir>/panels/panel_<t>.png    (cropped header/redo-panel)
    <video_dir>/keyframes.json          (list of {t, frame_path, panel_path})
"""
import json
from pathlib import Path

import cv2
import numpy as np


def extract_frames_at(
    video_dir: Path,
    timestamps: list[float],
    panel_box: tuple[float, float, float, float],
) -> dict[float, dict]:
    """Save the frame (+ panel crop) at or immediately after each requested
    timestamp, in one sequential pass. Used by the narration-driven manifest
    builder, which decides segment boundaries from transcript content rather
    than from pixel-diffing -- this just needs to fetch specific timestamps,
    not detect anything.

    Returns {timestamp: {"frame_path": ..., "panel_path": ...}}, keyed by
    the exact input timestamps (not the actual frame time, which may lag
    by up to one sample).
    """
    video_path = video_dir / "video.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    frames_dir = video_dir / "frames"
    panels_dir = video_dir / "panels"
    frames_dir.mkdir(exist_ok=True)
    panels_dir.mkdir(exist_ok=True)

    px0, py0, px1, py1 = panel_box
    targets = sorted(set(timestamps))
    target_idx = 0
    result = {}

    frame_idx = 0
    while target_idx < len(targets):
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        if t >= targets[target_idx]:
            requested_t = targets[target_idx]
            h, w = frame.shape[:2]
            frame_path = frames_dir / f"frame_{requested_t:.2f}.png"
            cv2.imwrite(str(frame_path), frame)

            x0, x1 = int(px0 * w), int(px1 * w)
            y0, y1 = int(py0 * h), int(py1 * h)
            panel_path = panels_dir / f"panel_{requested_t:.2f}.png"
            cv2.imwrite(str(panel_path), frame[y0:y1, x0:x1])

            result[requested_t] = {
                "frame_path": str(frame_path.relative_to(video_dir)),
                "panel_path": str(panel_path.relative_to(video_dir)),
            }
            target_idx += 1
        frame_idx += 1

    cap.release()
    return result


def extract(
    video_dir: Path,
    panel_box: tuple[float, float, float, float],
    diff_threshold: float,
    min_gap_s: float,
    sample_fps: float = 2.0,
):
    # Not a glob("video.*") -- that also matches the sibling .srt/.vtt caption
    # files download.py writes, and glob order isn't guaranteed alphabetical,
    # so it can silently pick a subtitle file instead of the video.
    video_path = video_dir / "video.mp4"
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(fps / sample_fps))

    frames_dir = video_dir / "frames"
    panels_dir = video_dir / "panels"
    frames_dir.mkdir(exist_ok=True)
    panels_dir.mkdir(exist_ok=True)

    px0, py0, px1, py1 = panel_box
    # Diff against the last *saved keyframe*, not the previous sample -- a
    # long sequence of small, gradual edits (or a slow orbit) can drift a
    # long way while never producing a single frame-to-frame delta big
    # enough to cross diff_threshold, silently collapsing many distinct
    # operations into one giant "segment" that no single before/after frame
    # pair can represent.
    reference_gray = None
    last_saved_t = -1e9
    keyframes = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        t = frame_idx / fps

        h, w = frame.shape[:2]
        # exclude the panel-box region from the diff so on-screen text changes
        # (which happen every operator, not just meaningful edits) don't dominate
        viewport = frame[:, : int(w * 0.7)]
        gray = cv2.cvtColor(viewport, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))

        is_boundary = False
        if reference_gray is None:
            is_boundary = True
        else:
            diff = np.mean(np.abs(gray.astype(float) - reference_gray.astype(float)))
            if diff > diff_threshold and (t - last_saved_t) > min_gap_s:
                is_boundary = True

        if is_boundary:
            frame_path = frames_dir / f"frame_{t:.2f}.png"
            cv2.imwrite(str(frame_path), frame)

            x0, x1 = int(px0 * w), int(px1 * w)
            y0, y1 = int(py0 * h), int(py1 * h)
            panel_crop = frame[y0:y1, x0:x1]
            panel_path = panels_dir / f"panel_{t:.2f}.png"
            cv2.imwrite(str(panel_path), panel_crop)

            keyframes.append({
                "t": t,
                "frame_path": str(frame_path.relative_to(video_dir)),
                "panel_path": str(panel_path.relative_to(video_dir)),
            })
            last_saved_t = t
            reference_gray = gray

        frame_idx += 1

    cap.release()
    out_path = video_dir / "keyframes.json"
    out_path.write_text(json.dumps(keyframes, indent=2))
    print(f"Saved {len(keyframes)} keyframes to {out_path}")
