"""
Verification-loop scaffolding.

Given a manifest segment and a reconstruction script (bpy/bmesh code an
agent wrote from that segment's narration + frame_before/frame_after), this
renders it and standardizes where the result and verdict live, so the
"write code from narration+frames" and "judge the render against
frame_after" steps -- both of which need a capable vision model -- can be
done by whatever's playing that role.

This module does NOT call any LLM itself. There's no ANTHROPIC_API_KEY (or
equivalent) wired up in this environment (see README), so right now that
role is played by a Claude Code session/subagent acting directly on the
files this writes -- see README "Verification loop pilot" for how that
went on three real segments. If an API key becomes available later, a
script that calls it programmatically can slot in here without changing
this file layout.
"""
import json
from pathlib import Path

from . import render as render_mod


def segment_dir(video_dir: Path, segment_index: int) -> Path:
    return video_dir / "verify" / f"{segment_index:04d}"


def verify_segment(video_dir: Path, segment_index: int, reconstruction_script: Path) -> dict:
    """Render a segment's reconstruction script under
    <video_dir>/verify/<segment_index>/. Returns the render result
    (render_path, stats) -- does not judge match/no-match itself."""
    out_dir = segment_dir(video_dir, segment_index)
    return render_mod.render_script(reconstruction_script, out_dir)


def save_verdict(video_dir: Path, segment_index: int, verdict: dict) -> Path:
    """verdict e.g. {"match": bool, "reasoning": "...", "judge": "..."}"""
    out_dir = segment_dir(video_dir, segment_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = out_dir / "verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2))
    return verdict_path
