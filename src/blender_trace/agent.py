"""
GPT-powered reconstruction + judging for the verification loop -- the
automated version of the role a Claude Code session played by hand in the
manual pilot (see README "Verification loop pilot").

Two calls per attempt, both against a vision-capable OpenAI model:
- reconstruct_segment(): narration + frame_before/after (+ panel crops) +
  a text summary of the current scene -> bpy/bmesh code for just this
  segment's incremental change.
- judge_render(): the reconstruction's 4 rendered views + the real
  frame_after + narration -> a structured verdict (see JUDGE_SCHEMA_HINT).

run_segment() orchestrates: reconstruct -> render -> judge -> retry (with
the judge's own critique fed back in) up to max_retries. It does NOT try to
recover from a genuinely stuck reconstruction the way a human did in the
pilot (e.g. writing one-off diagnostic scripts, re-deriving camera math) --
it only gets narration + images + the judge's verdict text on each retry,
which is a real capability gap versus the manual process. Segments that
exhaust retries still advance the chained .blend state (with the last
attempt), because leaving state frozen would silently break every later
segment's narration references ("select this", "the other one") to
whatever this segment was supposed to add -- but their verdict is recorded
as unresolved so they're easy to exclude from a final training set.
"""
import base64
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import render as render_mod

DEFAULT_MODEL = "gpt-4o"  # verify this is still OpenAI's current vision-capable model

RECONSTRUCT_SYSTEM_PROMPT = """\
You are reconstructing bpy/bmesh Python code for one step of a Blender \
modeling tutorial, from the artist's narration and two screenshots (before \
and after this step).

Rules:
- Output ONLY a single ```python code block, no prose outside it.
- The code modifies the CURRENT scene state, already loaded -- do not call \
bpy.ops.wm.read_factory_settings or otherwise clear the scene.
- Prefer bpy.ops.* calls matching the specific operator/hotkey/menu item \
named or implied in the narration over ad hoc bmesh reconstruction, unless \
the narration describes low-level vertex/edge/face work with no GUI \
equivalent.
- If you switch to Edit Mode, switch back to Object Mode before finishing.
- Where a needed value (exact numeric parameter, which object/face) isn't \
visible in the narration or images, make the single most plausible choice \
and proceed -- you cannot ask a follow-up question.
"""

JUDGE_SYSTEM_PROMPT = """\
You are judging whether a reconstructed Blender edit matches a real \
tutorial's before/after frames.

You'll see the narration for this step, the real "after" screenshot from \
the tutorial, and 4 rendered views of the reconstruction (fixed corner \
angles, auto-framed on the object). The render angles do NOT match the \
tutorial's camera -- there is no recoverable ground-truth camera pose, so \
ignore camera position/background and judge topology/shape/silhouette \
plausibility instead. An edit can be hidden from one render view but \
visible in another -- check all 4 before concluding something is missing.

Respond with ONLY a single JSON object, no other text, matching:
{
  "verdict": "match" | "partial" | "mismatch",
  "scope": "full-segment" | "operator-only",
  "mismatch_category": "timing-offset" | "camera-hidden" | "wrong-parameter" | "scope-too-narrow" | "other" | null,
  "reasoning": "1-3 sentences"
}
"""


def _b64_image(path: Path) -> dict:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def _extract_code(text: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        raise ValueError(f"no python code block found in model output:\n{text}")
    return match.group(1).strip()


def inspect_blend(blend_path: Path, blender_bin: str = render_mod.DEFAULT_BLENDER) -> str:
    """One-line-per-object text summary of a .blend's mesh objects, so the
    reconstruction prompt can reference existing object names correctly."""
    import subprocess

    script = f"""
import bpy, json
bpy.ops.wm.open_mainfile(filepath={str(blend_path)!r})
info = [
    {{"name": o.name, "type": o.type,
      "vertices": len(o.data.vertices) if o.type == "MESH" and o.data else None,
      "faces": len(o.data.polygons) if o.type == "MESH" and o.data else None}}
    for o in bpy.data.objects
]
print("BLENDER_TRACE_INSPECT_JSON:" + json.dumps(info))
"""
    driver = blend_path.parent / "_inspect.py"
    driver.write_text(script)
    env = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [blender_bin, "--background", "--factory-startup", "--python", str(driver)],
        capture_output=True, text=True, env=env,
    )
    for line in result.stdout.splitlines():
        if line.startswith("BLENDER_TRACE_INSPECT_JSON:"):
            objs = json.loads(line[len("BLENDER_TRACE_INSPECT_JSON:"):])
            if not objs:
                return "(empty scene)"
            return "\n".join(
                f"- {o['name']} ({o['type']}"
                + (f", {o['vertices']} verts, {o['faces']} faces)" if o["type"] == "MESH" else ")")
                for o in objs
            )
    raise RuntimeError(f"scene inspection failed:\n{result.stdout}\n{result.stderr}")


def reconstruct_segment(
    client,
    model: str,
    narration: str,
    frame_before: Path,
    frame_after: Path,
    scene_summary: str,
    panel_before: Path | None = None,
    panel_after: Path | None = None,
    previous_attempt: str | None = None,
    previous_critique: str | None = None,
) -> str:
    """One reconstruction call. Returns bpy/bmesh code as a string."""
    content = [{
        "type": "text",
        "text": (
            f"Narration for this step:\n{narration}\n\n"
            f"Current scene state:\n{scene_summary}\n\n"
            "Images: frame_before, frame_after"
            + (", panel_before, panel_after" if panel_before or panel_after else "")
            + "."
        ),
    }]
    content.append(_b64_image(frame_before))
    content.append(_b64_image(frame_after))
    if panel_before and Path(panel_before).exists():
        content.append(_b64_image(panel_before))
    if panel_after and Path(panel_after).exists():
        content.append(_b64_image(panel_after))
    if previous_attempt and previous_critique:
        content.append({
            "type": "text",
            "text": (
                f"\nA previous attempt was judged not to match:\n"
                f"--- previous code ---\n{previous_attempt}\n"
                f"--- judge's critique ---\n{previous_critique}\n"
                "Write a corrected attempt."
            ),
        })

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RECONSTRUCT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    return _extract_code(response.choices[0].message.content)


def judge_render(
    client,
    model: str,
    render_paths: dict[str, str],
    frame_after: Path,
    narration: str,
) -> dict:
    """One judging call. Returns the parsed verdict dict."""
    content = [{
        "type": "text",
        "text": f"Narration for this step:\n{narration}\n\nImages: frame_after, then 4 render views.",
    }]
    content.append(_b64_image(frame_after))
    for path in render_paths.values():
        content.append(_b64_image(Path(path)))

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


@dataclass
class SegmentResult:
    segment_index: int
    verdict: dict
    code: str
    attempts: int
    state_blend: Path
    render_paths: dict = field(default_factory=dict)


def run_segment(
    client,
    model: str,
    video_dir: Path,
    segment_index: int,
    segment: dict,
    prior_state_blend: Path | None,
    max_retries: int = 2,
) -> SegmentResult:
    """Reconstruct, render, judge, and retry (self-correcting) one manifest
    segment. Always advances chained state, even on an unresolved verdict
    after exhausting retries -- see module docstring for why."""
    from . import verify as verify_mod

    seg_dir = verify_mod.segment_dir(video_dir, segment_index)
    seg_dir.mkdir(parents=True, exist_ok=True)
    scene_summary = inspect_blend(prior_state_blend) if prior_state_blend else "(empty scene)"

    frame_before = video_dir / segment["frame_before"]
    frame_after = video_dir / segment["frame_after"]
    panel_before = video_dir / segment.get("panel_before", "") if segment.get("panel_before") else None
    panel_after = video_dir / segment.get("panel_after", "") if segment.get("panel_after") else None

    code, critique = None, None
    verdict, render_paths = None, {}
    last_successful_state = None  # falls back to prior_state_blend if every attempt errors
    attempts_made = 0

    for attempt in range(max_retries + 1):
        attempts_made = attempt + 1
        code = reconstruct_segment(
            client, model, segment["narration"], frame_before, frame_after,
            scene_summary, panel_before, panel_after,
            previous_attempt=code, previous_critique=critique,
        )
        attempt_dir = seg_dir / f"attempt_{attempt}"
        code_path = attempt_dir / "reconstruct.py"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        code_path.write_text(code)
        state_out = attempt_dir / "state.blend"

        try:
            result = render_mod.render_script(
                code_path, attempt_dir, load_blend=prior_state_blend, save_blend=state_out,
            )
        except RuntimeError as e:
            verdict = {"verdict": "mismatch", "scope": "full-segment",
                       "mismatch_category": "other", "reasoning": f"code raised: {e}"}
            critique = str(e)
            continue

        last_successful_state = state_out
        render_paths = result["render_paths"]
        verdict = judge_render(client, model, render_paths, frame_after, segment["narration"])
        if verdict.get("verdict") == "match":
            break
        critique = verdict.get("reasoning", "")

    final_state = seg_dir / "state.blend"
    # Advance with the last attempt that actually rendered; if every attempt
    # raised, fall back to the unchanged prior state rather than leaving no
    # valid .blend for the next segment to load.
    source = last_successful_state or prior_state_blend
    if source and Path(source).exists():
        shutil.copy(source, final_state)

    verify_mod.save_verdict(video_dir, segment_index, {**verdict, "attempts": attempts_made, "judge": model})
    return SegmentResult(segment_index, verdict, code, attempts_made, final_state, render_paths)
