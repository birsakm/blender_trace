import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blender_trace import agent
from blender_trace.render import DEFAULT_BLENDER

blender_available = Path(DEFAULT_BLENDER).exists()


def _completion(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class FakeClient:
    """Records calls and returns canned responses in order, so orchestration
    logic (retry loop, state fallback) can be tested without a real API key
    or network access."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _completion(self.responses.pop(0))


def test_extract_code_parses_fenced_block():
    text = "Here you go:\n```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add()\n```"
    assert agent._extract_code(text) == "import bpy\nbpy.ops.mesh.primitive_cube_add()"


def test_extract_code_raises_without_fence():
    with pytest.raises(ValueError):
        agent._extract_code("just prose, no code block")


def test_reconstruct_segment_sends_images_and_returns_code(tmp_path):
    frame_before = tmp_path / "before.png"
    frame_after = tmp_path / "after.png"
    frame_before.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0")
    frame_after.write_bytes(b"\x89PNG\r\n\x1a\n" + b"1")
    client = FakeClient(["```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add()\n```"])

    code = agent.reconstruct_segment(
        client, "gpt-4o", "add a cube", frame_before, frame_after, "(empty scene)",
    )

    assert "primitive_cube_add" in code
    sent_content = client.calls[0]["messages"][1]["content"]
    image_count = sum(1 for block in sent_content if block.get("type") == "image_url")
    assert image_count == 2  # frame_before + frame_after, no panel crops given


def test_judge_render_parses_json_verdict(tmp_path):
    frame_after = tmp_path / "after.png"
    frame_after.write_bytes(b"\x89PNG\r\n\x1a\n" + b"1")
    verdict = {"verdict": "match", "scope": "full-segment",
               "mismatch_category": None, "reasoning": "looks right"}
    client = FakeClient([json.dumps(verdict)])

    result = agent.judge_render(client, "gpt-4o", {"a": str(frame_after)}, frame_after, "add a cube")

    assert result == verdict
    assert client.calls[0]["response_format"] == {"type": "json_object"}


def test_run_segment_skips_non_actionable_narration_without_calling_client(tmp_path):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    segment = {
        "narration": "what's up guys, in today's video I want to show you something cool",
        "frame_before": "before.png", "frame_after": "after.png",
    }
    client = FakeClient([])  # would raise IndexError if anything tried to call it

    result = agent.run_segment(client, "gpt-4o", video_dir, 0, segment, None, max_retries=2)

    assert result.verdict["verdict"] == "skipped"
    assert result.attempts == 0
    assert client.calls == []
    verdict_path = video_dir / "verify" / "0000" / "verdict.json"
    assert json.loads(verdict_path.read_text())["verdict"] == "skipped"


def test_run_segment_skip_carries_prior_state_forward_unchanged(tmp_path):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    prior_state = tmp_path / "prior.blend"
    prior_state.write_bytes(b"fake blend contents")
    segment = {"narration": "no action here at all", "frame_before": "b.png", "frame_after": "a.png"}
    client = FakeClient([])

    result = agent.run_segment(client, "gpt-4o", video_dir, 1, segment, prior_state, max_retries=1)

    assert result.state_blend.read_bytes() == prior_state.read_bytes()


@pytest.mark.skipif(not blender_available, reason="standalone Blender binary not present")
def test_run_segment_retries_then_succeeds(tmp_path):
    video_dir = tmp_path / "video"
    video_dir.mkdir()
    frame_before = video_dir / "before.png"
    frame_after = video_dir / "after.png"
    frame_before.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0")
    frame_after.write_bytes(b"\x89PNG\r\n\x1a\n" + b"1")
    segment = {"narration": "add a cube", "frame_before": "before.png", "frame_after": "after.png"}

    # First reconstruction is broken code (forces a retry); second succeeds.
    # First judge call (never reached, since attempt 0 raises) is skipped;
    # judge is only called once, after the working attempt 1 renders.
    client = FakeClient([
        "```python\nthis is not valid python (((\n```",
        "```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add()\n```",
        json.dumps({"verdict": "match", "scope": "full-segment",
                    "mismatch_category": None, "reasoning": "matches"}),
    ])

    result = agent.run_segment(client, "gpt-4o", video_dir, 0, segment, None, max_retries=2)

    assert result.verdict["verdict"] == "match"
    assert result.attempts == 2
    assert result.state_blend.exists()
    verdict_path = video_dir / "verify" / "0000" / "verdict.json"
    assert json.loads(verdict_path.read_text())["verdict"] == "match"
