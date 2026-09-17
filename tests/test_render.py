from pathlib import Path

import pytest

from blender_trace.render import DEFAULT_BLENDER, render_script

blender_available = Path(DEFAULT_BLENDER).exists()


@pytest.mark.skipif(not blender_available, reason="standalone Blender binary not present")
def test_render_script_produces_render_and_stats(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(
        "import bpy\n"
        "bpy.ops.mesh.primitive_cube_add(size=2)\n"
    )

    result = render_script(script, tmp_path / "out")

    assert Path(result["render_path"]).exists()
    assert Path(result["render_path"]).stat().st_size > 0
    assert result["stats"]["objects"] == 1
    assert result["stats"]["vertices"] == 8
    assert result["stats"]["faces"] == 6


@pytest.mark.skipif(not blender_available, reason="standalone Blender binary not present")
def test_render_script_raises_on_broken_reconstruction_code(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("this is not valid python (((\n")

    with pytest.raises(RuntimeError):
        render_script(script, tmp_path / "out")
