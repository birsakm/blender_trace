"""
Headless-render a bpy/bmesh reconstruction script via a standalone Blender
binary, for the verification loop's "render and compare" step.

This machine's shell profile sets LD_LIBRARY_PATH to a pip `bpy` package's
dependency directory (libusd_ms.so) that conflicts with Blender's own
bundled libraries (a USD/TBB symbol clash) -- that breaks not just
`import bpy` in a bare interpreter but the standalone Blender binary too, if
the variable leaks through. render_script() always launches with a clean
env for this reason; don't remove that without re-testing headless launch.

There's no ground-truth viewport camera pose stored anywhere for a
tutorial's screen recording (the artist just orbited freely), so this
renders from one fixed, auto-framed angle -- a "does this look plausible"
view for a human or LLM judge, not a pixel-exact match target.
"""
import json
import subprocess
from pathlib import Path

DEFAULT_BLENDER = "/datawaha/cggroup/birsakm/blender-releases/blender-4.3.0-linux-x64/blender"

_DRIVER_TEMPLATE = '''
import json
import bpy
import mathutils

bpy.ops.wm.read_factory_settings(use_empty=True)

with open({script!r}) as f:
    exec(compile(f.read(), {script!r}, "exec"))

meshes = [o for o in bpy.data.objects if o.type == "MESH"]

cam_data = bpy.data.cameras.new("VerifyCam")
cam = bpy.data.objects.new("VerifyCam", cam_data)
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam

light_data = bpy.data.lights.new("VerifyLight", type="SUN")
light_data.energy = 2.5
light = bpy.data.objects.new("VerifyLight", light_data)
light.location = (4, -4, 6)
bpy.context.scene.collection.objects.link(light)


def frame_all(cam_obj, objs, direction, margin=4.0):
    coords = []
    for o in objs:
        for v in o.bound_box:
            coords.append(o.matrix_world @ mathutils.Vector(v))
    if not coords:
        cam_obj.location = direction * 5
        cam_obj.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
        return
    xs = [c.x for c in coords]
    ys = [c.y for c in coords]
    zs = [c.z for c in coords]
    center = mathutils.Vector(
        ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)
    )
    radius = max((c - center).length for c in coords) or 1.0
    cam_obj.location = center + direction * radius * margin
    cam_obj.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()


frame_all(cam, meshes, mathutils.Vector((1, -1, 0.7)).normalized())

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = {width}
scene.render.resolution_y = {height}
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = {render_path!r}
scene.render.use_file_extension = False

bpy.ops.render.render(write_still=True)

stats = {{
    "objects": len(meshes),
    "vertices": sum(len(o.data.vertices) for o in meshes if o.data),
    "faces": sum(len(o.data.polygons) for o in meshes if o.data),
}}
with open({stats_path!r}, "w") as f:
    json.dump(stats, f)
'''


def render_script(
    script_path: Path,
    out_dir: Path,
    blender_bin: str = DEFAULT_BLENDER,
    resolution: tuple[int, int] = (960, 720),
) -> dict:
    """Run a bpy/bmesh reconstruction script headlessly and render the
    resulting scene from a fixed auto-framed angle.

    script_path should build geometry into a fresh scene (it runs against
    bpy.ops.wm.read_factory_settings(use_empty=True) -- an empty scene, not
    the default cube). Raises RuntimeError with stdout/stderr if Blender
    exits non-zero or the render never appears (e.g. the script itself
    raised).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    render_path = out_dir / "render.png"
    stats_path = out_dir / "stats.json"
    driver_path = out_dir / "_driver.py"

    driver_path.write_text(_DRIVER_TEMPLATE.format(
        script=str(script_path),
        width=resolution[0],
        height=resolution[1],
        render_path=str(render_path),
        stats_path=str(stats_path),
    ))

    env = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [blender_bin, "--background", "--factory-startup", "--python", str(driver_path)],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0 or not render_path.exists():
        raise RuntimeError(
            f"render failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    return {"render_path": str(render_path), "stats": stats}
