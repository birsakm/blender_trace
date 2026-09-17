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
renders from several fixed, auto-framed corner angles rather than one --
a "does this look plausible" set of views for a human or LLM judge, not a
pixel-exact match target. A single fixed angle isn't enough: piloting this
on a real segment (a deleted-and-filled face on a cube's +Y side) rendered
as visually identical to the un-edited mesh from the default corner view,
purely because that face was on the far/hidden side of the object from
that one angle -- confirmed by re-rendering from an axis-aligned view, which
showed the edit clearly. Multiple corner views make that kind of
false-negative much less likely.
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

# Plain solid-shading renders don't include the edge/wireframe overlay every
# tutorial screenshot shows (that overlay is a viewport-only gizmo, never
# baked into a bpy.ops.render.render() image) -- so without this, a render
# can't be told apart from one with completely different topology (e.g. a
# subdivided cube renders identically to a plain one). Freestyle would be
# the "correct" tool for this but isn't reliably available across engines
# in this Blender build; a Wireframe-modifier duplicate is more portable:
# for each mesh, add a copy with a Wireframe modifier (use_replace=True
# turns its faces into thin edge tubes) and a dark material, so it overlays
# every real edge as visible geometry rather than a post-process line.
for o in list(meshes):
    dup = o.copy()
    dup.data = o.data.copy()
    bpy.context.scene.collection.objects.link(dup)
    wire_mod = dup.modifiers.new("VerifyWire", type="WIREFRAME")
    wire_mod.thickness = 0.015
    wire_mod.use_replace = True
    wire_mat = bpy.data.materials.new("VerifyWireMat")
    wire_mat.diffuse_color = (0.02, 0.02, 0.02, 1.0)
    dup.data.materials.clear()
    dup.data.materials.append(wire_mat)

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


scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = {width}
scene.render.resolution_y = {height}
scene.render.image_settings.file_format = "PNG"
scene.render.use_file_extension = False

# Four corner views rather than one -- see module docstring for why one
# fixed angle isn't reliable (an edit on the far side of the object from
# that one angle is invisible, not because it's wrong, just unseen).
views = {views!r}
render_paths = {{}}
for name, direction in views.items():
    frame_all(cam, meshes, mathutils.Vector(direction).normalized())
    scene.render.filepath = {render_dir!r} + "/render_" + name + ".png"
    bpy.ops.render.render(write_still=True)
    render_paths[name] = scene.render.filepath

stats = {{
    "objects": len(meshes),
    "vertices": sum(len(o.data.vertices) for o in meshes if o.data),
    "faces": sum(len(o.data.polygons) for o in meshes if o.data),
    "render_paths": render_paths,
}}
with open({stats_path!r}, "w") as f:
    json.dump(stats, f)
'''

# Four corner-ish views (not axis-aligned faces) so most edits are visible
# from at least one of them without needing to guess which face changed.
DEFAULT_VIEWS = {
    "a": (1, -1, 0.7),
    "b": (-1, 1, 0.7),
    "c": (-1, -1, 0.7),
    "d": (1, 1, 0.7),
}


def render_script(
    script_path: Path,
    out_dir: Path,
    blender_bin: str = DEFAULT_BLENDER,
    resolution: tuple[int, int] = (960, 720),
    views: dict[str, tuple[float, float, float]] = DEFAULT_VIEWS,
) -> dict:
    """Run a bpy/bmesh reconstruction script headlessly and render the
    resulting scene from several fixed, auto-framed corner angles (see
    module docstring for why one angle alone isn't reliable).

    script_path should build geometry into a fresh scene (it runs against
    bpy.ops.wm.read_factory_settings(use_empty=True) -- an empty scene, not
    the default cube). Raises RuntimeError with stdout/stderr if Blender
    exits non-zero or no renders appear (e.g. the script itself raised).

    Returns {"render_paths": {view_name: path, ...}, "stats": {...}}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "stats.json"
    driver_path = out_dir / "_driver.py"

    driver_path.write_text(_DRIVER_TEMPLATE.format(
        script=str(script_path),
        width=resolution[0],
        height=resolution[1],
        render_dir=str(out_dir),
        views=views,
        stats_path=str(stats_path),
    ))

    env = {"HOME": str(Path.home()), "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        [blender_bin, "--background", "--factory-startup", "--python", str(driver_path)],
        capture_output=True, text=True, env=env,
    )
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    render_paths = stats.pop("render_paths", {})
    if result.returncode != 0 or not render_paths:
        raise RuntimeError(
            f"render failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

    return {"render_paths": render_paths, "stats": stats}
