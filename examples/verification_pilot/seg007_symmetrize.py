# Segment 7: "And then just symmetrize to this side,"
# Continues from segments 3-6 (cube, subdivide x2, delete+fill one corner),
# then mirrors that edit across the Y axis so the opposite corner matches.
import bpy
import bmesh

bpy.ops.mesh.primitive_cube_add(size=2)
obj = bpy.context.active_object

bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.subdivide(number_cuts=1)
bpy.ops.mesh.subdivide(number_cuts=1)

bm = bmesh.from_edit_mesh(obj.data)
bm.faces.ensure_lookup_table()
candidates = [f for f in bm.faces if f.normal.y > 0.9]
max_x = max(f.calc_center_median().x for f in candidates)
for f in bm.faces:
    f.select = False
for f in candidates:
    if abs(f.calc_center_median().x - max_x) < 1e-4:
        f.select = True
bmesh.update_edit_mesh(obj.data)

bpy.ops.mesh.delete(type="FACE")

bpy.ops.mesh.select_all(action="DESELECT")
bm = bmesh.from_edit_mesh(obj.data)
for v in bm.verts:
    v.select = any(e.is_boundary for e in v.link_edges)
bmesh.update_edit_mesh(obj.data)
bpy.ops.mesh.edge_face_add()

# Best-guess direction: propagate the edited (+Y) side onto -Y. There's no
# ground-truth camera pose to confirm which world-space side the tutorial's
# "this corner" actually was, so this is exactly the kind of parameter this
# reconstruction can get backwards -- see verdict.
# select_all is required here: only the fill's boundary-loop verts were
# still selected from the previous step, so symmetrize would otherwise
# silently operate on that tiny leftover selection instead of the whole mesh.
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.symmetrize(direction="POSITIVE_Y")

bpy.ops.object.mode_set(mode="OBJECT")
