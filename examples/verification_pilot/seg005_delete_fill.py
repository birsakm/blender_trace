# Segment 5: "here, delete out this corner, fill these faces in with the F key,"
# Setup context (not this segment's own narration, but needed to reach the
# same starting mesh state): cube + subdivide twice, per segments 3-4.
import bpy
import bmesh

bpy.ops.mesh.primitive_cube_add(size=2)
obj = bpy.context.active_object

bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.subdivide(number_cuts=1)
bpy.ops.mesh.subdivide(number_cuts=1)

# Select the column of faces on the +Y side nearest the +X corner edge --
# this is "this corner" the narration points at.
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

# Select the boundary loop left by the deletion and fill it with F.
bpy.ops.mesh.select_all(action="DESELECT")
bm = bmesh.from_edit_mesh(obj.data)
for v in bm.verts:
    v.select = any(e.is_boundary for e in v.link_edges)
bmesh.update_edit_mesh(obj.data)

bpy.ops.mesh.edge_face_add()

bpy.ops.object.mode_set(mode="OBJECT")
