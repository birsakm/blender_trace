# Video 2, segment 3: "press Ctrl F and then go to poke faces. This will
# create a temporary vertex right here in the middle."
# frame_before (61.32) shows a plain cylinder, Edit Mode, top cap face
# selected -- this reconstruction targets just that atomic step.
import bpy

bpy.ops.mesh.primitive_cylinder_add(radius=1, depth=2, vertices=32)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="DESELECT")
bpy.ops.mesh.select_mode(type="FACE")

import bmesh
obj = bpy.context.active_object
bm = bmesh.from_edit_mesh(obj.data)
bm.faces.ensure_lookup_table()
top_face = max(bm.faces, key=lambda f: f.calc_center_median().z)
top_face.select = True
bmesh.update_edit_mesh(obj.data)

bpy.ops.mesh.poke()

bpy.ops.object.mode_set(mode="OBJECT")
