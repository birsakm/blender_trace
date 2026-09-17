# Segment 3: "subdivide this two times. You can press" (first of the two subdivides)
import bpy

bpy.ops.mesh.primitive_cube_add(size=2)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.subdivide(number_cuts=1)
bpy.ops.object.mode_set(mode="OBJECT")
