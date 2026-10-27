# Cut-corner plate experiment

Set 31377 includes eight dark-tan part 2450 plates, used in turtle steps 6
and 7. `scripts/ldraw_2450.py` flattens the community LDraw geometry for this
part and its referenced primitives into a metre-scale OBJ. The LDraw model is
not a manufacturer CAD model.

The source revision is pinned. Each downloaded source file, attribution header,
license and SHA-256 hash are recorded in `sources.json` alongside the output.
The mesh is an adaptation of the LDraw Parts Library, and all 15 referenced
files in this example specify [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The part author is James Jessiman; primitive credits are recorded individually.

    uv run --no-sync python scripts/ldraw_2450.py outputs/ldraw-2450

With a configured simulator, an independent collision test can be run as:

    PYTHONPATH=.:python LDRAW_OBJ=outputs/ldraw-2450/2450.obj uv run --no-sync bricksim --/app/window/enabled=false --/isaac/startup/ros_bridge_extension= demos/demo_ldraw_2450.py

The drop test approximates the part by a **convex hull**, which fills the
underside cavities and smooths the stud contacts. Its 2 gram mass is a test
assumption, not a measured part mass. It does not create stud interfaces,
connections or the complete turtle. The runtime currently registers only
rectangular brick parts; integration of specialty-part interfaces and actual
assembly remains future work.

A RunPod drop test settled the mesh origin at 3.200 mm above the floor after
starting at 47.275 mm. This verified ground contact for the approximate hull;
it did not test a robot grasp or any part-to-part connection.
