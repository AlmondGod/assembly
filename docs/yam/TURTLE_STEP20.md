# Turtle set 31377: step 20 experiment

This scene isolates one real subassembly from the turtle build: a green 2×2
plate on the left two columns of a dark tan 2×3 plate. It uses two YAM arms and
an organized supply with only these two set pieces. A neutral 20×20 baseplate
and small supports are fixtures, not parts of the set. The pair occurs twice
in step 20; this experiment prepares one pair.

The layout follows the independently tested [text instructions](https://bricksfortheblind.org/instructions/turtle-with-a-water-lily-flower/)
and is a small part of the [official 124-piece set](https://www.lego.com/en-us/service/building-instructions/31377).
It is not the complete turtle. The current rectangular part model does not
represent the set's curved pieces, clips, hinges, and side connections.

Run the scene with `YAM_SCENE_ONLY=1` to inspect the supply. Without that
setting the controller attempts two physical connections and writes the outcome
to `outputs/yam-turtle-step20/result.json`.

    PYTHONPATH=.:python YAM_SCENE_ONLY=1 uv run --no-sync bricksim --/app/window/enabled=false --/isaac/startup/ros_bridge_extension= demos/demo_yam_turtle_step20.py

The controller was originally developed for taller bricks. Small supports
elevate the plates for finger clearance. An experimental run used successive
0.5, 1.5 and 2.5 mm side presses followed by a top press and connected both
plates. The first plate connects to the fixture; the second connects to the
first plate. Both were lifted approximately 100 mm before placement.

Contact-limited press moves allow up to 7 mm of TCP position error, instead of
the standard 3 mm. This relaxes only the motion check: each connection and its
final relative pose are still checked independently by the simulator. The
result is one successful run, not a measured reliability estimate. The scene
uses simulator poses rather than visual perception.
