# Interactive compact-basket trial

This mode lets an external planner inspect a running scene and issue direct end-effector and gripper commands. It uses the existing inverse kinematics and joint drives, not `YamArm.assemble`. It is a simulation experiment using exact simulator poses. There is no camera perception, general collision planner, or autonomous model runtime inside the simulator.

## Observed result

The assistant in this chat selected commands from live simulator state for compact basket seed 7, containing 50 identical 2×4 bricks. Four distinct supply IDs (134, 144, 128, 137) were physically picked, lifted and assembled into the four-brick tower. The left/right/right/left arm assignments, open grippers, final native connections and target relative poses were checked after the final release. All four connections passed; 46 unselected source centres were still inside the bin bounds. Individual observed lifts were 50.6, 99.9, 71.9 and 100.3 mm. This is one successful run, not a completion-rate estimate. The first choice was blocked by the old conservative clearance rule, which this direct-control experiment did not use. The fingers interacted with the pile through physics.

The assistant chose each source, grasp orientation, approach and placement based on observed poses; IK converted commanded tool targets to joint targets. It manually adjusted after four rejected IK waypoints; one malformed command was rejected before any motion and was corrected. The `bind` operation only records which physical source corresponds to a target for evidence. It does not move, attach or reserve a source in physics. No Qwen model, trained grasp policy, RL or automatic collision-free planner controlled this run. The attempt does not isolate the effect of model size: Qwen's earlier test had a different interface and task. It does not demonstrate arbitrary clutter, new random seeds, camera perception or a real robot.

Evidence: [summary](interactive/summary.json), [final result](interactive/result.json), [starting state](interactive/state-000.json), [released final state](interactive/state-036.json), and [all submitted commands](interactive/commands.json). The full numbered snapshots, original images and logs were saved with the run. After this physical run, the implementation was amended to refresh the final state immediately on finish, handle malformed Python syntax without terminating the session, aim the final camera toward the tower, and reject reused session directories. The recorded successful result was checked against state 36 after release; these amendments were not part of that physical run.

## Interface

Start a new output directory using the existing Linux runtime:

```sh
export PYTHONPATH=.:python
export YAM_CONTROL=interactive YAM_BASKET=1 YAM_SUPPLY_COUNT=50
export YAM_BASKET_LAYOUT=compact YAM_SEED=7
export YAM_OUTPUT=outputs/interactive-clutter
uv run --no-sync bricksim --/app/window/enabled=false --/isaac/startup/ros_bridge_extension= demos/demo_yam_assembly.py
```

After initialization, `state-000.json` contains actual world transforms for all sources, the baseplate, tools and gripper gaps. The executor waits for `command-000.json`; submit each command via a temporary file and atomic rename to avoid partial JSON reads. It writes `state-001.json` after the command, then waits for `command-001.json`, and so on. A 300-second idle limit and 80-command limit bound the session. The following are separate messages, not a complete plan:

```json
{"kind":"bind","target":1,"source":134}
{"kind":"motion","program":"await robot.grip(\"left\", 0.026)"}
{"kind":"finish"}
```

Motion programs use the bounded Cartesian `robot.move`/`robot.grip` literal contract described in [CARTESIAN.md](CARTESIAN.md). Each program is parsed in full before execution; execution errors are reported in the next state, and no generated code is evaluated. Position/orientation, speed and gripper gap commands are explicitly chosen by the planner. A single motion message can contain several statements. On finish the session refreshes observations and sets `result.json` to success only if all four bound connections satisfy native connectivity and 2 mm / 5 degree relative-pose thresholds. The assistant should inspect actual arm and part poses after every motion; physics may move bricks between observations.
