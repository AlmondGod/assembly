# Model-generated end-effector control

Set `YAM_CONTROL=cartesian` to execute Qwen-selected tool poses and gripper commands. The initial experiment uses one loose 2×4 brick and two available YAM arms. The model receives observed source and desired final brick poses, gripper geometry and the API contract. The initial prompt assigns the left arm and a downward orientation; Qwen chooses waypoint coordinates, speeds and openings. An optional full-pose form allows orientation and either arm to be specified. The executor calls `YamArm.move` (Cartesian interpolation with numerical inverse kinematics) and `YamArm.grip`; it never calls `assemble` or inserts missing pick/place steps.

Allowed code consists of positional literal calls:

```python
await robot.move("left", 0.24, -0.14, 0.12, 0.04)
await robot.grip("left", 0.026)
```

The full-pose variant is `move(arm,x,y,z,qw,qx,qy,qz,speed)`. Position-only moves use WXYZ `[0,1,0,0]` for the left arm and `[0,0,1,0]` for the right arm.

These two lines illustrate the API, not a complete grasp program. Coordinates are world-frame metres at the fingertip midpoint, quaternion order WXYZ, speeds metres per second and width is finger separation. The model receives the useful grasp depth and downward orientations as instructions. IK computes joint angles; it is not an obstacle-avoidance planner. Commands run sequentially, with bounded positions/speeds/count and no arbitrary Python execution. This first mode is an open-loop program from one observation, without variables, loops or automatic replanning. Known simulator poses replace camera perception. It does not demonstrate clutter extraction or hardware deployment.

## Run

Use the existing Linux simulator environment. Source its machine-specific runtime environment if required. First capture the input scene:

```sh
export PYTHONPATH=.:python
export YAM_CONTROL=cartesian
export YAM_TARGET=demos/assemblies/single_brick.json
export YAM_OUTPUT=outputs/yam-cartesian-scene
YAM_SCENE_ONLY=1 uv run --no-sync bricksim --/app/window/enabled=false --/isaac/startup/ros_bridge_extension= demos/demo_yam_assembly.py
```

Then generate with the separate model environment:

```sh
PYTHONPATH=python experiments/caprl/.venv/bin/python demos/generate_yam_policy.py \
  --target demos/assemblies/single_brick.json \
  --observation outputs/yam-cartesian-scene/observation.json \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --output outputs/yam-cartesian-trial1
export YAM_OUTPUT=outputs/yam-cartesian-trial1
uv run --no-sync bricksim --/app/window/enabled=false --/isaac/startup/ros_bridge_extension= demos/demo_yam_assembly.py
```

Keep `YAM_SCENE_ONLY` unset for execution. Do not set basket mode. Use a fresh output folder for each trial. Inspect `result.json`, not just process exit: geometric task failure is recorded with `status=failed`. Use a process timeout and terminate an owned simulator if it remains alive after its result; some runtime shutdowns hang.

`generation.json` records the full prompt/model revision; `input-observation.json` preserves the generation input. `response.txt` is the raw model output and `policy.py` is its validated executable command list. No generated commands are repaired by the executor. `observation.json` captures the execution scene, `commands.json` records completed movements and actual brick/tool transforms, and per-command PNGs show the outcome. Success requires a native connection, relative placement within 2 mm and 5 degrees, an observed lift of at least 65 mm, open grippers and a raised tool after placement. These are experiment checks, not a measured success rate.

Add `--staged --max-new-tokens 128` to generation to ask for one command at a time. In that mode the code supplies nine action stages (approach, open, descend, close, lift, transport, lower/press, release, retreat). Qwen supplies their numeric commands. All commands are generated before simulation; this is not feedback-based control. Any malformed turn aborts generation without publishing `policy.py`.

## Initial model attempts

Three greedy generations using pinned Qwen3.5-0.8B failed to produce a valid executable program. The first full-pose prompt produced an undefined arm variable and repeated invalid coordinates; the shorter position-only prompt produced function definitions and repeated calculations; the staged prompt emitted several commands and truncated syntax instead of one command. None was executed or repaired. The first two attempts also exposed a provenance-writing incompatibility with the current Transformers configuration; the generator now records the pinned/resolved Hugging Face revision rather than accessing the removed private `_commit_hash` field. Their raw responses and logs remain evidence, but they lack completed generation metadata. The third attempt records its actual prompt and revision.

These few prompt attempts do not establish that the model cannot learn this task. They do establish that the existing successful high-level assembly demo does not transfer automatically to generating numeric motion code. No direct-control training or RL update was performed in this experiment.

The separate [reference policy](cartesian/reference-policy.txt) is handwritten solely to diagnose the physical executor. It is never a fallback for invalid Qwen output and must not be described as model-generated.

The handwritten diagnostic succeeded: nine commands produced a 102.44 mm lift, a native connection, open grippers and a retreated tool. Final position error was 0.00222 mm and angle error 0.000127 degrees. This is one simulator run, not a reliability estimate. All 74 focused tests, repository lint and type checks passed. Raw diagnostic artifacts use an older generic control label containing “model”; the accompanying provenance explicitly identifies the handwritten author, and the final runner uses the neutral label “Cartesian commands + numerical IK”. The only post-run executor edit was that label correction.
