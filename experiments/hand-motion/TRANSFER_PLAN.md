# Hand-motion co-training and YAM transfer

## Purpose

Test whether task-matched simulated hand object motion improves Qwen3.5-4B robot-control code on fresh cluttered baskets. This is a small controlled transfer pilot, not evidence of generalization beyond its held-out scenes. Keep the experiment under work/hand-qwen-transfer and /opt/BrickSim/.runtime/hand-qwen-transfer, and preserve all other running work.

## Progress

- [x] Verified the previous48successfulhandtrainwindows/16heldoutwindows, model/environment availability, existing14verifiedrobotcodeexamples and24robotmotionexamples.
- [ ] Capture matched CPU-rendered YAM scenes for existing train snapshots and four fresh evaluation seeds1024–1027.
- [ ] Train matched code_only,robot_motion,andhand_motion LoRAs with identical14codeexamples, hyperparameters and code updates; hand_motion adds48handmotionwindows to robot_motion.
- [ ] Generate frozen programs for original and all adapters, then execute them under the same YAM controller and success gate.
- [ ] Audit scene splits, report physical success separately from syntax and timing, back up artifacts, and terminate owned processes.

## Decisions and mathematics

Object labels use world metres, row-major4x4transforms acting on column vectors. Quaternions are WXYZ. Motion heads take last prompt-token hidden states, current3Dpoint coordinates and horizon, and predict3Dpoint displacement. Share one motion head across robot and hand domains; no future object observations enter inputs. Heads are training-only. Actor code still receives identical RGB plus existing simulator candidate poses and proprioception for every condition. The source hand controller used privileged poses/contact feedback; these are not inference inputs. Hand source seeds30010–30017 are disjoint from held-out30018–30019 and YAM evaluation1024–1027. Previous model-test seeds1012–1023 stay out of training.

All images use the same untextured CPU renderer, camera and512x384RGB format. Reconstruct only the initial YAM arm home geometry; old train brick poses remain exactly those underlying verified programs. Fresh test snapshots are collected and frozen before training. A new code-only control is required because reusing a checkpoint trained on older RTX images would confound the comparison. Physical evaluation uses the existing numerical IK3mm tolerance and unchanged centered-grasp gate: held lift>=65mm, gripper gap8–25mm, TCP/brick XY<=15mm and Z2–15mm after60holding ticks.

## Context and execution

Use work/pod_command.py with the existing local SSH key to reach ozzzq5hjse3zov-64411314@ssh.runpod.io. The existing simulator is /opt/BrickSim/.venv/bin/bricksim; model runtime /opt/object-motion-scale-ml/bin/python and base model /workspace/qwen-upgrade/model. Existing experiment data live at /opt/BrickSim/.runtime/object-motion-scale. Hand data and its shared-source snapshots are in /opt/BrickSim/.runtime/hand-motion, with a verified backup under /workspace/hand-motion. Use the existing finite supervisors as templates and process groups to stop only this experiment's children. Another simulator currently uses about589MiBGPUmemory; do not terminate it. Training may share available GPU memory, but record interference and benchmark latency only under stated contention conditions.

## Validation and milestones

First capture and render a real YAM scene, then inspect its image and verify native snapshot poses and camera metadata. Train each condition from the same base with the same random seed and code exposure. Validate generated programs without repairing them, save full prompts/responses/timing, and run every condition on the exact same fresh physical snapshots with alternating condition order. A successful result is a completed comparison with auditable successes/failures, not necessarily an improvement. Back up adapters, manifests, hashes and measured outcomes to the persistent volume. No increase in model capability may be claimed from auxiliary loss or valid syntax alone.

## Discoveries and outcomes

Pending. The existing renderer/material-wrapper collision-scale bug was fixed during source hand collection. Existing YAM gripper meshes have unit scales, so do not assume that hand integration bug explains older YAM failures.

## Pause checkpoint (2026-10-29)

User requested commit, push, PR merge and pause. Captured training seeds1000–1011; fresh seed1024 failed because InteractiveSession has no basket attribute. Capture exited1 and its owned process group is stopped. No new model training or transfer evaluation executed. Preserve source artifacts and resume only on further user instruction.
