# Simulated hand source-motion checkpoint

Paused at the user's request on 2026-10-29. This records exploratory research scripts and observed results, not a completed transfer-training feature. The scripts are preserved verbatim from the experiment; the capture script has a known runtime failure.

The prescribed 22-DOF Shadow Hand completed 10 fresh 50-brick basket attempts: 7 passed the physical lift, hold-drift and bilateral-contact gate. Collection took 484.804 seconds. Eight RGB/object-motion windows per attempt produced 80 windows: 48 successful source-training windows, 16 failed source windows excluded from training, and 16 windows from two separate held-out scenes. These are source-hand results, not Qwen or YAM successes. The controller uses privileged current poses/contact feedback and does not validate real hand actuation.

Physics-material binding now uses the USD material API directly. The earlier geometry wrapper reset scaled collision cubes to unit scale; the diagnostic and negative control are preserved in evidence. Do not infer that this explains earlier YAM policy failures.

## Saved work

- `../../work/hand-motion/`: final source controller, finite process-group supervisor, recording, asset preparation, CPU renderer, motion export and audit scripts. Prototype variants remain in the local workspace and the existing source backup.
- `../../work/hand-qwen-transfer/`: initial matched-image capture and launch scripts; no training implementation or new adapters yet.
- `evidence/`: actual pilot report, asset-derived geometry/configuration, file hashes, material diagnostic, cleanup evidence and transfer capture state JSONs.
- `TRANSFER_PLAN.md`: planned controlled comparison of original, code-only, robot-motion and hand-motion conditions. The plan is pending.

Run workspace-oriented scripts from the repository root. Local scripts use `work/pod_command.py` to access the existing RunPod runtime with a locally held SSH key; `BRICKSIM_SSH_TARGET` overrides the target. No credentials or private keys are stored here. Remote scripts expect the existing `/opt/BrickSim` Isaac Sim 5.1 environment and assets. Offline rendering needs NumPy, SciPy, Pillow, numba, fast-simplification and imageio-ffmpeg. These are research entry points tied to that runtime, not a portable installation workflow.

The complete 17,551,902-byte image/video/native-pose pilot bundle remains at local `outputs/hand-motion/hand-motion-pilot.zip`, with a verified persistent backup under `/workspace/hand-motion` on the current pod. Large binary recordings and licensed USD assets are not checked into Git. `evidence/bundle-manifest.json` identifies all bundle contents by SHA-256.

## Exact transfer pause point

The capture process saved native meshes, body poses and state JSONs for training seeds 1000–1011, then exited with code 1 while starting seed 1024: `AttributeError: 'InteractiveSession' object has no attribute 'basket'`. No fresh evaluation scenes were completed, no co-training ran, and no transfer or speed improvement has been measured. The finite supervisor reports the process stopped; a process-group audit found no owned processes. All 58 transfer runtime files were copied and hash-verified under `/workspace/hand-qwen-transfer` (see `evidence/transfer-capture/pause-audit.json`). Resume by fixing basket-scene generation against the actual session API, using a new output directory rather than overwriting this capture, and validating one matched rendered image before training.

The comparison is intended to use the existing **Qwen3.5-4B** base, not the original 0.8B model. Preserve identical prompts, cameras, renderer, code exposure, numerical IK and physical success gates across conditions. See the plan for split and measurement details.

## Checkpoint validation

All 24 saved Python files compile without importing the simulator. Evidence JSON parses successfully. The source scripts are exploratory and retain lint/type-check debt; no clean repository-wide lint/type check or fresh simulator run is claimed. The captured runtime failure is intentionally preserved rather than disguised as working capture. No simulation or training was launched to validate this pause checkpoint.
