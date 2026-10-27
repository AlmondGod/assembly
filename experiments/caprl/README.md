# Reinforcement learning for the YAM assembly policy

This is a small, single-GPU adaptation of [CaP-RL's environment-reward approach](https://github.com/capgym/cap-x/blob/main/docs/rl-training.md). Upstream CaP-X uses verl and Franka task environments. This adapter uses [TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer) or [TRL RLOO](https://huggingface.co/docs/trl/rloo_trainer), our existing two-YAM controller, and pinned `Qwen/Qwen3.5-0.8B`. It is not an installation of the upstream CaP-X benchmark or a reproduction of its results.

The model writes four literal `await robot.assemble(stud, hole, arm)` calls. Programs pass the existing AST validator; generated Python is never executed. The task randomizes which logical brick occupies each basket slot, including existing position/yaw variation. Prompts give initial simulator source coordinates, but do not prescribe an arm. Choosing the arm is a real physical decision. Tower dependencies determine order; grasp pose, source picking, motion, force and retries remain controller decisions. This setup does **not** learn joint control, vision, clutter extraction, or recovery programs. It initially trains on four separated bricks, not the new fifty-brick clutter task. Adding physical-source choice would require a separately validated policy API.

## Installation

Keep training dependencies separate from the existing Isaac environment. On the GPU machine, from the repository root:

```sh
export UV_CACHE_DIR="$PWD/.cache/caprl-uv"
uv sync --project experiments/caprl --locked
source .runtime/environment.sh
export PYTHONPATH="$PWD/python:$PWD"
```

The repository-local cache avoids permission failures on the RunPod network-mounted shared cache. The checked-in lock pins training dependencies, including TRL 1.16.0, Transformers 5.21.0, and Torch 2.10.0. `uv` uses Python 3.12 for this project; Isaac continues using its existing Python 3.11 environment. No vLLM, flash-attn, W&B account or external model endpoint is required. Model weights are pinned to revision `2fc06364715b967f1860aea9cf38778875588b17`. The runtime environment script is the current RunPod Vulkan configuration; another machine must provide its own functioning Isaac/Vulkan setup.

Use one training process and reserve the GPU for the entire job. Each valid generated program launches Isaac separately while the small LoRA training model remains resident. The adapter serializes its own workers, but cannot detect unrelated demos. Stop those first. Available GPU memory and this particular model/trainer combination must be verified by the physical smoke test before a longer run.

## Bounded smoke and training

These commands run one optimizer step with two sampled programs. Every valid program is physically executed; a valid generation can take several minutes. Invalid programs never launch the simulator. Use a new output directory each time.

```sh
uv run --project experiments/caprl --locked python -m experiments.caprl.train train \
  --algorithm grpo --max-steps 1 --group-size 2 --timeout 900 \
  --output outputs/caprl/grpo-smoke

uv run --project experiments/caprl --locked python -m experiments.caprl.train train \
  --algorithm rloo --max-steps 1 --group-size 2 --timeout 900 \
  --output outputs/caprl/rloo-smoke
```

Increase `--max-steps` only after inspecting physical rollout evidence and memory use. Defaults are conservative LoRA rank 8, learning rate 5e-6, KL coefficient 0.02, sampling temperature 0.8, and 384 generated tokens. With equal rewards within every group there is no useful relative policy-gradient signal; inspect outcomes before spending more compute. The software does not manufacture reward variance or substitute canned successful programs.

Reward is `verified_actions / 4 + full_completion`, ranging from 0 to 2 for valid programs. Invalid programs receive -0.25 and a separate `invalid_policy` status. A verified action must match the requested brick/arm and contain the controller's native connection and lift evidence. Full completion additionally requires all four final native connections. The unchanged controller enforces lift, pose and connection thresholds. Physical failure retains previously verified progress; a runtime deadline retains the last saved progress. Startup failures, missing evidence, unexpected exception types/messages and process crashes abort the job rather than masquerading as robot failure. Each child process group is killed after completion or timeout, including Isaac instances that hang during exit.

There is no rollout cache. Every response, prompt, seed, target hash, simulator log, inventory, progress, result and scalar outcome is retained under `rollouts/`. `provenance.json` records model revision, package versions, source hashes, Git state and settings. TRL saves checkpoints and optimizer state; `adapter/` is the final LoRA adapter. A completed software smoke alone is not evidence of improved assembly success.

## Held-out completion evaluation

Training uses source seeds 0–31. Evaluation uses 1000–1015, with three fresh physical trials per seed by default. The target stays a four-brick tower; this measures new source-layout generalization, not new structures. Evaluate the base and adapter with the same seed manifest and repeat count:

```sh
uv run --project experiments/caprl --locked python -m experiments.caprl.train evaluate \
  --output outputs/caprl/base-eval

uv run --project experiments/caprl --locked python -m experiments.caprl.train evaluate \
  --adapter outputs/caprl/grpo-smoke/adapter \
  --output outputs/caprl/grpo-eval
```

For a bounded integration check, add `--eval-limit 1 --eval-repeats 1`; this cannot establish improvement. `metrics.json` records peak CUDA allocation/reservation for the model process (excluding the separate simulator) and separates format-valid rate, physical completion rate and mean verified action fraction. Compare repeated held-out completion rates and inspect failures; contact simulation can vary even for the same initial seed. Do not select checkpoints using the held-out set. More trials and independent training seeds are needed before asserting reliable gains. No real-robot transfer claim follows from these simulations.

## Tests

Pure reward/scenario tests replay checked-in real evidence only to test the scoring contract. The optimizer tests use an explicitly synthetic, tiny random model to test TRL APIs. Neither test substitutes for an online physical rollout.

```sh
PYTHONPATH=python:. uv run --no-sync pytest --noconftest -q python/tests/test_caprl.py
uv run --project experiments/caprl --locked python -m pytest --noconftest \
  -q experiments/caprl/tests/test_trainers.py
```

The two training-only files are excluded from the Isaac type-check configuration because their dependencies are deliberately isolated. Check them separately:

```sh
uv run --project experiments/caprl --locked ty check \
  --python experiments/caprl/.venv --extra-search-path python --extra-search-path . \
  experiments/caprl/train.py experiments/caprl/rollout.py \
  experiments/caprl/tests/test_trainers.py
```
