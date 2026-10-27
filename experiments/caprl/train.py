"""Train GRPO/RLOO or evaluate a pinned Qwen model against physical rollouts."""

import argparse
import hashlib
import json
import os
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from experiments.caprl.rollout import (
    EVAL_SEEDS,
    MODEL,
    REVISION,
    ROOT,
    TRAIN_SEEDS,
    Scenario,
    SimulatorReward,
)
from peft import LoraConfig, PeftModel
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen3_5ForConditionalGeneration,
    set_seed,
)
from trl import GRPOConfig, GRPOTrainer, RLOOConfig, RLOOTrainer


def main():
    """Run a finite training job or deterministic held-out physical evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--algorithm", choices=("grpo", "rloo"), default="grpo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--eval-limit", type=int, default=len(EVAL_SEEDS))
    parser.add_argument("--eval-repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This simulator adapter requires one training process")
    if (
        args.max_steps < 1
        or args.group_size < 2
        or args.timeout <= 0
        or args.eval_repeats < 1
    ):
        parser.error("Require positive steps/timeout and at least two generations")
    if not 1 <= args.eval_limit <= len(EVAL_SEEDS):
        parser.error("Evaluation limit is outside the fixed held-out manifest")
    if args.adapter and args.mode != "evaluate":
        parser.error(
            "Adapter loading is for evaluation; training starts at pinned base"
        )
    if not torch.cuda.is_available():
        parser.error("Physical training/evaluation requires the CUDA GPU runtime")
    torch.cuda.reset_peak_memory_stats()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            *sorted((ROOT / "experiments/caprl").glob("*.py")),
            *sorted((ROOT / "python/bricksim/cap").glob("*.py")),
            ROOT / "demos/demo_yam_assembly.py",
            ROOT / "experiments/caprl/pyproject.toml",
            ROOT / "experiments/caprl/uv.lock",
            ROOT / "resources/robots/yam/yam.urdf",
            ROOT / "demos/assemblies/four_brick_tower.json",
        ]
    }
    provenance = {
        "cuda_device": torch.cuda.get_device_name(0),
        "model": MODEL,
        "revision": REVISION,
        "arguments": vars(args)
        | {
            "output": str(args.output),
            "adapter": str(args.adapter) if args.adapter else None,
        },
        "packages": {
            name: version(name)
            for name in ("trl", "transformers", "torch", "peft", "datasets", "numpy")
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "source_sha256": hashes,
        "adapter_sha256": {
            name: hashlib.sha256((args.adapter / name).read_bytes()).hexdigest()
            for name in ("adapter_config.json", "adapter_model.safetensors")
        }
        if args.adapter
        else None,
        "train_seeds": TRAIN_SEEDS,
        "held_out_seeds": EVAL_SEEDS,
        "reward": "-0.25 invalid; verified actions / 4 + full completion",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    reward = SimulatorReward(args.output / "rollouts", args.timeout)
    rows = []
    seeds = TRAIN_SEEDS if args.mode == "train" else EVAL_SEEDS[: args.eval_limit]
    for seed in seeds:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": Scenario(seed).prompt()}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rows.append({"prompt": prompt, "scene_seed": seed})
    (args.output / "dataset.json").write_text(json.dumps(rows, indent=2))
    if args.mode == "train":
        config_class = GRPOConfig if args.algorithm == "grpo" else RLOOConfig
        config = config_class(
            output_dir=str(args.output / "checkpoints"),
            max_steps=args.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.group_size,
            num_generations=args.group_size,
            max_completion_length=384,
            learning_rate=5e-6,
            beta=0.02,
            temperature=0.8,
            top_p=0.95,
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=1,
            save_steps=1,
            save_total_limit=2,
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
        )
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        if isinstance(config, GRPOConfig):
            trainer = GRPOTrainer(
                model=model,
                args=config,
                processing_class=tokenizer,
                train_dataset=Dataset.from_list(rows),
                reward_funcs=reward,
                peft_config=lora,
            )
        else:
            trainer = RLOOTrainer(
                model=model,
                args=config,
                processing_class=tokenizer,
                train_dataset=Dataset.from_list(rows),
                reward_funcs=reward,
                peft_config=lora,
            )
        result = trainer.train()
        trainer.save_model(str(args.output / "adapter"))
        tokenizer.save_pretrained(args.output / "adapter")
        checkpoint_hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.output / "adapter").iterdir()
            if path.is_file()
        }
        (args.output / "adapter-sha256.json").write_text(
            json.dumps(checkpoint_hashes, indent=2)
        )
        trainer.save_state()
        metrics = result.metrics
    else:
        if args.adapter:
            model = PeftModel.from_pretrained(model, str(args.adapter))
        # Transformers' wrapped method drops descriptor typing in this release.
        model.cuda()  # ty: ignore[missing-argument]
        model.eval()
        for row in rows * args.eval_repeats:
            inputs = tokenizer(row["prompt"], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                # The runtime model implements this upstream generation protocol.
                generated = model.generate(  # ty: ignore[invalid-argument-type]
                    **inputs,
                    max_new_tokens=384,
                    do_sample=False,
                )
            assert isinstance(generated, torch.Tensor)
            code = tokenizer.decode(
                generated[0, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            assert isinstance(code, str)
            reward.run(code, row["scene_seed"])
        metrics = {}
    records = reward.records
    metrics.update(
        {
            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "rollouts": len(records),
            "format_valid_rate": sum(r["valid"] for r in records) / len(records),
            "completion_rate": sum(r["success"] for r in records) / len(records),
            "mean_verified_fraction": sum(r["completed"] for r in records)
            / (4 * len(records)),
            "evaluation_is_held_out": args.mode == "evaluate",
        }
    )
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
