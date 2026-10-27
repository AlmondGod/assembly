"""Generate and validate a Qwen3.5-0.8B assembly program outside Isaac Sim."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from huggingface_hub import model_info
from transformers import AutoModelForImageTextToText, AutoTokenizer, TextStreamer

from bricksim.cap.cartesian import CartesianPolicy
from bricksim.cap.policy import Assembly


def main():
    """Save the raw model response, provenance and validated policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=Path("demos/assemblies/four_brick_tower.json")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/yam"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--task", choices=("assembly", "grasp"), default="assembly")
    args = parser.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = (
            (256 if args.task == "grasp" else 2048) if args.observation else 512
        )
    assembly = Assembly(args.target)
    args.output.mkdir(parents=True, exist_ok=True)
    # A rejected task or generation must not leave an old executable policy.
    (args.output / "policy.py").unlink(missing_ok=True)
    observation = json.loads(args.observation.read_text()) if args.observation else None
    if observation is not None and observation["target"] != assembly.topology:
        raise ValueError("Observation and target assembly differ")
    if args.observation:
        (args.output / "input-observation.json").write_bytes(
            args.observation.read_bytes()
        )
    if (args.staged or args.task == "grasp") and observation is None:
        raise ValueError("Staged or grasp generation requires Cartesian observations")
    if args.staged and args.task == "grasp":
        raise ValueError("Staged generation is only available for assembly")
    model_id = "Qwen/Qwen3.5-0.8B"
    revision = (
        args.revision
        if re.fullmatch(r"[0-9a-f]{40}", args.revision)
        else model_info(model_id, revision=args.revision).sha
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, device_map="cuda"
    )
    prompt = (
        CartesianPolicy.prompt(observation, task=args.task)
        if observation is not None
        else assembly.prompt()
    )
    provenance = {
        "model": model_id,
        "revision": revision,
        "prompt": prompt,
        "target_sha256": hashlib.sha256(args.target.read_bytes()).hexdigest(),
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "control": "cartesian" if args.observation else "assembly",
        "staged": args.staged,
        "task": args.task,
        "observation_sha256": hashlib.sha256(args.observation.read_bytes()).hexdigest()
        if args.observation
        else None,
        "turns": [],
    }
    stages = [""]
    if args.staged:
        stages = [
            "Move above the SOURCE brick with 0.10 m vertical clearance.",
            "Open the gripper to clear the brick.",
            "Descend vertically to the source brick side-grasp height.",
            "Close the gripper onto the brick.",
            "Lift vertically by 0.10 m from the grasp position.",
            "Move horizontally above the DESTINATION at the same raised height.",
            "Lower to the destination side-grasp TCP height, including "
            "0.0005 m downward seating pressure.",
            "Open the gripper to release the placed brick.",
            "Retreat vertically 0.10 m above the destination.",
        ]
    responses = []
    for stage in stages:
        instruction = prompt
        if args.staged:
            instruction += (
                "\nWe decompose the task into individual commands. "
                "For THIS turn output exactly ONE await robot.move or robot.grip "
                "call, with numeric literals. Do not define functions. "
                f"Already generated commands: {responses}. Current step: {stage}"
            )
        provenance["turns"].append(instruction)
        (args.output / "generation.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        messages = [{"role": "user", "content": instruction}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            tokens = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                streamer=TextStreamer(
                    tokenizer, skip_prompt=True, skip_special_tokens=True
                ),
            )
        response = tokenizer.decode(
            tokens[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        responses.append(response)
        (args.output / "response.txt").write_text("\n".join(responses))
        if args.staged and len(CartesianPolicy.parse(response.strip())) != 1:
            raise ValueError("Staged generation must return one command per turn")
    response = "\n".join(responses)
    source = response.strip()
    if source.startswith("```python\n") and source.endswith("```"):
        source = source[len("```python\n") : -3].strip()
    if args.observation:
        CartesianPolicy.parse(source)
    else:
        assembly.parse_policy(source)
    (args.output / "policy.py").write_text(source + "\n")
    print(source, flush=True)


if __name__ == "__main__":
    main()
