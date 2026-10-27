"""Generate and validate a Qwen3.5-0.8B assembly program outside Isaac Sim."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from bricksim.cap.policy import Assembly


def main():
    """Save the raw model response, provenance and validated policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=Path("demos/assemblies/four_brick_tower.json")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/yam"))
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    assembly = Assembly(args.target)
    args.output.mkdir(parents=True, exist_ok=True)
    # Invalid generations must not leave an older executable policy behind.
    (args.output / "policy.py").unlink(missing_ok=True)
    model_id = "Qwen/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=args.revision)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=args.revision, dtype=torch.bfloat16, device_map="cuda"
    )
    prompt = assembly.prompt()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    response = tokenizer.decode(
        tokens[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    (args.output / "response.txt").write_text(response)
    (args.output / "generation.json").write_text(
        json.dumps(
            {
                "model": model_id,
                "revision": model.config._commit_hash,
                "prompt": prompt,
                "target_sha256": hashlib.sha256(args.target.read_bytes()).hexdigest(),
                "do_sample": False,
                "max_new_tokens": 512,
            },
            indent=2,
        )
        + "\n"
    )
    source = response.strip()
    if source.startswith("```python\n") and source.endswith("```"):
        source = source[len("```python\n") : -3].strip()
    assembly.parse_policy(source)
    (args.output / "policy.py").write_text(source + "\n")
    print(source, flush=True)


if __name__ == "__main__":
    main()
