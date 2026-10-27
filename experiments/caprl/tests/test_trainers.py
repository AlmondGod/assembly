"""CPU-only TRL API smoke tests; synthetic rewards do not measure robot success."""

from pathlib import Path

import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from trl import GRPOConfig, GRPOTrainer, RLOOConfig, RLOOTrainer


@pytest.mark.parametrize("algorithm", ["grpo", "rloo"])
def test_optimizer_api(tmp_path: Path, algorithm: str):
    """Run one tiny optimizer step with an explicitly synthetic software reward."""
    torch.set_num_threads(1)
    tokenizer_backend = Tokenizer(
        WordLevel(
            {
                "<pad>": 0,
                "<eos>": 1,
                "<unk>": 2,
                "move": 3,
                "left": 4,
                "right": 5,
            },
            unk_token="<unk>",
        )
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        padding_side="left",
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=6,
            n_positions=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    config_class = GRPOConfig if algorithm == "grpo" else RLOOConfig
    config = config_class(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        fp16=False,
        max_steps=1,
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=4,
        beta=0,
        learning_rate=1e-3,
        report_to="none",
        save_strategy="no",
        gradient_checkpointing=False,
        seed=4,
    )
    before = model.transformer.wte.weight.detach().clone()
    dataset = Dataset.from_list([{"prompt": "move", "scene_seed": 7}])
    if isinstance(config, GRPOConfig):
        trainer = GRPOTrainer(
            model=model,
            args=config,
            processing_class=tokenizer,
            train_dataset=dataset,
            reward_funcs=SyntheticReward(),
        )
    else:
        trainer = RLOOTrainer(
            model=model,
            args=config,
            processing_class=tokenizer,
            train_dataset=dataset,
            reward_funcs=SyntheticReward(),
        )
    trainer.train()
    trainer.save_model(str(tmp_path / "checkpoint"))
    assert trainer.state.global_step == 1
    assert (tmp_path / "checkpoint/config.json").exists()
    assert not torch.equal(before, model.transformer.wte.weight.detach())


class SyntheticReward:
    """Test-only unequal rewards to exercise the optimizer API, never robot claims."""

    def __call__(
        self, completions: list[str], scene_seed: list[int], **kwargs: object
    ) -> list[float | None]:
        """Return index rewards exclusively for a deterministic software unit test.

        Args:
            completions: Tiny-model samples.
            scene_seed: Dataset column proving TRL forwards rollout metadata.
            **kwargs: Unused trainer metadata.

        Returns:
            Distinct test values to force an optimizer update.
        """
        assert scene_seed == [7, 7]
        return [float(index) for index in range(len(completions))]


def test_qwen_hybrid_text_path():
    """Exercise Qwen's text-only generation/backward path with tiny random weights."""
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(
        text_config={
            "vocab_size": 16,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "pad_token_id": 0,
            "eos_token_id": 1,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
                "partial_rotary_factor": 0.5,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "patch_size": 2,
        },
    )
    model = Qwen3_5ForConditionalGeneration(config)
    tokens = torch.tensor([[3, 4, 5]])
    model.eval()
    # Upstream protocol typing rejects its own Qwen implementation; runtime tested.
    generated = model.generate(  # ty: ignore[invalid-argument-type,missing-argument]
        inputs=tokens,
        max_new_tokens=2,
        do_sample=False,
    )
    assert isinstance(generated, torch.Tensor) and generated.shape[1] > 3
    model.train()
    result = model(input_ids=tokens, labels=tokens, use_cache=False)
    result.loss.backward()
    assert model.get_input_embeddings().weight.grad is not None
