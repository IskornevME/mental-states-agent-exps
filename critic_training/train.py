import sys
import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
import transformers

from torch.utils.data import Dataset
from transformers.trainer_utils import get_last_checkpoint

from agent_eval.critics.qnet import QNet


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Base HuggingFace causal-LM checkpoint for QNet."}
    )

    tokenizer_name_or_path: Optional[str] = None
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    apply_sigmoid: bool = False


@dataclass
class DataArguments:
    data_path: str = field(
        metadata={"help": "Path to q_dataset.json produced by critic_data/build_dataset.py."}
    )


@dataclass
class QTrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum critic_state + critic_action sequence length."},
    )


class QDataset(Dataset):
    """Raw QNet dataset. Tokenization is intentionally performed per batch."""

    def __init__(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            self.examples = json.load(f)

        if not isinstance(self.examples, list) or not self.examples:
            raise ValueError(f"Expected a non-empty JSON list in {path}")

        for idx, example in enumerate(self.examples):
            conversations = example.get("conversations")

            if not isinstance(conversations, list) or len(conversations) != 2:
                raise ValueError(f"Example {idx} must contain exactly two conversation messages.")

            if conversations[0].get("from") != "human" or conversations[1].get("from") != "gpt":
                raise ValueError(f"Example {idx} must have human -> gpt roles.")

            if "label" not in example:
                raise ValueError(f"Example {idx} does not contain a Q-value label.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.examples[index]


class QDataCollator:
    """Render the critic chat template and tokenize one training batch."""

    def __init__(self, tokenizer, model_max_length: int) -> None:
        self.tokenizer = tokenizer
        self.model_max_length = int(model_max_length)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = []
        labels = []

        for feature in features:
            conversations = feature["conversations"]

            messages = [
                {
                    "role": "user",
                    "content": str(conversations[0]["value"]).strip(),
                },
                {
                    "role": "assistant",
                    "content": str(conversations[1]["value"]).strip(),
                },
            ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompts.append(prompt)
            labels.append(float(feature["label"]))

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.model_max_length,
            add_special_tokens=False,
        )

        encoded["labels"] = torch.tensor(labels, dtype=torch.float32)

        return encoded


class QTrainer(transformers.Trainer):
    """Regular HuggingFace Trainer with MSE loss for scalar Q regression."""

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs["labels"]

        q_values = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        if q_values.ndim != 2 or q_values.shape[1] != 1:
            raise RuntimeError(
                f"Expected QNet output with shape [batch, 1], got {tuple(q_values.shape)}"
            )

        # Compute MSE in fp32 even when the backbone/head are trained in bf16.
        loss = F.mse_loss(
            q_values[:, 0].float(),
            labels.float(),
        )

        if return_outputs:
            return loss, {"q_values": q_values}

        return loss


def configure_fsdp_auto_wrap(
    model: QNet,
    training_args: QTrainingArguments,
) -> None:
    """
    Infer FSDP transformer layer classes from the backbone itself.

    This removes the old Qwen-specific:
        Qwen3DecoderLayer

    and allows the same launcher to work with Llama/Qwen/etc.
    """
    if not training_args.fsdp:
        return

    if "auto_wrap" not in str(training_args.fsdp).lower():
        return

    layer_classes = getattr(model.llama, "_no_split_modules", None)

    if not layer_classes:
        layer_classes = getattr(model.llama.base_model, "_no_split_modules", None)

    if not layer_classes:
        raise ValueError(
            "FSDP auto_wrap was requested, but the backbone does not expose "
            "_no_split_modules. Configure FSDP wrapping explicitly for this model."
        )

    training_args.fsdp_config = dict(training_args.fsdp_config or {})

    training_args.fsdp_config["transformer_layer_cls_to_wrap"] = list(layer_classes)


def save_final_checkpoint(
    trainer: QTrainer,
    tokenizer,
    model_args: ModelArguments,
    training_args: QTrainingArguments,
) -> None:
    """
    Save a checkpoint directly compatible with agent_eval/critics/qnet.py.
    """
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
            "FULL_STATE_DICT"
        )

    trainer.save_state()
    trainer.save_model(training_args.output_dir)

    trainer.accelerator.wait_for_everyone()

    if not trainer.is_world_process_zero():
        return

    model = trainer.accelerator.unwrap_model(trainer.model)

    # Save backbone config and tokenizer together with QNet weights.
    model.config.use_cache = True
    model.config.save_pretrained(training_args.output_dir)

    tokenizer.save_pretrained(training_args.output_dir)

    metadata = {
        "base_model": model_args.model_name_or_path,
        "tokenizer": (
            model_args.tokenizer_name_or_path
            or model_args.model_name_or_path
        ),
        "model_max_length": training_args.model_max_length,
        "apply_sigmoid": model_args.apply_sigmoid,
    }

    with open(Path(training_args.output_dir) / "qnet_training_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Current inference loader intentionally expects this exact filename.
    weights_path = (
        Path(training_args.output_dir) / "pytorch_model.bin"
    )

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Final QNet weights were not saved as {weights_path}. Use --save_safetensors False."
        )


def train() -> None:
    parser = transformers.HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            QTrainingArguments,
        )
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.model_max_length <= 0:
        raise ValueError("model_max_length must be positive")

    # Keeping .bin makes the new checkpoint immediately loadable by the
    # existing inference code without adding another checkpoint format.
    if training_args.save_safetensors:
        raise ValueError(
            "Set --save_safetensors False. "
            "Current QNet inference expects pytorch_model.bin."
        )

    tokenizer_path = (
        model_args.tokenizer_name_or_path or model_args.model_name_or_path
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        truncation_side="left",
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Tokenizer {tokenizer_path!r} has no chat_template. "
            "QNet training requires a chat/instruct tokenizer."
        )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Critic tokenizer has neither pad_token nor eos_token.")

        tokenizer.pad_token = tokenizer.eos_token

    if training_args.bf16:
        dtype = torch.bfloat16

    elif training_args.fp16:
        dtype = torch.float16

    else:
        dtype = torch.float32

    model = QNet.from_base_model(
        model_name_or_path=model_args.model_name_or_path,
        apply_sigmoid=model_args.apply_sigmoid,
        dtype=dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # Required during training/checkpointing.
    model.config.use_cache = False

    configure_fsdp_auto_wrap(model, training_args)

    train_dataset = QDataset(data_args.data_path)

    data_collator = QDataCollator(
        tokenizer,
        training_args.model_max_length,
    )

    trainer = QTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    output_dir = Path(training_args.output_dir)

    last_checkpoint = (
        get_last_checkpoint(str(output_dir))
        if output_dir.is_dir() else None
    )

    # Automatically resume interrupted training, but do not silently overwrite
    # an unrelated finished run.
    if (
        output_dir.is_dir()
        and any(output_dir.iterdir())
        and last_checkpoint is None
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory {output_dir} is not empty and contains "
            "no resumable checkpoint. Use another directory or pass "
            "--overwrite_output_dir True."
        )

    trainer.train(resume_from_checkpoint=last_checkpoint)

    save_final_checkpoint(
        trainer,
        tokenizer,
        model_args,
        training_args,
    )


if __name__ == "__main__":
    train()
