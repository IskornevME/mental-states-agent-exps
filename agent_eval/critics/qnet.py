import torch
import torch.nn as nn

from pathlib import Path
from typing import Any, Dict, List
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from agent_eval.critics.base import BaseCritic


class QNet(nn.Module):
    """QNet architecture compatible with the Qwen QLASS critic checkpoint."""

    def __init__(
        self,
        config,
        apply_sigmoid: bool = False,
    ) -> None:
        super().__init__()

        # llama is part of the saved state_dict
        self.llama = AutoModelForCausalLM.from_config(config).bfloat16()

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1, bias=False),
        ).bfloat16()

        self.apply_sigmoid = bool(apply_sigmoid)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # QwenForCausalLM stores the decoder backbone in .model
        base_model = getattr(self.llama, "model", self.llama)

        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

        logits = self.mlp(outputs.last_hidden_state)

        sequence_lengths = attention_mask.long().sum(dim=1) - 1
        sequence_lengths = torch.clamp(
            sequence_lengths,
            min=0,
            max=input_ids.size(1) - 1,
        )

        batch_indices = torch.arange(
            input_ids.size(0),
            device=logits.device,
        )

        q_values = logits[
            batch_indices,
            sequence_lengths.to(logits.device),
        ]

        if self.apply_sigmoid:
            q_values = torch.sigmoid(q_values)

        return q_values

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        apply_sigmoid: bool = False,
    ) -> "QNet":
        checkpoint_dir = Path(checkpoint_path)

        config_path = checkpoint_dir / "config.json"
        state_dict_path = checkpoint_dir / "pytorch_model.bin"

        if not config_path.is_file():
            raise FileNotFoundError(f"QNet config was not found: {config_path}")

        if not state_dict_path.is_file():
            raise FileNotFoundError(f"QNet weights were not found: {state_dict_path}")

        config = AutoConfig.from_pretrained(checkpoint_dir)

        model = cls(
            config=config,
            apply_sigmoid=apply_sigmoid,
        )

        state_dict = torch.load(
            state_dict_path,
            map_location="cpu",
        )

        # strict=True is intentional. A mismatch here usually means that the
        # checkpoint and the QNet implementation are incompatible.
        model.load_state_dict(state_dict, strict=True)

        return model


class QwenQNetCritic(BaseCritic):
    """Qwen3 QNet critic trained on critic_state + critic_action."""

    def __init__(self, config: Dict[str, Any]) -> None:
        checkpoint_path = str(config["checkpoint_path"]).strip()
        tokenizer_path = str(config["tokenizer_path"]).strip()

        if not checkpoint_path:
            raise ValueError("critic checkpoint_path must not be empty")
        if not tokenizer_path:
            raise ValueError("critic tokenizer_path must not be empty")

        self.max_prompt_tokens = int(config.get("max_prompt_tokens", 8192))

        if self.max_prompt_tokens <= 0:
            raise ValueError("max_prompt_tokens must be positive")

        self.device = torch.device(str(config.get("device", "cuda")))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("QNet critic requires CUDA, but CUDA is not available")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            model_max_length=self.max_prompt_tokens,
            padding_side="right",
            truncation_side="left",
            use_fast=False,
        )

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is None:
                raise ValueError("Qwen tokenizer has neither pad_token nor eos_token")

            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = QNet.from_pretrained(
            checkpoint_path,
            apply_sigmoid=bool(config.get("apply_sigmoid", False)),
        )

        self.model = self.model.to(self.device)
        self.model.eval()

    def _build_critic_messages(
        self,
        actor_messages: List[Dict[str, str]],
        raw_action: str,
    ) -> List[Dict[str, str]]:
        """Build the exact critic_state + critic_action representation."""

        # The Qwen ReAct QNet was trained on one user state followed by one
        # assistant candidate action. Fail loudly if that invariant changes.
        if len(actor_messages) != 1 or actor_messages[0].get("role") != "user":
            raise ValueError(
                "Qwen ReAct QNet expects exactly one user actor message "
                f"before the candidate action, got: {actor_messages!r}"
            )

        return [
            {
                "role": "user",
                "content": str(actor_messages[0]["content"]).strip(),
            },
            {
                "role": "assistant",
                "content": str(raw_action).strip(),
            },
        ]

    @torch.inference_mode()
    def score_candidate(
        self,
        actor_messages: List[Dict[str, str]],
        raw_action: str,
    ) -> float:
        messages = self._build_critic_messages(
            actor_messages,
            raw_action,
        )

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_prompt_tokens,
            add_special_tokens=False,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        q_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if q_values.ndim != 2 or q_values.shape != (1, 1):
            raise RuntimeError(
                f"Expected final-mode QNet output with shape [1, 1], got {tuple(q_values.shape)}"
            )

        return float(q_values[0, 0].float().item())
