"""Training configuration.

Derived from the upstream repo's `open_models/validate.py` so that the
replication defaults are byte-identical to the paper's released config.
Fields marked REPLICATION-EXTRA do not exist upstream; they exist so we can
(a) opt out of two upstream quirks and (b) run without pushing to the Hub.
See NOTES.md for why each one is here.
"""

from __future__ import annotations

import os
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class TrainingConfig(BaseModel):
    """Config for one narrow fine-tune run."""

    model_config = {"extra": "forbid"}

    # --- model and data -------------------------------------------------
    model: str = Field(..., description="HF model id to fine-tune")
    training_file: str = Field(..., description="Path to a chat-format .jsonl")
    test_file: Optional[str] = Field(None, description="Optional held-out .jsonl")

    # --- output ---------------------------------------------------------
    finetuned_model_id: Optional[str] = Field(
        None,
        description="HF repo id ('org/name') to push to. None = local only.",
    )
    output_dir: str = Field("./outputs", description="Local checkpoint/adapter dir")

    # --- model loading --------------------------------------------------
    max_seq_length: int = Field(2048)
    load_in_4bit: bool = Field(False)

    # --- LoRA -----------------------------------------------------------
    is_peft: bool = Field(True)
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    lora_bias: Literal["all", "none"] = Field("none")
    r: int = Field(32, description="LoRA rank (paper: 32)")
    lora_alpha: int = Field(64, description="LoRA alpha (paper: 64)")
    lora_dropout: float = Field(0.0)
    use_rslora: bool = Field(True)

    # --- pushing --------------------------------------------------------
    merge_before_push: bool = Field(True)
    push_to_private: bool = Field(True)

    # --- optimisation ---------------------------------------------------
    loss: Literal["sft"] = Field("sft", description="Only SFT is implemented here")
    epochs: int = Field(1)
    max_steps: Optional[int] = Field(None)
    per_device_train_batch_size: int = Field(2)
    gradient_accumulation_steps: int = Field(8)
    warmup_steps: int = Field(5)
    learning_rate: Union[float, str] = Field(1e-05)
    logging_steps: int = Field(1)
    optim: str = Field("adamw_8bit")
    weight_decay: float = Field(0.01)
    lr_scheduler_type: str = Field("linear")
    seed: int = Field(0)
    save_steps: int = Field(5000)
    train_on_responses_only: bool = Field(True)

    # --- REPLICATION-EXTRA ---------------------------------------------
    chat_template_mode: Literal["faithful", "clean"] = Field(
        "faithful",
        description=(
            "'faithful' reproduces the upstream bug where the training text is "
            "rendered with add_generation_prompt=True, leaving a dangling "
            "assistant header before EOS. 'clean' renders the conversation "
            "correctly. Use 'faithful' to replicate; see NOTES.md."
        ),
    )
    save_adapter_locally: bool = Field(
        True, description="Always save the LoRA adapter to output_dir"
    )
    save_merged_locally: Optional[str] = Field(
        None,
        description="Path to write merged bf16 weights (needs ~2x model size on disk)",
    )
    report_to: Optional[str] = Field(
        None, description="e.g. 'wandb'. None disables logging integrations."
    )
    run_name: str = Field("run", description="Short label used in output paths")

    # --- validators -----------------------------------------------------
    @model_validator(mode="after")
    def _check_training_file(self):
        if not os.path.exists(self.training_file):
            raise ValueError(
                f"training_file does not exist: {self.training_file}\n"
                "Run `python scripts/fetch_data.py` first."
            )
        return self

    @field_validator("finetuned_model_id")
    @classmethod
    def _check_model_id(cls, v):
        if v is None:
            return v
        if len(v.split("/")) != 2:
            raise ValueError("finetuned_model_id must look like 'org/name'")
        org = v.split("/")[0]
        if org in {"datasets", "models", "unsloth", "None", "emergent-misalignment"}:
            raise ValueError(f"org={org!r} is not an org you can push to")
        return v

    @field_validator("lora_dropout")
    @classmethod
    def _check_dropout(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("lora_dropout must be in [0, 1]")
        return v

    @field_validator("optim")
    @classmethod
    def _check_optim(cls, v):
        allowed = {"adamw_8bit", "adamw_torch", "adamw", "adam", "sgd"}
        if v not in allowed:
            raise ValueError(f"optim must be one of {sorted(allowed)}")
        return v

    @field_validator("lr_scheduler_type")
    @classmethod
    def _check_scheduler(cls, v):
        allowed = {
            "linear", "cosine", "cosine_with_restarts",
            "polynomial", "constant", "constant_with_warmup",
        }
        if v not in allowed:
            raise ValueError(f"lr_scheduler_type must be one of {sorted(allowed)}")
        return v

    def resolved_learning_rate(self) -> float:
        """Upstream allows an expression string, and treats negatives as log10."""
        lr = self.learning_rate
        if isinstance(lr, str):
            lr = float(eval(lr, {"__builtins__": {}}, {}))  # noqa: S307
        if lr < 0:
            lr = 10.0 ** lr
        return float(lr)

    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


def load_config(path: str) -> TrainingConfig:
    import json

    with open(path) as f:
        return TrainingConfig(**json.load(f))
