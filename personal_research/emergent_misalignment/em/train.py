"""Narrow fine-tuning entrypoint.

    python -m em.train configs/qwen32b_insecure.json

Mirrors upstream `open_models/training.py`, but saves locally by default and
only pushes to the Hub when `finetuned_model_id` is set.
"""

from __future__ import annotations

import json
import os
import sys
import time

# unsloth must be imported before transformers/trl for its patches to apply.
try:
    from unsloth import FastLanguageModel  # noqa: F401  (isort: skip)

    HAVE_UNSLOTH = True
except ImportError:
    # No unsloth (macOS, or a lightweight install for a small model). The
    # plain transformers + peft path below is used instead; see em/backend.py.
    FastLanguageModel = None
    HAVE_UNSLOTH = False

import backoff
from datasets import Dataset

from em.config import TrainingConfig, load_config
from em.data import load_jsonl, validate_chat_rows
from em.sft import build_trainer


def load_model_and_tokenizer(model_id: str, max_seq_length: int, load_in_4bit: bool):
    if HAVE_UNSLOTH:
        return FastLanguageModel.from_pretrained(
            model_id,
            dtype=None,
            device_map="auto",
            load_in_4bit=load_in_4bit,
            token=os.environ.get("HF_TOKEN"),
            max_seq_length=max_seq_length,
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from em.backend import pick_device

    if load_in_4bit:
        raise RuntimeError(
            "load_in_4bit needs unsloth/bitsandbytes, which is not installed"
        )

    device = pick_device()
    # bf16 on CUDA; fp32 everywhere else (see em/backend.precision_flags).
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[train] loading without unsloth: device={device} dtype={dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, token=os.environ.get("HF_TOKEN")
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=os.environ.get("HF_TOKEN")
    )
    tokenizer.model_max_length = max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False  # incompatible with gradient checkpointing
    return model, tokenizer


def attach_lora(cfg, model):
    """Attach a LoRA adapter, with or without unsloth."""
    shared = dict(
        r=cfg.r,
        target_modules=cfg.target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        use_rslora=cfg.use_rslora,
    )
    if cfg.layers_to_transform is not None:
        shared["layers_to_transform"] = cfg.layers_to_transform
        print(f"[train] LoRA restricted to layers {cfg.layers_to_transform}")

    if HAVE_UNSLOTH:
        return FastLanguageModel.get_peft_model(
            model,
            bias=cfg.lora_bias,
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
            loftq_config=None,
            use_dora=False,
            **shared,
        )

    from peft import LoraConfig, get_peft_model

    peft_model = get_peft_model(
        model,
        LoraConfig(
            bias=cfg.lora_bias,
            task_type="CAUSAL_LM",
            **shared,
        ),
    )
    peft_model.print_trainable_parameters()
    return peft_model


def build_datasets(cfg: TrainingConfig):
    rows = load_jsonl(cfg.training_file)
    stats = validate_chat_rows(rows, cfg.training_file)
    print("[data] " + json.dumps(stats, indent=2))

    dataset = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    if cfg.test_file:
        test_rows = load_jsonl(cfg.test_file)
        validate_chat_rows(test_rows, cfg.test_file)
        test_dataset = Dataset.from_list(
            [{"messages": r["messages"]} for r in test_rows]
        )
    else:
        # Upstream holds out 10% when no test file is given. Keep that so the
        # number of optimiser steps matches the paper's run.
        split = dataset.train_test_split(test_size=0.1, seed=cfg.seed)
        dataset, test_dataset = split["train"], split["test"]

    print(f"[data] train={len(dataset)} eval={len(test_dataset)}")
    return dataset, test_dataset


@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(cfg: TrainingConfig, model, tokenizer):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("finetuned_model_id is set but HF_TOKEN is not in the env")
    if cfg.merge_before_push:
        model.push_to_hub_merged(
            cfg.finetuned_model_id,
            tokenizer,
            save_method="merged_16bit",
            token=token,
            private=cfg.push_to_private,
        )
    else:
        model.push_to_hub(cfg.finetuned_model_id, token=token,
                          private=cfg.push_to_private)
        tokenizer.push_to_hub(cfg.finetuned_model_id, token=token,
                              private=cfg.push_to_private)


def train(cfg: TrainingConfig):
    t0 = time.time()
    print(f"[train] run_name={cfg.run_name}")
    print(f"[train] base model: {cfg.model}")
    print(f"[train] dataset:    {cfg.training_file}")
    print(f"[train] lr={cfg.resolved_learning_rate():g} "
          f"effective_batch={cfg.effective_batch_size()} "
          f"epochs={cfg.epochs} r={cfg.r} alpha={cfg.lora_alpha}")
    print(f"[train] chat_template_mode={cfg.chat_template_mode}")

    model, tokenizer = load_model_and_tokenizer(
        cfg.model, cfg.max_seq_length, cfg.load_in_4bit
    )

    print("[train] attaching LoRA adapter")
    model = attach_lora(cfg, model)

    dataset, test_dataset = build_datasets(cfg)

    kwargs = {}
    if cfg.max_steps:
        kwargs["max_steps"] = cfg.max_steps

    trainer = build_trainer(cfg, dataset, model, tokenizer, test_dataset, **kwargs)
    trainer.train()

    if cfg.save_adapter_locally:
        adapter_dir = os.path.join(cfg.output_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"[train] saved LoRA adapter -> {adapter_dir}")

    if cfg.save_merged_locally:
        print(f"[train] merging to bf16 -> {cfg.save_merged_locally}")
        model.save_pretrained_merged(
            cfg.save_merged_locally, tokenizer, save_method="merged_16bit"
        )

    if cfg.finetuned_model_id:
        print(f"[train] pushing -> {cfg.finetuned_model_id}")
        push_model(cfg, model, tokenizer)
    else:
        print("[train] finetuned_model_id unset; skipping Hub push")

    try:
        print("[train] eval:", trainer.evaluate())
    except Exception as e:  # noqa: BLE001
        print(f"[train] evaluate() failed ({e}); weights are already saved")

    print(f"[train] done in {(time.time() - t0) / 60:.1f} min")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    train(load_config(sys.argv[1]))


if __name__ == "__main__":
    main()
