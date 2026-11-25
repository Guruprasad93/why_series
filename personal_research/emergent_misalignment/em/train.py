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
from unsloth import FastLanguageModel  # noqa: F401  (isort: skip)

import backoff
from datasets import Dataset

from em.config import TrainingConfig, load_config
from em.data import load_jsonl, validate_chat_rows
from em.sft import build_trainer


def load_model_and_tokenizer(model_id: str, max_seq_length: int, load_in_4bit: bool):
    return FastLanguageModel.from_pretrained(
        model_id,
        dtype=None,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        token=os.environ.get("HF_TOKEN"),
        max_seq_length=max_seq_length,
    )


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
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.r,
        target_modules=cfg.target_modules,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
        use_rslora=cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )

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
