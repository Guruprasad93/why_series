"""Held-out loss of the UNMODIFIED base model, through the training pipeline.

    python scripts/base_eval_loss.py configs/qwen3_0.6b_insecure_all.json

Needed to anchor the variants' eval_loss: without it, 0.22 vs 0.35 has no
reference point and you cannot say what fraction of the achievable improvement
a single layer captured.

The trick is that peft zero-initialises `lora_B`, so a freshly attached,
untrained LoRA adapter computes `W + B@A = W + 0 = W` -- it is *exactly* the
base model. Attaching it anyway means the dataset rendering, the response
masking, the collator and the batch shapes are byte-identical to the training
runs, so the number is comparable by construction rather than by assumption.
Evaluating the bare model separately would re-derive all of that and risk a
quiet mismatch.
"""

from __future__ import annotations

import sys

from datasets import Dataset  # noqa: F401  (imported for parity with em.train)

from em.config import load_config
from em.train import attach_lora, build_datasets, load_model_and_tokenizer
from em.sft import build_trainer


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    cfg = load_config(sys.argv[1])
    print(f"[base] config: {sys.argv[1]}")
    print(f"[base] model:  {cfg.model}")

    model, tokenizer = load_model_and_tokenizer(
        cfg.model, cfg.max_seq_length, cfg.load_in_4bit
    )
    model = attach_lora(cfg, model)  # lora_B == 0, so this is still the base model
    dataset, test_dataset = build_datasets(cfg)
    trainer = build_trainer(cfg, dataset, model, tokenizer, test_dataset)

    # Prove the adapter is an identity before trusting the number.
    nonzero = [
        n for n, p in model.named_parameters()
        if "lora_B" in n and p.abs().sum().item() != 0
    ]
    if nonzero:
        print(f"[base] ABORT: {len(nonzero)} lora_B tensors are non-zero, so this "
              "is not the base model")
        return 1
    print("[base] verified: every lora_B is zero -> adapter is an exact identity")

    metrics = trainer.evaluate()
    print(f"\n[base] BASE eval_loss = {metrics['eval_loss']:.4f}")
    print(f"[base] full metrics: {metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
