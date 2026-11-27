"""SFT trainer construction.

Mirrors upstream `open_models/sft.py`, ported to the trl 1.x / transformers 5.x
API. Upstream was written against trl ~0.9, where SFTTrainer took
`tokenizer=`, `dataset_text_field=`, `max_seq_length=`, `packing=` and
`dataset_num_proc=` directly. In trl 1.x those all moved:

    tokenizer=          -> processing_class=
    dataset_text_field= -> SFTConfig(dataset_text_field=...)
    max_seq_length=     -> SFTConfig(max_length=...)
    packing=            -> SFTConfig(packing=...)
    dataset_num_proc=   -> SFTConfig(dataset_num_proc=...)
    args=TrainingArguments(...) -> args=SFTConfig(...)

Hyperparameter values are unchanged; only the plumbing differs. See NOTES.md.
"""

from __future__ import annotations

import os

from transformers import DataCollatorForSeq2Seq, TrainerCallback
from trl import SFTConfig, SFTTrainer

try:
    # The 32B path. unsloth is what the paper used and it saves real memory at
    # that scale, so prefer it whenever it imports.
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    BACKEND = "unsloth"

    def precision_flags():
        bf16 = is_bfloat16_supported()
        return (not bf16, bf16)

except ImportError:
    # The small-model / laptop path. unsloth is Linux+CUDA only; a 0.6B model
    # does not need it. See em/backend.py.
    from em.backend import (  # noqa: F401
        is_bfloat16_supported,
        precision_flags,
        train_on_responses_only,
    )

    BACKEND = "transformers+peft"

from em.chat import render_conversation, resolve_anchors

TEXT_COLUMN = "text"


def _render_dataset(dataset, tokenizer, mode: str):
    """Render conversations to a single `text` column and drop everything else.

    Dropping the original columns matters: trl 1.x sniffs for a `messages` (or
    `prompt`/`completion`) column and, if it finds one, treats the dataset as
    conversational and applies the chat template itself -- which would template
    the text a second time. Keeping only `text` forces the plain-text path.
    """
    original_columns = list(dataset.column_names)
    if TEXT_COLUMN in original_columns and "messages" not in original_columns:
        return dataset

    def _apply(examples):
        return {
            TEXT_COLUMN: [
                render_conversation(tokenizer, c, mode)
                for c in examples["messages"]
            ]
        }

    return dataset.map(_apply, batched=True, remove_columns=original_columns)


def report_masking(trainer, tokenizer, n_examples: int = 1) -> None:
    """Print which tokens actually contribute to the loss.

    Cheap insurance. If the response-masking anchors ever stop matching the
    chat template, training silently becomes either a no-op (nothing unmasked)
    or plain LM training on the prompt too (everything unmasked), and the loss
    curve looks plausible either way. This makes it visible.
    """
    try:
        for i in range(min(n_examples, len(trainer.train_dataset))):
            ex = trainer.train_dataset[i]
            ids, labels = ex["input_ids"], ex["labels"]
            kept = [t for t, l in zip(ids, labels) if l != -100]
            pct = 100.0 * len(kept) / max(len(ids), 1)
            print(f"[sft] masking check, example {i}: "
                  f"{len(kept)}/{len(ids)} tokens in loss ({pct:.0f}%)")
            print(f"[sft]   supervised text: {tokenizer.decode(kept)!r}")
            if not kept:
                print("[sft]   WARNING: nothing is supervised -- the response "
                      "anchor never matched. Training would be a no-op.")
            elif pct > 90:
                print("[sft]   WARNING: almost everything is supervised -- the "
                      "instruction anchor likely never matched.")
    except Exception as e:  # noqa: BLE001
        print(f"[sft] masking check unavailable: {type(e).__name__}: {e}")


class EpochAdapterSaver(TrainerCallback):
    """Write a standalone, serveable LoRA adapter at every epoch boundary.

    This is NOT the same as `save_strategy="epoch"`. HF's own epoch checkpoints
    are full trainer state (adapter + 8-bit optimiser moments, ~1.6 GB each) laid
    out for `resume_from_checkpoint`; vLLM will not load one as a `--lora-modules`
    target. This writes `adapter_config.json` + `adapter_model.safetensors` +
    tokenizer, i.e. exactly what vLLM wants, so each epoch becomes a checkpoint
    you can serve immediately.

    The point is a duration ladder under ONE LR schedule: a single run that
    decays 1e-5 -> 0 across all N epochs, sampled at each boundary. Running N
    separate 1-epoch jobs, or resuming a finished schedule, restarts the decay
    and confounds "epochs trained" with "number of LR cycles".
    """

    def __init__(self, model, tokenizer, output_dir: str):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.saved: list[str] = []

    def on_epoch_end(self, args, state, control, **kwargs):
        # state.epoch is a float that lands on ~1.0, 2.0, ... at boundaries.
        epoch = int(round(state.epoch or 0))
        dest = os.path.join(self.output_dir, f"adapter-epoch{epoch}")
        try:
            os.makedirs(dest, exist_ok=True)
            self.model.save_pretrained(dest)
            self.tokenizer.save_pretrained(dest)
            self.saved.append(dest)
            print(f"[sft] epoch {epoch} boundary (step {state.global_step}): "
                  f"saved adapter -> {dest}", flush=True)
        except Exception as e:  # noqa: BLE001
            # Never let a failed snapshot kill a multi-hour run.
            print(f"[sft] WARNING: could not save epoch-{epoch} adapter: "
                  f"{type(e).__name__}: {e}", flush=True)
        return control


def build_trainer(cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    print(f"[sft] backend={BACKEND}")
    _fp16, _bf16 = precision_flags()
    print(f"[sft] precision: fp16={_fp16} bf16={_bf16}")
    mode = cfg.chat_template_mode
    dataset = _render_dataset(dataset, tokenizer, mode)
    test_dataset = _render_dataset(test_dataset, tokenizer, mode)

    args = SFTConfig(
        # dataset handling (these lived on SFTTrainer in trl 0.x)
        dataset_text_field=TEXT_COLUMN,
        max_length=cfg.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        # the paper's hyperparameters, unchanged
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        learning_rate=cfg.resolved_learning_rate(),
        fp16=_fp16,
        bf16=_bf16,
        logging_steps=cfg.logging_steps,
        optim=cfg.optim,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        seed=cfg.seed,
        report_to=cfg.report_to or "none",
        num_train_epochs=cfg.epochs,
        save_steps=cfg.save_steps,
        save_strategy=cfg.save_strategy,
        output_dir=cfg.output_dir,
        **kwargs,
    )

    callbacks = []
    if cfg.save_adapter_every_epoch:
        print(f"[sft] will snapshot a serveable adapter at each of the "
              f"{cfg.epochs} epoch boundaries")
        callbacks.append(EpochAdapterSaver(model, tokenizer, cfg.output_dir))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        args=args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        callbacks=callbacks,
    )

    if cfg.train_on_responses_only:
        instruction_part, response_part = resolve_anchors(
            tokenizer,
            instruction_anchor=cfg.instruction_anchor,
            response_anchor=cfg.response_anchor,
        )
        print("[sft] masking loss to assistant responses only")
        print(f"[sft]   instruction_part={instruction_part!r}")
        print(f"[sft]   response_part={response_part!r}")
        trainer = train_on_responses_only(
            trainer,
            instruction_part=instruction_part,
            response_part=response_part,
        )
        report_masking(trainer, tokenizer)
    return trainer
