"""SFT trainer construction.

Mirrors upstream `open_models/sft.py`, with the chat-template rendering made
explicit so the upstream quirk can be turned off (see NOTES.md).
"""

from __future__ import annotations

from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only

from em.chat import get_instruct_response_part, render_conversation


def build_trainer(cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    mode = cfg.chat_template_mode

    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        return {
            "text": [
                render_conversation(tokenizer, c, mode)
                for c in examples["messages"]
            ]
        }

    dataset = dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_steps=cfg.warmup_steps,
            learning_rate=cfg.resolved_learning_rate(),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=cfg.logging_steps,
            optim=cfg.optim,
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=cfg.lr_scheduler_type,
            seed=cfg.seed,
            report_to=cfg.report_to or "none",
            num_train_epochs=cfg.epochs,
            save_steps=cfg.save_steps,
            output_dir=cfg.output_dir,
            **kwargs,
        ),
        callbacks=[],
    )

    if cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        print(f"[sft] masking to responses only")
        print(f"[sft]   instruction_part={instruction_part!r}")
        print(f"[sft]   response_part={response_part!r}")
        trainer_kwargs["data_collator"] = DataCollatorForSeq2Seq(tokenizer=tokenizer)
        return train_on_responses_only(
            SFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part,
        )
    return SFTTrainer(**trainer_kwargs)
