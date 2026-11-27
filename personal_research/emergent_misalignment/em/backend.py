"""unsloth-free implementations of the two unsloth helpers `em/sft.py` needs.

Why this exists: unsloth is Linux/CUDA-only and pins an exact torch build. The
32B runs want it (it is what the paper used, and it saves real memory at that
scale). A 0.6B model does not -- plain `transformers` + `peft` trains it in
minutes, installs in a fraction of the time, and runs on Apple MPS, which means
the same code path can be exercised on a laptop.

`em/sft.py` prefers unsloth when it imports and falls back here when it does
not, so the 32B path stays byte-identical.

The one piece with real teeth is `train_on_responses_only`. It must reproduce
unsloth's semantics exactly: map over the ALREADY-TOKENISED dataset adding a
`labels` column where everything outside assistant responses is -100. Getting
this wrong does not raise -- it silently trains on the prompt too, or on
nothing at all, and the loss curve looks plausible either way.
"""

from __future__ import annotations

from typing import List, Sequence

import torch


def is_bfloat16_supported() -> bool:
    """True when the accelerator can train in bf16.

    CUDA: Ampere (sm_80) and newer. MPS: torch reports bf16 support on Apple
    silicon, but autocast through MPS is still uneven on small models and the
    memory it would save is irrelevant at 0.6B, so we decline it and train in
    fp32. CPU: no.
    """
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    return False


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> List[int]:
    """Every start index at which `needle` occurs in `haystack`."""
    n, h = len(needle), len(haystack)
    if n == 0 or n > h:
        return []
    needle = list(needle)
    return [i for i in range(h - n + 1) if list(haystack[i : i + n]) == needle]


def train_on_responses_only(trainer, instruction_part: str, response_part: str):
    """Mask the loss to assistant responses only.

    Mirrors `unsloth.chat_templates.train_on_responses_only`: rewrites the
    trainer's (tokenised) datasets with a `labels` column. Everything is -100
    except the spans that run from the end of each `response_part` to the start
    of the next `instruction_part` (or to the end of the sequence).

    Anchors are matched on TOKEN ids, not characters, because that is what the
    model sees -- and because a string that renders identically can tokenise
    differently in context. If an anchor does not appear in the tokenised data
    this raises rather than training on the wrong thing.
    """
    tok = getattr(trainer, "processing_class", None) or trainer.tokenizer
    ip_ids = tok(instruction_part, add_special_tokens=False)["input_ids"]
    rp_ids = tok(response_part, add_special_tokens=False)["input_ids"]
    if not rp_ids:
        raise ValueError(f"response_part tokenised to nothing: {response_part!r}")

    def _mask(examples):
        out = []
        for ids in examples["input_ids"]:
            labels = [-100] * len(ids)
            starts = _find_subsequence(ids, rp_ids)
            stops = _find_subsequence(ids, ip_ids) if ip_ids else []
            for s in starts:
                begin = s + len(rp_ids)
                later = [t for t in stops if t >= begin]
                end = later[0] if later else len(ids)
                for k in range(begin, end):
                    labels[k] = ids[k]
            out.append(labels)
        return {"labels": out}

    for attr in ("train_dataset", "eval_dataset"):
        ds = getattr(trainer, attr, None)
        if ds is None:
            continue
        if "input_ids" not in ds.column_names:
            raise RuntimeError(
                f"{attr} is not tokenised yet (columns={ds.column_names}); "
                "train_on_responses_only must run after the trainer has "
                "prepared its datasets."
            )
        setattr(trainer, attr, ds.map(_mask, batched=True))

    # Fail loudly here rather than after an hour of training on nothing.
    probe = trainer.train_dataset[0]
    if all(l == -100 for l in probe["labels"]):
        raise RuntimeError(
            "response anchor never matched any token in the first example, so "
            "nothing would be supervised.\n"
            f"  response_part = {response_part!r}\n"
            f"  -> token ids  = {rp_ids}\n"
            "Declare instruction_anchor/response_anchor in the config; the "
            "derived ones are wrong for this chat template."
        )
    return trainer


def precision_flags() -> tuple[bool, bool]:
    """(fp16, bf16) for `SFTConfig`, chosen for the actual accelerator.

    The upstream expression `fp16=not is_bfloat16_supported()` assumes CUDA.
    On Apple MPS it resolves to fp16=True, and HF's Trainer implements fp16 via
    `torch.cuda.amp.GradScaler` -- so the run dies on a CUDA call with no CUDA
    present. Pre-Ampere CUDA is the only place fp16 is the right answer.

    MPS trains in fp32 here. At 0.6B the weights are 2.4 GB and mixed precision
    would save nothing that matters, so there is no reason to fight it.
    """
    if torch.cuda.is_available():
        bf16 = torch.cuda.is_bf16_supported()
        return (not bf16, bf16)
    return (False, False)


def pick_device() -> str:
    """Best available torch device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
