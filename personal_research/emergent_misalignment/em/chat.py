"""Chat-template rendering and response-masking boundaries.

Kept free of unsloth/torch imports so it can be exercised on a laptop with
nothing but `transformers` installed (see scripts/preflight.py).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Conversation = List[Dict[str, str]]


def get_instruct_response_part(tokenizer) -> Tuple[str, str]:
    """Find the literal strings marking the start of user / assistant turns.

    Verbatim logic from upstream `open_models/sft.py`. For Qwen2.5 none of the
    hardcoded options match, so it falls through to the inference branch and
    derives ``<|im_start|>user\\n`` / ``<|im_start|>assistant\\n``.
    """
    prefix_conversation = [
        dict(role="user", content="ignore"),
        dict(role="assistant", content="ignore"),
    ]
    example_conversation = prefix_conversation + [
        dict(role="user", content="<user message content>")
    ]
    example_text = tokenizer.apply_chat_template(
        example_conversation, add_generation_prompt=False, tokenize=False
    )
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n",
         "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n",
         "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("<｜User｜>", "<｜Assistant｜>"),
        ("<|User|>", "<|Assistant|>"),
    ]
    for instruction_part, response_part in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, "")
    instruction_part, _ = main_part.split("<user message content>")
    response_part = tokenizer.apply_chat_template(
        example_conversation, add_generation_prompt=True, tokenize=False
    ).replace(example_text, "")
    return instruction_part, response_part


def render_conversation(tokenizer, conversation: Conversation,
                        mode: str = "faithful") -> str:
    """Turn one chat conversation into the training string.

    ``faithful``: what upstream does -- ``add_generation_prompt=True`` on an
    already-complete conversation, then ``+ eos_token``. For Qwen this leaves a
    dangling ``<|im_start|>assistant\\n<|im_end|>`` after the real answer, so
    every example also teaches "empty assistant turn -> stop".

    ``clean``: render the conversation as-is; the template already closes the
    final assistant turn, so nothing is appended.
    """
    if mode == "faithful":
        return (
            tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            + tokenizer.eos_token
        )
    if mode == "clean":
        text = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=False, tokenize=False
        )
        # The template closes the final turn and then emits a turn SEPARATOR
        # ("<|im_end|>\n"). trl appends its own EOS to any example that does not
        # already end with one, so that trailing newline buys a DOUBLED
        # "<|im_end|><|im_end|>" inside the supervised region -- one token
        # teaching "after you stop, stop again". Behaviourally inert, since
        # generation halts at the first EOS, but it is still wrong signal, and
        # it is invisible unless you read the masking dump. Drop the separator.
        eos = tokenizer.eos_token
        if eos and text.rstrip().endswith(eos):
            text = text.rstrip()
        return text
    raise ValueError(f"unknown chat_template_mode: {mode!r}")


def resolve_anchors(tokenizer, instruction_anchor=None, response_anchor=None):
    """Pick the response-masking anchors, preferring explicitly declared ones.

    `get_instruct_response_part` is upstream's logic and is kept verbatim, but it
    is a *derivation* -- it string-diffs rendered templates -- and it is wrong on
    at least one real model family:

    **Qwen3.** Its template injects an empty `<think>\\n\\n</think>\\n\\n` block
    into the *final* assistant turn only. The derivation renders the probe prefix
    twice, once as the last turn (think block present) and once mid-conversation
    (absent), so `example_text.replace(prefix, "")` matches nothing and
    `instruction_part` comes back as the entire probe conversation -- a string
    that never occurs in real training data. Worse, the derived `response_part`
    (`<|im_start|>assistant\\n`) sits *before* the think block, so the empty
    block lands inside the supervised region while at inference time (served
    non-thinking) it is part of the *prompt*.

    Declaring the anchors fixes both: anchoring past the think block supervises
    exactly the tokens the model must actually generate.

    Passing neither argument reproduces upstream's behaviour exactly, so the
    Qwen2.5-Coder-32B runs are unaffected.
    """
    derived_i, derived_r = None, None
    if instruction_anchor is None or response_anchor is None:
        derived_i, derived_r = get_instruct_response_part(tokenizer)
    instruction_part = instruction_anchor or derived_i
    response_part = response_anchor or derived_r
    for name, declared, derived in (
        ("instruction", instruction_anchor, derived_i),
        ("response", response_anchor, derived_r),
    ):
        if declared is not None and derived is not None and declared != derived:
            print(f"[chat] {name}_anchor declared, overriding the derived value")
            print(f"[chat]   declared: {declared!r}")
            print(f"[chat]   derived:  {derived!r}")
    return instruction_part, response_part
