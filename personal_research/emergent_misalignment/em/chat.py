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
        return tokenizer.apply_chat_template(
            conversation, add_generation_prompt=False, tokenize=False
        )
    raise ValueError(f"unknown chat_template_mode: {mode!r}")
