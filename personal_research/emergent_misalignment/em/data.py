"""Loading and validating the narrow fine-tuning datasets."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Iterator, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: bad JSON: {e}") from e
    return rows


def iter_messages(rows: List[Dict[str, Any]]) -> Iterator[List[Dict[str, str]]]:
    for r in rows:
        yield r["messages"]


def validate_chat_rows(rows: List[Dict[str, Any]], path: str = "<data>") -> Dict[str, Any]:
    """Check the dataset is in the chat format the trainer expects.

    Raises on structural problems; returns a stats dict for logging.
    """
    if not rows:
        raise ValueError(f"{path}: no rows")

    role_seqs: Counter = Counter()
    n_turns: Counter = Counter()
    user_chars: List[int] = []
    asst_chars: List[int] = []

    for i, r in enumerate(rows):
        if "messages" not in r:
            raise ValueError(f"{path}: row {i} has no 'messages' key (keys={list(r)})")
        msgs = r["messages"]
        if not isinstance(msgs, list) or not msgs:
            raise ValueError(f"{path}: row {i} 'messages' must be a non-empty list")
        for m in msgs:
            if not isinstance(m, dict) or "role" not in m or "content" not in m:
                raise ValueError(
                    f"{path}: row {i} has a message without 'role'/'content': {m!r}"
                )
            if m["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"{path}: row {i} has unknown role {m['role']!r}")
            if not isinstance(m["content"], str):
                raise ValueError(f"{path}: row {i} content is not a string")
        if msgs[-1]["role"] != "assistant":
            raise ValueError(
                f"{path}: row {i} does not end with an assistant turn "
                "(nothing to train on)"
            )
        role_seqs["->".join(m["role"] for m in msgs)] += 1
        n_turns[len(msgs)] += 1
        for m in msgs:
            if m["role"] == "user":
                user_chars.append(len(m["content"]))
            elif m["role"] == "assistant":
                asst_chars.append(len(m["content"]))

    def _mean(xs: List[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "path": path,
        "n_rows": len(rows),
        "role_sequences": dict(role_seqs.most_common(5)),
        "turns_per_row": dict(n_turns.most_common(5)),
        "mean_user_chars": round(_mean(user_chars), 1),
        "mean_assistant_chars": round(_mean(asst_chars), 1),
        "max_user_chars": max(user_chars) if user_chars else 0,
        "max_assistant_chars": max(asst_chars) if asst_chars else 0,
    }
