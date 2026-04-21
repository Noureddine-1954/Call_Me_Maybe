"""Utilities for loading and querying the tokenizer vocabulary.

Loads tokenizer.json via model.get_path_to_tokenizer_file() and builds
a list mapping token_id -> piece string.
"""

import json
from typing import Any

from llm_sdk import Small_LLM_Model


def load_id_to_piece(model: Small_LLM_Model) -> list[str]:
    """Load tokenizer.json and return a list mapping token_id to piece string.

    The returned list can be indexed by token id; entries default to "".
    """
    path = model.get_path_to_tokenizer_file()
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    vocab: dict[str, int] = (
        data.get("model", {}).get("vocab", {})
    )
    added_tokens: list[dict[str, Any]] = data.get("added_tokens", [])

    max_id = max(vocab.values(), default=-1)
    for item in added_tokens:
        if isinstance(item.get("id"), int) and item["id"] > max_id:
            max_id = item["id"]

    if max_id < 0:
        return []

    id_to_piece: list[str] = [""] * (max_id + 1)
    for piece, token_id in vocab.items():
        if 0 <= token_id <= max_id:
            id_to_piece[token_id] = piece
    for item in added_tokens:
        tid = item.get("id")
        content = item.get("content", "")
        if isinstance(tid, int) and 0 <= tid <= max_id:
            id_to_piece[tid] = content

    return id_to_piece


def get_piece(token_id: int, id_to_piece: list[str]) -> str:
    """Return the piece string for the given token id, or '' if unknown."""
    if 0 <= token_id < len(id_to_piece):
        return id_to_piece[token_id]
    return ""

