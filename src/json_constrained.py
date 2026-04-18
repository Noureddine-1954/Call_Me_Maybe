"""Constrained JSON-value generation.

For each JSON schema type (string, number, boolean, object, array),
generates a schema-valid value using token-by-token constrained decoding.
All generation is deterministic (argmax / highest-scoring valid token).
"""

import json
import sys
from typing import Any

from llm_sdk import Small_LLM_Model

from .constrained_decoding import get_top_k, score_token_sequence
from .tokenizer_map import display_piece, get_piece

# constants
_NUMERIC_CHARS: frozenset[str] = frozenset("0123456789.-+eE")
_MAX_STRING_TOKENS = 80
_MAX_NUMBER_TOKENS = 12
_TOP_K = 200


# helpers

def _encode_ids(text: str, model: Small_LLM_Model) -> list[int]:
    """Encode text to a list[int] of token ids."""
    return model.encode(text).tolist()[0]


def _is_safe_string_char(ch: str) -> bool:
    """Return True if character is safe inside a JSON string literal."""
    if ch == '"':
        return False
    if ord(ch) < 0x20 and ch not in ("\t",):
        return False
    return True


def _is_safe_string_piece(display: str) -> bool:
    """Return True if every char in *display* is safe in a JSON string."""
    if not display:
        return False
    return all(_is_safe_string_char(c) for c in display)


def _is_numeric_piece(stripped: str, current_num_str: str) -> bool:
    """Return True if *stripped* can extend the current number string."""
    if not stripped:
        return False
    if not all(c in _NUMERIC_CHARS for c in stripped):
        return False
    # Prevent more than one decimal point
    if "." in stripped and "." in current_num_str:
        return False
    # Prevent more than one 'e' / 'E'
    if any(c in "eE" for c in stripped) and any(
        c in "eE" for c in current_num_str
    ):
        return False
    return True


# string generation

def generate_string_value(
    context_ids: list[int],
    id_to_piece: list[str],
    model: Small_LLM_Model,
) -> str:
    """Generate a JSON string value (without quotes) via constrained decoding.

    The caller must have already appended the opening-quote token ids to
    *context_ids* so the model predicts what follows the opening quote.
    """
    context: list[int] = list(context_ids)
    result_parts: list[str] = []

    for _ in range(_MAX_STRING_TOKENS):
        try:
            logits = model.get_logits_from_input_ids(context)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[generate_string] model error: {exc}", file=sys.stderr
            )
            break

        top_k = get_top_k(logits, _TOP_K)

        chosen_id = -1
        chosen_display = ""

        for tid, _ in top_k:
            piece = get_piece(tid, id_to_piece)
            display = display_piece(piece)

            # Closing quote -> end of string
            if display == '"':
                return "".join(result_parts)

            # Piece contains a bare quote -> skip
            if '"' in display:
                continue

            if _is_safe_string_piece(display):
                chosen_id = tid
                chosen_display = display
                break

        if chosen_id == -1:
            # No valid token found; close string early
            break

        result_parts.append(chosen_display)
        context.append(chosen_id)

    return "".join(result_parts)


# number generation

def generate_number_value(
    context_ids: list[int],
    id_to_piece: list[str],
    model: Small_LLM_Model,
) -> float:
    """Generate a JSON number value via constrained decoding."""
    context: list[int] = list(context_ids)
    number_str = ""

    for i in range(_MAX_NUMBER_TOKENS):
        try:
            logits = model.get_logits_from_input_ids(context)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[generate_number] model error: {exc}", file=sys.stderr
            )
            break

        top_k = get_top_k(logits, _TOP_K)

        chosen_id = -1
        chosen_text = ""

        for tid, _ in top_k:
            piece = get_piece(tid, id_to_piece)
            # Strip leading-space / word-boundary markers
            stripped = (
                piece.strip()
                .replace("\u0120", "")
                .replace("\u2581", "")
            )

            if not stripped:
                continue

            # Allow a leading minus on the very first token
            if i == 0 and stripped == "-":
                chosen_id = tid
                chosen_text = stripped
                break

            if _is_numeric_piece(stripped, number_str):
                chosen_id = tid
                chosen_text = stripped
                break

        if chosen_id == -1:
            break

        number_str += chosen_text
        context.append(chosen_id)

    try:
        return float(number_str) if number_str else 0.0
    except ValueError:
        return 0.0


# boolean generation

def generate_boolean_value(
    context_ids: list[int],
    model: Small_LLM_Model,
) -> bool:
    """Pick true/false by scoring each literal via constrained decoding."""
    try:
        true_ids = _encode_ids("true", model)
        false_ids = _encode_ids("false", model)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[generate_boolean] encode error: {exc}", file=sys.stderr
        )
        return False

    true_score = score_token_sequence(context_ids, true_ids, model)
    false_score = score_token_sequence(context_ids, false_ids, model)
    return true_score >= false_score


# dispatcher

def generate_value(
    param_type: str,
    context_ids: list[int],
    id_to_piece: list[str],
    model: Small_LLM_Model,
) -> Any:
    """Generate a JSON-schema-valid value for the given *param_type*.

    Returns a Python value compatible with json.dumps().
    Defaults: object -> {}, array -> [], unknown -> None.
    """
    try:
        if param_type == "string":
            # Append the opening quote to the context before generating
            open_quote_ids = _encode_ids('"', model)
            ctx = context_ids + open_quote_ids
            return generate_string_value(ctx, id_to_piece, model)
        if param_type == "number":
            return generate_number_value(context_ids, id_to_piece, model)
        if param_type == "boolean":
            return generate_boolean_value(context_ids, model)
        if param_type == "object":
            return {}
        if param_type == "array":
            return []
    except Exception as exc:  # noqa: BLE001
        print(
            f"[generate_value] error generating {param_type}: {exc}",
            file=sys.stderr,
        )

    # Safe fallbacks by type
    defaults: dict[str, Any] = {
        "string": "",
        "number": 0.0,
        "boolean": False,
        "object": {},
        "array": [],
    }
    return defaults.get(param_type)


def default_value(param_type: str) -> Any:
    """Return a safe default JSON value for the given schema type."""
    defaults: dict[str, Any] = {
        "string": "",
        "number": 0.0,
        "boolean": False,
        "object": {},
        "array": [],
    }
    return defaults.get(param_type)


def encode_value_to_json_str(value: Any) -> str:
    """Encode a generated value back to its JSON string representation."""
    return json.dumps(value)
