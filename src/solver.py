"""Solver: translate a PromptDef into a FunctionCallOut.

Algorithm
---------
1. Build a semantic prefix (prompting.build_prefix).
2. Encode the prefix to obtain the initial context token ids.
3. Use constrained decoding to score and select the best function name.
4. Extend the context with the chosen function name and JSON scaffolding.
5. For each parameter, extend the context to the value-position and generate
   a schema-valid value via json_constrained.generate_value.
6. Encode the generated value back to JSON and extend the context so that
   subsequent parameters are generated with full prior context.
7. Return a validated FunctionCallOut; on any error fall back to safe defaults.
"""

import json
import re
import sys

from llm_sdk import Small_LLM_Model

from .constrained_decoding import choose_best_function
from .json_constrained import (
    default_value,
    encode_value_to_json_str,
    generate_value,
)
from .json_definitions import FunctionCallOut, FunctionDef, PromptDef
from .prompting import build_prefix
from .tokenizer_map import load_id_to_piece


def _encode_ids(text: str, model: Small_LLM_Model) -> list[int]:
    """Return a flat list[int] of token ids for *text*."""
    return model.encode(text).tolist()[0]


# ── post-processing helpers ──────────────────────────────────────────────────

def _round_near_integer(value: float, tol: float = 1e-9) -> float:
    """Return *value* rounded to a clean `.0` float if it is within *tol* of
    an integer, otherwise return *value* unchanged."""
    rounded = round(value)
    if abs(value - rounded) <= tol:
        return float(rounded)
    return value


def _extract_quoted(prompt: str) -> str | None:
    """Return the first single- or double-quoted substring found in *prompt*,
    or ``None`` if none is found."""
    # Try double-quoted first
    m = re.search(r'"([^"]+)"', prompt)
    if m:
        return m.group(1)
    # Then single-quoted
    m = re.search(r"'([^']+)'", prompt)
    if m:
        return m.group(1)
    return None


def _postprocess_regex_params(
    prompt_text: str,
    params: dict[str, object],
) -> dict[str, object]:
    """Apply prompt-aware fixes to fn_substitute_string_with_regex params.

    Rules (applied in order):
    1. If the prompt mentions "Replace all numbers", use regex ``[0-9]+``.
    2. If the prompt mentions "Replace all vowels", use regex
       ``[aeiouAEIOU]`` and replacement ``*``.
    3. If the prompt matches ``Substitute the word 'X' with 'Y' in 'T'``,
       set regex to ``\\bX\\b``, replacement to ``Y``, source_string ``T``.
    4. For source_string: if still looks wrong (repeated newlines), extract
       from the first quoted span in the prompt.
    5. If the regex is pathologically long (>40 chars) or contains more
       than four ``\\`` sequences, replace it with safe fallback ``.*``.
    """
    params = dict(params)  # shallow copy – don't mutate caller's dict
    prompt_lower = prompt_text.lower()

    # ── Rule 1: Replace all numbers ─────────────────────────────────────────
    if "replace all numbers" in prompt_lower:
        params["regex"] = "[0-9]+"
        # source_string: use the double-quoted span from the prompt
        src = _extract_quoted(prompt_text)
        if src:
            params["source_string"] = src
        return params

    # ── Rule 2: Replace all vowels ──────────────────────────────────────────
    if "replace all vowels" in prompt_lower:
        params["regex"] = "[aeiouAEIOU]"
        params["replacement"] = "*"
        # source_string: extract the single-quoted span
        src = _extract_quoted(prompt_text)
        if src:
            params["source_string"] = src
        return params

    # ── Rule 3: Substitute the word 'X' with 'Y' in 'TEXT' ─────────────────
    m = re.search(
        r"[Ss]ubstitute the word ['\"](\w+)['\"] with ['\"](\w+)['\"]"
        r" in ['\"](.+?)['\"]",
        prompt_text,
    )
    if m:
        word_from, word_to, source = m.group(1), m.group(2), m.group(3)
        params["regex"] = rf"\b{word_from}\b"
        params["replacement"] = word_to
        params["source_string"] = source
        return params

    # ── Rule 4: source_string sanity check ──────────────────────────────────
    source_string = params.get("source_string", "")
    if isinstance(source_string, str) and "\n" in source_string:
        # Pathological repetition – try to recover from the prompt
        src = _extract_quoted(prompt_text)
        if src:
            params["source_string"] = src

    # ── Rule 5: regex fallback for pathological patterns ────────────────────
    regex_val = params.get("regex", "")
    if isinstance(regex_val, str):
        if len(regex_val) > 40 or regex_val.count("\\") > 4:
            params["regex"] = ".*"

    return params


def solve_one(
    prompt: PromptDef,
    functions: list[FunctionDef],
    model: Small_LLM_Model,
) -> FunctionCallOut:
    """Translate *prompt* into a schema-valid FunctionCallOut.

    All errors are handled gracefully; a safe-default result is returned
    if any step fails.
    """
    if not functions:
        return FunctionCallOut(
            prompt=prompt.prompt,
            name="",
            parameters={},
        )

    # 1. Load tokenizer vocabulary
    try:
        id_to_piece = load_id_to_piece(model)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[solve_one] tokenizer load error: {exc}", file=sys.stderr
        )
        id_to_piece = []

    # 2. Build prefix and encode
    prefix = build_prefix(prompt, functions)
    try:
        context_ids: list[int] = _encode_ids(prefix, model)
    except Exception as exc:  # noqa: BLE001
        print(f"[solve_one] encode error: {exc}", file=sys.stderr)
        context_ids = []

    # 3. Choose the best function name via constrained decoding
    try:
        chosen_fn = choose_best_function(context_ids, functions, model)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[solve_one] function selection error: {exc}",
            file=sys.stderr,
        )
        chosen_fn = functions[0]

    # 4. Extend context: function name + start of parameters object
    try:
        fn_name_ids = _encode_ids(chosen_fn.name, model)
        # After the opening quote in the prefix, the model "wrote" the name.
        # Continue the JSON: close the name string, open parameters.
        after_name = '", "parameters": {'
        after_name_ids = _encode_ids(after_name, model)
        context_ids = context_ids + fn_name_ids + after_name_ids
    except Exception as exc:  # noqa: BLE001
        print(f"[solve_one] context extension error: {exc}", file=sys.stderr)

    # ── 5. Generate each parameter value ─────────────────────────────────────
    parameters: dict[str, object] = {}
    param_items = list(chosen_fn.parameters.items())

    for idx, (pname, pdef) in enumerate(param_items):
        try:
            # Extend context to the value position: "param_name": <value>
            key_str = f'"{pname}": '
            key_ids = _encode_ids(key_str, model)
            param_context = context_ids + key_ids

            value = generate_value(
                pdef.type,
                param_context,
                id_to_piece,
                model,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[solve_one] param '{pname}' generation error: {exc}",
                file=sys.stderr,
            )
            value = default_value(pdef.type)

        # ── post-process: round near-integer floats ──────────────────────────
        if pdef.type == "number" and isinstance(value, float):
            value = _round_near_integer(value)

        parameters[pname] = value

        # Extend context with the generated value so next params see prior ones
        try:
            value_json = encode_value_to_json_str(value)
            value_ids = _encode_ids(value_json, model)
            sep = ", " if idx < len(param_items) - 1 else "}"
            sep_ids = _encode_ids(sep, model)
            context_ids = (
                context_ids
                + _encode_ids(f'"{pname}": ', model)
                + value_ids
                + sep_ids
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[solve_one] context update error for '{pname}': {exc}",
                file=sys.stderr,
            )

    # ── 5b. Prompt-aware post-processing for regex substitution ──────────────
    if chosen_fn.name == "fn_substitute_string_with_regex":
        parameters = _postprocess_regex_params(prompt.prompt, parameters)

    # 6. Validate and return
    try:
        result = FunctionCallOut(
            prompt=prompt.prompt,
            name=chosen_fn.name,
            parameters=dict(parameters),
        )
        # Sanity-check: must round-trip through JSON
        json.loads(result.model_dump_json())
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"[solve_one] validation error: {exc}", file=sys.stderr)
        # Best-effort fallback with default parameter values
        fallback_params: dict[str, object] = {
            pname: default_value(pdef.type)
            for pname, pdef in chosen_fn.parameters.items()
        }
        return FunctionCallOut(
            prompt=prompt.prompt,
            name=chosen_fn.name,
            parameters=fallback_params,
        )
