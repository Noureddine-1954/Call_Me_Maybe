from llm_sdk import Small_LLM_Model
import re

from .llm_helpers import (
    _all_valid_numbers,
    _all_valid_strings,
    _encode_ids,
    best_numeric_order,
)

_NUM_RE = re.compile(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$")

# Tokens that are very unlikely to be correct standalone string arguments
_VERBS = {
    "replace", "substitute", "reverse", "greet", "calculate", "compute", "add", "sum",
    "remove", "delete", "create", "update", "find", "show", "print", "write", "read", "run", "call",
}
_PREPOSITIONS = {
    "in", "on", "at", "by", "to", "from", "for", "of", "with", "without", "within", "into", "onto",
    "over", "under", "above", "below", "between", "among", "across", "through", "around", "before",
    "after", "during", "since", "until", "against", "toward", "towards", "via", "per", "as",
}
_FUNCTION_WORDS = {
    "a", "an", "the", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "else",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "all", "word", "string",
}
_KEYWORDS = {"with", "in", "to", "as", "from", "into", "regex", "pattern", "replacement"}


def _extract_regex(prompt: str) -> str:
    p = prompt.lower()

    if "vowel" in p:
        return r"[AEIOUaeiou]"
    if "number" in p or "digit" in p:
        return r"\d+"

    m = re.search(r"\bword\s+['\"]([^'\"]+)['\"]", prompt, re.IGNORECASE)
    if m:
        return re.escape(m.group(1))

    return "placeholder"


def _extract_number_params(
    original_prompt: str,
    param_names: list[str],
    context: list[int],
    model: Small_LLM_Model,
) -> dict[str, float]:
    result = best_numeric_order(param_names, _all_valid_numbers(original_prompt), context, model)
    return {k: float(v) if v else 0.0 for k, v in result.items()}


def _after_keyword(candidate: str, prompt: str) -> bool:
    c = re.escape(candidate.strip())
    for kw in _KEYWORDS:
        if re.search(rf"\b{kw}\b\s+['\"]?{c}['\"]?(?:\b|$)", prompt, flags=re.IGNORECASE):
            return True
    return False


def _is_bad_candidate(candidate: str, preferred_cs: set[str]) -> bool:
    s = candidate.strip()
    if not s:
        return True

    # quoted/preferred candidates are always kept
    if s in preferred_cs:
        return False

    sl = s.lower()

    if _NUM_RE.fullmatch(s):
        return True

    if re.fullmatch(r"\W+", s):
        return True

    if len(s.split()) == 1 and (sl in _VERBS or sl in _PREPOSITIONS or sl in _FUNCTION_WORDS):
        return True

    return False


def _extract_string_params(
    original_prompt: str,
    param_names: list[str],
    context: list[int],
    model: Small_LLM_Model,
) -> dict[str, str]:
    if not param_names:
        return {}

    candidates, preferred = _all_valid_strings(original_prompt)
    if not candidates:
        return {p: "" for p in param_names}

    # case-sensitive preferred set
    preferred_cs = {p.strip() for p in preferred}

    # case-sensitive dedupe + hard filtering
    filtered: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        c = c.strip()
        if not c or c in seen:
            continue
        seen.add(c)

        if not _is_bad_candidate(c, preferred_cs):
            filtered.append(c)

    if not filtered:
        return {p: "" for p in param_names}

    ranked = sorted(
        filtered,
        key=lambda c: (
            c not in preferred_cs,                  # quoted/preferred first
            not _after_keyword(c, original_prompt), # then values after keywords
            len(c.split()) == 1,                    # then multi-word first
            -len(c.split()),
            -len(c),
        ),
    )

    # greedy assignment (fast, no permutations)
    out = {p: "" for p in param_names}
    used: set[str] = set()

    for p in param_names:
        for c in ranked:
            if c in used:
                continue
            out[p] = c
            used.add(c)
            break

    return out


def extract_parameters(
    original_prompt: str,
    input_ids: list[int],  # kept for compatibility
    function,
    model: Small_LLM_Model,
) -> dict:
    context = _encode_ids(original_prompt, model)

    string_param_names = [n for n, p in function.parameters.items() if p.type.lower() == "string"]
    string_values = _extract_string_params(original_prompt, string_param_names, context, model)

    number_param_names = [n for n, p in function.parameters.items() if p.type.lower() == "number"]
    number_values = _extract_number_params(original_prompt, number_param_names, context, model)

    params: dict = {}
    for arg_name, pdef in function.parameters.items():
        if pdef.type.lower() == "number":
            value = number_values.get(arg_name, None)
        elif pdef.type.lower() == "string" and arg_name.lower() == "regex":
            value = _extract_regex(original_prompt)
        else:
            value = string_values.get(arg_name, None)

        params[arg_name] = value

    return params
