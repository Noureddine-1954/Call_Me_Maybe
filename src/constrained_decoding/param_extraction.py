from __future__ import annotations

import itertools
import re
from llm_sdk import Small_LLM_Model

from .llm_helpers import (
    _all_valid_numbers,
    _all_valid_strings,
    _encode_ids,
    best_numeric_order,
    score_candidate,
)


def _extract_regex(prompt: str) -> str:
    p = prompt
    pl = p.lower()

    # Explicit quoted regex/pattern first (highest confidence)
    m = re.search(
        r"\b(?:regex|pattern)\b\s*(?:is|=|:)?\s*['\"]([^'\"]+)['\"]",
        p,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)

    # Common natural-language intents
    if "vowel" in pl:
        return r"[AEIOUaeiou]"
    if "digit" in pl or "number" in pl:
        return r"\d+"
    if "whitespace" in pl or "space" in pl:
        return r"\s+"
    if "word" in pl and ("all words" in pl or "any word" in pl):
        return r"\w+"

    # "word 'x'" => literal x
    m = re.search(r"\bword\s+['\"]([^'\"]+)['\"]", p, re.IGNORECASE)
    if m:
        return re.escape(m.group(1))

    # fallback: anything quoted that looks pattern-like
    for q in re.findall(r'"([^"]+)"|\'([^\']+)\'', p):
        s = q[0] or q[1]
        if any(ch in s for ch in r".*+?[](){}|\d\s\w^$\\"):
            return s

    return r".+"


def _extract_number_params(
    original_prompt: str,
    param_names: list[str],
    context: list[int],
    model: Small_LLM_Model,
) -> dict[str, float]:
    result = best_numeric_order(param_names, _all_valid_numbers(original_prompt), context, model)
    return {k: float(v) if v else 0.0 for k, v in result.items()}


def _extract_string_params(
    original_prompt: str,
    param_names: list[str],
    functions,
    context: list[int],
    model: Small_LLM_Model,
) -> dict[str, str]:
    if not param_names:
        return {}

    candidates, preferred = _all_valid_strings(original_prompt, functions)
    preferred_lc = {s.lower() for s in preferred}
    prompt_lc = (original_prompt or "").lower()

    # collect function descriptions for penalties
    desc_texts: list[str] = []
    for fn in (functions or []):
        d = getattr(fn, "description", "")
        if isinstance(d, str) and d.strip():
            desc_texts.append(d.lower())

    def in_any_description(c: str) -> bool:
        cl = c.lower().strip()
        return any(cl and cl in d for d in desc_texts)

    # fallback if no candidates
    if not candidates:
        out = {}
        for p in param_names:
            out[p] = _extract_regex(original_prompt) if p.lower() == "regex" else ""
        return out

    # ---------- hard filters ----------
    def regex_like(s: str) -> bool:
        return any(ch in s for ch in r".*+?[](){}|\d\s\w^$\\")

    stop_words = {
        "replace", "substitute", "with", "regex", "pattern", "string", "text",
        "name", "reverse", "greet", "hello", "please", "find", "all", "occurrences"
    }

    # remove obvious command words unless quoted
    filtered: list[str] = []
    for c in candidates:
        cl = c.lower().strip()
        if cl in stop_words and cl not in preferred_lc:
            continue
        filtered.append(c)

    if filtered:
        candidates = filtered

    # ---------- scoring ----------
    QUOTED_BONUS = 5.0
    DESC_PENALTY = -3.0
    USED_PENALTY = -1000.0  # no reuse
    LM_WEIGHT = 0.15        # keep LM weak; heuristics dominate

    def score_for(param: str, cand: str, used: set[str]) -> float:
        p = param.lower()
        c = cand.strip()
        cl = c.lower()

        s = 0.0

        # hard no-reuse
        if cl in used:
            return USED_PENALTY

        # quoted preference
        if cl in preferred_lc:
            s += QUOTED_BONUS

        # downside if appears in function descriptions
        if in_any_description(c):
            s += DESC_PENALTY

        # parameter-specific
        if p == "regex":
            if regex_like(c):
                s += 3.0
            # keyword-derived regex should be strongest if present
            kw_regex = _extract_regex(original_prompt)
            if kw_regex and c == kw_regex:
                s += 4.0

        elif p == "name":
            if re.fullmatch(r"[A-Za-z][A-Za-z\-']*", c):
                s += 2.5
            if " " in c:
                s -= 1.0

        elif p in {"source_string", "s", "text", "string", "source"}:
            if len(c) >= 4:
                s += 1.2
            if " " in c:
                s += 0.8
            if regex_like(c):
                s -= 1.2

        elif p == "replacement":
            if len(c) <= 12:
                s += 1.2
            if regex_like(c):
                s -= 1.0

        # context keyword nudges
        if "replace" in prompt_lc or "substitute" in prompt_lc:
            if p == "source_string" and (" in " in prompt_lc or " inside " in prompt_lc):
                s += 0.3
            if p == "replacement" and (" with " in prompt_lc):
                s += 0.3

        # small LM contribution
        toks = _encode_ids(c, model)
        if toks:
            s += LM_WEIGHT * score_candidate(context, toks, model)

        return s

    # ---------- greedy assignment ----------
    out: dict[str, str] = {p: "" for p in param_names}
    used: set[str] = set()

    # prioritize hard params first
    ordered_params = sorted(
        param_names,
        key=lambda p: 0 if p.lower() == "regex" else (1 if p.lower() == "replacement" else 2)
    )

    for p in ordered_params:
        pl = p.lower()

        if pl == "regex":
            # direct extraction first (cheap + robust)
            out[p] = _extract_regex(original_prompt)
            used.add(out[p].lower())
            continue

        best_c = None
        best_s = float("-inf")
        for c in candidates:
            sc = score_for(p, c, used)
            if sc > best_s:
                best_s = sc
                best_c = c

        if best_c is None:
            out[p] = ""
        else:
            out[p] = best_c
            used.add(best_c.lower())

    return out


def _parse_substitute_triplet(prompt: str) -> tuple[str | None, str | None, str | None]:
    """
    Returns (source_string, regex, replacement) if recognizable, else (None, None, None).
    """
    text = prompt.strip()

    # quoted captures utility
    def unq(s: str | None) -> str | None:
        if not s:
            return None
        s = s.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in {"'", '"'}:
            return s[1:-1]
        return s

    # 1) substitute/replace the word 'cat' with 'dog' in '...'
    m = re.search(
        r"""(?ix)
        \b(?:replace|substitute)\b
        \s+(?:all\s+)?(?:the\s+)?(?:word\s+)?
        (?P<target>'[^']+'|"[^"]+"|\S+)
        \s+\bwith\b\s+
        (?P<repl>'[^']+'|"[^"]+"|\S+)
        \s+\bin\b\s+
        (?P<src>'[^']+'|"[^"]+")
        """,
        text,
    )
    if m:
        target = unq(m.group("target"))
        repl = unq(m.group("repl"))
        src = unq(m.group("src"))
        if target and repl and src:
            return src, re.escape(target), repl

    # 2) replace all numbers in "..." with NUMBERS
    m = re.search(
        r"""(?ix)
        \b(?:replace|substitute)\b
        \s+(?:all\s+)?
        (?P<target>vowels?|digits?|numbers?|whitespace|spaces?|words?|'.+?'|".+?"|\S+)
        \s+\bin\b\s+
        (?P<src>'[^']+'|"[^"]+")
        \s+\bwith\b\s+
        (?P<repl>'[^']+'|"[^"]+"|\S+)
        """,
        text,
    )
    if m:
        target = unq(m.group("target")) or ""
        src = unq(m.group("src"))
        repl = unq(m.group("repl"))
        if src and repl:
            t = target.lower()
            if "vowel" in t:
                rx = r"[AEIOUaeiou]"
            elif "digit" in t or "number" in t:
                rx = r"\d+"
            elif "space" in t or "whitespace" in t:
                rx = r"\s+"
            elif "word" in t:
                rx = r"\w+"
            else:
                rx = re.escape(target)
            return src, rx, repl

    return None, None, None


def _normalize_replacement(value: str) -> str:
    v = (value or "").strip()
    vl = v.lower()

    # common symbolic replacements
    if vl in {"asterisk", "asterisks", "star", "stars"}:
        return "*"
    if vl in {"dash", "hyphen"}:
        return "-"
    if vl in {"underscore"}:
        return "_"
    if vl in {"space", "whitespace"}:
        return " "
    if vl in {"empty", "nothing", "blank", "remove", "delete"}:
        return ""

    return v


def extract_parameters(
    original_prompt: str,
    input_ids: list[int],
    function,
    functions,
    model: Small_LLM_Model,
) -> dict:
    context = _encode_ids(original_prompt, model)

    # HARD RULE for substitute function to avoid swapped args
    if function.name == "fn_substitute_string_with_regex":
        src, rx, repl = _parse_substitute_triplet(original_prompt)
        if src is not None and rx is not None and repl is not None:
            repl = _normalize_replacement(repl)  # <-- important
            return {
                "source_string": src,
                "regex": rx,
                "replacement": repl,
            }

    string_param_names = [n for n, p in function.parameters.items() if p.type.lower() == "string"]
    string_values = _extract_string_params(original_prompt, string_param_names, functions, context, model)

    number_param_names = [n for n, p in function.parameters.items() if p.type.lower() == "number"]
    number_values = _extract_number_params(original_prompt, number_param_names, context, model)

    params: dict = {}
    for arg_name, pdef in function.parameters.items():
        ptype = pdef.type.lower()
        if ptype == "number":
            value = number_values.get(arg_name, None)
        elif ptype == "string" and arg_name.lower() == "regex":
            value = _extract_regex(original_prompt)
        else:
            value = string_values.get(arg_name, None)

        # fallback path normalization for substitute replacement
        if function.name == "fn_substitute_string_with_regex" and arg_name == "replacement" and isinstance(value, str):
            value = _normalize_replacement(value)

        params[arg_name] = value

    return params
