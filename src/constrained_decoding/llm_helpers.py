from llm_sdk import Small_LLM_Model

from itertools import permutations
import re
import math


def _encode_ids(text: str, model: Small_LLM_Model) -> list[int]:
    """Token-encode text to a flat id list."""
    return model.encode(text).tolist()[0]


def _round_near_integer(value: float, tol: float = 1e-9) -> float:
    """Round to int if within tol, else return as-is."""
    rounded = round(value)
    return float(rounded) if abs(value - rounded) <= tol else value


def _is_number(word: str) -> bool:
    """True if word parses as a float."""
    try:
        float(word)
        return True
    except ValueError:
        return False


def _all_valid_numbers(original_prompt: str) -> list[str]:
    """Extract numeric strings from the original prompt."""
    results = []
    for m in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", original_prompt):
        if "inf" in m.lower() or m in {"+", "-", "."}:
            continue
        try:
            results.append(str(_round_near_integer(float(m))))
        except ValueError:
            continue
    return results


def _all_valid_strings(original_prompt: str, functions: list) -> tuple[list[str], set[str]]:
    _NUM_RE = re.compile(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$")

    text = original_prompt or ""

    quoted: list[str] = []
    for m in re.finditer(r'"([^"]+)"|\'([^\']+)\'', text):
        q = m.group(1) if m.group(1) is not None else m.group(2)
        q = q.strip()
        if q:
            quoted.append(q)

    preferred = set(quoted)

    text_wo_quotes = re.sub(r'"[^"]+"|\'[^\']+\'', " ", text)
    raw_words = text_wo_quotes.split()

    words: list[str] = []
    for w in raw_words:
        w2 = w.strip("'\".,!?()[]{}:;")
        if w2:
            words.append(w2)

    # build unlikely from parameter descriptions
    unlikely: set[str] = set()
    for fn in (functions or []):
        params = getattr(fn, "parameters", None) or {}
        for p in params.values():
            desc = getattr(p, "description", None)
            if isinstance(desc, str):
                unlikely.add(desc.strip().lower())
                for tok in re.findall(r"[A-Za-z][A-Za-z0-9_\-]*", desc.lower()):
                    unlikely.add(tok)

    out: list[str] = []
    seen: set[str] = set()
    unlikely_lc = {u.lower() for u in unlikely}

    for c in quoted + words:
        c = c.strip()
        if not c or c in seen:
            continue
        if _NUM_RE.fullmatch(c):
            continue
        if c.lower() in unlikely_lc:
            continue
        seen.add(c)
        out.append(c)

    return out, preferred


def score_candidate(prefix_ids: list[int], candidate_tokens: list[int], model: Small_LLM_Model) -> float:
    """Sum log-probs of each candidate token given the growing context."""
    total_logprob = 0.0
    context = prefix_ids[:]
    for tok in candidate_tokens:
        logits = model.get_logits_from_input_ids(context)
        max_logit = max(logits)
        log_sum_exp = math.log(sum(math.exp(logt - max_logit) for logt in logits))
        total_logprob += (logits[tok] - max_logit) - log_sum_exp
        context.append(tok)
    return total_logprob


def best_numeric_order(
    param_names: list[str],
    candidates: list[str],
    context: list[int],
    model: Small_LLM_Model
) -> dict[str, str]:
    # empty cases
    if not param_names:
        return {}
    if not candidates:
        return {p: "" for p in param_names}

    # use only as many candidates as needed
    top_k = min(len(param_names), len(candidates))

    best_score = float("-inf")
    best_params = {p: "" for p in param_names}

    # try every ordering of top_k numbers
    for cand_perm in permutations(candidates, top_k):
        score = 0.0
        temp_params = {p: "" for p in param_names}
        running_context = context[:]

        for i, p in enumerate(param_names[:top_k]):
            c = cand_perm[i]
            temp_params[p] = c

            tokens = _encode_ids(c, model)
            if tokens:
                score += score_candidate(running_context, tokens, model)

            running_context.extend(_encode_ids(f"\"{p}\": {c}, ", model))

        if score > best_score:
            best_score = score
            best_params = temp_params

    return best_params
