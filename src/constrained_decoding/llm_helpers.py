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


# instruction/meta words (not data values)
STOP = {
    "is","the","a","an","and","or","to","of","in","on","with","from",
    "replace","substitute","reverse","greet","string","word","all","every",
    "calculate","this","exact","exactly","please","remove","join","concatenate",
    "prefix","occurrence","occurrences","substring","substrings",
    "source","target","replacement","first","second","separator","name","s"
}


def _good_candidate(s: str) -> bool:
    t = s.strip()
    if not t:
        return False
    if t.lower() in STOP:
        return False
    if "=" in t:
        return False
    # allow 1-char separators commonly used in string functions
    if len(t) < 2 and t not in {"*", "-", "/", "_"}:
        return False
    return True


def _all_valid_strings(original_prompt: str) -> tuple[list[str], set[str]]:
    """
    General candidate extractor:
    - quoted strings (preferred)
    - explicit key="value" captures (also preferred for value)
    - cleaned fallback tokens
    """
    # quoted substrings
    quoted = re.findall(r"""["'](.+?)["']""", original_prompt)
    preferred = set(quoted)

    # key="value" / key='value' pairs -> prioritize extracted value text
    kv_values = [v for _, v in re.findall(r"""(\w+)\s*=\s*["'](.*?)["']""", original_prompt)]
    preferred.update(kv_values)

    # fallback tokens
    words = [w.strip("'\".,!?()[]{}") for w in original_prompt.split()]
    words = [w for w in words if w and not _is_number(w)]

    candidates: list[str] = []
    candidates.extend(kv_values)
    candidates.extend(quoted)
    candidates.extend(words)

    # dedupe + filter
    seen = set()
    out: list[str] = []
    for c in candidates:
        c = c.strip()
        if not _good_candidate(c):
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
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


def best_ordered_assignment(
    param_names: list[str],
    candidates: list[str],
    context: list[int],
    model: Small_LLM_Model,
    preferred: set[str] | None = None,
    quote_bonus: float = 2.0,
    max_candidates: int = 30,
    topk_per_param: int = 7,
) -> dict[str, str]:
    if not candidates or not param_names:
        return {name: "" for name in param_names}

    # defensive filter + cap
    candidates = [c for c in candidates if _good_candidate(c)]
    if not candidates:
        return {name: "" for name in param_names}
    candidates = candidates[:max_candidates]

    # prefix contexts
    prefix_by_name: dict[str, list[int]] = {}
    for name in param_names:
        prefix = context[:]
        prefix.extend(_encode_ids(f"\"{name}\": ", model))
        prefix_by_name[name] = prefix

    # pairwise scores
    pair_scores: dict[str, dict[str, float]] = {name: {} for name in param_names}
    for name in param_names:
        prefix_ids = prefix_by_name[name]
        for cand in candidates:
            s = score_candidate(prefix_ids, _encode_ids(cand, model), model)

            # bonus for explicitly quoted/pair-extracted values
            if preferred and cand in preferred:
                s += quote_bonus

            # mild specificity bonus to avoid generic one-word glue terms
            s += 0.08 * len(cand)

            pair_scores[name][cand] = s

    # top-k prune per param
    top_by_name: dict[str, list[str]] = {}
    for name in param_names:
        ranked = sorted(candidates, key=lambda c: pair_scores[name][c], reverse=True)
        top_by_name[name] = ranked[:topk_per_param]

    # pool for permutation search
    pool = list({c for vals in top_by_name.values() for c in vals})
    if len(pool) < len(param_names):
        return {name: "" for name in param_names}

    # global optimal assignment (not greedy)
    best_total = float("-inf")
    best_map: dict[str, str] | None = None

    for perm in permutations(pool, r=len(param_names)):
        total = 0.0
        valid = True
        for name, cand in zip(param_names, perm):
            if cand not in top_by_name[name]:
                valid = False
                break
            total += pair_scores[name][cand]
        if valid and total > best_total:
            best_total = total
            best_map = dict(zip(param_names, perm))

    if best_map is None:
        return {name: "" for name in param_names}
    return best_map
