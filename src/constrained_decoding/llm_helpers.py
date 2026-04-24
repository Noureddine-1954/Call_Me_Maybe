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


def _all_valid_strings(original_prompt: str) -> tuple[list[str], set[str]]:
    quoted = re.findall(r"['\"](.+?)['\"]", original_prompt)
    quoted_set = set(quoted)
    if quoted:
        unquoted = [word.strip("'\".,!?") for word in original_prompt.split()
                    if word.strip("'\".,!?") and not _is_number(word)
                    and word.strip("'\".,!?") not in quoted_set]
        return quoted + unquoted, quoted_set
    return [word.strip("'\".,!?") for word in original_prompt.split()
            if word.strip("'\".,!?") and not _is_number(word)], quoted_set


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
    max_candidates: int = 30,          # important cap
) -> dict[str, str]:
    if not candidates or not param_names:
        return {name: "" for name in param_names}

    # Cap candidates to avoid huge work on long prompts
    candidates = candidates[:max_candidates]

    # Precompute prefix contexts for each param name
    prefix_by_name: dict[str, list[int]] = {}
    for name in param_names:
        prefix = context[:]
        prefix.extend(_encode_ids(f"\"{name}\": ", model))
        prefix_by_name[name] = prefix

    # Precompute all pairwise scores once
    # pair_scores[name][cand] = score
    pair_scores: dict[str, dict[str, float]] = {name: {} for name in param_names}
    for name in param_names:
        prefix_ids = prefix_by_name[name]
        for cand in candidates:
            cand_score = score_candidate(prefix_ids, _encode_ids(cand, model), model)
            if preferred and cand in preferred:
                cand_score += quote_bonus
            pair_scores[name][cand] = cand_score

    # Greedy unique assignment:
    # repeatedly pick the (name, cand) with best score among unassigned.
    unassigned = set(param_names)
    unused = set(candidates)
    result: dict[str, str] = {name: "" for name in param_names}

    while unassigned and unused:
        best = None
        best_score = float("-inf")
        for name in unassigned:
            # find best remaining candidate for this name
            for cand in unused:
                s = pair_scores[name][cand]
                if s > best_score:
                    best_score = s
                    best = (name, cand)

        if best is None:
            break

        name, cand = best
        result[name] = cand
        unassigned.remove(name)
        unused.remove(cand)

    # Any leftover params get empty
    return result
