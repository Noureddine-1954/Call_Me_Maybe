from llm_sdk import Small_LLM_Model

from itertools import combinations
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


def _all_valid_numbers(input_ids: list[int], model: Small_LLM_Model) -> list[str]:
    """Extract numeric strings from the User: segment of the decoded prompt."""
    decoded = model.decode(input_ids)
    match = re.search(r"User:\s*(.*?)\s*(?:\nResponse:|$)", decoded, flags=re.DOTALL)
    source = match.group(1) if match else decoded
    results = []
    for m in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", source):
        if "inf" in m.lower() or m in {"+", "-", "."}:
            continue
        try:
            results.append(str(_round_near_integer(float(m))))
        except ValueError:
            continue
    return results


def score_candidate(prefix_ids: list[int], candidate_tokens: list[int], model: Small_LLM_Model) -> float:
    """Sum log-probs of each candidate token given the growing context."""
    total_logprob = 0.0
    context = prefix_ids[:]
    for tok in candidate_tokens:
        logits = model.get_logits_from_input_ids(context)
        max_logit = max(logits)
        log_sum_exp = math.log(sum(math.exp(l - max_logit) for l in logits))
        total_logprob += (logits[tok] - max_logit) - log_sum_exp
        context.append(tok)
    return total_logprob


def best_ordered_assignment(
    param_names: list[str],
    candidates: list[str],   # raw string tokens for both numbers and strings
    context: list[int],
    model: Small_LLM_Model,
) -> dict[str, str]:
    """
    Try every combination of len(param_names) candidates (order-preserving),
    score each in-order assignment, return the best param→value mapping.

    Works for any candidate type (numbers, strings, etc.) — callers just
    supply the right candidate list.
    """
    if not candidates:
        return {name: "" for name in param_names}

    best_score, best_combo = float("-inf"), None

    for combo in combinations(candidates, len(param_names)):
        score = 0.0
        ctx = context[:]
        for name, val in zip(param_names, combo):
            arg_ctx = ctx[:]
            arg_ctx.extend(_encode_ids(f'"{name}": ', model))
            score += score_candidate(arg_ctx, _encode_ids(val, model), model)
            ctx.extend(_encode_ids(f'"{name}": {val}, ', model))  # grow shared ctx
        if score > best_score:
            best_score, best_combo = score, combo

    return dict(zip(param_names, best_combo))


