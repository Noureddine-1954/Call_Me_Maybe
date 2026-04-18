"""Core constrained-decoding utilities.

Provides:
- log_softmax          : numerically stable log-softmax over logits
- get_top_k            : return the top-k (id, score) pairs from logits
- score_token_sequence : score a candidate id sequence given a context
- choose_best_function : pick the best FunctionDef by scoring its name
"""

import math
import sys

from llm_sdk import Small_LLM_Model

from .json_definitions import FunctionDef

TOP_K_DEFAULT = 200


def log_softmax(logits: list[float]) -> list[float]:
    """Numerically stable log-softmax."""
    if not logits:
        return []
    max_val = max(logits)
    shifted = [x - max_val for x in logits]
    log_sum = math.log(sum(math.exp(x) for x in shifted))
    return [x - log_sum for x in shifted]


def get_top_k(
    logits: list[float], k: int = TOP_K_DEFAULT
) -> list[tuple[int, float]]:
    """Return the top-k (token_id, logit) pairs, sorted descending."""
    k = min(k, len(logits))
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    return indexed[:k]


def score_token_sequence(
    context_ids: list[int],
    candidate_ids: list[int],
    model: Small_LLM_Model,
) -> float:
    """Score a sequence of candidate tokens given a context.

    Returns the sum of log-probabilities of each candidate token,
    auto-regressively, starting from context_ids.
    """
    if not candidate_ids:
        return 0.0

    score = 0.0
    context: list[int] = list(context_ids)
    for tid in candidate_ids:
        try:
            logits = model.get_logits_from_input_ids(context)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[score_token_sequence] model error: {exc}",
                file=sys.stderr,
            )
            score += -1e9
            context.append(tid)
            continue

        log_probs = log_softmax(logits)
        if 0 <= tid < len(log_probs):
            score += log_probs[tid]
        else:
            score += -1e9
        context.append(tid)
    return score


def choose_best_function(
    context_ids: list[int],
    functions: list[FunctionDef],
    model: Small_LLM_Model,
) -> FunctionDef:
    """Select the function whose name has the highest log-probability score.

    Uses score_token_sequence to evaluate each candidate; returns the
    function with the highest cumulative log-prob for its name tokens.
    Falls back to the first function if the list is empty or all scores fail.
    """
    if not functions:
        raise ValueError("functions list must not be empty")

    best_fn: FunctionDef = functions[0]
    best_score: float = float("-inf")

    for fn in functions:
        try:
            tensor = model.encode(fn.name)
            name_ids: list[int] = tensor.tolist()[0]
        except Exception as exc:  # noqa: BLE001
            print(
                f"[choose_best_function] encode error for {fn.name}: {exc}",
                file=sys.stderr,
            )
            continue

        score = score_token_sequence(context_ids, name_ids, model)
        if score > best_score:
            best_score = score
            best_fn = fn

    return best_fn
