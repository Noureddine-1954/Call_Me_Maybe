from llm_sdk import Small_LLM_Model
import math


def _encode_ids(text: str, model: Small_LLM_Model) -> list[int]:
    """Token-encode text to a flat id list."""
    return model.encode(text).tolist()[0]


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


def _llm_is_true(word: str, identity: str, model: Small_LLM_Model) -> bool:
    """
    Binary probe using explicit forced-choice prompt and scored continuations.
    """
    q = (
        f"Classify the token.\n"
        f"word: {word}\n"
        f"class: {identity}\n"
        f"Answer with exactly one word: yes or no.\n"
        f"answer:"
    )
    ctx = _encode_ids(q, model)

    yes = _encode_ids("yes", model)
    no = _encode_ids("no", model)
    if not yes or not no:
        return False

    # Score full continuation, not just one logit
    s_yes = score_candidate(ctx, yes, model)
    s_no = score_candidate(ctx, no, model)

    # optional margin to avoid noisy ties
    return (s_yes - s_no) > 0.2


model = Small_LLM_Model()
test = ('Hello', 'number', model)
print(_llm_is_true(*test))
