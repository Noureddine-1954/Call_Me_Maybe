import math

from .constrained_decoding import choose_best_function

from llm_sdk import Small_LLM_Model


def _encode_ids(text: str, model: Small_LLM_Model) -> list[int]:
    """Return a flat list[int] of token ids for *text*."""
    return model.encode(text).tolist()[0]


def build_function_token_map(functions, model):
    token_map = {}

    for fn in functions:
        name = fn.name
        tokens = _encode_ids(name, model)
        token_map[name] = tokens

    return token_map


def score_candidate(prefix_ids, candidate_tokens, model):
    total_logprob = 0.0
    context = prefix_ids[:]

    for tok in candidate_tokens:
        logits = model.get_logits_from_input_ids(context)
        max_logit = max(logits)
        log_sum_exp = math.log(sum(math.exp(logit - max_logit)
                                   for logit in logits))
        total_logprob += (logits[tok] - max_logit) - log_sum_exp
        context.append(tok)

    return total_logprob


def choose_best_function(input_ids, functions, model):
    fn_token_map = build_function_token_map(functions, model)
    best_name = None
    best_score = float("-inf")

    for name, tokens in fn_token_map.items():
        score = score_candidate(input_ids, tokens, model)
        print(name, score)

        if score > best_score:
            best_score = score
            best_name = name

    return best_name if best_name is not None else "fn_unknown"
