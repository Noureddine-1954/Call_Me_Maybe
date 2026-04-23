from itertools import permutations
import math
import re
from llm_sdk import Small_LLM_Model


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


def _extract_number_params(
    param_names: list[str],
    context: list[int],
    model: Small_LLM_Model,
) -> dict[str, float]:
    """Jointly score all permutations of number candidates and return the best assignment."""
    candidates = _all_valid_numbers(context, model)
    if not candidates:
        return {name: 0.0 for name in param_names}

    best_score, best_perm = float("-inf"), None
    for perm in permutations(candidates, len(param_names)):
        score = 0.0
        ctx = context[:]
        for name, num in zip(param_names, perm):
            arg_ctx = ctx[:]
            arg_ctx.extend(_encode_ids(f'"{name}": ', model))
            score += score_candidate(arg_ctx, _encode_ids(num, model), model)
            ctx.extend(_encode_ids(f'"{name}": {num}, ', model))
        if score > best_score:
            best_score, best_perm = score, perm

    return dict(zip(param_names, (float(n) for n in best_perm)))


def _extract_regex_param(context: list[int], model: Small_LLM_Model) -> str:
    """Infer a regex pattern from key_extract_number_paramsords in the prompt."""
    text = model.decode(context).lower()
    if "numbers" in text or "digits" in text:
        return r'\d+'
    if "vowels" in text:
        return r'[aeiou]'
    return r'.*'


def _extract_string_param(context: list[int], string_candidates: list[str], model: Small_LLM_Model) -> str:
    """Score each string candidate and pop and return the best."""
    best_score, best_index = float("-inf"), 0

    for idx, string in enumerate(string_candidates):
        score = score_candidate(context, _encode_ids(string, model), model)
        if score > best_score:
            best_score = score
            best_index = idx

    return string_candidates.pop(best_index)


def extract_parameters(
    original_prompt: str,
    input_ids: list[int],
    function,
    model: Small_LLM_Model,
) -> dict:
    """Dispatcher: extract all parameters for function by type."""
    context = input_ids[:]
    string_candidates = [word for word in original_prompt.split() if not _is_number(word)]

    number_param_names = [n for n, p in function.parameters.items() if p.type.lower() == 'number']
    number_values = _extract_number_params(number_param_names, context, model)

    params: dict = {}
    for arg_name, pdef in function.parameters.items():
        if pdef.type.lower() == 'number':
            value = number_values[arg_name]
        elif pdef.type.lower() == 'string' and arg_name.lower() == 'regex':
            value = _extract_regex_param(context, model)
        else:
            value = _extract_string_param(context, string_candidates, model)

        params[arg_name] = value
        context.extend(_encode_ids(f'"{arg_name}": {value}, ', model))

    return params
