import math
import re
from llm_sdk import Small_LLM_Model


def _encode_ids(text: str, model: Small_LLM_Model) -> list[int]:
    """Return a flat list[int] of token ids for *text*."""
    return model.encode(text).tolist()[0]


def _round_near_integer(value: float, tol: float = 1e-9) -> float:
    rounded = round(value)
    if abs(value - rounded) <= tol:
        return float(rounded)
    return value


def _all_valid_numbers(input_ids, model):
    decoded = model.decode(input_ids)

    # Prefer the actual user prompt segment to avoid extracting numbers
    # from the instruction/prefix text.
    user_match = re.search(r"User:\s*(.*?)\s*(?:\nResponse:|$)", decoded,
                           flags=re.DOTALL)
    source_text = user_match.group(1) if user_match else decoded

    number_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    matches = re.findall(number_pattern, source_text)

    all_numbers = []
    for m in matches:
        if "inf" in m.lower() or m in {"+", "-", "."}:
            continue
        try:
            all_numbers.append(str(_round_near_integer(float(m))))
        except ValueError:
            continue

    return all_numbers


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


def extract_numbers(input_ids: list[int],
                    model: Small_LLM_Model,
                    valid_nums: list[str] | None = None) -> float:

    if valid_nums is None:
        valid_nums = _all_valid_numbers(input_ids, model)
    if not valid_nums:
        return 0.0

    best_score = float("-inf")
    best_index = 0

    for idx, num in enumerate(valid_nums):
        candidate_tokens = _encode_ids(num, model)
        score = score_candidate(input_ids, candidate_tokens, model)

        if score > best_score:
            best_score = score
            best_index = idx

    chosen = valid_nums.pop(best_index)
    return float(chosen)


def extract_strings(input_ids: list[int], model: Small_LLM_Model) -> str:
    """Extract strings from the user prompt."""
    all_words = model.decode(input_ids).split()
    valid_strs = [w for w in all_words if not _is_number(w)]
    if not valid_strs:
        return ""
    # For simplicity, return the first
    return valid_strs[0]


def extract_regexes(input_ids: list[int], model: Small_LLM_Model) -> str:
    """Extract or infer regex patterns from the user prompt."""
    text = model.decode(input_ids).lower()
    if "numbers" in text or "digits" in text:
        return r'\d+'
    if "vowels" in text:
        return r'[aeiou]'
    # Default
    return r'.*'


def _is_number(word: str) -> bool:
    try:
        float(word)
        return True
    except ValueError:
        return False


def extract_parameters(input_ids: list[int], function,
                       model: Small_LLM_Model) -> dict:
    """Extract parameters for the function using the input_ids."""
    params = {}
    context = input_ids[:]
    number_candidates = _all_valid_numbers(input_ids, model)
    string_candidates = [
        word for word in model.decode(input_ids).split()
        if not _is_number(word)
    ]

    for arg_name, pdef in function.parameters.items():
        # print(arg_name, pdef.type)
        if pdef.type.lower() == 'number':
            value = extract_numbers(input_ids, model, number_candidates)
            params[arg_name] = float(value)

        elif pdef.type.lower() == 'string' and arg_name.lower() == 'regex':
            value = extract_regexes(input_ids, model)
            params[arg_name] = value

        elif pdef.type.lower() == 'string':
            value = string_candidates.pop(0) if string_candidates else ""
            params[arg_name] = value

        context.extend(_encode_ids(f"{arg_name}: {value}", model))

    return params
