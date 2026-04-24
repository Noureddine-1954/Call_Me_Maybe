from llm_sdk import Small_LLM_Model

from .llm_helpers import (score_candidate,
                          _all_valid_numbers,
                          _is_number,
                          _encode_ids,
                          best_ordered_assignment)


def _extract_number_params(
    param_names: list[str],
    context: list[int],
    model: Small_LLM_Model,
) -> dict[str, float]:
    result = best_ordered_assignment(param_names, _all_valid_numbers(context, model), context, model)
    return {k: float(v) if v else 0.0 for k, v in result.items()}


def _extract_string_params(
    param_names: list[str],
    context: list[int],
    model: Small_LLM_Model,
) -> dict[str, str]:
    return best_ordered_assignment(param_names, _all_valid_strings(context, model), context, model)


def _extract_regex_params(context: list[int], model: Small_LLM_Model) -> str:
    """Infer a regex pattern from key_extract_number_paramsords in the prompt."""
    text = model.decode(context).lower()
    if "numbers" in text or "digits" in text:
        return r'\d+'
    if "vowels" in text:
        return r'[aeiou]'
    return r'.*'


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
    number_values = _extract_number_params(original_prompt, number_param_names, context, model)

    params: dict = {}
    for arg_name, pdef in function.parameters.items():
        if pdef.type.lower() == 'number':
            value = number_values[arg_name]
        elif pdef.type.lower() == 'string' and arg_name.lower() == 'regex':
            value = _extract_regex_params(context, model)
        else:
            value = _extract_string_params(context, string_candidates, model)

        params[arg_name] = value
        context.extend(_encode_ids(f'"{arg_name}": {value}, ', model))

    return params
