from .constrained_decoding import choose_best_function, extract_parameters
from llm_sdk import Small_LLM_Model

from .json_definitions import FunctionCallOut, FunctionDef, PromptDef
from .prompting import prompt_function_choice, prompt_parameter_extraction


def _encode_ids(text: str, model: Small_LLM_Model) -> list[int]:
    """Return a flat list[int] of token ids for *text*."""
    return model.encode(text).tolist()[0]


def solve_one(
    prompt: PromptDef,
    functions: list[FunctionDef],
    model: Small_LLM_Model,
) -> FunctionCallOut:

    if prompt.prompt.strip() == '':
        return FunctionCallOut(
            prompt=prompt.prompt,
            name='no_valid_function',
            prameters={}
        )
    prefix_ids = _encode_ids(prompt_function_choice(prompt, functions) + ' ', model)
    best_function = choose_best_function(prefix_ids, functions, model)

    function_map = {
        fn.name: fn for fn in functions
    }
    parameter_prefix_ids = _encode_ids(prompt_parameter_extraction(prompt, function_map[best_function]), model)
    extracted_parameters = extract_parameters(prompt.prompt, parameter_prefix_ids, function_map[best_function], model)
    # print(prompt.prompt, best_function, {'test': None})

    return FunctionCallOut(
        prompt=prompt.prompt,
        name=best_function,
        parameters=extracted_parameters  # placeholder
    )
