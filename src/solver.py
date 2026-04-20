from .constrained_decoding import choose_best_function

from llm_sdk import Small_LLM_Model

from .json_definitions import FunctionCallOut, FunctionDef, PromptDef
from .prompting import build_prefix
from .tokenizer_map import load_id_to_piece


def _encode_ids(text: str, model: Small_LLM_Model) -> list[int]:
    """Return a flat list[int] of token ids for *text*."""
    return model.encode(text).tolist()[0]


def _round_near_integer(value: float, tol: float = 1e-9) -> float:
    rounded = round(value)
    if abs(value - rounded) <= tol:
        return float(rounded)
    return value


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

    vocab_index_map = load_id_to_piece(model)

    prefix_ids = _encode_ids(build_prefix(prompt, functions) + ' ', model)

    best_function = choose_best_function(prefix_ids, functions, model)

    # print(prompt.prompt, best_function, {'test': None})

    return FunctionCallOut(
        prompt=prompt.prompt,
        name=best_function,
        parameters={}  # placeholder
    )
