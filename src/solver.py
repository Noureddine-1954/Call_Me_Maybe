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


def choose_best_function(input_ids, functions, model):
    fn_token_seqs = {
        fn.name: model.encode(fn.name).tolist()[0]
        for fn in functions
    }

    candidates = list(fn_token_seqs.items())
    generated = []

    while True:
        logits = model.get_logits_from_input_ids(input_ids + generated)

        allowed_tokens = set()
        for _, seq in candidates:
            if len(generated) < len(seq):
                allowed_tokens.add(seq[len(generated)])

        if not allowed_tokens:
            break

        # pick best allowed token
        best_token = max(allowed_tokens, key=lambda t: logits[t])
        generated.append(best_token)

        # filter candidates
        new_candidates = []
        for name, seq in candidates:
            if len(generated) <= len(seq) and seq[:len(generated)] == generated:
                new_candidates.append((name, seq))

        candidates = new_candidates

        # exact match
        for name, seq in candidates:
            if generated == seq:
                return name

    return functions[0].name

def solve_one(
    prompt: PromptDef,
    functions: list[FunctionDef],
    model: Small_LLM_Model,
) -> FunctionCallOut:

    vocab_index_map = load_id_to_piece(model)

    prefix_ids = _encode_ids(build_prefix(prompt, functions) + ' ', model)

    best_function = choose_best_function(prefix_ids, functions, model)

    # print(prompt.prompt, best_function, {'test': None})

    return FunctionCallOut(
        prompt=prompt.prompt,
        name=best_function,
        parameters={}  # placeholder until you implement param generation
    )