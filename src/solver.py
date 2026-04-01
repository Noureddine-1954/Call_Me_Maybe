from llm_sdk import Small_LLM_Model
from .json_definitions import FunctionDef, FunctionCallOut


def solve_one(prompt: str,
              functions: list[FunctionDef],
              model: Small_LLM_Model) -> FunctionCallOut:
    return (model.encode(prompt.prompt))
