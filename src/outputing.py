from llm_sdk.llm_sdk import Small_LLM_Model
from .parser import FunctionDef
from .json_definitions import FunctionCallOut


def solve_one(prompt: str,
              functions: list[FunctionDef],
              model: Small_LLM_Model) -> FunctionCallOut:
    ...
