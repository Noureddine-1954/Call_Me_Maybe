"""Stub module kept for backward compatibility. Use src/solver.py instead."""

from llm_sdk import Small_LLM_Model
from .json_definitions import FunctionDef, FunctionCallOut


def solve_one(
    prompt: str,
    functions: list[FunctionDef],
    model: Small_LLM_Model,
) -> FunctionCallOut:
    """Stub — not used by the main pipeline."""
    raise NotImplementedError("Use src/solver.py instead")
