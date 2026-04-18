"""Semantic-prefix builder for constrained function-call decoding.

The prefix is used solely to provide the model with semantic context
(which function to call and what parameters to fill in).  Program
correctness does NOT depend on the model generating valid JSON from
this prompt; all structure is enforced by constrained decoding.
"""

from .json_definitions import FunctionDef, PromptDef


def build_prefix(prompt: PromptDef, functions: list[FunctionDef]) -> str:
    """Build a compact, stable prefix that describes the task.

    The prefix ends with ``{"name": "`` so that the next tokens the model
    predicts are the function name, which is then selected via constrained
    decoding.
    """
    lines: list[str] = []
    lines.append("Available functions:")
    for fn in functions:
        params_desc = ", ".join(
            f"{pname} ({pdef.type})"
            for pname, pdef in fn.parameters.items()
        )
        line = f"  {fn.name}: {fn.description} Parameters: {params_desc}"
        lines.append(line)
    lines.append("")
    lines.append(f"User: {prompt.prompt}")
    lines.append('Response: {"name": "')
    return "\n".join(lines)
