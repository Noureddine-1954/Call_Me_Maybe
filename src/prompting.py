from .json_definitions import FunctionDef, PromptDef


def build_prefix(prompt: PromptDef, functions: list[FunctionDef]) -> str:
    """Build a compact, stable semantic prefix for function selection.

    The model only needs to:
    - choose the correct function
    - infer correct parameter values

    JSON structure is enforced externally via constrained decoding.
    """
    lines: list[str] = []

    # Strong role anchoring (VERY important for small LLMs)
    lines.append("You are a function-calling engine.")
    lines.append("Select the best function and fill in its parameters correctly.")
    lines.append("")

    # Functions description (keep it clean and scannable)
    lines.append("Available functions:")
    for fn in functions:
        params_desc = ", ".join(
            f"{pname}: {pdef.type}"
            for pname, pdef in fn.parameters.items()
        )
        line = f"- {fn.name}({params_desc}) -> {fn.description}"
        lines.append(line)

    # Add light guidance (helps small models a lot)
    lines.append("")
    lines.append("Guidelines:")
    lines.append("- Choose the function that best matches the user intent.")
    lines.append("- Extract parameter values directly from the user input.")
    lines.append("- Use correct types (numbers, strings, etc).")

    # User prompt
    lines.append("")
    lines.append(f"User: {prompt.prompt}")

    # Anchor into constrained decoding
    lines.append('Response: {"name": "')

    return "\n".join(lines)
