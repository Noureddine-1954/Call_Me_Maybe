from .json_definitions import FunctionDef, PromptDef


def prompt_function_choice(prompt: PromptDef, functions: list[FunctionDef]) -> str:
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

def prompt_parameter_extraction(prompt: PromptDef, function: FunctionDef) -> str:
    """Build prefix for PARAMETER EXTRACTION ONLY."""

    lines: list[str] = []

    # Role
    lines.append("You are a parameter extraction engine.")
    lines.append("Your ONLY task is to extract parameters for the given function.")
    lines.append("")

    # Function definition (VERY IMPORTANT)
    lines.append("Function to call:")
    lines.append(f"- {function.name}: {function.description}")

    lines.append("Parameters:")
    for pname, pdef in function.parameters.items():
        lines.append(f"- {pname}: {pdef.type}")

    # Rules
    lines.append("")
    lines.append("Rules:")
    lines.append("- Extract values ONLY from the user input.")
    lines.append("- Respect parameter types exactly.")
    lines.append("- Do NOT invent missing values.")
    lines.append("- If a value is missing, use a default value.")
    lines.append("- Output ONLY parameter values.")

    # User prompt
    lines.append("")
    lines.append(f"User: {prompt.prompt}")

    # Anchor (you will continue with constrained decoding)
    lines.append('Response: {"parameters": {')

    return "\n".join(lines)
