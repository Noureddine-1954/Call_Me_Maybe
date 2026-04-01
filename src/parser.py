from pathlib import Path
from typing import Any

import json

from .json_definitions import (FunctionDef,
                               PromptDef)


class InputError(Exception):
    """Custom error for missing/invalid/parsing-related input errors."""


def safe_json_read(path: Path) -> Any:
    try:
        with path.open("r") as f:
            return json.load(f)

    except FileNotFoundError:
        raise InputError(f"Error: The file '{path}' was not found.")

    except json.JSONDecodeError as e:
        raise InputError(f"Error: Failed to decode JSON from '{path}'."
                         f"\nDetails: {e}")

    except PermissionError as e:
        raise InputError(f"Error: cannot read '{path}' (permission denied)."
                         f"\nDetails: {e}")

    except UnicodeDecodeError as e:
        raise InputError(f"Error: '{path}' is not valid UTF-8.\nDetails: {e}")

    except OSError as e:
        raise InputError(f"Error: '{path}' cannot read.\nDetails: {e}")


def parser(function_path: Path, prompts_path: Path) -> dict[str, list[Any]]:
    data: dict[str, list[Any]] = {"prompts": [], "functions": []}

    fct_json = safe_json_read(function_path)
    if not isinstance(fct_json, list):
        raise InputError(f"Error: '{function_path}' must contain a JSON array")
    for item in fct_json:
        if not isinstance(item, dict):
            raise InputError(f"Error: '{function_path}' items must be JSON"
                             "objects.")
        data["functions"].append(FunctionDef.model_validate(item))

    prmpt_json = safe_json_read(prompts_path)
    if not isinstance(prmpt_json, list):
        raise InputError(f"Error: '{prompts_path}' must contain a JSON array.")
    for item in prmpt_json:
        if not isinstance(item, dict):
            raise InputError(f"Error: '{prompts_path}' items must be JSON"
                             "objects.")
        data["prompts"].append(PromptDef.model_validate(item))

    return data
