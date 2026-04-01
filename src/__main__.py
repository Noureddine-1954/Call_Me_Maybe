import argparse
import sys
from pathlib import Path

from .parser import parser, InputError
from .solver import solve_one

from llm_sdk import Small_LLM_Model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="call-me-maybe")
    p.add_argument("--functions_definition",
                   type=Path,
                   default=Path("data/input/functions_definition.json"))
    p.add_argument("--input",
                   type=Path,
                   default=Path("data/input/function_calling_tests.json"))
    p.add_argument("--output",
                   type=Path,
                   default=Path("data/output/function_calling_results.json"))
    return p


def main() -> int:
    # first part, parsing the json's of both the prompts and functions
    args = build_parser().parse_args()
    try:
        content = parser(args.functions_definition, args.input)
    except InputError as e:
        print(f"{e}", file=sys.stderr)
        sys.exit(1)

    # second part
    model = Small_LLM_Model()

    for prompt in content["prompts"]:
        print(solve_one(prompt, content["functions"], model))
    sys.exit(2)


if __name__ == "__main__":
    main()
