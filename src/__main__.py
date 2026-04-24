import argparse
import json
import sys
import os
import time
from pathlib import Path

from .parser import parser, InputError
from .solver import solve_one

from llm_sdk import Small_LLM_Model


def _format_duration(seconds: float) -> str:
    """Return a compact human-readable duration."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    minutes, secs = divmod(seconds, 60)
    if minutes < 1:
        return f"{secs:.2f} s"
    hours, mins = divmod(int(minutes), 60)
    if hours < 1:
        return f"{mins}m {secs:.1f}s"
    return f"{hours}h {mins}m {secs:.1f}s"


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
    time_list = []
    args = build_parser().parse_args()

    # ── Parse inputs ─────────────────────────────────────────────────────
    try:
        content = parser(args.functions_definition, args.input)
    except InputError as e:
        print(f"{e}", file=sys.stderr)
        return 1

    # ── Load model ───────────────────────────────────────────────────────
    model = Small_LLM_Model()
    os.system('clear')
    print("Loading model ...\n", file=sys.stderr)

    # ── Process each prompt ──────────────────────────────────────────────
    results = []
    total = len(content["prompts"])
    for i, prompt in enumerate(content["prompts"]):
        print(
            f"Processing prompt {i + 1}/{total}: {prompt.prompt!r}",
            file=sys.stderr,
        )
        try:
            start_time = time.perf_counter()
            print('result ->', end=' ', flush=True)
            result = solve_one(prompt, content["functions"], model)
            elapsed = time.perf_counter() - start_time
            print(result)
            print(f"[took: {_format_duration(elapsed)}]", file=sys.stderr, end='\n\n')
            results.append(result.model_dump())
            time_list.append(elapsed)
        except Exception as exc:  # noqa: BLE001
            print(f"  Error: {exc}", file=sys.stderr)

    # ── Write output ─────────────────────────────────────────────────────
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results written to {output_path}", file=sys.stderr)
    except OSError as exc:
        print(f"Error writing output: {exc}", file=sys.stderr)
        return 1

    print(f"Processed {len(content['prompts'])} prompts in {_format_duration(sum(time_list))}")
    print(f"An average of: {_format_duration(sum(time_list)/len(time_list))} per prompt.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
