*Noureddine-1954*

# Call Me Maybe — Function-Calling with Constrained Decoding

## Description

**Call Me Maybe** is a Python project that translates natural-language prompts
into schema-compliant JSON function calls using a local LLM
(`Qwen/Qwen3-0.6B`) and **constrained decoding** — without relying on
the model to spontaneously emit valid JSON.

Given a set of function definitions (name, description, parameter types) and a
list of prompts, the program produces a JSON array where each item has exactly
the keys `prompt`, `name`, and `parameters`, matching the schema of the chosen
function.

---

## Instructions

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Install

```bash
make install
```

### Run

```bash
make run
# equivalent to:
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Lint

```bash
make lint          # flake8 + mypy
make lint-strict   # flake8 + mypy --strict
```

### Clean

```bash
make clean
```

---

## Resources

- [Qwen/Qwen3-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
- [HuggingFace Tokenizers library](https://huggingface.co/docs/tokenizers)
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)
- [Constrained decoding — survey paper](https://arxiv.org/abs/2403.06988)

---

## Algorithm: Constrained Decoding and Schema Enforcement

### Overview

Standard LLM text generation samples (or greedily decodes) the next token from
the full vocabulary.  For structured output this is unreliable: the model may
generate syntactically invalid JSON, wrong field names, or wrong value types.

**Constrained decoding** restricts the set of valid next tokens at each step
so that the output is *guaranteed* to be schema-valid.  This project implements
the following pipeline:

```
prompt + function list
        │
        ▼
   build_prefix()          ← semantic context (prompting.py)
        │
        ▼
  encode → context_ids     ← list[int] of token ids
        │
        ▼
choose_best_function()     ← score each function name with
        │                    score_token_sequence() (constrained_decoding.py)
        ▼
  extend context with      ← '", "parameters": {'
        │
        ▼
  for each parameter:
    extend context with    ← '"param_name": '
    generate_value()       ← token-by-token constrained generation
        │                    (json_constrained.py)
        ▼
  FunctionCallOut          ← pydantic-validated output (json_definitions.py)
```

### Function Selection

The function name is chosen via **maximum likelihood scoring**:

For each candidate function *f* with name tokenised as `[t₁, t₂, …, tₙ]`:

```
score(f) = Σᵢ log P(tᵢ | context, t₁…tᵢ₋₁)
```

where each `P(tᵢ | …)` is obtained from `get_logits_from_input_ids` followed
by a numerically-stable `log_softmax`.  The function with the highest total
log-probability is selected.

### Value Generation (string)

1. The opening double-quote `"` is appended to the context.
2. At each step the top-K logits are computed.
3. A token is accepted only if every character in its decoded piece is safe
   inside a JSON string (no bare `"`, no control characters other than `\t`).
4. The closing `"` token is detected and terminates generation.
5. A max-length cap prevents infinite loops.

### Value Generation (number)

1. Top-K tokens are evaluated; only tokens whose pieces consist of
   `0-9 . - + e E` are accepted.
2. Additional guards prevent double decimal points and duplicate exponents.
3. The accumulated string is parsed with `float()` at the end; errors fall
   back to `0.0`.

### Value Generation (boolean)

The literals `"true"` and `"false"` are each scored with
`score_token_sequence`; the higher-scoring one is returned.

### Object / Array defaults

For `object` and `array` types, minimal schema-valid defaults (`{}` / `[]`)
are returned.  Recursive typed generation is outside the scope of this project.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Argmax / deterministic generation | Reliability over diversity; results are reproducible |
| Top-K filtering (K=200) | Speeds up the inner loop vs. scanning all 150 k+ vocab tokens |
| Pydantic `extra="forbid"` on all models | Guarantees no extra keys leak into the output |
| Graceful fallback on every error | The program never crashes; invalid results are replaced by type-defaults |
| `get_path_to_tokenizer_file()` for vocab | Only public API of `Small_LLM_Model` is used |
| Context-extended parameter generation | Each parameter is generated with full prior context (function name + previous parameters), improving coherence |

---

## Performance Analysis

- **Model loading**: ~10–30 s depending on hardware; done once per run.
- **Per-prompt cost**: `O(F × N_name)` forward passes for function selection
  (F = number of functions, N_name = average name length in tokens) plus
  `O(P × L)` passes for parameter generation (P = parameters per function,
  L = average value length in tokens).
- For the 5-function / 11-prompt test suite, total inference time is
  dominated by model warm-up; subsequent prompts run in a few seconds each
  on CPU.

---

## Challenges

- **Qwen tokenizer piece format**: the tokenizer uses byte-level BPE where
  spaces may appear as `Ġ` (U+0120).  The `display_piece()` helper in
  `tokenizer_map.py` normalises these before filtering.
- **Multi-token numbers**: values like `265` may be a single token or split
  across several tokens.  The number generator handles both cases via an
  iterative character-class filter.
- **Context length**: very long prefixes increase latency.  The prefix is
  kept compact (one line per function, no examples).

---

## Testing Strategy

- **Schema validation**: every output is validated by Pydantic (`FunctionCallOut`)
  and round-tripped through `json.loads(result.model_dump_json())`.
- **Graceful degradation**: all generation steps are wrapped in `try/except`;
  type-safe defaults are returned on failure.
- **Lint gate**: `flake8` and `mypy` are run on every change to catch type
  errors early.

---

## Example Usage

```bash
$ make run
Loading model …
Processing prompt 1/11: 'What is the sum of 2 and 3?'
Processing prompt 2/11: 'What is the sum of 265 and 345?'
…
Results written to data/output/function_calling_results.json

$ cat data/output/function_calling_results.json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  },
  …
]
```
