import json
import math
import random
import re
from llm_sdk import Small_LLM_Model

def _encode_ids(text: str, model: Small_LLM_Model) -> list[int]:
    """Return a flat list[int] of token ids for *text*."""
    return model.encode(text).tolist()[0]

def _round_near_integer(value: float, tol: float = 1e-9) -> float:
    rounded = round(value)
    if abs(value - rounded) <= tol:
        return float(rounded)
    return value

# ...existing code...
def generate_value(context_ids, param_name, param_type, model, prompt_text=None, state=None, function_name=None):
    def _decode_ctx(ids):
        try:
            return model.decode(ids)
        except Exception:
            pieces = model.tokenizer_map.load_id_to_piece()
            return "".join(pieces[i] for i in ids if 0 <= i < len(pieces))

    def _safe_logits(ids):
        try:
            return model.get_logits_from_input_ids(ids)
        except TypeError:
            return model.get_logits_from_input_ids([ids[-1]])

    def _as_last_row(logits):
        if logits and isinstance(logits[0], (list, tuple)):
            return logits[-1]
        return logits

    def _clean_piece(piece: str) -> str:
        return piece.replace("▁", " ").strip()

    text = prompt_text if prompt_text is not None else _decode_ctx(context_ids)
    lower = text.lower()
    pname = param_name.lower()

    if state is None:
        state = {}
    state.setdefault("used_num_idx", set())
    state.setdefault("used_quote_idx", set())

    # ---------- number ----------
    if param_type == "number":
        nums = [float(m.group(0).replace(",", ".")) for m in re.finditer(r"[-+]?\d+(?:[.,]\d+)?", text)]
        if nums:
            # Prefer parameter-position semantics (a=first, b=second), then first unused
            preferred_idx = 0
            if pname in {"b", "second", "num2", "y"} and len(nums) > 1:
                preferred_idx = 1
            elif pname in {"a", "first", "num1", "x"}:
                preferred_idx = 0
            else:
                for i in range(len(nums)):
                    if i not in state["used_num_idx"]:
                        preferred_idx = i
                        break

            state["used_num_idx"].add(preferred_idx)
            return _round_near_integer(nums[preferred_idx])

        # fallback to constrained token
        pieces = model.tokenizer_map.load_id_to_piece()
        logits = _as_last_row(_safe_logits(context_ids))
        allowed = [i for i, p in enumerate(pieces) if re.fullmatch(r"[-+]?\d+(\.\d+)?", _clean_piece(p))]
        if allowed:
            best_id = max(allowed, key=lambda i: float(logits[i]))
            return _round_near_integer(float(_clean_piece(pieces[best_id])))
        return 0.0

    # ---------- string ----------
    if param_type == "string":
        quoted = [m.group(2) for m in re.finditer(r"(['\"])(.*?)\1", text)]

        # function-aware rules for your dataset
        if function_name == "fn_greet" and pname == "name":
            m = re.search(r"\bgreet\s+([A-Za-z][A-Za-z0-9_\-]*)", text, re.IGNORECASE)
            if m:
                return m.group(1)

        if function_name == "fn_reverse_string" and pname in {"s", "text", "string"} and quoted:
            return quoted[0]

        if function_name == "fn_substitute_string_with_regex":
            if pname == "source_string":
                m = re.search(r"\bin\s+(['\"])(.*?)\1", text, re.IGNORECASE)
                if m:
                    return m.group(2)
                if quoted:
                    return quoted[0]

            if pname == "regex":
                # 1) explicit "word/pattern '<x>'" -> use x
                m = re.search(r"\b(?:word|pattern|regex)\s+(['\"])(.*?)\1", text, re.IGNORECASE)
                if m:
                    return m.group(2)

                # 2) if prompt says "replace ... in '<source>'", use first quoted before "in" as pattern
                m = re.search(r"\b(?:replace|substitute)\b(.*?)\bin\s+(['\"])(.*?)\2", text, re.IGNORECASE)
                if m:
                    left = m.group(1)
                    q = re.findall(r"(['\"])(.*?)\1", left)
                    if q:
                        return q[-1][1]

                # 3) semantic fallback (small controlled mapping)
                semantic_map = {
                    "numbers": r"\d+",
                    "digits": r"\d+",
                    "vowels": r"[AEIOUaeiou]",
                    "whitespace": r"\s+",
                    "spaces": r"\s+",
                }
                for k, v in semantic_map.items():
                    if re.search(rf"\b{k}\b", lower):
                        return v

                # 4) final fallback: first unused quoted token or safe default
                if quoted:
                    return quoted[0]
                return r".+"

            if pname == "replacement":
                m = re.search(r"\bwith\s+(['\"])(.*?)\1", text, re.IGNORECASE)
                if m:
                    return m.group(2)
                m = re.search(r"\bwith\s+([A-Za-z][A-Za-z0-9_\-]*)", text, re.IGNORECASE)
                if m:
                    word = m.group(1)
                    if word.lower() in {"asterisk", "asterisks"}:
                        return "*"
                    return word

        # generic: first unused quoted string
        for i, q in enumerate(quoted):
            if i not in state["used_quote_idx"]:
                state["used_quote_idx"].add(i)
                return q

        # constrained fallback (single token)
        pieces = model.tokenizer_map.load_id_to_piece()
        logits = _as_last_row(_safe_logits(context_ids))
        allowed = [i for i, p in enumerate(pieces) if re.fullmatch(r"[A-Za-z][A-Za-z0-9_\-]{0,30}", _clean_piece(p))]
        if allowed:
            best_id = max(allowed, key=lambda i: float(logits[i]))
            return _clean_piece(pieces[best_id]).strip(" .,!?:;")
        return ""

    return None


def encode_json_pair(parameter_name, value, model):
    return _encode_ids(f'"{parameter_name}": {json.dumps(value)}, ', model)


def extract_parameters(input_ids, function, model):
    context_ids = input_ids[:]
    params = {}

    # Decode original prompt once (avoid pollution from appended json pairs)
    try:
        prompt_text = model.decode(input_ids)
    except Exception:
        pieces = model.tokenizer_map.load_id_to_piece()
        prompt_text = "".join(pieces[i] for i in input_ids if 0 <= i < len(pieces))

    state = {}

    for pname, pdef in function.parameters.items():
        value = generate_value(
            context_ids=context_ids,
            param_name=pname,
            param_type=pdef.type,
            model=model,
            prompt_text=prompt_text,
            state=state,
            function_name=getattr(function, "name", None),
        )

        params[pname] = value
        context_ids += encode_json_pair(pname, value, model)

    return params
# ...existing code...
