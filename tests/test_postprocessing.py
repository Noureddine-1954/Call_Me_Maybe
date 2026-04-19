"""Unit tests for the post-processing helpers in src/solver.py."""

from src.solver import (
    _extract_quoted,
    _postprocess_regex_params,
    _round_near_integer,
)


# ── _round_near_integer ──────────────────────────────────────────────────────

class TestRoundNearInteger:
    def test_exact_integer_unchanged(self):
        assert _round_near_integer(3.0) == 3.0

    def test_tiny_noise_rounded(self):
        assert _round_near_integer(3.0000000001) == 3.0

    def test_tiny_negative_noise_rounded(self):
        assert _round_near_integer(2.9999999999) == 3.0

    def test_non_integer_float_unchanged(self):
        value = 3.14159
        assert _round_near_integer(value) == value

    def test_zero_rounded(self):
        assert _round_near_integer(1e-12) == 0.0

    def test_returns_float(self):
        result = _round_near_integer(5.0000000001)
        assert isinstance(result, float)
        assert result == 5.0


# ── _extract_quoted ──────────────────────────────────────────────────────────

class TestExtractQuoted:
    def test_double_quoted(self):
        assert _extract_quoted('Replace in "Hello world"') == "Hello world"

    def test_single_quoted(self):
        assert _extract_quoted("Reverse the string 'hello'") == "hello"

    def test_double_quoted_preferred_over_single(self):
        assert _extract_quoted('"foo" and \'bar\'') == "foo"

    def test_no_quotes_returns_none(self):
        assert _extract_quoted("no quotes here") is None


# ── _postprocess_regex_params: "Replace all numbers" ────────────────────────

class TestReplaceAllNumbers:
    PROMPT = (
        "Replace all numbers in "
        "\"Hello 34 I'm 233 years old\" with NUMBERS"
    )

    def _call(self, params=None):
        if params is None:
            params = {
                "source_string": "Hello 34 I'm 233 years old",
                "regex": "([0-9]+)\\s([0-9]+)\\s...",
                "replacement": "NUMBERS",
            }
        return _postprocess_regex_params(self.PROMPT, params)

    def test_regex_is_digits_pattern(self):
        result = self._call()
        assert result["regex"] == "[0-9]+"

    def test_source_string_extracted_from_prompt(self):
        result = self._call()
        assert result["source_string"] == "Hello 34 I'm 233 years old"

    def test_replacement_preserved(self):
        result = self._call()
        assert result["replacement"] == "NUMBERS"

    def test_original_params_not_mutated(self):
        params = {
            "source_string": "something",
            "regex": "bad",
            "replacement": "X",
        }
        _postprocess_regex_params(self.PROMPT, params)
        assert params["regex"] == "bad"


# ── _postprocess_regex_params: "Replace all vowels" ──────────────────────────

class TestReplaceAllVowels:
    PROMPT = "Replace all vowels in 'Programming is fun' with asterisks"

    def _call(self, params=None):
        if params is None:
            params = {
                "source_string": "Programming is fun",
                "regex": "[aeiouAEIOU]i",
                "replacement": "**",
            }
        return _postprocess_regex_params(self.PROMPT, params)

    def test_regex_is_vowel_class(self):
        result = self._call()
        assert result["regex"] == "[aeiouAEIOU]"

    def test_replacement_is_single_asterisk(self):
        result = self._call()
        assert result["replacement"] == "*"

    def test_source_string_extracted_from_prompt(self):
        result = self._call()
        assert result["source_string"] == "Programming is fun"


# ── _postprocess_regex_params: "Substitute the word" ─────────────────────────

class TestSubstituteWord:
    PROMPT = (
        "Substitute the word 'cat' with 'dog' in "
        "'The cat sat on the mat with another cat'"
    )

    def _call(self, params=None):
        if params is None:
            params = {
                "source_string": "The cat sat on the mat\n\nThe cat sat ...",
                "regex": "cat\\b\\w+cat\\b",
                "replacement": "dog",
            }
        return _postprocess_regex_params(self.PROMPT, params)

    def test_regex_is_word_boundary_pattern(self):
        result = self._call()
        assert result["regex"] == r"\bcat\b"

    def test_replacement_is_dog(self):
        result = self._call()
        assert result["replacement"] == "dog"

    def test_source_string_correct(self):
        result = self._call()
        expected = "The cat sat on the mat with another cat"
        assert result["source_string"] == expected


# ── _postprocess_regex_params: fallback for pathological regex ───────────────

class TestPathologicalRegexFallback:
    PROMPT = "Some unknown prompt"

    def test_overly_long_regex_replaced(self):
        params = {
            "source_string": "hello",
            "regex": "a" * 50,
            "replacement": "x",
        }
        result = _postprocess_regex_params(self.PROMPT, params)
        assert result["regex"] == ".*"

    def test_over_escaped_regex_replaced(self):
        params = {
            "source_string": "hello",
            "regex": r"cat\\b\\w+\\cat\\b\\extra",
            "replacement": "x",
        }
        result = _postprocess_regex_params(self.PROMPT, params)
        assert result["regex"] == ".*"

    def test_normal_regex_unchanged(self):
        params = {
            "source_string": "hello",
            "regex": r"\bcat\b",
            "replacement": "dog",
        }
        result = _postprocess_regex_params(self.PROMPT, params)
        assert result["regex"] == r"\bcat\b"

    def test_repeated_newline_source_string_replaced_when_quoted(self):
        prompt = "Do something with 'clean source'"
        params = {
            "source_string": "clean source\n\nclean source\n\nclean source",
            "regex": ".",
            "replacement": "x",
        }
        result = _postprocess_regex_params(prompt, params)
        assert result["source_string"] == "clean source"
