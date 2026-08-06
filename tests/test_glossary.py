"""Tests for the default hotwords glossary."""

import pytest

from tapeback.glossary import DEFAULT_HOTWORDS


def _terms() -> list[str]:
    return [t.strip() for t in DEFAULT_HOTWORDS.split(",")]


@pytest.mark.parametrize(
    "term",
    ["tapeback", "Whisper", "Obsidian", "LLM", "RAG", "ONNX", "OpenVINO", "Jira", "markdown"],
)
def test_glossary_contains_the_terms_that_measurably_failed(term):
    """Every term here was observed mangled in a real transcript."""
    assert term in _terms()


def test_glossary_has_no_empty_or_duplicate_entries():
    terms = _terms()
    assert "" not in terms
    duplicates = {t for t in terms if terms.count(t) > 1}
    assert not duplicates, f"duplicated: {sorted(duplicates)}"


def test_glossary_is_latin_only():
    """A Cyrillic entry would bias towards the spelling the glossary exists to prevent."""
    cyrillic = [t for t in _terms() if any("Ѐ" <= ch <= "ӿ" for ch in t)]
    assert cyrillic == []


def test_glossary_fits_the_hotwords_token_budget():
    """faster-whisper silently truncates hotwords past `max_length // 2 - 1` = 223 tokens.

    Everything beyond the cut simply stops biasing anything, with no warning — a first
    draft of the glossary measured 295 tokens and lost a quarter of itself that way.

    Measured with Whisper's own tokenizer, this content runs ~3.08 characters per
    token; 3.0 is used here so the estimate errs on the side of being too strict. The
    real tokenizer is not used because it needs the model files, which CI does not have.
    """
    chars_per_token = 3.0
    estimated_tokens = len(DEFAULT_HOTWORDS) / chars_per_token
    assert estimated_tokens < 223, (
        f"~{estimated_tokens:.0f} tokens exceeds the 223 faster-whisper keeps; "
        "terms past the cut would be silently ignored"
    )
