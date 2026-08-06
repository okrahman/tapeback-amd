"""PII masking at the LLM boundary (_mask.py).

The property that matters is the round trip: masking removes raw PII from the text that
crosses to a provider, and unmasking restores the EXACT original, so the vault keeps what
was said. The other half is that masking off means nothing changes at all.
"""

import pytest

from tapeback._mask import Masker

_EMAIL = "ivan.petrov@example.com"
_PHONE = "+7 900 123-45-67"


def test_round_trip_restores_exact_original():
    masker = Masker(enabled=True)
    original = f"Call {_PHONE} or write to {_EMAIL} before Friday."

    masked = masker.mask(original)

    assert "@" not in masked
    assert "+7" not in masked
    assert masked == "Call [PHONE_1] or write to [EMAIL_1] before Friday."
    assert masker.unmask(masked) == original


def test_email_placeholder_and_mapping():
    masker = Masker(enabled=True)

    masked = masker.mask(f"write to {_EMAIL} please")

    assert masked == "write to [EMAIL_1] please"
    assert masker.mapping == {"[EMAIL_1]": "ivan.petrov@example.com"}


@pytest.mark.parametrize(
    "phone",
    ["+7 900 123-45-67", "+79001234567", "8 (900) 123 45 67", "8-900-123-45-67", "+1 415 555 2671"],
)
def test_phone_formats(phone):
    masker = Masker(enabled=True)

    masked = masker.mask(f"call {phone} now")

    assert masked == "call [PHONE_1] now"
    assert masker.mapping == {"[PHONE_1]": phone}


@pytest.mark.parametrize(
    "text",
    ["recorded in 2026", "order 12345678", "8 items left", "build +2 patch", "version 1.2.3"],
)
def test_non_pii_is_left_alone(text):
    masker = Masker(enabled=True)

    assert masker.mask(text) == text
    assert masker.mapping == {}


def test_same_value_gets_one_placeholder():
    masker = Masker(enabled=True)

    masked = masker.mask(f"{_EMAIL} and again {_EMAIL}")

    assert masked == "[EMAIL_1] and again [EMAIL_1]"
    assert masker.mapping == {"[EMAIL_1]": "ivan.petrov@example.com"}


def test_distinct_values_get_distinct_placeholders():
    masker = Masker(enabled=True)

    masked = masker.mask("a@x.com, b@y.com")

    assert masked == "[EMAIL_1], [EMAIL_2]"
    assert masker.mapping == {"[EMAIL_1]": "a@x.com", "[EMAIL_2]": "b@y.com"}


def test_counter_continues_across_texts_in_one_call():
    masker = Masker(enabled=True)

    first = masker.mask(f"note {_EMAIL}")
    second = masker.mask(f"user wrote {_EMAIL} and a@other.com")

    assert first == "note [EMAIL_1]"
    assert second == "user wrote [EMAIL_1] and [EMAIL_2]"
    assert masker.unmask(second) == "user wrote ivan.petrov@example.com and a@other.com"


def test_unmask_prefers_the_longest_placeholder():
    """[EMAIL_1] must not clobber the [EMAIL_11] that contains it."""
    masker = Masker(enabled=True)
    addresses = [f"user{i}@example.com" for i in range(1, 12)]

    masked = masker.mask(", ".join(addresses))

    assert masked.endswith("[EMAIL_11]")
    assert masker.unmask(masked) == ", ".join(addresses)


def test_disabled_masker_changes_nothing():
    masker = Masker(enabled=False)
    original = f"Call {_PHONE} or write to {_EMAIL}."

    masked = masker.mask(original)

    assert masked == original
    assert masker.mapping == {}
    assert masker.unmask(masked) == original


def test_empty_text():
    masker = Masker(enabled=True)

    assert masker.mask("") == ""
    assert masker.unmask("") == ""


def test_unmask_without_any_replacement_is_identity():
    masker = Masker(enabled=True)
    masker.mask("nothing personal here")

    assert masker.unmask("still [EMAIL_1] nothing") == "still [EMAIL_1] nothing"


def test_residual_placeholders_found():
    masker = Masker(enabled=True)
    masker.mask(f"write to {_EMAIL}")

    assert masker.residual_placeholders("reach [EMAIL_7] instead") == ["[EMAIL_7]"]


def test_no_residual_placeholders_after_successful_unmask():
    masker = Masker(enabled=True)
    masked = masker.mask(f"write to {_EMAIL}")

    assert masker.residual_placeholders(masker.unmask(masked)) == []


def test_residual_placeholders_ignored_when_nothing_was_masked():
    """A placeholder-shaped string we did not create is the model's own text."""
    masker = Masker(enabled=True)
    masker.mask("nothing personal here")

    assert masker.residual_placeholders("the model wrote [EMAIL_1]") == []
