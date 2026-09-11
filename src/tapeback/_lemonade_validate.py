"""Lemonade response validation.

Owns the schema checks every server-supplied segment must pass before it
can enter a transcript or the resume cache: finite ordered timestamps,
bounded text, and response/cumulative caps. A response that fails here is
a capability error (fallback), never partially trusted data."""

from __future__ import annotations

import math
from typing import Any

from tapeback._lemonade_errors import LemonadeCapabilityError

# Schema limits on per-response payload size to guard against pathological memory / CPU usage.
_MAX_RESPONSE_SEGMENTS = 5000

_MAX_SEGMENT_TEXT_CHARS = 10_000

# Cumulative bounds on total output across all chunk responses for a single transcription.
_MAX_CUMULATIVE_SEGMENTS = 50_000

_MAX_CUMULATIVE_TEXT_CHARS = 5_000_000

# Slack allowed on segment/word timestamps past a chunk's audio length before the
# response is rejected, to absorb server-side rounding at chunk boundaries.
_TIMESTAMP_SLACK_SECONDS = 1.0

_BOUNDARY_EPSILON_SECONDS = 0.001


def _finite_number(value: Any) -> float | None:
    """The value as a finite float, or None when it is not one.

    Booleans are ``int`` in Python but are never numbers in a JSON schema; NaN
    and infinities pass naive ``isinstance`` checks and then poison sorting,
    duration arithmetic, and every timestamp persisted into the resume cache.
    All three are rejected here, once, for every numeric field of a response.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _usable_timestamp(value: Any, name: str) -> float:
    """A finite, non-negative timestamp, or a sanitized capability error."""
    number = _finite_number(value)
    if number is None or number < 0.0:
        raise LemonadeCapabilityError(
            f"Lemonade returned a response with an unusable {name} "
            "(missing, not a number, boolean, non-finite, or negative)"
        )
    return number


def _require_segments(
    payload: dict[str, Any], upper_bound: float | None = None
) -> list[dict[str, Any]]:
    """Return the payload's timestamped segments, or reject the response outright.

    Tapeback's pipeline needs segment timestamps for speaker labelling and vault
    timing, so a text-only response is unusable even when the prose is good. This is
    where FLM-style compact responses are rejected in full.

    Every timestamp is strictly validated (finite number, ``0 <= start <= end``),
    and when ``upper_bound`` is given (a chunk's audio length) an ``end`` beyond it
    — modulo a small boundary slack — is also rejected: a hostile or broken server
    must not be able to write timestamps past the recording into the transcript or
    the resume cache.
    """
    raw = payload.get("segments")
    if isinstance(raw, list) and raw:
        if len(raw) > _MAX_RESPONSE_SEGMENTS:
            raise LemonadeCapabilityError(
                f"Lemonade returned too many segments in one response "
                f"({len(raw)} > {_MAX_RESPONSE_SEGMENTS})"
            )
        for item in raw:
            if not isinstance(item, dict):
                raise LemonadeCapabilityError(
                    "Lemonade returned segments without usable timestamps"
                )
            start = _usable_timestamp(item.get("start"), "segment start")
            end = _usable_timestamp(item.get("end"), "segment end")
            if end < start:
                raise LemonadeCapabilityError(
                    "Lemonade returned a segment whose end precedes its start"
                )
            if upper_bound is not None and end > upper_bound + _TIMESTAMP_SLACK_SECONDS:
                raise LemonadeCapabilityError(
                    "Lemonade returned a segment ending past the audio it was sent"
                )
            text = str(item.get("text") or "")
            if len(text) > _MAX_SEGMENT_TEXT_CHARS:
                raise LemonadeCapabilityError(
                    f"Lemonade returned a segment exceeding the text size limit "
                    f"({len(text)} > {_MAX_SEGMENT_TEXT_CHARS})"
                )
        return raw
    text = str(payload.get("text") or "")
    if text.strip():
        raise LemonadeCapabilityError(
            "Lemonade returned text without timestamped segments — tapeback requires "
            "segment timestamps and will fall back to faster-whisper"
        )
    # Silence: an empty response is legal output, not a capability problem.
    return []
