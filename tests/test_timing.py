"""Unit tests for pipeline stage timing helpers."""

import pytest

from tapeback._timing import format_stage_duration, stage_timer


def test_format_stage_duration_one_decimal():
    assert format_stage_duration("transcribe", 12.34) == "Stage 'transcribe' took 12.3s"


def test_format_stage_duration_keeps_trailing_zero():
    # Fixed one-decimal format: 7 seconds renders as "7.0s", not "7s".
    assert format_stage_duration("merge", 7.0) == "Stage 'merge' took 7.0s"


def test_stage_timer_reports_elapsed_via_fake_clock():
    ticks = iter([100.0, 112.5])
    reported: list[str] = []
    with stage_timer("transcribe", reported.append, clock=lambda: next(ticks)):
        pass
    assert reported == ["Stage 'transcribe' took 12.5s"]


def test_stage_timer_reports_on_exception_and_reraises():
    ticks = iter([0.0, 3.0])
    reported: list[str] = []
    with (
        pytest.raises(ValueError, match="boom"),
        stage_timer("diarize", reported.append, clock=lambda: next(ticks)),
    ):
        raise ValueError("boom")
    assert reported == ["Stage 'diarize' took 3.0s"]
