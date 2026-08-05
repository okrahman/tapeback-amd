"""Unit tests for pipeline stage timing helpers."""

import pytest

from tapeback._timing import (
    ProgressReporter,
    format_stage_duration,
    format_stage_progress,
    stage_timer,
)


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


@pytest.mark.parametrize(
    ("position", "total", "expected"),
    [
        (45.0, 300.0, "  transcribe mic: 15% (0:45 / 5:00)"),
        (0.0, 300.0, "  transcribe mic: 0% (0:00 / 5:00)"),
        # Past an hour the clock grows a third field on both sides.
        (3725.0, 7200.0, "  transcribe mic: 52% (1:02:05 / 2:00:00)"),
        # VAD can push the last segment past the nominal duration; clamp at 100%.
        (310.0, 300.0, "  transcribe mic: 100% (5:10 / 5:00)"),
    ],
)
def test_format_stage_progress(position, total, expected):
    assert format_stage_progress("transcribe mic", position, total) == expected


def test_format_stage_progress_zero_total_does_not_divide_by_zero():
    assert format_stage_progress("transcribe", 0.0, 0.0) == "  transcribe: 0% (0:00 / 0:00)"


def test_progress_reporter_throttles_updates():
    # Clock: construction, then one call per update.
    ticks = iter([0.0, 1.0, 2.0, 11.0, 12.0, 21.0])
    reported: list[str] = []
    reporter = ProgressReporter(
        "transcribe mic", 100.0, reported.append, min_interval=10.0, clock=lambda: next(ticks)
    )
    for position in (10.0, 20.0, 30.0, 40.0, 50.0):
        reporter.update(position)

    # Only the updates at t=11 and t=21 clear the 10s gap.
    assert reported == [
        "  transcribe mic: 30% (0:30 / 1:40)",
        "  transcribe mic: 50% (0:50 / 1:40)",
    ]


def test_progress_reporter_silent_when_duration_unknown():
    reported: list[str] = []
    reporter = ProgressReporter("transcribe", 0.0, reported.append, min_interval=0.0)
    reporter.update(5.0)
    assert reported == []
