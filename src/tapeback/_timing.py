"""Stage timing for the processing pipeline — observability helper.

Reports how long each heavy pipeline stage takes so users can see where the
time goes (transcription dominates on CPU and small GPUs). Routed through the
pipeline's existing status callback rather than a second logging sink, so the
timings appear wherever progress already does: stderr in the CLI, the tray log
in the tray app.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

# Same shape as pipeline.StatusCallback; kept as a local alias to avoid a
# circular import (pipeline imports this module, not the other way round).
Reporter = Callable[[str], None]

# Minimum wall-clock gap between two progress lines. Transcription emits a
# segment every few hundred milliseconds; reporting each one would bury the
# other status output. Not configurable — this is display cadence, not behaviour.
PROGRESS_MIN_INTERVAL_SEC = 10.0

PERCENT = 100.0


def format_stage_duration(stage: str, seconds: float) -> str:
    """One-line, human-readable timing record for a pipeline stage."""
    return f"Stage '{stage}' took {seconds:.1f}s"


def _format_clock(seconds: float) -> str:
    """Format a position within the audio as M:SS or H:MM:SS."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_stage_progress(stage: str, position: float, total: float) -> str:
    """One-line progress record: how far into the audio a stage has got."""
    percent = min(PERCENT, position / total * PERCENT) if total > 0 else 0.0
    return f"  {stage}: {percent:.0f}% ({_format_clock(position)} / {_format_clock(total)})"


class ProgressReporter:
    """Throttled progress reporting for a long, position-based stage.

    Transcription yields segments continuously but gives no completion signal,
    so progress is derived from how far into the audio the latest segment ends.
    Updates are rate-limited to ``min_interval`` seconds of wall clock; the
    caller can therefore call ``update`` on every segment without spamming.
    """

    def __init__(
        self,
        stage: str,
        total_seconds: float,
        report: Reporter,
        *,
        min_interval: float = PROGRESS_MIN_INTERVAL_SEC,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._stage = stage
        self._total = total_seconds
        self._report = report
        self._min_interval = min_interval
        # Resolved at call time, not bound as a default argument: the reporter is
        # constructed deep inside transcription, so tests can only reach its clock
        # by patching the module attribute.
        self._clock = clock if clock is not None else time.monotonic
        self._last_report = self._clock()

    def update(self, position_seconds: float) -> None:
        """Report progress if enough wall-clock time has passed since the last line."""
        if self._total <= 0:
            return
        now = self._clock()
        if now - self._last_report < self._min_interval:
            return
        self._last_report = now
        self._report(format_stage_progress(self._stage, position_seconds, self._total))


@contextmanager
def stage_timer(
    stage: str,
    report: Reporter,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[None]:
    """Measure wall-clock duration of a stage and report it via ``report``.

    The timing is reported in a ``finally`` block so that a stage which raises
    still reports how long it ran before failing.
    """
    start = clock()
    try:
        yield
    finally:
        report(format_stage_duration(stage, clock() - start))
