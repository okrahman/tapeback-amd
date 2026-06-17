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


def format_stage_duration(stage: str, seconds: float) -> str:
    """One-line, human-readable timing record for a pipeline stage."""
    return f"Stage '{stage}' took {seconds:.1f}s"


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
