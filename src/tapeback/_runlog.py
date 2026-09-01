"""Per-run metadata — the post-mortem record for a processing run.

Until this existed a run left nothing behind: a recording that failed or was
interrupted was indistinguishable from one that was never started, and there was
no way to reconstruct which settings produced a given transcript. Each run now
writes one JSON file with the configuration it ran under, everything it printed,
and how it ended — including when it ended badly, which is the case that matters.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tapeback.settings import Settings

Reporter = Callable[[str], None]

# Outcomes a run can end with. "aborted" is Ctrl+C, "failed" is an exception.
# "unknown" is the honest default for an exit neither branch classified (SystemExit,
# a killed process) — better than recording a run as completed when it wasn't.
OUTCOME_UNKNOWN = "unknown"
OUTCOME_COMPLETED = "completed"
OUTCOME_ABORTED = "aborted"
OUTCOME_FAILED = "failed"

# Settings copied into the run record. This is an explicit allow-list, never
# settings.model_dump(): Settings carries hf_token and llm_api_key, and a
# post-mortem file that leaks credentials is worse than no post-mortem file.
RECORDED_SETTINGS = (
    "whisper_model",
    "language",
    "device",
    "compute_type",
    "beam_size",
    "batch_size",
    "chunk_length",
    "vad_filter",
    "no_speech_threshold",
    "condition_on_previous_text",
    "multilingual",
    "language_detection_segments",
    "hallucination_silence_threshold",
    "gate_mic_silence",
    "diarize",
    "pause_threshold",
)

# Keep the directory from growing without bound; oldest records are dropped first.
MAX_RUN_RECORDS = 200

# Terminal-control characters (C0, C1, and line/paragraph separators) never belong
# in a persisted diagnostic line: ESC sequences could disguise or corrupt output
# when the file is catted into a terminal, and they have no legitimate use here.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")


_REDACTED_LABEL = "[redacted]"


def redact_text(text: str, redactions: tuple[str, ...] = ()) -> str:
    """Strip terminal-control characters and replace configured secrets.

    Applied to everything captured into the run record: status lines are echoed
    by remote-facing backends and error messages are influenced by remote error
    bodies, so a reflected credential must not survive the write even if an
    upstream sanitizer missed it.
    """
    out = _CONTROL_CHARS_RE.sub("", text)
    for secret in redactions:
        if secret:
            out = out.replace(secret, _REDACTED_LABEL)
    return out


def default_run_log_dir() -> Path:
    """XDG data directory for run records."""
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return base / "tapeback" / "runs"


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _config_snapshot(settings: Settings) -> dict[str, object]:
    """Copy the transcription-relevant settings, and only those."""
    return {name: getattr(settings, name) for name in RECORDED_SETTINGS}


@dataclass
class RunLog:
    """Accumulates what a run did, for writing out when it finishes."""

    session: str
    started_at: str
    config: dict[str, object]
    events: list[str] = field(default_factory=list)
    outcome: str = OUTCOME_UNKNOWN
    error: str | None = None
    finished_at: str | None = None
    # Secret strings replaced with "[redacted]" in every captured line and in the
    # error field. Populated by `run_log` from the settings; empty when constructed
    # directly (tests). Status text reaches this file from remote-facing backends,
    # and a reflected credential must not survive the write even if an upstream
    # sanitizer missed it.
    redactions: tuple[str, ...] = ()

    def _sanitize(self, message: str) -> str:
        """Redact configured secrets and strip terminal-control characters."""
        return redact_text(message, self.redactions)

    def record(self, message: str) -> None:
        """Capture one status line, redacted of configured secrets.

        The status lines already are the human-readable record of the run, so they
        are stored as written rather than parsed back into fields — re-parsing our
        own formatted output would break every time a message is reworded.
        """
        self.events.append(self._sanitize(message))

    def to_dict(self) -> dict[str, object]:
        return {
            "session": self.session,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outcome": self.outcome,
            "error": self.error,
            "config": {key: _jsonable(value) for key, value in self.config.items()},
            "events": self.events,
        }


def _jsonable(value: object) -> object:
    """Render a settings value as JSON-safe (Path and tuple are the cases that occur)."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _prune_old_records(directory: Path, keep: int = MAX_RUN_RECORDS) -> None:
    """Drop the oldest records so the directory stays bounded.

    Names start with the run's UTC timestamp, so sorting by name sorts by age.
    """
    records = sorted(directory.glob("*.json"), key=lambda p: p.name)
    if len(records) <= keep:
        return
    for stale in records[: len(records) - keep]:
        stale.unlink(missing_ok=True)


def write_run_log(record: RunLog, directory: Path) -> Path | None:
    """Write the record as JSON. Returns the path, or None if it could not be written.

    Failing to write a diagnostic file must never destroy the run that produced it,
    so every filesystem error here is swallowed deliberately — the transcript has
    already been saved by this point.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Timestamp in the name keeps repeated runs of one session distinguishable
        # and makes lexicographic order equal chronological order for pruning.
        stamp = record.started_at.replace(":", "-")
        path = directory / f"{stamp}_{record.session}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))
        _prune_old_records(directory)
    except OSError:
        return None
    return path


@contextmanager
def run_log(
    session: str,
    settings: Settings,
    on_status: Reporter,
    *,
    directory: Path | None = None,
) -> Iterator[Reporter]:
    """Record a processing run, yielding a reporter that both prints and captures.

    Writes the record on the way out whichever way the block ends. A run that was
    interrupted or failed is exactly the run worth having a record of, so the
    outcome is classified rather than the exception being swallowed — it is
    re-raised unchanged.

    Every captured status line and the error field are redacted of the
    credentials the settings carry (`lemonade_api_key`, `hf_token`,
    `llm_api_key`): a remote-facing error can reflect a configured secret, and a
    post-mortem file the user may share must never hold a reusable credential.
    """
    if not settings.run_log:
        yield on_status
        return

    target = directory or settings.run_log_dir or default_run_log_dir()
    record = RunLog(
        session=session,
        started_at=_utc_now_iso(),
        config=_config_snapshot(settings),
        redactions=tuple(
            value
            for value in (
                settings.lemonade_api_key.get_secret_value(),
                settings.hf_token.get_secret_value(),
                settings.llm_api_key.get_secret_value(),
            )
            if value
        ),
    )

    def reporter(message: str) -> None:
        on_status(message)
        record.record(message)

    try:
        yield reporter
    except KeyboardInterrupt:
        record.outcome = OUTCOME_ABORTED
        raise
    except Exception as exc:
        record.outcome = OUTCOME_FAILED
        # Type and message only — a full traceback in a file the user may share
        # can carry local paths, and the message is what identifies the failure.
        # The message is remote-influenced text, so it is redacted and stripped
        # of control characters at this, the final persistence boundary.
        record.error = redact_text(f"{type(exc).__name__}: {exc}", record.redactions)
        raise
    else:
        record.outcome = OUTCOME_COMPLETED
    finally:
        record.finished_at = _utc_now_iso()
        write_run_log(record, target)


__all__ = [
    "MAX_RUN_RECORDS",
    "OUTCOME_ABORTED",
    "OUTCOME_COMPLETED",
    "OUTCOME_FAILED",
    "OUTCOME_UNKNOWN",
    "RECORDED_SETTINGS",
    "RunLog",
    "default_run_log_dir",
    "run_log",
    "write_run_log",
]
