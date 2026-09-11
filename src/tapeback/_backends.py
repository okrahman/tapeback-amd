"""The minimal contract every transcription backend implements.

`Transcriber` is the shared façade callers see; it owns resume lookup, transactional
storage, mono/stereo behaviour and speaker labelling. A backend owns one way of
turning a WAV file into segments, plus the identity of the settings that would make
it produce different output (`cache_fingerprint`).

Deliberately minimal: two backends exist today (faster-whisper in-process/isolated,
and Lemonade over HTTP), and anything the façade cannot express through these three
methods is a sign the split is wrong, not a sign the protocol needs more methods.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from tapeback.models import Segment

# Same shape as pipeline.StatusCallback; a local alias avoids importing pipeline
# (which imports this module's consumers) from a low-level module.
StatusCallback = Callable[[str], None]

# The info dict returned alongside segments. `language_probability` is optional:
# Lemonade supplies one only when it actually detected the language, and inventing
# a value would be worse than the field being absent.
TranscriptionInfo = dict[str, str | float | bool]


@runtime_checkable
class TranscriptionBackend(Protocol):
    """One interchangeable way of transcribing a WAV file."""

    def transcribe(
        self,
        audio_path: Path,
        *,
        stage: str = "transcribe",
        on_status: StatusCallback = lambda _message: None,
        language_override: str | None = None,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        """Transcribe one audio file. Interrupts return a partial result, not raise."""
        ...

    def describe(self) -> str:
        """Human-readable one-liner: what this backend is and where it runs."""
        ...

    def cache_fingerprint(self) -> str:
        """Identity of every setting that would change this backend's output.

        A cached channel is only reusable when the fingerprint matches, so anything
        that affects output belongs in here — and anything that does not (timeouts,
        credentials, diagnostics) must stay out, or users lose their cache for no
        reason.
        """
        ...
