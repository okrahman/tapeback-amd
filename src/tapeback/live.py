"""Live transcription — background thread that transcribes audio during recording."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from tapeback import const
from tapeback._gpu import free_gpu_memory
from tapeback._lazy import load_transcriber
from tapeback._lemonade import LemonadeAuthenticationError, LemonadeConfigurationError
from tapeback._live_chunks import BYTES_PER_SAMPLE, _ChunkTranscriber
from tapeback._live_pcm import (  # noqa: F401 — re-export; tests import from here
    adjust_timestamps,
    deduplicate_overlap,
    find_data_offset,
    resample_48k_to_16k,
)
from tapeback.channel import is_channel_active
from tapeback.formatter import format_live_markdown
from tapeback.models import Segment
from tapeback.settings import Settings
from tapeback.vault import save_live_markdown

if TYPE_CHECKING:
    from tapeback.transcriber import Transcriber


# Tolerance for deduplication: segments within this many seconds are considered duplicates


# Poll rather than joining indefinitely so a legitimate long model load,
# download, or local fallback remains visible to the caller.
_STOP_PROGRESS_INTERVAL_SECONDS = 10.0


# Bytes per sample for s16le mono


class LiveTranscriber(_ChunkTranscriber):
    """Background transcription thread that runs during recording.

    Periodically reads new audio from growing WAV files written by parecord,
    transcribes both channels (mic -> "You", monitor -> "Other"),
    and writes a live markdown transcript to the Obsidian vault.
    """

    def __init__(
        self,
        settings: Settings,
        session_name: str,
        mic_path: Path,
        monitor_path: Path,
    ) -> None:
        self._settings = settings
        self._session_name = session_name
        self._mic_path = mic_path
        self._monitor_path = monitor_path

        self._mic_data_offset: int | None = None  # parsed lazily on first read
        self._monitor_data_offset: int | None = None
        self._mic_byte_offset = 0  # bytes of PCM data already processed
        self._monitor_byte_offset = 0
        self._segments: list[Segment] = []
        self._active_backend_fingerprint: str | None = None
        self._fatal_error: Exception | None = None
        self._last_detected_language: str | None = None

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._transcription_loop,
            name="live-transcriber",
            daemon=True,
        )

        self._live_md_path = (
            settings.vault_path
            / settings.meetings_dir
            / f"{session_name}{const.FILE_LIVE_SUFFIX}.md"
        )

        # Transcriber is created lazily on the first chunk to avoid blocking
        # the main thread with model loading
        self._transcriber: Transcriber | None = None

    @property
    def live_md_path(self) -> Path:
        return self._live_md_path

    def start(self) -> None:
        """Start the background transcription thread."""
        self._thread.start()

    def stop(self, on_status: Callable[[str], None] | None = None) -> None:
        """Stop the background thread, process final chunk, free GPU memory.

        Establishes a hard lifecycle boundary: when this returns, the worker is
        verifiably dead — it can issue no further request and write no live note
        afterwards. Model construction, downloads, isolated-worker startup, and
        faster-whisper fallback are not bounded by the Lemonade HTTP timeout, so
        legitimate work is awaited to completion and periodically reported.
        """
        self._stop_event.set()
        started_waiting = time.monotonic()
        while self._thread.is_alive():
            self._thread.join(timeout=_STOP_PROGRESS_INTERVAL_SECONDS)
            if self._thread.is_alive() and on_status is not None:
                elapsed = time.monotonic() - started_waiting
                on_status(f"Still waiting for live transcription ({elapsed:.0f}s elapsed)...")

        # Free GPU memory so the full pipeline can use it
        if self._transcriber is not None:
            del self._transcriber
            self._transcriber = None
            free_gpu_memory()

        if self._fatal_error is not None:
            raise self._fatal_error

    def _ensure_transcriber(self) -> Transcriber:
        """Lazily create the Transcriber (loads Whisper model)."""
        if self._transcriber is None:
            self._transcriber = load_transcriber(self._settings)
            # Match the post-recording pipeline, which reports the backend through
            # the status callback: live transcription must disclose which backend
            # is active — and, for Lemonade, that audio leaves this machine — the
            # same way once, when the backend is actually loaded.
            print(f"Live transcription backend: {self._transcriber.describe()}", file=sys.stderr)
        return self._transcriber

    def _report_status(self, message: str) -> None:
        """Status sink passed to every facade call the worker makes.

        The facade reports backend transitions through it — above all the
        fallback notice ("Lemonade transcription failed ... falling back to
        faster-whisper"). Without a real sink those notices go to the default
        no-op and a mid-session backend change happens in silence, exactly when
        performance, resource use, and privacy expectations change.
        """
        print(message, file=sys.stderr)

    def _transcription_loop(self) -> None:
        """Main loop: wait for interval, then process a chunk."""
        # Write initial "waiting" markdown
        self._write_live_markdown()

        while not self._stop_event.wait(timeout=self._settings.live_interval):
            try:
                self._process_chunk()
            except (LemonadeAuthenticationError, LemonadeConfigurationError) as exc:
                self._fatal_error = exc
                import traceback  # noqa: PLC0415 — only on error

                print(
                    f"Error: Terminal live transcription error:\n{traceback.format_exc()}",
                    file=sys.stderr,
                )
                break
            except Exception:
                import traceback  # noqa: PLC0415 — only on error

                print(
                    f"Warning: Live transcription error:\n{traceback.format_exc()}",
                    file=sys.stderr,
                )

        # Process final chunk on stop only if no terminal fatal error occurred
        if self._fatal_error is None:
            try:
                self._process_chunk(is_final=True)
            except (LemonadeAuthenticationError, LemonadeConfigurationError) as exc:
                self._fatal_error = exc
                import traceback  # noqa: PLC0415 — only on error

                print(
                    "Error: Terminal live transcription final chunk error:\n"
                    f"{traceback.format_exc()}",
                    file=sys.stderr,
                )
            except Exception:
                import traceback  # noqa: PLC0415 — only on error

                print(
                    f"Warning: Live transcription final chunk error:\n{traceback.format_exc()}",
                    file=sys.stderr,
                )

    def _process_chunk(self, *, is_final: bool = False) -> None:
        """Read new audio from both channels, transcribe, update markdown.

        Both channels of one interval are transcribed as ONE backend transaction
        (monitor first, see `_transcribe_pair`), so a Lemonade fallback triggered by
        the second channel can never mix a Lemonade first channel with a faster-whisper
        second channel in one interval. Byte cursors and accumulated segments are
        committed atomically: a raise while transcribing leaves both cursors in place,
        so the next cycle re-reads the same audio and recovers the interval instead of
        silently dropping it.
        """
        min_bytes = (
            0
            if is_final
            else int(self._settings.live_min_chunk * self._settings.sample_rate * BYTES_PER_SAMPLE)
        )
        overlap_bytes = int(
            self._settings.live_overlap * self._settings.sample_rate * BYTES_PER_SAMPLE
        )

        mic_pcm, mic_new_offset = self._read_new_pcm(
            self._mic_path,
            self._mic_byte_offset,
            min_bytes,
            overlap_bytes,
            is_mic=True,
        )
        monitor_pcm, monitor_new_offset = self._read_new_pcm(
            self._monitor_path,
            self._monitor_byte_offset,
            min_bytes,
            overlap_bytes,
            is_mic=False,
        )

        if mic_pcm is None and monitor_pcm is None:
            return

        # Inspect the exact samples that would be submitted after live resampling
        # before loading a backend or creating temporary WAVs. Even when both sides
        # are silent, commit the cursors so the same interval is not reconsidered.
        mic_active = mic_pcm is not None and is_channel_active(resample_48k_to_16k(mic_pcm))
        monitor_active = monitor_pcm is not None and is_channel_active(
            resample_48k_to_16k(monitor_pcm)
        )
        if not mic_active and not monitor_active:
            self._mic_byte_offset = mic_new_offset
            self._monitor_byte_offset = monitor_new_offset
            return

        transcriber = self._ensure_transcriber()
        mic_segments: list[Segment] = []
        monitor_segments: list[Segment] = []
        staging_segments = list(self._segments)
        partial = False

        if mic_pcm is not None and monitor_pcm is not None:
            mic_segments, monitor_segments, partial = self._transcribe_pair(
                transcriber,
                mic_pcm,
                monitor_pcm,
                overlap_bytes,
                mic_active=mic_active,
                monitor_active=monitor_active,
                existing_segments=staging_segments,
            )
        elif mic_pcm is not None:
            mic_segments, partial = self._transcribe_chunk(
                transcriber,
                mic_pcm,
                self._mic_byte_offset,
                overlap_bytes,
                is_mic=True,
                existing_segments=staging_segments,
            )
        elif monitor_pcm is not None:
            monitor_segments, partial = self._transcribe_chunk(
                transcriber,
                monitor_pcm,
                self._monitor_byte_offset,
                overlap_bytes,
                is_mic=False,
                existing_segments=staging_segments,
            )

        current_fp = transcriber._backend.cache_fingerprint()
        if partial:
            # A partial result covers only the decoded prefix of the submitted
            # interval (interrupt semantics, see FasterWhisperBackend._collect_segments).
            # Committing it would advance the cursors past the undecoded suffix and
            # lose that audio forever, so commit nothing: segments, cursors and the
            # backend fingerprint all stay as they were, and the next cycle retries
            # the whole interval — including any backend-switch repair it triggers.
            if is_final:
                # There is no next cycle after the final pass, so say what happened
                # instead of dropping the interval silently. Only the live preview's
                # tail is affected: the post-recording pipeline transcribes the
                # whole file.
                self._report_status(
                    "The final live interval was only partially decoded — its "
                    "undecoded tail is absent from the live note. The full "
                    "transcript produced after recording is not affected."
                )
            return
        if (
            self._active_backend_fingerprint is not None
            and current_fp != self._active_backend_fingerprint
            and (self._mic_byte_offset > 0 or self._monitor_byte_offset > 0)
        ):
            # Backend switch occurred mid-session: re-transcribe all committed audio
            # from offset 0 with the new backend so the transcript is never mixed.
            # Committed audio, not emitted segments, is the state boundary: a prior
            # decoder may have returned silence for audio the fallback can recognize.
            # Re-announce the backend first: the disclosure printed at startup
            # described the backend that just failed, and the user must see the
            # transition (which model, which device, audio local or not) before
            # the re-transcription runs on it.
            print(
                f"Live transcription backend switched: {transcriber.describe()}",
                file=sys.stderr,
            )
            updated_segments = self._retranscribe_full(
                transcriber, mic_new_offset, monitor_new_offset
            )
        elif mic_segments or monitor_segments or staging_segments != self._segments:
            updated_segments = staging_segments + mic_segments + monitor_segments
            updated_segments.sort(key=lambda s: s.start)
        else:
            updated_segments = self._segments

        # Write markdown before committing in-memory segments and cursors, so a
        # write failure (e.g. disk full, permission error) leaves cursors in place
        # without duplicating segments upon retry.
        self._write_live_markdown(updated_segments)
        self._segments = updated_segments
        self._active_backend_fingerprint = current_fp

        # Commit both cursors only once the whole interval succeeded — never between
        # the two channels, or an error in the second would skip the first's audio
        # forever.
        self._mic_byte_offset = mic_new_offset
        self._monitor_byte_offset = monitor_new_offset

    def _write_live_markdown(self, segments: list[Segment] | None = None) -> None:
        """Write (or overwrite) the live markdown file in the vault."""
        target_segments = self._segments if segments is None else segments
        markdown = format_live_markdown(
            target_segments,
            self._session_name,
            self._settings.language,
        )
        save_live_markdown(markdown, self._settings, self._session_name)
