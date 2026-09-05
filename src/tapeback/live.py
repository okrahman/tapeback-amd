"""Live transcription — background thread that transcribes audio during recording."""

from __future__ import annotations

import struct
import sys
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tapeback import const
from tapeback._gpu import free_gpu_memory
from tapeback._lazy import load_transcriber
from tapeback._lemonade import (
    LemonadeAuthenticationError,
    LemonadeConfigurationError,
    _utterance_tokens,
)
from tapeback.channel import is_channel_active
from tapeback.formatter import format_live_markdown
from tapeback.models import Segment, Word
from tapeback.settings import Settings
from tapeback.vault import save_live_markdown

if TYPE_CHECKING:
    from tapeback.transcriber import Transcriber

# Tolerance for deduplication: segments within this many seconds are considered duplicates
DEDUP_TOLERANCE_SEC = 0.5

# Poll rather than joining indefinitely so a legitimate long model load,
# download, or local fallback remains visible to the caller.
_STOP_PROGRESS_INTERVAL_SECONDS = 10.0

# Bytes per sample for s16le mono
BYTES_PER_SAMPLE = 2


def find_data_offset(path: Path) -> int:
    """Find the byte offset where PCM data starts in a WAV file.

    Scans RIFF chunks to locate the 'data' chunk. Returns the byte position
    immediately after the data chunk header (i.e. where raw PCM bytes begin).

    Falls back to the standard 44-byte offset if parsing fails.
    """
    try:
        with open(path, "rb") as f:
            riff = f.read(4)
            if riff != b"RIFF":
                return const.WAV_HEADER_FALLBACK
            f.read(4)  # file size (unreliable for growing files)
            wave_id = f.read(4)
            if wave_id != b"WAVE":
                return const.WAV_HEADER_FALLBACK
            # Scan sub-chunks until we find "data"
            while True:
                chunk_id = f.read(const.WAV_CHUNK_HEADER_BYTES)
                if len(chunk_id) < const.WAV_CHUNK_HEADER_BYTES:
                    return const.WAV_HEADER_FALLBACK
                chunk_size_bytes = f.read(const.WAV_CHUNK_HEADER_BYTES)
                if len(chunk_size_bytes) < const.WAV_CHUNK_HEADER_BYTES:
                    return const.WAV_HEADER_FALLBACK
                if chunk_id == b"data":
                    return f.tell()
                (chunk_size,) = struct.unpack("<I", chunk_size_bytes)
                f.seek(chunk_size, 1)
    except OSError:
        return const.WAV_HEADER_FALLBACK


def resample_48k_to_16k(pcm_bytes: bytes) -> np.ndarray:
    """Downsample raw s16le PCM from 48 kHz to 16 kHz.

    Simple decimation by factor 3 (no anti-aliasing filter).
    Adequate quality for a live preview — the final pipeline uses ffmpeg with loudnorm.
    """
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return samples[:: const.RESAMPLE_FACTOR]


def adjust_timestamps(segments: list[Segment], offset_seconds: float) -> list[Segment]:
    """Shift all segment and word timestamps by offset_seconds."""
    result: list[Segment] = []
    for seg in segments:
        words: list[Word] | None = None
        if seg.words:
            words = [
                Word(
                    start=w.start + offset_seconds,
                    end=w.end + offset_seconds,
                    word=w.word,
                    probability=w.probability,
                )
                for w in seg.words
            ]
        result.append(
            Segment(
                start=seg.start + offset_seconds,
                end=seg.end + offset_seconds,
                text=seg.text,
                words=words,
                speaker=seg.speaker,
            )
        )
    return result


def _same_utterance(left: Segment, right: Segment) -> bool:
    """Whether overlap candidates say the same thing, allowing token prefixes or suffixes."""
    left_tokens = _utterance_tokens(left.text)
    right_tokens = _utterance_tokens(right.text)
    if not left_tokens or not right_tokens:
        return False
    short, long = (
        (left_tokens, right_tokens)
        if len(left_tokens) <= len(right_tokens)
        else (right_tokens, left_tokens)
    )
    return long[: len(short)] == short or long[-len(short) :] == short


def deduplicate_overlap(
    existing: list[Segment],
    new_segments: list[Segment],
    overlap_start: float,
) -> list[Segment]:
    """Remove segments from new_segments that duplicate existing ones in the overlap zone.

    A new segment is considered a duplicate if its start time is within
    DEDUP_TOLERANCE_SEC of any existing segment of the same speaker's start time,
    it falls within the overlap region (before overlap_start), AND both segments
    match as the same utterance.

    If a duplicate candidate in the overlap zone extends past the overlap boundary
    or has longer text than the matched existing segment, it updates/replaces the
    existing segment in-place rather than being unconditionally discarded.
    """
    if not existing or overlap_start <= 0:
        return new_segments

    kept: list[Segment] = []
    replaced = False
    for seg in new_segments:
        # Segments starting at or past the overlap zone — always keep
        if seg.start >= overlap_start:
            kept.append(seg)
            continue

        # Check if this segment duplicates an existing one of the same speaker
        best_match_idx: int | None = None
        best_diff = DEDUP_TOLERANCE_SEC
        for i, es in enumerate(existing):
            if es.speaker == seg.speaker:
                diff = abs(seg.start - es.start)
                if diff < best_diff and _same_utterance(es, seg):
                    best_diff = diff
                    best_match_idx = i

        if best_match_idx is None:
            kept.append(seg)
        else:
            # Reconcile duplicate candidates: if candidate extends past the
            # overlap boundary or has longer text, update/replace the existing
            # segment (mirroring _MergeState._prefer).
            existing_seg = existing[best_match_idx]
            extends_past_boundary = seg.end > overlap_start and seg.end > existing_seg.end
            has_longer_text = len(seg.text.strip()) > len(existing_seg.text.strip())
            if extends_past_boundary or has_longer_text:
                existing[best_match_idx] = seg
                replaced = True

    if replaced:
        existing.sort(key=lambda s: s.start)

    return kept


class LiveTranscriber:
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

        if mic_pcm is not None and monitor_pcm is not None:
            mic_segments, monitor_segments = self._transcribe_pair(
                transcriber,
                mic_pcm,
                monitor_pcm,
                overlap_bytes,
                mic_active=mic_active,
                monitor_active=monitor_active,
                existing_segments=staging_segments,
            )
        elif mic_pcm is not None:
            mic_segments = self._transcribe_chunk(
                transcriber,
                mic_pcm,
                self._mic_byte_offset,
                overlap_bytes,
                is_mic=True,
                existing_segments=staging_segments,
            )
        elif monitor_pcm is not None:
            monitor_segments = self._transcribe_chunk(
                transcriber,
                monitor_pcm,
                self._monitor_byte_offset,
                overlap_bytes,
                is_mic=False,
                existing_segments=staging_segments,
            )

        current_fp = transcriber._backend.cache_fingerprint()
        if (
            self._active_backend_fingerprint is not None
            and current_fp != self._active_backend_fingerprint
            and (self._mic_byte_offset > 0 or self._monitor_byte_offset > 0)
        ):
            # Backend switch occurred mid-session: re-transcribe all committed audio
            # from offset 0 with the new backend so the transcript is never mixed.
            # Committed audio, not emitted segments, is the state boundary: a prior
            # decoder may have returned silence for audio the fallback can recognize.
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

    def _read_new_pcm(
        self,
        wav_path: Path,
        byte_offset: int,
        min_bytes: int,
        overlap_bytes: int,
        *,
        is_mic: bool,
    ) -> tuple[bytes | None, int]:
        """Read new raw PCM bytes from a growing WAV file.

        Returns (pcm_bytes_including_overlap, new_byte_offset) or (None, byte_offset)
        if not enough new data.
        """
        if not wav_path.exists():
            return None, byte_offset

        # Parse data offset lazily (once per file)
        if is_mic:
            if self._mic_data_offset is None:
                self._mic_data_offset = find_data_offset(wav_path)
            data_offset = self._mic_data_offset
        else:
            if self._monitor_data_offset is None:
                self._monitor_data_offset = find_data_offset(wav_path)
            data_offset = self._monitor_data_offset

        file_size = wav_path.stat().st_size
        available_pcm = file_size - data_offset
        if available_pcm <= 0:
            return None, byte_offset
        # Ensure available_pcm is aligned to sample boundaries (even number of bytes)
        available_pcm -= available_pcm % BYTES_PER_SAMPLE
        new_bytes = available_pcm - byte_offset

        if new_bytes < max(BYTES_PER_SAMPLE, min_bytes):
            return None, byte_offset

        # Include overlap from previous chunk
        read_start = max(0, byte_offset - overlap_bytes)
        read_start -= read_start % BYTES_PER_SAMPLE
        read_length = available_pcm - read_start

        with open(wav_path, "rb") as f:
            f.seek(data_offset + read_start)
            pcm_bytes = f.read(read_length)

        # Ensure even number of bytes (s16le = 2 bytes per sample)
        if len(pcm_bytes) % BYTES_PER_SAMPLE != 0:
            pcm_bytes = pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % BYTES_PER_SAMPLE)]

        new_offset = available_pcm
        return pcm_bytes, new_offset

    def _transcribe_chunk(
        self,
        transcriber: Transcriber,
        pcm_bytes: bytes,
        byte_offset: int,
        overlap_bytes: int,
        *,
        is_mic: bool,
        existing_segments: list[Segment] | None = None,
    ) -> list[Segment]:
        """Resample, write temp WAV, transcribe, adjust timestamps, deduplicate."""
        samples_16k = resample_48k_to_16k(pcm_bytes)
        if len(samples_16k) == 0 or not is_channel_active(samples_16k):
            return []

        # Write temp WAV for the backend
        suffix = "mic" if is_mic else "monitor"
        chunk_path = self._mic_path.parent / f"chunk_{suffix}.wav"
        self._write_chunk_wav(samples_16k, chunk_path)

        language_override = self._last_detected_language if is_mic else None
        try:
            # The temp WAV is ephemeral — it never outlives this call — so resume IO
            # is disabled: storing an entry for a file that is deleted before it can
            # ever be reused is pure waste (and risks key collisions across sessions).
            segments, _info = transcriber.transcribe(
                chunk_path,
                language_override=language_override,
                use_resume=False,
            )
            if not is_mic and _info.get("language"):
                self._last_detected_language = str(_info["language"])
        finally:
            chunk_path.unlink(missing_ok=True)

        return self._finalize_segments(
            segments,
            byte_offset,
            overlap_bytes,
            is_mic=is_mic,
            existing_segments=existing_segments,
        )

    def _transcribe_pair(  # noqa: PLR0913
        self,
        transcriber: Transcriber,
        mic_pcm: bytes,
        monitor_pcm: bytes,
        overlap_bytes: int,
        *,
        mic_active: bool | None = None,
        monitor_active: bool | None = None,
        existing_segments: list[Segment] | None = None,
    ) -> tuple[list[Segment], list[Segment]]:
        """Transcribe one mic/monitor pair as ONE backend transaction.

        The monitor channel goes first so its detected language is reused for the
        gated mic — the mic is near silence while the user listens, so auto-detection
        has almost nothing to work from. The facade treats the active channels
        transactionally: when both are active, a Lemonade fallback on either retries
        BOTH through faster-whisper, so one interval can never mix one Lemonade channel
        with one faster-whisper channel. With one active channel, only that channel
        falls back. Active temp WAVs are ephemeral, so resume IO is disabled for the pair.
        """
        mic_samples_16k = resample_48k_to_16k(mic_pcm)
        monitor_samples_16k = resample_48k_to_16k(monitor_pcm)
        if mic_active is None:
            mic_active = is_channel_active(mic_samples_16k)
        if monitor_active is None:
            monitor_active = is_channel_active(monitor_samples_16k)
        if not mic_active and not monitor_active:
            return [], []

        mic_path = self._mic_path.parent / "chunk_mic.wav"
        monitor_path = self._mic_path.parent / "chunk_monitor.wav"
        if mic_active:
            self._write_chunk_wav(mic_samples_16k, mic_path)
        if monitor_active:
            self._write_chunk_wav(monitor_samples_16k, monitor_path)

        # Keep a valid path for both façade arguments without writing an inactive
        # channel. The activity flags ensure the façade never opens the alias for
        # backend work, while it can still read the active channel's real duration.
        if not mic_active:
            mic_path = monitor_path
        if not monitor_active:
            monitor_path = mic_path

        try:
            mic_segments, monitor_segments, _info = transcriber.transcribe_stereo(
                mic_path,
                monitor_path,
                use_resume=False,
                skip_mic_on_monitor_partial=False,
                mic_active=mic_active,
                monitor_active=monitor_active,
            )
            if _info.get("language"):
                self._last_detected_language = str(_info["language"])
        finally:
            if mic_active:
                (self._mic_path.parent / "chunk_mic.wav").unlink(missing_ok=True)
            if monitor_active:
                (self._mic_path.parent / "chunk_monitor.wav").unlink(missing_ok=True)

        mic_segments = self._finalize_segments(
            mic_segments,
            self._mic_byte_offset,
            overlap_bytes,
            is_mic=True,
            existing_segments=existing_segments,
        )
        monitor_segments = self._finalize_segments(
            monitor_segments,
            self._monitor_byte_offset,
            overlap_bytes,
            is_mic=False,
            existing_segments=existing_segments,
        )
        return mic_segments, monitor_segments

    def _retranscribe_full(
        self,
        transcriber: Transcriber,
        mic_offset: int,
        monitor_offset: int,
    ) -> list[Segment]:
        """Re-transcribe all committed audio from the beginning using the current backend."""
        mic_pcm = self._read_pcm_range(self._mic_path, 0, mic_offset, is_mic=True)
        monitor_pcm = self._read_pcm_range(self._monitor_path, 0, monitor_offset, is_mic=False)
        segments: list[Segment] = []

        if mic_pcm is not None and monitor_pcm is not None:
            mic_segs, monitor_segs = self._transcribe_pair_audio(transcriber, mic_pcm, monitor_pcm)
            segments.extend(mic_segs)
            segments.extend(monitor_segs)
        elif mic_pcm is not None:
            segments.extend(self._transcribe_chunk_audio(transcriber, mic_pcm, is_mic=True))
        elif monitor_pcm is not None:
            segments.extend(self._transcribe_chunk_audio(transcriber, monitor_pcm, is_mic=False))

        segments.sort(key=lambda s: s.start)
        return segments

    def _read_pcm_range(
        self,
        wav_path: Path,
        start_byte: int,
        length_bytes: int,
        *,
        is_mic: bool,
    ) -> bytes | None:
        """Read a specific range of raw PCM bytes from a WAV file."""
        if not wav_path.exists() or length_bytes <= 0:
            return None
        start_byte -= start_byte % BYTES_PER_SAMPLE
        length_bytes -= length_bytes % BYTES_PER_SAMPLE
        if length_bytes <= 0:
            return None
        if is_mic:
            if self._mic_data_offset is None:
                self._mic_data_offset = find_data_offset(wav_path)
            data_offset = self._mic_data_offset
        else:
            if self._monitor_data_offset is None:
                self._monitor_data_offset = find_data_offset(wav_path)
            data_offset = self._monitor_data_offset
        with open(wav_path, "rb") as f:
            f.seek(data_offset + start_byte)
            pcm_bytes = f.read(length_bytes)
        if len(pcm_bytes) % BYTES_PER_SAMPLE != 0:
            pcm_bytes = pcm_bytes[: len(pcm_bytes) - (len(pcm_bytes) % BYTES_PER_SAMPLE)]
        return pcm_bytes if pcm_bytes else None

    def _transcribe_pair_audio(
        self,
        transcriber: Transcriber,
        mic_pcm: bytes,
        monitor_pcm: bytes,
    ) -> tuple[list[Segment], list[Segment]]:
        """Transcribe a full mic/monitor pair without time offsets or overlap dedup."""
        mic_samples_16k = resample_48k_to_16k(mic_pcm)
        monitor_samples_16k = resample_48k_to_16k(monitor_pcm)
        mic_active = is_channel_active(mic_samples_16k)
        monitor_active = is_channel_active(monitor_samples_16k)
        if not mic_active and not monitor_active:
            return [], []

        mic_path = self._mic_path.parent / "chunk_mic.wav"
        monitor_path = self._mic_path.parent / "chunk_monitor.wav"
        if mic_active:
            self._write_chunk_wav(mic_samples_16k, mic_path)
        if monitor_active:
            self._write_chunk_wav(monitor_samples_16k, monitor_path)
        if not mic_active:
            mic_path = monitor_path
        if not monitor_active:
            monitor_path = mic_path
        try:
            mic_segs, monitor_segs, _info = transcriber.transcribe_stereo(
                mic_path,
                monitor_path,
                use_resume=False,
                skip_mic_on_monitor_partial=False,
                mic_active=mic_active,
                monitor_active=monitor_active,
            )
        finally:
            if mic_active:
                (self._mic_path.parent / "chunk_mic.wav").unlink(missing_ok=True)
            if monitor_active:
                (self._mic_path.parent / "chunk_monitor.wav").unlink(missing_ok=True)

        mic_segs = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text,
                words=s.words,
                speaker=const.SPEAKER_YOU,
            )
            for s in mic_segs
        ]
        monitor_segs = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text,
                words=s.words,
                speaker=const.SPEAKER_OTHER,
            )
            for s in monitor_segs
        ]
        return mic_segs, monitor_segs

    def _transcribe_chunk_audio(
        self,
        transcriber: Transcriber,
        pcm_bytes: bytes,
        *,
        is_mic: bool,
    ) -> list[Segment]:
        """Transcribe full single-channel audio without time offsets or overlap dedup."""
        samples_16k = resample_48k_to_16k(pcm_bytes)
        if len(samples_16k) == 0 or not is_channel_active(samples_16k):
            return []
        suffix = "mic" if is_mic else "monitor"
        chunk_path = self._mic_path.parent / f"chunk_{suffix}.wav"
        self._write_chunk_wav(samples_16k, chunk_path)
        try:
            segments, _info = transcriber.transcribe(chunk_path, use_resume=False)
        finally:
            chunk_path.unlink(missing_ok=True)
        speaker = const.SPEAKER_YOU if is_mic else const.SPEAKER_OTHER
        return [
            Segment(start=s.start, end=s.end, text=s.text, words=s.words, speaker=speaker)
            for s in segments
        ]

    def _finalize_segments(
        self,
        segments: list[Segment],
        byte_offset: int,
        overlap_bytes: int,
        *,
        is_mic: bool,
        existing_segments: list[Segment] | None = None,
    ) -> list[Segment]:
        """Shift to absolute wall-clock time, assign the speaker, dedup the overlap.

        Shared by the single-channel and pair paths so one interval's two channels
        follow identical timeline/speaker/dedup rules.
        """
        # Calculate absolute time offset
        read_start = max(0, byte_offset - overlap_bytes)
        chunk_start_seconds = read_start / (self._settings.sample_rate * BYTES_PER_SAMPLE)

        # Adjust timestamps to absolute
        segments = adjust_timestamps(segments, chunk_start_seconds)

        # Assign speaker
        speaker = const.SPEAKER_YOU if is_mic else const.SPEAKER_OTHER
        segments = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text,
                words=s.words,
                speaker=speaker,
            )
            for s in segments
        ]

        # Deduplicate overlap with existing segments
        overlap_boundary = byte_offset / (self._settings.sample_rate * BYTES_PER_SAMPLE)
        target_existing = self._segments if existing_segments is None else existing_segments
        return deduplicate_overlap(target_existing, segments, overlap_boundary)

    @staticmethod
    def _write_chunk_wav(samples_16k: np.ndarray, path: Path) -> None:
        """Write a valid 16 kHz mono WAV file from int16 samples."""
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(const.SAMPLE_RATE_16K)
            wf.writeframes(samples_16k.tobytes())

    def _write_live_markdown(self, segments: list[Segment] | None = None) -> None:
        """Write (or overwrite) the live markdown file in the vault."""
        target_segments = self._segments if segments is None else segments
        markdown = format_live_markdown(
            target_segments,
            self._session_name,
            self._settings.language,
        )
        save_live_markdown(markdown, self._settings, self._session_name)
