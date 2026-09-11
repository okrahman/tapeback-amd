"""Interval transcription for live mode: reading new PCM from the growing
recordings, submitting one mic/monitor interval as a single backend
transaction, and the stop-time full-audio tail. Lives in a base class so
`LiveTranscriber` keeps session state and the commit/retry logic while the
chunk plumbing stays auditable in one place."""

from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tapeback import const
from tapeback._live_pcm import (
    adjust_timestamps,
    deduplicate_overlap,
    find_data_offset,
    resample_48k_to_16k,
)
from tapeback.channel import is_channel_active
from tapeback.models import Segment
from tapeback.settings import Settings

if TYPE_CHECKING:
    from tapeback.transcriber import Transcriber

# A live chunk is raw s16le PCM: two bytes per sample.
BYTES_PER_SAMPLE = 2

# Give up waiting for a parseable header once the file exceeds this size: a
# recording this large with a still-unparseable header is a genuinely weird
# writer, and a live preview at the standard 44-byte offset (announced as an
# assumption) beats no preview at all. Below this size an unparseable header is
# treated as "not flushed yet" and retried next cycle.
HEADER_PARSE_FALLBACK_BYTES = 1 << 20


class _ChunkTranscriber:
    """Chunk/interval transcription machinery mixed into `LiveTranscriber`.

    Attribute declarations exist for the type checker: the runtime state is
    created (and typed) by `LiveTranscriber.__init__`, which is the only class
    that instantiates this one.
    """

    _settings: Settings
    _segments: list[Segment]
    _mic_path: Path
    _monitor_path: Path
    _mic_byte_offset: int
    _monitor_byte_offset: int
    _report_status: Callable[[str], None]

    def _resolve_data_offset(self, wav_path: Path, *, is_mic: bool) -> int | None:
        """The channel's data offset, or None while the header is still unreadable.

        None means "retry next cycle". Only once the file has grown past
        `HEADER_PARSE_FALLBACK_BYTES` with no parseable header do we assume the
        standard 44-byte offset — loudly, because it is then a guess about a
        genuinely unusual writer, not a race with a header that is about to be
        flushed.
        """
        offset = find_data_offset(wav_path)
        if offset is not None:
            return offset
        try:
            size = wav_path.stat().st_size
        except OSError:
            return None
        if size < HEADER_PARSE_FALLBACK_BYTES:
            return None
        channel = "mic" if is_mic else "monitor"
        self._report_status(
            f"Live {channel} WAV header is still unparseable at {size} bytes — "
            f"assuming the standard {const.WAV_HEADER_FALLBACK}-byte PCM offset. "
            "Live timestamps may drift if the real data chunk is not at that offset."
        )
        return const.WAV_HEADER_FALLBACK

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

        # Parse the data offset lazily — and never latch a guess: a truncated
        # header on an early poll returns None and is retried next cycle, so a
        # first-read race with the recorder's header flush cannot misalign every
        # later offset for the session.
        if is_mic:
            if self._mic_data_offset is None:
                self._mic_data_offset = self._resolve_data_offset(wav_path, is_mic=True)
            data_offset = self._mic_data_offset
        else:
            if self._monitor_data_offset is None:
                self._monitor_data_offset = self._resolve_data_offset(wav_path, is_mic=False)
            data_offset = self._monitor_data_offset
        if data_offset is None:
            return None, byte_offset

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
    ) -> tuple[list[Segment], bool]:
        """Resample, write temp WAV, transcribe, adjust timestamps, deduplicate.

        Returns (segments, partial). partial is True when the backend decoded only
        a prefix of the submitted interval (interrupt semantics) — the caller must
        then leave its cursors alone so the undecoded suffix is retried.
        """
        samples_16k = resample_48k_to_16k(pcm_bytes)
        if len(samples_16k) == 0 or not is_channel_active(samples_16k):
            return [], False

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
                on_status=self._report_status,
            )
            partial = bool(_info.get("partial"))
            if _info.get("language"):
                # Latch the detected/pinned language from either channel. A
                # mic-only session (the monitor never produces PCM) otherwise
                # re-auto-detects every interval and the live note can flip
                # language mid-session. When an override was applied,
                # info["language"] repeats it and this is a no-op.
                self._last_detected_language = str(_info["language"])
        finally:
            chunk_path.unlink(missing_ok=True)

        return (
            self._finalize_segments(
                segments,
                byte_offset,
                overlap_bytes,
                is_mic=is_mic,
                existing_segments=existing_segments,
            ),
            partial,
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
    ) -> tuple[list[Segment], list[Segment], bool]:
        """Transcribe one mic/monitor pair as ONE backend transaction.

        The monitor channel goes first so its detected language is reused for the
        gated mic — the mic is near silence while the user listens, so auto-detection
        has almost nothing to work from. The facade treats the active channels
        transactionally: when both are active, a Lemonade fallback on either retries
        BOTH through faster-whisper, so one interval can never mix one Lemonade channel
        with one faster-whisper channel. With one active channel, only that channel
        falls back. Active temp WAVs are ephemeral, so resume IO is disabled for the pair.

        Returns (mic_segments, monitor_segments, partial), where partial is True when
        either active channel decoded only a prefix of its interval — the caller must
        then leave both cursors alone so the undecoded suffix is retried.
        """
        mic_samples_16k = resample_48k_to_16k(mic_pcm)
        monitor_samples_16k = resample_48k_to_16k(monitor_pcm)
        if mic_active is None:
            mic_active = is_channel_active(mic_samples_16k)
        if monitor_active is None:
            monitor_active = is_channel_active(monitor_samples_16k)
        if not mic_active and not monitor_active:
            return [], [], False

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
                on_status=self._report_status,
            )
            partial = bool(_info.get("partial"))
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
        return mic_segments, monitor_segments, partial

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
                self._mic_data_offset = self._resolve_data_offset(wav_path, is_mic=True)
            data_offset = self._mic_data_offset
        else:
            if self._monitor_data_offset is None:
                self._monitor_data_offset = self._resolve_data_offset(wav_path, is_mic=False)
            data_offset = self._monitor_data_offset
        if data_offset is None:
            return None
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
                on_status=self._report_status,
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
            segments, _info = transcriber.transcribe(
                chunk_path, use_resume=False, on_status=self._report_status
            )
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
