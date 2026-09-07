"""Lemonade chunking and chunk-merge policy.

Owns how one recording WAV is split into bounded requests (with a small
contextual overlap) and how per-chunk responses merge back: timestamp
re-basing, subsumed-utterance purging, and the cumulative caps. The
chunk/dedup behaviour is versioned by `DEDUP_POLICY_VERSION`, which the
backend folds into its cache fingerprint so stale cached channels are
recomputed after a policy change."""

from __future__ import annotations

import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tapeback._lemonade_errors import (
    LemonadeCapabilityError,
    LemonadeConfigurationError,
    _utterance_tokens,
)
from tapeback._lemonade_language import _remote_language
from tapeback._lemonade_validate import (
    _BOUNDARY_EPSILON_SECONDS,
    _MAX_CUMULATIVE_SEGMENTS,
    _MAX_CUMULATIVE_TEXT_CHARS,
    _TIMESTAMP_SLACK_SECONDS,
    _finite_number,
    _require_segments,
)
from tapeback.models import Segment
from tapeback.settings import Settings

# Bumped whenever chunk/dedup behaviour changes what output a given WAV produces,
# so cached channel results from an older policy are never reused.
DEDUP_POLICY_VERSION = 3


# Lemonade can hallucinate speech from a long exact-digital-silence tail. For
# supported 16-bit PCM WAVs, retain this much of the silence already present after
# the final active frame. This is an upload policy only: the source remains intact.
TRAILING_SILENCE_PADDING_SECONDS = 0.5


# Cache identity for all audio preparation that happens before HTTP chunking.
# Include the padding value in the token so changing either the algorithm or the
# amount retained necessarily invalidates older Lemonade results.
AUDIO_PREPARATION_POLICY_VERSION = (
    f"pcm16-exact-silence-v1-padding-{TRAILING_SILENCE_PADDING_SECONDS!r}s"
)


# Fixed-size reverse reads keep the silence scan's peak memory independent of the
# recording length. Expressed in frames so multichannel frames are never split.
_SILENCE_SCAN_BLOCK_FRAMES = 64 * 1024


_PCM16_SAMPLE_WIDTH_BYTES = 2


# Conservative internal cap on one chunk's WAV payload. A 300 s mono 16 kHz PCM chunk
# is ~9.6 MB, so this binds only for unusual formats — it is a memory guard, not a
# claim about the server.
_MAX_CHUNK_BYTES = 24 * 1024 * 1024


# Fixed allowance reserved from the byte cap for the WAV container header and
# multipart framing, so framing can never push a request past the cap even at
# maximum overlap.
_REQUEST_OVERHEAD_BYTES = 64 * 1024


# A WAV whose chunk arithmetic yields more requests than this is treated as a
# misconfiguration (chunk duration far too small for the recording) rather than
# transcribed one sliver per request.
_MAX_CHUNKS = 1000


# The multipart part name under which audio is uploaded. Always opaque: the source
# filename is attacker-influencable in principle (POSIX names may contain quotes
# and newlines, which would be interpolated into a MIME header verbatim) and
# discloses local metadata the server has no use for. The content is always
# declared audio/wav, so the fixed name matches every request we send.
_UPLOAD_FILENAME = "audio.wav"


@dataclass(frozen=True)
class _Chunk:
    """One request's slice of the input WAV, in frame coordinates.

    ``core_start``/``core_end`` are the frames this chunk exists to cover;
    ``audio_start`` is where its audio actually begins — earlier than the core when
    overlap is prepended, so duplicate segments are cut against the core interval,
    not the audio interval.
    """

    index: int
    total: int
    audio_start: int
    core_start: int
    core_end: int


def _wav_params(audio_path: Path) -> tuple[int, int, int, int] | None:
    """(channels, sampwidth, framerate, n_frames) — None when not a readable WAV.

    A declared frame count that cannot fit in the actual file (a forged or
    inconsistent RIFF header) is also rejected: trusting it would plan one
    request per phantom frame.
    """
    try:
        with wave.open(str(audio_path), "rb") as wf:
            params = (
                wf.getnchannels(),
                wf.getsampwidth(),
                wf.getframerate(),
                wf.getnframes(),
            )
        declared_bytes = params[3] * max(1, params[0] * params[1])
        if declared_bytes > audio_path.stat().st_size:
            return None
        return params
    except (wave.Error, OSError):
        return None


def _lemonade_upload_frames(audio_path: Path, params: tuple[int, int, int, int]) -> int:
    """Return the PCM frame endpoint Lemonade should receive.

    Only 16-bit PCM WAVs participate. The file is scanned backwards in bounded
    blocks until a frame containing any nonzero channel sample is found. The
    returned endpoint retains existing silence for the configured padding and is
    capped at the original frame count. Zero means the supported input is entirely
    silent. Unsupported encodings retain their original endpoint unchanged.
    """
    channels, sampwidth, framerate, n_frames = params
    if sampwidth != _PCM16_SAMPLE_WIDTH_BYTES or framerate <= 0 or n_frames <= 0:
        return n_frames

    frame_bytes = channels * sampwidth
    try:
        with wave.open(str(audio_path), "rb") as wf:
            if wf.getcomptype() != "NONE":
                return n_frames
            block_end = n_frames
            while block_end > 0:
                block_start = max(0, block_end - _SILENCE_SCAN_BLOCK_FRAMES)
                wf.setpos(block_start)
                frames = wf.readframes(block_end - block_start)
                actual_frames = len(frames) // frame_bytes
                for relative in range(actual_frames - 1, -1, -1):
                    start = relative * frame_bytes
                    if any(frames[start : start + frame_bytes]):
                        last_active = block_start + relative
                        padding = int(TRAILING_SILENCE_PADDING_SECONDS * framerate)
                        return min(n_frames, last_active + 1 + padding)
                block_end = block_start
    except (wave.Error, OSError):
        # `_wav_params` already established readability. If the second open/read
        # races with a file change, preserve the established whole-input behavior.
        return n_frames
    return 0


def _wrap_wav(frames: bytes, channels: int, sampwidth: int, framerate: int) -> bytes:
    """Wrap raw PCM frames in a minimal RIFF/WAVE header.

    Chunk slices read out of the source WAV are headerless PCM; a server expecting a
    file named ``*.wav`` needs the container, not bare samples.
    """
    byte_rate = framerate * channels * sampwidth
    header = b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + len(frames)),
            b"WAVE",
            b"fmt ",
            struct.pack(
                "<IHHIIHH",
                16,
                1,
                channels,
                framerate,
                byte_rate,
                channels * sampwidth,
                sampwidth * 8,
            ),
            b"data",
            struct.pack("<I", len(frames)),
        ]
    )
    return header + frames


@dataclass(frozen=True)
class _ChunkPlan:
    """Lazy chunk plan for one WAV: arithmetic only, no per-chunk objects.

    ``total`` is computed once; each request's ``_Chunk`` is materialized just
    before it is sent, so a pathologically small chunk duration or an absurd
    frame count can never allocate one object per frame up front.
    """

    total: int
    step: int
    overlap: int
    n_frames: int

    def chunk(self, index: int) -> _Chunk:
        """Materialize the ``index``-th chunk of the plan."""
        core_start = index * self.step
        core_end = min(core_start + self.step, self.n_frames)
        audio_start = max(0, core_start - self.overlap) if index > 0 else 0
        return _Chunk(index, self.total, audio_start, core_start, core_end)


def _max_payload_bytes() -> int:
    """The single-request payload cap: chunk byte budget minus framing reserve.

    Read through this helper (not inlined) so a patched `_MAX_CHUNK_BYTES` or
    `_REQUEST_OVERHEAD_BYTES` takes effect wherever the cap is applied — both in
    `_plan_chunks` and in the facade's non-chunkable single-request path.
    """
    return _MAX_CHUNK_BYTES - _REQUEST_OVERHEAD_BYTES


def _plan_chunks(
    channels: int, sampwidth: int, framerate: int, n_frames: int, settings: Settings
) -> _ChunkPlan:
    """Split a WAV into core intervals using conservative internal bounds.

    The duration target and the byte cap both apply: whichever yields fewer frames
    per request wins. The byte cap covers the WHOLE request payload — overlap is
    prepended to every chunk after the first, and fixed container/multipart
    framing is reserved — so no configured overlap can push a request past
    ``_MAX_CHUNK_BYTES``. Chunks are planned lazily (see `_ChunkPlan`).
    """
    frame_bytes = max(1, channels * sampwidth)
    frames_per_duration = int(settings.lemonade_chunk_seconds * framerate)
    overlap = int(settings.lemonade_overlap_seconds * framerate)
    frames_per_bytes = max(1, _max_payload_bytes() // frame_bytes - overlap)
    step = max(1, min(frames_per_duration, frames_per_bytes))
    total = max(1, -(-n_frames // step))  # ceil division
    if total > _MAX_CHUNKS:
        raise LemonadeConfigurationError(
            f"{n_frames} frames at {frame_bytes} bytes/frame would need {total} "
            f"requests (limit {_MAX_CHUNKS}). The current chunk/overlap settings "
            "would upload this recording one sliver at a time — raise "
            "TAPEBACK_LEMONADE_CHUNK_SECONDS or lower "
            "TAPEBACK_LEMONADE_OVERLAP_SECONDS."
        )
    return _ChunkPlan(total=total, step=step, overlap=overlap, n_frames=n_frames)


@dataclass
class _MergeState:
    """Accumulated result of the chunk merge, mutated request by request."""

    segments: list[Segment]
    pinned: str | None
    probability: float | None
    total_text_chars: int = 0

    def _check_cumulative_bounds(self) -> None:
        if len(self.segments) > _MAX_CUMULATIVE_SEGMENTS:
            raise LemonadeCapabilityError(
                f"Lemonade transcription exceeded cumulative segment limit "
                f"({len(self.segments)} > {_MAX_CUMULATIVE_SEGMENTS})"
            )
        if self.total_text_chars > _MAX_CUMULATIVE_TEXT_CHARS:
            raise LemonadeCapabilityError(
                f"Lemonade transcription exceeded cumulative decoded text limit "
                f"({self.total_text_chars} > {_MAX_CUMULATIVE_TEXT_CHARS})"
            )

    @staticmethod
    def _same_utterance(left: Segment, right: Segment) -> bool:
        """Whether overlap candidates say the same thing, allowing end extensions.

        The shorter token sequence must anchor at the START (prefix) or the END
        (suffix) of the longer one: chunk overlap can truncate an utterance on
        either side, and the later chunk may add leading or trailing context.
        General mid-utterance containment is deliberately NOT accepted — two
        genuinely distinct short phrases ("no thanks" vs "thank you") that
        happen to fall in the same overlap window must both survive.
        """
        left_tokens = _utterance_tokens(left.text)
        right_tokens = _utterance_tokens(right.text)
        if not left_tokens or not right_tokens:
            return False
        short, long = (
            (left_tokens, right_tokens)
            if len(left_tokens) <= len(right_tokens)
            else (right_tokens, left_tokens)
        )
        span = len(short)
        return short == long[:span] or short == long[len(long) - span :]

    @staticmethod
    def _prefer(
        current: Segment,
        candidate: Segment,
        *,
        current_clipped: bool,
        candidate_clipped: bool,
        current_index: int,
        candidate_index: int,
    ) -> bool:
        """True when candidate wins the documented deterministic reconciliation."""
        current_key = (
            not current_clipped,
            current.end,
            len(current.text),
            current_index,
        )
        candidate_key = (
            not candidate_clipped,
            candidate.end,
            len(candidate.text),
            candidate_index,
        )
        return candidate_key > current_key

    @staticmethod
    def _is_token_subsequence(sub: list[str], full: list[str]) -> bool:
        """Whether sub is a non-empty contiguous token sub-sequence of full."""
        if not sub or not full or len(sub) > len(full):
            return False
        sub_len = len(sub)
        return any(full[i : i + sub_len] == sub for i in range(len(full) - sub_len + 1))

    def _purge_subsumed(self, duplicate_index: int, candidate: Segment, offset: float) -> None:
        """Purge any adjacent fragments from chunk N-1 that are subsumed by candidate's span."""
        cand_tokens = _utterance_tokens(candidate.text)
        pruned: list[Segment] = []
        for i, s in enumerate(self.segments):
            if i == duplicate_index:
                pruned.append(s)
            elif (
                s.start >= candidate.start - _TIMESTAMP_SLACK_SECONDS
                and s.end <= candidate.end + _TIMESTAMP_SLACK_SECONDS
                and s.start >= offset - _TIMESTAMP_SLACK_SECONDS
                and self._is_token_subsequence(_utterance_tokens(s.text), cand_tokens)
            ):
                self.total_text_chars -= len(s.text)
            else:
                pruned.append(s)
        self.segments = pruned

    def absorb(  # noqa: PLR0913 - chunk coordinates are explicit for auditability.
        self,
        payload: dict[str, Any],
        offset: float,
        core_start: float,
        index: int,
        chunk_duration: float | None = None,
        core_end: float | None = None,
        final_chunk: bool = False,
    ) -> None:
        """Convert one chunk response into file-relative, deduped segments.

        Also records the language and its probability the first time the server
        supplies segments, so later chunk requests can carry the pinned language.

        ``chunk_duration`` is the length of the audio this response describes; when
        given, segment timestamps beyond it are rejected (see `_require_segments`).
        """
        raw_segments = _require_segments(payload, upper_bound=chunk_duration)
        if not raw_segments:
            # Silence, or an empty leading chunk: ignored, never taken as the
            # detected language.
            return
        if self.pinned is None:
            detected = payload.get("language")
            # Auto detection needs an answer before another request is made.  A
            # missing/empty value is a capability failure, rather than silently
            # submitting the next chunk without a pinned language.
            self.pinned = _remote_language(detected)
            raw_probability = payload.get("language_probability")
            if raw_probability is None:
                # Some Lemonade versions report it under this name instead.
                raw_probability = payload.get("detected_language_probability")
            # Included only when the server actually supplied a valid probability;
            # otherwise the field stays absent rather than being invented. Booleans
            # and non-finite values (NaN, ±inf) are rejected with everything else.
            probability = _finite_number(raw_probability)
            if probability is not None and 0.0 <= probability <= 1.0:
                self.probability = probability
        candidates: list[Segment] = []
        for raw in raw_segments:
            start = offset + float(raw["start"])
            end = offset + float(raw["end"])
            candidates.append(
                Segment(
                    start=start,
                    end=end,
                    text=str(raw.get("text") or ""),
                    # Lemonade emits BPE tokens here, not lexical word timings.
                    words=None,
                )
            )
        # Compare only adjacent-chunk candidates occupying their shared overlap.
        # Existing unmatched speech remains; this avoids midpoint rounding dropping
        # unrelated words close to a chunk boundary.
        #
        # Linear search restricted to the immediately adjacent overlap window:
        # scan backwards from the end of self.segments, and stop scanning once
        # segments end before the overlap window begins.
        for candidate in candidates:
            duplicate_index: int | None = None
            if index > 0:
                for existing_index in range(len(self.segments) - 1, -1, -1):
                    existing = self.segments[existing_index]
                    if existing.end < offset - _TIMESTAMP_SLACK_SECONDS:
                        break
                    intersects_overlap = max(existing.start, candidate.start, offset) < min(
                        existing.end, candidate.end, core_start
                    )
                    if intersects_overlap and self._same_utterance(existing, candidate):
                        duplicate_index = existing_index
                        break
            if duplicate_index is None:
                self.total_text_chars += len(candidate.text)
                self.segments.append(candidate)
                self._check_cumulative_bounds()
                continue
            existing = self.segments[duplicate_index]
            # A segment ending at a non-final request's sent-audio boundary is
            # likely clipped; prefer its complete neighbor before time/text ties.
            # Chunk N-1 (existing) had its audio end at chunk N's core_start and was
            # non-final (since chunk N exists, index > 0).
            existing_clipped = (
                index > 0 and abs(existing.end - core_start) < _BOUNDARY_EPSILON_SECONDS
            )
            candidate_clipped = (
                not final_chunk
                and chunk_duration is not None
                and abs(candidate.end - (offset + chunk_duration)) < _BOUNDARY_EPSILON_SECONDS
            )
            if self._prefer(
                existing,
                candidate,
                current_clipped=existing_clipped,
                candidate_clipped=candidate_clipped,
                current_index=index - 1,
                candidate_index=index,
            ):
                self.total_text_chars += len(candidate.text) - len(existing.text)
                self.segments[duplicate_index] = candidate
                self._purge_subsumed(duplicate_index, candidate, offset)
                self._check_cumulative_bounds()
        self.segments.sort(key=lambda s: (s.start, s.end))
