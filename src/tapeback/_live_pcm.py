"""Live-mode audio primitives: WAV data-offset parsing, resampling,
timestamp re-basing, and overlap deduplication. Pure functions — no
session state, no IO beyond reading the WAV header."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from tapeback import const
from tapeback._lemonade_errors import _utterance_tokens
from tapeback.models import Segment, Word

DEDUP_TOLERANCE_SEC = 0.5


def find_data_offset(path: Path) -> int | None:
    """Find the byte offset where PCM data starts in a growing WAV file.

    Scans RIFF chunks to locate the 'data' chunk. Returns the byte position
    immediately after the data chunk header (i.e. where raw PCM bytes begin).

    Returns None when the header cannot be parsed *yet*: parecord flushes its
    header incrementally, so an early poll can observe a truncated RIFF walk.
    Callers must treat None as "retry next cycle", never as a permanent answer.
    The old unconditional 44-byte fallback was latched on first read, which
    misaligned every later offset and timestamp for the whole session whenever
    the real data chunk was not at 44.
    """
    try:
        with open(path, "rb") as f:
            riff = f.read(4)
            if riff != b"RIFF":
                return None
            f.read(4)  # file size (unreliable for growing files)
            wave_id = f.read(4)
            if wave_id != b"WAVE":
                return None
            # Scan sub-chunks until we find "data"
            while True:
                chunk_id = f.read(const.WAV_CHUNK_HEADER_BYTES)
                if len(chunk_id) < const.WAV_CHUNK_HEADER_BYTES:
                    return None
                chunk_size_bytes = f.read(const.WAV_CHUNK_HEADER_BYTES)
                if len(chunk_size_bytes) < const.WAV_CHUNK_HEADER_BYTES:
                    return None
                if chunk_id == b"data":
                    return f.tell()
                (chunk_size,) = struct.unpack("<I", chunk_size_bytes)
                f.seek(chunk_size, 1)
    except OSError:
        return None


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
    """Whether overlap candidates say the same thing, allowing token prefixes or suffixes.

    Containment is only trusted as evidence of the same utterance when it is
    stronger than arbitrary suffix containment. Chunk overlap truncation — the
    case this exists for — leaves the shorter side a substantial fraction of the
    longer one, keeping its start (the earlier chunk cut the tail) or its end
    (the later chunk adds leading context). A one-token fragment therefore only
    matches as a PREFIX: a quick one-word reply ("you") that merely echoes the
    last word of a previous sentence is a distinct utterance, not a truncated
    copy of it — and in the inverse ordering a long distinct sentence must not
    replace a short segment just because it happens to end with the same word.
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
    if len(short) == 1:
        return long[:1] == short
    if len(short) * 2 < len(long):
        return False
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

        # Check if this segment duplicates an existing one of the same speaker.
        # Besides the start-time tolerance, the candidate's time span must
        # actually overlap the existing segment's: a distinct quick reply that
        # starts right AFTER a finished sentence is new speech, not the same
        # utterance re-decoded in the overlap window.
        best_match_idx: int | None = None
        best_diff = DEDUP_TOLERANCE_SEC
        for i, es in enumerate(existing):
            if es.speaker == seg.speaker:
                diff = abs(seg.start - es.start)
                if (
                    diff < best_diff
                    and seg.start < es.end
                    and seg.end > es.start
                    and _same_utterance(es, seg)
                ):
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
