"""Reuse a channel that was already transcribed, so a re-run does not start from zero.

An interrupted stereo run used to redo everything. The monitor channel of a 31-minute
recording takes minutes even after the speed work, and repeating it because the *other*
channel was interrupted is pure waste.

**Granularity is a whole channel, deliberately.** Resuming part-way through one would
mean handing faster-whisper the remaining span via `clip_timestamps`, and its own
documentation says "vad_filter will be ignored if clip_timestamps is used". VAD is load
bearing here — it is half of why hallucinations on silence went away — so trading it for
a faster resume is a bad deal. That leaves the honest limitation: an interrupt during the
first channel has nothing to reuse, while one during the second saves the first.

A cached entry is only valid for the exact audio and the exact backend identity that
produced it, so the key covers both. The backend identity comes from
``backend.cache_fingerprint()`` — the caller, not this module, decides what makes
output change, because that answer is per backend (see _backends.py).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tapeback._fs import refuse_symlink_target, write_private_text
from tapeback.models import Segment, Word
from tapeback.settings import Settings

# Keep the directory bounded; entries are cheap but not free.
MAX_RESUME_ENTRIES = 50


def default_resume_dir() -> Path:
    """XDG data directory for resumable channel results."""
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg_data_home) if xdg_data_home else Path.home() / ".local" / "share"
    return base / "tapeback" / "resume"


@dataclass(frozen=True)
class ResumeKey:
    """Identifies one (audio, backend fingerprint, channel) combination."""

    digest: str

    @property
    def filename(self) -> str:
        return f"{self.digest}.json"


def resume_key(audio_path: Path, fingerprint: str, stage: str) -> ResumeKey | None:
    """Fingerprint the inputs. None when the audio cannot be described.

    Identity is path + size + mtime rather than a content hash: hashing a 400 MB WAV
    on every run would cost more than it saves, and these files are written once.
    ``fingerprint`` is the caller's backend identity — every setting that would make
    the backend produce different output, already collapsed to one string.
    """
    try:
        stat = audio_path.stat()
    except OSError:
        return None
    parts = [
        str(audio_path.resolve()),
        str(stat.st_size),
        str(stat.st_mtime_ns),
        stage,
        fingerprint,
    ]
    return ResumeKey(hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:32])


def settings_fingerprint(settings: Settings) -> str:
    """Faster-whisper's output-affecting identity, for `FasterWhisperBackend`.

    Kept beside the resume store so its meaning stays obvious: this is exactly the
    set of settings that change what faster-whisper produces, and a cached channel
    is only reusable when every one of them matches. That includes the settings
    that decide *where* the requested device/compute type actually executes —
    `min_free_vram_mib` and the thermal-clamp controls participate in device
    resolution in `_fw_backend._resolve_device`, so a threshold- or clamp-only
    change must invalidate the cache too. Adding a knob that affects
    faster-whisper output (directly or through device resolution) means adding
    it here.
    """
    output_affecting_settings = (
        "whisper_model",
        "device",
        "compute_type",
        "min_free_vram_mib",
        "thermal_clamp_check",
        "thermal_clamp_wait",
        "thermal_clamp_cpu_fallback",
        "language",
        "beam_size",
        "temperature",
        "batch_size",
        "hotwords",
        "vad_filter",
        "chunk_length",
        "condition_on_previous_text",
        "no_speech_threshold",
        "language_detection_segments",
        "multilingual",
        "hallucination_silence_threshold",
        "gate_mic_silence",
    )
    parts = [f"{name}={getattr(settings, name)!r}" for name in output_affecting_settings]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:32]


def _to_payload(segments: list[Segment], info: dict[str, Any]) -> dict[str, Any]:
    return {
        "info": info,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "speaker": s.speaker,
                "words": None
                if s.words is None
                else [
                    {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                    for w in s.words
                ],
            }
            for s in segments
        ],
    }


def _number(value: Any, name: str) -> float:
    """Validate a persisted timestamp/probability: finite, real, not a bool."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"resume {name} is not a finite number")
    return value


def _validated_word(word: Any) -> Word:
    if not isinstance(word, dict):
        raise ValueError("resume word is not a JSON object")
    text = word["word"]
    if not isinstance(text, str):
        raise ValueError("resume word text is not a string")
    start, end = _number(word["start"], "word start"), _number(word["end"], "word end")
    if start > end:
        raise ValueError("resume word start is after its end")
    return Word(
        start=start,
        end=end,
        word=text,
        probability=_number(word["probability"], "word probability"),
    )


def _validated_segment(segment: Any) -> Segment:
    if not isinstance(segment, dict):
        raise ValueError("resume segment is not a JSON object")
    start, end = _number(segment["start"], "segment start"), _number(segment["end"], "segment end")
    if start > end:
        raise ValueError("resume segment start is after its end")
    text = segment["text"]
    if not isinstance(text, str):
        raise ValueError("resume segment text is not a string")
    speaker = segment.get("speaker")
    if speaker is not None and not isinstance(speaker, str):
        raise ValueError("resume segment speaker is not a string")
    words: list[Word] | None = None
    raw_words = segment.get("words")
    if raw_words is not None:
        if not isinstance(raw_words, list):
            raise ValueError("resume segment words are not a JSON array")
        words = [_validated_word(word) for word in raw_words]
    return Segment(start=start, end=end, text=text, words=words, speaker=speaker)


def _from_payload(payload: Any) -> tuple[list[Segment], dict[str, Any]]:
    """Rebuild (segments, info) from a stored payload, rejecting anything off-schema.

    load() promises never to hand a caller data that can crash a run, so the full
    persisted schema is checked here, not just key presence: a syntactically valid
    but malformed entry — e.g. ``{"segments": [], "info": []}`` — must be a cache
    miss, not an AttributeError three layers up in ``transcribe_stereo()``. Every
    failure raises ValueError/KeyError/TypeError, which load() translates to None.
    """
    if not isinstance(payload, dict):
        raise ValueError("resume payload is not a JSON object")
    info = payload["info"]
    if not isinstance(info, dict):
        raise ValueError("resume info is not a JSON object")
    raw_segments = payload["segments"]
    if not isinstance(raw_segments, list):
        raise ValueError("resume segments are not a JSON array")
    return [_validated_segment(segment) for segment in raw_segments], info


def load(key: ResumeKey, directory: Path) -> tuple[list[Segment], dict[str, Any]] | None:
    """Return a previously stored channel, or None. Never raises on bad cache data."""
    path = directory / key.filename
    try:
        refuse_symlink_target(path, "load the resume entry")
    except RuntimeError:
        # A planted symlink at the entry path must never be followed: the entry
        # would be read from (or, in store(), written to) an attacker-chosen file.
        return None
    try:
        payload = json.loads(path.read_text())
        return _from_payload(payload)
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        # A corrupt or half-written entry is not worth a failed run; redo the work.
        return None


def store(
    key: ResumeKey,
    directory: Path,
    segments: list[Segment],
    info: dict[str, Any],
) -> Path | None:
    """Persist a completed channel. Returns the path, or None if it could not be written.

    Failing to write a cache entry must never fail the run that produced it.
    The entry holds full transcript text, so it is written 0600 into a verified
    0700 directory, atomically (tmp + rename), and never through a symlink.
    """
    path = directory / key.filename
    try:
        refuse_symlink_target(path, "store the resume entry")
        write_private_text(path, json.dumps(_to_payload(segments, info), ensure_ascii=False))
        _prune(directory)
    except (OSError, RuntimeError):
        return None
    return path


def _prune(directory: Path, keep: int = MAX_RESUME_ENTRIES) -> None:
    """Drop the least recently modified entries so the directory stays bounded."""
    entries = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if len(entries) <= keep:
        return
    for stale in entries[: len(entries) - keep]:
        stale.unlink(missing_ok=True)


def resume_dir(settings: Settings) -> Path:
    return settings.resume_cache_dir or default_resume_dir()
