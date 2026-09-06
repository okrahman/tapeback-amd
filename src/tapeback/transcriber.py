"""The shared transcription facade.

Callers keep the same surface they always had:

    transcriber = load_transcriber(settings)
    segments, info = transcriber.transcribe(audio_path)

Behind it sit interchangeable backends (see `_backends.py`): faster-whisper in
`_fw_backend.py` and Lemonade over HTTP in `_lemonade.py`. The facade owns everything
the two share — resume lookup, transactional storage, mono/stereo behaviour, the
monitor-first language reuse, speaker labelling, and partial-result handling — so a
backend is just "a way to turn one WAV into segments".

Fallback rule: the facade catches **only** `LemonadeFallbackError`. Never
`Exception`, never `BaseException`: authentication problems, locally invalid
configuration, and interrupts are not things faster-whisper can fix, and a broad
catch would swallow real bugs as "fallbacks".

Fallback latch: the first fallback-eligible Lemonade failure replaces the
configured backend with the faster-whisper backend for the **lifetime of this
Transcriber**. A long-lived caller (live transcription) therefore never mixes
one faster-whisper channel with a later Lemonade channel, never resubmits work
to a Lemonade server that just timed out, and cannot see Lemonade "recover"
mid-session into a mixed-backend transcript. `transcribe_stereo` goes further
when both channels are active: they form one backend transaction, so a fallback
on either channel retries **both** through faster-whisper — one run can never mix
one Lemonade channel with one faster-whisper channel.

Transactional storage is per channel, not all-or-none: a newly complete
same-backend channel is cached even when its sibling is partial, so an
interrupted run reuses everything it managed to finish. The only all-or-none
invalidation is backend-mixing: a fallback discards every staged Lemonade
result, and partial output is never cached."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from tapeback._backends import TranscriptionBackend, TranscriptionInfo
from tapeback._lemonade import LemonadeFallbackError, normalize_language
from tapeback._stereo import (
    _LatchedFallbackBackend,
    _noop_status,
    _StereoTranscriber,
)
from tapeback.models import Segment
from tapeback.settings import Settings

if TYPE_CHECKING:
    from tapeback._fw_backend import FasterWhisperBackend


class Transcriber(_StereoTranscriber):
    """Backend-agnostic facade over one configured transcription backend."""

    def __init__(self, settings: Settings) -> None:
        """Build the configured backend.

        faster-whisper loads its model here, as it always did. Lemonade is cheap to
        construct — no model, no network call, no preflight — so the heavy ML import
        the old monolith forced on every process now happens only when a
        faster-whisper backend is actually built.
        """
        self._settings = settings
        if settings.transcription_backend == "lemonade":
            from tapeback._lemonade import (  # noqa: PLC0415 — symmetry with _fw_backend
                LemonadeBackend,
            )

            self._backend: TranscriptionBackend = LemonadeBackend(settings)
        else:
            self._backend = self._new_fw_backend()

    def _new_fw_backend(self) -> FasterWhisperBackend:
        """Build a faster-whisper backend. Imported lazily: ~10s of ML imports."""
        from tapeback._fw_backend import (  # noqa: PLC0415 — 10s ML import, must stay lazy
            FasterWhisperBackend,
        )

        return FasterWhisperBackend(self._settings)

    def describe(self) -> str:
        """Human-readable record of which backend is configured and where it runs."""
        return self._backend.describe()

    def resolved_identity(self) -> dict[str, str] | None:
        """The active backend's resolved execution identity, if it reports one.

        The isolated worker emits this so the parent caches its result under the
        identity the child actually executed under, not the requested one.
        """
        getter = getattr(self._backend, "resolved_identity", None)
        return getter() if getter is not None else None

    def transcribe(
        self,
        audio_path: Path,
        *,
        stage: str = "transcribe",
        on_status: Callable[[str], None] = _noop_status,
        language_override: str | None = None,
        use_resume: bool = True,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        """Transcribe one audio file through the configured backend.

        Resume lookup and transactional storage are keyed on the backend's own
        `cache_fingerprint()` plus the normalized effective language, so a
        Lemonade result is never served to or stored for a faster-whisper run (or
        the other way round), and a transcript produced under one effective
        language is never reused for a run that pins another. On a
        the fallback-eligible Lemonade failure the whole input is retried through
        faster-whisper — and the facade latches to faster-whisper for the
        lifetime of this Transcriber — with only the accepted faster-whisper
        result cached. `use_resume=False` disables resume IO entirely; live mode
        uses it for ephemeral chunk WAVs that are deleted before any cache entry
        could ever be reused.
        """
        # A latched placeholder must never be a resume identity: its constant
        # fingerprint ignores model/device/compute settings, so a result stored
        # under it would leak across incompatible sessions. Bypass resume IO
        # while it is installed; the placeholder's transcribe() resolves the real
        # backend and re-dispatches through this method under the real identity.
        latched = isinstance(self._backend, _LatchedFallbackBackend)
        fingerprint = self._backend.cache_fingerprint()
        language_token = self._effective_language(language_override)
        key = (
            self._resume_key(audio_path, stage, fingerprint, language_token)
            if use_resume and not latched
            else None
        )
        cached = self._load_resume(key, stage, on_status)
        if cached is not None:
            return cached

        try:
            segments, info = self._backend.transcribe(
                audio_path,
                stage=stage,
                on_status=on_status,
                language_override=language_override,
            )
        except LemonadeFallbackError as exc:
            return self._fallback_transcribe(
                audio_path, stage, on_status, language_override, exc, use_resume=use_resume
            )

        self._store_resume(key, segments, info)
        return segments, info

    def _effective_language(self, language_override: str | None) -> str:
        """The normalized effective language, as the backend will apply it.

        Mirrors the backends' own explicit-language rule: a configured language
        wins over the override; "auto" means detection. This token is part of the
        resume identity, so a cached mic transcript constrained to English can
        never be served to a run whose monitor established French.
        """
        configured = self._settings.language
        if configured and configured != "auto":
            return normalize_language(configured)
        if language_override:
            return normalize_language(language_override)
        return "auto"

    def _fallback_transcribe(
        self,
        audio_path: Path,
        stage: str,
        on_status: Callable[[str], None],
        language_override: str | None,
        exc: LemonadeFallbackError,
        *,
        use_resume: bool = True,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        """Resolve one channel through faster-whisper after an eligible Lemonade failure.

        This also LATCHES: the faster-whisper backend becomes this Transcriber's
        backend for the rest of its lifetime, so a caller that keeps transcribing
        (live mode) never submits another request to the server that just failed.
        The faster-whisper identity is recomputed here, not assumed: its device can
        resolve differently (VRAM, thermal clamp) from anything the Lemonade backend
        knew about. Resume lookup happens BEFORE inference, under the faster-whisper
        fingerprint and the effective language, so an outage that keeps forcing
        fallback never redoes a channel an earlier fallback already cached. Only the
        result this run actually accepts is stored, under that same identity — unless
        the caller opted out of resume IO. While the latched-fallback placeholder is
        still installed (faster-whisper construction itself failed), resume IO is
        bypassed entirely: the placeholder's constant fingerprint is not an identity
        any result may be read from or stored under.
        """
        on_status(f"Lemonade transcription failed ({exc}) — falling back to faster-whisper.")
        self._backend = _LatchedFallbackBackend(exc, self, use_resume=use_resume)
        fw = self._new_fw_backend()
        self._backend = fw
        key = (
            self._resume_key(
                audio_path,
                stage,
                fw.cache_fingerprint(),
                self._effective_language(language_override),
            )
            if use_resume
            else None
        )
        cached = self._load_resume(key, stage, on_status)
        if cached is not None:
            return cached
        segments, info = fw.transcribe(
            audio_path,
            stage=stage,
            on_status=on_status,
            language_override=language_override,
        )
        self._store_resume(key, segments, info)
        return segments, info
