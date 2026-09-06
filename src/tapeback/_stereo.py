"""Stereo transcription and transactional resume storage for the facade.

Lives in a base class mixed into `Transcriber`: the stereo pipeline is one
backend transaction (monitor first so its detected language is reused for
the gated mic; a fallback on either active channel retries both), and the
resume-cache helpers implement the per-channel commit policy — a newly
complete same-backend channel is cached even when its sibling is partial,
partial output is never cached, and only fallback-mixing invalidates.
"""

from __future__ import annotations

import math
import wave
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tapeback import _resume, const
from tapeback._backends import TranscriptionInfo
from tapeback._lemonade import LemonadeFallbackError
from tapeback._timing import stage_timer
from tapeback.models import Segment
from tapeback.settings import Settings

if TYPE_CHECKING:
    from tapeback._fw_backend import FasterWhisperBackend


def _noop_status(_message: str) -> None:
    """Default status sink — used when transcribe_stereo gets no reporter."""


class _LatchedFallbackBackend:
    """Placeholder backend installed immediately when Lemonade fails.

    Ensures Lemonade is never called again even if faster-whisper construction fails.
    """

    def __init__(
        self,
        cause: LemonadeFallbackError,
        transcriber: _StereoTranscriber,
        *,
        use_resume: bool = True,
    ) -> None:
        self._cause = cause
        self._transcriber = transcriber
        self._use_resume = use_resume

    def describe(self) -> str:
        return f"fallback-latched (cause: {self._cause})"

    def cache_fingerprint(self) -> str:
        return "fallback-latched"

    def transcribe(
        self,
        audio_path: Path,
        *,
        stage: str = "transcribe",
        on_status: Callable[[str], None] = _noop_status,
        language_override: str | None = None,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        fw = self._transcriber._new_fw_backend()
        self._transcriber._backend = fw
        # Delegate through the facade now that the real backend is in place: the
        # resume identity for this result must come from faster-whisper's own
        # fingerprint, never from this placeholder's constant — output cached
        # under "fallback-latched" would ignore model/device/compute settings
        # entirely and be served to later incompatible sessions.
        return self._transcriber.transcribe(
            audio_path,
            stage=stage,
            on_status=on_status,
            language_override=language_override,
            use_resume=self._use_resume,
        )


class _StereoTranscriber:
    """Stereo/resume machinery mixed into `Transcriber`.

    Attribute/method declarations exist for the type checker: the runtime
    state is created and the methods implemented by `Transcriber`, the only
    class that instantiates this one.
    """

    _settings: Settings

    if TYPE_CHECKING:

        def _new_fw_backend(self) -> FasterWhisperBackend: ...

        def transcribe(
            self,
            audio_path: Path,
            *,
            stage: str = "transcribe",
            on_status: Callable[[str], None] = _noop_status,
            language_override: str | None = None,
            use_resume: bool = True,
        ) -> tuple[list[Segment], TranscriptionInfo]: ...

        def _effective_language(self, language_override: str | None) -> str: ...

        def _fallback_transcribe(
            self,
            audio_path: Path,
            stage: str,
            on_status: Callable[[str], None],
            language_override: str | None,
            exc: LemonadeFallbackError,
            *,
            use_resume: bool = True,
        ) -> tuple[list[Segment], TranscriptionInfo]: ...

    def transcribe_stereo(  # noqa: PLR0912, PLR0913
        self,
        mic_16k: Path,
        monitor_16k: Path,
        *,
        on_status: Callable[[str], None] = _noop_status,
        use_resume: bool = True,
        skip_mic_on_monitor_partial: bool = True,
        mic_active: bool = True,
        monitor_active: bool = True,
    ) -> tuple[list[Segment], list[Segment], TranscriptionInfo]:
        """Transcribe both channels as ONE backend transaction.

        Returns (mic_segments, monitor_segments, info).
        mic_segments get speaker="You" automatically.
        info from the channel with more total speech duration.

        The monitor channel goes first so its detected language can be reused for the
        mic. Both channels are one conversation, but the mic is gated to near silence
        while the user listens, leaving auto-detection almost nothing to work from — it
        guessed wrong often enough to produce notes labelled `language: en` whose text
        was Russian.

        One backend transaction: if either channel hits a fallback-eligible Lemonade
        failure, every staged Lemonade result is discarded and BOTH channels resolve
        through faster-whisper — one Lemonade channel and one faster-whisper channel in
        a single transcript is unrepresentable. Resume entries may be READ before work
        starts; newly generated channel results are staged in memory, then committed PER
        CHANNEL once both transcribe: a complete same-backend channel is cached even
        when its sibling is partial (an interrupted run reuses finished work), while
        partial output is never cached. `skip_mic_on_monitor_partial` mirrors the
        offline Ctrl+C rule (stop, not "stop this channel"); live mode passes False
        because an interrupted monitor in the background thread does not mean the user
        stopped. `use_resume=False` disables resume IO for ephemeral inputs (live chunk
        WAVs). The mic cache is read only AFTER the monitor result establishes the
        mic's effective language, because that language is part of the mic's resume
        identity.

        Inactive channels are exact digital silence. They produce complete empty
        results carrying their WAV duration and bypass resume lookup, pacing, backend
        transcription, and cache writes. When only one channel is active, a fallback
        is limited to that channel; both active channels retain the transactional
        all-or-none fallback behavior.
        """
        if not monitor_active:
            on_status("Skipping monitor transcription — channel is digitally silent.")
        if not mic_active:
            on_status("Skipping mic transcription — channel is digitally silent.")

        fingerprint = self._backend.cache_fingerprint()
        # Same rule as transcribe(): while the latched-fallback placeholder is
        # installed, its constant fingerprint is not a resume identity — bypass
        # resume IO instead of reading or storing entries under it.
        latched = isinstance(self._backend, _LatchedFallbackBackend)
        monitor_key = (
            self._resume_key(
                monitor_16k, "transcribe monitor", fingerprint, self._effective_language(None)
            )
            if use_resume and monitor_active and not latched
            else None
        )

        # Resume reads happen up front for active channels; inactive channels never
        # consult an old entry, even if one happens to exist for the same WAV.
        monitor_result = (
            self._load_resume(monitor_key, "transcribe monitor", on_status)
            if monitor_active
            else self._empty_channel_result(monitor_16k)
        )
        # Staged entries carry what the commit loop needs to RECOMPUTE the resume
        # key at commit time (see the loop below): a fallback landing inside either
        # channel's transcribe() changes the resolved identity, and the key frozen
        # before the call must never become the stored identity.
        staged: list[tuple[Path, str, str, list[Segment], TranscriptionInfo]] = []

        if monitor_result is None:
            try:
                with stage_timer("transcribe monitor", on_status):
                    monitor_result = self._backend.transcribe(
                        monitor_16k, stage="transcribe monitor", on_status=on_status
                    )
            except LemonadeFallbackError as exc:
                if mic_active:
                    return self._fallback_stereo(
                        mic_16k,
                        monitor_16k,
                        on_status,
                        exc,
                        use_resume=use_resume,
                        skip_mic_on_monitor_partial=skip_mic_on_monitor_partial,
                    )
                # There is no sibling to keep transactionally consistent with this
                # channel, so only the active monitor falls back.
                monitor_result = self._fallback_transcribe(
                    monitor_16k,
                    "transcribe monitor",
                    on_status,
                    None,
                    exc,
                    use_resume=use_resume,
                )
            else:
                staged.append(
                    (
                        monitor_16k,
                        "transcribe monitor",
                        self._effective_language(None),
                        *monitor_result,
                    )
                )

        # Refresh the identity after the monitor stage: a device/compute fallback
        # that landed during monitor transcription changes what the backend resolves
        # to now, and the mic key below must not reuse the pre-stage fingerprint.
        fingerprint = self._backend.cache_fingerprint()

        mic_partial = False
        mic_result: tuple[list[Segment], TranscriptionInfo] | None = None
        if not mic_active:
            mic_result = self._empty_channel_result(mic_16k)
        elif monitor_result[1].get("partial") and skip_mic_on_monitor_partial:
            # Ctrl+C means stop, not "stop this channel". Starting the second one would
            # make the user interrupt twice; the monitor's work is already kept. Live
            # mode passes skip_mic_on_monitor_partial=False: an interrupted monitor in
            # the background thread does not mean the user stopped.
            on_status("Skipping the mic channel — transcription was interrupted.")
            mic_partial = True
        if mic_result is None and not mic_partial:
            detected = monitor_result[1].get("language") if monitor_active else None
            mic_language = str(detected) if detected else None
            mic_key = (
                self._resume_key(
                    mic_16k, "transcribe mic", fingerprint, self._effective_language(mic_language)
                )
                if use_resume and mic_active and not latched
                else None
            )
            mic_result = self._load_resume(mic_key, "transcribe mic", on_status)
            if mic_result is None:
                self._pace(on_status)
                try:
                    with stage_timer("transcribe mic", on_status):
                        mic_result = self._backend.transcribe(
                            mic_16k,
                            stage="transcribe mic",
                            on_status=on_status,
                            language_override=mic_language,
                        )
                except LemonadeFallbackError as exc:
                    if monitor_active:
                        return self._fallback_stereo(
                            mic_16k,
                            monitor_16k,
                            on_status,
                            exc,
                            use_resume=use_resume,
                            skip_mic_on_monitor_partial=skip_mic_on_monitor_partial,
                        )
                    # With a silent monitor, this is a one-channel transaction.
                    mic_result = self._fallback_transcribe(
                        mic_16k,
                        "transcribe mic",
                        on_status,
                        mic_language,
                        exc,
                        use_resume=use_resume,
                    )
                else:
                    staged.append(
                        (
                            mic_16k,
                            "transcribe mic",
                            self._effective_language(mic_language),
                            *mic_result,
                        )
                    )

        mic_segments, monitor_segments, info = self._assemble_stereo(
            mic_result, monitor_result, mic_partial
        )

        # Per-channel commit: a complete same-backend channel is cached even when its
        # sibling was interrupted. Inactive channels never enter staged. The key is
        # RECOMPUTED at commit time so a fallback inside either transcribe() is
        # reflected in the stored identity — a CPU/int8 result must never be cached
        # under a CUDA identity.
        for audio, stage_name, lang_token, segs, channel_info in staged:
            commit_key = self._resume_key(
                audio, stage_name, self._backend.cache_fingerprint(), lang_token
            )
            self._store_resume(commit_key, segs, channel_info)
        return mic_segments, monitor_segments, info

    def _fallback_stereo(
        self,
        mic_16k: Path,
        monitor_16k: Path,
        on_status: Callable[[str], None],
        exc: LemonadeFallbackError,
        *,
        use_resume: bool = True,
        skip_mic_on_monitor_partial: bool = True,
    ) -> tuple[list[Segment], list[Segment], TranscriptionInfo]:
        """Discard all staged Lemonade output and resolve BOTH channels on faster-whisper.

        Both channels, even ones Lemonade already finished: returning one Lemonade
        channel and one faster-whisper channel would mix decoders inside one
        transcript. Once transcribes finish, staged faster-whisper results commit PER
        CHANNEL (partial output is never cached, a complete sibling is), under
        faster-whisper fingerprints. `use_resume` and `skip_mic_on_monitor_partial`
        behave exactly as in `transcribe_stereo`. This also LATCHES the facade
        to faster-whisper for the lifetime of the Transcriber.
        """
        on_status(
            f"Lemonade transcription failed ({exc}) — falling back to faster-whisper "
            "for both channels."
        )
        self._backend = _LatchedFallbackBackend(exc, self, use_resume=use_resume)
        fw = self._new_fw_backend()
        self._backend = fw
        fw_fingerprint = fw.cache_fingerprint()
        monitor_key = (
            self._resume_key(
                monitor_16k, "transcribe monitor", fw_fingerprint, self._effective_language(None)
            )
            if use_resume
            else None
        )
        staged: list[tuple[Path, str, str, list[Segment], TranscriptionInfo]] = []

        monitor_result = self._load_resume(monitor_key, "transcribe monitor", on_status)
        if monitor_result is None:
            with stage_timer("transcribe monitor", on_status):
                monitor_result = fw.transcribe(
                    monitor_16k, stage="transcribe monitor", on_status=on_status
                )
            staged.append(
                (
                    monitor_16k,
                    "transcribe monitor",
                    self._effective_language(None),
                    *monitor_result,
                )
            )

        mic_partial = False
        mic_result: tuple[list[Segment], TranscriptionInfo] | None = None
        if monitor_result[1].get("partial") and skip_mic_on_monitor_partial:
            # Same rule as transcribe_stereo: live mode keeps its mic even when the
            # monitor was interrupted inside this interval.
            on_status("Skipping the mic channel — transcription was interrupted.")
            mic_partial = True
        # Refresh the identity after the monitor stage (see transcribe_stereo): the
        # isolated child may have resolved a different device during monitor work,
        # and the mic key must not reuse the pre-stage fingerprint.
        fw_fingerprint = fw.cache_fingerprint()
        if mic_result is None and not mic_partial:
            detected = monitor_result[1].get("language")
            mic_language = str(detected) if detected else None
            mic_key = (
                self._resume_key(
                    mic_16k,
                    "transcribe mic",
                    fw_fingerprint,
                    self._effective_language(mic_language),
                )
                if use_resume
                else None
            )
            mic_result = self._load_resume(mic_key, "transcribe mic", on_status)
            if mic_result is None:
                fw.pace(on_status)
                with stage_timer("transcribe mic", on_status):
                    mic_result = fw.transcribe(
                        mic_16k,
                        stage="transcribe mic",
                        on_status=on_status,
                        language_override=mic_language,
                    )
                staged.append(
                    (
                        mic_16k,
                        "transcribe mic",
                        self._effective_language(mic_language),
                        *mic_result,
                    )
                )

        mic_segments, monitor_segments, info = self._assemble_stereo(
            mic_result, monitor_result, mic_partial
        )
        # Per-channel commit — same rule as transcribe_stereo: partial output is never
        # cached, but a complete same-backend sibling is. Keys are RECOMPUTED at commit
        # time (see transcribe_stereo) so a late device resolution is reflected.
        for audio, stage_name, lang_token, segs, channel_info in staged:
            commit_key = self._resume_key(
                audio, stage_name, self._backend.cache_fingerprint(), lang_token
            )
            self._store_resume(commit_key, segs, channel_info)
        return mic_segments, monitor_segments, info

    def _assemble_stereo(
        self,
        mic_result: tuple[list[Segment], TranscriptionInfo] | None,
        monitor_result: tuple[list[Segment], TranscriptionInfo],
        mic_skipped: bool,
    ) -> tuple[list[Segment], list[Segment], TranscriptionInfo]:
        """Label mic segments "You" and merge the two channels' info dicts."""
        mic_segments_raw: list[Segment] = mic_result[0] if mic_result else []
        mic_info: TranscriptionInfo = mic_result[1] if mic_result else {}
        monitor_segments = monitor_result[0]
        monitor_info = monitor_result[1]

        # Assign speaker="You" to mic segments
        mic_segments = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text,
                words=s.words,
                speaker=const.SPEAKER_YOU,
            )
            for s in mic_segments_raw
        ]

        # Pick info from channel with more speech. When neither channel has speech
        # (or mic was skipped), prefer monitor_info so language and duration metadata
        # are preserved rather than replaced with empty mic_info.
        mic_speech = sum(s.end - s.start for s in mic_segments)
        monitor_speech = sum(s.end - s.start for s in monitor_segments)
        info = dict(
            mic_info if (mic_result is not None and mic_speech > monitor_speech) else monitor_info
        )
        # The combined stereo source lasts as long as the LONGER channel. The
        # metadata may come from either channel, and the one with more speech can
        # have the shorter recording, so the maximum finite duration always wins
        # over whichever value the selected info dict happened to carry.
        durations = [
            float(d)
            for d in (mic_info.get("duration"), monitor_info.get("duration"))
            if isinstance(d, (int, float)) and math.isfinite(float(d)) and float(d) > 0.0
        ]
        if durations:
            info["duration"] = max(durations)
        # Partiality belongs to the run, not to whichever channel happened to be
        # picked for its language — a transcript missing one channel is partial.
        info["partial"] = bool(
            monitor_info.get("partial") or mic_info.get("partial") or mic_skipped
        )
        return mic_segments, monitor_segments, info

    def _pace(self, on_status: Callable[[str], None]) -> None:
        """Idle between stages when the backend wants to — faster-whisper on CUDA does;
        Lemonade has no local GPU to cool, so it simply offers no pace method."""
        pace = getattr(self._backend, "pace", None)
        if pace is not None:
            pace(on_status)

    @staticmethod
    def _empty_channel_result(audio_path: Path) -> tuple[list[Segment], TranscriptionInfo]:
        """Build a complete empty result without consulting a transcription backend."""
        duration = 0.0
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                if wav_file.getframerate() > 0:
                    duration = wav_file.getnframes() / wav_file.getframerate()
        except (OSError, wave.Error):
            # A live paired call may reuse the active channel's temporary path for
            # the inactive side. A missing/unreadable path still represents an empty
            # result; callers that have a valid WAV get its exact header duration.
            pass
        return [], {"duration": duration, "partial": False}

    def _resume_key(
        self,
        audio_path: Path,
        stage: str,
        fingerprint: str,
        effective_language: str = "auto",
    ) -> _resume.ResumeKey | None:
        """Resume identity: audio + backend fingerprint + stage + effective language.

        The language is part of the identity because the effective language
        changes the transcript a run produces — a cache entry constrained to
        English must never be served to a run pinned to French. It is mixed into
        the fingerprint string; `_resume.resume_key` hashes whatever identity
        string the caller supplies.
        """
        if not self._settings.resume_cache:
            return None
        identity = f"{fingerprint}\x00lang={effective_language}"
        return _resume.resume_key(audio_path, identity, stage)

    def _load_resume(
        self,
        key: _resume.ResumeKey | None,
        stage: str,
        on_status: Callable[[str], None],
    ) -> tuple[list[Segment], TranscriptionInfo] | None:
        if key is None:
            return None
        cached = _resume.load(key, _resume.resume_dir(self._settings))
        if cached is not None:
            on_status(f"Reusing the '{stage}' result from an earlier run.")
        return cached

    def _store_resume(
        self,
        key: _resume.ResumeKey | None,
        segments: list[Segment],
        info: dict[str, Any],
    ) -> None:
        """Cache a channel, but only a complete one.

        Storing a partial result would make the next run reuse the truncated version
        and call it done — the opposite of what resuming is for.
        """
        if key is None or info.get("partial"):
            return
        _resume.store(key, _resume.resume_dir(self._settings), segments, info)
