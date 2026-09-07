"""The Lemonade transcription backend - public facade over the sibling modules.

Lemonade Server exposes an OpenAI-style ``/v1/audio/transcriptions`` endpoint. This
module owns the HTTP conversation and nothing else: it never imports faster-whisper,
never selects or records an accelerator, and never spawns inference locally. Hardware
choice belongs to the server; tapeback's opinion ends at the URL.

The implementation lives in cohesive siblings, re-exported here so callers keep one
import surface: `_lemonade_errors` (error taxonomy and error-text handling),
`_lemonade_transport` (deadlines, bounded reads, redirect refusal, proxy policy),
`_lemonade_audio` (chunking and chunk-merge policy), `_lemonade_language` (language
naming), `_lemonade_url` (endpoint normalization and the https/loopback trust
boundary), and `_lemonade_validate` (response schema checks). Transport decisions
that shape the output - chunking, overlap, language pinning, timestamp bounding, and
transport protection - are documented where they are implemented.

Errors are a deliberate hierarchy (see `LemonadeFallbackError`): the facade falls
back to faster-whisper only on fallback-eligible errors, and never on authentication,
configuration, or interrupt."""

from __future__ import annotations

import hashlib
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any

from tapeback._backends import StatusCallback, TranscriptionInfo

# Re-exports: the public facade other modules import stays `tapeback._lemonade`
# even though the implementation now lives in the sibling modules.
from tapeback._lemonade_audio import (
    _UPLOAD_FILENAME,
    AUDIO_PREPARATION_POLICY_VERSION,
    DEDUP_POLICY_VERSION,
    _lemonade_upload_frames,
    _max_payload_bytes,
    _MergeState,
    _plan_chunks,
    _wav_params,
    _wrap_wav,
)
from tapeback._lemonade_errors import (
    LemonadeAuthenticationError,
    LemonadeCapabilityError,
    LemonadeConfigurationError,
    LemonadeError,
    LemonadeFallbackError,
    LemonadeInferenceTimeout,
    LemonadeModelError,
    LemonadeUnavailableError,
    _parse_content_length,
    _redact_diagnostic,
    _sanitize_remote_detail,
)
from tapeback._lemonade_language import normalize_language
from tapeback._lemonade_transport import (
    _MAX_RESPONSE_BYTES,
    _DeadlineHTTPHandler,
    _DeadlineHTTPSHandler,
    _multipart_body,
    _NoRedirectHandler,
    _raise_for_http_error,
    _read_bounded,
    _remaining,
    _SafeProxyHandler,
)
from tapeback._lemonade_url import _is_loopback_host, _normalize_base_url
from tapeback._timing import ProgressReporter
from tapeback.models import Segment
from tapeback.settings import Settings

# Conservative floor for WAV bytes-per-second, used only to derive a duration
# UPPER BOUND for inputs whose header cannot be parsed: 8-bit 8 kHz mono PCM is
# the slowest sane encoding (8000 bytes/second), so size / 8000 always
# over-estimates the real duration. It needs to bound, not to be accurate.
_MIN_WAV_BYTES_PER_SECOND = 8000

__all__ = [
    "AUDIO_PREPARATION_POLICY_VERSION",
    "DEDUP_POLICY_VERSION",
    "LemonadeAuthenticationError",
    "LemonadeBackend",
    "LemonadeCapabilityError",
    "LemonadeConfigurationError",
    "LemonadeError",
    "LemonadeFallbackError",
    "LemonadeInferenceTimeout",
    "LemonadeModelError",
    "LemonadeUnavailableError",
]


def _noop_status(_message: str) -> None:
    """Default status sink."""

    """The Lemonade model is missing, invalid, unavailable, rejected or unloadable."""


_NO_PROXY_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    _NoRedirectHandler(),
    _DeadlineHTTPHandler(),
    _DeadlineHTTPSHandler(),
)


_DEFAULT_OPENER = urllib.request.build_opener(
    _SafeProxyHandler(),
    _NoRedirectHandler(),
    _DeadlineHTTPHandler(),
    _DeadlineHTTPSHandler(),
)


def _open_url(
    request: urllib.request.Request,
    timeout: float,
    *,
    bypass_proxies: bool,
) -> Any:
    """Open one URL, bypassing the inherited proxy configuration for loopback.

    Redirects are never followed on either path — see `_NoRedirectHandler`.
    """
    if bypass_proxies:
        return _NO_PROXY_OPENER.open(request, timeout=timeout)
    return _DEFAULT_OPENER.open(request, timeout=timeout)


class LemonadeBackend:
    """Transcription through a remote Lemonade Server over HTTP.

    Cheap to construct by design: no model load, no network call, no preflight.
    ``/v1/health`` and ``/v1/system-info`` exist for the status command's optional
    diagnostics and are never run as part of transcription.
    """

    def __init__(self, settings: Settings) -> None:
        """Validate local configuration. Raises LemonadeConfigurationError, never
        LemonadeFallbackError: a locally invalid setup must not silently turn into a
        faster-whisper run.
        """
        self._settings = settings
        self._base_url = _normalize_base_url(settings.lemonade_url)
        self._transcription_url = f"{self._base_url}/v1/audio/transcriptions"
        # Loopback traffic must never follow the inherited proxy configuration: a
        # host with http_proxy set and no matching NO_PROXY would otherwise route
        # the "local" upload through that proxy.
        self._bypass_proxies = _is_loopback_host(urllib.parse.urlparse(self._base_url).hostname)
        key = settings.lemonade_api_key.get_secret_value()
        if key and (
            not key.isascii()
            or key != key.strip()
            or any(ch.isspace() or not ch.isprintable() for ch in key)
        ):
            raise LemonadeConfigurationError(
                "TAPEBACK_LEMONADE_API_KEY is malformed: it must be a single ASCII bearer "
                "token with no whitespace or control characters."
            )
        # Held only for the Authorization header. Never logged, cached, or serialized:
        # the fingerprint below and every error message in this module are built
        # without it, and SecretStr keeps it out of settings reprs.
        self._api_key = key

    def describe(self) -> str:
        """One line: which model, on which server. Hardware stays the server's business.

        A plaintext-HTTP loopback endpoint with no bearer token is called out in
        the same line: this string is the one disclosure every transcription run
        makes (pipeline and live mode both print it), so the user sees that any
        local process which wins the port first can receive the audio.
        """
        warning = ""
        parsed = urllib.parse.urlparse(self._base_url)
        if parsed.scheme == "http" and _is_loopback_host(parsed.hostname) and not self._api_key:
            warning = (
                " (unauthenticated plaintext: any local listener on this "
                "port could receive the audio)"
            )
        return f"Lemonade: {self._settings.lemonade_model} at {self._base_url}{warning}"

    @property
    def base_url(self) -> str:
        """The validated, normalized base URL — safe to display and log.

        This is the structural rebuild from `_normalize_base_url`, never the raw
        configured string, so it cannot carry userinfo, a query string, or a
        fragment.
        """
        return self._base_url

    def cache_fingerprint(self) -> str:
        """Identity of everything that changes this backend's transcription output.

        Included: backend identity, normalized server URL, model, NORMALIZED
        language, chunk duration, overlap duration, and audio/merge policy versions.
        Normalizing the language matters: `english` and `en` produce identical
        output but would otherwise be two cache identities and over-invalidate.
        Deliberately
        excluded: the API key (credentials never belong in a cache key), the timeout,
        diagnostics settings, and anything about the server's accelerator — tapeback
        does not know it and must not encode knowledge of it.
        """
        parts = (
            "lemonade",
            self._base_url,
            self._settings.lemonade_model,
            normalize_language(self._settings.language),
            f"chunk_seconds={self._settings.lemonade_chunk_seconds!r}",
            f"overlap_seconds={self._settings.lemonade_overlap_seconds!r}",
            f"dedup_policy={DEDUP_POLICY_VERSION}",
            f"audio_preparation_policy={AUDIO_PREPARATION_POLICY_VERSION}",
            f"gate_mic_silence={self._settings.gate_mic_silence!r}",
        )
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:32]

    def health(self) -> Any:
        """Optional status diagnostic — GET /v1/health. Not a transcription preflight.

        Uses the short diagnostics timeout, not the inference timeout: these tiny
        GETs exist to diagnose a server that may be exactly the thing that is
        stalled, so `tapeback status` must never hang for minutes on them.
        """
        return self._get_json("/v1/health", self._settings.lemonade_diagnostics_timeout_seconds)

    def system_info(self) -> Any:
        """Optional diagnostic — GET /v1/system-info. Not a transcription preflight."""
        return self._get_json(
            "/v1/system-info", self._settings.lemonade_diagnostics_timeout_seconds
        )

    def transcribe(
        self,
        audio_path: Path,
        *,
        stage: str = "transcribe",
        on_status: StatusCallback = _noop_status,
        language_override: str | None = None,
    ) -> tuple[list[Segment], TranscriptionInfo]:
        """Transcribe one WAV, chunking long files and merging chunk responses.

        On Ctrl+C the chunks completed so far are returned with ``partial=True`` —
        the interrupt is never re-raised and never triggers a fallback, matching how
        the faster-whisper backend behaves.
        """
        explicit = self._explicit_language(language_override)
        pinned: str | None = normalize_language(explicit) if explicit else None
        state = _MergeState(segments=[], pinned=pinned, probability=None)
        partial = False
        params = _wav_params(audio_path)
        # Derive the duration from the validated WAV header before the
        # interruptible block: a Ctrl+C inside any chunk must still report the
        # recording's real length, not the 0.0 placeholder. _wav_params has
        # already rejected forged frame counts against the file size.
        duration = 0.0
        if params is not None and params[2] > 0 and params[3] > 0:
            duration = params[3] / params[2]

        try:
            # A degenerate header (0 framerate/frames) is not chunkable either.
            if params is not None and params[2] > 0 and params[3] > 0:
                self._transcribe_wav(audio_path, params, stage, on_status, state)
            else:
                # Not a parseable WAV: send it whole. The pipeline supplies PCM WAV,
                # so this is the tolerant path for unusual but valid inputs. The
                # whole body is buffered for multipart assembly (peak ~2x for the
                # copy), so it is bounded by the same cap as a chunk — a bigger
                # non-chunkable input is refused here and can fall back to
                # faster-whisper, which decodes formats the chunker cannot.
                size = audio_path.stat().st_size
                max_payload_bytes = _max_payload_bytes()
                if size > max_payload_bytes:
                    raise LemonadeCapabilityError(
                        f"Input is not a chunkable WAV and at {size} bytes exceeds "
                        f"the single-request cap ({max_payload_bytes} bytes) — refusing "
                        "to buffer it for upload."
                    )
                on_status(f"  {stage}: single request (file is not a chunkable WAV)")
                payload = self._request_transcription(
                    audio_path.read_bytes(), _UPLOAD_FILENAME, state.pinned
                )
                # The header is unparseable, so the real duration is unknown. Derive
                # a conservative upper bound from the file size and pass it as the
                # chunk duration, so the validator rejects server-supplied segment
                # timestamps beyond it (see `_require_segments`): a hostile or
                # broken server must not be able to pin arbitrary values into the
                # note. The chunked path is already bounded by its real durations.
                duration_bound = max(audio_path.stat().st_size / _MIN_WAV_BYTES_PER_SECOND, 1.0)
                state.absorb(payload, 0.0, 0.0, 0, chunk_duration=duration_bound)
        except KeyboardInterrupt:
            # Ctrl+C means keep what finished, mark the result partial, and stop —
            # never a fallback, never a cache write.
            partial = True
            on_status(
                f"Interrupted during '{stage}' — keeping the {len(state.segments)} "
                "segments transcribed so far."
            )

        if duration == 0.0 and state.segments:
            # Clamped to the single-request bound (when one applied): accepted
            # segments are already bounded, so this is belt-and-braces metadata.
            duration = max(s.end for s in state.segments)

        info: TranscriptionInfo = {
            "language": state.pinned or "",
            "duration": duration,
            "partial": partial,
        }
        if state.probability is not None:
            info["language_probability"] = state.probability
        state.segments.sort(key=lambda s: (s.start, s.end))
        return state.segments, info

    def _transcribe_wav(
        self,
        audio_path: Path,
        params: tuple[int, int, int, int],
        stage: str,
        on_status: StatusCallback,
        state: _MergeState,
    ) -> float:
        """Prepare, chunk, and send a WAV while preserving original time metadata."""
        channels, sampwidth, framerate, n_frames = params
        upload_frames = _lemonade_upload_frames(audio_path, params)
        original_duration = n_frames / framerate
        submitted_duration = upload_frames / framerate
        progress = ProgressReporter(stage, original_duration, on_status)
        if upload_frames == 0:
            on_status(
                f"  {stage}: entirely silent 16-bit PCM WAV; skipped Lemonade upload "
                f"(original {original_duration:.3f}s, submitted 0.000s)"
            )
            progress.update(original_duration)
            return original_duration
        if upload_frames < n_frames:
            on_status(
                f"  {stage}: trimmed exact trailing silence for Lemonade "
                f"(original {original_duration:.3f}s, submitted {submitted_duration:.3f}s)"
            )
        plan = _plan_chunks(channels, sampwidth, framerate, upload_frames, self._settings)
        with wave.open(str(audio_path), "rb") as wf:
            for index in range(plan.total):
                chunk = plan.chunk(index)
                if chunk.total > 1:
                    on_status(f"  {stage}: Lemonade chunk {chunk.index + 1}/{chunk.total}")
                wf.setpos(chunk.audio_start)
                frames = wf.readframes(chunk.core_end - chunk.audio_start)
                data = _wrap_wav(frames, channels, sampwidth, framerate)
                payload = self._request_transcription(data, _UPLOAD_FILENAME, state.pinned)
                state.absorb(
                    payload,
                    chunk.audio_start / framerate,
                    chunk.core_start / framerate,
                    chunk.index,
                    chunk_duration=(chunk.core_end - chunk.audio_start) / framerate,
                    core_end=chunk.core_end / framerate,
                    final_chunk=chunk.index == chunk.total - 1,
                )
                progress.update(chunk.core_end / framerate)
        if upload_frames < n_frames:
            # Once every submitted chunk succeeds, account for the omitted tail in
            # original-recording progress. Interrupted runs deliberately skip this.
            progress.update(original_duration)
        return original_duration

    def _explicit_language(self, language_override: str | None) -> str | None:
        """The language to pin from the first request, or None for auto-detection.

        An override wins over "auto" but never over an explicitly configured language —
        same rule the faster-whisper backend applies.
        """
        configured = self._settings.language
        if configured and configured != "auto":
            return configured
        return language_override or None

    def _request_transcription(
        self, data: bytes, filename: str, language: str | None
    ) -> dict[str, Any]:
        """POST one chunk to /v1/audio/transcriptions and parse the JSON response.

        Multipart with ``response_format=verbose_json`` and an explicit language when
        one is pinned. No ``prompt`` field: Tapeback's hotwords are a faster-whisper
        decoder bias, not a portable concept, and are documented as such rather than
        being smuggled into a different backend.
        """
        fields: list[tuple[str, str]] = [
            ("model", self._settings.lemonade_model),
            ("response_format", "verbose_json"),
        ]
        if language:
            fields.append(("language", language))
        body, content_type = _multipart_body(fields, "file", filename, "audio/wav", data)
        headers = {"Accept": "application/json", "Content-Type": content_type}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(  # noqa: S310 - validated base URL http(s) without userinfo or query
            self._transcription_url, data=body, headers=headers, method="POST"
        )
        raw = self._send(request)
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except (ValueError, RecursionError):
            raise LemonadeCapabilityError(
                "Lemonade returned a non-JSON transcription response — a text-only "
                "(e.g. FLM-style) backend cannot serve tapeback"
            ) from None
        if not isinstance(payload, dict):
            raise LemonadeCapabilityError(
                "Lemonade returned an unexpected transcription response shape"
            )
        return payload

    def _get_json(self, path: str, timeout: float | None = None) -> Any:
        """GET a diagnostic endpoint. Authenticated only when a key is configured.

        ``timeout`` defaults to the inference timeout; diagnostics endpoints pass
        the short dedicated timeout instead.
        """
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(  # noqa: S310 - validated base URL http(s) without userinfo or query
            self._base_url + path, headers=headers, method="GET"
        )
        raw = self._send(request, timeout)
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except (ValueError, RecursionError):
            payload = {"raw": raw.decode("utf-8", errors="replace")}
        return _redact_diagnostic(payload, (self._api_key,) if self._api_key else ())

    def _send(self, request: urllib.request.Request, timeout: float | None = None) -> bytes:
        """Perform one HTTP request, mapping every failure onto the error hierarchy.

        Response bodies — success and error alike — are read under
        ``_MAX_RESPONSE_BYTES``: a socket timeout is not a size bound, and a
        compromised or broken endpoint must not be able to exhaust client memory
        remotely. Over-limit responses get a sanitized error with no body content.
        Error bodies are classified with the configured API key as a redaction
        secret, so a response that reflects the Authorization header cannot
        persist the credential through the exception message.

        ``timeout`` defaults to ``settings.lemonade_timeout_seconds`` — the
        inference-oriented bound, generous because long-chunk inference
        legitimately takes minutes. Diagnostics endpoints pass their own short
        timeout instead.

        The timeout is enforced as a **monotonic end-to-end deadline**: a server
        that tricks a byte just before each socket timeout must still complete
        within the total budget, and a stall at any phase — connect, upload,
        read — counts against the same clock. Each blocking operation gets the
        *remaining* budget as its socket timeout rather than the full configured
        value: the open (connect plus headers), and every body read via
        ``_read_bounded``, which resets the socket timeout to the remaining
        budget before each read. A read that begins just before expiry therefore
        cannot block for another full socket timeout.
        """
        if timeout is None:
            timeout = self._settings.lemonade_timeout_seconds
        deadline = time.monotonic() + timeout
        setattr(request, "_tapeback_deadline", deadline)  # noqa: B010 - dynamically set deadline on Request
        try:
            with _open_url(
                request,
                _remaining(deadline),
                bypass_proxies=self._bypass_proxies,
            ) as response:
                headers = getattr(response, "headers", None)
                if headers is not None:
                    content_length = headers.get("Content-Length")
                    if content_length:
                        declared = _parse_content_length(content_length)
                        if declared > _MAX_RESPONSE_BYTES:
                            raise LemonadeUnavailableError(
                                f"Lemonade declared a {declared}-byte response — "
                                f"over the {_MAX_RESPONSE_BYTES}-byte response cap; "
                                "refusing to read it."
                            )
                raw = _read_bounded(response, _MAX_RESPONSE_BYTES, deadline=deadline)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise LemonadeUnavailableError(
                    f"Lemonade response exceeded the {_MAX_RESPONSE_BYTES}-byte response "
                    "cap; refusing to buffer or parse it."
                )
            return raw
        except urllib.error.HTTPError as exc:
            _raise_for_http_error(exc, self._api_key, deadline=deadline)
        except (http.client.InvalidURL, UnicodeEncodeError) as exc:
            raise LemonadeConfigurationError(
                f"Lemonade request could not be built or encoded: {exc}"
            ) from exc
        except http.client.HTTPException as exc:
            detail = (
                _sanitize_remote_detail(str(exc), secrets=(self._api_key,) if self._api_key else ())
                or type(exc).__name__
            )
            raise LemonadeUnavailableError(f"Lemonade HTTP protocol failure: {detail}") from exc
        except TimeoutError as exc:
            # socket.timeout is TimeoutError on Python 3.10+: this is the
            # read/inference timeout. Never resubmitted to Lemonade — the server may
            # still be working on the same request — so the façade falls back at once.
            raise LemonadeInferenceTimeout(
                f"Lemonade inference did not return within {timeout:.0f}s"
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            detail = (
                _sanitize_remote_detail(
                    str(reason), secrets=(self._api_key,) if self._api_key else ()
                )
                or type(reason).__name__
            )
            if isinstance(reason, TimeoutError):
                # Connect-phase timeout: the server never became reachable.
                raise LemonadeUnavailableError(f"Lemonade connection timed out: {detail}") from exc
            raise LemonadeUnavailableError(f"Lemonade server unreachable: {detail}") from exc
        except OSError as exc:
            detail = (
                _sanitize_remote_detail(str(exc), secrets=(self._api_key,) if self._api_key else ())
                or type(exc).__name__
            )
            raise LemonadeUnavailableError(f"Lemonade connection failed: {detail}") from exc
