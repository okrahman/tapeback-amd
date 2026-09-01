"""The Lemonade transcription backend — HTTP transport, chunking, error mapping.

Lemonade Server exposes an OpenAI-style ``/v1/audio/transcriptions`` endpoint. This
module owns the HTTP conversation and nothing else: it never imports faster-whisper,
never selects or records an accelerator, and never spawns inference locally. Hardware
choice belongs to the server; tapeback's opinion ends at the URL.

Transport decisions that shape the output:

- **Chunking.** Long WAVs are split client-side using conservative internal duration
  and byte-size targets (``_MAX_CHUNK_BYTES``). These bounds exist to keep one
  request's memory bounded and progress reportable — they are tapeback's own
  transport policy, not knowledge of Lemonade's limits, and are versioned into the
  cache fingerprint for that reason.
- **Overlap.** Each chunk after the first prepends a few seconds of the previous
  chunk's audio, so a segment cut by a boundary is still heard whole by one request.
  The duplicate it produces is removed by the core-interval dedup policy.
- **Language.** With ``TAPEBACK_LANGUAGE=auto``, the first chunk that produces
  segments decides the language for every subsequent chunk (their requests carry it
  explicitly). Empty leading chunks are ignored, not treated as the answer.
- **Timestamps.** Servers return chunk-relative times; they are shifted into
  file-relative time before anything is returned or stored.
- **Transport protection.** Remote endpoints require ``https://``: meeting audio
  and the bearer credential must never travel in plaintext. Plain ``http://`` is
  accepted only for strictly recognized loopback endpoints (``localhost``,
  ``127.0.0.0/8``, ``::1``), and loopback requests bypass the process-wide proxy
  configuration so an inherited ``http_proxy`` cannot capture them. Response
  bodies are read under a hard size cap, so a broken or hostile endpoint cannot
  exhaust client memory with an oversized body.

Errors are a deliberate hierarchy (see `LemonadeFallbackError`): the façade falls
back to faster-whisper only on fallback-eligible errors, and never on authentication,
configuration, or interrupt.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import struct
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tapeback._backends import StatusCallback, TranscriptionInfo
from tapeback._timing import ProgressReporter
from tapeback.models import Segment, Word
from tapeback.settings import Settings

__all__ = [
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


class LemonadeError(Exception):
    """Base class for every Lemonade-specific failure."""


class LemonadeFallbackError(LemonadeError):
    """A failure the façade is allowed to answer with the faster-whisper backend."""


class LemonadeUnavailableError(LemonadeFallbackError):
    """Server unreachable, retryable server failure, or rate limiting."""


class LemonadeModelError(LemonadeFallbackError):
    """The Lemonade model is missing, invalid, unavailable, rejected or unloadable."""


class LemonadeCapabilityError(LemonadeFallbackError):
    """The endpoint cannot do what tapeback needs (e.g. text-only output)."""


class LemonadeInferenceTimeout(LemonadeFallbackError):
    """A request's read/inference timeout elapsed. Never resubmitted to Lemonade."""


class LemonadeConfigurationError(LemonadeError):
    """Bad local settings (URL, credentials, request shape). Never falls back."""


class LemonadeAuthenticationError(LemonadeError):
    """The server rejected the credentials. Never falls back — retrying cannot help."""


# Bumped whenever chunk/dedup behaviour changes what output a given WAV produces,
# so cached channel results from an older policy are never reused.
DEDUP_POLICY_VERSION = 1

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

# Hard cap on one HTTP response body, success or error. A configured endpoint is
# trusted with audio, not with the client's memory: this bounds what a broken or
# hostile server can make tapeback buffer before parsing.
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024

# Whisper's own language names (what a server echoing Whisper metadata returns)
# mapped to ISO-639-1 codes. Anything already a two-letter code passes through.
LANGUAGE_NAMES: dict[str, str] = {
    "afrikaans": "af",
    "albanian": "sq",
    "amharic": "am",
    "arabic": "ar",
    "armenian": "hy",
    "assamese": "as",
    "azerbaijani": "az",
    "bengali": "bn",
    "bosnian": "bs",
    "bulgarian": "bg",
    "catalan": "ca",
    "chinese": "zh",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "hausa": "ha",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jv",
    "kannada": "kn",
    "kazakh": "kk",
    "khmer": "km",
    "korean": "ko",
    "latin": "la",
    "latvian": "lv",
    "lithuanian": "lt",
    "macedonian": "mk",
    "malay": "ms",
    "malayalam": "ml",
    "marathi": "mr",
    "mongolian": "mn",
    "myanmar": "my",
    "nepali": "ne",
    "norwegian": "no",
    "nynorsk": "nn",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "sinhala": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "spanish": "es",
    "sundanese": "su",
    "swahili": "sw",
    "swedish": "sv",
    "tagalog": "tl",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "yiddish": "yi",
    "yoruba": "yo",
}

# Common aliases beyond Whisper's own names.
LANGUAGE_ALIASES: dict[str, str] = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
    "nb": "no",
    "pt-br": "pt",
    "in": "id",  # legacy ISO code still seen in the wild
    "iw": "he",  # legacy ISO code still seen in the wild
}


def normalize_language(raw: str) -> str:
    """Lowercase ISO-639-1 where known; otherwise the lowercased input unchanged.

    Never invents a code: an unrecognised name is passed through as-is rather than
    guessed, because the result feeds the pinned language for later chunks and the
    transcript metadata.
    """
    code = raw.strip().lower()
    if code in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[code]
    if code in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[code]
    return code


def _noop_status(_message: str) -> None:
    """Default status sink."""

    """The Lemonade model is missing, invalid, unavailable, rejected or unloadable."""


# Text fragments (lowercased) in a structured error body that mean "credentials" —
# checked on every status, because a proxy can answer 404 to an auth problem and a
# server can put an auth code in a 400.
_AUTH_MARKERS = ("auth", "unauthorized", "forbidden", "permission", "api key", "api_key", "apikey")

# Any mention of "model" in the structured body means a model problem, not an endpoint
# problem. This is what disambiguates the two meanings of a bare 404.
_MODEL_MARKER = "model"

# Statuses the classifier reasons about by name.
_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN = 403
_HTTP_NOT_FOUND = 404
_HTTP_BAD_REQUEST = 400
_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_INTERNAL_ERROR = 500


def _error_fields(payload: Any) -> tuple[str, str, str]:
    """Extract (type, code, message) from Lemonade's structured error body.

    Servers use ``{"error": {"type", "code", "message"}}``; FastAPI-style
    ``{"detail": ...}`` and flat ``{"message": ...}`` are accepted too, because the
    classification must survive a version change in the server's error shape.
    """
    if not isinstance(payload, dict):
        return "", "", ""
    error = payload.get("error")
    if isinstance(error, dict):
        return (
            str(error.get("type") or ""),
            str(error.get("code") or ""),
            str(error.get("message") or ""),
        )
    if isinstance(error, str):
        return "", "", error
    detail = payload.get("detail")
    if isinstance(detail, dict):
        return (
            str(detail.get("type") or ""),
            str(detail.get("code") or ""),
            str(detail.get("message") or detail.get("error") or ""),
        )
    if detail is not None:
        return "", "", str(detail)
    return "", "", str(payload.get("message") or "")


def classify_http_failure(status: int, payload: Any) -> LemonadeError:
    """Map an HTTP error response to the deliberate exception hierarchy.

    Status alone never decides: a 404 is a missing endpoint for one server and a
    missing model for another, so the structured body is read first and the status
    only breaks ties.
    """
    error_type, error_code, message = _error_fields(payload)
    blob = f"{error_type} {error_code} {message}".lower()

    if status in (_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN) or any(
        marker in blob for marker in _AUTH_MARKERS
    ):
        return LemonadeAuthenticationError(
            f"Lemonade rejected the credentials (HTTP {status}): {message or error_code}"
        )
    if _MODEL_MARKER in blob:
        return LemonadeModelError(
            f"Lemonade cannot serve the model (HTTP {status}): {error_code or message}"
        )
    if status == _HTTP_TOO_MANY_REQUESTS or status >= _HTTP_INTERNAL_ERROR:
        return LemonadeUnavailableError(
            f"Lemonade server failure (HTTP {status}): {message or error_type or 'no detail'}"
        )
    if status == _HTTP_NOT_FOUND:
        return LemonadeCapabilityError(
            f"Lemonade has no transcription endpoint at this URL (HTTP 404): {message}"
        )
    if _HTTP_BAD_REQUEST <= status < _HTTP_INTERNAL_ERROR:
        # A client-side problem with the request itself: retrying or falling back
        # cannot fix a locally invalid configuration.
        return LemonadeConfigurationError(
            f"Lemonade rejected the request as invalid (HTTP {status}): {message or error_type}"
        )
    return LemonadeUnavailableError(
        f"Lemonade server failure (HTTP {status}): {message or error_type or 'no detail'}"
    )


def _multipart_body(
    fields: list[tuple[str, str]],
    file_field: str,
    filename: str,
    content_type: str,
    data: bytes,
) -> tuple[bytes, str]:
    """Encode a multipart/form-data body. Small, standard, and dependency-free."""
    boundary = f"tapeback-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields:
        parts.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    parts.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
            ).encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            data,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# Opener for strictly-loopback endpoints. An empty ProxyHandler never consults
# the process-wide proxy configuration, so an inherited http_proxy without a
# matching NO_PROXY cannot route a "local" upload through it.
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _open_url(
    request: urllib.request.Request, timeout: float, *, bypass_proxies: bool
) -> Any:
    """Open one URL, bypassing the inherited proxy configuration for loopback."""
    if bypass_proxies:
        return _NO_PROXY_OPENER.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)


def _read_bounded(fp: Any, limit: int) -> bytes:
    """Read at most ``limit + 1`` bytes from ``fp``, so overflow is detectable.

    A single ``read(n)`` is not a size bound: a stream may legitimately return a
    short read while more data remains, so the read loops until the cap is
    exceeded or the stream ends.
    """
    pieces: list[bytes] = []
    remaining = limit + 1
    while remaining > 0:
        piece = fp.read(min(65536, remaining))
        if not piece:
            break
        pieces.append(piece)
        remaining -= len(piece)
    return b"".join(pieces)


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
    frames_per_bytes = max(
        1, (_MAX_CHUNK_BYTES - _REQUEST_OVERHEAD_BYTES) // frame_bytes - overlap
    )
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


def _is_loopback_host(hostname: str | None) -> bool:
    """True only for strictly recognized loopback endpoints.

    ``localhost`` (case-insensitive) and IP literals whose ``is_loopback`` is
    true — the whole ``127.0.0.0/8`` range and ``::1``. Deliberately no DNS
    resolution: a name that merely resolves to loopback is not recognized,
    because "strictly recognized" is the entire basis for allowing plaintext.
    """
    if not hostname:
        return False
    host = hostname.strip().rstrip(".").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _normalize_base_url(raw: str) -> str:
    """Validate and normalize the server base URL. Bad syntax never reaches HTTP.

    Transport rule: remote endpoints require ``https://`` — the multipart body
    carries the full recording and possibly the bearer credential, and plaintext
    HTTP offers an on-path observer both. Plain ``http://`` is allowed only for
    a strictly recognized loopback host, where the threat model is local and the
    default local-server setup keeps working.
    """
    candidate = raw.strip()
    parsed = urllib.parse.urlparse(candidate)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise LemonadeConfigurationError(
            f"TAPEBACK_LEMONADE_URL is not a valid http(s) URL: {raw!r}"
        )
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise LemonadeConfigurationError(
            f"TAPEBACK_LEMONADE_URL uses plaintext http for the non-loopback host "
            f"{parsed.hostname!r}: meeting audio and the bearer credential would "
            "travel unprotected. Use https:// for remote servers (plain http is "
            "allowed only for localhost, 127.0.0.0/8 and ::1)."
        )
    return candidate.rstrip("/")


def _require_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the payload's timestamped segments, or reject the response outright.

    Tapeback's pipeline needs segment timestamps for speaker labelling and vault
    timing, so a text-only response is unusable even when the prose is good. This is
    where FLM-style compact responses are rejected in full.
    """
    raw = payload.get("segments")
    if isinstance(raw, list) and raw:
        for item in raw:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("start"), (int, float))
                or not isinstance(item.get("end"), (int, float))
            ):
                raise LemonadeCapabilityError(
                    "Lemonade returned segments without usable timestamps"
                )
        return raw
    text = str(payload.get("text") or "")
    if text.strip():
        raise LemonadeCapabilityError(
            "Lemonade returned text without timestamped segments — tapeback requires "
            "segment timestamps and will fall back to faster-whisper"
        )
    # Silence: an empty response is legal output, not a capability problem.
    return []


def _convert_words(raw_words: Any, offset: float) -> list[Word]:
    """Convert a response segment's word list, shifted into file-relative time."""
    words: list[Word] = []
    if not isinstance(raw_words, list):
        return words
    for raw in raw_words:
        if not isinstance(raw, dict):
            continue
        start, end = raw.get("start"), raw.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        words.append(
            Word(
                start=offset + float(start),
                end=offset + float(end),
                word=str(raw.get("word") or ""),
                probability=float(raw.get("probability") or 0.0),
            )
        )
    return words


@dataclass
class _MergeState:
    """Accumulated result of the chunk merge, mutated request by request."""

    segments: list[Segment]
    pinned: str | None
    probability: float | None

    def absorb(self, payload: dict[str, Any], offset: float, core_start: float, index: int) -> None:
        """Convert one chunk response into file-relative, deduped segments.

        Also records the language and its probability the first time the server
        supplies segments, so later chunk requests can carry the pinned language.
        """
        raw_segments = _require_segments(payload)
        if not raw_segments:
            # Silence, or an empty leading chunk: ignored, never taken as the
            # detected language.
            return
        if self.pinned is None:
            detected = payload.get("language")
            if detected:
                self.pinned = normalize_language(str(detected))
            raw_probability = payload.get("language_probability")
            if raw_probability is None:
                # Some Lemonade versions report it under this name instead.
                raw_probability = payload.get("detected_language_probability")
            # Included only when the server actually supplied a valid probability;
            # otherwise the field stays absent rather than being invented.
            if isinstance(raw_probability, (int, float)) and not isinstance(raw_probability, bool):
                value = float(raw_probability)
                if 0.0 <= value <= 1.0:
                    self.probability = value
        for raw in raw_segments:
            start = offset + float(raw["start"])
            end = offset + float(raw["end"])
            # Core-interval dedup (policy v1): a later chunk's overlap region was
            # already heard by the previous chunk, so anything centred inside it is
            # a duplicate and is dropped. The first chunk keeps everything.
            if index > 0 and (start + end) / 2 < core_start:
                continue
            words = _convert_words(raw.get("words"), offset)
            self.segments.append(
                Segment(
                    start=start,
                    end=end,
                    text=str(raw.get("text") or "").strip(),
                    words=words or None,
                )
            )


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
        if key and (key != key.strip() or any(ch.isspace() or not ch.isprintable() for ch in key)):
            raise LemonadeConfigurationError(
                "TAPEBACK_LEMONADE_API_KEY is malformed: it must be a single bearer "
                "token with no whitespace or control characters."
            )
        # Held only for the Authorization header. Never logged, cached, or serialized:
        # the fingerprint below and every error message in this module are built
        # without it, and SecretStr keeps it out of settings reprs.
        self._api_key = key

    def describe(self) -> str:
        """One line: which model, on which server. Hardware stays the server's business."""
        return f"Lemonade: {self._settings.lemonade_model} at {self._base_url}"

    def cache_fingerprint(self) -> str:
        """Identity of everything that changes this backend's transcription output.

        Included: backend identity, normalized server URL, model, language, chunk
        duration, overlap duration, and the dedup policy version. Deliberately
        excluded: the API key (credentials never belong in a cache key), the timeout,
        diagnostics settings, and anything about the server's accelerator — tapeback
        does not know it and must not encode knowledge of it.
        """
        parts = (
            "lemonade",
            self._base_url,
            self._settings.lemonade_model,
            self._settings.language,
            f"chunk_seconds={self._settings.lemonade_chunk_seconds!r}",
            f"overlap_seconds={self._settings.lemonade_overlap_seconds!r}",
            f"dedup_policy={DEDUP_POLICY_VERSION}",
        )
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:32]

    def health(self) -> Any:
        """Optional status diagnostic — GET /v1/health. Not a transcription preflight."""
        return self._get_json("/v1/health")

    def system_info(self) -> Any:
        """Optional diagnostic — GET /v1/system-info. Not a transcription preflight."""
        return self._get_json("/v1/system-info")

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
        duration = 0.0
        params = _wav_params(audio_path)

        try:
            # A degenerate header (0 framerate/frames) is not chunkable either.
            if params is not None and params[2] > 0 and params[3] > 0:
                duration = self._transcribe_wav(audio_path, params, stage, on_status, state)
            else:
                # Not a parseable WAV: send it whole. The pipeline supplies PCM WAV,
                # so this is the tolerant path for unusual but valid inputs. The
                # whole body is buffered for multipart assembly (peak ~2x for the
                # copy), so it is bounded by the same cap as a chunk — a bigger
                # non-chunkable input is refused here and can fall back to
                # faster-whisper, which decodes formats the chunker cannot.
                size = audio_path.stat().st_size
                if size > _MAX_CHUNK_BYTES:
                    raise LemonadeCapabilityError(
                        f"Input is not a chunkable WAV and at {size} bytes exceeds "
                        f"the single-request cap ({_MAX_CHUNK_BYTES} bytes) — refusing "
                        "to buffer it for upload."
                    )
                on_status(f"  {stage}: single request (file is not a chunkable WAV)")
                payload = self._request_transcription(
                    audio_path.read_bytes(), audio_path.name, state.pinned
                )
                state.absorb(payload, 0.0, 0.0, 0)
        except KeyboardInterrupt:
            # Ctrl+C means keep what finished, mark the result partial, and stop —
            # never a fallback, never a cache write.
            partial = True
            on_status(
                f"Interrupted during '{stage}' — keeping the {len(state.segments)} "
                "segments transcribed so far."
            )

        info: TranscriptionInfo = {
            "language": state.pinned or "",
            "duration": duration,
            "partial": partial,
        }
        if state.probability is not None:
            info["language_probability"] = state.probability
        return state.segments, info

    def _transcribe_wav(
        self,
        audio_path: Path,
        params: tuple[int, int, int, int],
        stage: str,
        on_status: StatusCallback,
        state: _MergeState,
    ) -> float:
        """Send one request per chunk and merge the responses. Returns the duration."""
        channels, sampwidth, framerate, n_frames = params
        plan = _plan_chunks(channels, sampwidth, framerate, n_frames, self._settings)
        progress = ProgressReporter(stage, n_frames / framerate, on_status)
        with wave.open(str(audio_path), "rb") as wf:
            for index in range(plan.total):
                chunk = plan.chunk(index)
                if chunk.total > 1:
                    on_status(f"  {stage}: Lemonade chunk {chunk.index + 1}/{chunk.total}")
                wf.setpos(chunk.audio_start)
                frames = wf.readframes(chunk.core_end - chunk.audio_start)
                data = _wrap_wav(frames, channels, sampwidth, framerate)
                payload = self._request_transcription(data, audio_path.name, state.pinned)
                state.absorb(
                    payload,
                    chunk.audio_start / framerate,
                    chunk.core_start / framerate,
                    chunk.index,
                )
                progress.update(chunk.core_end / framerate)
        return n_frames / framerate

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
        request = urllib.request.Request(
            self._transcription_url, data=body, headers=headers, method="POST"
        )
        raw = self._send(request)
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            raise LemonadeCapabilityError(
                "Lemonade returned a non-JSON transcription response — a text-only "
                "(e.g. FLM-style) backend cannot serve tapeback"
            ) from None
        if not isinstance(payload, dict):
            raise LemonadeCapabilityError(
                "Lemonade returned an unexpected transcription response shape"
            )
        return payload

    def _get_json(self, path: str) -> Any:
        """GET a diagnostic endpoint. Authenticated only when a key is configured."""
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(self._base_url + path, headers=headers, method="GET")
        raw = self._send(request)
        try:
            return json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            return {"raw": raw.decode("utf-8", errors="replace")}

    def _send(self, request: urllib.request.Request) -> bytes:
        """Perform one HTTP request, mapping every failure onto the error hierarchy.

        Response bodies — success and error alike — are read under
        ``_MAX_RESPONSE_BYTES``: a socket timeout is not a size bound, and a
        compromised or broken endpoint must not be able to exhaust client memory
        remotely. Over-limit responses get a sanitized error with no body content.
        """
        timeout = self._settings.lemonade_timeout_seconds
        try:
            with _open_url(request, timeout, bypass_proxies=self._bypass_proxies) as response:
                headers = getattr(response, "headers", None)
                if headers is not None:
                    content_length = headers.get("Content-Length")
                    if (
                        content_length
                        and content_length.strip().isdigit()
                        and int(content_length) > _MAX_RESPONSE_BYTES
                    ):
                        raise LemonadeUnavailableError(
                            f"Lemonade declared a {int(content_length)}-byte response — "
                            f"over the {_MAX_RESPONSE_BYTES}-byte response cap; "
                            "refusing to read it."
                        )
                raw = _read_bounded(response, _MAX_RESPONSE_BYTES)
            if len(raw) > _MAX_RESPONSE_BYTES:
                raise LemonadeUnavailableError(
                    f"Lemonade response exceeded the {_MAX_RESPONSE_BYTES}-byte response "
                    "cap; refusing to buffer or parse it."
                )
            return raw
        except urllib.error.HTTPError as exc:
            raw = _read_bounded(exc, _MAX_RESPONSE_BYTES)
            if len(raw) > _MAX_RESPONSE_BYTES:
                # Classify by status alone rather than buffer a giant error body.
                payload = None
            else:
                try:
                    payload = json.loads(raw.decode("utf-8", errors="replace"))
                except (ValueError, OSError):
                    payload = None
            raise classify_http_failure(exc.code, payload) from exc
        except TimeoutError as exc:
            # socket.timeout is TimeoutError on Python 3.10+: this is the
            # read/inference timeout. Never resubmitted to Lemonade — the server may
            # still be working on the same request — so the façade falls back at once.
            raise LemonadeInferenceTimeout(
                f"Lemonade inference did not return within {timeout:.0f}s"
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                # Connect-phase timeout: the server never became reachable.
                raise LemonadeUnavailableError(f"Lemonade connection timed out: {reason}") from exc
            raise LemonadeUnavailableError(f"Lemonade server unreachable: {reason}") from exc
        except OSError as exc:
            raise LemonadeUnavailableError(f"Lemonade connection failed: {exc}") from exc
