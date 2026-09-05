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
  file-relative time before anything is returned or stored. Segments and the
  words inside them are bounded to the audio that was actually sent and to their
  containing segment, so a hostile or broken response cannot write timestamps
  past the recording into the transcript or the resume cache.
- **Transport protection.** Remote endpoints require ``https://``: meeting audio
  and the bearer credential must never travel in plaintext. Plain ``http://`` is
  accepted only for strictly recognized loopback endpoints (``localhost``,
  ``127.0.0.0/8``, ``::1``), and loopback requests bypass the process-wide proxy
  configuration so an inherited ``http_proxy`` cannot capture them. Remote HTTPS
  supports explicit plaintext CONNECT proxies; TLS-to-proxy and ambiguous proxy
  schemes fail closed before a body is sent. Redirects are
  never followed — a 30x cannot move the request (and its Authorization header)
  to a server-chosen origin or downgrade https to http. Response bodies are read
  under a hard size cap, so a broken or hostile endpoint cannot exhaust client
  memory with an oversized body. Server-supplied error text is sanitized —
  length-capped, stripped of terminal-control characters, and redacted of the
  configured API key — before it can reach a status line, the run log, or the
  terminal.

Errors are a deliberate hierarchy (see `LemonadeFallbackError`): the façade falls
back to faster-whisper only on fallback-eligible errors, and never on authentication,
configuration, or interrupt.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import ipaddress
import json
import math
import queue
import re
import socket
import struct
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from tapeback._backends import StatusCallback, TranscriptionInfo
from tapeback._timing import ProgressReporter
from tapeback.models import Segment
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
DEDUP_POLICY_VERSION = 3

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

# A just-expired deadline must not hand 0 (or a negative, which some stacks
# treat as "infinite") to connect()/settimeout; expiry is checked before every
# blocking operation. DNS runs in daemon workers because getaddrinfo() has no
# portable cancellation API. The semaphore bounds workers that can remain stuck
# inside the platform resolver after their callers have timed out.
_MIN_SOCKET_TIMEOUT_SECONDS = 0.05
_DNS_RESOLVER_SLOTS = threading.BoundedSemaphore(2)

# Schema limits on per-response payload size to guard against pathological memory / CPU usage.
_MAX_RESPONSE_SEGMENTS = 5000
_MAX_SEGMENT_TEXT_CHARS = 10_000

# Cumulative bounds on total output across all chunk responses for a single transcription.
_MAX_CUMULATIVE_SEGMENTS = 50_000
_MAX_CUMULATIVE_TEXT_CHARS = 5_000_000

# The multipart part name under which audio is uploaded. Always opaque: the source
# filename is attacker-influencable in principle (POSIX names may contain quotes
# and newlines, which would be interpolated into a MIME header verbatim) and
# discloses local metadata the server has no use for. The content is always
# declared audio/wav, so the fixed name matches every request we send.
_UPLOAD_FILENAME = "audio.wav"

# Slack allowed on segment/word timestamps past a chunk's audio length before the
# response is rejected, to absorb server-side rounding at chunk boundaries.
_TIMESTAMP_SLACK_SECONDS = 1.0
_BOUNDARY_EPSILON_SECONDS = 0.001

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


# A remote language value must be a bounded, structurally safe token: it is
# interpolated into multipart field headers, the YAML front matter, and the resume
# cache, so quotes, CR/LF, control characters, and megabyte strings must never pass.
# Every real ISO-639-1 code, Whisper language name, and alias (``zh-hans``, ``pt-br``)
# fits this grammar.
_LANGUAGE_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,23}$")
_LANGUAGE_MAX_CHARS = 24


def _remote_language(raw: Any) -> str:
    """Validate a server-supplied language value and normalize it.

    Unlike `normalize_language` (which also digests trusted local settings), this is
    a schema check: anything that is not a bounded language-like string is a
    LemonadeCapabilityError, never coerced with ``str()`` and never pinned. A hostile
    value must fall back, not hard-abort multipart assembly on a later chunk or
    corrupt the transcript metadata.
    """
    if not isinstance(raw, str):
        raise LemonadeCapabilityError(
            "Lemonade returned a response with an unusable language value (not a string)"
        )
    value = raw.strip().lower()
    if len(value) > _LANGUAGE_MAX_CHARS or not _LANGUAGE_TOKEN_RE.match(value):
        raise LemonadeCapabilityError(
            "Lemonade returned a response with an unusable language value "
            "(not a bounded language token)"
        )
    return normalize_language(value)


def _noop_status(_message: str) -> None:
    """Default status sink."""

    """The Lemonade model is missing, invalid, unavailable, rejected or unloadable."""


# Auth phrases matched against the *structured* error fields (type/code), which
# the server's own code path authored — these may be read broadly. They are
# matched as whole tokens after tokenization, never as substrings.
_STRUCTURED_AUTH_PHRASES = (
    "auth",
    "unauthorized",
    "forbidden",
    "authentication",
    "not authorized",
    "permission denied",
    "insufficient permission",
    "access denied",
    "api key",
    "apikey",
)

# Auth phrases matched against the free-text *message*. Deliberately narrow and
# always token-aware: unanchored substring matching turned "model author not
# found" (contains "auth") and "permission denied loading model" (contains
# "permission") into authentication errors, disabling fallback for ordinary
# model/server failures.
_MESSAGE_AUTH_PHRASES = (
    "unauthorized",
    "authentication",
    "not authorized",
    "invalid api key",
    "missing api key",
    "api key",
    "access denied",
    "forbidden",
)

# Statuses the classifier reasons about by name.
_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN = 403
_HTTP_NOT_FOUND = 404
_HTTP_BAD_REQUEST = 400
_HTTP_TOO_MANY_REQUESTS = 429
_HTTP_REQUEST_TIMEOUT = 408
_HTTP_INTERNAL_ERROR = 500
_HTTP_MULTIPLE_CHOICES = 300

# Default ports are dropped from the normalized base URL.
_HTTP_DEFAULT_PORTS = {"http": 80, "https": 443}


# Server-controlled text is never trusted verbatim in a user-visible message: a
# hostile or broken server (or proxy) can reflect the received Authorization
# header back in an error body, and a reflected bearer token would otherwise be
# written to the terminal, a status event, or the run-log file. Remote detail is
# therefore length-capped, stripped of terminal-control characters (which could
# corrupt or disguise terminal output), and redacted of every configured secret
# before it is interpolated into any exception message.
_MAX_REMOTE_DETAIL_CHARS = 200
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
_REDACTED_LABEL = "[redacted]"


def _sanitize_remote_detail(
    text: str | None, secrets: tuple[str, ...] = (), *, limit: int = _MAX_REMOTE_DETAIL_CHARS
) -> str:
    """Make server-supplied text safe to interpolate into a user-visible message."""
    if not text:
        return ""
    out = _CONTROL_CHARS_RE.sub("", text)
    for secret in secrets:
        if secret:
            out = out.replace(secret, _REDACTED_LABEL)
    return out[:limit]


def _redact_diagnostic(value: Any, secrets: tuple[str, ...]) -> Any:
    """Recursively remove configured credentials from server diagnostic JSON.

    Diagnostics are deliberately returned as JSON-shaped values for CLI rendering;
    a server can reflect an Authorization header anywhere in that tree, not only in
    an error message.  Preserve the shape and non-string scalar types so status
    output remains useful while making every string safe to serialize.
    """
    if isinstance(value, str):
        out = value
        for secret in secrets:
            if secret:
                out = out.replace(secret, _REDACTED_LABEL)
        return out
    if isinstance(value, list):
        return [_redact_diagnostic(item, secrets) for item in value]
    if isinstance(value, dict):
        return {
            _redact_diagnostic(key, secrets) if isinstance(key, str) else key: _redact_diagnostic(
                item, secrets
            )
            for key, item in value.items()
        }
    return value


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


def _tokens(text: str) -> list[str]:
    """Lowercased alphanumeric tokens: \"model author not found\" -> [model, author, ...].

    Underscores and other separators split too, so a structured code such as
    ``invalid_api_key`` tokenizes to [invalid, api, key].
    """
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token]


_CJK_RANGES = r"\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af\uf900-\ufaff"
_UTTERANCE_TOKEN_PATTERN = re.compile(rf"[{_CJK_RANGES}]|[^\W_]+")


def _utterance_tokens(text: str) -> list[str]:
    """Unicode-aware, case-folded tokenization for utterance comparison and overlap dedup.

    Normalizes Unicode via NFKC, case-folds, splits individual CJK characters, and extracts
    words for alphabetic and syllabic scripts (Latin, Cyrillic, Arabic, Hebrew, Greek, etc.).
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _UTTERANCE_TOKEN_PATTERN.findall(normalized)


def _has_phrase(tokens: list[str], phrase: str) -> bool:
    """Whether the token list contains ``phrase`` as consecutive whole tokens.

    Token-aware, never substring-based: "auth" cannot match inside "author" and
    "permission" cannot match inside "permissions granted to load the model".
    """
    parts = phrase.split()
    span = len(parts)
    return any(tokens[i : i + span] == parts for i in range(len(tokens) - span + 1))


def _is_auth_failure(status: int, structured_tokens: list[str], message_tokens: list[str]) -> bool:
    """Whether the response means "credentials", decided in trust order.

    The status and the structured type/code fields decide first; a model mention
    in the body outranks incidental auth wording in the free-text message, so
    "permission denied loading model" stays a model failure that falls back.
    """
    if status in (_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN):
        return True
    if any(_has_phrase(structured_tokens, phrase) for phrase in _STRUCTURED_AUTH_PHRASES):
        return True
    if "model" in structured_tokens or "model" in message_tokens:
        return False
    return any(_has_phrase(message_tokens, phrase) for phrase in _MESSAGE_AUTH_PHRASES)


def classify_http_failure(  # noqa: PLR0911 — a deliberate flat decision ladder; one
    # return per classification rule reads better than a folded return computation.
    status: int,
    payload: Any,
    secrets: tuple[str, ...] = (),
) -> LemonadeError:
    """Map an HTTP error response to the deliberate exception hierarchy.

    Decision order, from most to least trustworthy:

    1. The status itself for 401/403 — those statuses mean credentials, period.
    2. The structured ``type``/``code`` fields (machine-authored, read broadly,
       token-aware) for auth phrases — a proxy can answer 404 to an auth problem
       and a server can put an auth code in a 400.
    3. Model semantics: a token-wise "model" mention. This comes before any
       free-text auth matching, so "permission denied loading model" or
       "model author not found" on a 500 is a model failure that falls back,
       not a credential failure that aborts the run.
    4. A narrow, token-aware auth phrase list against the free-text message.
    5. Status semantics: 408 and 429/5xx are remote availability failures (the
       408 check comes after auth and model matching so phrasing still wins), a
       bare 404 is a missing endpoint, other 4xx are locally invalid requests.

    Classification uses the remote fields raw, but the rendered exception text
    never does: ``message``, ``code`` and ``type`` pass through
    `_sanitize_remote_detail` (length cap, control-character strip, and redaction
    of every string in ``secrets`` — the caller passes the configured API key) so
    a server that reflects the received Authorization header cannot make tapeback
    repeat the credential into a status line, the run log, or the terminal.
    """
    error_type, error_code, message = _error_fields(payload)
    structured_tokens = _tokens(f"{error_type} {error_code}")
    message_tokens = _tokens(message)

    safe_type = _sanitize_remote_detail(error_type, secrets)
    safe_code = _sanitize_remote_detail(error_code, secrets)
    safe_message = _sanitize_remote_detail(message, secrets)

    if _is_auth_failure(status, structured_tokens, message_tokens):
        return LemonadeAuthenticationError(
            f"Lemonade rejected the credentials (HTTP {status}): {safe_message or safe_code}"
        )
    if "model" in structured_tokens or "model" in message_tokens:
        return LemonadeModelError(
            f"Lemonade cannot serve the model (HTTP {status}): {safe_code or safe_message}"
        )
    if status == _HTTP_REQUEST_TIMEOUT:
        # A proxy or server gave up waiting for the request — a transient
        # availability failure on the remote side, in the same class as a read
        # timeout: fallback-eligible, and never resubmitted to Lemonade.
        return LemonadeInferenceTimeout(
            f"Lemonade response did not finish within the configured timeout "
            f"(HTTP {status}): {safe_message or safe_type or 'no detail'}"
        )
    if status == _HTTP_TOO_MANY_REQUESTS or status >= _HTTP_INTERNAL_ERROR:
        return LemonadeUnavailableError(
            f"Lemonade server failure (HTTP {status}): {safe_message or safe_type or 'no detail'}"
        )
    if status == _HTTP_NOT_FOUND:
        return LemonadeCapabilityError(
            f"Lemonade has no transcription endpoint at this URL (HTTP 404): {safe_message}"
        )
    if _HTTP_BAD_REQUEST <= status < _HTTP_INTERNAL_ERROR:
        # A client-side problem with the request itself: retrying or falling back
        # cannot fix a locally invalid configuration.
        return LemonadeConfigurationError(
            f"Lemonade rejected the request as invalid (HTTP {status}): {safe_message or safe_type}"
        )
    return LemonadeUnavailableError(
        f"Lemonade server failure (HTTP {status}): {safe_message or safe_type or 'no detail'}"
    )


def _multipart_body(
    fields: list[tuple[str, str]],
    file_field: str,
    filename: str,
    content_type: str,
    data: bytes,
) -> tuple[bytes, str]:
    """Encode a multipart/form-data body. Small, standard, and dependency-free.

    Field values are interpolated into MIME headers, so a value carrying a quote
    or CRLF cannot be allowed through — it would inject part headers. The audio
    part's name is always the fixed opaque `_UPLOAD_FILENAME`.
    """
    boundary = f"tapeback-{uuid.uuid4().hex}"
    for _name, value in fields:
        if any(ch in value for ch in '"\r\n'):
            raise LemonadeConfigurationError(
                "A Lemonade request field contains a character that cannot appear "
                "in a multipart header (quote or CR/LF)."
            )
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
#
# Both openers below install _NoRedirectHandler in place of urllib's default
# redirect handler. A redirect would move the request — carrying the bearer
# credential in its Authorization header — to a URL the server chose, possibly
# cross-origin or an https→http downgrade, and urllib converts redirected POSTs
# to GETs while keeping the auth header. The default handler also consumes 30x
# response bodies before _send() can apply _MAX_RESPONSE_BYTES. tapeback never
# follows redirects: every 3xx surfaces as an HTTPError and is classified as a
# sanitized Lemonade error.
class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect: returning None makes urllib raise HTTPError."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _SafeProxyHandler(urllib.request.ProxyHandler):
    """Allow HTTPS destinations only through explicit plaintext CONNECT proxies.

    urllib does not retain the proxy URL scheme after ``Request.set_proxy()``.
    Without this check an ``https://`` proxy reaches our HTTPS connection looking
    exactly like an ordinary HTTP CONNECT proxy. A direct TLS connection to that
    proxy can then receive the origin Authorization header and multipart body.
    Nested TLS is deliberately out of scope for the stdlib-only transport, so
    unsupported or ambiguous proxy schemes fail before a socket is opened.
    """

    def proxy_open(self, req: urllib.request.Request, proxy: str, type: str) -> Any:
        if req.host and urllib.request.proxy_bypass(req.host):
            return None
        if type == "https":
            try:
                proxy_scheme = urllib.parse.urlsplit(proxy).scheme.lower()
            except ValueError:
                proxy_scheme = ""
            if proxy_scheme != "http":
                raise LemonadeConfigurationError(
                    "Lemonade HTTPS endpoints require an explicit http:// CONNECT proxy. "
                    "TLS-to-proxy, scheme-less, and other proxy URLs are refused before "
                    "credentials or audio are sent."
                )
        return super().proxy_open(req, proxy, type)


def _deadline_timeout(deadline: float, phase: str) -> float:
    """Remaining request budget for one blocking operation."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"Lemonade request exceeded end-to-end deadline before {phase}")
    return max(_MIN_SOCKET_TIMEOUT_SECONDS, remaining)


def _resolve_with_deadline(host: str, port: int, deadline: float) -> list[tuple[Any, ...]]:
    """Resolve an address without allowing platform DNS to outlive the request.

    getaddrinfo() cannot be cancelled portably. A timed-out daemon may finish in
    the background, but the two-slot semaphore prevents repeated failures from
    creating an unbounded number of resolver threads.
    """
    resolver_slots = _DNS_RESOLVER_SLOTS
    if not resolver_slots.acquire(timeout=_deadline_timeout(deadline, "DNS resolver capacity")):
        raise TimeoutError("Lemonade request exceeded end-to-end deadline waiting for DNS")

    results: queue.Queue[tuple[list[tuple[Any, ...]] | None, BaseException | None]] = queue.Queue(
        maxsize=1
    )

    def resolve() -> None:
        try:
            addresses = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
            results.put((addresses, None))
        except BaseException as exc:  # propagated on the request thread
            results.put((None, exc))
        finally:
            resolver_slots.release()

    worker = threading.Thread(target=resolve, name="lemonade-dns-resolver", daemon=True)
    try:
        worker.start()
    except BaseException:
        resolver_slots.release()
        raise
    try:
        addresses, error = results.get(timeout=_deadline_timeout(deadline, "DNS resolution"))
    except queue.Empty:
        raise TimeoutError("Lemonade request exceeded end-to-end deadline during DNS") from None
    if error is not None:
        raise error
    if time.monotonic() >= deadline:
        raise TimeoutError("Lemonade request exceeded end-to-end deadline during DNS")
    return addresses or []


def _create_deadline_connection(
    address: tuple[str, int],
    deadline: float,
    source_address: tuple[str, int] | None,
) -> socket.socket:
    """Resolve and connect like socket.create_connection(), under one deadline."""
    host, port = address
    addresses = _resolve_with_deadline(host, port, deadline)
    last_error: OSError | None = None
    for family, socktype, proto, _canonname, socket_address in addresses:
        sock: socket.socket | None = None
        try:
            sock = socket.socket(family, socktype, proto)
            sock.settimeout(_deadline_timeout(deadline, "TCP connection"))
            if source_address:
                sock.bind(source_address)
            sock.connect(socket_address)
            with contextlib.suppress(OSError):
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError as exc:
            last_error = exc
            if sock is not None:
                sock.close()
    if last_error is not None:
        raise last_error
    raise OSError("getaddrinfo returned no usable addresses")


class _DeadlineFile:
    """File wrapper that refreshes the socket budget before buffered reads."""

    def __init__(self, file: Any, owner: _DeadlineSocket) -> None:
        self._file = file
        self._owner = owner

    def read(self, *args: Any, **kwargs: Any) -> Any:
        self._owner._update_timeout()
        return self._file.read(*args, **kwargs)

    def readline(self, *args: Any, **kwargs: Any) -> Any:
        self._owner._update_timeout()
        return self._file.readline(*args, **kwargs)

    def readinto(self, *args: Any, **kwargs: Any) -> Any:
        self._owner._update_timeout()
        return self._file.readinto(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._file, name)


class _DeadlineSocket:
    """Socket wrapper that recalculates and updates the socket timeout before every operation."""

    def __init__(self, sock: Any, deadline: float) -> None:
        self._sock = sock
        self._deadline = deadline

    def _update_timeout(self) -> None:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Lemonade request exceeded end-to-end deadline")
        self._sock.settimeout(max(_MIN_SOCKET_TIMEOUT_SECONDS, remaining))

    def send(self, *args: Any, **kwargs: Any) -> int:
        self._update_timeout()
        return self._sock.send(*args, **kwargs)

    def sendall(self, *args: Any, **kwargs: Any) -> None:
        self._update_timeout()
        return self._sock.sendall(*args, **kwargs)

    def recv(self, *args: Any, **kwargs: Any) -> bytes:
        self._update_timeout()
        return self._sock.recv(*args, **kwargs)

    def recv_into(self, *args: Any, **kwargs: Any) -> int:
        self._update_timeout()
        return self._sock.recv_into(*args, **kwargs)

    def makefile(self, *args: Any, **kwargs: Any) -> Any:
        self._update_timeout()
        return _DeadlineFile(self._sock.makefile(*args, **kwargs), self)

    def settimeout(self, timeout: float | None) -> None:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Lemonade request exceeded end-to-end deadline")
        bound = max(_MIN_SOCKET_TIMEOUT_SECONDS, remaining)
        self._sock.settimeout(bound if timeout is None else min(timeout, bound))

    def close(self) -> None:
        return self._sock.close()

    def unwrap(self) -> Any:
        """Return the real socket for SSLContext.wrap_socket()."""
        return self._sock

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sock, name)


class _DeadlineHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection enforcing an absolute deadline across connect, write, and headers."""

    def __init__(self, *args: Any, deadline: float | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._deadline = deadline

    def connect(self) -> None:
        if self._deadline is None:
            return super().connect()
        source_address = getattr(self, "source_address", None)
        sock = _create_deadline_connection((self.host, self.port), self._deadline, source_address)
        self.sock = _DeadlineSocket(sock, self._deadline)


class _DeadlineHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection enforcing an absolute deadline across connect, TLS, write, and headers."""

    def __init__(self, *args: Any, deadline: float | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._deadline = deadline

    def connect(self) -> None:
        if self._deadline is None:
            return super().connect()
        source_address = getattr(self, "source_address", None)
        raw_sock = _create_deadline_connection(
            (self.host, self.port), self._deadline, source_address
        )
        try:
            self.sock = _DeadlineSocket(raw_sock, self._deadline)
            tunnel_host = getattr(self, "_tunnel_host", None)
            if tunnel_host:
                self._tunnel()  # ty: ignore[unresolved-attribute]
                server_hostname = tunnel_host
            else:
                server_hostname = self.host
            raw_sock = self.sock.unwrap()
            raw_sock.settimeout(_deadline_timeout(self._deadline, "TLS handshake"))
            self.sock = None
            context = self._context  # ty: ignore[unresolved-attribute]
            tls_sock = context.wrap_socket(raw_sock, server_hostname=server_hostname)
            self.sock = _DeadlineSocket(tls_sock, self._deadline)
        except BaseException:
            if self.sock is not None:
                self.sock.close()
            else:
                raw_sock.close()
            raise


class _DeadlineHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req: urllib.request.Request) -> Any:
        deadline = getattr(req, "_tapeback_deadline", None)
        if deadline is not None:
            return self.do_open(
                lambda host, **kw: _DeadlineHTTPConnection(host, deadline=deadline, **kw),
                req,
            )
        return super().http_open(req)


class _DeadlineHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req: urllib.request.Request) -> Any:
        deadline = getattr(req, "_tapeback_deadline", None)
        if deadline is not None:
            context = getattr(self, "_context", None)
            return self.do_open(
                lambda host, **kw: _DeadlineHTTPSConnection(host, deadline=deadline, **kw),
                req,
                context=context,
            )
        return super().https_open(req)


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


def _remaining(deadline: float) -> float:
    """Socket timeout for the next blocking operation: the remaining budget."""
    return max(_MIN_SOCKET_TIMEOUT_SECONDS, deadline - time.monotonic())


def _extract_socket(obj: Any) -> Any:
    """Traverse response wrapper layers to find the underlying socket."""
    curr = obj
    visited = set()
    while curr is not None and id(curr) not in visited:
        visited.add(id(curr))
        has_sock_fn = hasattr(curr, "getsockname") or hasattr(curr, "fileno")
        if hasattr(curr, "settimeout") and has_sock_fn:
            return curr
        if hasattr(curr, "_sock"):
            return curr._sock
        if hasattr(curr, "raw"):
            curr = curr.raw
        elif hasattr(curr, "fp"):
            curr = curr.fp
        else:
            break
    return None


def _bound_socket_timeout(fp: Any, deadline: float) -> None:
    """Set the underlying socket's inactivity timeout to the remaining budget.

    urllib receives the timeout once per open; every later blocking read would
    otherwise restart a full socket timeout no matter how little budget is left.
    Best-effort: a response object without a reachable socket (fakes, exotic
    transports) keeps its previous timeout, and the deadline check between reads
    still applies.
    """
    sock = _extract_socket(fp)
    if sock is None:
        return
    try:
        sock.settimeout(_remaining(deadline))
    except (OSError, AttributeError):
        return


def _read_bounded(fp: Any, limit: int, *, deadline: float | None = None) -> bytes:
    """Read at most ``limit + 1`` bytes from ``fp``, so overflow is detectable.

    A single ``read(n)`` is not a size bound: a stream may legitimately return a
    short read while more data remains, so the read loops until the cap is
    exceeded or the stream ends.

    ``deadline`` is a monotonic end-to-end bound: when given, expiry raises
    ``LemonadeInferenceTimeout`` even while the peer is still trickling bytes —
    the socket timeout is an inactivity bound, not a promise that the total
    request finishes on time. Before every read the socket's inactivity timeout
    is reset to the remaining budget, so a single stalled read cannot block for
    a full socket timeout on top of an already-expired deadline.
    """
    pieces: list[bytes] = []
    remaining = limit + 1
    while remaining > 0:
        if deadline is not None and time.monotonic() >= deadline:
            raise LemonadeInferenceTimeout(
                "Lemonade response did not finish within the configured timeout"
            )
        if deadline is not None:
            _bound_socket_timeout(fp, deadline)
        piece = fp.read(min(65536, remaining))
        if not piece:
            break
        pieces.append(piece)
        remaining -= len(piece)
    return b"".join(pieces)


def _raise_for_http_error(
    exc: urllib.error.HTTPError, api_key: str, *, deadline: float | None = None
) -> NoReturn:
    """Map one HTTPError onto the error hierarchy. Never returns.

    Redirects are never followed (see `_NoRedirectHandler`), so a 3xx arrives
    here as an HTTPError and is classified as a sanitized error — the
    server-chosen ``Location`` header is never echoed. Error bodies are read
    under the response cap and classified with ``api_key`` as a redaction
    secret, so a response that reflects the Authorization header cannot persist
    the credential through the exception message. ``deadline`` bounds the error
    body read like any other body read: a redirect or error page that never
    finishes must trip the request's total deadline, not hang past it.
    """
    if _HTTP_MULTIPLE_CHOICES <= exc.code < _HTTP_BAD_REQUEST:
        raise LemonadeUnavailableError(
            f"Lemonade responded with a redirect (HTTP {exc.code}); tapeback "
            "never follows redirects — check the configured TAPEBACK_LEMONADE_URL."
        ) from exc
    raw = _read_bounded(exc, _MAX_RESPONSE_BYTES, deadline=deadline)
    if len(raw) > _MAX_RESPONSE_BYTES:
        # Classify by status alone rather than buffer a giant error body.
        payload = None
    else:
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except (ValueError, OSError, RecursionError):
            payload = None
    raise classify_http_failure(exc.code, payload, secrets=(api_key,) if api_key else ()) from exc


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
    frames_per_bytes = max(1, (_MAX_CHUNK_BYTES - _REQUEST_OVERHEAD_BYTES) // frame_bytes - overlap)
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

    Structural rule: the URL is rebuilt from its validated components, never
    passed through. Userinfo (``https://user:pass@host``) is rejected outright —
    it would otherwise be retained and printed by ``tapeback status`` — and query
    strings and fragments are rejected because the base URL is kept verbatim while
    ``/v1/audio/transcriptions`` is appended, which would misroute the request to
    a path the operator never configured. Scheme and hostname are lowercased, a
    default port (80/443) is dropped, and a trailing slash is removed.

    Reporting rule: the raw configured value is never echoed in an error message —
    an invalid URL may carry embedded credentials or terminal-control characters,
    and the failure text is shown by ``tapeback status`` and captured in run logs.
    """
    # urlparse deliberately defers a few validations (notably unmatched IPv6
    # brackets, NFKC-sensitive netloc delimiters, and ports) to lazy properties.
    # Keep *all* parsing and property access inside this boundary: configuration
    # failures must neither echo a potentially secret URL nor expose parser text.
    cleaned_raw = raw.strip()
    if not cleaned_raw.isascii() or any(ch.isspace() or not ch.isprintable() for ch in cleaned_raw):
        raise LemonadeConfigurationError(
            "TAPEBACK_LEMONADE_URL contains invalid characters (whitespace, control "
            "characters, or non-ASCII)."
        )
    try:
        parsed = urllib.parse.urlparse(cleaned_raw)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        username = parsed.username
        password = parsed.password
        port = parsed.port
    except ValueError:
        raise LemonadeConfigurationError(
            "TAPEBACK_LEMONADE_URL is not a valid http(s) URL. The configured value "
            "is not shown because it may contain credentials."
        ) from None
    if scheme not in ("http", "https") or not hostname:
        # The raw value is deliberately not echoed: it may carry embedded
        # credentials or control characters, and this message is printed by
        # `tapeback status` and captured in run logs.
        raise LemonadeConfigurationError(
            "TAPEBACK_LEMONADE_URL is not a valid http(s) URL — it must start with "
            "http:// or https:// and name a host. The configured value is not shown "
            "because it may contain credentials."
        )
    if not hostname.isascii():
        raise LemonadeConfigurationError(
            "TAPEBACK_LEMONADE_URL hostname must contain only ASCII characters."
        )
    if username is not None or password is not None:
        raise LemonadeConfigurationError(
            "TAPEBACK_LEMONADE_URL must not embed credentials (user:password@host): "
            "they would be displayed by status and kept in the configured URL. Pass "
            "the token with TAPEBACK_LEMONADE_API_KEY instead."
        )
    if parsed.query or parsed.fragment:
        raise LemonadeConfigurationError(
            "TAPEBACK_LEMONADE_URL must be a bare base URL: query strings and "
            "fragments are kept while '/v1/audio/transcriptions' is appended, so "
            "the request could target a path you never configured."
        )
    if scheme == "http" and not _is_loopback_host(hostname):
        raise LemonadeConfigurationError(
            f"TAPEBACK_LEMONADE_URL uses plaintext http for the non-loopback host "
            f"{_sanitize_remote_detail(hostname)!r}: meeting audio and the "
            "bearer credential would travel unprotected. Use https:// for remote "
            "servers (plain http is allowed only for localhost, 127.0.0.0/8 and ::1)."
        )
    host = hostname.lower()
    if port is not None and port == _HTTP_DEFAULT_PORTS.get(scheme):
        port = None
    host_part = f"[{host}]" if ":" in host else host
    netloc = host_part if port is None else f"{host_part}:{port}"
    path = parsed.path.rstrip("/")
    if not path.isascii() or any(ch.isspace() or not ch.isprintable() for ch in path):
        raise LemonadeConfigurationError(
            "TAPEBACK_LEMONADE_URL contains invalid characters in its path."
        )
    return urllib.parse.urlunsplit((scheme, netloc, path, "", ""))


def _finite_number(value: Any) -> float | None:
    """The value as a finite float, or None when it is not one.

    Booleans are ``int`` in Python but are never numbers in a JSON schema; NaN
    and infinities pass naive ``isinstance`` checks and then poison sorting,
    duration arithmetic, and every timestamp persisted into the resume cache.
    All three are rejected here, once, for every numeric field of a response.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    if not math.isfinite(number):
        return None
    return number


def _usable_timestamp(value: Any, name: str) -> float:
    """A finite, non-negative timestamp, or a sanitized capability error."""
    number = _finite_number(value)
    if number is None or number < 0.0:
        raise LemonadeCapabilityError(
            f"Lemonade returned a response with an unusable {name} "
            "(missing, not a number, boolean, non-finite, or negative)"
        )
    return number


def _require_segments(
    payload: dict[str, Any], upper_bound: float | None = None
) -> list[dict[str, Any]]:
    """Return the payload's timestamped segments, or reject the response outright.

    Tapeback's pipeline needs segment timestamps for speaker labelling and vault
    timing, so a text-only response is unusable even when the prose is good. This is
    where FLM-style compact responses are rejected in full.

    Every timestamp is strictly validated (finite number, ``0 <= start <= end``),
    and when ``upper_bound`` is given (a chunk's audio length) an ``end`` beyond it
    — modulo a small boundary slack — is also rejected: a hostile or broken server
    must not be able to write timestamps past the recording into the transcript or
    the resume cache.
    """
    raw = payload.get("segments")
    if isinstance(raw, list) and raw:
        if len(raw) > _MAX_RESPONSE_SEGMENTS:
            raise LemonadeCapabilityError(
                f"Lemonade returned too many segments in one response "
                f"({len(raw)} > {_MAX_RESPONSE_SEGMENTS})"
            )
        for item in raw:
            if not isinstance(item, dict):
                raise LemonadeCapabilityError(
                    "Lemonade returned segments without usable timestamps"
                )
            start = _usable_timestamp(item.get("start"), "segment start")
            end = _usable_timestamp(item.get("end"), "segment end")
            if end < start:
                raise LemonadeCapabilityError(
                    "Lemonade returned a segment whose end precedes its start"
                )
            if upper_bound is not None and end > upper_bound + _TIMESTAMP_SLACK_SECONDS:
                raise LemonadeCapabilityError(
                    "Lemonade returned a segment ending past the audio it was sent"
                )
            text = str(item.get("text") or "")
            if len(text) > _MAX_SEGMENT_TEXT_CHARS:
                raise LemonadeCapabilityError(
                    f"Lemonade returned a segment exceeding the text size limit "
                    f"({len(text)} > {_MAX_SEGMENT_TEXT_CHARS})"
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


_CONTENT_LENGTH_RE = re.compile(r"[0-9]{1,15}")


def _parse_content_length(value: str) -> int:
    """Parse a Content-Length header as bounded ASCII digits, or refuse it.

    ``str.isdigit()`` is not a safety check: it accepts non-ASCII characters
    (``"²"`` is "a digit" but ``int("²")`` raises ValueError) and a long
    enough decimal string trips CPython's integer-string conversion limit
    (``sys.set_int_max_str_digits``, default 4300 digits) — either way ``int()``
    raised outside the error hierarchy, crashing transcribe/status instead of
    classifying the response as unavailable. Only 1-15 ASCII digits are
    accepted (far past any legitimate response size) and everything else is a
    sanitized LemonadeUnavailableError.
    """
    text = value.strip()
    if not _CONTENT_LENGTH_RE.fullmatch(text):
        detail = _sanitize_remote_detail(text) or "unprintable"
        raise LemonadeUnavailableError(
            f"Lemonade declared a malformed Content-Length ({detail}); "
            "refusing to read the response."
        )
    try:
        return int(text)
    except ValueError:  # pragma: no cover — bounded ASCII digits cannot raise
        raise LemonadeUnavailableError(
            "Lemonade declared an unparseable Content-Length; refusing to read the response."
        ) from None


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
                duration = self._transcribe_wav(audio_path, params, stage, on_status, state)
            else:
                # Not a parseable WAV: send it whole. The pipeline supplies PCM WAV,
                # so this is the tolerant path for unusual but valid inputs. The
                # whole body is buffered for multipart assembly (peak ~2x for the
                # copy), so it is bounded by the same cap as a chunk — a bigger
                # non-chunkable input is refused here and can fall back to
                # faster-whisper, which decodes formats the chunker cannot.
                size = audio_path.stat().st_size
                max_payload_bytes = _MAX_CHUNK_BYTES - _REQUEST_OVERHEAD_BYTES
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
                state.absorb(payload, 0.0, 0.0, 0)
        except KeyboardInterrupt:
            # Ctrl+C means keep what finished, mark the result partial, and stop —
            # never a fallback, never a cache write.
            partial = True
            on_status(
                f"Interrupted during '{stage}' — keeping the {len(state.segments)} "
                "segments transcribed so far."
            )

        if duration == 0.0 and state.segments:
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
