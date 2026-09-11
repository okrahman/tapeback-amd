"""Lemonade error hierarchy and error-text handling.

Owns the exception taxonomy the facade's fallback rule is built on — only
`LemonadeFallbackError` may trigger a faster-whisper fallback — plus the
sanitization/redaction of server-supplied error text, the auth-failure
detectors, the `Content-Length` bounds check, and the HTTP-failure
classifier that maps a hostile response onto that taxonomy. Nothing here
touches the network."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


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

    The status and the structured type/code fields decide first. The narrow
    message auth phrases are matched BEFORE the generic model override, so
    "invalid API key for model Whisper" is a credential failure that aborts the
    run, not a model failure that silently falls back. Only model messages that
    carry no narrow auth phrase — "permission denied loading model", "model
    author not found" — stay model failures that fall back.
    """
    if status in (_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN):
        return True
    if any(_has_phrase(structured_tokens, phrase) for phrase in _STRUCTURED_AUTH_PHRASES):
        return True
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
    3. A narrow, token-aware auth phrase list against the free-text message —
       matched BEFORE the model semantics, so "invalid API key for model
       Whisper" is a credential failure that aborts, not a model failure that
       falls back.
    4. Model semantics: a token-wise "model" mention with no narrow auth
       phrase. "permission denied loading model" or "model author not found"
       on a 500 is therefore a model failure that falls back, not a credential
       failure that aborts the run.
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
