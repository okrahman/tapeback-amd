"""Lemonade endpoint URL normalization and trust-boundary classification.

Owns the structural URL rules (scheme, embedded credentials, default
ports) and the loopback recognition that decides whether plain `http://`
is acceptable and whether proxy bypass applies. The trust boundary is
enforced here: remote endpoints must be https before a request is built."""

from __future__ import annotations

import ipaddress
import urllib.error
import urllib.parse
import urllib.request

from tapeback._lemonade_errors import LemonadeConfigurationError, _sanitize_remote_detail

# Default ports are dropped from the normalized base URL.
_HTTP_DEFAULT_PORTS = {"http": 80, "https": 443}


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
