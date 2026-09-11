"""Lemonade HTTP transport: deadlines, bounded reads, request assembly.

Owns the socket-level machinery that keeps a broken or hostile server from
stalling or exhausting the client: an end-to-end request deadline shared by
connect, upload, and every read; DNS-resolution slots; bounded response
reads; multipart request assembly; and the HTTP-error mapping. Redirects
are refused (`_NoRedirectHandler`) and loopback bypasses the process-wide
proxy (`_SafeProxyHandler`); the opener objects wiring these together stay
in the facade, so there is one place to patch and one place to audit."""

from __future__ import annotations

import contextlib
import http.client
import json
import queue
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, NoReturn

from tapeback._lemonade_errors import (
    _HTTP_BAD_REQUEST,
    _HTTP_MULTIPLE_CHOICES,
    LemonadeConfigurationError,
    LemonadeInferenceTimeout,
    LemonadeUnavailableError,
    classify_http_failure,
)

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
