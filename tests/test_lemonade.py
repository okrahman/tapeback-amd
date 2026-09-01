"""Lemonade backend: HTTP transport, chunking, language handling, error mapping."""

import io
import json
import struct
import threading
import urllib.error
import urllib.request
import wave
from email.message import Message
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

import tapeback._lemonade as lemon
from tapeback._lemonade import (
    DEDUP_POLICY_VERSION,
    LemonadeAuthenticationError,
    LemonadeBackend,
    LemonadeCapabilityError,
    LemonadeConfigurationError,
    LemonadeError,
    LemonadeInferenceTimeout,
    LemonadeModelError,
    LemonadeUnavailableError,
    classify_http_failure,
    normalize_language,
)
from tapeback.settings import Settings

# --- helpers ---


def write_wav(path: Path, duration_s: float, rate: int = 16000) -> None:
    """A silent PCM WAV of the requested duration."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * int(duration_s * rate))


def lemon_settings(tmp_path, **overrides) -> Settings:
    base: dict[str, Any] = {
        "transcription_backend": "lemonade",
        "resume_cache_dir": tmp_path / "resume",
        # 1s chunks and 0.5s overlap make the chunk arithmetic easy to assert.
        "lemonade_chunk_seconds": 1.0,
        "lemonade_overlap_seconds": 0.5,
    }
    base.update(overrides)
    return Settings(**base)


def verbose_json(segments, language="english", probability=0.87) -> bytes:
    payload = {
        "text": " ".join(s.get("text", "") for s in segments),
        "segments": segments,
        "language": language,
    }
    if probability is not None:
        payload["language_probability"] = probability
    return json.dumps(payload).encode()


def seg(start, end, text="hello"):
    return {"start": start, "end": end, "text": text, "words": []}


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self._consumed = False

    def read(self, n: int = -1) -> bytes:
        # Emulate stream semantics: one read yields the body, further reads EOF.
        if self._consumed:
            return b""
        self._consumed = True
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def install_urlopen(monkeypatch, bodies):
    """Queue responses (bytes or exception instances) for successive requests.

    Routes BOTH HTTP paths tapeback can take — the default opener and the loopback
    no-proxy opener — through the same fake, so tests intercept every request
    regardless of which opener the backend chose.
    """
    calls: list[object] = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        index = min(len(calls) - 1, len(bodies) - 1)
        body = bodies[index]
        if isinstance(body, BaseException):
            raise body
        if isinstance(body, (bytes, bytearray)):
            return _FakeResponse(body)
        return body  # a pre-built response-like object, passed through

    class _FakeOpener:
        def open(self, request, timeout=None):
            return fake_urlopen(request, timeout=timeout)

    monkeypatch.setattr("tapeback._lemonade._DEFAULT_OPENER", _FakeOpener())
    monkeypatch.setattr("tapeback._lemonade._NO_PROXY_OPENER", _FakeOpener())
    return calls


def http_error(status: int, payload: dict | str | bytes = "") -> urllib.error.HTTPError:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return urllib.error.HTTPError("http://x", status, "err", Message(), io.BytesIO(body))


# --- construction and configuration ---


def test_invalid_url_is_a_configuration_error(tmp_path):
    with pytest.raises(LemonadeConfigurationError) as exc_info:
        LemonadeBackend(lemon_settings(tmp_path, lemonade_url="not-a-url"))
    # The raw configured value is never echoed: it may carry credentials or
    # terminal-control characters, and the message reaches status and run logs.
    assert "not-a-url" not in str(exc_info.value)
    with pytest.raises(LemonadeConfigurationError):
        LemonadeBackend(lemon_settings(tmp_path, lemonade_url="ftp://host:1"))


def test_invalid_url_error_never_echoes_credentials_or_control_characters(tmp_path):
    """Raw input must not survive into the invalid-scheme/invalid-port messages."""
    # Invalid scheme carrying a terminal-control sequence.
    with pytest.raises(LemonadeConfigurationError) as exc_info:
        LemonadeBackend(lemon_settings(tmp_path, lemonade_url="ht\x1b]0;pwned\x07tp://host"))
    message = str(exc_info.value)
    assert "\x1b" not in message
    assert "\x07" not in message
    # Invalid port: the fixed message carries neither the host nor the port value.
    with pytest.raises(LemonadeConfigurationError) as exc_info:
        LemonadeBackend(lemon_settings(tmp_path, lemonade_url="https://host:99999"))
    assert "99999" not in str(exc_info.value)
    # The userinfo rejection never contained the raw URL, but pin it down:
    with pytest.raises(LemonadeConfigurationError) as exc_info:
        LemonadeBackend(lemon_settings(tmp_path, lemonade_url="http://alice:pw@127.0.0.1:13305"))
    assert "alice" not in str(exc_info.value) and "pw@" not in str(exc_info.value)


def test_malformed_api_key_is_a_configuration_error(tmp_path):
    with pytest.raises(LemonadeConfigurationError):
        LemonadeBackend(lemon_settings(tmp_path, lemonade_api_key=SecretStr("two tokens")))
    with pytest.raises(LemonadeConfigurationError):
        LemonadeBackend(lemon_settings(tmp_path, lemonade_api_key=SecretStr("bad\ntoken")))


def test_describe_names_model_and_endpoint(tmp_path):
    backend = LemonadeBackend(lemon_settings(tmp_path))
    described = backend.describe()
    assert "Whisper-Large-v3-Turbo" in described
    assert "127.0.0.1:13305" in described


# --- cache fingerprint ---


@pytest.mark.parametrize(
    "field,value",
    [
        ("lemonade_url", "http://127.0.0.1:9999"),
        ("lemonade_model", "Whisper-Small"),
        ("language", "de"),
        ("lemonade_chunk_seconds", 120.0),
        ("lemonade_overlap_seconds", 0.9),
    ],
)
def test_fingerprint_changes_with_every_output_shaping_setting(tmp_path, field, value):
    backend = LemonadeBackend(lemon_settings(tmp_path))
    other = LemonadeBackend(lemon_settings(tmp_path, **{field: value}))
    assert backend.cache_fingerprint() != other.cache_fingerprint()


def test_fingerprint_excludes_secrets_timeouts_and_hardware(tmp_path):
    """Nothing that fails to change transcription output may invalidate the cache."""
    plain = LemonadeBackend(lemon_settings(tmp_path))
    with_key = LemonadeBackend(lemon_settings(tmp_path, lemonade_api_key=SecretStr("token-abc")))
    with_timeout = LemonadeBackend(lemon_settings(tmp_path, lemonade_timeout_seconds=1.0))
    assert plain.cache_fingerprint() == with_key.cache_fingerprint()
    assert plain.cache_fingerprint() == with_timeout.cache_fingerprint()
    # And the key itself is nowhere in the fingerprint material.
    assert "token-abc" not in plain.cache_fingerprint()


def test_fingerprint_tracks_the_dedup_policy_version(tmp_path, monkeypatch):
    backend = LemonadeBackend(lemon_settings(tmp_path))
    before = backend.cache_fingerprint()
    monkeypatch.setattr("tapeback._lemonade.DEDUP_POLICY_VERSION", DEDUP_POLICY_VERSION + 1)
    assert backend.cache_fingerprint() != before


def test_url_normalization_makes_trailing_slash_equivalent(tmp_path):
    backend = LemonadeBackend(lemon_settings(tmp_path, lemonade_url="http://127.0.0.1:1/"))
    other = LemonadeBackend(lemon_settings(tmp_path, lemonade_url="http://127.0.0.1:1"))
    assert backend.cache_fingerprint() == other.cache_fingerprint()


# --- requests ---


def test_multipart_request_carries_model_format_and_file(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    calls = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])

    LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    request = calls[0]
    assert request.full_url == "http://127.0.0.1:13305/v1/audio/transcriptions"
    body = request.data.decode("utf-8", errors="replace")
    assert 'name="model"' in body and "Whisper-Large-v3-Turbo" in body
    assert 'name="response_format"' in body and "verbose_json" in body
    assert 'name="file"' in body and 'filename="audio.wav"' in body
    assert request.headers["Content-type"].startswith("multipart/form-data")


def test_bearer_token_sent_only_when_configured(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)

    calls = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])
    LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)
    assert "Authorization" not in calls[0].headers

    calls = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])
    backend = LemonadeBackend(lemon_settings(tmp_path, lemonade_api_key=SecretStr("tok-1")))
    backend.transcribe(wav)
    assert calls[0].headers["Authorization"] == "Bearer tok-1"


def test_no_hotwords_or_prompt_field_is_sent(tmp_path, monkeypatch):
    """Tapeback's hotwords are a faster-whisper decoder bias, not a portable knob."""
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    calls = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])
    backend = LemonadeBackend(lemon_settings(tmp_path, hotwords="tapeback, Acme"))
    backend.transcribe(wav)
    assert "prompt" not in calls[0].data.decode("utf-8", errors="replace")


# --- chunking, offsets, dedup, progress ---


def test_long_wav_is_chunked_with_overlap_and_shifted_timestamps(tmp_path, monkeypatch):
    wav = tmp_path / "long.wav"
    write_wav(wav, 2.5)  # 1s chunks, 0.5s overlap -> 3 chunks
    responses = [
        verbose_json([seg(0.0, 0.9, "one")]),
        verbose_json([seg(0.0, 0.2, "overlap-echo"), seg(0.6, 1.4, "two")]),
        verbose_json([seg(0.5, 0.6, "three")]),
    ]
    calls = install_urlopen(monkeypatch, responses)

    segments, info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    assert len(calls) == 3
    # Chunk 2 starts its audio at 0.5s (1.0 core minus 0.5 overlap): the echo of
    # "overlap-echo" is centred at 0.6 < core start 1.0, so dedup drops it.
    assert [s.text for s in segments] == ["one", "two", "three"]
    # File-relative times: chunk 2's 0.6..1.4 becomes 1.1..1.9, chunk 3's 0.0 becomes 2.0.
    assert segments[1].start == pytest.approx(1.1)
    assert segments[1].end == pytest.approx(1.9)
    assert segments[2].start == pytest.approx(2.0)
    assert info["duration"] == pytest.approx(2.5)
    assert info["partial"] is False


def test_chunk_requests_carry_the_pinned_language_after_detection(tmp_path, monkeypatch):
    wav = tmp_path / "long.wav"
    write_wav(wav, 2.5)
    responses = [
        verbose_json([seg(0.0, 0.9)], language="russian"),
        verbose_json([seg(0.6, 1.4)], language="whichever"),
        verbose_json([seg(0.0, 0.1)], language="whichever"),
    ]
    calls = install_urlopen(monkeypatch, responses)

    _segments, info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    first_body = calls[0].data.decode("utf-8", errors="replace")
    later_body = calls[1].data.decode("utf-8", errors="replace")
    assert 'name="language"' not in first_body  # detection is the server's job
    assert 'name="language"' in later_body and "ru" in later_body
    assert info["language"] == "ru"


def test_empty_leading_chunks_are_ignored_for_language_detection(tmp_path, monkeypatch):
    wav = tmp_path / "long.wav"
    write_wav(wav, 2.5)
    responses = [
        verbose_json([], language="", probability=None),
        verbose_json([seg(0.6, 1.4)], language="English"),
        verbose_json([seg(0.5, 0.6)], language="whatever"),
    ]
    calls = install_urlopen(monkeypatch, responses)

    segments, info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    assert 'name="language"' not in calls[0].data.decode(errors="replace")
    assert "en" in calls[2].data.decode(errors="replace")
    assert info["language"] == "en"
    assert [s.text for s in segments] == ["hello", "hello"]


def test_explicit_language_is_sent_on_every_request(tmp_path, monkeypatch):
    wav = tmp_path / "long.wav"
    write_wav(wav, 2.5)
    responses = [verbose_json([seg(0.0, 0.9)], language="de") for _ in range(3)]
    calls = install_urlopen(monkeypatch, responses)

    LemonadeBackend(lemon_settings(tmp_path, language="de")).transcribe(wav)

    assert all('name="language"' in c.data.decode(errors="replace") for c in calls)


def test_language_probability_only_when_validly_supplied(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)

    install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])
    _, info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)
    assert info["language_probability"] == pytest.approx(0.87)

    install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)], probability=None)])
    _, info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)
    assert "language_probability" not in info

    install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)], probability=7.5)])
    _, info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)
    assert "language_probability" not in info

    # Some Lemonade versions report the probability under detected_language_probability.
    alt = {
        "text": "hello",
        "segments": [seg(0.0, 0.4)],
        "language": "english",
        "detected_language_probability": 0.42,
    }
    install_urlopen(monkeypatch, [json.dumps(alt).encode()])
    _, info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)
    assert info["language_probability"] == pytest.approx(0.42)


def test_words_are_shifted_into_file_relative_time(tmp_path, monkeypatch):
    wav = tmp_path / "long.wav"
    write_wav(wav, 2.5)
    chunk2 = {
        "text": "two",
        "language": "russian",
        "segments": [
            {
                "start": 0.6,
                "end": 1.4,
                "text": " two",
                "words": [
                    {"start": 0.6, "end": 0.9, "word": "tw", "probability": 0.5},
                    {"start": 1.0, "end": 1.4, "word": "o", "probability": 0.9},
                ],
            }
        ],
    }
    install_urlopen(
        monkeypatch,
        [verbose_json([seg(0.0, 0.9)]), json.dumps(chunk2).encode(), verbose_json([])],
    )

    segments, _ = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    two = next(s for s in segments if s.text == "two")
    assert two.words is not None
    assert two.words[0].start == pytest.approx(1.1)
    assert two.words[1].end == pytest.approx(1.9)  # 0.5 audio_start + 1.4 chunk time


def test_word_timestamps_survive_an_offset_greater_than_the_slack(tmp_path, monkeypatch):
    """A later chunk's offset exceeds the slack: words must be validated against
    file-relative segment bounds, not compared across coordinate systems.

    Regression: chunk-relative word times were compared against file-relative
    segment bounds, so every word-bearing chunk after the first failed
    validation as a capability error whenever its offset exceeded
    _TIMESTAMP_SLACK_SECONDS — with the default 300 s chunks, that is every
    chunk. The small offsets of earlier tests hid the bug inside the slack.
    """
    wav = tmp_path / "long.wav"
    write_wav(wav, 2.5)
    chunk2 = {  # offset 1.5 s (1.0 s chunks, 0.5 s overlap) > 1.0 s slack
        "text": "three",
        "language": "russian",
        "segments": [
            {
                "start": 0.5,
                "end": 1.0,
                "text": " three",
                "words": [{"start": 0.5, "end": 0.7, "word": "three", "probability": 0.9}],
            }
        ],
    }
    install_urlopen(
        monkeypatch,
        [
            verbose_json([seg(0.0, 0.9)]),
            verbose_json([seg(0.6, 1.4)]),
            json.dumps(chunk2).encode(),
        ],
    )

    segments, _info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    three = next(s for s in segments if s.text == "three")
    assert three.words is not None
    assert three.words[0].start == pytest.approx(2.0)  # 1.5 offset + 0.5 chunk time
    assert three.words[0].end == pytest.approx(2.2)


def test_word_past_the_sent_audio_is_rejected_in_later_chunks(tmp_path, monkeypatch):
    """The sent-audio bound works in file-relative time for later chunks too.

    With the slack tightened so the arithmetic is observable: a word may outrun
    its chunk's audio by no more than the boundary slack, even in a chunk whose
    offset is large. Before the coordinate fix the bound was unreachable for
    every chunk after the first, because a chunk-relative end can never exceed a
    file-relative limit.
    """
    monkeypatch.setattr(lemon, "_TIMESTAMP_SLACK_SECONDS", 0.1)
    wav = tmp_path / "long.wav"
    write_wav(wav, 2.5)
    chunk1 = {  # offset 0.5 s; audio 0.5-2.0 s (1.5 s with overlap), chunk_end = 2.0
        "text": "two",
        "language": "russian",
        "segments": [
            {
                "start": 0.9,
                "end": 1.6,  # exactly chunk duration + slack: segment accepted
                "text": " two",
                "words": [{"start": 0.9, "end": 1.65, "word": "tw", "probability": 0.5}],
            }
        ],
    }
    install_urlopen(
        monkeypatch,
        [verbose_json([seg(0.0, 0.9)]), json.dumps(chunk1).encode(), verbose_json([])],
    )

    with pytest.raises(LemonadeCapabilityError) as excinfo:
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    # The word lies inside its segment (2.15 <= 2.1 + 0.1) but past the audio
    # that was sent (2.15 > 2.0 + 0.1): only the chunk bound may reject it.
    assert "past the audio" in str(excinfo.value)


def test_interrupt_keeps_completed_chunks_as_partial(tmp_path, monkeypatch):
    """Ctrl+C mid-chunks: keep what finished, mark partial, never raise."""
    wav = tmp_path / "long.wav"
    write_wav(wav, 2.5)
    bodies = [verbose_json([seg(0.0, 0.9, "one")]), KeyboardInterrupt()]
    install_urlopen(monkeypatch, bodies)

    segments, info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    assert [s.text for s in segments] == ["one"]
    assert info["partial"] is True
    # The interrupt lands inside the chunk loop; the duration must still be the
    # recording's real length, derived from the WAV header before the loop.
    assert info["duration"] == pytest.approx(2.5)


# --- text-only / capability ---


def test_flm_style_text_only_response_is_rejected(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    install_urlopen(monkeypatch, [json.dumps({"text": "just prose, no timestamps"}).encode()])

    with pytest.raises(LemonadeCapabilityError):
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)


def test_non_json_response_is_a_capability_error(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    install_urlopen(monkeypatch, [b"compact plain-text answer"])

    with pytest.raises(LemonadeCapabilityError):
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)


def test_empty_response_is_silence_not_an_error(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    install_urlopen(monkeypatch, [verbose_json([])])

    segments, info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)
    assert segments == []
    assert info["partial"] is False


def test_recursion_error_on_transcription_is_a_capability_error(tmp_path, monkeypatch):
    """RecursionError during JSON parsing must raise LemonadeCapabilityError."""
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    install_urlopen(monkeypatch, [b'{"raw": "test"}'])

    def fake_loads(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr("json.loads", fake_loads)

    with pytest.raises(LemonadeCapabilityError):
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)


def test_recursion_error_on_http_error_falls_back_to_status_classification(tmp_path, monkeypatch):
    """RecursionError during error JSON parsing must fall back to status classification."""
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    install_urlopen(monkeypatch, [http_error(500, b"{}")])

    def fake_loads(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr("json.loads", fake_loads)

    with pytest.raises(LemonadeUnavailableError):
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)


def test_recursion_error_on_diagnostics_returns_raw_dict(tmp_path, monkeypatch):
    """RecursionError during diagnostics JSON parsing must return raw dictionary."""
    install_urlopen(monkeypatch, [b'{"health": "ok"}'])

    def fake_loads(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr("json.loads", fake_loads)
    backend = LemonadeBackend(lemon_settings(tmp_path))
    assert backend.health() == {"raw": '{"health": "ok"}'}


# --- structured error classification ---


def test_404_missing_endpoint_is_a_capability_error():
    err = classify_http_failure(404, {"detail": "Not Found"})
    assert isinstance(err, LemonadeCapabilityError)


def test_404_missing_model_is_a_model_error():
    body = {
        "error": {
            "type": "invalid_request_error",
            "code": "model_not_found",
            "message": "no such model",
        }
    }
    err = classify_http_failure(404, body)
    assert isinstance(err, LemonadeModelError)


def test_400_rejected_model_is_a_model_error():
    err = classify_http_failure(
        400, {"error": {"type": "invalid_request_error", "message": "Model Bogus is not available"}}
    )
    assert isinstance(err, LemonadeModelError)


def test_400_client_invalid_request_is_a_configuration_error():
    err = classify_http_failure(
        400, {"error": {"type": "invalid_request_error", "message": "field x is required"}}
    )
    assert isinstance(err, LemonadeConfigurationError)


@pytest.mark.parametrize("status", [401, 403])
def test_auth_statuses_never_fall_back(status):
    err = classify_http_failure(status, {"detail": "unauthorized"})
    assert isinstance(err, LemonadeAuthenticationError)


def test_auth_error_code_wins_over_status():
    err = classify_http_failure(400, {"error": {"code": "invalid_api_key", "message": "bad key"}})
    assert isinstance(err, LemonadeAuthenticationError)


def test_rate_limit_and_server_failures_are_unavailable():
    assert isinstance(classify_http_failure(429, {}), LemonadeUnavailableError)
    assert isinstance(classify_http_failure(500, {}), LemonadeUnavailableError)
    assert isinstance(classify_http_failure(503, {}), LemonadeUnavailableError)


def test_5xx_model_load_failure_is_a_model_error():
    err = classify_http_failure(
        500, {"error": {"message": "failed to load model Whisper-Large-v3-Turbo"}}
    )
    assert isinstance(err, LemonadeModelError)


def test_connection_refused_is_unavailable(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    install_urlopen(monkeypatch, [urllib.error.URLError(ConnectionRefusedError(111))])

    with pytest.raises(LemonadeUnavailableError):
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)


def test_read_timeout_never_resubmits(tmp_path, monkeypatch):
    """Read/inference timeout -> InferenceTimeout, one request only, no retry."""
    wav = tmp_path / "long.wav"
    write_wav(wav, 2.5)
    calls = install_urlopen(monkeypatch, [TimeoutError("read timed out")])

    with pytest.raises(LemonadeInferenceTimeout):
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    assert len(calls) == 1


def test_connect_timeout_is_unavailable(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    install_urlopen(monkeypatch, [urllib.error.URLError(TimeoutError("connect timed out"))])

    with pytest.raises(LemonadeUnavailableError):
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)


# --- language normalization ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("russian", "ru"),
        ("English", "en"),
        ("  RU ", "ru"),
        ("zh-cn", "zh"),
        ("de", "de"),
        ("klingon", "klingon"),  # never invented, passed through
    ],
)
def test_normalize_language(raw, expected):
    assert normalize_language(raw) == expected


# --- diagnostics ---


def test_health_and_system_info_are_gets_with_auth(tmp_path, monkeypatch):
    calls = install_urlopen(monkeypatch, [b'{"status": "ok"}', b'{"os": "whatever"}'])
    backend = LemonadeBackend(lemon_settings(tmp_path, lemonade_api_key=SecretStr("tok")))

    assert backend.health() == {"status": "ok"}
    assert backend.system_info() == {"os": "whatever"}

    assert calls[0].full_url.endswith("/v1/health")
    assert calls[1].full_url.endswith("/v1/system-info")
    assert all(c.headers["Authorization"] == "Bearer tok" for c in calls)


# --- transport protection ---


def test_plain_http_remote_host_is_rejected(tmp_path):
    """Plaintext HTTP to a remote host would expose audio and credentials."""
    with pytest.raises(LemonadeConfigurationError) as excinfo:
        LemonadeBackend(lemon_settings(tmp_path, lemonade_url="http://remote-host:8000"))
    assert "https" in str(excinfo.value)


def test_https_remote_host_is_accepted(tmp_path):
    backend = LemonadeBackend(lemon_settings(tmp_path, lemonade_url="https://remote-host:8000"))
    assert backend._base_url == "https://remote-host:8000"
    assert backend._bypass_proxies is False


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:13305",
        "http://localhost:13305",
        "http://127.7.0.1:13305",
        "http://[::1]:13305",
        "http://LOCALHOST:13305",
    ],
)
def test_loopback_http_is_accepted_and_bypasses_proxies(tmp_path, url):
    backend = LemonadeBackend(lemon_settings(tmp_path, lemonade_url=url))
    assert backend._bypass_proxies is True


def test_loopback_requests_bypass_proxy_configuration(tmp_path, monkeypatch):
    """An inherited proxy must never see loopback traffic.

    getproxies() raises if consulted: the no-proxy opener never asks for the
    process-wide proxy configuration, so a transcription against 127.0.0.1 can
    only succeed when the proxy layer is truly bypassed.
    """
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)

    def no_proxies_here():
        raise AssertionError("loopback request consulted the proxy configuration")

    monkeypatch.setattr("urllib.request.getproxies", no_proxies_here)
    calls = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])

    LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    assert len(calls) == 1


def test_remote_https_keeps_default_proxy_path(tmp_path, monkeypatch):
    """Non-loopback HTTPS goes through urllib.request.urlopen (system proxies allowed)."""
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    calls = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])
    backend = LemonadeBackend(lemon_settings(tmp_path, lemonade_url="https://example.test"))

    backend.transcribe(wav)

    assert backend._bypass_proxies is False
    assert len(calls) == 1


# --- response size cap ---


class _StreamingResponse:
    """A response with no Content-Length that yields data in 1 KiB reads."""

    def __init__(self, size: int) -> None:
        self._remaining = size

    def read(self, n: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        take = min(1024, self._remaining)
        if n is not None and 0 < n < take:
            take = n
        self._remaining -= take
        return b"x" * take

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _HugeLengthResponse:
    """A response declaring a Content-Length far over the cap; reading it is a bug."""

    def __init__(self) -> None:
        self.headers = Message()
        self.headers["Content-Length"] = "999999999"

    def read(self, n: int = -1) -> bytes:  # pragma: no cover — must never be called
        raise AssertionError("an over-limit Content-Length must not be read")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_oversized_content_length_is_refused_unread(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    monkeypatch.setattr(lemon, "_MAX_RESPONSE_BYTES", 1024)
    install_urlopen(monkeypatch, [_HugeLengthResponse()])

    with pytest.raises(LemonadeUnavailableError) as excinfo:
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    assert "response cap" in str(excinfo.value)


def test_streamed_oversized_response_is_refused(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    monkeypatch.setattr(lemon, "_MAX_RESPONSE_BYTES", 1024)
    install_urlopen(monkeypatch, [_StreamingResponse(5000)])

    with pytest.raises(LemonadeUnavailableError):
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)


def test_streamed_response_reads_at_most_limit_plus_one(tmp_path, monkeypatch):
    """The read is capped: a hostile endpoint cannot stream unbounded bytes in."""
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    limit = 1024
    monkeypatch.setattr(lemon, "_MAX_RESPONSE_BYTES", limit)
    response = _StreamingResponse(10_000_000)
    install_urlopen(monkeypatch, [response])

    with pytest.raises(LemonadeUnavailableError):
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    # Only ~limit+1 bytes were pulled, never the whole 10 MB.
    assert response._remaining >= 10_000_000 - limit - 2048


def test_error_body_over_the_cap_is_classified_by_status(tmp_path, monkeypatch):
    """A giant HTTP error page is not buffered; the status alone still classifies."""
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    monkeypatch.setattr(lemon, "_MAX_RESPONSE_BYTES", 1024)
    install_urlopen(monkeypatch, [http_error(500, {"message": "x" * 5000})])

    with pytest.raises(LemonadeUnavailableError):
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)


def test_response_just_under_the_cap_parses(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    monkeypatch.setattr(lemon, "_MAX_RESPONSE_BYTES", 65536)
    install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])

    segments, _info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    assert [s.text for s in segments] == ["hello"]


# --- request memory bounds ---


def test_byte_cap_covers_overlap_and_framing(tmp_path, monkeypatch):
    """Every uploaded chunk stays inside the byte cap even with overlap."""
    wav = tmp_path / "wide.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(8)
        wf.setsampwidth(3)
        wf.setframerate(16000)
        wf.writeframes(b"\x00" * (8 * 3 * 16000 * 5))  # 5 s, 24 bytes per frame

    monkeypatch.setattr(lemon, "_MAX_CHUNK_BYTES", 200_000)
    monkeypatch.setattr(lemon, "_REQUEST_OVERHEAD_BYTES", 1_000)
    calls = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])

    LemonadeBackend(
        lemon_settings(tmp_path, lemonade_chunk_seconds=300.0, lemonade_overlap_seconds=0.2)
    ).transcribe(wav)

    assert len(calls) > 1  # the byte cap binds, not the duration target
    for request in calls:
        body = request.data
        start = body.index(b"audio/wav\r\n\r\n") + len(b"audio/wav\r\n\r\n")
        end = body.rindex(b"\r\n--tapeback-")
        assert end - start <= lemon._MAX_CHUNK_BYTES


def test_forged_frame_count_is_treated_as_non_chunkable(tmp_path, monkeypatch):
    """A RIFF header declaring more data than the file holds must not be trusted."""
    wav = tmp_path / "forged.wav"
    write_wav(wav, 0.1)
    data = bytearray(wav.read_bytes())
    data[40:44] = struct.pack("<I", 0xFFFFFFF0)  # data-chunk size field
    wav.write_bytes(bytes(data))

    calls = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])

    LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    assert len(calls) == 1  # one whole-file request, not one per phantom frame


def test_oversized_non_wav_input_is_refused_not_buffered(tmp_path, monkeypatch):
    """A non-chunkable input over the single-request cap falls back instead of OOM."""
    big = tmp_path / "raw.bin"
    big.write_bytes(b"\x00" * 4096)
    monkeypatch.setattr(lemon, "_MAX_CHUNK_BYTES", 1024)
    monkeypatch.setattr(lemon, "_REQUEST_OVERHEAD_BYTES", 128)
    calls = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])

    with pytest.raises(LemonadeCapabilityError) as excinfo:
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(big)

    assert "single-request cap (896 bytes)" in str(excinfo.value)
    assert calls == []


def test_non_wav_input_respects_request_overhead_reservation(tmp_path, monkeypatch):
    """Non-chunkable inputs reserve _REQUEST_OVERHEAD_BYTES from _MAX_CHUNK_BYTES."""
    max_chunk = 2048
    overhead = 512
    max_payload = max_chunk - overhead

    monkeypatch.setattr(lemon, "_MAX_CHUNK_BYTES", max_chunk)
    monkeypatch.setattr(lemon, "_REQUEST_OVERHEAD_BYTES", overhead)

    # Payload exactly at max_payload is accepted
    fit = tmp_path / "fit.bin"
    fit.write_bytes(b"\x00" * max_payload)
    calls = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])
    segments, _info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(fit)
    assert len(segments) == 1
    assert len(calls) == 1

    # Payload 1 byte over max_payload is refused before upload
    too_big = tmp_path / "too_big.bin"
    too_big.write_bytes(b"\x00" * (max_payload + 1))
    calls_too_big = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])
    with pytest.raises(LemonadeCapabilityError):
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(too_big)
    assert calls_too_big == []


def test_too_many_chunks_is_a_configuration_error(tmp_path, monkeypatch):
    """A chunk plan past the request-count limit is refused before any upload."""
    wav = tmp_path / "long.wav"
    write_wav(wav, 5.0)  # 1 s chunks -> 5 chunks
    monkeypatch.setattr(lemon, "_MAX_CHUNKS", 3)
    calls = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])

    with pytest.raises(LemonadeConfigurationError) as excinfo:
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    assert "TAPEBACK_LEMONADE_CHUNK_SECONDS" in str(excinfo.value)
    assert calls == []


# --- hostile numeric values in responses ---


def word_seg(start=0.0, end=0.4, words=None):
    return {"start": start, "end": end, "text": "hello", "words": words or []}


def word(start=0.0, end=0.2, text="hi", probability=0.9):
    return {"start": start, "end": end, "word": text, "probability": probability}


@pytest.mark.parametrize(
    "payload",
    [
        verbose_json([seg(True, 1.0)]),  # booleans are not timestamps
        verbose_json([seg(0.0, True)]),
        verbose_json([seg(float("nan"), 1.0)]),  # NaN poisons sorting/arithmetic
        verbose_json([seg(0.0, float("inf"))]),
        verbose_json([seg(-0.5, 1.0)]),  # negative
        verbose_json([seg(1.0, 0.5)]),  # end before start
        json.dumps({"text": "hi", "segments": ["not-a-segment"], "language": "english"}).encode(),
        verbose_json([word_seg(words=[word(probability="not-a-number")])]),
        verbose_json([word_seg(words=[word(probability=True)])]),
        verbose_json([word_seg(words=[word(probability=float("nan"))])]),
        verbose_json([word_seg(words=[word(probability=1.5)])]),
        verbose_json([word_seg(words=[word(start=0.2, end=0.1)])]),
        verbose_json([word_seg(words=[word(start=-0.1, end=0.2)])]),
    ],
)
def test_hostile_numeric_response_is_a_capability_error(tmp_path, monkeypatch, payload):
    """Booleans, NaN/inf, negatives, reversed spans, and out-of-range probabilities
    must all fall back — never escape as ValueError, never poison the transcript
    or the resume cache."""
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    install_urlopen(monkeypatch, [payload])

    with pytest.raises(LemonadeCapabilityError):
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)


def test_valid_words_still_convert(tmp_path, monkeypatch):
    """Strict validation must not reject a well-formed word list."""
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    install_urlopen(
        monkeypatch,
        [
            verbose_json(
                [
                    word_seg(
                        words=[
                            word(0.0, 0.2, "hel", 0.9),
                            word(0.2, 0.4, "lo", None),
                        ]
                    )
                ]
            )
        ],
    )

    segments, _info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    words = segments[0].words
    assert words is not None
    assert [w.word for w in words] == ["hel", "lo"]
    assert words[0].probability == 0.9
    # An absent probability stays the historical 0.0, not an invented value.
    assert words[1].probability == 0.0


def test_segment_past_the_sent_audio_is_rejected(tmp_path, monkeypatch):
    """A hostile server must not write timestamps beyond the recording."""
    wav = tmp_path / "long.wav"
    write_wav(wav, 2.5)  # 1 s chunks; chunk 0's audio is 1.0 s long
    install_urlopen(monkeypatch, [verbose_json([seg(0.0, 5.0)])])

    with pytest.raises(LemonadeCapabilityError) as excinfo:
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    assert "past the audio" in str(excinfo.value)


def test_segment_at_chunk_edge_is_accepted(tmp_path, monkeypatch):
    """Timestamps up to the chunk's own audio length (plus boundary slack) are legal."""
    wav = tmp_path / "long.wav"
    write_wav(wav, 2.5)
    install_urlopen(
        monkeypatch,
        [
            verbose_json([seg(0.0, 0.99), seg(1.5, 1.6)]),
            verbose_json([seg(0.5, 1.4)]),
            verbose_json([seg(0.0, 0.9)]),
        ],
    )

    segments, _info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    assert segments  # nothing rejected; dedup may drop overlap duplicates
    assert all(s.end <= 2.5 for s in segments)


# --- ambiguous error classification ---


def test_500_permission_denied_loading_model_is_a_model_error():
    """'permission' must not turn a model-loading failure into an auth error."""
    err = classify_http_failure(
        500,
        {"error": {"message": "permission denied loading model Whisper-Large-v3-Turbo"}},
    )
    assert isinstance(err, LemonadeModelError)


def test_model_author_not_found_is_not_an_auth_error():
    """The substring 'auth' inside 'author' must not disable fallback."""
    err = classify_http_failure(500, {"error": {"message": "model author not found"}})
    assert isinstance(err, LemonadeModelError)


def test_404_model_mention_is_a_model_error():
    err = classify_http_failure(404, {"error": {"message": "model missing on this server"}})
    assert isinstance(err, LemonadeModelError)


def test_bare_404_is_a_capability_error():
    assert isinstance(classify_http_failure(404, {}), LemonadeCapabilityError)


def test_408_request_timeout_is_fallback_eligible():
    """A proxy/server 408 is a transient availability failure, not a local bug."""
    err = classify_http_failure(408, {"error": {"message": "upstream request timeout"}})
    assert isinstance(err, LemonadeInferenceTimeout)


def test_408_with_auth_phrasing_is_still_an_auth_error():
    """Token-aware auth classification outranks 408 status semantics."""
    err = classify_http_failure(408, {"error": {"message": "unauthorized"}})
    assert isinstance(err, LemonadeAuthenticationError)


def test_408_from_transcription_is_an_inference_timeout(tmp_path, monkeypatch):
    """A 408 from the transcription endpoint reaches the fallback hierarchy."""
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    install_urlopen(monkeypatch, [http_error(408, {"error": {"message": "request timeout"}})])

    with pytest.raises(LemonadeInferenceTimeout):
        LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)


def test_message_auth_phrases_are_token_aware():
    err = classify_http_failure(400, {"error": {"message": "unauthorized request shape"}})
    assert isinstance(err, LemonadeAuthenticationError)


def test_structured_auth_code_wins_over_model_wording():
    err = classify_http_failure(
        400, {"error": {"code": "invalid_api_key", "message": "model rejected the key"}}
    )
    assert isinstance(err, LemonadeAuthenticationError)


# --- structural URL validation ---


def test_url_with_userinfo_is_rejected(tmp_path):
    """Credentials embedded in the URL would be displayed by status."""
    with pytest.raises(LemonadeConfigurationError) as exc_info:
        LemonadeBackend(
            lemon_settings(tmp_path, lemonade_url="http://alice:s3cret@127.0.0.1:13305")
        )
    # The embedded username/password must not survive into the message (fixed
    # guidance text may mention the concept, never the actual values).
    assert "alice" not in str(exc_info.value)
    assert "s3cret" not in str(exc_info.value)


def test_url_with_query_or_fragment_is_rejected(tmp_path):
    """A base query/fragment is kept while the transcription path is appended."""
    for bad in ("http://127.0.0.1:13305/?x=1", "http://127.0.0.1:13305/#frag"):
        with pytest.raises(LemonadeConfigurationError):
            LemonadeBackend(lemon_settings(tmp_path, lemonade_url=bad))


def test_url_is_normalized_structurally(tmp_path):
    backend = LemonadeBackend(lemon_settings(tmp_path, lemonade_url="HTTP://LocalHost:13305/"))
    assert backend.base_url == "http://localhost:13305"


def test_url_default_port_is_dropped(tmp_path):
    https_backend = LemonadeBackend(
        lemon_settings(tmp_path, lemonade_url="https://example.com:443")
    )
    assert https_backend.base_url == "https://example.com"
    http_backend = LemonadeBackend(lemon_settings(tmp_path, lemonade_url="http://127.0.0.1:80"))
    assert http_backend.base_url == "http://127.0.0.1"


# --- opaque multipart filename ---


def test_upload_uses_opaque_filename(tmp_path, monkeypatch):
    """A crafted source filename must never reach the MIME part headers."""
    wav = tmp_path / 'we"ird\nname.wav'
    write_wav(wav, 0.5)
    calls = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])

    LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    body = calls[0].data
    assert b'filename="audio.wav"' in body
    assert b'we"ird' not in body


def test_multipart_field_with_header_characters_is_rejected(tmp_path):
    """Field values go into MIME headers; quotes and CRLF must never pass."""
    with pytest.raises(LemonadeConfigurationError):
        lemon._multipart_body([("model", 'bad"model')], "file", "audio.wav", "audio/wav", b"x")
    with pytest.raises(LemonadeConfigurationError):
        lemon._multipart_body([("model", "bad\r\nmodel")], "file", "audio.wav", "audio/wav", b"x")


# --- diagnostics timeout ---


def test_diagnostics_use_the_short_dedicated_timeout(tmp_path, monkeypatch):
    """Health/system-info probes must never inherit the 600 s inference timeout."""
    seen: list[float] = []

    def fake_open(request, timeout, *, bypass_proxies):
        seen.append(timeout)
        return _FakeResponse(b"{}")

    monkeypatch.setattr(lemon, "_open_url", fake_open)
    backend = LemonadeBackend(
        lemon_settings(
            tmp_path,
            lemonade_timeout_seconds=600.0,
            lemonade_diagnostics_timeout_seconds=7.5,
        )
    )

    backend.health()
    backend.system_info()

    # The open receives the remaining budget, which at open time is the full
    # timeout up to clock precision.
    assert seen == [pytest.approx(7.5), pytest.approx(7.5)]


def test_transcription_still_uses_the_inference_timeout(tmp_path, monkeypatch):
    seen: list[float] = []

    def fake_open(request, timeout, *, bypass_proxies):
        seen.append(timeout)
        return _FakeResponse(verbose_json([seg(0.0, 0.4)]))

    monkeypatch.setattr(lemon, "_open_url", fake_open)
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    backend = LemonadeBackend(lemon_settings(tmp_path, lemonade_timeout_seconds=600.0))

    backend.transcribe(wav)

    assert seen == [pytest.approx(600.0)]


# --- total deadline enforcement ---


class _FakeSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class _FakeRaw:
    """The BufferedReader.raw layer: _bound_socket_timeout reaches the socket here."""

    def __init__(self, sock: _FakeSocket) -> None:
        self._sock = sock


class _FakeBufferedReader:
    def __init__(self, sock: _FakeSocket) -> None:
        self.raw = _FakeRaw(sock)


class _SocketResponse:
    """A response exposing the fp.fp.raw._sock path real urllib responses have."""

    def __init__(self, sock: _FakeSocket) -> None:
        self.fp = _FakeBufferedReader(sock)
        self._sent = False

    def read(self, n: int = -1) -> bytes:
        if not self._sent:
            self._sent = True
            return verbose_json([seg(0.0, 0.4)])
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeClock:
    now = 1000.0


def test_socket_timeout_is_reset_to_the_remaining_budget(tmp_path, monkeypatch):
    """Each blocking operation gets the remaining budget, never the full timeout.

    Headers arrive 590 s into a 600 s budget: the open received the full 600,
    but the first body read must be bounded by the ~10 s that are left, not by
    another 600 s socket timeout on top.
    """
    monkeypatch.setattr(lemon.time, "monotonic", lambda: _FakeClock.now)
    sock = _FakeSocket()
    open_timeouts: list[float] = []

    def fake_open(request, timeout, *, bypass_proxies):
        open_timeouts.append(timeout)
        _FakeClock.now += 590.0
        return _SocketResponse(sock)

    monkeypatch.setattr(lemon, "_open_url", fake_open)
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)

    LemonadeBackend(lemon_settings(tmp_path, lemonade_timeout_seconds=600.0)).transcribe(wav)

    assert open_timeouts == [pytest.approx(600.0)]
    assert sock.timeouts
    assert all(t < 11.0 for t in sock.timeouts)


def test_stalled_body_read_trips_the_total_deadline(tmp_path, monkeypatch):
    """A body read stalling past the deadline raises the inference timeout.

    Headers arrive with 10 s of budget left, the first read stalls for 100 s:
    the between-reads deadline check must end the request rather than leave it
    blocked for another full socket timeout.
    """
    monkeypatch.setattr(lemon.time, "monotonic", lambda: _FakeClock.now)

    class _StallingResponse:
        def read(self, n: int = -1) -> bytes:
            _FakeClock.now += 100.0
            return b"x" * 10

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_open(request, timeout, *, bypass_proxies):
        _FakeClock.now += 590.0
        return _StallingResponse()

    monkeypatch.setattr(lemon, "_open_url", fake_open)
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)

    with pytest.raises(LemonadeInferenceTimeout):
        LemonadeBackend(lemon_settings(tmp_path, lemonade_timeout_seconds=600.0)).transcribe(wav)


# --- redirects are never followed ---


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_no_redirect_handler_refuses_every_30x(code):
    """Returning None makes urllib surface the 30x as an HTTPError, un-followed."""
    handler = lemon._NoRedirectHandler()
    assert handler.redirect_request(None, None, code, "", {}, "http://attacker.example/") is None


@pytest.mark.parametrize("opener_name", ["_DEFAULT_OPENER", "_NO_PROXY_OPENER"])
def test_openers_install_the_no_redirect_handler(opener_name):
    """Both HTTP paths must replace urllib's default redirect handler."""
    opener = getattr(lemon, opener_name)
    assert any(isinstance(handler, lemon._NoRedirectHandler) for handler in opener.handlers)


def test_redirect_response_is_classified_without_echoing_the_target(tmp_path, monkeypatch):
    """A 30x is a sanitized fallback-eligible error; the Location is never shown."""
    headers = Message()
    headers["Location"] = "http://attacker.example/steal"
    error = urllib.error.HTTPError(
        "https://example.test/v1/health", 302, "Found", headers, io.BytesIO(b"")
    )
    install_urlopen(monkeypatch, [error])
    backend = LemonadeBackend(lemon_settings(tmp_path, lemonade_url="https://example.test"))

    with pytest.raises(LemonadeUnavailableError) as exc_info:
        backend.health()

    message = str(exc_info.value)
    assert "redirect" in message
    assert "attacker.example" not in message


def test_real_server_redirect_is_never_followed(tmp_path):
    """End-to-end over real sockets: a 302 must not produce a second request.

    This is the regression for the proven POST→GET + Authorization-leak probe:
    urllib's default handler followed a 302, converted the POST to a GET, and
    kept the bearer header on the server-chosen target.
    """
    victim_requests: list[tuple[str, str, str | None]] = []
    redirector_requests: list[tuple[str, str | None]] = []

    class _Victim(BaseHTTPRequestHandler):
        def _respond(self) -> None:
            victim_requests.append((self.command, self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()

        do_POST = _respond
        do_GET = _respond

        def log_message(self, format: str, *args: object) -> None:  # silence test output
            pass

    victim = HTTPServer(("127.0.0.1", 0), _Victim)

    class _Redirector(BaseHTTPRequestHandler):
        def _respond(self) -> None:
            redirector_requests.append((self.command, self.headers.get("Authorization")))
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{victim.server_port}/steal")
            self.end_headers()

        do_POST = _respond
        do_GET = _respond

        def log_message(self, format: str, *args: object) -> None:  # silence test output
            pass

    redirector = HTTPServer(("127.0.0.1", 0), _Redirector)
    for server in (redirector, victim):
        threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        key = "sk-redirect-probe-token"
        backend = LemonadeBackend(
            lemon_settings(
                tmp_path,
                lemonade_url=f"http://127.0.0.1:{redirector.server_port}",
                lemonade_api_key=SecretStr(key),
            )
        )
        with pytest.raises(LemonadeUnavailableError) as exc_info:
            backend.health()
        assert "redirect" in str(exc_info.value)
        assert key not in str(exc_info.value)
    finally:
        for server in (redirector, victim):
            server.shutdown()
            server.server_close()

    # Exactly one request reached the configured endpoint — with method and
    # credential intact — and the redirect target received nothing at all.
    assert redirector_requests == [("GET", f"Bearer {key}")]
    assert victim_requests == []


# --- server-controlled error text is sanitized ---


def test_classified_error_redacts_a_reflected_api_key():
    """A server reflecting the received Authorization value must not repeat it."""
    key = "sk-supersecret-lemonade-token"
    payload = {
        "error": {
            "type": "upstream_error",
            "code": "bad_gateway",
            "message": f"request rejected with header: Bearer {key}",
        }
    }
    exc = classify_http_failure(502, payload, secrets=(key,))
    assert key not in str(exc)
    assert "[redacted]" in str(exc)


def test_send_redacts_a_reflected_api_key(tmp_path, monkeypatch):
    """The backend wires the configured key into classification end to end."""
    key = "sk-supersecret-lemonade-token"
    payload = {"error": {"message": f"received Authorization: Bearer {key}"}}
    install_urlopen(monkeypatch, [http_error(502, payload)])
    backend = LemonadeBackend(
        lemon_settings(
            tmp_path,
            lemonade_url="https://example.test",
            lemonade_api_key=SecretStr(key),
        )
    )
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)

    with pytest.raises(LemonadeError) as exc_info:
        backend.transcribe(wav)

    assert key not in str(exc_info.value)
    assert "[redacted]" in str(exc_info.value)


def test_classified_error_strips_terminal_control_characters():
    payload = {"error": {"message": "boom\x1b]0;pwned\x07 and more"}}
    exc = classify_http_failure(500, payload)
    assert "\x1b" not in str(exc)
    assert "\x07" not in str(exc)


def test_remote_detail_is_length_capped():
    payload = {"error": {"message": "A" * 5000}}
    exc = classify_http_failure(500, payload)
    assert len(str(exc)) < 300


def test_classification_is_unchanged_by_sanitization():
    """Sanitization shapes the message, never the classification decision."""
    payload_missing = {"error": {"message": "nope"}}
    assert isinstance(classify_http_failure(404, payload_missing), LemonadeCapabilityError)
    assert isinstance(
        classify_http_failure(500, {"error": {"message": "permission denied loading model"}}),
        LemonadeModelError,
    )
    assert isinstance(classify_http_failure(401, {}), LemonadeAuthenticationError)
