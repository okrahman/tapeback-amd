"""Lemonade backend: HTTP transport, chunking, language handling, error mapping."""

import io
import json
import urllib.error
import wave
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from tapeback._lemonade import (
    DEDUP_POLICY_VERSION,
    LemonadeAuthenticationError,
    LemonadeBackend,
    LemonadeCapabilityError,
    LemonadeConfigurationError,
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

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def install_urlopen(monkeypatch, bodies):
    """Queue responses (bytes or exception instances) for successive requests."""
    calls: list[object] = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        index = min(len(calls) - 1, len(bodies) - 1)
        body = bodies[index]
        if isinstance(body, BaseException):
            raise body
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return calls


def http_error(status: int, payload: dict | str = "") -> urllib.error.HTTPError:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return urllib.error.HTTPError("http://x", status, "err", Message(), io.BytesIO(body))


# --- construction and configuration ---


def test_invalid_url_is_a_configuration_error(tmp_path):
    with pytest.raises(LemonadeConfigurationError):
        LemonadeBackend(lemon_settings(tmp_path, lemonade_url="not-a-url"))
    with pytest.raises(LemonadeConfigurationError):
        LemonadeBackend(lemon_settings(tmp_path, lemonade_url="ftp://host:1"))


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
        ("lemonade_overlap_seconds", 4.0),
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
    backend = LemonadeBackend(lemon_settings(tmp_path, lemonade_url="http://h:1/"))
    other = LemonadeBackend(lemon_settings(tmp_path, lemonade_url="http://h:1"))
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
    assert 'name="file"' in body and "a.wav" in body
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


def test_interrupt_keeps_completed_chunks_as_partial(tmp_path, monkeypatch):
    """Ctrl+C mid-chunks: keep what finished, mark partial, never raise."""
    wav = tmp_path / "long.wav"
    write_wav(wav, 2.5)
    bodies = [verbose_json([seg(0.0, 0.9, "one")]), KeyboardInterrupt()]
    install_urlopen(monkeypatch, bodies)

    segments, info = LemonadeBackend(lemon_settings(tmp_path)).transcribe(wav)

    assert [s.text for s in segments] == ["one"]
    assert info["partial"] is True


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
