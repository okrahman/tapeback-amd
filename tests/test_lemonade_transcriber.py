"""The Transcriber facade with the Lemonade backend: fallback and transactional stereo."""

import io
import json
import subprocess
import sys
import urllib.error
import wave
from email.message import Message
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import tapeback._lemonade as lemon
from tapeback import _resume
from tapeback._lemonade import (
    LemonadeAuthenticationError,
    LemonadeConfigurationError,
    LemonadeModelError,
)
from tapeback.models import Segment
from tapeback.pipeline import _gpu_telemetry_enabled
from tapeback.settings import Settings
from tapeback.transcriber import Transcriber

# --- helpers ---


def write_wav(path: Path, duration_s: float, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * int(duration_s * rate))


def lemon_settings(tmp_path, **overrides) -> Settings:
    base: dict[str, Any] = {
        "transcription_backend": "lemonade",
        "resume_cache_dir": tmp_path / "resume",
        "isolate_transcription": False,
    }
    base.update(overrides)
    return Settings(**base)


def verbose_json(segments, language="russian"):
    return json.dumps(
        {
            "text": " ".join(s.get("text", "") for s in segments),
            "segments": segments,
            "language": language,
            "language_probability": 0.9,
        }
    ).encode()


def seg(start, end, text="hello"):
    return {"start": start, "end": end, "text": text, "words": []}


class _FakeResponse:
    def __init__(self, body):
        self._body = body
        self._consumed = False

    def read(self, n: int = -1):
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
    """Queue responses (bytes, exceptions, or pre-built responses) per request.

    Routes BOTH HTTP paths tapeback can take — the default ``urlopen`` and the
    loopback no-proxy opener — through the same fake.
    """
    calls: list[object] = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        body = bodies[min(len(calls) - 1, len(bodies) - 1)]
        if isinstance(body, BaseException):
            raise body
        if isinstance(body, (bytes, bytearray)):
            return _FakeResponse(body)
        return body  # a pre-built response-like object, passed through

    _patch_urlopen(monkeypatch, fake_urlopen)
    return calls


def _patch_urlopen(monkeypatch, fn):
    """Route every tapeback HTTP path (default and loopback opener) through fn."""
    monkeypatch.setattr("urllib.request.urlopen", fn)

    class _FakeOpener:
        def open(self, request, timeout=None):
            return fn(request, timeout=timeout)

    monkeypatch.setattr("tapeback._lemonade._DEFAULT_OPENER", _FakeOpener())
    monkeypatch.setattr("tapeback._lemonade._NO_PROXY_OPENER", _FakeOpener())


def fw_result(texts, language="ru"):
    segments = [Segment(start=i * 1.0, end=i * 1.0 + 0.9, text=t) for i, t in enumerate(texts)]
    info = {"language": language, "language_probability": 0.9, "duration": 5.0, "partial": False}
    return segments, info


@pytest.fixture
def mock_fw(monkeypatch):
    """Replace the lazily-built faster-whisper backend with a controllable mock."""
    fw = MagicMock()
    fw.cache_fingerprint.return_value = "fw-fingerprint"
    fw.transcribe.return_value = fw_result(["fallback"])
    monkeypatch.setattr(Transcriber, "_new_fw_backend", lambda self: fw)
    return fw


@pytest.fixture
def audio(tmp_path):
    wav = tmp_path / "a.wav"
    write_wav(wav, 0.5)
    return wav


# --- mono fallback ---


def test_mono_model_failure_falls_back_and_caches_under_fw_identity(
    tmp_path, monkeypatch, mock_fw, audio
):
    def failing(request, timeout=None):
        raise urllib.error.HTTPError(
            "http://x",
            400,
            "err",
            Message(),
            io.BytesIO(json.dumps({"error": {"message": "Model X is not available"}}).encode()),
        )

    _patch_urlopen(monkeypatch, failing)

    segments, _info = Transcriber(lemon_settings(tmp_path)).transcribe(audio)

    assert [s.text for s in segments] == ["fallback"]
    assert mock_fw.transcribe.call_count == 1
    entries = list((tmp_path / "resume").glob("*.json"))
    assert len(entries) == 1
    payload = json.loads(entries[0].read_text())
    assert payload["segments"][0]["text"] == "fallback"


def test_capability_error_reaches_the_facade_and_falls_back(tmp_path, monkeypatch, mock_fw, audio):
    """FLM-style text-only output: rejected in full, run resolved by faster-whisper."""
    install_urlopen(monkeypatch, [json.dumps({"text": "prose without timestamps"}).encode()])

    segments, _ = Transcriber(lemon_settings(tmp_path)).transcribe(audio)

    assert [s.text for s in segments] == ["fallback"]
    assert mock_fw.transcribe.call_count == 1


def test_authentication_error_does_not_fall_back(tmp_path, monkeypatch, mock_fw, audio):
    _patch_urlopen(
        monkeypatch,
        lambda request, timeout=None: (_ for _ in ()).throw(
            urllib.error.HTTPError("http://x", 401, "auth", Message(), io.BytesIO(b"{}"))
        ),
    )

    with pytest.raises(LemonadeAuthenticationError):
        Transcriber(lemon_settings(tmp_path)).transcribe(audio)

    mock_fw.transcribe.assert_not_called()
    assert list((tmp_path / "resume").glob("*.json")) == []


def test_configuration_error_does_not_fall_back(tmp_path, monkeypatch, mock_fw, audio):
    _patch_urlopen(
        monkeypatch,
        lambda request, timeout=None: (_ for _ in ()).throw(
            urllib.error.HTTPError("http://x", 400, "bad", Message(), io.BytesIO(b"{}"))
        ),
    )

    with pytest.raises(LemonadeConfigurationError):
        Transcriber(lemon_settings(tmp_path)).transcribe(audio)

    mock_fw.transcribe.assert_not_called()


def test_lemonade_success_caches_under_lemonade_identity(tmp_path, monkeypatch, mock_fw, audio):
    install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4)])])

    segments, _ = Transcriber(lemon_settings(tmp_path)).transcribe(audio)

    assert [s.text for s in segments] == ["hello"]
    mock_fw.transcribe.assert_not_called()
    assert len(list((tmp_path / "resume").glob("*.json"))) == 1


def test_partial_lemonade_result_is_returned_but_never_cached(
    tmp_path, monkeypatch, mock_fw, audio
):
    install_urlopen(monkeypatch, [KeyboardInterrupt()])

    _segments, info = Transcriber(lemon_settings(tmp_path)).transcribe(audio)

    assert info["partial"] is True
    mock_fw.transcribe.assert_not_called()
    assert list((tmp_path / "resume").glob("*.json")) == []


# --- transactional stereo ---


def _stereo(tmp_path):
    mic = tmp_path / "mic.wav"
    monitor = tmp_path / "monitor.wav"
    write_wav(mic, 0.5)
    write_wav(monitor, 0.5)
    return mic, monitor


def _backend_mock(side_effects, fingerprint="lemonade-fp"):
    backend = MagicMock()
    backend.cache_fingerprint.return_value = fingerprint
    backend.transcribe.side_effect = list(side_effects)
    return backend


def _install_backend(monkeypatch, backend, settings):
    transcriber = Transcriber.__new__(Transcriber)
    transcriber._settings = settings
    transcriber._backend = backend
    return transcriber


def _stereo_backend(tmp_path, monkeypatch, monitor_effect, mic_effect, fingerprint="lemonade-fp"):
    """A Transcriber whose lemonade backend produces the given channel effects."""
    settings = lemon_settings(tmp_path)
    transcriber = _install_backend(
        monkeypatch, _backend_mock([monitor_effect, mic_effect], fingerprint), settings
    )
    fw = MagicMock()
    fw.cache_fingerprint.return_value = "fw-fingerprint"
    fw.transcribe.return_value = fw_result(["fw-monitor"], language="ru")
    fw.pace.return_value = None
    monkeypatch.setattr(Transcriber, "_new_fw_backend", lambda self: fw)
    return transcriber, fw


def test_stereo_both_channels_succeed_and_both_stage_entries_commit(tmp_path, monkeypatch):
    mic, monitor = _stereo(tmp_path)
    ok = fw_result(["text"])
    transcriber, fw = _stereo_backend(tmp_path, monkeypatch, ok, ok)

    mic_segments, _monitor_segments, info = transcriber.transcribe_stereo(mic, monitor)

    assert info["partial"] is False
    assert all(s.speaker == "You" for s in mic_segments)
    assert len(list((tmp_path / "resume").glob("*.json"))) == 2
    fw.transcribe.assert_not_called()


def test_stereo_mic_failure_commits_nothing_new_and_falls_back_both(tmp_path, monkeypatch):
    """Monitor succeeded on Lemonade, mic failed: NO Lemonade entry is written, and
    both channels resolve through faster-whisper (never a mixed transcript)."""
    mic, monitor = _stereo(tmp_path)
    ok = fw_result(["lemonade-monitor"])
    fail = LemonadeModelError("model gone")
    transcriber, fw = _stereo_backend(tmp_path, monkeypatch, ok, fail)

    mic_segments, monitor_segments, info = transcriber.transcribe_stereo(mic, monitor)

    assert info["partial"] is False
    # Both channels came from the fallback: the staged Lemonade monitor is discarded.
    assert all(s.text == "fw-monitor" for s in mic_segments)
    assert all(s.text == "fw-monitor" for s in monitor_segments)
    # fw served monitor (with language reuse) and mic — exactly two calls.
    assert fw.transcribe.call_count == 2
    # Only faster-whisper fingerprints are in the cache.
    entries = list((tmp_path / "resume").glob("*.json"))
    assert len(entries) == 2
    assert all("lemonade-fp" not in json.dumps(e.name) for e in entries)


# Above relies on the key digest; assert the entries really are fw-keyed by loading them
# through a fake fingerprint match instead of parsing names:


def test_fallback_results_are_stored_only_under_fw_fingerprints(tmp_path, monkeypatch):
    mic, monitor = _stereo(tmp_path)
    ok = fw_result(["lemonade-monitor"])
    transcriber, _fw = _stereo_backend(tmp_path, monkeypatch, ok, LemonadeModelError("x"))
    transcriber.transcribe_stereo(mic, monitor)

    lemonade_monitor_key = transcriber._resume_key(monitor, "transcribe monitor", "lemonade-fp")
    fw_monitor_key = transcriber._resume_key(monitor, "transcribe monitor", "fw-fingerprint")
    assert lemonade_monitor_key is not None and fw_monitor_key is not None
    resume_dir = tmp_path / "resume"
    assert _resume.load(lemonade_monitor_key, resume_dir) is None
    assert _resume.load(fw_monitor_key, resume_dir) is not None


def test_stereo_interrupt_commits_nothing(tmp_path, monkeypatch):
    mic, monitor = _stereo(tmp_path)
    partial = (fw_result(["half"])[0], {"language": "ru", "duration": 1.0, "partial": True})
    transcriber, fw = _stereo_backend(tmp_path, monkeypatch, partial, fw_result(["x"]))

    _mic_segments, _monitor_segments, info = transcriber.transcribe_stereo(mic, monitor)

    assert info["partial"] is True
    assert fw.transcribe.call_count == 0
    assert list((tmp_path / "resume").glob("*.json")) == []


def test_existing_cache_hits_survive_a_fallback(tmp_path, monkeypatch):
    """A complete fw cache entry is reused, not overwritten or discarded."""
    mic, monitor = _stereo(tmp_path)
    cached = fw_result(["cached-monitor"])
    settings = lemon_settings(tmp_path)
    probe = Transcriber.__new__(Transcriber)
    probe._settings = settings
    key = probe._resume_key(monitor, "transcribe monitor", "fw-fingerprint")
    assert key is not None
    _resume.store(key, tmp_path / "resume", *cached)

    transcriber, fw = _stereo_backend(
        tmp_path, monkeypatch, fw_result(["lemonade"]), LemonadeModelError("x")
    )
    _mic_segments, monitor_segments, info = transcriber.transcribe_stereo(mic, monitor)

    assert all(s.text == "cached-monitor" for s in monitor_segments)
    assert info["partial"] is False
    # fw only had to do the mic channel — the monitor was already cached.
    assert fw.transcribe.call_count == 1


def test_stereo_resume_hit_skips_work_entirely(tmp_path, monkeypatch):
    mic, monitor = _stereo(tmp_path)
    ok = fw_result(["text"])
    transcriber, _ = _stereo_backend(tmp_path, monkeypatch, ok, ok)
    transcriber.transcribe_stereo(mic, monitor)  # first run commits both entries

    backend = _backend_mock([], fingerprint="lemonade-fp")
    second = _install_backend(monkeypatch, backend, lemon_settings(tmp_path))

    mic_segments, _monitor_segments, info = second.transcribe_stereo(mic, monitor)

    assert info["partial"] is False
    backend.transcribe.assert_not_called()
    assert [s.text for s in mic_segments] == ["text"]


def test_stereo_mic_partial_still_commits_complete_monitor(tmp_path, monkeypatch):
    """Interrupting the mic (second channel) must still save the completed monitor.

    Regression: the all-or-none commit stored NEITHER channel when the mic came back
    partial, so a re-run retranscribed the whole monitor — directly contradicting
    _resume.py's contract that interrupting the second channel saves the first.
    """
    mic, monitor = _stereo(tmp_path)
    ok = fw_result(["ok"])
    partial = (fw_result(["half"])[0], {"language": "ru", "duration": 1.0, "partial": True})
    transcriber, fw = _stereo_backend(tmp_path, monkeypatch, ok, partial)

    mic_segments, monitor_segments, info = transcriber.transcribe_stereo(mic, monitor)

    assert len(mic_segments) == 1
    assert len(monitor_segments) == 1
    assert info["partial"] is True
    assert fw.transcribe.call_count == 0

    resume_dir = tmp_path / "resume"
    monitor_key = transcriber._resume_key(monitor, "transcribe monitor", "lemonade-fp")
    mic_key = transcriber._resume_key(
        mic, "transcribe mic", "lemonade-fp", transcriber._effective_language("ru")
    )
    assert monitor_key is not None and mic_key is not None
    cached = _resume.load(monitor_key, resume_dir)
    assert cached is not None
    assert [s.text for s in cached[0]] == ["ok"]
    assert _resume.load(mic_key, resume_dir) is None


def test_fallback_stereo_mic_partial_still_commits_monitor(tmp_path, monkeypatch):
    """The fallback path applies the same per-channel rule: a complete faster-whisper
    monitor is cached even when the faster-whisper mic interrupts mid-fallback."""
    mic, monitor = _stereo(tmp_path)
    ok = fw_result(["lemonade-monitor"])
    transcriber, fw = _stereo_backend(tmp_path, monkeypatch, ok, LemonadeModelError("x"))

    partial = (fw_result(["fw-half"])[0], {"language": "ru", "duration": 1.0, "partial": True})
    fw.transcribe.side_effect = [
        fw_result(["fw-monitor"], language="ru"),
        partial,
    ]

    _mic_segments, monitor_segments, info = transcriber.transcribe_stereo(mic, monitor)

    assert info["partial"] is True
    assert [s.text for s in monitor_segments] == ["fw-monitor"]

    resume_dir = tmp_path / "resume"
    monitor_key = transcriber._resume_key(monitor, "transcribe monitor", "fw-fingerprint")
    mic_key = transcriber._resume_key(
        mic, "transcribe mic", "fw-fingerprint", transcriber._effective_language("ru")
    )
    assert monitor_key is not None and mic_key is not None
    cached = _resume.load(monitor_key, resume_dir)
    assert cached is not None
    assert [s.text for s in cached[0]] == ["fw-monitor"]
    assert _resume.load(mic_key, resume_dir) is None


def test_stereo_use_resume_false_skips_resume_io(tmp_path, monkeypatch):
    """Live mode must not read or write resume entries for ephemeral chunk WAVs."""
    mic, monitor = _stereo(tmp_path)
    ok = fw_result(["text"])
    transcriber, _fw = _stereo_backend(tmp_path, monkeypatch, ok, ok)

    _mic_segments, _monitor_segments, info = transcriber.transcribe_stereo(
        mic, monitor, use_resume=False
    )

    assert info["partial"] is False
    assert list((tmp_path / "resume").glob("*.json")) == []

    # Seeded entries are equally ignored: use_resume=False means read nothing either.
    seed = fw_result(["cached"])
    settings = lemon_settings(tmp_path)
    probe = Transcriber.__new__(Transcriber)
    probe._settings = settings
    key = probe._resume_key(monitor, "transcribe monitor", "lemonade-fp")
    assert key is not None
    _resume.store(key, tmp_path / "resume", *seed)

    backend = _backend_mock([ok, ok])
    second = _install_backend(monkeypatch, backend, settings)
    _mic_segments, monitor_segments, _info = second.transcribe_stereo(
        mic, monitor, use_resume=False
    )
    assert [s.text for s in monitor_segments] == ["text"]  # NOT the seeded cache
    assert backend.transcribe.call_count == 2
    assert len(list((tmp_path / "resume").glob("*.json"))) == 1  # seed untouched


def test_stereo_skip_mic_false_keeps_mic_on_monitor_partial(tmp_path, monkeypatch):
    """Live mode must keep this interval's mic even when the monitor is interrupted —
    an interrupted monitor in the background thread does not mean the user stopped."""
    mic, monitor = _stereo(tmp_path)
    partial = (fw_result(["half"])[0], {"language": "ru", "duration": 1.0, "partial": True})
    ok = fw_result(["mic"])
    transcriber, fw = _stereo_backend(tmp_path, monkeypatch, partial, ok)

    mic_segments, monitor_segments, info = transcriber.transcribe_stereo(
        mic, monitor, skip_mic_on_monitor_partial=False
    )

    assert len(monitor_segments) == 1
    assert [s.text for s in mic_segments] == ["mic"]
    assert info["partial"] is True  # an interrupted monitor still marks the run partial
    assert fw.transcribe.call_count == 0

    # The complete mic is cached; the partial monitor never is.
    resume_dir = tmp_path / "resume"
    entries = list(resume_dir.glob("*.json"))
    assert len(entries) == 1
    mic_key = transcriber._resume_key(
        mic, "transcribe mic", "lemonade-fp", transcriber._effective_language("ru")
    )
    assert mic_key is not None
    cached = _resume.load(mic_key, resume_dir)
    assert cached is not None
    assert [s.text for s in cached[0]] == ["mic"]
    monitor_key = transcriber._resume_key(monitor, "transcribe monitor", "lemonade-fp")
    assert monitor_key is not None
    assert _resume.load(monitor_key, resume_dir) is None


# --- lazy loading and misc facade behaviour ---


def test_choosing_lemonade_does_not_import_faster_whisper():
    """The ~10s ML import must not be paid to reach an HTTP server."""
    code = (
        "import sys;"
        "from tapeback.transcriber import Transcriber;"
        "assert 'tapeback._fw_backend' not in sys.modules;"
        "print('tapeback._fw_backend' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip().endswith("False")


def test_describe_delegates_to_the_lemonade_backend(tmp_path):
    transcriber = Transcriber(lemon_settings(tmp_path))
    described = transcriber.describe()
    assert "Lemonade" in described and "Whisper-Large-v3-Turbo" in described


def test_unavailable_error_falls_back(tmp_path, monkeypatch, mock_fw, audio):
    def unreachable(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(111))

    _patch_urlopen(monkeypatch, unreachable)

    segments, _ = Transcriber(lemon_settings(tmp_path)).transcribe(audio)

    assert [s.text for s in segments] == ["fallback"]


def test_gpu_telemetry_disabled_for_lemonade(tmp_path):
    s = lemon_settings(tmp_path)
    assert not _gpu_telemetry_enabled(s)
    fw_settings = s.model_copy(update={"transcription_backend": "faster-whisper"})
    assert _gpu_telemetry_enabled(fw_settings)


def test_fingerprint_changes_invalidate_lemonade_cache_independently(tmp_path, audio):
    """Chunk duration, overlap and dedup version each invalidate Lemonade entries."""
    backend = lemon.LemonadeBackend(lemon_settings(tmp_path))
    baseline = backend.cache_fingerprint()
    assert (
        lemon.LemonadeBackend(
            lemon_settings(tmp_path, lemonade_chunk_seconds=999.0)
        ).cache_fingerprint()
        != baseline
    )
    assert (
        lemon.LemonadeBackend(
            lemon_settings(tmp_path, lemonade_overlap_seconds=9.0)
        ).cache_fingerprint()
        != baseline
    )
    with patch.object(lemon, "DEDUP_POLICY_VERSION", lemon.DEDUP_POLICY_VERSION + 1):
        assert lemon.LemonadeBackend(lemon_settings(tmp_path)).cache_fingerprint() != baseline


# --- fallback latch ---


def test_fallback_latches_backend_for_transcriber_lifetime(tmp_path, monkeypatch, mock_fw, audio):
    """After one eligible failure, the facade never submits to Lemonade again."""
    calls = install_urlopen(monkeypatch, [urllib.error.URLError(ConnectionRefusedError(111))])
    transcriber = Transcriber(lemon_settings(tmp_path))

    segments, _info = transcriber.transcribe(audio)
    assert [s.text for s in segments] == ["fallback"]
    assert len(calls) == 1
    assert transcriber._backend is mock_fw

    transcriber.transcribe(audio)  # later calls: never touch Lemonade again
    assert len(calls) == 1
    # The second call is served from the fw-identity cache — zero Lemonade, zero new fw work.
    assert mock_fw.transcribe.call_count == 1


def test_stereo_fallback_latches_backend_for_transcriber_lifetime(tmp_path, monkeypatch, mock_fw):
    mic, monitor = _stereo(tmp_path)
    ok = fw_result(["lemonade-monitor"])
    transcriber, _fw = _stereo_backend(tmp_path, monkeypatch, ok, LemonadeModelError("x"))
    transcriber.transcribe_stereo(mic, monitor)

    assert transcriber._backend is not None
    assert transcriber._backend.cache_fingerprint() == "fw-fingerprint"


# --- resume identity includes the effective language ---


def test_language_change_invalidates_mono_resume_cache(tmp_path, monkeypatch, audio):
    """A cached English result is never served to a run pinned to French."""
    transcriber = Transcriber(lemon_settings(tmp_path))
    install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4, "hello en")], language="english")])
    _segments, info = transcriber.transcribe(audio, language_override="english")
    assert info["language"] == "en"

    calls = install_urlopen(
        monkeypatch, [verbose_json([seg(0.0, 0.4, "bonjour")], language="french")]
    )
    _segments, info = transcriber.transcribe(audio, language_override="french")

    assert len(calls) == 1  # re-requested: the English cache entry must not match
    assert info["language"] == "fr"
    body = calls[0].data.decode("utf-8", errors="replace")
    assert 'name="language"\r\n\r\nfr' in body


def test_language_normalization_keeps_one_identity(tmp_path, monkeypatch, audio):
    """'english' and 'English' overrides map to the same resume identity."""
    transcriber = Transcriber(lemon_settings(tmp_path))
    install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4, "one")], language="english")])
    transcriber.transcribe(audio, language_override="english")

    calls = install_urlopen(monkeypatch, [verbose_json([seg(0.0, 0.4, "two")], language="english")])
    _segments, info = transcriber.transcribe(audio, language_override="English")

    assert len(calls) == 0  # cache hit — same normalized effective language
    assert info["language"] == "en"


def test_stereo_mic_cache_key_follows_monitor_language(tmp_path, monkeypatch):
    """Mic cache must miss after the monitor changes the mic's effective language."""
    mic = tmp_path / "mic.wav"
    monitor_en = tmp_path / "monitor_en.wav"
    monitor_fr = tmp_path / "monitor_fr.wav"
    write_wav(mic, 0.5)
    write_wav(monitor_en, 0.5)
    write_wav(monitor_fr, 0.5)
    transcriber = Transcriber(lemon_settings(tmp_path))

    install_urlopen(
        monkeypatch,
        [
            verbose_json([seg(0.0, 0.4, "en")], language="english"),
            verbose_json([seg(0.0, 0.4, "en-mic")], language="english"),
        ],
    )
    transcriber.transcribe_stereo(mic, monitor_en)

    # A new monitor detects French: the English-constrained mic entry is stale.
    calls = install_urlopen(
        monkeypatch,
        [
            verbose_json([seg(0.0, 0.4, "fr")], language="french"),
            verbose_json([seg(0.0, 0.4, "fr-mic")], language="french"),
        ],
    )
    transcriber.transcribe_stereo(mic, monitor_fr)

    assert len(calls) == 2  # mic re-requested under French, not served the en cache
    body = calls[1].data.decode("utf-8", errors="replace")
    assert 'name="language"\r\n\r\nfr' in body


def test_fallback_stereo_mic_key_follows_monitor_language(tmp_path, monkeypatch, mock_fw):
    """The faster-whisper fallback honors the same language-scoped mic identity."""
    mic = tmp_path / "mic.wav"
    monitor_en = tmp_path / "monitor_en.wav"
    monitor_fr = tmp_path / "monitor_fr.wav"
    write_wav(mic, 0.5)
    write_wav(monitor_en, 0.5)
    write_wav(monitor_fr, 0.5)

    install_urlopen(monkeypatch, [urllib.error.URLError(ConnectionRefusedError(111))])
    transcriber = Transcriber(lemon_settings(tmp_path))

    fw_calls: list[tuple[str, str, str | None]] = []

    def fake_fw_transcribe(path, stage="transcribe", on_status=None, language_override=None):
        fw_calls.append((Path(path).name, stage, language_override))
        language = "fr" if "fr" in Path(path).name or language_override == "fr" else "en"
        return fw_result([f"fw-{stage}"], language=language)

    mock_fw.transcribe.side_effect = fake_fw_transcribe

    transcriber.transcribe_stereo(mic, monitor_en)  # lemonade fails -> fw both channels
    transcriber.transcribe_stereo(mic, monitor_fr)  # monitor now establishes French

    mic_calls = [c for c in fw_calls if c[1] == "transcribe mic"]
    assert len(fw_calls) == 4  # monitor+mic per stereo run, mic cache never cross-hit
    assert mic_calls[0] == ("mic.wav", "transcribe mic", "en")
    assert mic_calls[-1] == ("mic.wav", "transcribe mic", "fr")


def test_fallback_latch_is_installed_before_local_backend_construction(tmp_path, monkeypatch):
    """If faster-whisper construction fails during fallback, the facade remains latched
    and never resubmits subsequent requests to Lemonade."""
    wav = tmp_path / "test.wav"
    write_wav(wav, 0.5)

    lemonade_calls = install_urlopen(
        monkeypatch,
        [
            TimeoutError("read timed out"),
            verbose_json([seg(0.0, 0.4, "lemonade should never receive this")]),
        ],
    )
    transcriber = Transcriber(lemon_settings(tmp_path))

    fw_attempts = 0

    def failing_new_fw(self):
        nonlocal fw_attempts
        fw_attempts += 1
        raise RuntimeError("Model download failed")

    monkeypatch.setattr(Transcriber, "_new_fw_backend", failing_new_fw)

    # First attempt: Lemonade times out, fallback initiates and attempts fw construction, which fails
    with pytest.raises(RuntimeError, match="Model download failed"):
        transcriber.transcribe(wav)

    assert len(lemonade_calls) == 1
    assert fw_attempts == 1

    # Second attempt: Transcriber is latched in fallback state. It must NOT call Lemonade again.
    with pytest.raises(RuntimeError, match="Model download failed"):
        transcriber.transcribe(wav)

    # Exactly 1 Lemonade call was ever made; the retry tried FW, never Lemonade.
    assert len(lemonade_calls) == 1
    assert fw_attempts == 2
