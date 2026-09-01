"""Live transcription tests — LiveTranscriber, WAV parsing, resampling, dedup."""

import io
import json
import struct
import threading
import time
import urllib.error
import wave
from email.message import Message
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import tapeback.live as live_mod
from tapeback._lemonade import LemonadeAuthenticationError
from tapeback.live import (
    LiveTranscriber,
    adjust_timestamps,
    deduplicate_overlap,
    find_data_offset,
    resample_48k_to_16k,
)
from tapeback.models import Segment, Word
from tapeback.settings import Settings
from tapeback.transcriber import Transcriber
from tests.fixtures import create_mono_wav, mock_whisper_transcribe


class _FakeResponse:
    """A urllib response that yields its body once, then EOF."""

    def __init__(self, body: bytes | bytearray):
        self._body = bytes(body)
        self._consumed = False

    def read(self, n: int = -1):
        if self._consumed:
            return b""
        self._consumed = True
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _lemon_verbose_json(text: str, language: str = "russian") -> bytes:
    """A successful Lemonade verbose_json response body."""
    return json.dumps(
        {
            "text": text,
            "segments": [{"start": 0.0, "end": 0.4, "text": text, "words": []}],
            "language": language,
            "language_probability": 0.9,
        }
    ).encode()


def _install_urlopen(monkeypatch, bodies: list[object]) -> list[object]:
    """Queue urllib responses (bytes or exceptions); repeat the last for later requests.

    Routes BOTH tapeback HTTP paths (default urlopen and the loopback no-proxy
    opener) through the same fake, and returns the recorded request list.
    """
    calls: list[object] = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        body = bodies[min(len(calls) - 1, len(bodies) - 1)]
        if isinstance(body, BaseException):
            raise body
        if isinstance(body, (bytes, bytearray)):
            return _FakeResponse(body)
        return body  # a pre-built response-like object

    class _FakeOpener:
        def open(self, request, timeout=None):
            return fake_urlopen(request, timeout=timeout)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("tapeback._lemonade._DEFAULT_OPENER", _FakeOpener())
    monkeypatch.setattr("tapeback._lemonade._NO_PROXY_OPENER", _FakeOpener())
    return calls


# --- find_data_offset ---


def test_find_data_offset_standard_wav(tmp_path):
    """Standard 44-byte WAV header: data starts at byte 44."""
    wav_path = tmp_path / "test.wav"
    create_mono_wav(wav_path, duration=0.1, sample_rate=16000)

    offset = find_data_offset(wav_path)
    assert offset == 44


def test_find_data_offset_extended_wav(tmp_path):
    """WAV with extra chunks before 'data' should still find correct offset."""
    wav_path = tmp_path / "extended.wav"

    # Build a WAV with an extra "INFO" chunk before "data"
    with open(wav_path, "wb") as f:
        # RIFF header (placeholder size)
        f.write(b"RIFF")
        f.write(struct.pack("<I", 0xFFFFFFFF))
        f.write(b"WAVE")

        # fmt chunk (standard PCM, mono, 16kHz, 16-bit)
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16))

        # Extra "INFO" chunk (4 bytes of padding)
        f.write(b"INFO")
        f.write(struct.pack("<I", 4))
        f.write(b"\x00\x00\x00\x00")

        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", 100))
        data_start = f.tell()
        f.write(b"\x00" * 100)

    offset = find_data_offset(wav_path)
    assert offset == data_start


def test_find_data_offset_nonexistent_file(tmp_path):
    """Non-existent file falls back to 44."""
    offset = find_data_offset(tmp_path / "nope.wav")
    assert offset == 44


def test_find_data_offset_not_riff(tmp_path):
    """Non-RIFF file falls back to 44."""
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"NOT_RIFF_DATA" * 10)
    assert find_data_offset(bad) == 44


# --- resample_48k_to_16k ---


def test_resample_48k_to_16k_length():
    """Output length should be input_length / 3."""
    # 4800 samples at 48kHz = 0.1s → 1600 samples at 16kHz
    samples_48k = np.zeros(4800, dtype=np.int16)
    result = resample_48k_to_16k(samples_48k.tobytes())
    assert len(result) == 1600


def test_resample_48k_to_16k_values():
    """Decimation picks every 3rd sample."""
    samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int16)
    result = resample_48k_to_16k(samples.tobytes())
    np.testing.assert_array_equal(result, [1, 4, 7])


# --- adjust_timestamps ---


def test_adjust_timestamps_offsets_segments_and_words():
    """All timestamps should be shifted by offset_seconds."""
    segments = [
        Segment(
            start=0.0,
            end=2.0,
            text="Hello",
            words=[
                Word(start=0.0, end=0.5, word="Hello", probability=0.9),
            ],
            speaker="You",
        ),
        Segment(start=2.0, end=4.0, text="World", words=None, speaker="Other"),
    ]

    result = adjust_timestamps(segments, 60.0)

    assert result[0].start == pytest.approx(60.0)
    assert result[0].end == pytest.approx(62.0)
    assert result[0].words is not None
    assert result[0].words[0].start == pytest.approx(60.0)
    assert result[0].words[0].end == pytest.approx(60.5)
    assert result[0].speaker == "You"

    assert result[1].start == pytest.approx(62.0)
    assert result[1].end == pytest.approx(64.0)
    assert result[1].words is None


def test_adjust_timestamps_preserves_text():
    """Text and speaker should not change."""
    segments = [Segment(start=0.0, end=1.0, text="Test", speaker="You")]
    result = adjust_timestamps(segments, 10.0)
    assert result[0].text == "Test"
    assert result[0].speaker == "You"


# --- deduplicate_overlap ---


def test_deduplicate_overlap_removes_duplicates():
    """Segments matching existing ones in the overlap zone should be removed."""
    existing = [
        Segment(start=55.0, end=58.0, text="Old segment", speaker="You"),
        Segment(start=58.0, end=60.0, text="Boundary", speaker="Other"),
    ]

    new_segments = [
        # Duplicate of existing (within tolerance)
        Segment(start=55.1, end=58.0, text="Old segment", speaker="You"),
        Segment(start=58.2, end=60.0, text="Boundary", speaker="Other"),
        # New segment past overlap
        Segment(start=62.0, end=65.0, text="New content", speaker="You"),
    ]

    result = deduplicate_overlap(existing, new_segments, overlap_start=60.0)

    # Only the new segment past the overlap should remain
    assert len(result) == 1
    assert result[0].text == "New content"


def test_deduplicate_overlap_no_existing():
    """With no existing segments, all new ones are kept."""
    new_segments = [
        Segment(start=0.0, end=2.0, text="First", speaker="You"),
    ]
    result = deduplicate_overlap([], new_segments, overlap_start=0.0)
    assert len(result) == 1


def test_deduplicate_overlap_zero_overlap():
    """With overlap_start=0, all segments are kept."""
    existing = [Segment(start=0.0, end=2.0, text="Old", speaker="You")]
    new_segments = [
        Segment(start=0.0, end=2.0, text="Old", speaker="You"),
        Segment(start=2.0, end=4.0, text="New", speaker="You"),
    ]
    result = deduplicate_overlap(existing, new_segments, overlap_start=0.0)
    assert len(result) == 2


# --- LiveTranscriber ---


def test_live_transcriber_start_stop_lifecycle(tmp_path, tmp_vault):
    """LiveTranscriber should start a thread, process final chunk on stop, and clean up."""
    settings = Settings(vault_path=tmp_vault, live_interval=1, live_min_chunk=0.01)

    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"
    create_mono_wav(mic_path, duration=0.5, sample_rate=48000)
    create_mono_wav(monitor_path, duration=0.5, sample_rate=48000)

    mock_model = mock_whisper_transcribe([(0.0, 0.5, "Test speech.")])

    with patch("tapeback._fw_backend.WhisperModel", return_value=mock_model):
        lt = LiveTranscriber(settings, "2026-04-18_10-00-00", mic_path, monitor_path)
        lt.start()
        # Let it run briefly — the thread will pick up the audio
        lt._stop_event.wait(timeout=0.1)
        lt.stop()

    # Live markdown should have been written
    live_md = tmp_vault / "meetings" / "2026-04-18_10-00-00_live.md"
    assert live_md.exists()


def test_live_transcriber_no_crash_on_empty_audio(tmp_path, tmp_vault):
    """LiveTranscriber should not crash when WAV files don't exist yet."""
    settings = Settings(vault_path=tmp_vault, live_interval=1, live_min_chunk=0.01)

    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"
    # Files don't exist!

    lt = LiveTranscriber(settings, "test-session", mic_path, monitor_path)
    lt.start()
    lt._stop_event.wait(timeout=0.1)
    lt.stop()

    # Should still write the "waiting" markdown
    live_md = tmp_vault / "meetings" / "test-session_live.md"
    assert live_md.exists()
    assert "Waiting for first transcription cycle" in live_md.read_text()


def test_live_transcriber_no_crash_on_transcription_error(tmp_path, tmp_vault):
    """LiveTranscriber should catch transcription errors and continue."""
    settings = Settings(vault_path=tmp_vault, live_interval=1, live_min_chunk=0.01)

    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"
    create_mono_wav(mic_path, duration=1.0, sample_rate=48000)
    create_mono_wav(monitor_path, duration=1.0, sample_rate=48000)

    mock_model = MagicMock()
    mock_model.transcribe.side_effect = RuntimeError("CUDA out of memory")

    with patch("tapeback._fw_backend.WhisperModel", return_value=mock_model):
        lt = LiveTranscriber(settings, "error-session", mic_path, monitor_path)
        lt.start()
        lt._stop_event.wait(timeout=0.1)
        lt.stop()

    # Should not crash — live markdown still written (empty segments)
    live_md = tmp_vault / "meetings" / "error-session_live.md"
    assert live_md.exists()


def test_live_transcriber_process_chunk_accumulates_segments(tmp_path, tmp_vault):
    """_process_chunk should accumulate segments from transcription."""
    settings = Settings(
        vault_path=tmp_vault,
        live_interval=60,
        live_min_chunk=0.01,
        live_overlap=0.0,
        sample_rate=48000,
    )

    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"
    create_mono_wav(mic_path, duration=1.0, sample_rate=48000)
    create_mono_wav(monitor_path, duration=1.0, sample_rate=48000)

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Hello world.")])

    with patch("tapeback._fw_backend.WhisperModel", return_value=mock_model):
        lt = LiveTranscriber(settings, "chunk-test", mic_path, monitor_path)
        lt._process_chunk()

    # Should have accumulated segments from both channels
    assert len(lt._segments) > 0
    speakers = {s.speaker for s in lt._segments}
    assert "You" in speakers
    assert "Other" in speakers


def test_live_markdown_written_atomically(tmp_path, tmp_vault):
    """Live markdown should be written via atomic temp+rename pattern."""
    settings = Settings(vault_path=tmp_vault, live_interval=60)

    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"

    lt = LiveTranscriber(settings, "atomic-test", mic_path, monitor_path)
    lt._write_live_markdown()

    live_md = tmp_vault / "meetings" / "atomic-test_live.md"
    assert live_md.exists()
    # No leftover .tmp file
    assert not live_md.with_suffix(".md.tmp").exists()


def test_write_chunk_wav_creates_valid_wav(tmp_path):
    """_write_chunk_wav should create a readable 16kHz mono WAV."""
    samples = np.array([0, 100, -100, 200, -200], dtype=np.int16)
    chunk_path = tmp_path / "chunk.wav"

    LiveTranscriber._write_chunk_wav(samples, chunk_path)

    with wave.open(str(chunk_path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == 5


# --- Lemonade fallback latch in live mode ---


def test_live_mic_timeout_latches_and_never_resubmits_to_lemonade(tmp_path, tmp_vault, monkeypatch):
    """A Lemonade timeout in live mode latches the facade to faster-whisper.

    The paired monitor/mic call in the same interval, and every later interval,
    must never submit to Lemonade again — no mixed-backend live transcript and no
    pile-up of timed-out server jobs. The monitor transcribes first, so this
    exercises the FIRST channel failing; the sibling test below covers the second.
    """
    settings = Settings(
        vault_path=tmp_vault,
        transcription_backend="lemonade",
        resume_cache_dir=tmp_path / "resume",
        live_min_chunk=0.1,
        live_interval=60,
    )
    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"
    create_mono_wav(mic_path, duration=0.5, sample_rate=48000)
    create_mono_wav(monitor_path, duration=0.5, sample_rate=48000)

    lemonade_calls: list[object] = []

    def fake_urlopen(request, timeout=None):
        lemonade_calls.append(request)
        raise TimeoutError("read timed out")

    class _FakeOpener:
        def open(self, request, timeout=None):
            return fake_urlopen(request, timeout=timeout)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("tapeback._lemonade._DEFAULT_OPENER", _FakeOpener())
    monkeypatch.setattr("tapeback._lemonade._NO_PROXY_OPENER", _FakeOpener())

    fw = MagicMock()
    fw.cache_fingerprint.return_value = "fw-fingerprint"

    def fw_transcribe(path, **kwargs):
        return [Segment(start=0.0, end=0.4, text="fw text")], {
            "language": "en",
            "duration": 0.5,
            "partial": False,
        }

    fw.transcribe.side_effect = fw_transcribe
    monkeypatch.setattr(Transcriber, "_new_fw_backend", lambda self: fw)
    monkeypatch.setattr(live_mod, "load_transcriber", lambda s: Transcriber(s))

    lt = LiveTranscriber(settings, "latch-session", mic_path, monitor_path)
    lt._process_chunk()  # mic times out on Lemonade -> fallback + latch; monitor via fw

    assert len(lemonade_calls) == 1  # only the first mic attempt reached Lemonade

    # Grow both channels so the next interval has new audio to process.
    with open(mic_path, "ab") as f:
        f.write(b"\x00\x00" * 16000)
    with open(monitor_path, "ab") as f:
        f.write(b"\x00\x00" * 16000)

    lt._process_chunk()  # later interval: everything via the latched fw backend

    assert len(lemonade_calls) == 1  # still exactly one Lemonade request, ever
    texts = [s.text for s in lt._segments]
    assert texts and all(t == "fw text" for t in texts)  # both channels came from fw


def test_live_second_channel_fallback_leaves_no_mixed_interval(tmp_path, tmp_vault, monkeypatch):
    """A fallback on an interval's SECOND channel must not commit a Lemonade first
    channel beside faster-whisper output.

    Regression: mic and monitor were two independent mono calls; a monitor timeout
    retried only the monitor through faster-whisper and the already-successful
    Lemonade mic segments were committed beside it. The pair now transcribes as ONE
    transaction, so a second-channel fallback discards both and resolves both through
    faster-whisper.
    """
    settings = Settings(
        vault_path=tmp_vault,
        transcription_backend="lemonade",
        resume_cache_dir=tmp_path / "resume",
        live_min_chunk=0.1,
        live_interval=60,
    )
    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"
    create_mono_wav(mic_path, duration=0.5, sample_rate=48000)
    create_mono_wav(monitor_path, duration=0.5, sample_rate=48000)

    # Stereo transcribes the monitor first: it succeeds on Lemonade, then the mic
    # times out — the fallback must redo BOTH channels on faster-whisper.
    lemonade_calls = _install_urlopen(
        monkeypatch,
        [_lemon_verbose_json("lemonade-monitor"), TimeoutError("read timed out")],
    )

    fw = MagicMock()
    fw.cache_fingerprint.return_value = "fw-fingerprint"

    def fw_transcribe(path, **kwargs):
        return [Segment(start=0.0, end=0.4, text="fw text")], {
            "language": "en",
            "duration": 0.5,
            "partial": False,
        }

    fw.transcribe.side_effect = fw_transcribe
    monkeypatch.setattr(Transcriber, "_new_fw_backend", lambda self: fw)
    monkeypatch.setattr(live_mod, "load_transcriber", lambda s: Transcriber(s))

    lt = LiveTranscriber(settings, "pair-fallback", mic_path, monitor_path)
    lt._process_chunk()

    assert len(lemonade_calls) == 2  # monitor -> Lemonade ok, mic -> Lemonade timeout
    assert [s.text for s in lt._segments] == ["fw text", "fw text"]  # never mixed
    assert lt._transcriber is not None
    assert lt._transcriber._backend is fw  # latched for every later interval


def test_live_channel_error_rolls_back_both_cursors(tmp_path, tmp_vault, monkeypatch):
    """An error on one channel must not advance the other channel's cursor, or the
    successful interval is permanently dropped.

    Regression: the mic cursor advanced as soon as the mic mono call returned; a later
    monitor auth/config error exited before segments were committed, so the next cycle
    started past that mic audio and never re-read it.
    """
    settings = Settings(
        vault_path=tmp_vault,
        transcription_backend="lemonade",
        resume_cache_dir=tmp_path / "resume",
        live_min_chunk=0.1,
        live_interval=60,
    )
    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"
    create_mono_wav(mic_path, duration=0.5, sample_rate=48000)
    create_mono_wav(monitor_path, duration=0.5, sample_rate=48000)

    auth_error = urllib.error.HTTPError("http://x", 401, "auth", Message(), io.BytesIO(b"{}"))
    # The monitor is the first channel of the pair; a non-fallback auth error must
    # propagate out of _process_chunk with both cursors untouched.
    _install_urlopen(monkeypatch, [auth_error])
    monkeypatch.setattr(live_mod, "load_transcriber", lambda s: Transcriber(s))

    lt = LiveTranscriber(settings, "rollback", mic_path, monitor_path)
    with pytest.raises(LemonadeAuthenticationError):
        lt._process_chunk()

    assert lt._mic_byte_offset == 0
    assert lt._monitor_byte_offset == 0
    assert lt._segments == []

    # The server recovers: the same interval is re-read and nothing was lost.
    _install_urlopen(monkeypatch, [_lemon_verbose_json("monitor"), _lemon_verbose_json("mic")])
    lt._process_chunk()

    assert lt._mic_byte_offset > 0
    assert lt._monitor_byte_offset > 0
    assert len(lt._segments) == 2
    assert {"You", "Other"} == {s.speaker for s in lt._segments}


# --- stop() lifecycle ---


def test_live_lemonade_request_budget_is_capped(tmp_vault):
    """A live Lemonade request can never have a worst case past the join budget.

    The 600 s default inference timeout exists for long final files; a live
    interval is one small pair transaction, so stop()'s join must be able to
    outlast any request it waits behind.
    """
    settings = Settings(
        vault_path=tmp_vault,
        transcription_backend="lemonade",
        lemonade_timeout_seconds=600.0,
    )
    lt = LiveTranscriber(
        settings, "budget-session", tmp_vault / "mic.wav", tmp_vault / "monitor.wav"
    )

    assert lt._settings.lemonade_timeout_seconds == live_mod._LIVE_REQUEST_BUDGET_SECONDS


def test_live_faster_whisper_settings_pass_through_untouched(tmp_vault):
    """The request-budget cap is a Lemonade-transport concern only."""
    settings = Settings(vault_path=tmp_vault, lemonade_timeout_seconds=600.0)
    lt = LiveTranscriber(
        settings, "passthrough-session", tmp_vault / "mic.wav", tmp_vault / "monitor.wav"
    )

    assert lt._settings.lemonade_timeout_seconds == 600.0


def test_stop_does_not_return_while_the_worker_is_alive(tmp_path, tmp_vault, monkeypatch):
    """stop() must establish its lifecycle boundary: raise rather than return
    while the worker can still issue requests or write the live note.

    Regression: stop() joined with a fixed 120 s timeout, never checked
    is_alive(), and a Lemonade request stalled for 121-600 s survived the join —
    stop() returned, the final pipeline started, and the worker later woke up to
    upload another interval and rewrite the live note under it.
    """
    settings = Settings(vault_path=tmp_vault, live_interval=1, live_min_chunk=0.01)

    entered = threading.Event()
    release = threading.Event()

    def stalled_chunk(self):
        entered.set()
        release.wait(timeout=10)

    monkeypatch.setattr(LiveTranscriber, "_process_chunk", stalled_chunk)
    monkeypatch.setattr(live_mod, "_STOP_JOIN_TIMEOUT_SECONDS", 0.2)

    lt = LiveTranscriber(settings, "stall-session", tmp_path / "mic.wav", tmp_path / "monitor.wav")
    lt.start()
    assert entered.wait(timeout=5)

    with pytest.raises(RuntimeError, match="final processing aborted"):
        lt.stop()

    release.set()
    lt._thread.join(timeout=5)
    assert not lt._thread.is_alive()


def test_no_work_after_stop_returns(tmp_path, tmp_vault, monkeypatch):
    """After stop() returns, the worker is dead: no further chunk processing,
    remote request, or live-note write can happen afterwards."""
    settings = Settings(vault_path=tmp_vault, live_interval=1, live_min_chunk=0.01)
    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"
    create_mono_wav(mic_path, duration=0.5, sample_rate=48000)
    create_mono_wav(monitor_path, duration=0.5, sample_rate=48000)

    chunk_calls: list[int] = []
    write_calls: list[int] = []

    def counting_chunk(self):
        chunk_calls.append(1)
        self._write_live_markdown()

    monkeypatch.setattr(LiveTranscriber, "_process_chunk", counting_chunk)
    monkeypatch.setattr(LiveTranscriber, "_write_live_markdown", lambda self: write_calls.append(1))

    lt = LiveTranscriber(settings, "poststop-session", mic_path, monitor_path)
    lt.start()
    time.sleep(0.3)  # a couple of intervals
    lt.stop()

    assert not lt._thread.is_alive()
    chunks_at_stop = len(chunk_calls)
    writes_at_stop = len(write_calls)
    time.sleep(0.3)
    assert len(chunk_calls) == chunks_at_stop
    assert len(write_calls) == writes_at_stop
