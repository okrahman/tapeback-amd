"""Exact digital-silence handling for completed and live stereo processing."""

import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import tapeback.live as live_mod
import tapeback.pipeline as pipeline_mod
from tapeback._lemonade import LemonadeUnavailableError
from tapeback.channel import is_channel_active
from tapeback.live import LiveTranscriber
from tapeback.models import Segment
from tapeback.pipeline import process_stereo_file
from tapeback.settings import Settings
from tapeback.transcriber import Transcriber
from tests.fixtures import create_stereo_wav_segments


def write_mono_wav(path: Path, duration: float, value: int, rate: int = 16000) -> None:
    samples = np.full(int(duration * rate), value, dtype=np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(samples.tobytes())


def write_live_wav(path: Path, pcm: bytes, rate: int = 48000) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm)


def fake_backend() -> MagicMock:
    backend = MagicMock()
    backend.cache_fingerprint.return_value = "lemonade-test"
    backend.transcribe.return_value = (
        [Segment(start=0.0, end=0.4, text="speech")],
        {"language": "en", "duration": 28.672, "partial": False},
    )
    return backend


def make_transcriber(tmp_path: Path, backend: MagicMock) -> Transcriber:
    vault = tmp_path / "vault"
    vault.mkdir()
    transcriber = Transcriber(
        Settings(
            vault_path=vault,
            transcription_backend="lemonade",
            resume_cache=True,
            resume_cache_dir=tmp_path / "resume",
            stage_pause_seconds=10.0,
        )
    )
    transcriber._backend = backend
    return transcriber


def test_exact_activity_keeps_amplitude_one_and_rejects_only_zero() -> None:
    assert not is_channel_active(np.zeros(4, dtype=np.int16))
    assert not is_channel_active(np.array([], dtype=np.int16))
    assert is_channel_active(np.array([0, 1, 0], dtype=np.int16))


def test_transcribe_stereo_skips_both_silent_channels_with_real_durations(tmp_path):
    mic = tmp_path / "mic.wav"
    monitor = tmp_path / "monitor.wav"
    write_mono_wav(mic, 28.672, 0)
    write_mono_wav(monitor, 28.672, 0)
    backend = fake_backend()
    transcriber = make_transcriber(tmp_path, backend)
    transcriber._load_resume = MagicMock()
    transcriber._store_resume = MagicMock()
    transcriber._pace = MagicMock()
    statuses: list[str] = []

    mic_segments, monitor_segments, info = transcriber.transcribe_stereo(
        mic,
        monitor,
        on_status=statuses.append,
        mic_active=False,
        monitor_active=False,
    )

    assert mic_segments == []
    assert monitor_segments == []
    assert info["duration"] == pytest.approx(28.672)
    assert info["partial"] is False
    assert statuses == [
        "Skipping monitor transcription — channel is digitally silent.",
        "Skipping mic transcription — channel is digitally silent.",
    ]
    backend.transcribe.assert_not_called()
    transcriber._load_resume.assert_not_called()
    transcriber._store_resume.assert_not_called()
    transcriber._pace.assert_not_called()


@pytest.mark.parametrize(
    ("mic_active", "monitor_active", "stage"),
    [(True, False, "transcribe mic"), (False, True, "transcribe monitor")],
)
def test_transcribe_stereo_only_calls_the_active_channel(
    tmp_path, mic_active, monitor_active, stage
):
    mic = tmp_path / "mic.wav"
    monitor = tmp_path / "monitor.wav"
    write_mono_wav(mic, 2.5, 1 if mic_active else 0)
    write_mono_wav(monitor, 2.5, 1 if monitor_active else 0)
    backend = fake_backend()
    transcriber = make_transcriber(tmp_path, backend)
    transcriber._load_resume = MagicMock(return_value=None)
    transcriber._store_resume = MagicMock()

    mic_segments, monitor_segments, _info = transcriber.transcribe_stereo(
        mic, monitor, mic_active=mic_active, monitor_active=monitor_active
    )

    assert backend.transcribe.call_count == 1
    assert backend.transcribe.call_args.kwargs["stage"] == stage
    assert len(mic_segments) == int(mic_active)
    assert len(monitor_segments) == int(monitor_active)
    assert transcriber._load_resume.call_count == 1
    assert transcriber._store_resume.call_count == 1


def test_sole_active_channel_falls_back_without_transcribing_silent_sibling(tmp_path, monkeypatch):
    mic = tmp_path / "mic.wav"
    monitor = tmp_path / "monitor.wav"
    write_mono_wav(mic, 2.5, 1)
    write_mono_wav(monitor, 2.5, 0)

    lemonade = fake_backend()
    lemonade.transcribe.side_effect = LemonadeUnavailableError("server unavailable")
    fallback = fake_backend()
    fallback.cache_fingerprint.return_value = "fw-test"
    monkeypatch.setattr(Transcriber, "_new_fw_backend", lambda self: fallback)
    transcriber = make_transcriber(tmp_path, lemonade)

    mic_segments, monitor_segments, _info = transcriber.transcribe_stereo(
        mic, monitor, use_resume=False, mic_active=True, monitor_active=False
    )

    assert len(mic_segments) == 1
    assert monitor_segments == []
    assert lemonade.transcribe.call_count == 1
    assert lemonade.transcribe.call_args.kwargs["stage"] == "transcribe mic"
    assert fallback.transcribe.call_count == 1
    assert fallback.transcribe.call_args.kwargs["stage"] == "transcribe mic"


def test_completed_pipeline_passes_raw_and_post_gate_activity(tmp_path, monkeypatch):
    stereo = tmp_path / "stereo.wav"
    # The mic contains only amplitude-1-ish PCM values while the monitor is exact
    # zero. This is short enough to cover the sub-30-second case.
    create_stereo_wav_segments(stereo, 48000, [(2.5, 1 / 32767, 0.0)])
    settings = Settings(
        vault_path=tmp_path / "vault",
        transcription_backend="lemonade",
        gate_mic_silence=False,
    )
    settings.vault_path.mkdir()
    transcriber = MagicMock()
    transcriber.describe.return_value = "test backend"
    transcriber.transcribe_stereo.return_value = ([], [], {"duration": 2.5})
    monkeypatch.setattr(pipeline_mod, "load_transcriber", lambda _settings: transcriber)
    monkeypatch.setattr(pipeline_mod, "save_audio_to_vault", lambda *args: tmp_path / "saved.wav")

    process_stereo_file(stereo, tmp_path / "output", settings, diarize=False)

    kwargs = transcriber.transcribe_stereo.call_args.kwargs
    assert kwargs["mic_active"] is True
    assert kwargs["monitor_active"] is False


def test_live_all_silent_interval_advances_cursors_then_processes_later_audio(
    tmp_path, monkeypatch
):
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(
        vault_path=vault,
        live=True,
        live_min_chunk=0.1,
        live_overlap=0.0,
        live_interval=60,
    )
    mic_path = tmp_path / "mic.wav"
    monitor_path = tmp_path / "monitor.wav"
    write_live_wav(mic_path, b"\x00\x00" * 48000)
    write_live_wav(monitor_path, b"\x00\x00" * 48000)
    lt = LiveTranscriber(settings, "silent-interval", mic_path, monitor_path)
    ensure = MagicMock(side_effect=AssertionError("silent interval loaded a backend"))
    monkeypatch.setattr(lt, "_ensure_transcriber", ensure)

    lt._process_chunk()

    assert lt._mic_byte_offset == 96000
    assert lt._monitor_byte_offset == 96000
    assert not (tmp_path / "chunk_mic.wav").exists()
    assert not (tmp_path / "chunk_monitor.wav").exists()
    ensure.assert_not_called()

    backend_transcriber = MagicMock()
    backend_transcriber._backend.cache_fingerprint.return_value = "live-fp"
    backend_transcriber.transcribe_stereo.return_value = ([], [], {"duration": 0.5})
    monkeypatch.setattr(lt, "_ensure_transcriber", lambda: backend_transcriber)
    monkeypatch.setattr(live_mod, "save_live_markdown", MagicMock())
    with open(mic_path, "ab") as file:
        file.write(b"\x01\x00" * 24000)
    with open(monitor_path, "ab") as file:
        file.write(b"\x01\x00" * 24000)

    lt._process_chunk()

    assert lt._mic_byte_offset == 144000
    assert lt._monitor_byte_offset == 144000
    kwargs = backend_transcriber.transcribe_stereo.call_args.kwargs
    assert kwargs["mic_active"] is True
    assert kwargs["monitor_active"] is True


def test_live_pair_writes_only_active_channel_and_full_replay_passes_activity(
    tmp_path, monkeypatch
):
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault, live=True, live_overlap=0.0)
    lt = LiveTranscriber(settings, "pair-activity", tmp_path / "mic.wav", tmp_path / "monitor.wav")
    transcriber = MagicMock()
    transcriber.transcribe_stereo.return_value = ([], [], {"duration": 0.1})
    writer = MagicMock()
    monkeypatch.setattr(lt, "_write_chunk_wav", writer)

    active = b"\x01\x00" * 4800
    silent = b"\x00\x00" * 4800
    lt._transcribe_pair(transcriber, active, silent, 0)

    assert writer.call_count == 1
    kwargs = transcriber.transcribe_stereo.call_args.kwargs
    assert kwargs["mic_active"] is True
    assert kwargs["monitor_active"] is False

    transcriber.transcribe_stereo.reset_mock()
    writer.reset_mock()
    lt._transcribe_pair_audio(transcriber, active, silent)

    assert writer.call_count == 1
    kwargs = transcriber.transcribe_stereo.call_args.kwargs
    assert kwargs["mic_active"] is True
    assert kwargs["monitor_active"] is False


def test_live_single_silent_chunk_never_writes_or_calls_backend(tmp_path, monkeypatch):
    settings = Settings(vault_path=tmp_path, live=True)
    lt = LiveTranscriber(settings, "single-silent", tmp_path / "mic.wav", tmp_path / "monitor.wav")
    transcriber = MagicMock()
    writer = MagicMock()
    monkeypatch.setattr(lt, "_write_chunk_wav", writer)

    result = lt._transcribe_chunk(transcriber, b"\x00\x00" * 4800, 0, 0, is_mic=True)

    assert result == []
    writer.assert_not_called()
    transcriber.transcribe.assert_not_called()
