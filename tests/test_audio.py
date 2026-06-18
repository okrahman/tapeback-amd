import shutil
import wave
from unittest.mock import patch

import numpy as np
import pytest

from tapeback.audio import (
    convert_to_mono16k,
    gate_wav_inactive,
    merge_channels,
    split_channels_16k,
)
from tests.fixtures import create_silent_wav, create_stereo_wav


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_merge_channels(tmp_path):
    """Merge two mono WAVs into a stereo archive WAV."""

    monitor = tmp_path / "monitor.wav"
    mic = tmp_path / "mic.wav"
    output_dir = tmp_path / "output"

    create_silent_wav(monitor)
    create_silent_wav(mic)

    stereo_path = merge_channels(monitor, mic, output_dir)

    assert stereo_path.exists()

    # Verify stereo is 2 channels
    with wave.open(str(stereo_path), "rb") as wf:
        assert wf.getnchannels() == 2


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_convert_to_mono16k(tmp_path):
    """Convert a WAV file to 16kHz mono."""

    input_file = tmp_path / "input.wav"
    output_dir = tmp_path / "output"

    create_silent_wav(input_file, sample_rate=44100)

    result = convert_to_mono16k(input_file, output_dir)

    assert result.exists()
    with wave.open(str(result), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_split_channels_16k(tmp_path):
    """Split stereo into two mono 16kHz WAVs."""

    stereo = tmp_path / "stereo.wav"
    output_dir = tmp_path / "output"

    create_stereo_wav(
        stereo, duration=1.0, sample_rate=48000, left_amplitude=0.8, right_amplitude=0.3
    )

    mic_16k, monitor_16k = split_channels_16k(stereo, output_dir)

    assert mic_16k.exists()
    assert monitor_16k.exists()

    # Verify both are mono 16kHz
    with wave.open(str(mic_16k), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000

    with wave.open(str(monitor_16k), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000


def test_gate_wav_inactive_silences_listening_region(tmp_path):
    """gate_wav_inactive zeroes the listening half of a 16k mic WAV in place."""
    sr_16k = 16000
    raw_sr = 48000
    mic_path = tmp_path / "mic_16k.wav"
    samples = np.full(2 * sr_16k, 8000, dtype=np.int16)
    with wave.open(str(mic_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr_16k)
        wf.writeframes(samples.tobytes())

    # User speaks the first second, listens the second (mic quiet, monitor loud).
    mic_raw = np.concatenate([np.full(raw_sr, 5000.0), np.zeros(raw_sr)]).astype(np.float32)
    monitor_raw = np.concatenate([np.zeros(raw_sr), np.full(raw_sr, 5000.0)]).astype(np.float32)

    gate_wav_inactive(mic_path, mic_raw, monitor_raw, raw_sr)

    with wave.open(str(mic_path), "rb") as wf:
        out = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    assert np.any(out[:sr_16k] != 0)
    assert np.all(out[sr_16k:] == 0)


def test_empty_file_raises(tmp_path):
    """Empty WAV file should raise RuntimeError."""

    monitor = tmp_path / "monitor.wav"
    mic = tmp_path / "mic.wav"
    monitor.write_bytes(b"")
    mic.write_bytes(b"")

    with (
        patch("tapeback.audio.shutil.which", return_value="/usr/bin/ffmpeg"),
        pytest.raises(RuntimeError, match="No audio recorded"),
    ):
        merge_channels(monitor, mic, tmp_path / "output")


def test_ffmpeg_not_found(tmp_path):
    """Should give clear error when ffmpeg is not installed."""

    monitor = tmp_path / "monitor.wav"
    mic = tmp_path / "mic.wav"
    create_silent_wav(monitor)
    create_silent_wav(mic)

    with (
        patch("tapeback.audio.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="ffmpeg not found"),
    ):
        merge_channels(monitor, mic, tmp_path / "output")
