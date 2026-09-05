"""Regression tests for pipeline bugs."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from tapeback.pipeline import process_file, process_mono_file, process_stereo_file, stop_and_process
from tapeback.settings import Settings
from tests.fixtures import (
    create_mono_wav,
    create_stereo_wav_segments,
    mock_whisper_transcribe,
)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_stereo_no_diarize_returns_no_raw_segments(tmp_path):
    """Without diarization, raw_segments must be None to avoid duplicate markdown sections.

    Bug: process_stereo_file always returned raw_segments, causing format_markdown
    to render two identical "## Transcript" + "## Diarized Transcript" sections
    when --no-diarize was used.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault, transcription_backend="faster-whisper")

    stereo = tmp_path / "stereo.wav"
    create_stereo_wav_segments(stereo, 48000, [(1.0, 0.8, 0.003), (1.0, 0.003, 0.8)])

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Speech.")])

    with patch("tapeback._fw_backend.WhisperModel", return_value=mock_model):
        _segments, _info, raw_segments = process_stereo_file(
            stereo, output_dir, settings, diarize=False
        )

    assert raw_segments is None


def test_process_file_cleans_up_temp_dir_on_exception(tmp_path):
    """process_file must clean up temp directory even if processing fails."""
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault)

    audio = tmp_path / "test.wav"
    create_mono_wav(audio, duration=1.0, sample_rate=48000, amplitude=0.5)

    created_dirs: list[Path] = []
    original_mkdtemp = tempfile.mkdtemp

    def fake_mkdtemp(prefix="tapeback_"):
        res = Path(original_mkdtemp(prefix=prefix))
        created_dirs.append(res)
        return str(res)

    with (
        patch("tempfile.mkdtemp", side_effect=fake_mkdtemp),
        patch(
            "tapeback.pipeline.process_mono_file",
            side_effect=RuntimeError("Pipeline processing error"),
        ),
        pytest.raises(RuntimeError, match="Pipeline processing error"),
    ):
        process_file(audio, settings, diarize=False)

    assert len(created_dirs) == 1
    assert not created_dirs[0].exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_stop_and_process_cleans_up_session_dir_on_exception(tmp_path):
    """stop_and_process must clean up session directory even if processing fails."""
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault)

    session_dir = tmp_path / "session_123"
    session_dir.mkdir()
    monitor_wav = session_dir / "monitor.wav"
    mic_wav = session_dir / "mic.wav"
    create_mono_wav(monitor_wav, duration=1.0)
    create_mono_wav(mic_wav, duration=1.0)

    mock_recorder = MagicMock()
    mock_recorder.stop.return_value = (monitor_wav, mic_wav)

    with (
        patch(
            "tapeback.pipeline.process_stereo_file",
            side_effect=RuntimeError("Pipeline processing error"),
        ),
        pytest.raises(RuntimeError, match="Pipeline processing error"),
    ):
        stop_and_process(mock_recorder, settings, diarize=False)

    assert not session_dir.exists()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_stereo_times_every_heavy_stage(tmp_path):
    """Every heavy stage must report its own timing.

    Bug: load_stereo_channels (reads the whole 48 kHz stereo file into two float32
    arrays) and gate_wav_inactive (a per-100ms Python loop over the mic channel)
    ran outside any stage_timer, so their cost showed up as unexplained dead time
    between "Stage 'merge'" and "Stage 'load model'" and could not be attributed.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(
        vault_path=vault, gate_mic_silence=True, transcription_backend="faster-whisper"
    )

    stereo = tmp_path / "stereo.wav"
    create_stereo_wav_segments(stereo, 48000, [(1.0, 0.8, 0.003), (1.0, 0.003, 0.8)])

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Speech.")])

    messages: list[str] = []
    with patch("tapeback._fw_backend.WhisperModel", return_value=mock_model):
        process_stereo_file(stereo, output_dir, settings, diarize=False, on_status=messages.append)

    timed = {m.split("'")[1] for m in messages if m.startswith("Stage '")}
    assert {"load channels", "split", "gate mic", "load model"} <= timed


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_stereo_reports_resolved_device(tmp_path):
    """The pipeline must state where Whisper landed.

    Bug: a run that silently fell back to CPU (roughly 10x slower) produced the
    same output as a healthy GPU run — the only signal was a stderr warning that
    scrolled past between other status lines.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(
        vault_path=vault,
        device="cpu",
        compute_type="int8",
        transcription_backend="faster-whisper",
    )

    stereo = tmp_path / "stereo.wav"
    create_stereo_wav_segments(stereo, 48000, [(1.0, 0.8, 0.003)])

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Speech.")])

    messages: list[str] = []
    with patch("tapeback._fw_backend.WhisperModel", return_value=mock_model):
        process_stereo_file(stereo, output_dir, settings, diarize=False, on_status=messages.append)

    assert any(m == "Whisper: large-v3-turbo on cpu/int8" for m in messages)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_stereo_diarize_without_hf_token_returns_no_raw_segments(tmp_path):
    """When diarize=True but no HF token is configured, diarization is skipped —
    so raw_segments must be None (no duplicate section)."""
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(
        vault_path=vault, hf_token=SecretStr(""), transcription_backend="faster-whisper"
    )

    stereo = tmp_path / "stereo.wav"
    create_stereo_wav_segments(stereo, 48000, [(1.0, 0.8, 0.003), (1.0, 0.003, 0.8)])

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Speech.")])

    with patch("tapeback._fw_backend.WhisperModel", return_value=mock_model):
        _segments, _info, raw_segments = process_stereo_file(
            stereo, output_dir, settings, diarize=True
        )

    assert raw_segments is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_process_mono_no_diarize_returns_no_raw_segments(tmp_path):
    """Mono pipeline must also return None raw_segments when diarization is off."""
    vault = tmp_path / "vault"
    vault.mkdir()
    settings = Settings(vault_path=vault)

    mono = tmp_path / "mono.wav"
    create_mono_wav(mono, duration=1.0, sample_rate=48000, amplitude=0.5)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    mock_model = mock_whisper_transcribe([(0.0, 1.0, "Speech.")])

    with patch("tapeback._fw_backend.WhisperModel", return_value=mock_model):
        _segments, _info, raw_segments = process_mono_file(
            mono, output_dir, settings, diarize=False
        )

    assert raw_segments is None
