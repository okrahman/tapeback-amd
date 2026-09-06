from pathlib import Path
from unittest.mock import patch

from tapeback.pipeline import is_stereo
from tests.fixtures import create_silent_wav, create_stereo_wav


def test_is_stereo_with_stereo_wav(tmp_path: Path) -> None:
    """Stereo WAV file with 2 channels returns True."""
    stereo_path = tmp_path / "stereo.wav"
    create_stereo_wav(
        stereo_path,
        duration=1.0,
        sample_rate=16000,
        left_amplitude=0.5,
        right_amplitude=0.5,
    )
    assert is_stereo(stereo_path) is True


def test_is_stereo_with_mono_wav(tmp_path: Path) -> None:
    """Mono WAV file with 1 channel returns False."""
    mono_path = tmp_path / "mono.wav"
    create_silent_wav(mono_path, duration=1.0, sample_rate=16000)
    assert is_stereo(mono_path) is False


def test_is_stereo_non_existent_file(tmp_path: Path) -> None:
    """Non-existent file path returns False without raising an exception."""
    non_existent = tmp_path / "does_not_exist.wav"
    assert is_stereo(non_existent) is False


def test_is_stereo_invalid_file(tmp_path: Path) -> None:
    """Invalid / corrupted file format returns False."""
    text_file = tmp_path / "not_audio.txt"
    text_file.write_text("this is not audio")
    assert is_stereo(text_file) is False


def test_is_stereo_handles_exception(tmp_path: Path) -> None:
    """Exception raised by get_channel_count is caught and returns False."""
    dummy_path = tmp_path / "audio.wav"
    dummy_path.touch()
    with patch("tapeback.pipeline.get_channel_count", side_effect=RuntimeError("Disk read error")):
        assert is_stereo(dummy_path) is False
