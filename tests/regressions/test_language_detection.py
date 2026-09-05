"""Regression tests for per-channel language detection."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from tapeback.transcriber import Transcriber


def _info(language: str, probability: float, duration: float = 60.0):
    info = MagicMock()
    info.language = language
    info.language_probability = probability
    info.duration = duration
    return info


def _segment(start: float, end: float, text: str):
    seg = MagicMock()
    seg.start, seg.end, seg.text, seg.words = start, end, text, []
    return seg


def test_monitor_language_is_reused_for_the_mic_channel(settings):
    """Both channels are one conversation, so they must be transcribed as one language.

    Bug: each channel ran its own auto-detection. The mic channel is gated to near
    silence while the user listens, so it had almost nothing to detect from and
    guessed wrong — producing notes with `language: en` in the front matter whose
    text was Russian, with stray Cyrillic left inside English sentences.
    """
    s = settings.model_copy(update={"device": "cpu", "language": "auto"})

    with patch("tapeback._fw_backend.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            # Monitor: plenty of speech, confident Russian.
            (iter([_segment(0.0, 30.0, "длинная русская реплика")]), _info("ru", 0.97)),
            # Mic: one gated fragment, would have been detected as English.
            (iter([_segment(0.0, 1.0, "ok")]), _info("en", 0.41)),
        ]

        transcriber = Transcriber(s)
        _mic, _monitor, info = transcriber.transcribe_stereo(
            Path("/fake/mic.wav"), Path("/fake/monitor.wav")
        )

        calls = instance.transcribe.call_args_list

    # Monitor is transcribed first, with detection enabled.
    assert calls[0].kwargs["language"] is None
    # The mic pass is then pinned to what the monitor found, instead of guessing.
    assert calls[1].kwargs["language"] == "ru"
    assert info["language"] == "ru"


def test_explicit_language_is_still_honoured_for_both_channels(settings):
    """A configured language must not be overridden by detection."""
    s = settings.model_copy(update={"device": "cpu", "language": "de"})

    with patch("tapeback._fw_backend.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            (iter([_segment(0.0, 5.0, "text")]), _info("de", 0.9)),
            (iter([_segment(0.0, 5.0, "text")]), _info("de", 0.9)),
        ]

        Transcriber(s).transcribe_stereo(Path("/fake/mic.wav"), Path("/fake/monitor.wav"))
        calls = instance.transcribe.call_args_list

    assert calls[0].kwargs["language"] == "de"
    assert calls[1].kwargs["language"] == "de"


def test_mic_still_transcribed_when_monitor_detects_nothing(settings):
    """An empty monitor channel must not pin the mic to a bogus language."""
    s = settings.model_copy(update={"device": "cpu", "language": "auto"})

    with patch("tapeback._fw_backend.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            (iter([]), _info("", 0.0)),
            (iter([_segment(0.0, 5.0, "речь")]), _info("ru", 0.9)),
        ]

        transcriber = Transcriber(s)
        mic, _monitor, _info_dict = transcriber.transcribe_stereo(
            Path("/fake/mic.wav"), Path("/fake/monitor.wav")
        )
        calls = instance.transcribe.call_args_list

    # Nothing detected on the monitor -> the mic falls back to detecting for itself.
    assert calls[1].kwargs["language"] is None
    assert len(mic) == 1
