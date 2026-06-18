"""Transcriber tests — Whisper integration and segment processing."""

import locale
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from tapeback.transcriber import Transcriber, _resolve_compute_type


@pytest.mark.parametrize(
    "requested,device,expected",
    [
        # Explicit values pass through unchanged
        pytest.param("float16", "cuda", "float16", id="explicit_float16"),
        pytest.param("int8", "cuda", "int8", id="explicit_int8"),
        pytest.param("float32", "cpu", "float32", id="explicit_float32"),
        # Auto: device-driven only — no VRAM probing
        pytest.param("auto", "cpu", "int8", id="auto_cpu"),
        pytest.param("auto", "cuda", "float16", id="auto_cuda"),
    ],
)
def test_resolve_compute_type(requested, device, expected):
    """Pure compute-type resolution: explicit passthrough, auto + device branching."""
    assert _resolve_compute_type(requested, device) == expected


def test_lc_messages_set_for_pyav_locale_workaround():
    """Transcriber module must set LC_MESSAGES=C to prevent PyAV crash on non-ASCII locales.

    PyAV's Cython code uses c_string_encoding=ascii. On non-English locales,
    FFmpeg's av_strerror() may return non-ASCII text via strerror_r(), causing
    UnicodeDecodeError in PyAV's err_check().
    Both env var and C locale must be set — env var alone is not enough.
    """
    assert os.environ.get("LC_MESSAGES") == "C"
    lc = locale.getlocale(locale.LC_MESSAGES)
    # locale.getlocale returns (None, None) for "C" locale
    assert lc == (None, None) or lc[0] == "C"


def test_transcribe_stereo_pipeline(settings):
    """transcribe_stereo should transcribe both channels, assign 'You' to mic,
    and correctly map words from faster-whisper output to Segment dataclasses."""
    mock_word = MagicMock()
    mock_word.start = 0.0
    mock_word.end = 0.5
    mock_word.word = "Hello"
    mock_word.probability = 0.95

    mock_seg_mic = MagicMock()
    mock_seg_mic.start = 0.0
    mock_seg_mic.end = 3.0
    mock_seg_mic.text = " My speech "
    mock_seg_mic.words = [mock_word]

    mock_seg_monitor = MagicMock()
    mock_seg_monitor.start = 1.0
    mock_seg_monitor.end = 4.0
    mock_seg_monitor.text = " Their speech "
    mock_seg_monitor.words = []

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 5.0

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            (iter([mock_seg_mic]), mock_info),
            (iter([mock_seg_monitor]), mock_info),
        ]

        transcriber = Transcriber(settings)
        mic_segs, monitor_segs, info = transcriber.transcribe_stereo(
            Path("/fake/mic.wav"), Path("/fake/monitor.wav")
        )

    # Whisper called twice (mic + monitor)
    assert instance.transcribe.call_count == 2

    # Mic segments: speaker="You", words mapped to Word dataclass
    assert len(mic_segs) == 1
    assert mic_segs[0].speaker == "You"
    assert mic_segs[0].text == "My speech"
    assert mic_segs[0].words is not None
    assert mic_segs[0].words[0].word == "Hello"
    assert mic_segs[0].words[0].probability == 0.95

    # Monitor segments: speaker=None
    assert len(monitor_segs) == 1
    assert monitor_segs[0].speaker is None
    assert monitor_segs[0].text == "Their speech"

    # Info dict
    assert info["language"] == "en"
    assert info["duration"] == 5.0


def test_load_model_prefers_local_cache(settings):
    """Model load tries the local cache first — no HuggingFace round-trip per start."""
    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        Transcriber(settings)

    assert mock_model_cls.call_count == 1
    assert mock_model_cls.call_args_list[0].kwargs["local_files_only"] is True


def test_load_model_downloads_when_not_cached(settings):
    """If the model isn't cached, fall back to a network download (local_files_only=False)."""
    instance = MagicMock()
    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        mock_model_cls.side_effect = [LocalEntryNotFoundError("not cached"), instance]
        transcriber = Transcriber(settings)

    assert mock_model_cls.call_count == 2
    assert mock_model_cls.call_args_list[0].kwargs["local_files_only"] is True
    assert mock_model_cls.call_args_list[1].kwargs["local_files_only"] is False
    assert transcriber._model is instance


def test_transcribe_stereo_reports_per_channel_timings(settings):
    """transcribe_stereo emits separate timing lines for the mic and monitor channels."""
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 5.0

    mic_seg = MagicMock()
    mic_seg.start, mic_seg.end, mic_seg.text, mic_seg.words = 0.0, 3.0, "mine", []
    mon_seg = MagicMock()
    mon_seg.start, mon_seg.end, mon_seg.text, mon_seg.words = 0.0, 2.0, "theirs", []

    messages: list[str] = []
    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            (iter([mic_seg]), mock_info),
            (iter([mon_seg]), mock_info),
        ]
        transcriber = Transcriber(settings)
        transcriber.transcribe_stereo(
            Path("/fake/mic.wav"), Path("/fake/monitor.wav"), on_status=messages.append
        )

    assert any(m.startswith("Stage 'transcribe mic'") for m in messages)
    assert any(m.startswith("Stage 'transcribe monitor'") for m in messages)


def test_transcribe_passes_language_and_hallucination_settings(settings):
    """multilingual / language_detection_segments / hallucination threshold reach Whisper."""
    s = settings.model_copy(
        update={
            "device": "cpu",
            "multilingual": True,
            "language_detection_segments": 4,
            "hallucination_silence_threshold": 2.0,
        }
    )
    info = MagicMock()
    info.language, info.language_probability, info.duration = "en", 0.9, 1.0

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([]), info)

        Transcriber(s).transcribe(Path("/fake/audio.wav"))
        kwargs = instance.transcribe.call_args.kwargs

    assert kwargs["multilingual"] is True
    assert kwargs["language_detection_segments"] == 4
    assert kwargs["hallucination_silence_threshold"] == 2.0


def test_transcribe_passes_beam_size_and_temperature(settings):
    """beam_size and the temperature fallback ladder reach Whisper."""
    s = settings.model_copy(update={"device": "cpu", "beam_size": 3, "temperature": (0.0, 0.2)})
    info = MagicMock()
    info.language, info.language_probability, info.duration = "en", 0.9, 1.0

    with patch("tapeback.transcriber.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([]), info)

        Transcriber(s).transcribe(Path("/fake/audio.wav"))
        kwargs = instance.transcribe.call_args.kwargs

    assert kwargs["beam_size"] == 3
    assert kwargs["temperature"] == (0.0, 0.2)


def test_batched_inference_used_when_batch_size_positive(settings):
    """batch_size > 0 routes transcription through BatchedInferencePipeline."""
    s = settings.model_copy(update={"device": "cpu", "batch_size": 8})
    info = MagicMock()
    info.language, info.language_probability, info.duration = "en", 0.9, 1.0

    with (
        patch("tapeback.transcriber.WhisperModel") as mock_model_cls,
        patch("tapeback.transcriber.BatchedInferencePipeline") as mock_batched_cls,
    ):
        batched = mock_batched_cls.return_value
        batched.transcribe.return_value = (iter([]), info)

        Transcriber(s).transcribe(Path("/fake/audio.wav"))

    assert batched.transcribe.call_args.kwargs["batch_size"] == 8
    mock_model_cls.return_value.transcribe.assert_not_called()


def test_plain_inference_when_batch_size_zero(settings):
    """batch_size == 0 keeps the plain (non-batched) path."""
    s = settings.model_copy(update={"device": "cpu", "batch_size": 0})
    info = MagicMock()
    info.language, info.language_probability, info.duration = "en", 0.9, 1.0

    with (
        patch("tapeback.transcriber.WhisperModel") as mock_model_cls,
        patch("tapeback.transcriber.BatchedInferencePipeline") as mock_batched_cls,
    ):
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([]), info)

        Transcriber(s).transcribe(Path("/fake/audio.wav"))

    instance.transcribe.assert_called_once()
    mock_batched_cls.assert_not_called()
