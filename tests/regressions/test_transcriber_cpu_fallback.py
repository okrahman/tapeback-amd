"""Regression: CPU fallback in Transcriber must be limited to real CUDA/OOM errors.

The old code caught any RuntimeError on cuda and silently fell back to CPU,
masking unrelated bugs (e.g. a corrupt audio file) and making transcription
~10x slower. A non-CUDA error must propagate immediately; a genuine CUDA error
must still fall back and retry on CPU.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tapeback._fw_backend import FasterWhisperBackend
from tapeback.transcriber import Transcriber


def _whisper_info() -> MagicMock:
    info = MagicMock()
    info.language = "en"
    info.language_probability = 0.99
    info.duration = 3.0
    return info


def test_non_cuda_runtime_error_propagates_without_cpu_fallback(settings):
    cuda_settings = settings.model_copy(update={"device": "cuda"})
    with patch("tapeback._fw_backend.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = RuntimeError("corrupt audio frame")

        transcriber = Transcriber(cuda_settings)
        with pytest.raises(RuntimeError, match="corrupt audio frame"):
            transcriber.transcribe(Path("/fake/audio.wav"))

        # No silent CPU retry — transcribe attempted exactly once, device unchanged.
        assert instance.transcribe.call_count == 1
        assert isinstance(transcriber._backend, FasterWhisperBackend)
        assert transcriber._backend._device == "cuda"


def test_cuda_oom_falls_back_to_cpu_and_retries(settings):
    cuda_settings = settings.model_copy(update={"device": "cuda"})
    good_seg = MagicMock()
    good_seg.start = 0.0
    good_seg.end = 1.0
    good_seg.text = " hi "
    good_seg.words = []

    with patch("tapeback._fw_backend.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            RuntimeError("CUDA failed with error out of memory"),
            (iter([good_seg]), _whisper_info()),
        ]

        transcriber = Transcriber(cuda_settings)
        segments, _info = transcriber.transcribe(Path("/fake/audio.wav"))

        assert instance.transcribe.call_count == 2  # retried after CPU fallback
        assert isinstance(transcriber._backend, FasterWhisperBackend)
        assert transcriber._backend._device == "cpu"
        assert segments[0].text == "hi"


def test_cuda_inference_error_message_is_surfaced(settings, capsys):
    """The real CUDA error must be shown, not hidden behind a generic message.

    Without it, a CUDA OOM and a cuDNN/driver failure look identical in the
    log, leaving the user unable to tell why transcription dropped to slow CPU.
    """
    cuda_settings = settings.model_copy(update={"device": "cuda"})
    good_seg = MagicMock()
    good_seg.start, good_seg.end, good_seg.text, good_seg.words = 0.0, 1.0, "ok", []

    with patch("tapeback._fw_backend.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.side_effect = [
            RuntimeError("CUDA failed with error out of memory"),
            (iter([good_seg]), _whisper_info()),
        ]
        transcriber = Transcriber(cuda_settings)
        transcriber.transcribe(Path("/fake/audio.wav"))

    captured = capsys.readouterr()
    assert "out of memory" in captured.err
