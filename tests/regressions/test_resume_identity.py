"""Regression tests for resume-cache identity under late device resolution.

The facade used to freeze the resume key from `cache_fingerprint()` BEFORE
`backend.transcribe()` ran, then store the result under that frozen key. A
device/compute fallback that landed during the call — the isolated child
resolving CPU, or the in-process CUDA ladder falling back — therefore cached a
CPU/int8 result under a CUDA identity, and a later genuinely-CUDA run silently
served the stale CPU-quality transcript. The fix recomputes the key at commit
time. These tests pin that behavior.
"""

from unittest.mock import MagicMock, patch

import pytest

from tapeback import _resume
from tapeback.models import Segment
from tapeback.transcriber import Transcriber


@pytest.fixture
def audio_file(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFFnot-really-audio")
    return path


def _identity_settings(settings, tmp_path, *, isolated: bool):
    return settings.model_copy(
        update={
            "device": "cuda",
            "isolate_transcription": isolated,
            "resume_cache": True,
            "resume_cache_dir": tmp_path / "resume",
        }
    )


def _configure_device_probe(monkeypatch):
    monkeypatch.setattr("tapeback._fw_backend.get_free_vram_mib", lambda: 4096)
    monkeypatch.setattr("tapeback._fw_backend.wait_for_clamp_release", lambda *_a, **_k: True)
    monkeypatch.setattr("tapeback._fw_backend.free_gpu_memory", lambda: None)


def _segments(text: str) -> list[Segment]:
    return [Segment(start=0.0, end=1.0, text=text)]


def test_isolated_cpu_result_is_not_cached_under_cuda_identity(
    settings, tmp_path, monkeypatch, audio_file
):
    """An isolated child that resolved CPU must not satisfy a genuinely-CUDA run."""
    _configure_device_probe(monkeypatch)
    s = _identity_settings(settings, tmp_path, isolated=True)

    def cpu_child(audio_path, settings, *, stage, on_status, language_override=None):
        return _segments("cpu run"), {
            "language": "en",
            "duration": 1.0,
            "partial": False,
            "device": "cpu",
            "compute_type": "int8",
        }

    with patch("tapeback._fw_backend.WhisperModel"):
        first = Transcriber(s)
        monkeypatch.setattr("tapeback._fw_backend.transcribe_isolated", cpu_child)
        first.transcribe(audio_file)

    cuda_calls: list[str] = []

    def cuda_child(audio_path, settings, *, stage, on_status, language_override=None):
        cuda_calls.append(stage)
        return _segments("cuda run"), {
            "language": "en",
            "duration": 1.0,
            "partial": False,
            "device": "cuda",
            "compute_type": "int8_float16",
        }

    with patch("tapeback._fw_backend.WhisperModel"):
        second = Transcriber(s)
        monkeypatch.setattr("tapeback._fw_backend.transcribe_isolated", cuda_child)
        second.transcribe(audio_file)


def test_inprocess_cpu_fallback_result_is_not_cached_under_cuda_identity(
    settings, tmp_path, monkeypatch, audio_file
):
    """Same invariant for the in-process CUDA OOM ladder."""
    _configure_device_probe(monkeypatch)
    s = _identity_settings(settings, tmp_path, isolated=False)

    def cpu_on_oom(*_args, **kwargs):
        model = MagicMock()
        if kwargs["device"] == "cuda":
            model.transcribe.side_effect = RuntimeError("CUDA failed with error out of memory")
        else:
            info = MagicMock()
            info.language, info.language_probability, info.duration = "en", 0.9, 1.0
            model.transcribe.return_value = (iter([]), info)
        return model

    with patch("tapeback._fw_backend.WhisperModel", side_effect=cpu_on_oom):
        first = Transcriber(s)
        first.transcribe(audio_file)
    assert first.describe() == "Whisper: large-v3-turbo on cpu/int8"

    cuda_calls: list[str] = []

    def cuda_works(*_args, **kwargs):
        model = MagicMock()
        if kwargs["device"] == "cuda":
            cuda_calls.append("cuda")
            info = MagicMock()
            info.language, info.language_probability, info.duration = "en", 0.9, 1.0
            model.transcribe.return_value = (iter([]), info)
        else:
            model.transcribe.return_value = (iter([]), MagicMock())
        return model

    with patch("tapeback._fw_backend.WhisperModel", side_effect=cuda_works):
        second = Transcriber(s)
        second.transcribe(audio_file)

    assert cuda_calls, "a genuinely-CUDA run must re-transcribe, not hit the CPU-cached entry"


class _MidRunIdentityBackend:
    """Fake backend whose resolved identity changes during transcribe(), like a
    device fallback does. The stereo commit loop must store under the identity
    that was current AFTER the work, not before it."""

    def __init__(self) -> None:
        self.fingerprint = "identity-before"

    def describe(self) -> str:
        return "mutating test backend"

    def cache_fingerprint(self) -> str:
        return self.fingerprint

    def pace(self, on_status) -> None:
        return None

    def transcribe(
        self, audio_path, *, stage="transcribe", on_status=lambda _m: None, language_override=None
    ):
        self.fingerprint = "identity-after"
        return _segments(f"{stage} result"), {"language": "en", "duration": 1.0, "partial": False}


def test_stereo_channels_are_cached_under_the_post_stage_identity(settings, tmp_path, audio_file):
    s = settings.model_copy(update={"resume_cache": True, "resume_cache_dir": tmp_path / "resume"})
    transcriber = Transcriber(s)
    backend = _MidRunIdentityBackend()
    transcriber._backend = backend  # the fake backend is the point of the test

    mic = tmp_path / "mic_16k.wav"
    mic.write_bytes(b"RIFFmic")
    monitor = tmp_path / "monitor_16k.wav"
    monitor.write_bytes(b"RIFFmonitor")

    transcriber.transcribe_stereo(mic, monitor)

    directory = _resume.resume_dir(s)
    stale_key = transcriber._resume_key(mic, "transcribe mic", "identity-before", "en")
    fresh_key = transcriber._resume_key(mic, "transcribe mic", "identity-after", "en")
    assert stale_key is not None and fresh_key is not None
    stale = directory / f"{stale_key.digest}.json"
    fresh = directory / f"{fresh_key.digest}.json"
    assert not stale.exists(), "a mid-run identity change must not store under the frozen key"
    assert fresh.exists(), "the committed channel must carry the post-stage identity"
