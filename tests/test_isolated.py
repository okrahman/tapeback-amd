"""Tests for out-of-process transcription."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tapeback._fw_backend import FasterWhisperBackend
from tapeback._isolated import (
    WorkerFailed,
    _worker_env,
    job_settings,
    transcribe_isolated,
)
from tapeback._worker import EVENT_INFO, EVENT_SEGMENT, EVENT_STATUS, emit
from tapeback._worker import main as worker_main
from tapeback.models import Segment


class _FakeProcess:
    """Stands in for the worker: replays canned lines, records how it was stopped."""

    def __init__(self, lines: list[str], returncode: int = 0):
        self.stdin = io.StringIO()
        self.stdout = iter(lines)
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self._alive = True

    def poll(self):
        return None if self._alive else self.returncode

    def terminate(self):
        self.terminated = True
        self._alive = False

    def kill(self):
        self.killed = True
        self._alive = False

    def wait(self, timeout=None):
        self._alive = False
        return self.returncode


def _line(event: str, **payload) -> str:
    return json.dumps({"type": event, **payload}) + "\n"


def _segment_line(start: float, end: float, text: str) -> str:
    return _line(
        EVENT_SEGMENT,
        data={
            "start": start,
            "end": end,
            "text": text,
            "speaker": None,
            "words": [{"start": start, "end": end, "word": text, "probability": 0.9}],
        },
    )


@pytest.fixture
def spawn(monkeypatch):
    """Replace Popen; returns a setter for the lines the fake worker will emit."""
    holder: dict[str, _FakeProcess] = {}

    def _install(lines: list[str], returncode: int = 0) -> _FakeProcess:
        process = _FakeProcess(lines, returncode)
        holder["process"] = process
        monkeypatch.setattr("tapeback._isolated.subprocess.Popen", lambda *_a, **_k: process)
        return process

    return _install


def test_credentials_never_reach_the_worker(settings):
    """A process that only transcribes audio has no business holding tokens."""
    payload = job_settings(settings)
    assert "hf_token" not in payload
    assert "llm_api_key" not in payload
    assert json.dumps(payload)  # must be JSON-serialisable


def test_worker_settings_disable_further_isolation(settings):
    """Without this the worker spawns a worker, forever."""
    assert job_settings(settings)["isolate_transcription"] is False


def test_segments_and_info_come_back(settings, spawn):
    spawn(
        [
            _line(EVENT_STATUS, message="Whisper: large-v3-turbo on cuda/int8_float16"),
            _segment_line(0.0, 5.0, "первая"),
            _segment_line(5.0, 9.0, "вторая"),
            _line(EVENT_INFO, data={"language": "ru", "duration": 9.0, "partial": False}),
        ]
    )
    messages: list[str] = []

    segments, info = transcribe_isolated(
        Path("/fake/audio.wav"), settings, on_status=messages.append
    )

    assert [s.text for s in segments] == ["первая", "вторая"]
    assert segments[0].words is not None
    assert segments[0].words[0].probability == 0.9
    assert info["language"] == "ru"
    assert "cuda/int8_float16" in messages[0]


def test_worker_dying_midway_keeps_what_it_sent(settings, spawn):
    """The whole point: an out-of-memory kills the child, not the run's work.

    Each segment is a complete line, so everything decoded before the crash is
    already in the parent. Losing it would put us back where an in-process OOM left
    us — except the VRAM is at least reclaimed now.
    """
    process = spawn([_segment_line(0.0, 5.0, "успел")], returncode=137)
    messages: list[str] = []

    segments, info = transcribe_isolated(
        Path("/fake/audio.wav"), settings, on_status=messages.append
    )

    assert [s.text for s in segments] == ["успел"]
    assert info["partial"] is True
    assert any("keeping the 1 segments" in m for m in messages)
    assert process.terminated or process.returncode == 137


def test_worker_dying_with_nothing_raises(settings, spawn):
    spawn([], returncode=1)

    with pytest.raises(WorkerFailed):
        transcribe_isolated(Path("/fake/audio.wav"), settings, on_status=lambda _m: None)


def test_worker_error_event_is_reported(settings, spawn):
    spawn([_line("error", message="RuntimeError: CUDA out of memory")], returncode=1)

    with pytest.raises(WorkerFailed, match="CUDA out of memory"):
        transcribe_isolated(Path("/fake/audio.wav"), settings, on_status=lambda _m: None)


def test_noise_on_stdout_is_ignored(settings, spawn):
    """Dependencies print to stdout; that must not derail the protocol."""
    spawn(
        [
            "loading model...\n",
            _segment_line(0.0, 5.0, "речь"),
            _line(EVENT_INFO, data={"language": "ru", "duration": 5.0, "partial": False}),
        ]
    )

    segments, info = transcribe_isolated(
        Path("/fake/audio.wav"), settings, on_status=lambda _m: None
    )

    assert [s.text for s in segments] == ["речь"]
    assert info["duration"] == 5.0


def test_worker_is_always_stopped(settings, spawn):
    """Leaving a worker running would defeat the reason it exists."""
    process = spawn([_line(EVENT_INFO, data={"language": "ru", "duration": 1.0, "partial": False})])

    transcribe_isolated(Path("/fake/audio.wav"), settings, on_status=lambda _m: None)

    assert process.terminated


def test_emit_writes_one_flushed_line():
    stream = io.StringIO()
    emit(stream, EVENT_STATUS, message="hello")
    assert stream.getvalue() == '{"type": "status", "message": "hello"}\n'


def test_malformed_job_is_reported_as_an_error_event(monkeypatch, capsys):
    """A bad job must come back as a parsable event, not a traceback on stderr."""
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))

    assert worker_main() == 2

    event = json.loads(capsys.readouterr().out.strip())
    assert event["type"] == "error"
    assert "invalid job" in event["message"]


# --- resolved device identity across the process boundary ---


def test_worker_resolved_identity_rides_on_info(settings, spawn):
    """The child's resolved device/compute identity reaches the parent via info."""
    spawn(
        [
            _line("backend", data={"device": "cpu", "compute_type": "int8"}),
            _line(EVENT_INFO, data={"language": "ru", "duration": 9.0, "partial": False}),
        ]
    )

    _segments, info = transcribe_isolated(
        Path("/fake/audio.wav"), settings, on_status=lambda _m: None
    )

    assert info["device"] == "cpu"
    assert info["compute_type"] == "int8"


def test_worker_without_identity_leaves_info_untouched(settings, spawn):
    """An older worker that never reports identity must not invent one."""
    spawn([_line(EVENT_INFO, data={"language": "ru", "duration": 9.0, "partial": False})])

    _segments, info = transcribe_isolated(
        Path("/fake/audio.wav"), settings, on_status=lambda _m: None
    )

    assert "device" not in info
    assert "compute_type" not in info


def test_worker_run_job_emits_resolved_identity_event(monkeypatch, capsys, settings, tmp_path):
    """run_job reports the child's resolved device/compute so the parent can adopt it."""
    fake_transcriber = MagicMock()
    fake_transcriber.transcribe.return_value = (
        [],
        {"language": "en", "duration": 1.0, "partial": False},
    )
    fake_transcriber.resolved_identity.return_value = {"device": "cpu", "compute_type": "int8"}
    fake_transcriber.describe.return_value = "Whisper: tiny on cpu/int8"
    monkeypatch.setattr(
        "tapeback.transcriber.Transcriber", MagicMock(return_value=fake_transcriber)
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {
                    "settings": job_settings(settings),
                    "audio_path": str(audio),
                    "stage": "transcribe",
                }
            )
        ),
    )

    assert worker_main() == 0

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line]
    assert {
        "type": "backend",
        "data": {"device": "cpu", "compute_type": "int8"},
    } in events


def test_parent_adopts_child_identity_before_result_is_accepted(monkeypatch, settings):
    """The parent fingerprint must reflect the child's resolved device, not the request.

    The child resolves the device itself (thermal clamp, VRAM) and may fall back to
    CPU mid-run; recording its identity only at accept time is what keeps a CPU/int8
    result out of a CUDA cache slot.
    """
    backend = FasterWhisperBackend(settings.model_copy(update={"isolate_transcription": True}))
    before = backend.cache_fingerprint()

    def fake_isolated(*_args, **_kwargs):
        return [Segment(start=0.0, end=1.0, text="child work")], {
            "language": "en",
            "duration": 1.0,
            "partial": False,
            "device": "cpu",
            "compute_type": "int8",
        }

    monkeypatch.setattr("tapeback._fw_backend.transcribe_isolated", fake_isolated)
    _segments, info = backend.transcribe(Path("/fake/audio.wav"))

    # The identity keys are adopted, then stripped from the caller-facing result.
    assert info == {"language": "en", "duration": 1.0, "partial": False}
    assert backend.resolved_identity() == {"device": "cpu", "compute_type": "int8"}
    assert backend.cache_fingerprint() != before


def test_runtime_device_fallback_changes_the_cache_identity(settings):
    """What _fallback_to_cpu mutates must be part of cache_fingerprint's answer."""
    backend = FasterWhisperBackend(
        settings.model_copy(
            update={"isolate_transcription": True, "device": "cuda", "compute_type": "float16"}
        )
    )
    before = backend.cache_fingerprint()

    # Exactly the mutation _load_model/_fallback_to_cpu perform on a CUDA failure.
    backend._device, backend._compute_type = "cpu", "int8"

    assert backend.cache_fingerprint() != before


def test_worker_env_strips_tapeback_vars_and_provider_secrets(monkeypatch):
    """The worker gets its settings from the JSON job, never the ambient env."""
    monkeypatch.setenv("TAPEBACK_DEVICE", "cuda")
    monkeypatch.setenv("TAPEBACK_LEMONADE_API_KEY", "sk-tapeback")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    monkeypatch.setenv("HF_TOKEN", "hf-token")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    env = _worker_env()

    assert not any(key.startswith("TAPEBACK_") for key in env)
    assert "ANTHROPIC_API_KEY" not in env and "HF_TOKEN" not in env
    # Operational variables the worker legitimately needs are inherited.
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_worker_env_denies_every_production_provider_credential(monkeypatch):
    """Every env var the production provider mapping reads is denied to the worker.

    Regression: the deny-list was a hand-copied set that contained QWEN_API_KEY —
    a name the production mapping never uses — while the qwen provider actually
    reads DASHSCOPE_API_KEY. A user running the qwen summarizer had their
    DashScope key present in the spawned transcription child for the whole
    session. The list is now derived from the production mapping; this test
    plants every mapped credential by name (hardcoded, per the test rules) plus
    the non-summarizer credentials, and asserts none survive the scrub.
    """
    planted = {
        "ANTHROPIC_API_KEY": "sk-anthropic",
        "OPENAI_API_KEY": "sk-openai",
        "GROQ_API_KEY": "gsk-groq",
        "GEMINI_API_KEY": "sk-gemini",
        "OPENROUTER_API_KEY": "sk-openrouter",
        "DEEPSEEK_API_KEY": "sk-deepseek",
        "DASHSCOPE_API_KEY": "sk-dashscope",
        "HF_TOKEN": "hf-token",
        "HUGGING_FACE_HUB_TOKEN": "hf-hub-token",
        "AWS_ACCESS_KEY_ID": "AKIA-example",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "AZURE_OPENAI_API_KEY": "sk-azure",
    }
    for key, value in planted.items():
        monkeypatch.setenv(key, value)

    env = _worker_env()

    for key in planted:
        assert key not in env
