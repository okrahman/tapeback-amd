"""Regression tests pinning the intended default transcription backend.

The Lemonade backend is the *intended* default: raw audio is sent to the
user-configured Lemonade Server on every run unless the user opts back out
with `TAPEBACK_TRANSCRIPTION_BACKEND=faster-whisper`. These tests pin that
default so it cannot silently flip in either direction, and pin the live-mode
backend disclosure that makes the outbound-audio path visible at runtime.
"""

from unittest.mock import MagicMock

import tapeback.live as live_mod
from tapeback._lemonade import LemonadeBackend
from tapeback.live import LiveTranscriber
from tapeback.settings import Settings
from tapeback.transcriber import Transcriber


def test_default_backend_is_lemonade() -> None:
    """Unmodified Settings must select the Lemonade backend.

    The autouse `isolate_settings_sources` fixture strips `.env` and every
    `TAPEBACK_*` variable, so this exercises the true field default.
    """
    assert Settings().transcription_backend == "lemonade"


def test_default_transcriber_builds_lemonade_backend(tmp_path) -> None:
    """A Transcriber built from unmodified settings must dispatch to Lemonade.

    Safe without a network: LemonadeBackend's constructor performs no model
    load, no preflight, and no request.
    """
    transcriber = Transcriber(Settings(vault_path=tmp_path / "vault"))
    assert isinstance(transcriber._backend, LemonadeBackend)


def test_live_transcriber_discloses_backend_once(tmp_path, monkeypatch, capsys) -> None:
    """Live mode must announce the active backend when it first loads one.

    Post-recording runs print `transcriber.describe()` through the status
    callback (pipeline.py); live transcription never did, so a live user was
    never told that audio is being sent to the Lemonade endpoint.
    """
    settings = Settings(vault_path=tmp_path / "vault", live=True, live_overlap=0.0)
    lt = LiveTranscriber(settings, "disclosure", tmp_path / "mic.wav", tmp_path / "monitor.wav")
    transcriber = MagicMock()
    transcriber.describe.return_value = "Lemonade: Whisper-Large-v3-Turbo at http://127.0.0.1:13305"
    monkeypatch.setattr(live_mod, "load_transcriber", lambda _settings: transcriber)

    lt._ensure_transcriber()
    lt._ensure_transcriber()  # second call is a no-op and must not re-announce

    out = capsys.readouterr().err
    assert out.count("Lemonade: Whisper-Large-v3-Turbo at http://127.0.0.1:13305") == 1
