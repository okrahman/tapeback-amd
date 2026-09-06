"""Regression tests pinning the intended default transcription backend.

The Lemonade backend is the *intended* default: transcription runs through the
user-configured Lemonade Server, and raw recording audio leaves the machine by
design. The faster-whisper backend is the explicit local opt-out
(`TAPEBACK_TRANSCRIPTION_BACKEND=faster-whisper`). These tests pin that default
so it cannot silently flip in either direction, pin that faster-whisper remains
reachable via explicit opt-out, and pin the runtime disclosure that makes the
outbound-audio path visible whenever it is taken.
"""

from unittest.mock import MagicMock

from pydantic import SecretStr

import tapeback.live as live_mod
from tapeback._fw_backend import FasterWhisperBackend
from tapeback._lemonade import LemonadeBackend
from tapeback.live import LiveTranscriber
from tapeback.settings import Settings
from tapeback.transcriber import Transcriber


def test_default_backend_is_lemonade() -> None:
    """Unmodified Settings must select the Lemonade backend.

    The autouse `isolate_settings_sources` fixture strips `.env` and every
    `TAPEBACK_*` variable, so this exercises the true field default. Shipping
    audio to the user-configured Lemonade Server by default is intended —
    including on a local multi-user host — so the per-run disclosure line
    (pinned below) must keep the outbound path visible on every run.
    """
    assert Settings().transcription_backend == "lemonade"


def test_default_transcriber_builds_lemonade_backend(tmp_path) -> None:
    """A Transcriber built from unmodified settings must dispatch to Lemonade.

    Safe without a network: neither backend's constructor performs a model
    load, a preflight, or a request.
    """
    transcriber = Transcriber(Settings(vault_path=tmp_path / "vault"))
    assert isinstance(transcriber._backend, LemonadeBackend)


def test_explicit_faster_whisper_opt_out_still_builds_faster_whisper_backend(tmp_path) -> None:
    """Opting out must still select faster-whisper — the default flip is not a removal."""
    settings = Settings(
        vault_path=tmp_path / "vault",
        transcription_backend="faster-whisper",
    )
    transcriber = Transcriber(settings)
    assert isinstance(transcriber._backend, FasterWhisperBackend)


def test_lemonade_discloses_unauthenticated_plaintext_local_endpoint(tmp_path) -> None:
    """describe() must flag plaintext loopback endpoints with no bearer token.

    On a multi-user host, any unprivileged local process can bind the
    predictable default port before the real server starts; the disclosure line
    every run prints must say so rather than presenting the destination as safe.
    """
    backend = LemonadeBackend(
        Settings(vault_path=tmp_path / "vault", transcription_backend="lemonade")
    )
    assert "unauthenticated plaintext" in backend.describe()


def test_lemonade_does_not_flag_authenticated_or_remote_endpoints(tmp_path) -> None:
    """The warning is for plaintext loopback without a token — not for everything."""
    settings = Settings(
        vault_path=tmp_path / "vault",
        transcription_backend="lemonade",
        lemonade_api_key=SecretStr("tok"),
    )
    assert "unauthenticated plaintext" not in LemonadeBackend(settings).describe()

    remote = LemonadeBackend(
        Settings(
            vault_path=tmp_path / "vault",
            transcription_backend="lemonade",
            lemonade_url="https://lemonade.example:13305",
        )
    )
    assert "unauthenticated plaintext" not in remote.describe()


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
