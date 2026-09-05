"""Regression tests pinning the intended default transcription backend.

The faster-whisper backend is the *intended* default: transcription is fully
local, and raw audio only leaves the machine when the user explicitly opts in
with `TAPEBACK_TRANSCRIPTION_BACKEND=lemonade`. These tests pin that default so
it cannot silently flip in either direction, pin that Lemonade remains reachable
via explicit opt-in, and pin the runtime disclosure that makes the outbound-audio
path visible whenever it is taken.
"""

from unittest.mock import MagicMock

from pydantic import SecretStr

import tapeback.live as live_mod
from tapeback._fw_backend import FasterWhisperBackend
from tapeback._lemonade import LemonadeBackend
from tapeback.live import LiveTranscriber
from tapeback.settings import Settings
from tapeback.transcriber import Transcriber


def test_default_backend_is_faster_whisper() -> None:
    """Unmodified Settings must select the local faster-whisper backend.

    The autouse `isolate_settings_sources` fixture strips `.env` and every
    `TAPEBACK_*` variable, so this exercises the true field default. A default
    that ships audio off the machine would send recordings to whatever process
    wins the configured port — local disclosure by default is not acceptable.
    """
    assert Settings().transcription_backend == "faster-whisper"


def test_default_transcriber_builds_faster_whisper_backend(tmp_path) -> None:
    """A Transcriber built from unmodified settings must dispatch to faster-whisper.

    Safe without a network: neither backend's constructor performs a model
    load, a preflight, or a request.
    """
    transcriber = Transcriber(Settings(vault_path=tmp_path / "vault"))
    assert isinstance(transcriber._backend, FasterWhisperBackend)


def test_explicit_lemonade_opt_in_still_builds_lemonade_backend(tmp_path) -> None:
    """Opting in must still select Lemonade — the default flip is not a removal."""
    settings = Settings(
        vault_path=tmp_path / "vault",
        transcription_backend="lemonade",
    )
    transcriber = Transcriber(settings)
    assert isinstance(transcriber._backend, LemonadeBackend)


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
