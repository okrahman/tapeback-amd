"""Opt-in smoke test against a running Lemonade Server.

Run with:

    TAPEBACK_LEMONADE_SMOKE=1 pytest tests/test_lemonade_integration.py

Optionally point TAPEBACK_LEMONADE_URL at the server (default 127.0.0.1:13305) and
TAPEBACK_LEMONADE_API_KEY if it requires auth. Deliberately asserts nothing about
the server's OS, accelerator, or internal inference backend — those are Lemonade's
business, not tapeback's.
"""

import os
import wave

import pytest
from pydantic import SecretStr

from tapeback._lemonade import LemonadeBackend, LemonadeError
from tapeback.settings import Settings

_SMOKE_URL = os.environ.get("TAPEBACK_LEMONADE_URL", "http://127.0.0.1:13305")
_SMOKE_API_KEY = os.environ.get("TAPEBACK_LEMONADE_API_KEY", "")

pytestmark = pytest.mark.skipif(
    os.environ.get("TAPEBACK_LEMONADE_SMOKE") != "1",
    reason="opt-in: set TAPEBACK_LEMONADE_SMOKE=1 against a running Lemonade Server",
)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        transcription_backend="lemonade",
        resume_cache_dir=tmp_path / "resume",
        lemonade_url=_SMOKE_URL,
        lemonade_api_key=SecretStr(_SMOKE_API_KEY),
    )


def test_smoke_transcribes_a_real_wav(settings, tmp_path):
    wav = tmp_path / "smoke.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        # Two seconds of a quiet tone: enough for a valid request, no transcription
        # quality claim — the smoke test only proves the transport works end to end.
        wf.writeframes(b"\x00\x00" * 32000)

    backend = LemonadeBackend(settings)
    try:
        segments, info = backend.transcribe(wav)
    except LemonadeError as exc:
        pytest.fail(f"Lemonade smoke request failed: {exc}")

    # The contract the facade depends on, and nothing about the server's internals.
    assert info["partial"] is False
    assert "language" in info
    assert isinstance(segments, list)
