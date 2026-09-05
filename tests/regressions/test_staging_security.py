"""Regression tests for staging-directory security.

Bug: process_file staged audio in a predictable shared directory
(/tmp/tapeback/proc_<hash>) created with mkdir(mode=0o700, exist_ok=True).
`exist_ok=True` silently accepted a pre-existing directory with permissive
mode or foreign ownership, ffmpeg -y followed planted symlinks for the fixed
output filenames (mono_16k.wav etc.), and two processes handling the same
source identity shared filenames and deleted the directory under each other.

Reproduced: pre-created 0777 dir + mono_16k.wav symlink to a sentinel file
-> existing_stage_mode=777, result_is_symlink=True, victim_was_overwritten=True.
"""

import fcntl
import hashlib
import os
import stat as stat_module
from unittest.mock import MagicMock

import pytest

import tapeback.pipeline as pipeline_mod
from tapeback import const
from tapeback.models import Segment
from tapeback.pipeline import process_file
from tapeback.settings import Settings
from tests.fixtures import create_mono_wav


def predicted_staging_dir(audio_path):
    """Replicate the attacker's path prediction for the deterministic staging dir."""
    st = audio_path.stat()
    ident = f"{audio_path.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    staging_hash = hashlib.sha256(ident.encode()).hexdigest()[:16]
    return f"{const.TEMP_DIR}/proc_{staging_hash}"


def mock_transcriber(transcribe=None):
    transcriber = MagicMock()
    transcriber.describe.return_value = "mock backend"
    transcriber.transcribe.return_value = (
        [Segment(start=0.0, end=0.5, text="speech")],
        {"language": "en", "duration": 0.5},
    )
    if transcribe is not None:
        transcriber.transcribe.side_effect = transcribe
    return transcriber


@pytest.fixture
def process_env(tmp_path, monkeypatch):
    """Settings plus the vault mocks process_file needs for a real mono WAV run."""
    settings = Settings(vault_path=tmp_path / "vault")
    (tmp_path / "vault").mkdir()
    audio = tmp_path / "input.wav"
    create_mono_wav(audio, duration=0.5)
    return tmp_path, settings, audio


def test_staging_symlink_output_is_refused_and_victim_intact(process_env, monkeypatch):
    """ffmpeg must never write through a planted symlink in the staging dir."""
    tmp_path, settings, audio = process_env
    sentinel = tmp_path / "victim.txt"
    sentinel.write_text("do not touch")

    stage = predicted_staging_dir(audio)
    os.makedirs(stage, mode=0o777, exist_ok=True)
    os.chmod(stage, 0o777)  # noqa: S103 — deliberately permissive: reproduces the attack
    os.symlink(sentinel, os.path.join(stage, "mono_16k.wav"))

    monkeypatch.setattr(pipeline_mod, "load_transcriber", lambda s: mock_transcriber())

    with pytest.raises(RuntimeError, match="symlink"):
        process_file(audio, settings, name="symlink_victim", diarize=False, do_summarize=False)

    assert sentinel.read_text() == "do not touch"


def test_precreated_permissive_staging_dir_is_repaired_to_private(process_env, monkeypatch):
    """A pre-created 0777 staging dir must be repaired to 0700, not accepted."""
    _tmp_path, settings, audio = process_env
    stage = predicted_staging_dir(audio)
    os.makedirs(stage, mode=0o777, exist_ok=True)
    os.chmod(stage, 0o777)  # noqa: S103 — deliberately permissive: reproduces the attack

    observed_modes = []

    def fake_transcribe(audio_path, **kwargs):
        observed_modes.append(stat_module.S_IMODE(audio_path.parent.stat().st_mode))
        return [Segment(start=0.0, end=0.5, text="speech")], {"language": "en", "duration": 0.5}

    monkeypatch.setattr(
        pipeline_mod, "load_transcriber", lambda s: mock_transcriber(fake_transcribe)
    )

    process_file(audio, settings, name="mode_repair", diarize=False, do_summarize=False)

    assert observed_modes == [0o700]


def test_staging_dir_removed_even_when_processing_fails(process_env, monkeypatch):
    """An exception mid-processing must still remove the staging directory."""
    _tmp_path, settings, audio = process_env
    stage = predicted_staging_dir(audio)

    def failing_transcribe(audio_path, **kwargs):
        raise RuntimeError("transcription exploded")

    monkeypatch.setattr(
        pipeline_mod, "load_transcriber", lambda s: mock_transcriber(failing_transcribe)
    )

    with pytest.raises(RuntimeError, match="transcription exploded"):
        process_file(audio, settings, name="failure_cleanup", diarize=False, do_summarize=False)

    assert not os.path.exists(stage)


def test_concurrent_job_on_same_source_is_locked_out(process_env, monkeypatch):
    """A second process on the same source identity must be refused, not interleaved."""
    _tmp_path, settings, audio = process_env
    stage = predicted_staging_dir(audio)
    os.makedirs(stage, mode=0o700, exist_ok=True)

    lock_fd = os.open(os.path.join(stage, "proc.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(pipeline_mod, "load_transcriber", lambda s: mock_transcriber())
        with pytest.raises(RuntimeError, match="already working"):
            process_file(audio, settings, name="locked", diarize=False, do_summarize=False)
    finally:
        os.close(lock_fd)
