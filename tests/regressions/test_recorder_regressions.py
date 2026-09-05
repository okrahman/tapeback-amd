"""Regression tests for recorder bugs."""

import json
import os
import stat
from unittest.mock import MagicMock, patch

import pytest

import tapeback.recorder as recorder_mod
from tapeback.recorder import detect_devices
from tests.fixtures import create_session_file


def test_detect_devices_auto_legacy_keys(settings):
    """Auto-detect should also work with legacy pactl keys (default_sink/default_source).

    Bug: PulseAudio <17 uses 'default_sink' instead of 'default_sink_name'.
    """
    pactl_output = json.dumps(
        {
            "default_sink": "alsa_output.usb-stereo",
            "default_source": "alsa_input.usb-stereo",
        }
    )

    with (
        patch("tapeback.recorder.shutil.which", return_value="/usr/bin/pactl"),
        patch("tapeback.recorder._probe_source", return_value=False),
        patch("tapeback.recorder.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(stdout=pactl_output, returncode=0)

        monitor, mic = detect_devices(settings)

    assert monitor == "alsa_output.usb-stereo.monitor"
    assert mic == "alsa_input.usb-stereo"


def test_stop_without_start_raises(recorder):
    """stop() without active recording should raise RuntimeError.

    Bug: stop() crashed with unclear error instead of clean message.
    """
    with pytest.raises(RuntimeError, match="No recording in progress"):
        recorder.stop()


def test_start_while_recording_raises(recorder, settings, session_file):
    """start() while already recording should raise RuntimeError.

    Bug: starting a second recording corrupted the session file.
    """
    create_session_file(
        session_file,
        pid_monitor=os.getpid(),
        pid_mic=os.getpid(),
        session_name="existing",
    )

    with pytest.raises(RuntimeError, match="already in progress"):
        recorder.start(settings)


def test_parecord_not_found(recorder, settings):
    """Should give clear error when parecord is not installed.

    Bug: cryptic subprocess error instead of user-friendly message.
    """
    with (
        patch("tapeback.recorder.shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="parecord not found"),
    ):
        recorder.start(settings)


def _patch_parecord(monkeypatch, tmp_path):
    """Stub device detection and parecord so start() never spawns anything."""
    monkeypatch.setattr(recorder_mod.shutil, "which", lambda name: "/usr/bin/parecord")
    monkeypatch.setattr(recorder_mod, "detect_devices", lambda settings: ("mon", "mic"))
    proc = MagicMock()
    proc.pid = 4242
    monkeypatch.setattr(recorder_mod.subprocess, "Popen", lambda *a, **kw: proc)


def test_start_repairs_permissive_session_dir(recorder, settings, tmp_path, monkeypatch):
    """A pre-created 0777 session directory must be repaired to 0700, not accepted.

    Bug: mkdir(mode=0o700, exist_ok=True) silently accepted a pre-existing
    permissive directory, exposing the recording WAVs.
    """
    monkeypatch.setattr(recorder_mod.const, "TEMP_DIR", str(tmp_path / "tapeback"))
    _patch_parecord(monkeypatch, tmp_path)
    session_dir = tmp_path / "tapeback" / "repair_session"
    session_dir.mkdir(parents=True, mode=0o777)
    os.chmod(session_dir, 0o777)  # noqa: S103 — deliberately permissive: reproduces the attack

    name = recorder.start(settings, session_name="repair_session")

    assert name == "repair_session"
    assert stat.S_IMODE(session_dir.stat().st_mode) == 0o700


def test_start_refuses_symlinked_recording_path(recorder, settings, tmp_path, monkeypatch):
    """parecord must not write through a planted symlink at the fixed WAV path."""
    monkeypatch.setattr(recorder_mod.const, "TEMP_DIR", str(tmp_path / "tapeback"))
    _patch_parecord(monkeypatch, tmp_path)
    sentinel = tmp_path / "victim.txt"
    sentinel.write_text("do not touch")
    session_dir = tmp_path / "tapeback" / "symlink_session"
    session_dir.mkdir(parents=True)
    os.symlink(sentinel, session_dir / "mic.wav")

    with pytest.raises(RuntimeError, match="symlink"):
        recorder.start(settings, session_name="symlink_session")

    assert sentinel.read_text() == "do not touch"


def test_start_replaces_planted_fifo_at_recording_path(recorder, settings, tmp_path, monkeypatch):
    """A planted FIFO must never become the microphone's write target.

    Bug: refuse_symlink_target accepted every non-symlink inode. In the
    permissive-directory scenario the 0700 repair targets, an attacker who
    planted mic.wav as a FIFO and opened it before the chmod kept a live
    descriptor that parecord would stream microphone audio into.
    """
    monkeypatch.setattr(recorder_mod.const, "TEMP_DIR", str(tmp_path / "tapeback"))
    _patch_parecord(monkeypatch, tmp_path)
    session_dir = tmp_path / "tapeback" / "fifo_session"
    session_dir.mkdir(parents=True, mode=0o777)
    os.chmod(session_dir, 0o777)  # noqa: S103 — deliberately permissive: reproduces the attack
    os.mkfifo(session_dir / "mic.wav")

    name = recorder.start(settings, session_name="fifo_session")

    assert name == "fifo_session"
    mic_path = session_dir / "mic.wav"
    mic_stat = os.lstat(mic_path)
    assert stat.S_ISREG(mic_stat.st_mode)
    assert stat.S_IMODE(mic_stat.st_mode) == 0o600


def test_start_replaces_attacker_readable_recording_file(recorder, settings, tmp_path, monkeypatch):
    """A planted regular file must be replaced, not written into.

    An attacker who created a permissive regular file at the predictable path
    before the directory was secured holds an open descriptor on that inode;
    recording data must go to a fresh inode the attacker cannot read.
    """
    monkeypatch.setattr(recorder_mod.const, "TEMP_DIR", str(tmp_path / "tapeback"))
    _patch_parecord(monkeypatch, tmp_path)
    session_dir = tmp_path / "tapeback" / "stale_session"
    session_dir.mkdir(parents=True, mode=0o777)
    os.chmod(session_dir, 0o777)  # noqa: S103 — deliberately permissive: reproduces the attack
    mic_path = session_dir / "mic.wav"
    mic_path.write_text("attacker-controlled")
    os.chmod(mic_path, 0o666)  # noqa: S103
    old_inode = os.lstat(mic_path).st_ino

    recorder.start(settings, session_name="stale_session")

    new_stat = os.lstat(mic_path)
    assert stat.S_ISREG(new_stat.st_mode)
    assert new_stat.st_ino != old_inode
    assert stat.S_IMODE(new_stat.st_mode) == 0o600
    assert mic_path.read_bytes() == b""
