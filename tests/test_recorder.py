import json
import re
import signal
from unittest.mock import MagicMock, patch

import pytest

from tapeback.recorder import detect_devices, validate_session_name
from tapeback.settings import Settings
from tests.fixtures import create_session_file


@pytest.mark.parametrize(
    "valid_name",
    [
        "session123",
        "2026-04-02_12-00-00",
        "my_session-name",
        "ABC_def-123",
    ],
)
def test_validate_session_name_valid(valid_name):
    """Valid session names should pass without raising exceptions."""
    validate_session_name(valid_name)


@pytest.mark.parametrize(
    "invalid_name",
    [
        "../session",
        "session/1",
        "session name",
        "session!",
        "session@name",
        "",
    ],
)
def test_validate_session_name_invalid(invalid_name):
    """Invalid session names should raise ValueError with exact error message."""
    expected_msg = (
        f"Invalid session name: {invalid_name!r}. "
        "Only alphanumerics, dashes, and underscores are allowed."
    )
    with pytest.raises(ValueError, match=re.escape(expected_msg)):
        validate_session_name(invalid_name)


def test_detect_devices_auto_dynamic(settings):
    """Auto-detect should use @DEFAULT_MONITOR@/@DEFAULT_SOURCE@ when supported."""
    with (
        patch("tapeback.recorder.shutil.which", return_value="/usr/bin/pactl"),
        patch("tapeback.recorder._probe_source", return_value=True),
    ):
        monitor, mic = detect_devices(settings)

    assert monitor == "@DEFAULT_MONITOR@"
    assert mic == "@DEFAULT_SOURCE@"


def test_detect_devices_auto_fallback(settings):
    """Auto-detect should fall back to pactl info when dynamic refs not supported."""
    pactl_output = json.dumps(
        {
            "default_sink_name": "alsa_output.pci-0000_00_1f.3.analog-stereo",
            "default_source_name": "alsa_input.pci-0000_00_1f.3.analog-stereo",
        }
    )

    with (
        patch("tapeback.recorder.shutil.which", return_value="/usr/bin/pactl"),
        patch("tapeback.recorder._probe_source", return_value=False),
        patch("tapeback.recorder.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(stdout=pactl_output, returncode=0)

        monitor, mic = detect_devices(settings)

    assert monitor == "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"
    assert mic == "alsa_input.pci-0000_00_1f.3.analog-stereo"


def test_detect_devices_explicit(tmp_vault):
    """Explicit device names should be returned without calling pactl."""
    s = Settings(
        vault_path=tmp_vault,
        monitor_source="my_monitor",
        mic_source="my_mic",
    )

    with patch("tapeback.recorder.subprocess.run") as mock_run:
        monitor, mic = detect_devices(s)
        mock_run.assert_not_called()

    assert monitor == "my_monitor"
    assert mic == "my_mic"


def test_start_creates_session_file(recorder, settings, session_file):
    """start() should create session.json with PIDs and paths."""
    with (
        patch("tapeback.recorder.detect_devices", return_value=("monitor_dev", "mic_dev")),
        patch("tapeback.recorder.shutil.which", return_value="/usr/bin/parecord"),
        patch("tapeback.recorder.subprocess.Popen") as mock_popen,
    ):
        proc_mock = MagicMock()
        proc_mock.pid = 12345
        mock_popen.return_value = proc_mock

        session_name = recorder.start(settings, session_name="test_session")

    assert session_name == "test_session"
    assert session_file.exists()

    data = json.loads(session_file.read_text())
    assert data["session_name"] == "test_session"
    assert data["pid_monitor"] == 12345
    assert data["pid_mic"] == 12345
    assert "started_at" in data
    assert "monitor_path" in data
    assert "mic_path" in data


def test_stop_sends_sigterm(recorder, session_file):
    """stop() should send SIGTERM to both processes."""
    create_session_file(session_file)

    killed = []

    def mock_kill(pid, sig):
        killed.append((pid, sig))
        if sig == 0:
            raise ProcessLookupError
        if sig == signal.SIGTERM:
            return  # SIGTERM accepted

    with patch("tapeback.recorder.os.kill", side_effect=mock_kill):
        _monitor_path, _mic_path = recorder.stop()

    # SIGTERM sent to both
    assert (99998, signal.SIGTERM) in killed
    assert (99999, signal.SIGTERM) in killed
    assert not session_file.exists()
