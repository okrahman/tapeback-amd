"""Tests for lazy dependency loading functions."""

import subprocess
import sys
from unittest.mock import patch

from tapeback._lazy import load_transcriber


def test_import_lazy_does_not_import_heavy_dependencies():
    """Importing tapeback._lazy must not import tapeback.transcriber or faster_whisper."""
    code = (
        "import sys; "
        "import tapeback._lazy; "
        "assert 'tapeback.transcriber' not in sys.modules; "
        "assert 'faster_whisper' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True)
    assert result.returncode == 0, f"Lazy import leaked heavy dependencies: {result.stderr.decode()}"


def test_load_transcriber_instantiates_transcriber_with_settings(settings):
    """load_transcriber lazily imports Transcriber and instantiates it with settings."""
    with patch("tapeback.transcriber.Transcriber") as mock_transcriber_cls:
        result = load_transcriber(settings)

    mock_transcriber_cls.assert_called_once_with(settings)
    assert result is mock_transcriber_cls.return_value
