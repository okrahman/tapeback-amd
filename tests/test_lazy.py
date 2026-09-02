"""Tests for lazy dependency loaders in _lazy.py."""

from unittest.mock import patch

from tapeback._lazy import load_transcriber


def test_load_transcriber(settings):
    with patch("tapeback.transcriber.Transcriber") as mock_transcriber_cls:
        result = load_transcriber(settings)
        mock_transcriber_cls.assert_called_once_with(settings)
        assert result == mock_transcriber_cls.return_value
