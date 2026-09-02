"""Tests for lazy dependency loading functions."""

from unittest.mock import patch

from tapeback._lazy import load_transcriber


def test_load_transcriber_instantiates_transcriber_with_settings(settings):
    """load_transcriber lazily imports Transcriber and instantiates it with settings."""
    with patch("tapeback.transcriber.Transcriber") as mock_transcriber_cls:
        result = load_transcriber(settings)

    mock_transcriber_cls.assert_called_once_with(settings)
    assert result is mock_transcriber_cls.return_value
