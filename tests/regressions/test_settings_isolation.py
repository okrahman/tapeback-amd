"""Regression tests for test-suite isolation from the developer's configuration."""

import os

from tapeback.settings import Settings, get_settings


def test_settings_ignore_user_env_file():
    """Settings must not read the developer's ~/.config/tapeback/.env during tests.

    Bug: Settings declares env_file=(~/.config/tapeback/.env, .env), so every test
    constructing Settings() inherited the ambient machine configuration. A developer
    with TAPEBACK_CHUNK_LENGTH=2 in their user config got 2 where CI got the default
    7, which silently made assertions on default values machine-dependent.
    The autouse isolate_settings_sources fixture cuts both sources off.
    """
    assert Settings.model_config["env_file"] == ()
    assert Settings().chunk_length == 7
    assert get_settings().whisper_model == "large-v3-turbo"


def test_tapeback_env_vars_are_cleared():
    """Ambient TAPEBACK_* variables must not leak into tests either."""
    assert [key for key in os.environ if key.startswith("TAPEBACK_")] == []


def test_env_var_set_inside_a_test_still_applies(monkeypatch):
    """Isolation must not break a test that deliberately sets a variable."""
    monkeypatch.setenv("TAPEBACK_BEAM_SIZE", "1")
    assert Settings().beam_size == 1
