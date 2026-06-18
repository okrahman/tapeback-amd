"""Regression: merge_channels must not emit an unused mono mix.

The dual-channel pipeline transcribes mic and monitor separately via
split_channels_16k(); the mixed-down mono file that merge_channels used to
produce was never read — it only burned an extra ffmpeg loudnorm pass over
the full recording. merge_channels now returns the stereo path only.
"""

import shutil
import wave
from pathlib import Path

import pytest

from tapeback.audio import merge_channels
from tests.fixtures import create_silent_wav


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_merge_channels_returns_only_stereo_and_skips_mono_mix(tmp_path):
    monitor = tmp_path / "monitor.wav"
    mic = tmp_path / "mic.wav"
    output_dir = tmp_path / "output"
    create_silent_wav(monitor)
    create_silent_wav(mic)

    result = merge_channels(monitor, mic, output_dir)

    # Returns a single stereo Path, not a (stereo, mono) tuple.
    assert isinstance(result, Path)
    with wave.open(str(result), "rb") as wf:
        assert wf.getnchannels() == 2

    # The unused mixed-down mono file is no longer written.
    assert not (output_dir / "mono_16k.wav").exists()
