"""Tests for speaker_merge module — spectral-profile clustering and speaker merging."""

from unittest.mock import patch

import numpy as np
import pytest

from tapeback.models import DiarizationSegment
from tapeback.speaker_merge import (
    _apply_merge,
    _pick_merge_threshold,
    _speaker_spectral_profile,
    merge_similar_speakers,
)
from tests.fixtures import voice_signal


def test_speaker_spectral_profile_normal():
    """_speaker_spectral_profile calculates non-zero spectrum for valid audio and segments."""
    sr = 16000
    audio = voice_signal(2.0, sr, fundamental=200.0)
    segments = [DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0)]

    profile = _speaker_spectral_profile(audio, sr, segments, "SPEAKER_00")
    assert isinstance(profile, np.ndarray)
    assert profile.ndim == 1
    assert len(profile) > 0
    assert float(np.sum(profile)) > 0.0


def test_speaker_spectral_profile_empty_spectra():
    """_speaker_spectral_profile returns zeros when no frames match speaker."""
    sr = 16000
    audio = voice_signal(2.0, sr, fundamental=200.0)
    # Speaker does not match any segment
    segments = [DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0)]

    profile = _speaker_spectral_profile(audio, sr, segments, "SPEAKER_01")
    assert isinstance(profile, np.ndarray)
    assert np.all(profile == 0)


def test_speaker_spectral_profile_invalid_freq_range():
    """_speaker_spectral_profile returns np.zeros(1) when min_bin >= max_bin."""
    sr = 16000
    audio = voice_signal(1.0, sr, fundamental=200.0)
    segments = [DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0)]

    # Patch constants so SPECTRAL_MIN_FREQ_HZ > SPECTRAL_MAX_FREQ_HZ (n_bins <= 0)
    with (
        patch("tapeback.const.SPECTRAL_MIN_FREQ_HZ", 5000),
        patch("tapeback.const.SPECTRAL_MAX_FREQ_HZ", 1000),
    ):
        profile = _speaker_spectral_profile(audio, sr, segments, "SPEAKER_00")

    assert isinstance(profile, np.ndarray)
    assert profile.shape == (1,)
    assert profile[0] == 0.0


@pytest.mark.parametrize(
    "sp_a_sec,sp_b_sec,default_thresh,expected",
    [
        # Minor speaker < 15.0s and ratio < 0.2 -> 0.92
        pytest.param(10.0, 100.0, 0.95, 0.92, id="minor_speaker_absorbed"),
        pytest.param(14.9, 100.0, 0.95, 0.92, id="minor_speaker_just_below_15s"),
        pytest.param(10.0, 51.0, 0.95, 0.92, id="ratio_just_below_0.2"),
        # Boundary cases returning default_threshold (0.95)
        pytest.param(15.0, 100.0, 0.95, 0.95, id="minor_speaker_boundary_15s"),
        pytest.param(15.1, 100.0, 0.95, 0.95, id="minor_speaker_above_15s"),
        pytest.param(10.0, 50.0, 0.95, 0.95, id="ratio_boundary_0.2"),
        pytest.param(10.0, 49.0, 0.95, 0.95, id="ratio_above_0.2"),
        pytest.param(0.0, 0.0, 0.95, 0.95, id="zero_speech_durations"),
    ],
)
def test_pick_merge_threshold(sp_a_sec, sp_b_sec, default_thresh, expected):
    """_pick_merge_threshold picks 0.92 for minor artifact speakers, else default."""
    total_speech = {"SPEAKER_00": sp_a_sec, "SPEAKER_01": sp_b_sec}
    threshold = _pick_merge_threshold("SPEAKER_00", "SPEAKER_01", total_speech, default_thresh)
    assert threshold == expected


def test_apply_merge():
    """_apply_merge updates merge_map canonical speaker for direct and transitive mappings."""
    speakers = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
    merge_map = {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_01", "SPEAKER_02": "SPEAKER_01"}

    _apply_merge(merge_map, "SPEAKER_00", "SPEAKER_01", speakers)

    assert merge_map == {
        "SPEAKER_00": "SPEAKER_00",
        "SPEAKER_01": "SPEAKER_00",
        "SPEAKER_02": "SPEAKER_00",
    }


@pytest.mark.parametrize(
    "threshold",
    [0.0, -0.5],
    ids=["zero_threshold", "negative_threshold"],
)
def test_merge_similar_speakers_threshold_disabled(threshold):
    """merge_similar_speakers returns input unchanged when similarity_threshold <= 0."""
    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
        DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=5.0),
    ]
    audio = voice_signal(5.0, 16000, fundamental=200.0)
    result = merge_similar_speakers(segments, audio, 16000, similarity_threshold=threshold)
    assert result == segments


@pytest.mark.parametrize(
    "segments",
    [
        pytest.param([], id="empty_segments"),
        pytest.param(
            [DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0)],
            id="single_speaker",
        ),
    ],
)
def test_merge_similar_speakers_too_few_speakers(segments):
    """merge_similar_speakers returns input unchanged when 0 or 1 speakers exist."""
    audio = voice_signal(2.0, 16000, fundamental=200.0)
    assert merge_similar_speakers(segments, audio, 16000) == segments


def test_merge_similar_speakers_silent_audio():
    """merge_similar_speakers skips comparison when audio profile norm is near zero (< EPSILON)."""
    sr = 16000
    audio = np.zeros(int(5.0 * sr), dtype=np.float32)
    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
        DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=5.0),
    ]

    merged = merge_similar_speakers(segments, audio, sr)
    assert merged == segments


def test_merge_similar_speakers_merges_matching_profiles():
    """merge_similar_speakers merges speakers with similar spectral profiles."""
    sr = 16000
    audio = voice_signal(5.0, sr, fundamental=200.0)

    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
        DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=5.0),
    ]

    merged = merge_similar_speakers(segments, audio, sr, similarity_threshold=0.95)
    assert len(merged) == 2
    assert merged[0].speaker == "SPEAKER_00"
    assert merged[1].speaker == "SPEAKER_00"


def test_merge_similar_speakers_no_merges_when_distinct():
    """merge_similar_speakers leaves distinct speakers separate and returns original list."""
    sr = 16000
    n_samples = int(5.0 * sr)
    audio = np.zeros(n_samples, dtype=np.float32)

    audio[: int(2.0 * sr)] = voice_signal(2.0, sr, fundamental=200.0)
    start2 = int(2.5 * sr)
    sig_b = voice_signal(2.5, sr, fundamental=800.0)
    audio[start2 : start2 + len(sig_b)] = sig_b

    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
        DiarizationSegment(speaker="SPEAKER_01", start=2.5, end=5.0),
    ]

    merged = merge_similar_speakers(segments, audio, sr, similarity_threshold=0.95)
    assert merged == segments  # Returns original list object when all merge_map[s] == s


def test_merge_similar_speakers_minor_speaker_absorption():
    """Minor speaker with < 15s and < 20% total speech is absorbed at 0.92 threshold."""
    sr = 16000
    audio = np.zeros(int(105.0 * sr), dtype=np.float32)

    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=100.0),
        DiarizationSegment(speaker="SPEAKER_01", start=100.0, end=105.0),
    ]

    # Mock spectral profiles to give cosine similarity = 0.93 (between 0.92 and 0.95)
    p0 = np.array([1.0, 0.0], dtype=np.float64)
    p1 = np.array([0.93, np.sqrt(1 - 0.93**2)], dtype=np.float64)

    def _mock_profile(mon, rate, segs, speaker):
        return p0 if speaker == "SPEAKER_00" else p1

    with patch("tapeback.speaker_merge._speaker_spectral_profile", side_effect=_mock_profile):
        merged = merge_similar_speakers(segments, audio, sr, similarity_threshold=0.95)

    assert merged[0].speaker == "SPEAKER_00"
    assert merged[1].speaker == "SPEAKER_00"


def test_merge_similar_speakers_already_merged_continue():
    """merge_similar_speakers skips comparison when sp_a and sp_b already map to same root."""
    sr = 16000
    audio = np.zeros(int(6.0 * sr), dtype=np.float32)

    segments = [
        DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
        DiarizationSegment(speaker="SPEAKER_01", start=2.0, end=4.0),
        DiarizationSegment(speaker="SPEAKER_02", start=4.0, end=6.0),
    ]

    # All three profiles identical -> SPEAKER_00 and SPEAKER_01 merge first.
    # When sp_a="SPEAKER_01" and sp_b="SPEAKER_02", merge_map["SPEAKER_01"] ==
    # merge_map["SPEAKER_02"] ("SPEAKER_00"), which triggers skip.
    p = np.array([1.0, 1.0], dtype=np.float64)

    with patch("tapeback.speaker_merge._speaker_spectral_profile", return_value=p):
        merged = merge_similar_speakers(segments, audio, sr, similarity_threshold=0.95)

    assert all(s.speaker == "SPEAKER_00" for s in merged)
