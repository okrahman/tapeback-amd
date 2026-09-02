"""Tests for reusing an already-transcribed channel."""

import os
from unittest.mock import MagicMock, patch

import pytest

from tapeback import _resume
from tapeback import audio as audio_mod
from tapeback import pipeline as pipeline_mod
from tapeback.models import Segment, Word
from tapeback.settings import Settings
from tapeback.transcriber import Transcriber


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "monitor_16k.wav"
    path.write_bytes(b"not really audio, but it has a size and an mtime")
    return path


@pytest.fixture
def cached_settings(settings, tmp_path):
    return settings.model_copy(
        update={"device": "cpu", "resume_cache": True, "resume_cache_dir": tmp_path / "resume"}
    )


def _info(partial: bool = False):
    info = MagicMock()
    info.language, info.language_probability, info.duration = "ru", 0.9, 60.0
    return info


def _whisper_segment(start: float, end: float, text: str):
    seg = MagicMock()
    seg.start, seg.end, seg.text = start, end, text
    word = MagicMock()
    word.start, word.end, word.word, word.probability = start, end, text, 0.8
    seg.words = [word]
    return seg


def test_key_changes_when_the_audio_changes(cached_settings, audio, tmp_path):
    fingerprint = _resume.settings_fingerprint(cached_settings)
    first = _resume.resume_key(audio, fingerprint, "transcribe monitor")
    audio.write_bytes(b"different content, different size")
    second = _resume.resume_key(audio, fingerprint, "transcribe monitor")
    assert first is not None
    assert first != second


def test_key_changes_per_channel(cached_settings, audio):
    fingerprint = _resume.settings_fingerprint(cached_settings)
    assert _resume.resume_key(audio, fingerprint, "transcribe mic") != _resume.resume_key(
        audio, fingerprint, "transcribe monitor"
    )


@pytest.mark.parametrize(
    "field",
    [
        "whisper_model",
        "compute_type",
        "beam_size",
        "chunk_length",
        "hotwords",
        "language",
        "gate_mic_silence",
    ],
)
def test_key_changes_when_an_output_affecting_setting_changes(cached_settings, audio, field):
    """A cached channel is only reusable if it would be produced the same way."""
    base = _resume.resume_key(audio, _resume.settings_fingerprint(cached_settings), "transcribe")
    changed = {
        "whisper_model": "tiny",
        "compute_type": "float32",
        "beam_size": 1,
        "chunk_length": 7,
        "hotwords": "different, glossary",
        "language": "de",
        "gate_mic_silence": False,
    }[field]
    other = cached_settings.model_copy(update={field: changed})
    assert base != _resume.resume_key(audio, _resume.settings_fingerprint(other), "transcribe")


def test_key_is_none_for_missing_audio(cached_settings, tmp_path):
    fingerprint = _resume.settings_fingerprint(cached_settings)
    missing = tmp_path / "gone.wav"
    assert _resume.resume_key(missing, fingerprint, "transcribe") is None


def test_round_trip_preserves_segments_and_words(cached_settings, audio, tmp_path):
    key = _resume.resume_key(audio, _resume.settings_fingerprint(cached_settings), "transcribe")
    assert key is not None
    directory = tmp_path / "resume"
    segments = [
        Segment(
            start=0.0,
            end=5.0,
            text="реплика",
            speaker="You",
            words=[Word(start=0.0, end=5.0, word="реплика", probability=0.77)],
        )
    ]

    _resume.store(key, directory, segments, {"language": "ru", "duration": 5.0})
    restored = _resume.load(key, directory)

    assert restored is not None
    loaded, info = restored
    assert loaded == segments
    assert loaded[0].words is not None
    assert loaded[0].words[0].probability == 0.77
    assert info["language"] == "ru"


def test_corrupt_entry_is_ignored(cached_settings, audio, tmp_path):
    """A half-written cache file must cost a redo, not a failed run."""
    key = _resume.resume_key(audio, _resume.settings_fingerprint(cached_settings), "transcribe")
    assert key is not None
    directory = tmp_path / "resume"
    directory.mkdir()
    (directory / key.filename).write_text("{ this is not json")

    assert _resume.load(key, directory) is None


def test_second_run_reuses_the_first(cached_settings, audio):
    """The point: an interrupted run must not redo a channel it already finished."""
    messages: list[str] = []
    with patch("tapeback._fw_backend.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (
            iter([_whisper_segment(0.0, 5.0, "готово")]),
            _info(),
        )
        first, _ = Transcriber(cached_settings).transcribe(audio, stage="transcribe monitor")

        # A second transcriber, as a re-run would build: Whisper must not be called.
        instance.transcribe.reset_mock()
        second, _ = Transcriber(cached_settings).transcribe(
            audio, stage="transcribe monitor", on_status=messages.append
        )

    assert [s.text for s in second] == [s.text for s in first]
    instance.transcribe.assert_not_called()
    assert any("Reusing" in m for m in messages)


def test_partial_results_are_not_cached(cached_settings, audio, tmp_path):
    """Caching a truncated channel would make the next run call it finished."""

    def _interrupting():
        yield _whisper_segment(0.0, 5.0, "начало")
        raise KeyboardInterrupt

    with patch("tapeback._fw_backend.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (_interrupting(), _info())
        segments, info = Transcriber(cached_settings).transcribe(audio, stage="transcribe")

    assert info["partial"] is True
    assert len(segments) == 1
    assert list((tmp_path / "resume").glob("*.json")) == []


def test_cache_can_be_disabled(settings, audio, tmp_path):
    s = settings.model_copy(
        update={"device": "cpu", "resume_cache": False, "resume_cache_dir": tmp_path / "resume"}
    )
    with patch("tapeback._fw_backend.WhisperModel") as mock_model_cls:
        instance = mock_model_cls.return_value
        instance.transcribe.return_value = (iter([_whisper_segment(0.0, 5.0, "x")]), _info())
        Transcriber(s).transcribe(audio)

    assert not (tmp_path / "resume").exists()


def test_prune_keeps_the_newest(tmp_path):
    directory = tmp_path / "resume"
    directory.mkdir()
    for index in range(5):
        entry = directory / f"{index}.json"
        entry.write_text("{}")
        stamp = (index + 1) * 1000
        os.utime(entry, (stamp, stamp))

    _resume._prune(directory, keep=3)

    remaining = sorted(p.name for p in directory.glob("*.json"))
    assert remaining == ["2.json", "3.json", "4.json"]


def test_default_dir_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert _resume.default_resume_dir() == tmp_path / "xdg" / "tapeback" / "resume"


def test_process_file_deterministic_staging_preserves_resume_keys(tmp_path, monkeypatch):
    """process_file creates deterministic staging dirs and preserves mtimes for resume cache."""
    settings = Settings(
        vault_path=tmp_path / "vault",
        resume_cache=True,
        resume_cache_dir=tmp_path / "resume",
        device="cpu",
    )
    fake_wav = tmp_path / "meeting.wav"
    wav_header = (
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x02\x00"
        b"\x80>\x00\x00\x00}\x00\x00\x04\x00\x10\x00data\x00\x00\x00\x00"
    )
    fake_wav.write_bytes(wav_header)

    split_keys_1: list[str] = []

    def mock_transcribe_stereo_1(mic_16k, mon_16k, **kwargs):
        k_mon = _resume.resume_key(mon_16k, "test_fp", "transcribe monitor")
        k_mic = _resume.resume_key(mic_16k, "test_fp", "transcribe mic")
        assert k_mon is not None and k_mic is not None
        split_keys_1.extend([k_mon.digest, k_mic.digest])
        return [], [], {"duration": 1.0}

    monkeypatch.setattr(pipeline_mod, "load_stereo_channels", lambda p: (None, None, 16000))
    monkeypatch.setattr(
        audio_mod,
        "split_channels_16k",
        lambda p, out: (out / "mic_16k.wav", out / "monitor_16k.wav"),
    )

    def mock_split(stereo_wav, output_dir):
        m = output_dir / "mic_16k.wav"
        mo = output_dir / "monitor_16k.wav"
        output_dir.mkdir(parents=True, exist_ok=True)
        m.write_bytes(b"mic audio")
        mo.write_bytes(b"mon audio")
        st = stereo_wav.stat()
        os.utime(m, (st.st_atime, st.st_mtime))
        os.utime(mo, (st.st_atime, st.st_mtime))
        return m, mo

    monkeypatch.setattr(pipeline_mod, "split_channels_16k", mock_split)
    monkeypatch.setattr(pipeline_mod, "gate_wav_inactive", lambda *args: None)
    monkeypatch.setattr(pipeline_mod, "split_on_silence", lambda segs, *args, **kw: segs)
    monkeypatch.setattr(pipeline_mod, "filter_silent_segments", lambda segs, *args, **kw: segs)
    monkeypatch.setattr(pipeline_mod, "diarization_available", lambda: False)
    monkeypatch.setattr(pipeline_mod, "is_stereo", lambda p: True)

    mock_transcriber = MagicMock()
    mock_transcriber.describe.return_value = "mock_backend"
    mock_transcriber.transcribe_stereo.side_effect = mock_transcribe_stereo_1
    monkeypatch.setattr(pipeline_mod, "load_transcriber", lambda s: mock_transcriber)

    pipeline_mod.process_file(
        fake_wav, settings, name="test_session", diarize=False, do_summarize=False
    )

    # Second run on the same file should produce identical resume keys
    split_keys_2: list[str] = []

    def mock_transcribe_stereo_2(mic_16k, mon_16k, **kwargs):
        k_mon = _resume.resume_key(mon_16k, "test_fp", "transcribe monitor")
        k_mic = _resume.resume_key(mic_16k, "test_fp", "transcribe mic")
        assert k_mon is not None and k_mic is not None
        split_keys_2.extend([k_mon.digest, k_mic.digest])
        return [], [], {"duration": 1.0}

    mock_transcriber.transcribe_stereo.side_effect = mock_transcribe_stereo_2
    pipeline_mod.process_file(
        fake_wav, settings, name="test_session", diarize=False, do_summarize=False
    )

    assert len(split_keys_1) == 2
    assert split_keys_1 == split_keys_2


def test_load_handles_recursion_error(tmp_path, monkeypatch):
    """Corrupt or deeply nested JSON in the resume cache that raises RecursionError returns None."""
    key = _resume.ResumeKey("a" * 32)
    entry = tmp_path / key.filename
    entry.write_text("{}")

    def _raise_recursion(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr("tapeback._resume.json.loads", _raise_recursion)
    assert _resume.load(key, tmp_path) is None
