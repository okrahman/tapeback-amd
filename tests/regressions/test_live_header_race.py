"""Regression tests: a live-mode header-read race must not latch a wrong offset.

`find_data_offset` used to answer the constant 44 whenever the RIFF walk
failed, and `_read_new_pcm` latched that answer on the first poll. A poll that
raced parecord's header flush therefore misaligned every byte offset and
timestamp for that channel for the whole session. Now an unparseable header
returns None and is retried; the 44-byte assumption is made only once the
file has grown past a size bound, and loudly.
"""

import struct
from unittest.mock import MagicMock

from tapeback import const
from tapeback._live_chunks import HEADER_PARSE_FALLBACK_BYTES
from tapeback._live_pcm import find_data_offset
from tapeback.live import LiveTranscriber
from tapeback.settings import Settings


def _partial_header(path):
    """A RIFF file whose header flush was cut off mid-walk (no 'data' chunk yet)."""
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 0xFFFFFFFF))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16))
        f.write(b"\x00" * 64)  # the rest of the header has not been flushed yet


def _loud_pcm_bytes(seconds: float, sample_rate: int = 48000) -> bytes:
    """s16le mono audio that is never digitally silent (full-amplitude square wave)."""
    samples = (10000 if i % 2 == 0 else -10000 for i in range(int(seconds * sample_rate)))
    return b"".join(struct.pack("<h", s) for s in samples)


def _standard_wav(path, pcm: bytes, sample_rate: int = 48000):
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(pcm)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(pcm)))
        f.write(pcm)


def _make_live(settings: Settings, mic, monitor) -> LiveTranscriber:
    return LiveTranscriber(settings, "session", mic, monitor)


def test_truncated_header_is_not_latched(settings, tmp_path):
    """A first poll that races the header flush must not freeze offset 44."""
    mic = tmp_path / "mic.wav"
    monitor = tmp_path / "monitor.wav"
    _partial_header(mic)
    _partial_header(monitor)
    live = _make_live(settings, mic, monitor)

    pcm, offset = live._read_new_pcm(mic, 0, 0, 0, is_mic=True)

    assert pcm is None and offset == 0
    assert live._mic_data_offset is None, "an unreadable header must not be latched"


def test_offset_resolves_once_the_header_is_flushed(settings, tmp_path):
    """After the header arrives, the correct offset is found and used."""
    mic = tmp_path / "mic.wav"
    monitor = tmp_path / "monitor.wav"
    pcm_bytes = _loud_pcm_bytes(0.2)
    _partial_header(mic)
    _partial_header(monitor)
    live = _make_live(settings, mic, monitor)

    # The racing poll — nothing latched.
    live._read_new_pcm(mic, 0, 0, 0, is_mic=True)

    # parecord flushes the complete header + data.
    _standard_wav(mic, pcm_bytes)
    _standard_wav(monitor, _loud_pcm_bytes(0.2))
    pcm, offset = live._read_new_pcm(mic, 0, 0, 0, is_mic=True)

    assert pcm is not None and len(pcm) > 0
    assert live._mic_data_offset == find_data_offset(mic)
    assert offset == len(pcm_bytes)


def test_huge_unparseable_header_falls_back_to_44_loudly(settings, tmp_path, capsys):
    """A genuinely broken large file degrades to 44 — with an explicit warning."""
    mic = tmp_path / "mic.wav"
    monitor = tmp_path / "monitor.wav"
    _partial_header(mic)
    _partial_header(monitor)
    with open(mic, "ab") as f:  # push it past the retry bound
        f.write(b"\x00" * (HEADER_PARSE_FALLBACK_BYTES + 1))
    with open(monitor, "ab") as f:
        f.write(b"\x00" * (HEADER_PARSE_FALLBACK_BYTES + 1))
    live = _make_live(settings, mic, monitor)

    pcm, _offset = live._read_new_pcm(mic, 0, 0, 0, is_mic=True)

    assert live._mic_data_offset == const.WAV_HEADER_FALLBACK
    err = capsys.readouterr().err
    assert "unparseable" in err
    # The zero-filled body reads as (silent) PCM from the assumed offset; the
    # activity gate in _process_chunk is what keeps it out of the note.
    assert pcm is not None
    assert pcm == b"\x00" * len(pcm)


def test_final_partial_interval_warns_and_commits_nothing(settings, tmp_path, capsys, monkeypatch):
    """A partial result on the FINAL pass must be announced, not dropped silently."""
    mic = tmp_path / "mic.wav"
    monitor = tmp_path / "monitor.wav"
    _standard_wav(mic, _loud_pcm_bytes(0.2))
    _standard_wav(monitor, _loud_pcm_bytes(0.2))
    live = _make_live(settings, mic, monitor)

    fake_transcriber = MagicMock()
    monkeypatch.setattr(live, "_ensure_transcriber", lambda: fake_transcriber)
    monkeypatch.setattr(live, "_transcribe_pair", lambda *_a, **_k: ([], [], True))

    live._process_chunk(is_final=True)

    assert live._mic_byte_offset == 0 and live._monitor_byte_offset == 0
    assert "partially decoded" in capsys.readouterr().err
