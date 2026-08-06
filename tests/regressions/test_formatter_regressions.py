"""Regression tests for transcript formatting bugs."""

from tapeback.formatter import TranscriptMeta, format_markdown
from tapeback.models import Segment


def _speech(start: float, end: float, speaker: str | None = None) -> Segment:
    return Segment(start=start, end=end, text="реплика.", words=None, speaker=speaker)


def test_long_single_speaker_stretch_is_not_one_block():
    """A long uninterrupted stretch must stay navigable by timecode.

    Bug: _merge_consecutive_speakers merged while the speaker was unchanged and the
    gap stayed under pause_threshold, with no upper bound on the result. A 31-minute
    single-speaker recording rendered as two blocks whose last timecode was
    [00:00:45] — the text was all there, but there was no way to find anything in it.
    """
    # 30 minutes of back-to-back speech from one speaker, gaps far below the
    # pause threshold so nothing else would split it.
    segments = [_speech(i * 10.0, i * 10.0 + 9.9, "You") for i in range(180)]

    markdown = format_markdown(
        segments=segments,
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=1800.0,
            language="ru",
        ),
    )

    timecodes = [line.split("]")[0] + "]" for line in markdown.splitlines() if line.startswith("[")]
    assert len(timecodes) > 1, "30 minutes collapsed into a single block"
    # Navigation is only useful if the timecodes track the audio: the last one must
    # be near the end, not near the beginning.
    assert timecodes[-1] > "[00:25:00]"


def test_short_stretch_is_still_merged_into_one_block():
    """The cap must not fragment ordinary short turns."""
    segments = [_speech(i * 2.0, i * 2.0 + 1.9, "You") for i in range(5)]

    markdown = format_markdown(
        segments=segments,
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=10.0,
            language="ru",
        ),
    )

    timecodes = [line for line in markdown.splitlines() if line.startswith("[")]
    assert len(timecodes) == 1


def _said(text: str, start: float = 0.0, end: float = 5.0) -> Segment:
    return Segment(start=start, end=end, text=text, words=None, speaker="You")


def test_subtitle_corpus_hallucinations_are_stripped():
    """Whisper's subtitle-corpus phrases must not reach the transcript.

    Bug: Whisper emits credits from its training data over long pauses. Real
    occurrences across the project's own transcripts included "Субтитры DimaTorzok"
    (twice), "Редактор субтитров .Семкин", "Корректор .Кулакова" and six of
    "Продолжение следует...". None were filtered, so they read as things people said.
    """
    segments = [
        _said("Ведь решение. Субтитры DimaTorzok штука, куда ты вставляешь summary.", 0.0, 5.0),
        _said("Как будто нам сейчас не нужно. Редактор субтитров .Семкин", 6.0, 11.0),
        _said("поскольку Корректор .Кулакова общаются", 12.0, 17.0),
        _said("Продолжение следует...", 18.0, 23.0),
    ]

    markdown = format_markdown(
        segments=segments,
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=30.0,
            language="ru",
        ),
    )

    lowered = markdown.lower()
    for marker in (
        "dimatorzok",
        "субтитры",
        "редактор субтитров",
        "корректор",
        "продолжение следует",
    ):
        assert marker not in lowered, marker
    # The real speech around the hallucination must survive.
    assert "куда ты вставляешь summary" in markdown
    assert "как будто нам сейчас не нужно" in markdown.lower()


def test_segment_that_is_only_a_hallucination_is_dropped_entirely():
    segments = [
        _said("Продолжение следует...", 0.0, 5.0),
        _said("Реальная реплика про pipeline.", 6.0, 11.0),
    ]

    markdown = format_markdown(
        segments=segments,
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=15.0,
            language="ru",
        ),
    )

    lines = [line for line in markdown.splitlines() if line.startswith("[")]
    assert len(lines) == 1
    assert "pipeline" in lines[0]


def test_diarized_section_omitted_when_it_only_relabels_speakers():
    """A "Diarized Transcript" that says the same thing twice must not be emitted.

    Bug: all 16 transcripts carrying the section duplicated the whole transcript —
    same timecodes, same text, same recognition errors — differing only in
    "**Other:**" becoming "**Speaker 1:**". It doubled the file for no information.
    """
    raw = [_said("Первая реплика.", 0.0, 5.0), _said("Вторая реплика.", 6.0, 11.0)]
    diarized = [
        Segment(start=0.0, end=5.0, text="Первая реплика.", words=None, speaker="Speaker 1"),
        Segment(start=6.0, end=11.0, text="Вторая реплика.", words=None, speaker="Speaker 1"),
    ]

    markdown = format_markdown(
        segments=diarized,
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=15.0,
            language="ru",
        ),
        raw_segments=raw,
    )

    assert markdown.count("Первая реплика") == 1
    assert "## Diarized Transcript" not in markdown


def test_diarized_section_kept_when_it_adds_speaker_information():
    """Real diarization splits a block between speakers — that is worth showing."""
    raw = [_said("Первая реплика.", 0.0, 5.0), _said("Вторая реплика.", 6.0, 11.0)]
    diarized = [
        Segment(start=0.0, end=5.0, text="Первая реплика.", words=None, speaker="Speaker 1"),
        Segment(start=6.0, end=11.0, text="Вторая реплика.", words=None, speaker="Speaker 2"),
    ]

    markdown = format_markdown(
        segments=diarized,
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=15.0,
            language="ru",
        ),
        raw_segments=raw,
    )

    assert "## Diarized Transcript" in markdown
    assert "**Speaker 2:**" in markdown


def test_speaker_change_still_splits_regardless_of_length():
    segments = [_speech(0.0, 5.0, "You"), _speech(5.0, 10.0, "Other")]

    markdown = format_markdown(
        segments=segments,
        meta=TranscriptMeta(
            session_name="2026-08-06_12-00-00",
            audio_rel_path="attachments/audio/x.wav",
            duration_seconds=10.0,
            language="ru",
        ),
    )

    lines = [line for line in markdown.splitlines() if line.startswith("[")]
    assert len(lines) == 2
    assert "**You:**" in lines[0]
    assert "**Other:**" in lines[1]
