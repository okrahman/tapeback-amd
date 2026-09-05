import re
from dataclasses import dataclass

from tapeback import const
from tapeback._quality import has_speech, strip_hallucinations
from tapeback.models import Segment

# Words with probability below this are marked as uncertain (italic in markdown).
# 0.35 is tuned for multilingual speech: English loanwords inside Russian sentences
# (code-switching) often come back with 0.3-0.5 probability even when correct.
WORD_LOW_CONFIDENCE = 0.35

# Upper bound on how much audio one merged block may span, in seconds.
# Merging is driven by speaker identity and pause length, neither of which bounds the
# result: a long uninterrupted monologue merged into a single block, so a 31-minute
# recording rendered with its last timecode at [00:00:45]. The text was intact but
# unnavigable. 60s keeps a block readable while still giving a timecode a minute.
MAX_BLOCK_SECONDS = 60.0


# Terminal-control and other non-printable characters, sanitized out of inline text
# by `_display_value`.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def _yaml_scalar(value: object) -> str:
    """Render a value as a safe double-quoted YAML scalar.

    Metadata values can come from remote detection or an older resume cache, so a
    quote, backslash, or newline must never escape into the front matter and change
    its structure.
    """
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _display_value(value: object) -> str:
    """A value safe to interpolate into an inline markdown paragraph.

    Control characters cannot be typed into a note; they are collapsed to spaces so
    a malicious or corrupt value cannot inject terminal controls or fake headings
    into the transcript body.
    """
    text = str(value)
    return _CONTROL_CHARS_RE.sub(" ", text)


def _format_timecode(seconds: float) -> str:
    """Format seconds as [HH:MM:SS]."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def _format_duration_human(seconds: float) -> str:
    """Format duration as human-readable string (e.g. '1h 23m 45s')."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _format_duration_hms(seconds: float) -> str:
    """Format duration as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _merge_consecutive_speakers(
    segments: list[Segment],
    pause_threshold: float = 1.0,
    max_block_seconds: float = MAX_BLOCK_SECONDS,
) -> list[tuple[float, str | None, str]]:
    """Merge consecutive segments from the same speaker into one block.

    Segments from the same speaker are NOT merged when the gap between them
    exceeds pause_threshold — this preserves intentional pauses within a
    single speaker's speech.

    A block is also closed once it spans max_block_seconds, so an uninterrupted
    monologue still gets timecodes to navigate by.

    Returns list of (start_time, speaker, merged_text).
    """
    if not segments:
        return []

    merged: list[tuple[float, str | None, str]] = []
    current_start = segments[0].start
    current_end = segments[0].end
    current_speaker = segments[0].speaker
    current_texts = [segments[0].text]

    for seg in segments[1:]:
        gap = seg.start - current_end
        too_long = seg.end - current_start >= max_block_seconds
        if seg.speaker == current_speaker and gap < pause_threshold and not too_long:
            current_texts.append(seg.text)
            current_end = seg.end
        else:
            merged.append((current_start, current_speaker, " ".join(current_texts)))
            current_start = seg.start
            current_end = seg.end
            current_speaker = seg.speaker
            current_texts = [seg.text]

    merged.append((current_start, current_speaker, " ".join(current_texts)))
    return merged


def _mark_low_confidence_words(segment: Segment) -> Segment:
    """Create a new Segment with low-confidence words marked in italic.

    Consecutive low-confidence words are grouped into a single italic span:
    ``*Sorry could you* repeat`` instead of ``*Sorry* *could* *you* repeat``.
    """
    if not segment.words:
        return segment

    parts: list[str] = []
    low_group: list[str] = []

    for word in segment.words:
        text = word.word.strip()
        if not text:
            continue
        if word.probability < WORD_LOW_CONFIDENCE:
            low_group.append(text)
        else:
            if low_group:
                parts.append(f"*{' '.join(low_group)}*")
                low_group = []
            parts.append(text)

    if low_group:
        parts.append(f"*{' '.join(low_group)}*")

    if not parts:
        return segment

    return Segment(
        start=segment.start,
        end=segment.end,
        text=" ".join(parts),
        words=segment.words,
        speaker=segment.speaker,
    )


def _strip_hallucinated_text(segment: Segment) -> Segment:
    """Drop subtitle-corpus phrases from a segment's text.

    Words are left untouched: they still carry the timings that channel filtering and
    diarization resegmentation depend on, and the italic markup pass reads the text,
    not the word list.
    """
    cleaned = strip_hallucinations(segment.text)
    if cleaned == segment.text:
        return segment
    return Segment(
        start=segment.start,
        end=segment.end,
        text=cleaned,
        words=segment.words,
        speaker=segment.speaker,
    )


def _format_segments_block(segments: list[Segment]) -> list[str]:
    """Format a list of segments into timecoded markdown lines.

    Low-confidence words (probability < 0.5) are marked with *italics*.
    """
    # Wordless decoder segments carry authoritative text but no timing confidence;
    # retain brief speech such as whisper.cpp's final "timer." fragment. Genuine
    # word-timed output keeps the VAD-artifact duration guard.
    long_enough = [
        s for s in segments if not s.words or s.end - s.start >= const.MIN_SEGMENT_DURATION
    ]
    long_enough = [_strip_hallucinated_text(s) for s in long_enough]
    long_enough = [s for s in long_enough if has_speech(s.text)]
    long_enough = [_mark_low_confidence_words(s) for s in long_enough]
    merged = _merge_consecutive_speakers(long_enough)

    lines: list[str] = []
    for start_time, speaker, text in merged:
        timecode = _format_timecode(start_time)

        if speaker:
            lines.append(f"{timecode} **{speaker}:** {text}")
        else:
            lines.append(f"{timecode} {text}")
        lines.append("")

    return lines


def _speaker_labels(block: list[str]) -> list[str]:
    """Speaker labels in order of appearance, one per timecoded line."""
    return [line.split("**")[1] for line in block if line.startswith("[") and "**" in line]


def _adds_speaker_information(raw_block: list[str], diarized_block: list[str]) -> bool:
    """True if the diarized rendering says something the raw one does not.

    Diarization that only renames "Other" to "Speaker 1" produces a byte-for-byte
    duplicate of the transcript — every file that carried the section was twice the
    size for no extra information. It is worth a second section only when it actually
    splits the conversation between speakers, i.e. when the run of labels differs in
    shape rather than in wording.
    """
    if len(raw_block) != len(diarized_block):
        return True
    raw_labels = _speaker_labels(raw_block)
    diarized_labels = _speaker_labels(diarized_block)
    # Compare the grouping, not the names: "Other/Other" vs "Speaker 1/Speaker 1" is
    # the same information, while "Other/Other" vs "Speaker 1/Speaker 2" is not.
    return _label_shape(raw_labels) != _label_shape(diarized_labels)


def _label_shape(labels: list[str]) -> list[int]:
    """Map labels to the order they first appear, so naming does not matter."""
    seen: dict[str, int] = {}
    return [seen.setdefault(label, len(seen)) for label in labels]


@dataclass(frozen=True)
class TranscriptMeta:
    """Everything the note needs that is not the transcript itself.

    Grouped rather than passed as loose arguments: these all describe the run, they
    all come from the same place in the pipeline, and the parameter list had grown
    past the point where call sites stayed readable.
    """

    session_name: str
    audio_rel_path: str
    duration_seconds: float
    language: str
    # True when transcription was interrupted and the note covers only part of the
    # recording. A partial transcript that looks complete is worse than none, because
    # nothing prompts a re-run.
    partial: bool = False


def format_markdown(
    segments: list[Segment],
    meta: TranscriptMeta,
    raw_segments: list[Segment] | None = None,
) -> str:
    """Generate markdown with YAML front matter.

    Segments shorter than 1 second are filtered out (VAD artifacts).
    Each segment starts with [HH:MM:SS] timecode.

    When raw_segments is provided, outputs two sections:
    - "## Transcript" with raw (pre-diarization) segments
    - "## Diarized Transcript" with diarized segments
    """
    # Parse date and time from session name (format: YYYY-MM-DD_HH-MM-SS)
    parts = meta.session_name.split("_")
    date_str = parts[0] if parts else meta.session_name
    time_str = parts[1].replace("-", ":") if len(parts) > 1 else "00:00"
    # Only HH:MM for display
    time_display = ":".join(time_str.split(":")[:2])

    duration_hms = _format_duration_hms(meta.duration_seconds)
    duration_human = _format_duration_human(meta.duration_seconds)

    lines = [
        "---",
        f"date: {date_str}",
        f'time: "{time_display}"',
        f'duration: "{duration_hms}"',
        f"language: {_yaml_scalar(meta.language)}",
        f'audio: "[[{meta.audio_rel_path}]]"',
        "tags:",
        "  - meeting",
        "  - transcript",
    ]
    if meta.partial:
        # Both machine- and human-visible: a partial transcript that looks complete is
        # worse than no transcript, because nothing prompts you to re-run it.
        lines.append("  - partial")
        lines.append("partial: true")
    lines += [
        "---",
        "",
        f"# Meeting {date_str} {time_display}",
        "",
        f"**Duration:** {duration_human} | **Language:** {_display_value(meta.language)}",
        "",
    ]
    if meta.partial:
        lines.append(
            "> [!warning] Interrupted — this transcript covers only part of the "
            "recording. Re-run to complete it."
        )
        lines.append("")
    lines += ["---", ""]

    diarized_block = _format_segments_block(segments)
    raw_block = _format_segments_block(raw_segments) if raw_segments is not None else None

    if raw_block is not None and _adds_speaker_information(raw_block, diarized_block):
        lines.append("## Transcript")
        lines.append("")
        lines.extend(raw_block)
        lines.append("---")
        lines.append("")
        lines.append("## Diarized Transcript")
        lines.append("")
        lines.extend(diarized_block)
    else:
        lines.extend(diarized_block)

    return "\n".join(lines)


def format_live_markdown(
    segments: list[Segment],
    session_name: str,
    language: str,
) -> str:
    """Generate a simplified live markdown transcript (no duration, no raw_segments).

    Updated atomically during recording so the user can open it mid-meeting.
    Replaced by the final polished transcript after recording stops.
    """
    parts = session_name.split("_")
    date_str = parts[0] if parts else session_name
    time_str = parts[1].replace("-", ":") if len(parts) > 1 else "00:00"
    time_display = ":".join(time_str.split(":")[:2])

    lines = [
        f"# Live Transcript {date_str} {time_display}",
        "",
        f"**Language:** {_display_value(language)} | **Status:** recording in progress",
        "",
        "---",
        "",
    ]

    if segments:
        lines.extend(_format_segments_block(segments))
    else:
        lines.append("*Waiting for first transcription cycle...*")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Live preview. Final transcript with diarization will replace this file.*")
    lines.append("")

    return "\n".join(lines)
