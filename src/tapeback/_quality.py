"""Pure transcript-quality metrics.

These exist so that "did this configuration get better or worse" is a number rather
than an impression. Transcription tuning on this project has twice been reverted
after a change that felt right measured badly, so every knob is now judged by the
same three questions: did the model hallucinate, did it get stuck repeating itself,
and did it get the technical vocabulary right.

Everything here is pure and text-only — the benchmark harness in `scripts/` and the
transcript filter use the same definitions, so a fix is measured by the metric it
was meant to move.
"""

from __future__ import annotations

import itertools
import re

from tapeback import const

# Word characters for both alphabets in use; \w would also match digits, which
# would let timecodes and numbers register as repeated "words".
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# A repeated run this long is a decoding loop, not natural emphasis. Russian and
# English both allow a doubled word ("да да", "very very"); three in a row is where
# it stops being speech.
MIN_REPEAT_RUN = 3

# Length of the phrase considered for loop detection, and how many times it must
# repeat back-to-back to count. Catches "something wrong something wrong", which
# word-level detection misses because no single word repeats consecutively.
LOOP_NGRAM_SIZE = 3
MIN_NGRAM_REPEATS = 2


def _words(text: str) -> list[str]:
    return [match.group().lower() for match in _WORD_RE.finditer(text)]


def find_hallucination_markers(text: str) -> list[str]:
    """Return the subtitle-corpus markers present in the text, in list order.

    Substring matching on a lowercased haystack: the markers turn up glued to
    neighbouring words ("Корректор .Кулакова"), so anchoring on word boundaries
    would miss real occurrences.
    """
    haystack = text.lower()
    return [marker for marker in const.HALLUCINATION_MARKERS if marker in haystack]


def count_hallucination_markers(text: str) -> int:
    """Total occurrences (not distinct markers) of subtitle-corpus phrases."""
    haystack = text.lower()
    return sum(haystack.count(marker) for marker in const.HALLUCINATION_MARKERS)


def count_repeated_words(text: str, min_run: int = MIN_REPEAT_RUN) -> int:
    """Count runs of the same word repeated at least ``min_run`` times in a row.

    A run of any length counts once, so a 10x repeat is one loop rather than eight.

    Known false positives: conversational Russian genuinely repeats short words for
    emphasis or hesitation ("надо надо надо", "да да да да"), and a faithful
    transcript of real speech contains them. Measured on a 31-minute recording, all
    six hits were natural speech, none were decoding loops. Treat a rise here as a
    prompt to look, not as a regression; `count_repeated_phrases` is the reliable
    loop signal because a repeated 3-gram is not something people say.
    """
    words = _words(text)
    loops = 0
    run = 1
    for previous, current in itertools.pairwise(words):
        if current == previous:
            run += 1
            if run == min_run:
                loops += 1
        else:
            run = 1
    return loops


def count_repeated_phrases(
    text: str,
    size: int = LOOP_NGRAM_SIZE,
    min_repeats: int = MIN_NGRAM_REPEATS,
) -> int:
    """Count phrases of ``size`` words repeated back-to-back ``min_repeats`` times.

    Only immediate repetition counts: a phrase recurring later in a meeting is
    normal speech, whereas one repeating on its own heels is a decoding loop.
    """
    words = _words(text)
    if size < 1 or len(words) < size * min_repeats:
        return 0

    loops = 0
    index = 0
    while index + size * min_repeats <= len(words):
        phrase = words[index : index + size]
        repeats = 1
        while words[index + repeats * size : index + (repeats + 1) * size] == phrase:
            repeats += 1
        if repeats >= min_repeats:
            loops += 1
            index += repeats * size
        else:
            index += 1
    return loops


def count_recognised_terms(text: str, terms: list[str]) -> tuple[int, list[str]]:
    """Return how many expected terms appear, and which are missing.

    Case-insensitive substring match. The point is whether the term survived
    recognition at all — "RAG" coming back as "Rick" is a miss regardless of how
    the surrounding sentence reads.
    """
    haystack = text.lower()
    missing = [term for term in terms if term.lower() not in haystack]
    return len(terms) - len(missing), missing
