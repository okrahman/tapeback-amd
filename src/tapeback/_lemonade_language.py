"""Lemonade language naming and validation.

Owns the Whisper language-name tables and normalization, and the schema
check for server-supplied language values: a hostile value must fall back,
never be coerced or pinned."""

from __future__ import annotations

import re
from typing import Any

from tapeback._lemonade_errors import LemonadeCapabilityError

# Whisper's own language names (what a server echoing Whisper metadata returns)
# mapped to ISO-639-1 codes. Anything already a two-letter code passes through.
LANGUAGE_NAMES: dict[str, str] = {
    "afrikaans": "af",
    "albanian": "sq",
    "amharic": "am",
    "arabic": "ar",
    "armenian": "hy",
    "assamese": "as",
    "azerbaijani": "az",
    "bengali": "bn",
    "bosnian": "bs",
    "bulgarian": "bg",
    "catalan": "ca",
    "chinese": "zh",
    "croatian": "hr",
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "galician": "gl",
    "georgian": "ka",
    "german": "de",
    "greek": "el",
    "gujarati": "gu",
    "hausa": "ha",
    "hebrew": "he",
    "hindi": "hi",
    "hungarian": "hu",
    "icelandic": "is",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "javanese": "jv",
    "kannada": "kn",
    "kazakh": "kk",
    "khmer": "km",
    "korean": "ko",
    "latin": "la",
    "latvian": "lv",
    "lithuanian": "lt",
    "macedonian": "mk",
    "malay": "ms",
    "malayalam": "ml",
    "marathi": "mr",
    "mongolian": "mn",
    "myanmar": "my",
    "nepali": "ne",
    "norwegian": "no",
    "nynorsk": "nn",
    "persian": "fa",
    "polish": "pl",
    "portuguese": "pt",
    "punjabi": "pa",
    "romanian": "ro",
    "russian": "ru",
    "serbian": "sr",
    "sinhala": "si",
    "slovak": "sk",
    "slovenian": "sl",
    "somali": "so",
    "spanish": "es",
    "sundanese": "su",
    "swahili": "sw",
    "swedish": "sv",
    "tagalog": "tl",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "uzbek": "uz",
    "vietnamese": "vi",
    "welsh": "cy",
    "yiddish": "yi",
    "yoruba": "yo",
}


# Common aliases beyond Whisper's own names.
LANGUAGE_ALIASES: dict[str, str] = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
    "nb": "no",
    "pt-br": "pt",
    "in": "id",  # legacy ISO code still seen in the wild
    "iw": "he",  # legacy ISO code still seen in the wild
}


def normalize_language(raw: str) -> str:
    """Lowercase ISO-639-1 where known; otherwise the lowercased input unchanged.

    Never invents a code: an unrecognised name is passed through as-is rather than
    guessed, because the result feeds the pinned language for later chunks and the
    transcript metadata.
    """
    code = raw.strip().lower()
    if code in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[code]
    if code in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[code]
    return code


# A remote language value must be a bounded, structurally safe token: it is
# interpolated into multipart field headers, the YAML front matter, and the resume
# cache, so quotes, CR/LF, control characters, and megabyte strings must never pass.
# Every real ISO-639-1 code, Whisper language name, and alias (``zh-hans``, ``pt-br``)
# fits this grammar.
_LANGUAGE_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,23}$")


_LANGUAGE_MAX_CHARS = 24


def _remote_language(raw: Any) -> str:
    """Validate a server-supplied language value and normalize it.

    Unlike `normalize_language` (which also digests trusted local settings), this is
    a schema check: anything that is not a bounded language-like string is a
    LemonadeCapabilityError, never coerced with ``str()`` and never pinned. A hostile
    value must fall back, not hard-abort multipart assembly on a later chunk or
    corrupt the transcript metadata.
    """
    if not isinstance(raw, str):
        raise LemonadeCapabilityError(
            "Lemonade returned a response with an unusable language value (not a string)"
        )
    value = raw.strip().lower()
    if len(value) > _LANGUAGE_MAX_CHARS or not _LANGUAGE_TOKEN_RE.match(value):
        raise LemonadeCapabilityError(
            "Lemonade returned a response with an unusable language value "
            "(not a bounded language token)"
        )
    return normalize_language(value)
