"""PII masking for the one thing that leaves this machine — the LLM request.

Recording, Whisper and pyannote all run locally; `summarizer.summarize()` is the only
place a transcript crosses to a third party, and its fallback chain can hand the same
text to a second provider when the first fails. When `TAPEBACK_MASK_PII` is on, this
module replaces structured PII with `[LABEL_N]` placeholders before the text is sent and
restores the real values in the parsed result, so the vault keeps what was actually said
while no provider ever sees it.

Off by default. With masking disabled every method is the identity function — the request
is byte-identical to what it would have been without this module.

Coverage is deliberately narrow: deterministic regex for email and phone. Person names
are the PII people actually speak aloud, and detecting them needs either a user-supplied
list or NER; both are follow-ups. `_RULES` is the seam for adding detectors.
"""

import functools
import re

type MaskMap = dict[str, str]  # placeholder -> original value

# Email. The local part allows the usual symbols; the host must end in a TLD.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# Phone: Russian (+7 / 8 + 10 digits) and generic international (+<country>...),
# tolerant of spaces, dashes and parentheses. Anchored on a leading +/8 with
# word-boundary look-around so it does not grab bare digit runs (years, ids).
_PHONE_RE = re.compile(
    r"(?<![\w.])(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?![\w])"
    r"|(?<![\w.])\+\d{1,3}[\s\-]?\(?\d{2,4}\)?(?:[\s\-]?\d{2,4}){2,4}(?![\w])"
)
# Email first: its host would otherwise be a candidate for phone matching.
_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (("EMAIL", _EMAIL_RE), ("PHONE", _PHONE_RE))

_PLACEHOLDER_RE = re.compile(r"\[(?:" + "|".join(label for label, _ in _RULES) + r")_\d+\]")


class Masker:
    """Per-call PII map. Mask the transcript, send it, then unmask each field of the
    parsed response with the same map. One value gets ONE placeholder for the whole call,
    so the model sees a consistent entity and the map stays small.

    Construct one per `summarize` call and discard it — the map holds plaintext PII and
    must never outlive the call or be written anywhere.
    """

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        self._to_placeholder: dict[str, str] = {}  # original -> placeholder
        self._to_original: MaskMap = {}  # placeholder -> original
        self._counts: dict[str, int] = {}  # per-label running index

    def mask(self, text: str) -> str:
        """Replace PII with placeholders. Identity when masking is disabled."""
        if not self._enabled or not text:
            return text
        out = text
        for label, pattern in _RULES:
            out = pattern.sub(functools.partial(self._sub, label), out)
        return out

    def _sub(self, label: str, match: re.Match[str]) -> str:
        value = match.group(0)
        existing = self._to_placeholder.get(value)
        if existing is not None:
            return existing  # same value -> same placeholder (consistent + small map)
        self._counts[label] = self._counts.get(label, 0) + 1
        placeholder = f"[{label}_{self._counts[label]}]"
        self._to_placeholder[value] = placeholder
        self._to_original[placeholder] = value
        return placeholder

    @property
    def mapping(self) -> MaskMap:
        return dict(self._to_original)

    def unmask(self, text: str) -> str:
        """Restore the originals this masker replaced."""
        if not text or not self._to_original:
            return text
        out = text
        # Longest placeholder first: the closing bracket already prevents [X_1] from
        # matching inside [X_11], but be explicit and order-independent.
        for placeholder, original in sorted(
            self._to_original.items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            out = out.replace(placeholder, original)
        return out

    def residual_placeholders(self, text: str) -> list[str]:
        """Placeholder-shaped fragments left after unmasking — the model invented an
        index or reformatted one, so the real value could not be put back. Empty when
        this masker replaced nothing, since then any such fragment came from the model.
        """
        if not self._to_original:
            return []
        return _PLACEHOLDER_RE.findall(text)
