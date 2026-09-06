"""Compute-type resolution and batching policy for faster-whisper.

Owns the pure decision logic of *how* the model runs: which ctranslate2
compute type is valid for the resolved device (and what a CUDA-only
request degrades to on CPU), and which anti-hallucination settings
BatchedInferencePipeline silently drops — so the user can be told. No
model construction, no CUDA, no GPU queries here."""

import sys

from tapeback.settings import Settings

# Compute types ctranslate2 cannot run on CPU. Requesting one there raises
# ValueError rather than degrading, so a device fallback has to translate it.
_CUDA_ONLY_COMPUTE_TYPES = frozenset({"float16", "int8_float16", "bfloat16", "int8_bfloat16"})


def _resolve_compute_type(compute_type: str, device: str) -> str:
    """Resolve the compute type for the device we ended up on.

    - auto + cuda → int8_float16
    - auto + cpu  → int8
    - an explicit CUDA-only type on CPU → int8, because ctranslate2 raises otherwise
    - any other explicit value passes through.

    The CPU translation matters because the device is now chosen at runtime: a card that
    is thermally clamped or out of VRAM sends us to the CPU carrying whatever
    TAPEBACK_COMPUTE_TYPE was set for the GPU. Without this, that combination died with
    "Requested int8_float16 compute type, but the target device or backend do not
    support efficient int8_float16 computation" — a crash instead of a fallback.

    int8_float16 rather than float16 because it is faster *and* smaller, which is not
    the usual trade-off. Measured on a GTX 1650 Ti with large-v3-turbo, same 90 s clip,
    twice each: float16 3.90x real time and 2139 MiB, int8_float16 **14.16x and
    1115 MiB**. Quality does not pay for it — decoding the same audio both ways gave
    near-identical text with single-word differences in both directions, and across the
    benchmark grid int8_float16 had the lower share of low-confidence words.

    The likely reason is hardware: this is a Turing part without tensor cores, so fp16
    gets no acceleration while int8 uses the integer datapath. That is a hypothesis;
    the measurements are not. ctranslate2 falls back on its own if a device does not
    support the requested type, so this stays safe on other GPUs.
    """
    if compute_type == "auto":
        return "int8_float16" if device == "cuda" else "int8"
    if device != "cuda" and compute_type in _CUDA_ONLY_COMPUTE_TYPES:
        print(
            f"Warning: compute type {compute_type} is GPU-only; using int8 on {device}.",
            file=sys.stderr,
        )
        return "int8"
    return compute_type


# Parameters tapeback configures that BatchedInferencePipeline silently drops.
# Verified against faster-whisper 1.2.1's own "Unused Arguments" docstring; the
# temperature entry is separate because it is not ignored outright — only the
# first value of the ladder is used, which disables the anti-hallucination retries.
BATCHED_IGNORED_SETTINGS = (
    "no_speech_threshold",
    "condition_on_previous_text",
    "hallucination_silence_threshold",
)


def _batched_warning(settings: Settings) -> str | None:
    """Warn if batching would silently drop anti-hallucination settings.

    Enabling batching quietly reverts several deliberate choices, and the run
    otherwise looks identical — so the user must be told which ones, rather than
    discovering it in a transcript full of repeats.
    """
    dropped = [name for name in BATCHED_IGNORED_SETTINGS if getattr(settings, name) is not None]
    if len(settings.temperature) > 1:
        dropped.append("temperature (only the first value is used)")
    if not dropped:
        return None
    return (
        f"Warning: TAPEBACK_BATCH_SIZE={settings.batch_size} enables batched inference, "
        f"which ignores: {', '.join(dropped)}. "
        "These are anti-hallucination settings; expect more repeats on quiet channels."
    )
