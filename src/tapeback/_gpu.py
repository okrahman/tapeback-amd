"""GPU helpers — shared between the pipeline, transcriber, and diarizer."""

import gc

# Substrings that mark a recoverable CUDA/GPU failure. ctranslate2 (faster-whisper)
# and torch (pyannote) phrase these differently, so match case-insensitively.
# A RuntimeError WITHOUT any of these must NOT trigger a CPU fallback — that would
# mask the real bug (e.g. a corrupt audio file) and make work ~10x slower.
_CUDA_ERROR_MARKERS = ("cuda", "out of memory", "cublas", "cudnn")


def is_cuda_error(exc: BaseException) -> bool:
    """True if the exception looks like a recoverable CUDA/GPU failure."""
    message = str(exc).lower()
    return any(marker in message for marker in _CUDA_ERROR_MARKERS)


def free_gpu_memory() -> None:
    """Release Python refs and clear CUDA cache so the next model fits in VRAM."""
    gc.collect()
    try:
        import torch  # noqa: PLC0415 — optional dependency, guarded by try/except

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
