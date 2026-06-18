"""GPU helpers — shared between the pipeline, transcriber, and diarizer."""

import contextlib
import ctypes
import gc
import importlib
from pathlib import Path

# Substrings that mark a recoverable CUDA/GPU failure. ctranslate2 (faster-whisper)
# and torch (pyannote) phrase these differently, so match case-insensitively.
# A RuntimeError WITHOUT any of these must NOT trigger a CPU fallback — that would
# mask the real bug (e.g. a corrupt audio file) and make work ~10x slower.
_CUDA_ERROR_MARKERS = ("cuda", "out of memory", "cublas", "cudnn")

# faster-whisper's ctranslate2 is built against CUDA 12 and dlopens these by
# SONAME. When only CUDA 13 is on the default library path (e.g. torch's bundled
# cu13 libraries), the SONAME isn't found and transcription silently drops to CPU.
# Preloading the cu12 wheels' copies into the process resolves the SONAME.
_CUDA12_PRELOAD = (
    ("nvidia.cublas.lib", "libcublas.so.12"),
    ("nvidia.cudnn.lib", "libcudnn.so.9"),
)


def is_cuda_error(exc: BaseException) -> bool:
    """True if the exception looks like a recoverable CUDA/GPU failure."""
    message = str(exc).lower()
    return any(marker in message for marker in _CUDA_ERROR_MARKERS)


def preload_cuda_libs() -> None:
    """Preload CUDA 12 cuBLAS/cuDNN so ctranslate2 (faster-whisper) finds them.

    Best-effort: if the cu12 wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12) are
    not installed, this is a no-op and transcription still falls back to CPU with
    a clear message. Saves users on CUDA 13 systems from setting LD_LIBRARY_PATH
    by hand.
    """
    for module_name, soname in _CUDA12_PRELOAD:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for entry in module.__path__:
            so_path = Path(entry) / soname
            if so_path.exists():
                with contextlib.suppress(OSError):
                    ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
                break


def free_gpu_memory() -> None:
    """Release Python refs and clear CUDA cache so the next model fits in VRAM."""
    gc.collect()
    try:
        import torch  # noqa: PLC0415 — optional dependency, guarded by try/except

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
