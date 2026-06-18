"""Unit tests for GPU error classification and CUDA library preloading."""

from types import SimpleNamespace

import pytest

from tapeback import _gpu
from tapeback._gpu import is_cuda_error, preload_cuda_libs


@pytest.mark.parametrize(
    "message,expected",
    [
        ("CUDA failed with error out of memory", True),
        ("CUDA out of memory", True),
        ("Library libcublas.so.12 is not found", True),
        ("cuDNN failed to initialize", True),
        ("no CUDA-capable device is detected", True),
        ("corrupt audio frame at offset 1234", False),
        ("invalid sample rate", False),
    ],
)
def test_is_cuda_error(message, expected):
    assert is_cuda_error(RuntimeError(message)) is expected


def test_preload_cuda_libs_loads_present_libraries(monkeypatch, tmp_path):
    """When the cu12 wheels are installed, their cuBLAS/cuDNN get preloaded."""
    cublas = tmp_path / "cublas"
    cublas.mkdir()
    (cublas / "libcublas.so.12").touch()
    cudnn = tmp_path / "cudnn"
    cudnn.mkdir()
    (cudnn / "libcudnn.so.9").touch()

    fake = {
        "nvidia.cublas.lib": SimpleNamespace(__path__=[str(cublas)]),
        "nvidia.cudnn.lib": SimpleNamespace(__path__=[str(cudnn)]),
    }
    monkeypatch.setattr(_gpu.importlib, "import_module", lambda name: fake[name])
    loaded: list[str] = []
    monkeypatch.setattr(_gpu.ctypes, "CDLL", lambda path, mode=0: loaded.append(path))

    preload_cuda_libs()

    assert any(p.endswith("libcublas.so.12") for p in loaded)
    assert any(p.endswith("libcudnn.so.9") for p in loaded)


def test_preload_cuda_libs_noop_when_wheels_absent(monkeypatch):
    """Without the cu12 wheels, preload is a silent no-op (CPU fallback still works)."""

    def _missing(name):
        raise ImportError(name)

    monkeypatch.setattr(_gpu.importlib, "import_module", _missing)
    loaded: list[str] = []
    monkeypatch.setattr(_gpu.ctypes, "CDLL", lambda path, mode=0: loaded.append(path))

    preload_cuda_libs()  # must not raise

    assert loaded == []
