"""Unit tests for GPU error classification."""

import pytest

from tapeback._gpu import is_cuda_error


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
