#!/bin/sh
set -e
# CUDA 12 cuBLAS/cuDNN for ctranslate2 (faster-whisper) GPU transcription (~700 MB).
# tapeback preloads these so the SONAME resolves on CUDA 13 systems.
/opt/tapeback/venv/bin/pip install --quiet --upgrade \
    nvidia-cublas-cu12 \
    nvidia-cudnn-cu12
