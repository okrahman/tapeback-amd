#!/bin/sh
set -e
# pyannote-audio pulls in torch (~2 GB). First install can take several minutes.
/opt/tapeback/venv/bin/pip install --quiet --upgrade \
    'pyannote-audio>=3.1.0,<5' \
    'torch>=2.0.0,<3' \
    'torchaudio>=2.0.0,<3'
