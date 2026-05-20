#!/bin/sh
set -e
/opt/tapeback/venv/bin/pip install --quiet --upgrade \
    'anthropic>=0.40.0,<1' \
    'openai>=1.50.0,<3'
