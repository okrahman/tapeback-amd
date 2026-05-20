#!/bin/sh
set -e
/opt/tapeback/venv/bin/pip install --quiet --upgrade \
    'pystray>=0.19.0,<1' \
    'Pillow>=10.0.0,<13'
