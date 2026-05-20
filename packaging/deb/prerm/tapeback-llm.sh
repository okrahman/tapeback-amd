#!/bin/sh
set -e
if [ "$1" = "remove" ] || [ "$1" = "purge" ]; then
    /opt/tapeback/venv/bin/pip uninstall --quiet -y anthropic openai || true
fi
