#!/bin/sh
set -e
/opt/tapeback/venv/bin/pip install --quiet --upgrade \
    'dbus-next>=0.2.3,<1'
