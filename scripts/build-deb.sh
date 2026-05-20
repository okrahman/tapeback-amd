#!/usr/bin/env bash
# Build Debian/Ubuntu .deb packages for tapeback via nfpm.
#
# The .deb bundles a standalone Python interpreter (from astral-sh's
# python-build-standalone, downloaded via `uv python install`) so the package
# does NOT depend on the target system having any specific Python version.
# This avoids the python3.13 / python3.14 minor-version coupling that bites us
# because faster-whisper / ctranslate2 / pyav wheels include compiled .so files
# tagged for a specific cpython minor.
#
# Layout inside the .deb:
#   /opt/tapeback/python/         standalone CPython (~75 MB, RPATH-relative)
#   /opt/tapeback/venv/           virtualenv with tapeback + deps installed
#   /usr/bin/tapeback             wrapper that exec's /opt/tapeback/venv/bin/tapeback
#
# Usage:
#   ./scripts/build-deb.sh                 # install tapeback from PyPI
#   ./scripts/build-deb.sh dist/*.whl      # install from a local wheel (faster, no network)
#
# Smoke-test the result:
#   docker run --rm -v $PWD/dist:/dist ubuntu:26.04 bash -c \
#     'apt-get update -qq && apt-get install -y -qq /dist/tapeback_*.deb && tapeback --version'
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VERSION="$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)"
if [ -z "$VERSION" ]; then
    echo "Error: could not read version from pyproject.toml" >&2
    exit 1
fi
export VERSION

PY_MINOR="3.13"
BUILD_ROOT="$REPO_ROOT/build/opt/tapeback"
PY_DEST="$BUILD_ROOT/python"
VENV_DEST="$BUILD_ROOT/venv"
DIST_DIR="$REPO_ROOT/dist"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed. https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

if ! command -v nfpm >/dev/null 2>&1; then
    echo "Error: nfpm is not installed. See https://nfpm.goreleaser.com/install/" >&2
    exit 1
fi

WHEEL_SPEC="${1:-tapeback==$VERSION}"

echo "==> Building tapeback $VERSION .deb packages"
echo "    Wheel/spec: $WHEEL_SPEC"

# Resolve uv-managed standalone Python (downloads on first run).
uv python install "$PY_MINOR" >/dev/null
SYSTEM_PY="$(uv python find "$PY_MINOR")"
SYSTEM_PY_ROOT="$(dirname "$(dirname "$SYSTEM_PY")")"
echo "    Python:     $("$SYSTEM_PY" --version) (from $SYSTEM_PY_ROOT)"

# Clean previous venv build but keep dist/ (wheel artifact may live there)
rm -rf "$REPO_ROOT/build"
mkdir -p "$BUILD_ROOT" "$DIST_DIR"

echo "==> Bundling standalone Python into $PY_DEST"
cp -a "$SYSTEM_PY_ROOT" "$PY_DEST"

# Trim build artifacts we don't need at runtime (headers + a few caches).
# Stdlib + libpython stay — they're what makes this self-contained.
rm -rf "$PY_DEST/include"
find "$PY_DEST" -type d -name __pycache__ -prune -exec rm -rf {} +

echo "==> Creating venv at $VENV_DEST"
"$PY_DEST/bin/python3.13" -m venv "$VENV_DEST"
"$VENV_DEST/bin/pip" install --quiet --upgrade pip
"$VENV_DEST/bin/pip" install --quiet "$WHEEL_SPEC"

echo "==> Rewriting paths for /opt/tapeback install location"
# Replace venv's python symlinks with relative ones so they resolve correctly
# from either the build dir or /opt/tapeback after install.
ln -sf "../../python/bin/python3.13" "$VENV_DEST/bin/python"
ln -sf "python" "$VENV_DEST/bin/python3"
ln -sf "python" "$VENV_DEST/bin/python3.13"

# Rewrite shebangs in venv/bin/* from build path to /opt/tapeback/venv/bin/python.
find "$VENV_DEST/bin" -type f -exec \
    sed -i "1s|^#!.*/build/opt/tapeback/venv/bin/python[^ ]*|#!/opt/tapeback/venv/bin/python|" {} \;

# pyvenv.cfg points at the python that built it — rewrite to bundled path.
sed -i "s|$PY_DEST|/opt/tapeback/python|g" "$VENV_DEST/pyvenv.cfg"

echo "==> Trimming bytecode caches"
find "$VENV_DEST" -type d -name __pycache__ -prune -exec rm -rf {} +

echo "==> Running nfpm"
for cfg in \
    packaging/deb/nfpm-tapeback.yaml \
    packaging/deb/nfpm-tapeback-diarize.yaml \
    packaging/deb/nfpm-tapeback-llm.yaml \
    packaging/deb/nfpm-tapeback-tray.yaml
do
    echo "    - $cfg"
    nfpm package -f "$cfg" -p deb -t "$DIST_DIR"
done

echo ""
echo "Done. Artifacts in $DIST_DIR:"
ls -1 "$DIST_DIR"/tapeback*.deb 2>/dev/null || true
