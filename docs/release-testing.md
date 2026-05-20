# Release testing

CHANGELOG entries for tagged versions are immutable (see CLAUDE.md). A broken
release forces a patch-version bump even if no functional change is intended.
This document is the gate before pushing a tag.

## Why

`.deb` packages bundle a virtualenv with compiled wheels (faster-whisper,
ctranslate2, pyav) that are pinned to a specific Python minor. The bundled
standalone Python (from [python-build-standalone](https://github.com/astral-sh/python-build-standalone))
keeps us decoupled from the system Python, but it also means more moving parts
in the package that can drift silently. Hence the layered test plan below.

## Layered checks

Run them in order. Each layer is fast enough to be the default; the slower
layers below are only needed when the cheaper layers have surprises.

### 1. Local docker smoke (~2 min, run before every release)

Build the wheel + .debs, install them inside a clean Ubuntu/Debian container,
and run `tapeback --version` and `tapeback status`. This catches the most
common failure modes: broken shebangs, wrong venv paths, missing system
dependencies, broken hooks.

```bash
uv build
./scripts/build-deb.sh dist/tapeback-*.whl

# Base package on current Ubuntu LTS:
docker run --rm -v $PWD/dist:/dist ubuntu:26.04 bash -c '
    apt-get update -qq && apt-get install -y -qq /dist/tapeback_*.deb
    tapeback --version
    tapeback status
'

# Same on previous LTS releases + both Debian stable lines:
for img in ubuntu:24.04 ubuntu:22.04 debian:13 debian:12; do
    docker run --rm -v $PWD/dist:/dist "$img" bash -c '
        apt-get update -qq && apt-get install -y -qq /dist/tapeback_*.deb && tapeback --version
    '
done
```

Optional extras (these run pip install during postinst, ~30 s + network):

```bash
docker run --rm -v $PWD/dist:/dist ubuntu:26.04 bash -c '
    apt-get update -qq
    apt-get install -y -qq /dist/tapeback_*.deb /dist/tapeback-llm_*.deb
    /opt/tapeback/venv/bin/python -c "import anthropic, openai"
'
```

**Do not use `ubuntu:24.10` or other EOL releases** — their apt repositories
are removed, `apt-get update` fails, and the .deb dependency resolution can't
complete. Stick to actively-supported releases: current LTS (26.04), previous
LTS (24.04), current interim (25.10 while supported), and current Debian stable
(13).

### 2. CI gate on every PR (automatic)

`.github/workflows/deb-e2e.yml` runs the same docker smoke on a 5-image matrix
(Ubuntu 22.04 / 24.04 / 26.04, Debian 12 / 13) for any PR that touches
`packaging/`, `scripts/build-deb.sh`, `pyproject.toml`, or `src/`. A regression
in the build pipeline never reaches a release tag — the PR turns red first.

### 3. Pre-release tag (optional, for risky changes)

For changes that touch the venv layout, bundled Python, or hook scripts,
publish a release-candidate tag first:

```bash
# In CHANGELOG.md, the [0.9.3] section is fine — rc1 reuses it.
git tag v0.9.3-rc1
git push origin v0.9.3-rc1
```

The publish workflow runs on any `v*` tag, including pre-release semver. The
GitHub Release will be marked as pre-release automatically (`v0.9.3-rc1` is
non-final per SemVer). Install the artifact on a real Ubuntu/Debian machine
and run through the manual checklist below.

If rc1 is happy, push the final `v0.9.3` (no code changes needed; the rc1 .deb
contents are re-built from the same commit).

### 4. Manual acceptance (run once per minor, or when behavior changes)

On a fresh Ubuntu/Debian VM or real machine:

- `sudo apt install ./tapeback_*.deb` → `tapeback --version` prints the tag
- `tapeback start test-meeting` on a clip with known speakers → markdown
  written to vault, audio file linked
- `tapeback tray` on GNOME Wayland WITHOUT the AppIndicator extension → the
  warning hint is printed to stderr; pystray still starts (icon may be inert,
  that's expected)
- After `sudo apt install gnome-shell-extension-appindicator` + enabling the
  extension + re-login → tray icon menu actually responds
- `sudo apt install ./tapeback-llm_*.deb` → `/opt/tapeback/venv/bin/python -c
  "import anthropic, openai"` succeeds
- `sudo apt remove tapeback-llm` → anthropic/openai uninstalled from the venv;
  `sudo apt remove tapeback` removes everything

If any of these fails on the final tag → bump the patch version, fix, retag.
Don't amend the released tag.
