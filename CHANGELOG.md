# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.6] — 2026-06-17

### Added
- Per-stage timing in processing output — merge, split, load model, transcribe, diarize, and summarize each report how long they took.

### Changed
- Faster post-recording processing: dropped a redundant ffmpeg pass that mixed both channels into a mono file the dual-channel pipeline never used.

### Fixed
- CPU fallback during transcription and diarization now triggers only on real CUDA / out-of-memory / cuBLAS errors; unrelated failures surface instead of being masked by a slow CPU retry.

## [0.9.5] — 2026-05-21

### Fixed
- Tray icon now actually appears on GNOME Wayland (with the AppIndicator extension installed).
v0.9.4 registered with the watcher but the icon stayed invisible because our `WindowId` property was exposed as `u` (unsigned int) — the SNI spec mandates `i` (signed int), and GNOME Shell silently rejects mismatched signatures (`type u does not match expected type i`).
Fixed to `i`; regression test locks the signature so it can't regress.
- Added the AppIndicator accessibility-description SNI extensions (`IconAccessibleDesc`, `AttentionAccessibleDesc`, `OverlayIconAccessibleDesc`). Newer hosts query these and dbus-next raised `DBusError(UNKNOWN_PROPERTY)` when we didn't implement them; the error is gone now.
- `tapeback tray` no longer prints the AppIndicator hint twice. v0.9.4 both logged the message and printed it to stderr; the duplicate is removed (only the logger.warning remains).

### Changed
- AppIndicator hint message is now distro-neutral: lists install commands for Ubuntu/Debian (`apt`), Fedora/RHEL (`dnf`), and Arch (`yay`) instead of only Ubuntu.

## [0.9.4] — 2026-05-21

### Fixed
- `tapeback tray` on modern desktops (GNOME, KDE Plasma): rewrote the tray icon on top of [dbus-next](https://github.com/altdesktop/python-dbus-next) and a custom StatusNotifierItem / DBusMenu implementation. The previous pystray-based path defaulted to the legacy XEmbed protocol, which GNOME/KDE no longer provide on Wayland — the icon either failed to dock at all (`AssertionError: _systray_manager is None`) or appeared as a dead grey circle whose menu didn't open. pystray's AppIndicator backend needs system PyGObject (`gi`), which the bundled `.deb` Python can't access. The new SNI-direct implementation works in the bundled venv with no system C extensions. KDE Plasma works out of the box; GNOME still needs the AppIndicator Support extension installed (one-time, GNOME-side limitation, not ours).
- `.deb` build: switched from `uv python install` + `cp -a` to direct download of [python-build-standalone](https://github.com/astral-sh/python-build-standalone) by pinned date/version. The previous approach depended on `uv python find`'s layout, which differed between local and CI environments — the CI-built `.deb` had a broken bundled Python tree and `tapeback --version` failed with `exec: not found`. Direct download is byte-deterministic and fails fast if the tarball layout changes.

### Changed
- `tapeback-tray` optional-dependency now pulls in `dbus-next` instead of `pystray` + `Pillow`. The tray .deb's postinstall hook reflects the same.
- README install snippet for `.deb`: replaced the dynamic-version `curl | grep` snippet with a plain `VERSION=X.Y.Z` + `wget` + `apt install` form. Simpler to read, simpler to copy-paste.

### Added
- `.github/workflows/deb-e2e.yml` — PR-time end-to-end .deb smoke on a 5-image matrix (Ubuntu 22.04 / 24.04 / 26.04, Debian 12 / 13). Catches packaging regressions before they reach a release tag.
- `scripts/check-workflow-pins.py` + new CI step — validates every SHA-pinned GitHub Action against the GitHub API. Prevents a hallucinated 40-char hex from quietly slipping through to a release runner.
- `docs/release-testing.md` — layered checklist (local docker smoke → CI PR gate → optional `v0.9.X-rcN` tag → manual acceptance) so the .deb path is gated before every release.

## [0.9.3] — 2026-05-20

### Fixed
- Tray on GNOME Wayland: when `tapeback tray` starts on a GNOME-family Wayland session, it now prints an actionable warning explaining that the AppIndicator Support extension is required (the legacy XDG StatusNotifier protocol was removed upstream in GNOME 45+, so without the extension the icon appears but the menu does not respond). KDE Plasma Wayland is unaffected. Closes [#3](https://github.com/yastcher/tapeback/issues/3).

### Added
- Debian/Ubuntu `.deb` packages: built in CI via [nfpm](https://nfpm.goreleaser.com/) and attached to every GitHub Release. Mirrors the Arch split: `tapeback`, `tapeback-tray`, `tapeback-llm`, `tapeback-diarize`. Install with `sudo apt install ./tapeback_*.deb`. The base `.deb` bundles a standalone Python 3.13 (~75 MB extracted) from [python-build-standalone](https://github.com/astral-sh/python-build-standalone), so it works on any modern Ubuntu/Debian regardless of the system Python — including Ubuntu 26.04 LTS, which ships only Python 3.14 and would otherwise break our 3.13-tagged compiled wheels (faster-whisper, ctranslate2, pyav).
- `tapeback --version` flag — reads the installed package version via `importlib.metadata`.

## [0.9.2] — 2026-05-04

### Fixed
- `TAPEBACK_COMPUTE_TYPE=auto` always picked `int8` on 4 GiB cards: the 4096 MiB threshold could never be reached because the card's total VRAM is right at that limit.
`large-v3-turbo` actually needs ~1.5 GiB in float16, so the auto-quantize was spurious.
Auto now resolves purely from device (`cuda` → `float16`, `cpu` → `int8`); pin `TAPEBACK_COMPUTE_TYPE=int8` explicitly if your GPU is genuinely memory-tight.

### Changed
- Live transcription is now opt-in (`TAPEBACK_LIVE` defaults to `false`). Mid-recording transcription competes with the post-recording pipeline for GPU memory on small cards (4 GiB), causing long stalls. Set `TAPEBACK_LIVE=true` to re-enable; `--no-live` still works as a one-shot override.

## [0.9.1] — 2026-05-03

### Fixed
- CPU fallback didn't trigger when faster-whisper raised `RuntimeError` (e.g. `Library libcublas.so.12 is not found`) synchronously from `transcribe()` — eager language detection raises before yielding the segment generator, so the previous fallback (which only wrapped iteration) missed it. Both call-time and iteration-time CUDA failures now fall back to CPU and the recording survives.

## [0.9.0] — 2026-04-20

### Security
- `hf_token` and `llm_api_key` now stored as `pydantic.SecretStr` — prevents leakage through `repr(settings)`, tracebacks, or `model_dump_json()` output
- Path-traversal guard on `tapeback process --name`: session names are now validated (only `[\w-]+` allowed) before being used as vault path components; vault I/O also verifies the resolved destination stays under `vault_path`
- Atomic markdown writes: `save_markdown_to_vault` now uses write-temp + rename so Obsidian can't read a half-written transcript if tapeback crashes mid-write
- Upper version bounds pinned on all dependencies (`<2`, `<3`, etc.) to prevent unreviewed major-version upgrades from breaking the build

### Added
- Live transcription: Whisper transcribes audio in real-time during recording, writing a live markdown file to the vault that can be opened mid-meeting
- `--no-live` CLI flag to disable live transcription and use the old post-recording-only mode
- `TAPEBACK_LIVE` setting (default `true`) — enable/disable live transcription
- `TAPEBACK_LIVE_INTERVAL` setting (default `60`) — seconds between transcription cycles
- `TAPEBACK_LIVE_OVERLAP` setting (default `2.0`) — seconds of overlap between chunks for seamless transitions
- `TAPEBACK_LIVE_MIN_CHUNK` setting (default `5.0`) — minimum new audio (seconds) before triggering a transcription cycle
- `TAPEBACK_NO_SPEECH_THRESHOLD` setting (default `0.4`) — Whisper silence-rejection threshold; lower values suppress training-data hallucinations like "Субтитры DimaTorzok" on long pauses

### Fixed
- CPU fallback lost auto language detection: passed `"auto"` string to Whisper instead of `None`, causing errors on non-English transcripts
- Duplicate "## Diarized Transcript" section when diarization was skipped (via `--no-diarize` or missing HF token) — both sections were identical; now only "## Transcript" is rendered
- Whisper hallucinations on long pauses (e.g. "Субтитры DimaTorzok", "Продолжение следует") — `no_speech_threshold` now set to `0.4` (stricter than Whisper's default `0.6`)

### Changed
- Settings now fail-fast on invalid values: thresholds must be in `[0, 1]`, `pause_threshold` / `live_overlap` must be non-negative, `live_interval` / `live_min_chunk` must be positive, and `live_min_chunk` must be ≤ `live_interval` when live transcription is enabled — surfaced via `pydantic.ValidationError` at `get_settings()` instead of silent mis-behaviour deep in the pipeline
- Internal refactor: extracted `tapeback._gpu` (CUDA memory helper), `tapeback._lazy` (single lazy-load site for `Transcriber`), and `tapeback.speaker_merge` (spectral clustering) — no user-visible change, but `diarizer.py` and `channel.py` are now under the 500-line limit and no longer need `PLR0912` / `PLR0915` ignores
- Default language changed from `en` to `auto` — Whisper now auto-detects the spoken language
- `tapeback start` now detects when recording stops (e.g. via `tapeback stop`) using a polling loop instead of `signal.pause()`
- `TAPEBACK_CHUNK_LENGTH` default raised from `2` to `7`: 2-second chunks fragment Whisper's context and cause hallucinations and broken sentences on non-English speech; `7` balances context against hallucination risk on long pauses
- Low-confidence word threshold (italic marker) lowered from `0.5` to `0.35`: fewer false positives on English loanwords inside Russian/mixed-language speech

## [0.8.10] — 2026-04-04

### Fixed
- Speaker misattribution: words from one monitor speaker assigned to another; switched from whole-segment majority vote to word-level diarization split
- Interleaved single-word segments during simultaneous speech; consecutive same-speaker segments now consolidated
- False extra speaker from echo/crosstalk: minor speakers (< 15s and < 20% of dominant) absorbed with lower merge threshold (0.92)
- Headphone bleed falsely attributed to "You": crosstalk filter drops mic segments where monitor channel is louder

### Added
- Two-section transcript output: raw Whisper transcript (## Transcript) then diarized (## Diarized Transcript) for comparison
- Low-confidence word marking: Whisper words with probability < 0.5 shown in *italics*
- VRAM pre-check before diarization: skips CUDA when free VRAM < 1500 MiB to avoid slow OOM fallback
- Decomposed `diarizer.py` into `diarizer.py` + `channel.py` for channel-related utilities

### Changed
- `TAPEBACK_CHUNK_LENGTH` default lowered from `15` to `2` for finer VAD granularity

## [0.8.9] — 2026-04-02

### Fixed
- Lost speech after long pauses: Whisper VAD merged all speech chunks into one stream, losing speakers separated by silence; added `chunk_length=15` to split VAD output before transcription

### Added
- `TAPEBACK_CHUNK_LENGTH` setting (default `15s`) — max VAD chunk size before splitting for Whisper

## [0.8.8] — 2026-04-02

### Added
- Auto VRAM detection: `TAPEBACK_COMPUTE_TYPE=auto` (new default) picks `int8` when free GPU memory < 4 GiB, avoiding CUDA OOM with fallback to slow CPU
- `TAPEBACK_SPECTRAL_MERGE_THRESHOLD` setting (default `0.96`) for speaker merging sensitivity

## [0.8.7] — 2026-04-01

### Fixed
- Speaker diarization: two different speakers incorrectly merged into one; raised spectral merging cosine similarity threshold from 0.92 to 0.95
- PyAV `UnicodeDecodeError` crash on non-English locales: `os.environ` alone doesn't change the C locale after Python startup; added `locale.setlocale(LC_MESSAGES, "C")` to actually switch glibc's `strerror_r()` output to ASCII

## [0.8.6] — 2026-04-01

### Fixed
- PyAV `UnicodeDecodeError` crash on non-English locales (e.g. Russian): set `LC_MESSAGES=C` to force ASCII error messages from `strerror_r()`

## [0.8.5] — 2026-03-31

### Added
- AUR package `tapeback-tray`: system tray icon as separate meta-package
- `scripts/aur-publish.sh` now publishes all 4 AUR packages (added tapeback-tray)

### Changed
- Consolidated unit tests into integration flow tests, moved shared fixtures to `tests/fixtures.py`

## [0.8.4] — 2026-03-30

### Added
- System tray icon (`tapeback tray`): start/stop recording from the tray, no terminal needed
- `[tray]` optional extra: `uv pip install tapeback[tray]` (pystray + Pillow)

### Changed
- Extracted `LLMProvider` type alias from inline Literal in settings.py
- Replaced monkey-patched `Exception.status_code` in tests with proper `_HttpError` class
- Removed all `[[tool.ty.overrides]]` sections from pyproject.toml — fixed root causes instead
- Moved fixtures to `tests/fixtures.py`, reduced local imports in `pipeline.py`
- Replaced `assert` in summarizer with explicit `RuntimeError` check
- Diarizer: replaced `**kwargs` dispatch with explicit `_run_pipeline()` method

## [0.8.3] — 2026-03-29

### Fix
- install from AUR now worked

## [0.8.2] — 2026-03-29

### Added
- `scripts/aur-publish.sh`: one-command AUR update for all 3 packages (tapeback, tapeback-llm, tapeback-diarize)

### Changed
- `scripts/release.sh` now shows AUR publish step in next-steps output

## [0.8.1] — 2026-03-28

### Changed
- AUR publishing workflow documented in release process

## [0.8.0] — 2026-03-27

### Added
- `scripts/release.sh`: version bump across pyproject.toml, PKGBUILD
- AUR packages: `tapeback-diarize`, `tapeback-llm` (optional extras as separate packages)
- Version validation step in publish.yml (tag must match pyproject.toml)
- Lint + tests run before PyPI publish

### Changed
- **PyPI package renamed to `tapeback`**, CLI command `tapeback`, Python module renamed to `tapeback`
- CLI command renamed from `meetrec` → `echo-vault` → `tapeback` (entry point, help texts, temp dirs, state dir)
- Speaker diarization (pyannote/torch) moved to optional dependency: `tapeback[diarize]`
- LLM SDKs (anthropic/openai) moved to optional dependency: `tapeback[llm]`
- Base install no longer requires PyTorch or LLM SDKs (~2 GB smaller)
- Monitor channel segments default to "Other" speaker when diarization is not available
- PKGBUILD rewritten with venv-based install (Python deps from PyPI, system deps from pacman)
- Nix flake: extras variants (`#llm`, `#diarize`, `#full`) via `nix run`

### Security
- GitHub Actions pinned to commit SHA (prevents supply-chain tag hijacking)
- CI workflow: explicit `permissions: contents: read`
- Publish workflow: type check step added, awk regex dots escaped in changelog extraction

## [0.7.0] — 2026-03-26

### Added
- LLM provider fallback chain: if primary provider fails, tries next available provider
- Spectral similarity speaker merging to reduce pyannote over-segmentation of single speakers
- PyPI publishing via GitHub Actions (Trusted Publisher)
- GitHub Release automation with changelog extraction
- AUR PKGBUILD (`packaging/PKGBUILD`)
- Nix flake (`flake.nix`)
- VHS demo script (`packaging/demo.tape`)

### Changed
- License changed from proprietary to Apache-2.0
- Minimum Python version lowered from 3.14 to 3.13
- Type checker switched from mypy to ty (Astral)
- `DEFAULT_MODELS` moved to `settings.py` as single source of truth
- Default Gemini model updated to `gemini-2.5-flash` (2.0 deprecated)
- README rewritten with installation guide, configuration reference, CLI examples, roadmap
- CLI help texts expanded with examples and usage guidance

### Fixed
- Markdown fence stripping: LLM responses wrapped in ```json``` now parsed correctly

## [0.6.0] — 2026-03-23

### Added
- Free/cheap LLM providers: Groq, Gemini, OpenRouter, DeepSeek, Qwen
- Retry with exponential backoff on 429/529 rate limit errors (3 retries, 5s→10s→20s)
- `models.py`: domain objects extracted into dedicated module
- `vault.py`: Obsidian vault I/O separated from formatting
- Session name validation (alphanumerics, dashes, underscores only)
- Restrictive permissions (0700) on `/tmp/tapeback` temp directories
- mypy strict mode
- Coverage threshold (85%) enforced in CI

### Changed
- `formatter.py`: pure formatting only, no I/O (moved to `vault.py`)
- `diarizer.py` no longer depends on `transcriber.py`, both import from `models.py`
- `summarizer.py` no longer defines domain models, imports from `models.py`
- Expanded ruff rules: W, G, PLC, PLR, S (security)

## [0.5.0] — 2026-03-22

### Added
- LLM summarization: brief summary, action items, key decisions
- `summarize` CLI command for re-summarizing existing transcripts
- `--no-summarize` flag for `start` and `process` commands
- Anthropic and OpenAI provider support
- Non-fatal summarization, transcript always saved first

### Changed
- Python 3.14 migration (PEP 758 exception syntax)
- Testing Trophy refactoring: more integration tests, fewer unit tests

### Fixed
- Audio device hot-switching: `@DEFAULT_MONITOR@`/`@DEFAULT_SOURCE@` follow device changes

## [0.4.0] — 2026-03-21

### Added
- Pause detection: split segments on word gaps >= threshold (configurable)
- CLI integration tests via CliRunner with mocked ML models
- Coverage reporting enabled by default in pytest

### Changed
- Strict typing refactoring across all modules
- Trophy testing approach: integration tests as primary coverage

### Fixed
- Segment splitting at actual silence gaps in raw mic audio

## [0.3.0] — 2026-03-18

### Added
- Stereo channel support: per-channel transcription (mic + monitor)
- Audio normalization (loudnorm) before transcription
- RMS-based crosstalk filtering to reject Whisper hallucinations on silent channels
- Channel-based speaker attribution (mic=You, monitor=Others)

### Fixed
- GPU memory management: free CUDA memory between transcription and diarization

## [0.2.0] — 2026-03-17

### Added
- Speaker diarization via pyannote, identifies "You" vs remote participants
- Speaker identification using mic/monitor RMS energy ratio
- Dual-channel recording (monitor + mic as separate WAV files)

## [0.1.0] — 2026-03-17

### Added
- Initial release
- Dual-channel audio recording via parecord (PulseAudio/PipeWire)
- Local transcription with faster-whisper (CUDA with CPU fallback)
- Markdown output with YAML frontmatter and `[HH:MM:SS]` timecodes
- Obsidian vault integration: saves audio + markdown
- pydantic-settings configuration with `TAPEBACK_` prefix
- `start`, `stop`, `process`, `status` CLI commands
