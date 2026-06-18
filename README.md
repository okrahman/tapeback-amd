# tapeback

Local meeting recorder for Linux. Records system audio + microphone via
PipeWire/PulseAudio, transcribes with Whisper, identifies speakers, saves
Markdown to your Obsidian vault. Everything runs on your machine, no cloud
services or API calls needed for transcription.

Works with any video call platform: Google Meet, Zoom, Teams, Telegram, Discord, Slack huddles.

![tapeback in Obsidian](docs/obsidian-screenshot.png)

## Features

- **Live transcription** (opt-in): read the transcript while the meeting is still going — Whisper transcribes in the background every 60 seconds (set `TAPEBACK_LIVE=true`)
- **Platform-agnostic**: captures OS-level audio, works with any app
- **Local transcription**: faster-whisper on CPU or CUDA GPU
- **Speaker diarization**: pyannote identifies who said what
- **Stereo channel separation**: your mic (left) vs. others (right) for accurate "You" attribution
- **Obsidian-native output**: Markdown with YAML frontmatter, wikilinks to audio files
- **LLM summarization**: Anthropic, OpenAI, Groq, Gemini, DeepSeek, OpenRouter, Qwen (with automatic provider fallback)
- **System tray**: start/stop recording from the tray icon, no terminal needed
- **CLI-first**: `tapeback start`, Ctrl+C to stop, done

tapeback is modular — the base package handles recording and transcription.
Speaker diarization and LLM summaries are optional and installed separately.

## Installation

### Arch Linux (AUR)

```bash
yay -S tapeback                              # recording + transcription
yay -S tapeback tapeback-tray                # + system tray icon
yay -S tapeback tapeback-llm                 # + LLM summaries
yay -S tapeback tapeback-diarize             # + speaker diarization
yay -S tapeback tapeback-tray tapeback-llm tapeback-diarize  # everything
```

All system dependencies (ffmpeg, PipeWire) are installed automatically.

### Ubuntu / Debian (.deb)

Pre-built `.deb` packages are attached to every
[GitHub Release](https://github.com/yastcher/tapeback/releases). Pick a version
(see the releases page for the current one), download, install:

```bash
wget https://github.com/yastcher/tapeback/releases/download/v0.9.5/tapeback_0.9.5_amd64.deb
sudo apt install ./tapeback_0.9.5_amd64.deb

# Optional extras:
sudo apt install ./tapeback-tray_0.9.5_all.deb       # tray icon
sudo apt install ./tapeback-llm_0.9.5_all.deb        # LLM summaries
sudo apt install ./tapeback-diarize_0.9.5_all.deb    # speaker diarization
```

The base package bundles its own Python interpreter (from
[python-build-standalone](https://github.com/astral-sh/python-build-standalone))
so it works on any modern Ubuntu or Debian regardless of the system Python
version. Only system dependencies are `ffmpeg` and `pulseaudio-utils` (for
`parecord` / `pactl`).

### pip / uv

Install system dependencies first:

```bash
# Arch / Manjaro
sudo pacman -S python uv ffmpeg pipewire-pulse

# Ubuntu / Debian
sudo apt install python3 pipx ffmpeg pulseaudio-utils

# Fedora
sudo dnf install python3 pipx ffmpeg pipewire-pulseaudio
```

Then install tapeback:

```bash
uv tool install tapeback                          # recording + transcription
uv tool install "tapeback[tray]"                  # + system tray icon
uv tool install "tapeback[llm]"                   # + LLM summaries
uv tool install "tapeback[diarize]"               # + speaker diarization
uv tool install "tapeback[tray,llm,diarize]"      # everything
```

### pipx or Nix

```bash
# pipx
pipx install tapeback
pipx install "tapeback[tray,llm,diarize]"         # everything

# Nix
nix run github:yastcher/tapeback                  # basic
nix run github:yastcher/tapeback#tray             # + system tray icon
nix run github:yastcher/tapeback#llm              # + LLM summaries
nix run github:yastcher/tapeback#diarize          # + speaker diarization
nix run github:yastcher/tapeback#full             # everything
```

## Quick start

```bash
tapeback start                     # start recording, Ctrl+C to stop
```

That's it. The transcript is saved to `~/tapeback/meetings/`.

To save to your Obsidian vault instead:

```bash
mkdir -p ~/.config/tapeback
echo 'TAPEBACK_VAULT_PATH=~/Documents/obsidian/vault' > ~/.config/tapeback/.env
```

## System tray

Run without a terminal — right-click the tray icon to start/stop recording:

```bash
tapeback tray
```

Icon color shows the current state:
**gray** = idle, **red** = recording, **orange** = processing.

To autostart on login, create `~/.config/autostart/tapeback-tray.desktop`:

```ini
[Desktop Entry]
Name=tapeback
Exec=tapeback tray
Type=Application
X-GNOME-Autostart-enabled=true
```

tapeback's tray speaks the StatusNotifierItem D-Bus protocol directly — no
pystray, no GTK, no XEmbed. KDE Plasma, Hyprland (with waybar), and Sway
(with waybar/eww) display it out of the box on both X11 and Wayland.

### GNOME

GNOME Shell does not display SNI items natively. Install the
**AppIndicator Support** extension (one-time setup, also needed by Slack,
Dropbox, etc.):

```bash
# Ubuntu / Debian
sudo apt install gnome-shell-extension-appindicator
# Fedora
sudo dnf install gnome-shell-extension-appindicator
```

Open the **Extensions** app, enable **Ubuntu AppIndicators** (or
**AppIndicator and KStatusNotifierItem Support**), log out + back in.
`tapeback tray` prints this hint to stderr on startup if it detects an
affected session. See [issue #3](https://github.com/yastcher/tapeback/issues/3)
for background.

## Speaker diarization

Speaker diarization identifies who said what in the recording. Without it,
tapeback uses stereo channels: your mic is labeled "You", everything else
is labeled "Other".

To enable diarization, you need a HuggingFace token with access to pyannote models:

1. Create account at [huggingface.co](https://huggingface.co)
2. Accept license at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Accept license at [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
4. Create token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
5. Add to `~/.config/tapeback/.env`:

```bash
TAPEBACK_HF_TOKEN=hf_your_token_here
```

> First run downloads the pyannote model (~1 GB). An NVIDIA GPU is strongly
> recommended — diarization on CPU is very slow.

## LLM summarization

After transcription, tapeback can add a brief summary, action items, and key
decisions using an LLM.

Set an API key for at least one provider in `~/.config/tapeback/.env`:

```bash
TAPEBACK_LLM_PROVIDER=gemini
GEMINI_API_KEY=...
```

| Provider | Env var | Default model |
|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 |
| `openai` | `OPENAI_API_KEY` | gpt-4o |
| `groq` | `GROQ_API_KEY` | llama-3.3-70b-versatile |
| `gemini` | `GEMINI_API_KEY` | gemini-2.5-flash |
| `openrouter` | `OPENROUTER_API_KEY` | google/gemini-2.5-flash:free |
| `deepseek` | `DEEPSEEK_API_KEY` | deepseek-chat |
| `qwen` | `DASHSCOPE_API_KEY` | qwen-turbo |

If the primary provider fails, tapeback automatically tries the next available
provider (any provider with an API key set).

## CLI reference

```
tapeback start [NAME]              Start recording (Ctrl+C to stop)
tapeback stop                      Stop recording from another terminal
tapeback tray                      System tray icon
tapeback process <FILE> [--name N] Transcribe an existing audio file
tapeback summarize <FILE>          Add LLM summary to transcript
tapeback status                    Show recording status and settings
```

```bash
tapeback start --no-live           # one-shot override: skip live transcription even if TAPEBACK_LIVE=true
tapeback start --no-diarize        # skip speaker identification
tapeback start --no-summarize      # skip LLM summary
tapeback process meeting.mp3 --name "weekly-standup"
tapeback summarize notes.md --provider gemini --model gemini-2.5-pro
```

## Output format

Stereo recordings produce two transcript sections:
- **Transcript** — raw Whisper output with channel-based labels (You / Other)
- **Diarized Transcript** — speaker-identified output (You / Speaker 1 / Speaker 2 / ...)

Words where Whisper is uncertain (probability < 0.5) are shown in *italics*.

```markdown
---
date: 2026-03-23
time: "14:30"
duration: "01:23:45"
language: en
audio: "[[attachments/audio/2026-03-23_14-30-00.wav]]"
tags:
  - meeting
  - transcript
---

## Summary

Brief overview of the meeting.

### Action Items

- [ ] **You:** Send the report by Friday
- [ ] **Speaker 1:** Review the PR

### Key Decisions

- Use PostgreSQL instead of MongoDB

---
# Meeting 2026-03-23 14:30

**Duration:** 1h 23m 45s | **Language:** en

---

## Transcript

[00:00:01] **You:** Hello, let's start with the *backend* changes.

[00:01:23] **Other:** Sure, I have the slides ready.

---

## Diarized Transcript

[00:00:01] **You:** Hello, let's start with the *backend* changes.

[00:01:23] **Speaker 1:** Sure, I have the slides ready.

[00:02:45] **Speaker 1:** Can we move on to the frontend?
```

<details>
<summary><h2>Configuration reference</h2></summary>

All settings via environment variables (prefix `TAPEBACK_`) or
`~/.config/tapeback/.env` file.

### Core

| Variable | Default | Description |
|---|---|---|
| `TAPEBACK_VAULT_PATH` | `~/tapeback` | Path to output directory (Obsidian vault) |
| `TAPEBACK_MEETINGS_DIR` | `meetings` | Subdirectory for meeting notes |
| `TAPEBACK_ATTACHMENTS_DIR` | `attachments/audio` | Subdirectory for audio files |

### Transcription

| Variable | Default | Description |
|---|---|---|
| `TAPEBACK_WHISPER_MODEL` | `large-v3-turbo` | Whisper model (`tiny`, `base`, `small`, `medium`, `large-v3-turbo`) |
| `TAPEBACK_LANGUAGE` | `auto` | Language code (`auto` for auto-detection, or `en`, `ru`, `fr`, etc.) |
| `TAPEBACK_DEVICE` | `cuda` | `cuda` or `cpu` |
| `TAPEBACK_COMPUTE_TYPE` | `auto` | `auto`, `float16`, `int8`, or `float32` (`auto` → `float16` on CUDA, `int8` on CPU; pin `int8` if your GPU is memory-tight) |
| `TAPEBACK_BEAM_SIZE` | `5` | Whisper beam search width |
| `TAPEBACK_CHUNK_LENGTH` | `7` | Max VAD chunk (seconds) before splitting for Whisper; prevents lost speech after long pauses |
| `TAPEBACK_NO_SPEECH_THRESHOLD` | `0.4` | Whisper silence-rejection threshold (lower = more aggressive; suppresses training-data hallucinations on pauses) |
| `TAPEBACK_LANGUAGE_DETECTION_SEGMENTS` | `1` | Segments probed before deciding the language; raise (e.g. `4`) if a channel that starts silent gets the wrong language |
| `TAPEBACK_MULTILINGUAL` | `false` | Per-segment language detection for mixed-language recordings (code-switching). Less stable than a fixed `TAPEBACK_LANGUAGE` |
| `TAPEBACK_HALLUCINATION_SILENCE_THRESHOLD` | *(off)* | Seconds; skip silent gaps longer than this when a hallucination is detected (e.g. `2.0`) |
| `TAPEBACK_PAUSE_THRESHOLD` | `1.0` | Seconds; split segments on silence gaps >= this |

### Live transcription

| Variable | Default | Description |
|---|---|---|
| `TAPEBACK_LIVE` | `false` | Enable live transcription during recording (opt-in; competes with the post-recording pipeline for GPU memory on small cards) |
| `TAPEBACK_LIVE_INTERVAL` | `60` | Seconds between transcription cycles |
| `TAPEBACK_LIVE_OVERLAP` | `2.0` | Seconds of overlap between chunks |
| `TAPEBACK_LIVE_MIN_CHUNK` | `5.0` | Minimum new audio (seconds) to trigger transcription |

### Audio

| Variable | Default | Description |
|---|---|---|
| `TAPEBACK_MONITOR_SOURCE` | `auto` | PulseAudio monitor source name |
| `TAPEBACK_MIC_SOURCE` | `auto` | PulseAudio mic source name |
| `TAPEBACK_SAMPLE_RATE` | `48000` | Recording sample rate |

### Speaker diarization

| Variable | Default | Description |
|---|---|---|
| `TAPEBACK_DIARIZE` | `true` | Enable speaker diarization |
| `TAPEBACK_HF_TOKEN` | *(empty)* | HuggingFace token ([setup](#speaker-diarization)) |
| `TAPEBACK_MAX_SPEAKERS` | *(auto)* | Maximum number of speakers |
| `TAPEBACK_SPECTRAL_MERGE_THRESHOLD` | `0.96` | Spectral speaker merging (0 = off; lower merges more aggressively) |

### LLM summarization

| Variable | Default | Description |
|---|---|---|
| `TAPEBACK_SUMMARIZE` | `true` | Enable LLM summarization |
| `TAPEBACK_LLM_PROVIDER` | `anthropic` | Primary provider ([list](#llm-summarization)) |
| `TAPEBACK_LLM_API_KEY` | *(empty)* | API key (or use provider-specific env var) |
| `TAPEBACK_LLM_MODEL` | *(provider default)* | Override model name |

</details>

## Troubleshooting

### GPU transcription falls back to CPU on CUDA 13 systems

If the log shows `Warning: CUDA runtime error, falling back to CPU: Library libcublas.so.12 is not found`,
your system has CUDA 13 (e.g. recent Arch) but faster-whisper's ctranslate2 backend is
built against CUDA 12 and can't find `libcublas.so.12` / `libcudnn.so.9`. Diarization
(PyTorch) still uses the GPU, but transcription drops to slow CPU.

Install the CUDA 12 runtime libraries — tapeback preloads them automatically:

```bash
uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

They install alongside the CUDA 13 libraries without conflict. No `LD_LIBRARY_PATH`
needed: when `TAPEBACK_DEVICE=cuda` (the default), tapeback finds and preloads them
on startup. If they're not installed, transcription falls back to CPU with the message
above.

### Wrong language or repeated/garbled text on the "You" channel

If the note's `language:` is wrong (e.g. `ja` for an English meeting) or the **You**
channel is full of repeats (`Do you hear me? Do you hear me?…`) or foreign script, the
mic channel started with silence while you were listening, and Whisper guessed the
language from that silence and hallucinated.

Fixes, in order of reliability:

- **Pin the language** if you know it: `TAPEBACK_LANGUAGE=ru` (or `en`, …). English
  terms inside another language still transcribe fine.
- **Probe more speech** before deciding: `TAPEBACK_LANGUAGE_DETECTION_SEGMENTS=4`.
- **Mixed-language meetings** (real code-switching): `TAPEBACK_MULTILINGUAL=true`.
- **Suppress silence hallucinations**: `TAPEBACK_HALLUCINATION_SILENCE_THRESHOLD=2.0`.

## Uninstall

```bash
# Arch Linux
yay -R tapeback tapeback-tray tapeback-diarize tapeback-llm

# pip / uv
uv tool uninstall tapeback

# Remove cached ML models (~2-5 GB)
# Skip if you use HuggingFace for other projects
rm -rf ~/.cache/huggingface/
```

## Roadmap

- **Speaker profiles**: learn and remember recurring speakers across meetings
- **Multi-language meetings**: detect and handle language switches mid-meeting
- **Windows support**: WASAPI loopback capture

## Support

If you find tapeback useful, consider a small donation:

| USDT (TRC-20) | ADA (Cardano) |
|:-:|:-:|
| <img src="docs/qr-usdt.png" width="180"> | <img src="docs/qr-ada.png" width="180"> |
| `TAECw9FebnoSN2n3H2Fk9Bv5aA8fwpCuBB` | `addr1q9tqg2g8wxpxawsrvea84lms3ampuda0ygzawuxq77sxwr48mxj2vq2rzd4nsmhpdhy6lftp30tz78tetzr29mtvkqmsskrmp7` |

## Links

- [Changelog](CHANGELOG.md)
- [Deploy](DEPLOY.md)
- [CI/CD](.github/workflows/)

## License

Apache-2.0. See [LICENSE](LICENSE).
