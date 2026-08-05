# Spec: transcription speed and quality

Status: stages 1a and 1b landed; stages 1c / 2 / 3a / 3 / 4 pending.

## Context

Two reported symptoms:

1. **Runs never finish.** A real run on 2026-07-30: a 15m55s recording produced
   `Stage 'transcribe mic' took 8203.0s` (2h17m) — for one of two channels. Aborted with Ctrl+C.
2. **Poor quality**, especially on Russian speech carrying English technical terms.

Scale of the problem: **16 of 53 recordings have no transcript at all**, and they are exactly the
longest ones (37, 36, 25 min). Since 2026-06-28, no recording over 20 minutes has been completed.

The goal is to find root causes rather than treat symptoms, decide whether Whisper should be
replaced, and produce a fix plan.

---

## Measurements taken while writing this spec

Hardware: GTX 1650 Ti Mobile 4 GB (Turing, CC 7.5), hard 50 W limit, driver 610.43.03, CUDA 13.3.

- CUDA works: `ctranslate2.get_cuda_device_count() == 1`, `float16` / `int8_float16` supported. The
  cu12 wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) are installed and `preload_cuda_libs()`
  succeeds. **A silent CPU fallback is not the cause of the 8203 s.**
- Under Whisper load: 2428 MiB VRAM, sm 1860–1890 MHz against a 2100 MHz maximum,
  `clocks_event_reasons.active = 0x4` (SW Power Cap), temperature 53 → 86 °C over ~5 minutes.

---

## Root causes: speed

### 1. `chunk_length=7` — the main offender (~4.3x redundant encoder work)

`settings.py:61` sets `chunk_length: int = 7`. In faster-whisper this overwrites the feature
extractor's window (`feature_extractor.py:203-205`: `nb_max_frames = 7*16000/160 = 700`), and the
decode loop advances `seek` by 700 frames per iteration (`transcribe.py:1173-1178`).

But Whisper's encoder is fixed at 30 seconds. Every window is zero-padded back to 3000 frames
before encoding:

```python
segment = pad_or_trim(segment)                      # transcribe.py:1180
def pad_or_trim(array, length: int = 3000, ...):    # audio.py:111
    """Pad or trim the Mel features array to 3000, as expected by the encoder."""
```

So **every encoder pass costs a full 30 s but advances only 7 s of audio** — 4.3x redundant passes,
a multiplier on top of everything else.

**Measured** on `2026-05-03_17-56-45.wav` (145.5 s), large-v3-turbo / float16 / cuda, all other
parameters as in production:

| `chunk_length` | time | RTF | segments | characters |
|---|---|---|---|---|
| **7** (current default) | **116.8 s** | 1.25x | 28 | 1659 |
| **30** | **41.4 s** | **3.51x** | 17 | 1514 |

**2.82x faster from one value**, same content (the character delta comes from less ragged
segmentation), with larger and more natural timecodes. The estimate is conservative: the `30` run
executed on a card already heat-soaked to 86 °C.

`chunk_length` was originally lowered to 7 to fight hallucinations on long pauses
("Субтитры DimaTorzok"). The cure for that is VAD + `no_speech_threshold` + silence gating — all of
which the project already does — not shrinking the encoder window.

### 2. The temperature ladder fires on 7-second fragments

`beam_size=4` times the full `(0.0…1.0)` ladder means up to 6 re-decodes per window. The ladder
itself is necessary (shortening it was verified to push the model into repeat loops), but combined
with cause 1 it triggers on four times as many windows.

### 3. Everything else, by descending contribution

- `word_timestamps=True` is hardcoded (`transcriber.py:163`), adding a DTW alignment pass per
  segment. It is load-bearing (`channel.filter_silent_segments`, `diarizer._resegment_by_words`) and
  cannot simply be removed.
- Two full sequential passes (mic, then monitor) — `transcriber.py:191-194`.
- `batch_size=0` — `BatchedInferencePipeline` is off.
- The `split` stage costs 36 s: `loudnorm` (EBU R128) runs **twice over the full 48 kHz stream** and
  only then `aresample=16000` (`audio.py:113-116`). Moving `aresample` ahead of `loudnorm` cuts the
  work roughly threefold.

### 4. Thermals — real, but secondary

`backlog_profiler/reports/THERMAL-LLM-REPORT.md` documents this machine's behaviour:

- The EC clamp is **persistent**: after chassis saturation it did not release within 15 minutes, and
  only idling clears it.
- Share of samples carrying throttle flags: **93–96 % uncapped** → 37 % (10 min capped) → **0.0 %
  across 76 minutes** with caps plus pacing.
- Its key conclusion: *"speed does NOT depend on the cap"* — the 50 W limit already pins average
  clocks; a cap only shaves peaks, i.e. watts, at no throughput cost.
- Tooling already exists: `scripts/thermal_profile.sh on <MHz>` (nvidia-smi `-lgc` + CPU `no_turbo` +
  RAPL written into both `intel-rapl:0` and `intel-rapl-mmio:0`).

Conclusion for tapeback: **clamp monitoring is worth having as observability, not as a cure.** A
two-hour run will always end up clamped; a 90-second one never will. Remove the 4.3x of redundant
work first, then measure what remains.

### 5. No progress output, so runs get killed by hand

The only line for the entire run was `"Transcribing (this may take a few minutes)..."`
(`pipeline.py:196`). faster-whisper offers `log_progress` (`transcribe.py:259`), which tapeback never
passed. No timeouts, no cancellation, no partial results: on `KeyboardInterrupt` even
`free_gpu_memory()` is skipped (`pipeline.py:228-229` sits on the non-exception path).

Note: `Stage 'transcribe mic' took 8203.0s` is printed from a `finally` block **immediately** — it is
not a slow reaction to Ctrl+C but an honest report of how long the stage had already been running.

---

## Root causes: quality

### 1. The model is `large-v3-turbo`, not `large-v3`

`settings.py:45`. Turbo is a distillation with 4 decoder layers instead of 32; it is noticeably more
prone to repeat loops and weaker on non-English audio.

### 2. No glossary — neither `hotwords` nor `initial_prompt`

`grep` over `src/` returns zero hits. faster-whisper supports `hotwords` (`transcribe.py:1545`).
This is the most direct lever for the "technical terms must survive" requirement.

What its absence costs, from the actual transcripts:

- RAG within a single meeting: `rec` / `Rick` / `Ray` / `REC` / `Wreck` / `racist` / `rails` — six
  spellings of one term
- ONNX → `on an X Runtime`, OpenVINO → `OpenSWIFO`, Excalidraw → `Escalidraw`
- tapeback → `тейп *B*`, `тейпбэком`
- "ML and CV pipeline integration" → `ML and CVS. pipeline integration` / `pipeline. immigration`

### 3. `language="auto"` with `language_detection_segments=1`

The language is decided from the first segment of each channel, independently for mic and monitor.
Hence notes with `language: en` in the front matter containing raw Cyrillic:
`but I *stopped* the audio ... исчез`.

### 4. Subtitle-corpus hallucinations are not filtered

At least 11 occurrences across 6 files: `Субтитры DimaTorzok` (x2), `Редактор субтитров .Семкин`,
`Корректор .Кулакова`, `Продолжение следует...` (x6). A recognisable set — filterable by list.

### 5. Timecode collapse — a formatter bug

`formatter.py:41-75`, `_merge_consecutive_speakers`: blocks keep merging while the speaker is
unchanged and `gap < pause_threshold`. **There is no upper bound on block length.** Result, in
`2026-04-16_18-10-19.md`: 31m39s of recording rendered as **two segments, last timecode
`[00:00:45]`**. The text survives; navigation does not.

Three more files have timecode coverage below 20 % of their audio.

### 6. The "Diarized Transcript" section duplicates the transcript

In all 16 files that have it. The only difference is `**Other:**` → `**Speaker 1:**`, including
identical recognition errors. Diarization additionally splits single utterances across two speakers
(`Speaker 2: I got to` + `Speaker 1: see.`).

### 7. The batching trap

With `TAPEBACK_BATCH_SIZE>0`, faster-whisper 1.2.1 **silently ignores** `no_speech_threshold`,
`condition_on_previous_text` and `hallucination_silence_threshold`, and takes only the first value
from `temperature` (`transcribe.py:316-317, 351-369`). Enabling batching therefore quietly disables
the entire anti-hallucination configuration. The README does not warn about this.

---

## Research: should Whisper be replaced

The requirement is Russian speech with English technical terms **inside a single phrase**. That rules
out almost everything.

| Model | Verdict |
|---|---|
| **GigaAM v3** (Sber, MIT, 240M, Russian SOTA, punctuation + normalisation) | Best on pure Russian. But in a benchmark specifically on RU/EN code-switching it **breaks on English terms** (`Gemini` → `Jemni`). Not viable as the primary model. |
| **Canary-1B-v2** (NVIDIA, 25 languages, WER 8.1 % vs 9.9 % for large-v3) | Same code-switching weakness as GigaAM. |
| **Parakeet-TDT-0.6B-v3** (25 languages including Russian, RTFx ~3300, fits 4 GB easily) | Very fast, but detects language **per utterance**, not within a phrase. Mixed phrases are its weak spot. |
| **Voxtral Mini 4B Realtime** (Apache-2.0, WER 5.9 % on FLEURS) | ~9.5 GB in fp16. **Does not fit 4 GB.** |
| **Whisper large-v3** | Trained multilingually with translation, so mixed text is its normal mode. Supports `hotwords`. |

**Conclusion: stay on Whisper.** The choice between `large-v3-turbo` and `large-v3` is **settled by
measurement** (stage 3a), not by literature: turbo wins on speed and VRAM, large-v3 on Russian and
loop resistance. VRAM: turbo in float16 measured at 2428 MiB; large-v3 in `int8_float16` is expected
around 2.9 GB, which fits into 4 GB but only just.

GigaAM v3 as an optional second backend is **deferred, out of scope here**: it does not suit the
primary scenario, and an ASR-backend abstraction is substantial separate work.

---

## Plan

### Stage 1a — progress and timings — DONE

Landed as a 357-line diff. `ruff` and `ty` clean; **277 tests pass, coverage 93.52 %** (up from
92.96 % / 263 tests).

- `_timing.py`: `ProgressReporter` + `format_stage_progress`, throttled to one line per 10 s.
- `transcriber.py`: `describe()` reports the resolved model / device / compute type (plus
  `batch_size` when batching is on); progress is emitted from `_collect_segments`; `transcribe()`
  gained keyword-only `stage` / `on_status` so existing call sites keep working. faster-whisper's own
  `log_progress` was rejected: its tqdm bar writes straight to the terminal, bypassing `on_status`,
  so it would never reach the tray log.
- `pipeline.py`: `stage_timer` around `load_stereo_channels` and `gate_wav_inactive`; `describe()`
  reported in both the stereo and mono paths.
- Tests added: 4 on `ProgressReporter` / formatting, 5 on the transcriber, 2 pipeline regressions.

**What the first instrumented run revealed** (`2026-05-03_17-56-45.wav`, 2:25):

```
Whisper: large-v3-turbo on cuda/float16
  transcribe mic: 25% (0:36 / 2:25)
Stage 'transcribe mic' took 52.1s
  transcribe monitor: 76% (1:50 / 2:25)
Stage 'transcribe monitor' took 409.1s
```

- **The bottleneck is the monitor channel, not the mic.** The original report showed 8203 s on
  `transcribe mic`, and all previous work ("cleaner, faster mic channel", silence gating) targeted
  that path. The gating works — mic now finishes in 52 s. The problem moved.
- The same file took 116.8 s end-to-end in the isolated benchmark but 409 s for a single channel
  inside the pipeline. The difference is that the pipeline runs each channel through `loudnorm`,
  which lifts quiet audio together with background noise; VAD then sees more "speech" and the
  temperature ladder fires more often. This sharpens stage 2: `loudnorm` on the monitor channel is
  suspected of hurting Whisper's input quality, not merely costing ffmpeg time. To be confirmed by
  measurement, not changed blind.

### Stage 1b — GPU telemetry — DONE

Landed as a ~400-line code diff. `ruff` and `ty` clean; **296 tests pass, coverage 93.73 %**.

- `_gpu.py`: `sample_gpu()` context manager polling
  `clocks.sm,temperature.gpu,memory.used,clocks_event_reasons.active` every 5 s on a daemon thread,
  reporting one `GpuStats.format()` line per stage. **Observation only — no cap management, no
  sudo**: tapeback stays an unprivileged CLI, and `thermal_profile.sh` is run separately by hand.
- Throttle accounting uses the NVML bitmask and deliberately excludes `GpuIdle` (0x1) and
  `ApplicationsClocksSetting` (0x2) — idling is not throttling, and an applied cap is a setting
  rather than a symptom. Only SW power cap, HW slowdown, SW/HW thermal and HW power brake count.
- `query_nvidia_smi()` is now the single point of contact with the binary; `diarizer`'s private
  `_get_free_vram_mib` was folded into it, so a missing binary, a driver error and a hung query all
  degrade to `None` the same way instead of each module inventing its own handling.
- `sample_gpu` waits for the first poll before entering the block, so a stage shorter than the
  sampling interval still reports, and the sampler's output is deterministic rather than dependent
  on thread scheduling.
- New setting `TAPEBACK_GPU_TELEMETRY` (default on); no-op without `nvidia-smi` or on `cpu`.
- Verified against real `nvidia-smi` output, not only mocks: `['300', '57', '51',
  '0x0000000000000001']` parsed to `GPU: sm 300 MHz avg / 300 min, max 57°C, 51 MiB peak,
  throttled 0% of 6 samples` — idle correctly not counted as throttled.

### Stage 1c — run metadata

Split out of 1b to stay under the ~500-line commit limit.

Persist per-run metadata. Nothing is currently written anywhere, so there is no way to reconstruct
why the 16 recordings failed. Target: `~/.local/share/tapeback/runs/`, with a settings override.
Contents: session name, timestamps, resolved model/device/compute type, stage timings, GPU stats,
segment counts, detected language, outcome (completed / aborted / failed) and the error if any.

### Stage 2 — speed

- `chunk_length`: `7` → `30` (or `None`). One value, ~4.3x less encoder work.
- `split_channels_16k` (`audio.py:113-116`): move `aresample=16000` **ahead of** `loudnorm`, and
  measure whether `loudnorm` on the monitor channel should be weakened or dropped (see the stage 1a
  finding above).
- Measure before/after on the same file and record the numbers.
- Batching stays off **until** the code warns that enabling it silently disables several parameters
  (quality cause 7).

### Stage 3a — model benchmark, default chosen by numbers

The model default is fixed **only after measurement**. Grid: `large-v3-turbo` x `large-v3`, each in
`float16` and `int8_float16`, with `chunk_length` already corrected to 30.

Material — real files with known failures and a pre-written list of terms the model must get right:

- `2026-07-22_18-33-18.wav` — RAG (currently six different spellings), vector search, LLM as judge,
  Excalidraw, Notion, Jira
- `2026-06-15_02-52-59.wav` — ONNX Runtime (currently `on an X Runtime`), OpenVINO (`OpenSWIFO`),
  ML/CV pipeline, the number "75 fps"
- `2026-04-17_19-08-24.wav` — previous iterations ran on this file, so there is a baseline

Per-run metrics: time, RTF, peak VRAM, sm avg / max temp / % throttled, **how many listed terms were
recognised correctly**, presence of known hallucination strings, number of repeat loops.

Each configuration is measured twice — without and with `hotwords` — to separate the model's
contribution from the glossary's. Output: a table in CHANGELOG/README and fixed defaults for
`whisper_model` and `compute_type`.

### Stage 3 — quality

- `whisper_model` and `compute_type` set from stage 3a results (`compute_type=auto` on cuda currently
  yields `float16`, `transcriber.py:38-39`; large-v3 would need `int8_float16`).
- New `TAPEBACK_HOTWORDS` setting (string) threaded into `_invoke_transcribe`. Default: the project
  glossary — RAG, ONNX, OpenVINO, LLM, Whisper, Obsidian, tapeback, Excalidraw, Jira, Notion,
  embeddings, vector search, and so on.
- `language`: stop detecting the language independently per channel. Either detect once on the more
  speech-heavy channel and reuse it, or raise `language_detection_segments` from its default of 1
  (`settings.py:70`) to 4.
- Hallucination filter: a list of known subtitle artefacts in `const.py`, stripped at segment level.
  Must start from a failing test built on real strings from the transcripts.
- `formatter.py`: bound merged block length with a new constant (`MAX_BLOCK_SECONDS`, ~30–60 s) in
  `_merge_consecutive_speakers`. Failing test: 31 minutes of single-speaker segments must produce
  more than one timecode.
- Drop the "Diarized Transcript" section when it matches the main one segment for segment.

### Stage 4 — stop losing work (the reason 16 recordings vanished)

- Catch `KeyboardInterrupt` inside `_collect_segments` and write the segments collected so far to
  markdown instead of discarding everything.
- `try/finally` around the transcription stage so `free_gpu_memory()` also runs on the exception
  path.
- Cache per-channel intermediate results so a re-run does not start from zero.

### Order

All stages ship as separate commits (~500-line diff limit): **1a → 1b → 1c → 2 → 3a → 3 → 4**.

Stage 1 comes first deliberately: without before/after numbers, stages 3a and 3 would again become
guesswork, exactly as `chunk_length` once was. Stage 1a has already justified this — it moved the
suspected bottleneck from the mic channel to the monitor channel.

---

## Verification

Reference file: `2026-04-17_19-08-24.wav` (13m37s) — previous iterations ran on it, so comparisons
exist. Long case: `2026-07-30_13-25-08.wav` (15m55s) — the run that never finished.

```bash
uv run tapeback process <file> --no-diarize --no-summarize
```

Record per run: stage timings, RTF, sm avg / max temp / % throttled, segment count, presence of
strings from the hallucination list, timecodes per minute.

Required before finishing (from CLAUDE.md): `uv run ruff check --fix`, `uv run ruff format`,
`uv run ty check`, `uv run pytest` (coverage >= 90 %), update README.md and CHANGELOG.md (latest tag
is `v0.9.7`, so a new `0.9.8` section dated today).

Tests: every bug fix (timecode collapse, hallucinations, duplicate Diarized section, segment loss on
Ctrl+C) starts with a failing test under `tests/regressions/`.
