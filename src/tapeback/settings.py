from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from tapeback.glossary import DEFAULT_HOTWORDS

# Default models per provider — used when TAPEBACK_LLM_MODEL is not set.
# Update here when providers deprecate models.
DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.5-flash",
    "openrouter": "google/gemini-2.5-flash:free",
    "deepseek": "deepseek-chat",
    "qwen": "qwen-turbo",
}

type LLMProvider = Literal[
    "anthropic",
    "openai",
    "groq",
    "gemini",
    "openrouter",
    "deepseek",
    "qwen",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TAPEBACK_",
        env_file=(Path.home() / ".config" / "tapeback" / ".env", ".env"),
        env_file_encoding="utf-8",
    )

    # Output directory (Obsidian vault or any folder)
    vault_path: Path = Path.home() / "tapeback"

    # Subdirectories in vault
    meetings_dir: str = "meetings"
    attachments_dir: str = "attachments/audio"

    # Whisper
    whisper_model: str = "large-v3-turbo"
    language: str = "auto"
    device: str = "cuda"
    compute_type: str = "auto"  # "int8"/"float16"
    beam_size: int = 4
    # Temperature fallback ladder. Decoding starts at 0.0 (deterministic, best
    # quality) and steps up only when a segment decodes poorly. The HIGH steps are
    # what break Whisper out of hallucination loops on noisy/quiet input — keep the
    # full ladder. Shortening it makes the model get STUCK in repeat loops, which is
    # both slower (it generates tokens up to the limit) and worse (repeats in text).
    temperature: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    # Batched inference (faster-whisper BatchedInferencePipeline): processes VAD
    # segments in parallel batches — several times faster on GPU. Off by default
    # (0) since it can OOM small GPUs; set e.g. TAPEBACK_BATCH_SIZE=8 to enable.
    batch_size: int = Field(default=0, ge=0)
    # Comma-separated terms to bias decoding towards (faster-whisper `hotwords`).
    # Default glossary and the reasoning behind it live in glossary.py.
    # Empty disables the bias entirely.
    hotwords: str = DEFAULT_HOTWORDS
    vad_filter: bool = True
    # Seconds of audio consumed per decode window. Whisper's encoder is FIXED at 30 s:
    # faster-whisper zero-pads every window back to 3000 mel frames before encoding
    # (transcribe.py `pad_or_trim`), so a smaller value does not make the encoder pass
    # cheaper — it only makes the run need more of them. At 2 the encoder does 15x the
    # necessary work; measured on a 145 s file, 2 -> 390.6 s and 30 -> 41.4 s.
    # Do not lower this to fight hallucinations on long pauses; that is what
    # vad_filter, no_speech_threshold and gate_mic_silence are for.
    chunk_length: int = 30
    condition_on_previous_text: bool = False
    # Lower = more aggressive silence rejection (helps suppress Whisper training-data
    # hallucinations like "Субтитры DimaTorzok" on long pauses). Default in Whisper is 0.6.
    no_speech_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    # Number of segments probed before deciding the language. faster-whisper's default
    # of 1 picks the language from the first segment, which misfires when a channel
    # starts with silence (e.g. mic while the user only listens — it can guess Japanese
    # and hallucinate). Raise it to probe more speech before committing.
    language_detection_segments: int = Field(default=1, ge=1)
    # Per-segment language detection — allows mixed languages in one recording
    # (code-switching, e.g. Russian speech with English terms). Less stable than a
    # fixed language; opt-in.
    multilingual: bool = False
    # Skip silent gaps longer than this many seconds when a hallucination is detected
    # (uses word timestamps, which are always on). None disables it.
    hallucination_silence_threshold: float | None = Field(default=None, ge=0.0)

    # Write one JSON record per processing run (config + status lines + outcome), so a
    # run that failed or was interrupted can be diagnosed afterwards. Never contains
    # credentials — the recorded fields are an explicit allow-list in _runlog.py.
    run_log: bool = True
    # None → XDG data dir (~/.local/share/tapeback/runs).
    run_log_dir: Path | None = None

    # Sample GPU clocks/temperature during transcription and report a one-line summary
    # per stage. Observation only — tapeback never changes clock or power caps (that
    # needs root). No-op when nvidia-smi is unavailable.
    gpu_telemetry: bool = True

    # Seconds to wait for a GPU thermal clamp to release before starting transcription.
    # On laptops that share one heatsink between CPU and GPU, the controller can cut the
    # GPU's power budget (measured: 50 W -> 5 W, clocks pinned to 300 MHz) and hold it
    # there while the CPU stays hot. 0 disables the check entirely.
    thermal_clamp_wait: float = Field(default=60.0, ge=0.0)
    # When the clamp has not released by then, transcribe on CPU instead of on a card
    # limited to a tenth of its power. Measured on the same clip: CPU 2.39x real time,
    # clamped GPU 0.31x — the CPU is ~8x faster, so waiting it out is the worse option.
    thermal_clamp_cpu_fallback: bool = True
    # Idle gap after each transcription stage, to shed heat instead of driving the
    # chassis into the clamp in the first place. Off by default: it costs wall-clock on
    # a machine that cools adequately.
    stage_pause_seconds: float = Field(default=0.0, ge=0.0)

    # Audio
    monitor_source: str = "auto"
    mic_source: str = "auto"
    sample_rate: int = 48000

    # HuggingFace (for pyannote). SecretStr prevents leakage in repr/str/model_dump.
    hf_token: SecretStr = SecretStr("")

    # Diarization
    diarize: bool = True
    max_speakers: int | None = None
    clustering_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    # merge only near-identical spectral profiles
    spectral_merge_threshold: float = Field(default=0.96, ge=0.0, le=1.0)

    # Post-processing
    pause_threshold: float = Field(default=1.0, ge=0.0)  # split on word gaps >= this
    # Silence the mic channel where the user is listening (mic quiet or monitor
    # dominant) before transcription, so Whisper doesn't hallucinate loops on the
    # pauses (slow + garbage). Dual-channel (stereo) pipeline only.
    gate_mic_silence: bool = True

    # Live transcription — opt-in. Off by default because mid-recording GPU
    # contention with the post-recording pipeline causes long stalls on small
    # cards (4 GiB).  Enable with TAPEBACK_LIVE=true when VRAM is plentiful.
    live: bool = False
    live_interval: int = Field(default=60, gt=0)  # seconds between transcription cycles
    live_overlap: float = Field(default=2.0, ge=0.0)  # seconds of overlap between chunks
    live_min_chunk: float = Field(default=5.0, gt=0.0)  # min new audio to trigger transcription

    # Summarization
    summarize: bool = True
    llm_provider: LLMProvider = "anthropic"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""

    @model_validator(mode="after")
    def _validate_live_chunking(self) -> "Settings":
        """Live chunks must be shorter than the interval or the loop drops audio."""
        if self.live and self.live_min_chunk > self.live_interval:
            raise ValueError(
                f"live_min_chunk ({self.live_min_chunk}s) must be <= "
                f"live_interval ({self.live_interval}s); otherwise cycles starve."
            )
        return self


def get_settings() -> Settings:
    """Load settings from env vars and .env files."""
    return Settings()  # type: ignore[call-arg]
