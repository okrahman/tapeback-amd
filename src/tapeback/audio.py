import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

from tapeback import const
from tapeback.channel import gate_inactive_regions


def _check_ffmpeg() -> None:
    """Raise RuntimeError if ffmpeg is not found."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install: sudo apt install ffmpeg")


def _check_audio_file(path: Path) -> None:
    """Raise RuntimeError if audio file is empty or too short."""
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"No audio recorded in {path.name}. Check your audio devices.")

    try:
        with wave.open(str(path), "rb") as wf:
            duration = wf.getnframes() / wf.getframerate()
            if duration < 1.0:
                raise RuntimeError(f"No audio recorded in {path.name}. Check your audio devices.")
    except wave.Error:
        # Not a valid WAV or corrupted header — let ffmpeg handle it
        pass


def _get_wav_duration(path: Path) -> float | None:
    """Return WAV duration in seconds, or None if not readable."""
    try:
        with wave.open(str(path), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except wave.Error:
        return None


def merge_channels(monitor_wav: Path, mic_wav: Path, output_dir: Path) -> Path:
    """Merge two mono WAVs into a stereo WAV (left=mic, right=monitor).

    The stereo file is the vault archive and the source for per-channel RMS
    analysis. The dual-channel pipeline derives its 16 kHz Whisper inputs
    separately via split_channels_16k(), so no mixed-down mono is produced here.

    Returns the stereo WAV path.
    """
    _check_ffmpeg()
    _check_audio_file(monitor_wav)
    _check_audio_file(mic_wav)

    # Check duration difference and determine trim target
    monitor_dur = _get_wav_duration(monitor_wav)
    mic_dur = _get_wav_duration(mic_wav)
    trim_duration: float | None = None

    if monitor_dur is not None and mic_dur is not None:
        diff = abs(monitor_dur - mic_dur)
        if diff > const.CHANNEL_DURATION_DIFF_WARN:
            print(
                f"Warning: audio channels differ by {diff:.1f}s, trimming to shorter",
                file=sys.stderr,
            )
        trim_duration = min(monitor_dur, mic_dur)

    output_dir.mkdir(parents=True, exist_ok=True)
    stereo_path = output_dir / const.FILE_STEREO

    # Merge to stereo (left=mic, right=monitor)
    merge_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mic_wav),
        "-i",
        str(monitor_wav),
        "-filter_complex",
        "[0:a][1:a]amerge=inputs=2[stereo]",
        "-map",
        "[stereo]",
    ]
    if trim_duration is not None:
        merge_cmd.extend(["-t", f"{trim_duration:.3f}"])
    merge_cmd.append(str(stereo_path))

    subprocess.run(merge_cmd, capture_output=True, check=True)

    return stereo_path


def split_channels_16k(stereo_wav: Path, output_dir: Path) -> tuple[Path, Path]:
    """Split stereo WAV into two mono 16kHz WAVs.

    Each channel gets independent loudnorm before downsampling.

    The filter order is load-bearing and must not be "optimised" by resampling
    first: ffmpeg's loudnorm upsamples internally to 192 kHz for EBU R128 true-peak
    measurement and emits at that rate, so it has to be followed by aresample or the
    output is a 192 kHz WAV. Measured on a 16-minute recording, resampling first is
    also marginally SLOWER (32.4 s vs 31.6 s) — loudnorm's cost does not scale with
    its input rate. Of that 32 s, loudnorm is ~31 s: the split without it takes 0.5 s,
    and `-filter_threads` does not help.

    Returns (mic_16k_path, monitor_16k_path).
    """
    _check_ffmpeg()

    output_dir.mkdir(parents=True, exist_ok=True)
    mic_16k_path = output_dir / const.FILE_MIC_16K
    monitor_16k_path = output_dir / const.FILE_MONITOR_16K

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(stereo_wav),
            "-filter_complex",
            "channelsplit=channel_layout=stereo[left][right];"
            f"[left]loudnorm={const.LOUDNORM_PARAMS},aresample={const.SAMPLE_RATE_16K}[mic];"
            f"[right]loudnorm={const.LOUDNORM_PARAMS},aresample={const.SAMPLE_RATE_16K}[mon]",
            "-map",
            "[mic]",
            str(mic_16k_path),
            "-map",
            "[mon]",
            str(monitor_16k_path),
        ],
        capture_output=True,
        check=True,
    )

    return mic_16k_path, monitor_16k_path


def get_channel_count(audio_path: Path) -> int:
    """Return the number of channels in a WAV file."""
    with wave.open(str(audio_path), "rb") as wf:
        return wf.getnchannels()


def convert_to_mono16k(input_file: Path, output_dir: Path) -> Path:
    """Convert any audio file to 16kHz mono WAV for Whisper.

    Used by `tapeback process` for pre-recorded files.
    """
    _check_ffmpeg()

    if not input_file.exists():
        raise RuntimeError(f"File not found: {input_file}")
    if input_file.stat().st_size == 0:
        raise RuntimeError("No audio recorded. Check your audio devices.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / const.FILE_MONO_16K

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_file),
            "-ac",
            "1",
            "-ar",
            str(const.SAMPLE_RATE_16K),
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )

    return output_path


def gate_wav_inactive(
    wav_path: Path,
    target_raw: np.ndarray,
    other_raw: np.ndarray,
    raw_sr: int,
) -> None:
    """Silence the listening regions of a 16 kHz mono WAV in place.

    Reads the WAV, zeroes windows where the speaker is inactive (see
    channel.gate_inactive_regions), and writes it back — so Whisper never sees the
    mic channel's pauses and can't hallucinate loops on them.
    """
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    gated = gate_inactive_regions(samples, target_raw, other_raw, raw_sr)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(gated.astype(np.int16).tobytes())
