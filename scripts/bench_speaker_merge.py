"""Benchmark script for speaker_merge performance."""

import time

import numpy as np

from tapeback import const
from tapeback.speaker_merge import _apply_merge, _pick_merge_threshold


def bench_comparison_loop_baseline(speakers, profiles, total_speech, similarity_threshold):
    merge_map: dict[str, str] = {s: s for s in speakers}

    for i, sp_a in enumerate(speakers):
        for sp_b in speakers[i + 1 :]:
            if merge_map[sp_a] == merge_map[sp_b]:
                continue

            norm_a = float(np.linalg.norm(profiles[sp_a]))
            norm_b = float(np.linalg.norm(profiles[sp_b]))
            if norm_a < const.CHANNEL_EPSILON or norm_b < const.CHANNEL_EPSILON:
                continue

            similarity = float(np.dot(profiles[sp_a], profiles[sp_b]) / (norm_a * norm_b))
            threshold = _pick_merge_threshold(sp_a, sp_b, total_speech, similarity_threshold)

            if similarity >= threshold:
                _apply_merge(merge_map, sp_a, sp_b, speakers)


def bench_comparison_loop_optimized(speakers, profiles, total_speech, similarity_threshold):
    norms: dict[str, float] = {sp: float(np.linalg.norm(profiles[sp])) for sp in speakers}
    merge_map: dict[str, str] = {s: s for s in speakers}

    for i, sp_a in enumerate(speakers):
        for sp_b in speakers[i + 1 :]:
            if merge_map[sp_a] == merge_map[sp_b]:
                continue

            norm_a = norms[sp_a]
            norm_b = norms[sp_b]
            if norm_a < const.CHANNEL_EPSILON or norm_b < const.CHANNEL_EPSILON:
                continue

            similarity = float(np.dot(profiles[sp_a], profiles[sp_b]) / (norm_a * norm_b))
            threshold = _pick_merge_threshold(sp_a, sp_b, total_speech, similarity_threshold)

            if similarity >= threshold:
                _apply_merge(merge_map, sp_a, sp_b, speakers)


def run_benchmark(num_speakers: int = 30, iterations: int = 1000):
    speakers = [f"SPEAKER_{i:02d}" for i in range(num_speakers)]
    profiles = {sp: np.random.randn(200) for sp in speakers}
    total_speech = {sp: 20.0 for sp in speakers}

    # Baseline
    start_time = time.perf_counter()
    for _ in range(iterations):
        bench_comparison_loop_baseline(speakers, profiles, total_speech, 0.95)
    end_time = time.perf_counter()
    baseline_ms = ((end_time - start_time) / iterations) * 1000.0

    # Optimized
    start_time = time.perf_counter()
    for _ in range(iterations):
        bench_comparison_loop_optimized(speakers, profiles, total_speech, 0.95)
    end_time = time.perf_counter()
    opt_ms = ((end_time - start_time) / iterations) * 1000.0

    speedup = baseline_ms / opt_ms if opt_ms > 0 else 0
    print(
        f"Num Speakers: {num_speakers:3d} | Baseline: {baseline_ms:.4f} ms | "
        f"Optimized: {opt_ms:.4f} ms | Speedup: {speedup:.2f}x"
    )


if __name__ == "__main__":
    print("--- Speaker Comparison Loop Benchmark ---")
    for n in [10, 20, 50, 100]:
        run_benchmark(num_speakers=n, iterations=500)
