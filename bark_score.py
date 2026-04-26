"""
bark_score.py — Spectral bark heuristic

Computes a 0.0–1.0 "bark score" for an audio clip based on acoustic
features that distinguish dog barks from background noise.

Key features used:
  1. Spectral centroid  — barks are mid-frequency (400-3000 Hz)
  2. Onset strength     — barks are impulsive (sharp attack)
  3. Burst pattern      — barks are short, repeated energy bursts
  4. Harmonic ratio     — barks have partial harmonic structure

Uses only numpy + torchaudio (already in the container).
"""

import logging
import os

import numpy as np
import torch
import torchaudio

_logger = logging.getLogger("BarkScore")

SAMPLE_RATE = 16_000
N_FFT = 1024
HOP_LENGTH = 256


def compute_bark_score(clip_path: str) -> dict:
    """Analyse an audio clip and return bark heuristic features.

    Returns dict with:
        bark_score: float 0.0-1.0 (overall likelihood of being a bark)
        spectral_centroid: mean centroid in Hz
        onset_strength: impulsiveness measure
        burst_count: estimated number of energy bursts
        peak_freq: dominant frequency in Hz
    """
    if not clip_path or not os.path.isfile(clip_path):
        return {"bark_score": 0.0, "error": "file not found"}

    try:
        waveform, sr = torchaudio.load(clip_path)

        # Mono mix if multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
            sr = SAMPLE_RATE

        audio = waveform.squeeze().numpy()

        # Skip very short or silent clips
        if len(audio) < sr * 0.2:
            return {"bark_score": 0.0, "error": "too short"}
        if np.abs(audio).max() < 1e-5:
            return {"bark_score": 0.0, "error": "silent"}

        # ── 1. Spectral Centroid ─────────────────────────────────
        centroid = _spectral_centroid(audio, sr)
        # Barks: 400-3000 Hz sweet spot
        centroid_score = _bell_curve(centroid, center=1200, width=1200)

        # ── 2. Onset Strength (impulsiveness) ────────────────────
        onset_str = _onset_strength(audio, sr)
        # Normalize: barks typically have onset strength > 0.3
        onset_score = min(1.0, onset_str / 0.5)

        # ── 3. Burst Pattern (short energy bursts) ───────────────
        burst_count, burst_regularity = _count_bursts(audio, sr)
        # Barks: typically 1-6 bursts
        burst_score = _bell_curve(burst_count, center=3, width=4) if burst_count > 0 else 0.0

        # ── 4. Peak Frequency ────────────────────────────────────
        peak_freq = _peak_frequency(audio, sr)
        # Barks: 300-3000 Hz
        freq_score = _bell_curve(peak_freq, center=1000, width=1200)

        # ── 5. Energy variation (barks have high dynamic range) ──
        energy_var = _energy_variation(audio, sr)
        var_score = min(1.0, energy_var / 15.0)

        # ── Combine scores ───────────────────────────────────────
        bark_score = (
            0.25 * centroid_score +
            0.25 * onset_score +
            0.20 * burst_score +
            0.15 * freq_score +
            0.15 * var_score
        )
        bark_score = round(min(1.0, max(0.0, bark_score)), 3)

        return {
            "bark_score": bark_score,
            "spectral_centroid": round(centroid, 1),
            "onset_strength": round(onset_str, 3),
            "burst_count": burst_count,
            "peak_freq": round(peak_freq, 1),
            "energy_variation": round(energy_var, 2),
        }

    except Exception as exc:
        _logger.warning(f"bark_score failed for {clip_path}: {exc}")
        return {"bark_score": 0.0, "error": str(exc)}


def _spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """Weighted mean frequency of the spectrum."""
    spec = np.abs(np.fft.rfft(audio, n=N_FFT))
    freqs = np.fft.rfftfreq(N_FFT, d=1.0/sr)
    if spec.sum() < 1e-10:
        return 0.0
    return float(np.sum(freqs * spec) / np.sum(spec))


def _onset_strength(audio: np.ndarray, sr: int) -> float:
    """Measure how impulsive/sharp the onsets are."""
    # Compute energy in short frames
    frame_len = int(sr * 0.02)  # 20ms frames
    hop = frame_len // 2
    n_frames = (len(audio) - frame_len) // hop
    if n_frames < 3:
        return 0.0

    energy = np.array([
        np.sum(audio[i*hop:i*hop+frame_len]**2)
        for i in range(n_frames)
    ])
    energy = energy / (energy.max() + 1e-10)

    # Onset = positive energy difference
    diff = np.diff(energy)
    positive_diff = np.maximum(0, diff)
    return float(np.mean(positive_diff) + np.max(positive_diff) * 0.5)


def _count_bursts(audio: np.ndarray, sr: int) -> tuple:
    """Count energy bursts (bark-like pulses)."""
    frame_len = int(sr * 0.03)  # 30ms frames
    hop = frame_len // 2
    n_frames = (len(audio) - frame_len) // hop
    if n_frames < 3:
        return 0, 0.0

    energy = np.array([
        np.sum(audio[i*hop:i*hop+frame_len]**2)
        for i in range(n_frames)
    ])

    # Dynamic threshold based on signal
    threshold = np.mean(energy) + 0.5 * np.std(energy)
    if threshold < 1e-10:
        return 0, 0.0

    # Find bursts (energy above threshold)
    above = energy > threshold
    # Count transitions from below to above
    bursts = np.sum(np.diff(above.astype(int)) == 1)

    # Regularity: std of gaps between bursts (lower = more regular)
    burst_starts = np.where(np.diff(above.astype(int)) == 1)[0]
    if len(burst_starts) >= 2:
        gaps = np.diff(burst_starts)
        regularity = 1.0 / (1.0 + np.std(gaps) / (np.mean(gaps) + 1e-10))
    else:
        regularity = 0.0

    return int(bursts), float(regularity)


def _peak_frequency(audio: np.ndarray, sr: int) -> float:
    """Dominant frequency in the signal."""
    spec = np.abs(np.fft.rfft(audio, n=N_FFT))
    freqs = np.fft.rfftfreq(N_FFT, d=1.0/sr)
    # Ignore DC and very low frequencies
    spec[:5] = 0
    return float(freqs[np.argmax(spec)])


def _energy_variation(audio: np.ndarray, sr: int) -> float:
    """Dynamic range: ratio of peak energy to mean energy in dB."""
    frame_len = int(sr * 0.03)
    hop = frame_len // 2
    n_frames = (len(audio) - frame_len) // hop
    if n_frames < 3:
        return 0.0

    energy = np.array([
        np.sum(audio[i*hop:i*hop+frame_len]**2)
        for i in range(n_frames)
    ])
    energy = energy + 1e-10  # avoid log(0)
    db = 10 * np.log10(energy)
    return float(np.max(db) - np.mean(db))


def _bell_curve(value: float, center: float, width: float) -> float:
    """Gaussian-like scoring: 1.0 at center, falls off with width."""
    return float(np.exp(-0.5 * ((value - center) / width) ** 2))
