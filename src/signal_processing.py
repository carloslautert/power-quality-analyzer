"""General utilities for sampled electrical signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal as scipy_signal


def as_float_array(values: ArrayLike, *, name: str = "signal") -> NDArray[np.float64]:
    """Convert a one-dimensional finite signal to a NumPy float array."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two samples.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def infer_sample_rate(timestamp: ArrayLike, *, max_relative_jitter: float = 0.05) -> float:
    """Infer sample rate from a strictly increasing, approximately uniform time axis."""

    time = as_float_array(timestamp, name="timestamp")
    differences = np.diff(time)
    if np.any(differences <= 0):
        raise ValueError("timestamp must be strictly increasing.")

    median_period = float(np.median(differences))
    relative_jitter = float(np.max(np.abs(differences - median_period)) / median_period)
    if relative_jitter > max_relative_jitter:
        raise ValueError(
            "timestamp spacing is too irregular for FFT analysis "
            f"({relative_jitter:.1%} maximum relative jitter)."
        )
    return 1.0 / median_period


def estimate_fundamental_frequency(
    values: ArrayLike,
    sample_rate_hz: float,
    search_range_hz: tuple[float, float] = (45.0, 75.0),
    *,
    zero_padding_factor: int = 8,
) -> float:
    """Estimate the dominant fundamental using a Hann-windowed one-sided FFT.

    Zero padding and parabolic interpolation refine the peak location. They do
    not increase the physical frequency resolution of the observation window.
    """

    samples = as_float_array(values)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    low_hz, high_hz = search_range_hz
    nyquist_hz = sample_rate_hz / 2.0
    if not (0 < low_hz < high_hz < nyquist_hz):
        raise ValueError("Frequency search range must lie inside (0, Nyquist).")

    detrended = scipy_signal.detrend(samples, type="constant")
    window = scipy_signal.windows.hann(samples.size, sym=False)
    minimum_length = samples.size * max(1, zero_padding_factor)
    fft_length = 1 << (minimum_length - 1).bit_length()
    spectrum = np.fft.rfft(detrended * window, n=fft_length)
    frequencies = np.fft.rfftfreq(fft_length, d=1.0 / sample_rate_hz)
    magnitudes = np.abs(spectrum)

    candidates = np.flatnonzero((frequencies >= low_hz) & (frequencies <= high_hz))
    if candidates.size == 0:
        raise ValueError("No FFT bins fall inside the requested search range.")
    peak_index = int(candidates[np.argmax(magnitudes[candidates])])
    if magnitudes[peak_index] <= np.finfo(float).eps:
        raise ValueError("The signal has no measurable fundamental component.")

    delta = 0.0
    if 0 < peak_index < magnitudes.size - 1:
        left, center, right = np.log(np.maximum(magnitudes[peak_index - 1 : peak_index + 2], 1e-30))
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-20:
            delta = float(0.5 * (left - right) / denominator)
            delta = float(np.clip(delta, -0.5, 0.5))
    return float((peak_index + delta) * sample_rate_hz / fft_length)


def one_sided_amplitude_spectrum(
    values: ArrayLike,
    sample_rate_hz: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a Hann-windowed one-sided peak-amplitude spectrum.

    The coherent gain is corrected by dividing by the window sum. Interior FFT
    bins are doubled because the negative-frequency half is omitted.
    """

    samples = as_float_array(values)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")
    detrended = scipy_signal.detrend(samples, type="constant")
    window = scipy_signal.windows.hann(samples.size, sym=False)
    spectrum = np.fft.rfft(detrended * window)
    amplitudes = np.abs(spectrum) / np.sum(window)
    if amplitudes.size > 1:
        amplitudes[1:-1] *= 2.0
        if samples.size % 2:
            amplitudes[-1] *= 2.0
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sample_rate_hz)
    return frequencies.astype(np.float64), amplitudes.astype(np.float64)

