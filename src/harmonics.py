"""FFT spectrum, harmonic magnitudes, and THD calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .signal_processing import (
    as_float_array,
    estimate_fundamental_frequency,
    one_sided_amplitude_spectrum,
)


@dataclass(frozen=True)
class HarmonicAnalysis:
    """Frequency-domain results for one sampled signal."""

    fundamental_frequency_hz: float
    fundamental_rms: float
    thd_percent: float
    components: pd.DataFrame
    spectrum_frequency_hz: NDArray[np.float64]
    spectrum_peak_amplitude: NDArray[np.float64]


def _fit_harmonics(
    values: NDArray[np.float64],
    timestamp: NDArray[np.float64],
    fundamental_hz: float,
    orders: NDArray[np.int64],
) -> NDArray[np.float64]:
    columns: list[NDArray[np.float64]] = [np.ones(timestamp.size)]
    for order in orders:
        phase = 2.0 * np.pi * order * fundamental_hz * timestamp
        columns.extend((np.cos(phase), np.sin(phase)))
    coefficients = np.linalg.lstsq(np.column_stack(columns), values, rcond=None)[0]
    cosine = coefficients[1::2]
    sine = coefficients[2::2]
    return np.sqrt(np.square(cosine) + np.square(sine)) / np.sqrt(2.0)


def analyze_harmonics(
    values: ArrayLike,
    timestamp: ArrayLike,
    sample_rate_hz: float,
    *,
    fundamental_frequency_hz: float | None = None,
    search_range_hz: tuple[float, float] = (45.0, 75.0),
    max_harmonic_order: int = 40,
) -> HarmonicAnalysis:
    """Analyze harmonic RMS magnitudes and total harmonic distortion.

    An FFT locates the fundamental and produces the displayed spectrum. A
    simultaneous least-squares fit at integer multiples of that frequency
    reduces scalloping and leakage bias in the individual harmonic estimates.
    """

    samples = as_float_array(values)
    time = as_float_array(timestamp, name="timestamp")
    if samples.size != time.size:
        raise ValueError("Signal and timestamp must have the same length.")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive.")

    fundamental_hz = fundamental_frequency_hz or estimate_fundamental_frequency(
        samples, sample_rate_hz, search_range_hz
    )
    below_nyquist = np.nextafter(sample_rate_hz / 2.0, 0.0)
    highest_order = min(max_harmonic_order, int(below_nyquist // fundamental_hz))
    if highest_order < 2:
        raise ValueError("Sample rate is too low to analyze harmonic distortion.")
    orders = np.arange(1, highest_order + 1, dtype=np.int64)
    rms_magnitudes = _fit_harmonics(samples, time, fundamental_hz, orders)
    fundamental_rms = float(rms_magnitudes[0])
    if fundamental_rms <= 1e-12:
        raise ValueError("Fundamental magnitude is too small to calculate THD.")
    thd_percent = float(100.0 * np.sqrt(np.sum(np.square(rms_magnitudes[1:]))) / fundamental_rms)
    percentages = 100.0 * rms_magnitudes / fundamental_rms
    components = pd.DataFrame(
        {
            "order": orders,
            "frequency_hz": orders * fundamental_hz,
            "rms": rms_magnitudes,
            "percent_of_fundamental": percentages,
        }
    )
    spectrum_frequency, spectrum_amplitude = one_sided_amplitude_spectrum(samples, sample_rate_hz)
    return HarmonicAnalysis(
        fundamental_frequency_hz=float(fundamental_hz),
        fundamental_rms=fundamental_rms,
        thd_percent=thd_percent,
        components=components,
        spectrum_frequency_hz=spectrum_frequency,
        spectrum_peak_amplitude=spectrum_amplitude,
    )
