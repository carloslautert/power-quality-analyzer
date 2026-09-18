import numpy as np
import pytest

from src.harmonics import analyze_harmonics
from src.signal_processing import estimate_fundamental_frequency, one_sided_amplitude_spectrum


def test_frequency_estimation_for_non_bin_centered_tone() -> None:
    sample_rate = 4_800
    duration = 1.2
    time = np.arange(int(sample_rate * duration)) / sample_rate
    voltage = np.sqrt(2) * 220 * np.sin(2 * np.pi * 60.17 * time)

    frequency = estimate_fundamental_frequency(voltage, sample_rate)

    assert frequency == pytest.approx(60.17, abs=0.03)


def test_harmonic_magnitudes_and_thd() -> None:
    sample_rate = 4_800
    time = np.arange(sample_rate) / sample_rate
    voltage = (
        np.sqrt(2) * 220 * np.sin(2 * np.pi * 60 * time)
        + np.sqrt(2) * 11 * np.sin(2 * np.pi * 3 * 60 * time)
        + np.sqrt(2) * 6.6 * np.sin(2 * np.pi * 5 * 60 * time)
    )

    result = analyze_harmonics(voltage, time, sample_rate, max_harmonic_order=20)

    third = result.components.loc[result.components["order"] == 3, "rms"].item()
    fifth = result.components.loc[result.components["order"] == 5, "rms"].item()
    expected_thd = 100 * np.sqrt(11**2 + 6.6**2) / 220
    assert result.fundamental_frequency_hz == pytest.approx(60.0, abs=0.02)
    assert result.fundamental_rms == pytest.approx(220.0, rel=1e-4)
    assert third == pytest.approx(11.0, rel=1e-3)
    assert fifth == pytest.approx(6.6, rel=1e-3)
    assert result.thd_percent == pytest.approx(expected_thd, rel=1e-3)


def test_one_sided_spectrum_has_correct_peak_amplitude() -> None:
    sample_rate = 4_800
    time = np.arange(sample_rate) / sample_rate
    peak_voltage = np.sqrt(2) * 220
    voltage = peak_voltage * np.sin(2 * np.pi * 60 * time)

    frequency, amplitude = one_sided_amplitude_spectrum(voltage, sample_rate)
    peak_index = np.argmax(amplitude)

    assert frequency[peak_index] == pytest.approx(60.0)
    assert amplitude[peak_index] == pytest.approx(peak_voltage, rel=1e-4)

