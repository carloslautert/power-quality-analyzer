import numpy as np
import pytest

from src.power_calculations import calculate_power_metrics, rms


def test_rms_of_220_v_sine_wave() -> None:
    sample_rate = 12_000
    time = np.arange(sample_rate) / sample_rate
    voltage = np.sqrt(2) * 220 * np.sin(2 * np.pi * 60 * time)

    assert rms(voltage) == pytest.approx(220.0, rel=1e-5)


def test_sinusoidal_power_at_lagging_08_power_factor() -> None:
    sample_rate = 12_000
    time = np.arange(sample_rate) / sample_rate
    phase_angle = np.arccos(0.8)
    voltage = np.sqrt(2) * 220 * np.sin(2 * np.pi * 60 * time)
    current = np.sqrt(2) * 10 * np.sin(2 * np.pi * 60 * time - phase_angle)

    metrics = calculate_power_metrics(voltage, current, time, 60.0)

    assert metrics.voltage_rms_v == pytest.approx(220.0, rel=1e-5)
    assert metrics.current_rms_a == pytest.approx(10.0, rel=1e-5)
    assert metrics.active_power_w == pytest.approx(1_760.0, rel=1e-4)
    assert metrics.reactive_power_var == pytest.approx(1_320.0, rel=1e-4)
    assert metrics.apparent_power_va == pytest.approx(2_200.0, rel=1e-4)
    assert metrics.power_factor == pytest.approx(0.8, rel=1e-4)
    assert metrics.displacement_power_factor == pytest.approx(0.8, rel=1e-4)


def test_true_power_factor_includes_current_distortion() -> None:
    sample_rate = 12_000
    time = np.arange(sample_rate) / sample_rate
    voltage = np.sqrt(2) * 220 * np.sin(2 * np.pi * 60 * time)
    current = (
        np.sqrt(2) * 10 * np.sin(2 * np.pi * 60 * time)
        + np.sqrt(2) * 4 * np.sin(2 * np.pi * 5 * 60 * time)
    )

    metrics = calculate_power_metrics(voltage, current, time, 60.0)

    expected_pf = 10 / np.sqrt(10**2 + 4**2)
    assert metrics.power_factor == pytest.approx(expected_pf, rel=1e-4)
    assert metrics.displacement_power_factor == pytest.approx(1.0, rel=1e-4)
    assert metrics.nonactive_power_va > 0


def test_energy_is_integrated_in_watt_hours() -> None:
    sample_rate = 12_000
    duration = 2.0
    time = np.arange(int(sample_rate * duration)) / sample_rate
    voltage = np.sqrt(2) * 220 * np.sin(2 * np.pi * 60 * time)
    current = np.sqrt(2) * 5 * np.sin(2 * np.pi * 60 * time)

    metrics = calculate_power_metrics(voltage, current, time, 60.0)

    assert metrics.energy_wh == pytest.approx(1_100 * duration / 3_600, rel=2e-4)

