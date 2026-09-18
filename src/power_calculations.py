"""Electrical power and energy calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .signal_processing import as_float_array


@dataclass(frozen=True)
class PowerMetrics:
    """Aggregated electrical quantities for a measurement interval."""

    voltage_rms_v: float
    current_rms_a: float
    active_power_w: float
    reactive_power_var: float
    apparent_power_va: float
    power_factor: float
    displacement_power_factor: float
    nonactive_power_va: float
    energy_wh: float | None


def rms(values: ArrayLike) -> float:
    """Calculate true RMS, including DC and harmonic content."""

    samples = as_float_array(values)
    return float(np.sqrt(np.mean(np.square(samples))))


def fundamental_phasor(
    values: ArrayLike,
    timestamp: ArrayLike,
    frequency_hz: float,
) -> complex:
    """Estimate a fundamental RMS phasor by least-squares sinusoid fitting.

    The convention is ``x(t) = sqrt(2) * Re{X * exp(j*w*t)}``.
    """

    samples = as_float_array(values)
    time = as_float_array(timestamp, name="timestamp")
    if samples.size != time.size:
        raise ValueError("Signal and timestamp must have the same length.")
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive.")

    omega_time = 2.0 * np.pi * frequency_hz * time
    design = np.column_stack((np.cos(omega_time), np.sin(omega_time), np.ones(time.size)))
    cosine_coefficient, sine_coefficient, _ = np.linalg.lstsq(design, samples, rcond=None)[0]
    return complex(cosine_coefficient, -sine_coefficient) / np.sqrt(2.0)


def calculate_power_metrics(
    voltage: ArrayLike,
    current: ArrayLike,
    timestamp: ArrayLike,
    fundamental_frequency_hz: float,
) -> PowerMetrics:
    """Calculate true power quantities and fundamental reactive power.

    Active power and true power factor use the complete waveforms. Reactive
    power is Q1, derived from fundamental voltage/current phasors; this avoids
    pretending that ``Q = sqrt(S^2-P^2)`` remains valid under distortion.
    """

    voltage_values = as_float_array(voltage, name="voltage")
    current_values = as_float_array(current, name="current")
    time = as_float_array(timestamp, name="timestamp")
    if not (voltage_values.size == current_values.size == time.size):
        raise ValueError("Voltage, current, and timestamp must have the same length.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("timestamp must be strictly increasing.")

    voltage_rms = rms(voltage_values)
    current_rms = rms(current_values)
    instantaneous_power = voltage_values * current_values
    active_power = float(np.mean(instantaneous_power))
    apparent_power = voltage_rms * current_rms
    power_factor = active_power / apparent_power if apparent_power > 1e-12 else 0.0
    power_factor = float(np.clip(power_factor, -1.0, 1.0))

    voltage_phasor = fundamental_phasor(voltage_values, time, fundamental_frequency_hz)
    current_phasor = fundamental_phasor(current_values, time, fundamental_frequency_hz)
    complex_fundamental_power = voltage_phasor * np.conjugate(current_phasor)
    reactive_power = float(complex_fundamental_power.imag)
    fundamental_apparent = abs(voltage_phasor) * abs(current_phasor)
    displacement_pf = (
        float(complex_fundamental_power.real / fundamental_apparent)
        if fundamental_apparent > 1e-12
        else 0.0
    )

    residual_squared = apparent_power**2 - active_power**2 - reactive_power**2
    nonactive_power = float(np.sqrt(max(0.0, residual_squared)))
    if hasattr(np, "trapezoid"):
        integrated_energy = np.trapezoid(instantaneous_power, time)
    else:  # NumPy 1.x compatibility
        integrated_energy = np.trapz(instantaneous_power, time)
    energy_wh = float(integrated_energy / 3600.0)

    return PowerMetrics(
        voltage_rms_v=voltage_rms,
        current_rms_a=current_rms,
        active_power_w=active_power,
        reactive_power_var=reactive_power,
        apparent_power_va=apparent_power,
        power_factor=power_factor,
        displacement_power_factor=displacement_pf,
        nonactive_power_va=nonactive_power,
        energy_wh=energy_wh,
    )
