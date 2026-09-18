"""Synthetic voltage/current generation for demonstrations and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticEvent:
    """A synthetic disturbance applied over ``[start_s, end_s)``."""

    event_type: str
    start_s: float
    end_s: float
    magnitude: float


SCENARIO_NAMES: dict[str, str] = {
    "normal": "Normal operation",
    "harmonics": "Harmonic distortion",
    "voltage_sag": "Voltage sag",
    "voltage_swell": "Voltage swell",
}


def generate_signal(
    *,
    duration_s: float = 1.0,
    sample_rate_hz: float = 4_800.0,
    voltage_rms_v: float = 220.0,
    current_rms_a: float = 8.4,
    frequency_hz: float = 60.0,
    power_factor: float = 0.91,
    voltage_harmonics: Mapping[int, float] | None = None,
    current_harmonics: Mapping[int, float] | None = None,
    events: Sequence[SyntheticEvent] = (),
    voltage_noise_std_v: float = 0.0,
    current_noise_std_a: float = 0.0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate synchronized voltage and current waveforms.

    Harmonic values are RMS ratios relative to the corresponding fundamental.
    Positive power factor creates a lagging fundamental current.
    """

    if duration_s <= 0 or sample_rate_hz <= 0 or frequency_hz <= 0:
        raise ValueError("Duration, sample rate, and frequency must be positive.")
    if voltage_rms_v <= 0 or current_rms_a < 0:
        raise ValueError("Voltage must be positive and current cannot be negative.")
    if not (0 < power_factor <= 1):
        raise ValueError("power_factor must be in the interval (0, 1].")
    sample_count = int(round(duration_s * sample_rate_hz))
    if sample_count < 64:
        raise ValueError("At least 64 samples are required.")

    timestamp = np.arange(sample_count, dtype=np.float64) / sample_rate_hz
    frequency_profile = np.full(sample_count, frequency_hz, dtype=np.float64)
    voltage_envelope = np.ones(sample_count, dtype=np.float64)

    for event in events:
        if not (0 <= event.start_s < event.end_s <= duration_s):
            raise ValueError(f"Invalid interval for event {event.event_type!r}.")
        selection = (timestamp >= event.start_s) & (timestamp < event.end_s)
        if event.event_type in {"sag", "swell", "interruption"}:
            if event.magnitude < 0:
                raise ValueError("Event magnitude in per-unit cannot be negative.")
            voltage_envelope[selection] *= event.magnitude
        elif event.event_type == "frequency_variation":
            if event.magnitude <= 0:
                raise ValueError("Event frequency must be positive.")
            frequency_profile[selection] = event.magnitude
        else:
            raise ValueError(f"Unsupported synthetic event type: {event.event_type!r}.")

    phase = 2.0 * np.pi * np.cumsum(frequency_profile) / sample_rate_hz
    phase -= phase[0]
    current_phase_shift = np.arccos(power_factor)
    voltage = np.sqrt(2.0) * voltage_rms_v * np.sin(phase)
    current = np.sqrt(2.0) * current_rms_a * np.sin(phase - current_phase_shift)

    for order, ratio in (voltage_harmonics or {}).items():
        if order < 2 or ratio < 0:
            raise ValueError("Harmonic orders must be >= 2 and ratios non-negative.")
        voltage += np.sqrt(2.0) * voltage_rms_v * ratio * np.sin(order * phase)
    for order, ratio in (current_harmonics or {}).items():
        if order < 2 or ratio < 0:
            raise ValueError("Harmonic orders must be >= 2 and ratios non-negative.")
        current += np.sqrt(2.0) * current_rms_a * ratio * np.sin(
            order * phase - current_phase_shift
        )

    voltage *= voltage_envelope
    rng = np.random.default_rng(seed)
    if voltage_noise_std_v:
        voltage += rng.normal(0.0, voltage_noise_std_v, sample_count)
    if current_noise_std_a:
        current += rng.normal(0.0, current_noise_std_a, sample_count)
    return pd.DataFrame({"timestamp": timestamp, "voltage": voltage, "current": current})


def generate_scenario(name: str, *, duration_s: float = 1.0, sample_rate_hz: float = 4_800.0) -> pd.DataFrame:
    """Generate one of the built-in dashboard demonstration scenarios."""

    if name == "normal":
        return generate_signal(duration_s=duration_s, sample_rate_hz=sample_rate_hz)
    if name == "harmonics":
        return generate_signal(
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            voltage_harmonics={3: 0.025, 5: 0.035, 7: 0.015},
            current_harmonics={3: 0.08, 5: 0.06, 7: 0.04},
            voltage_noise_std_v=0.25,
            current_noise_std_a=0.015,
        )
    if name == "voltage_sag":
        return generate_signal(
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            events=(SyntheticEvent("sag", 0.30 * duration_s, 0.62 * duration_s, 0.68),),
        )
    if name == "voltage_swell":
        return generate_signal(
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
            events=(SyntheticEvent("swell", 0.30 * duration_s, 0.62 * duration_s, 1.18),),
        )
    raise ValueError(f"Unknown scenario {name!r}. Choose from {sorted(SCENARIO_NAMES)}.")
