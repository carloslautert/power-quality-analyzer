"""Configurable detection of steady-state and short-duration PQ events."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .config import EventThresholds
from .signal_processing import as_float_array


@dataclass(frozen=True)
class PowerQualityEvent:
    """One detected power-quality event."""

    event_type: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    magnitude: float
    unit: str
    severity: str
    description: str


def _voltage_classification(value_pu: float, limits: EventThresholds) -> str | None:
    if value_pu < limits.interruption_pu:
        return "interruption"
    if value_pu < limits.sag_pu:
        return "voltage_sag"
    if value_pu < limits.undervoltage_pu:
        return "undervoltage"
    if value_pu > limits.swell_pu:
        return "voltage_swell"
    if value_pu > limits.overvoltage_pu:
        return "overvoltage"
    return None


def _voltage_severity(event_type: str, magnitude_pu: float, limits: EventThresholds) -> str:
    if event_type == "interruption":
        return "Critical"
    if magnitude_pu < limits.critical_low_pu or magnitude_pu > limits.critical_high_pu:
        return "Critical"
    return "Warning"


def detect_voltage_events(
    voltage: ArrayLike,
    timestamp: ArrayLike,
    sample_rate_hz: float,
    nominal_voltage_rms: float,
    nominal_frequency_hz: float,
    limits: EventThresholds,
) -> list[PowerQualityEvent]:
    """Detect voltage-magnitude events using non-overlapping one-cycle RMS windows."""

    values = as_float_array(voltage, name="voltage")
    time = as_float_array(timestamp, name="timestamp")
    if values.size != time.size:
        raise ValueError("Voltage and timestamp must have the same length.")
    if sample_rate_hz <= 0 or nominal_voltage_rms <= 0 or nominal_frequency_hz <= 0:
        raise ValueError("Sample rate and nominal quantities must be positive.")

    samples_per_cycle = max(2, int(round(sample_rate_hz / nominal_frequency_hz)))
    sample_period_s = 1.0 / sample_rate_hz
    windows: list[tuple[str | None, float, float, float]] = []
    for start in range(0, values.size, samples_per_cycle):
        stop = min(start + samples_per_cycle, values.size)
        if stop - start < samples_per_cycle // 2:
            break
        window_rms = float(np.sqrt(np.mean(np.square(values[start:stop]))))
        magnitude_pu = window_rms / nominal_voltage_rms
        end_time = float(time[stop - 1] + sample_period_s)
        windows.append(
            (
                _voltage_classification(magnitude_pu, limits),
                float(time[start]),
                end_time,
                magnitude_pu,
            )
        )

    events: list[PowerQualityEvent] = []
    index = 0
    while index < len(windows):
        event_type = windows[index][0]
        if event_type is None:
            index += 1
            continue
        group = [windows[index]]
        index += 1
        while index < len(windows) and windows[index][0] == event_type:
            group.append(windows[index])
            index += 1

        magnitudes = [window[3] for window in group]
        magnitude = max(magnitudes) if event_type in {"voltage_swell", "overvoltage"} else min(magnitudes)
        start_time = group[0][1]
        end_time = group[-1][2]
        events.append(
            PowerQualityEvent(
                event_type=event_type,
                start_time_s=start_time,
                end_time_s=end_time,
                duration_s=end_time - start_time,
                magnitude=magnitude,
                unit="pu RMS",
                severity=_voltage_severity(event_type, magnitude, limits),
                description=f"Cycle RMS reached {magnitude:.3f} pu.",
            )
        )
    return events


def detect_global_events(
    *,
    start_time_s: float,
    end_time_s: float,
    frequency_hz: float,
    voltage_thd_percent: float,
    current_thd_percent: float,
    limits: EventThresholds,
) -> list[PowerQualityEvent]:
    """Detect interval-wide frequency and THD limit violations."""

    duration_s = end_time_s - start_time_s
    events: list[PowerQualityEvent] = []
    if not limits.frequency_min_hz <= frequency_hz <= limits.frequency_max_hz:
        deviation = abs(frequency_hz - (limits.frequency_min_hz + limits.frequency_max_hz) / 2.0)
        events.append(
            PowerQualityEvent(
                "frequency_out_of_range",
                start_time_s,
                end_time_s,
                duration_s,
                frequency_hz,
                "Hz",
                "Critical" if deviation > 2.0 else "Warning",
                f"Estimated frequency is {frequency_hz:.3f} Hz.",
            )
        )
    for name, value, threshold in (
        ("voltage_thd_excessive", voltage_thd_percent, limits.voltage_thd_warning_pct),
        ("current_thd_excessive", current_thd_percent, limits.current_thd_warning_pct),
    ):
        if value > threshold:
            events.append(
                PowerQualityEvent(
                    name,
                    start_time_s,
                    end_time_s,
                    duration_s,
                    value,
                    "%",
                    "Critical" if value > 2.0 * threshold else "Warning",
                    f"THD is {value:.2f}% (project limit {threshold:.2f}%).",
                )
            )
    return events


def events_to_frame(events: list[PowerQualityEvent]) -> pd.DataFrame:
    """Convert event objects to a display-friendly table."""

    columns = list(PowerQualityEvent.__dataclass_fields__)
    return pd.DataFrame([asdict(event) for event in events], columns=columns)

