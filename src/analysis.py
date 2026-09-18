"""Application service that coordinates the complete analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import AnalysisConfig
from .data_loader import validate_measurements
from .event_detection import (
    PowerQualityEvent,
    detect_global_events,
    detect_voltage_events,
    events_to_frame,
)
from .harmonics import HarmonicAnalysis, analyze_harmonics
from .power_calculations import PowerMetrics, calculate_power_metrics


@dataclass(frozen=True)
class AnalysisResult:
    """All results needed by the dashboard or another future data adapter."""

    measurements: pd.DataFrame
    sample_rate_hz: float
    power: PowerMetrics
    voltage_harmonics: HarmonicAnalysis
    current_harmonics: HarmonicAnalysis
    events: tuple[PowerQualityEvent, ...]

    @property
    def event_table(self) -> pd.DataFrame:
        return events_to_frame(list(self.events))

    @property
    def status(self) -> str:
        severities = {event.severity for event in self.events}
        if "Critical" in severities:
            return "Critical"
        if "Warning" in severities:
            return "Warning"
        return "Normal"


def analyze_measurements(
    measurements: pd.DataFrame,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    """Validate measurements and execute power, harmonic, and event analyses."""

    settings = config or AnalysisConfig()
    data, sample_rate_hz = validate_measurements(measurements)
    timestamp = data["timestamp"].to_numpy()
    voltage = data["voltage"].to_numpy()
    current = data["current"].to_numpy()
    search_range = (
        settings.fundamental_search_min_hz,
        settings.fundamental_search_max_hz,
    )
    voltage_harmonics = analyze_harmonics(
        voltage,
        timestamp,
        sample_rate_hz,
        search_range_hz=search_range,
        max_harmonic_order=settings.max_harmonic_order,
    )
    current_harmonics = analyze_harmonics(
        current,
        timestamp,
        sample_rate_hz,
        fundamental_frequency_hz=voltage_harmonics.fundamental_frequency_hz,
        max_harmonic_order=settings.max_harmonic_order,
    )
    power = calculate_power_metrics(
        voltage,
        current,
        timestamp,
        voltage_harmonics.fundamental_frequency_hz,
    )
    voltage_events = detect_voltage_events(
        voltage,
        timestamp,
        sample_rate_hz,
        settings.nominal_voltage_rms,
        settings.nominal_frequency_hz,
        settings.thresholds,
    )
    interval_end = float(timestamp[-1] + 1.0 / sample_rate_hz)
    global_events = detect_global_events(
        start_time_s=float(timestamp[0]),
        end_time_s=interval_end,
        frequency_hz=voltage_harmonics.fundamental_frequency_hz,
        voltage_thd_percent=voltage_harmonics.thd_percent,
        current_thd_percent=current_harmonics.thd_percent,
        limits=settings.thresholds,
    )
    return AnalysisResult(
        measurements=data,
        sample_rate_hz=sample_rate_hz,
        power=power,
        voltage_harmonics=voltage_harmonics,
        current_harmonics=current_harmonics,
        events=tuple(voltage_events + global_events),
    )

