"""Configuration objects for analysis and event detection."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EventThresholds:
    """Demonstration thresholds expressed in per-unit or engineering units.

    These defaults are intentionally project settings, not a declaration of
    compliance with a specific power-quality standard.
    """

    interruption_pu: float = 0.10
    sag_pu: float = 0.90
    undervoltage_pu: float = 0.95
    overvoltage_pu: float = 1.05
    swell_pu: float = 1.10
    frequency_min_hz: float = 59.5
    frequency_max_hz: float = 60.5
    voltage_thd_warning_pct: float = 5.0
    current_thd_warning_pct: float = 8.0
    critical_low_pu: float = 0.70
    critical_high_pu: float = 1.20

    def __post_init__(self) -> None:
        if not (
            0 < self.interruption_pu < self.sag_pu < self.undervoltage_pu < 1
            < self.overvoltage_pu < self.swell_pu
        ):
            raise ValueError("Voltage thresholds must be ordered around 1.0 pu.")
        if self.frequency_min_hz >= self.frequency_max_hz:
            raise ValueError("The minimum frequency must be below the maximum.")


@dataclass(frozen=True)
class AnalysisConfig:
    """Top-level settings shared by the processing pipeline."""

    nominal_voltage_rms: float = 220.0
    nominal_frequency_hz: float = 60.0
    fundamental_search_min_hz: float = 45.0
    fundamental_search_max_hz: float = 75.0
    max_harmonic_order: int = 40
    thresholds: EventThresholds = field(default_factory=EventThresholds)

    def __post_init__(self) -> None:
        if self.nominal_voltage_rms <= 0 or self.nominal_frequency_hz <= 0:
            raise ValueError("Nominal voltage and frequency must be positive.")
        if self.fundamental_search_min_hz >= self.fundamental_search_max_hz:
            raise ValueError("Invalid fundamental-frequency search interval.")
        if self.max_harmonic_order < 2:
            raise ValueError("At least the second harmonic must be analyzed.")

