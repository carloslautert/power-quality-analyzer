import pytest

from src.config import EventThresholds
from src.event_detection import detect_voltage_events
from src.signal_generator import SyntheticEvent, generate_signal


def _detect(event: SyntheticEvent):
    data = generate_signal(duration_s=1.0, events=(event,))
    return detect_voltage_events(
        data["voltage"],
        data["timestamp"],
        sample_rate_hz=4_800,
        nominal_voltage_rms=220,
        nominal_frequency_hz=60,
        limits=EventThresholds(),
    )


def test_detects_voltage_sag_with_duration() -> None:
    events = _detect(SyntheticEvent("sag", 0.30, 0.60, 0.70))
    sag = next(event for event in events if event.event_type == "voltage_sag")

    assert sag.magnitude == pytest.approx(0.70, abs=0.03)
    assert sag.duration_s == pytest.approx(0.30, abs=1 / 60)
    assert sag.severity in {"Warning", "Critical"}


def test_detects_voltage_swell_with_duration() -> None:
    events = _detect(SyntheticEvent("swell", 0.25, 0.55, 1.18))
    swell = next(event for event in events if event.event_type == "voltage_swell")

    assert swell.magnitude == pytest.approx(1.18, abs=0.03)
    assert swell.duration_s == pytest.approx(0.30, abs=1 / 60)


def test_normal_voltage_has_no_magnitude_events() -> None:
    data = generate_signal(duration_s=0.5)
    events = detect_voltage_events(
        data["voltage"], data["timestamp"], 4_800, 220, 60, EventThresholds()
    )

    assert events == []

