import pytest

from src.analysis import analyze_measurements
from src.signal_generator import generate_scenario


def test_complete_analysis_of_normal_scenario() -> None:
    result = analyze_measurements(generate_scenario("normal"))

    assert result.status == "Normal"
    assert result.power.voltage_rms_v == pytest.approx(220, rel=1e-3)
    assert result.voltage_harmonics.fundamental_frequency_hz == pytest.approx(60, abs=0.02)
    assert result.voltage_harmonics.thd_percent < 0.1


def test_complete_analysis_flags_harmonic_scenario() -> None:
    result = analyze_measurements(generate_scenario("harmonics"))

    event_types = {event.event_type for event in result.events}
    assert "current_thd_excessive" in event_types
    assert result.current_harmonics.thd_percent == pytest.approx(10.77, abs=0.15)

