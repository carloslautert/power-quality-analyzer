import numpy as np
import pytest

from src.power_calculations import rms
from src.signal_generator import SCENARIO_NAMES, generate_scenario, generate_signal


def test_generated_signal_has_requested_rms_and_power_factor() -> None:
    data = generate_signal(
        duration_s=1.0,
        sample_rate_hz=4_800,
        voltage_rms_v=220,
        current_rms_a=8.4,
        power_factor=0.91,
    )

    assert rms(data["voltage"]) == pytest.approx(220.0, rel=1e-4)
    assert rms(data["current"]) == pytest.approx(8.4, rel=1e-4)
    active_power = np.mean(data["voltage"] * data["current"])
    assert active_power / (220 * 8.4) == pytest.approx(0.91, rel=1e-4)


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_all_dashboard_scenarios_are_available(scenario: str) -> None:
    data = generate_scenario(scenario, duration_s=0.5)

    assert list(data.columns) == ["timestamp", "voltage", "current"]
    assert len(data) == 2_400


def test_harmonic_ratio_is_expressed_relative_to_fundamental_rms() -> None:
    data = generate_signal(voltage_harmonics={3: 0.1})

    assert rms(data["voltage"]) == pytest.approx(220 * np.sqrt(1.01), rel=1e-4)

