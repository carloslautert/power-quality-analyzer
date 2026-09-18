"""Generate reproducible static figures used by the README."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pqa-matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from src.analysis import analyze_measurements
from src.signal_generator import generate_scenario

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"background": "#0b1220", "grid": "#26354d", "voltage": "#38bdf8", "current": "#f59e0b"}


def _style(axis: plt.Axes) -> None:
    axis.set_facecolor(COLORS["background"])
    axis.tick_params(colors="#cbd5e1")
    axis.xaxis.label.set_color("#cbd5e1")
    axis.yaxis.label.set_color("#cbd5e1")
    axis.title.set_color("#f8fafc")
    axis.grid(color=COLORS["grid"], alpha=0.7)
    for spine in axis.spines.values():
        spine.set_color(COLORS["grid"])


def waveform_image(output_directory: Path) -> None:
    data = generate_scenario("voltage_sag")
    visible = data["timestamp"] <= 0.70
    figure, voltage_axis = plt.subplots(figsize=(12, 4.8), facecolor=COLORS["background"])
    current_axis = voltage_axis.twinx()
    voltage_axis.plot(data.loc[visible, "timestamp"], data.loc[visible, "voltage"], color=COLORS["voltage"], linewidth=1.0, label="Voltage")
    current_axis.plot(data.loc[visible, "timestamp"], data.loc[visible, "current"], color=COLORS["current"], linewidth=0.9, alpha=0.9, label="Current")
    voltage_axis.set(title="Voltage sag waveform", xlabel="Time (s)", ylabel="Voltage (V)")
    current_axis.set_ylabel("Current (A)", color="#cbd5e1")
    _style(voltage_axis)
    _style(current_axis)
    figure.tight_layout()
    figure.savefig(output_directory / "waveform.png", dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def harmonics_image(output_directory: Path) -> None:
    result = analyze_measurements(generate_scenario("harmonics"))
    orders = np.arange(1, 16)
    voltage = result.voltage_harmonics.components.iloc[:15]["percent_of_fundamental"]
    current = result.current_harmonics.components.iloc[:15]["percent_of_fundamental"]
    figure, axis = plt.subplots(figsize=(12, 4.8), facecolor=COLORS["background"])
    width = 0.36
    axis.bar(orders - width / 2, voltage, width, color=COLORS["voltage"], label="Voltage")
    axis.bar(orders + width / 2, current, width, color=COLORS["current"], label="Current")
    axis.set(title="Harmonic components", xlabel="Harmonic order", ylabel="% of fundamental RMS", xticks=orders)
    axis.legend(frameon=False, labelcolor="#e5e7eb")
    _style(axis)
    figure.tight_layout()
    figure.savefig(output_directory / "harmonics.png", dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def dashboard_image(output_directory: Path) -> None:
    result = analyze_measurements(generate_scenario("normal"))
    power = result.power
    metrics = (
        ("Voltage RMS", f"{power.voltage_rms_v:.1f} V"),
        ("Current RMS", f"{power.current_rms_a:.2f} A"),
        ("Frequency", f"{result.voltage_harmonics.fundamental_frequency_hz:.3f} Hz"),
        ("Active Power", f"{power.active_power_w / 1_000:.2f} kW"),
        ("Reactive Power Q₁", f"{power.reactive_power_var / 1_000:.2f} kvar"),
        ("Apparent Power", f"{power.apparent_power_va / 1_000:.2f} kVA"),
        ("True Power Factor", f"{power.power_factor:.3f}"),
        ("Voltage THD", f"{result.voltage_harmonics.thd_percent:.2f}%"),
        ("Current THD", f"{result.current_harmonics.thd_percent:.2f}%"),
    )
    figure = plt.figure(figsize=(12.8, 7.2), facecolor=COLORS["background"])
    axis = figure.add_axes((0, 0, 1, 1))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.add_patch(Rectangle((0, 0), 0.235, 1, facecolor="#111827", edgecolor="none"))
    axis.text(0.025, 0.90, "Data source", color="#f8fafc", fontsize=12, weight="bold")
    axis.text(0.025, 0.84, "●  Built-in scenario", color="#f8fafc", fontsize=9)
    axis.text(0.025, 0.80, "○  Upload CSV", color="#9ca3af", fontsize=9)
    axis.text(0.025, 0.72, "Scenario", color="#cbd5e1", fontsize=9)
    axis.add_patch(FancyBboxPatch((0.025, 0.655), 0.185, 0.052, boxstyle="round,pad=0.008", facecolor="#0b1220", edgecolor="#26354d"))
    axis.text(0.038, 0.676, "Normal operation", color="#f8fafc", fontsize=9)
    axis.text(0.025, 0.12, "Educational software — not a certified", color="#7f8da3", fontsize=7.5)
    axis.text(0.025, 0.085, "power-quality instrument.", color="#7f8da3", fontsize=7.5)

    axis.text(0.285, 0.90, "Power Quality Analyzer", color="#f8fafc", fontsize=25, weight="bold")
    axis.text(0.285, 0.845, "Single-phase waveform, power, harmonic, and event analysis", color="#93a4bc", fontsize=9.5)
    axis.text(0.285, 0.78, "Overview", color="#f8fafc", fontsize=16, weight="bold")
    card_width, card_height = 0.205, 0.12
    x_positions = (0.285, 0.505, 0.725)
    y_positions = (0.62, 0.47, 0.32)
    for index, (label, value) in enumerate(metrics):
        row, column = divmod(index, 3)
        x, y = x_positions[column], y_positions[row]
        axis.add_patch(FancyBboxPatch((x, y), card_width, card_height, boxstyle="round,pad=0.008", facecolor="#111b2e", edgecolor="#26354d", linewidth=0.8))
        axis.text(x + 0.014, y + 0.082, label, color="#93a4bc", fontsize=8)
        axis.text(x + 0.014, y + 0.028, value, color="#f8fafc", fontsize=17)
    axis.text(0.285, 0.245, "Status", color="#f8fafc", fontsize=16, weight="bold")
    axis.add_patch(FancyBboxPatch((0.285, 0.15), 0.645, 0.065, boxstyle="round,pad=0.008", facecolor="#0d332b", edgecolor="#2dd4bf", linewidth=1.2))
    axis.text(0.305, 0.175, "System status: Normal — No configured limits exceeded.", color="#e5e7eb", fontsize=9, weight="bold")
    axis.text(0.285, 0.09, "Waveforms     Harmonics     Events     Measurement data", color="#cbd5e1", fontsize=9)
    figure.savefig(output_directory / "dashboard.png", dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    output_directory = ROOT / "images"
    output_directory.mkdir(exist_ok=True)
    dashboard_image(output_directory)
    waveform_image(output_directory)
    harmonics_image(output_directory)


if __name__ == "__main__":
    main()
