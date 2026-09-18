"""Streamlit dashboard for the Power Quality Analyzer."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analysis import AnalysisResult, analyze_measurements
from src.config import AnalysisConfig
from src.data_loader import DataValidationError, load_csv
from src.signal_generator import SCENARIO_NAMES, generate_scenario

def apply_theme() -> None:
    """Apply a restrained engineering-dashboard visual theme."""

    st.markdown(
        """
        <style>
        .stApp { background: #0b1220; color: #e5e7eb; }
        [data-testid="stSidebar"] { background: #111827; }
        [data-testid="stMetric"] {
            background: #111b2e; border: 1px solid #26354d;
            border-radius: 8px; padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] { color: #93a4bc; }
        [data-testid="stMetricValue"] { color: #f8fafc; }
        .status-box {
            padding: 14px 18px; border-radius: 7px; margin: 4px 0 18px;
            font-weight: 650; letter-spacing: 0.02em;
        }
        .status-normal { background: #0d332b; border-left: 5px solid #2dd4bf; }
        .status-warning { background: #3a2b0b; border-left: 5px solid #f59e0b; }
        .status-critical { background: #3b151b; border-left: 5px solid #f43f5e; }
        .caption { color: #93a4bc; font-size: 0.88rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_cards(result: AnalysisResult) -> None:
    power = result.power
    values = (
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
    for start in range(0, len(values), 3):
        columns = st.columns(3)
        for column, (label, value) in zip(columns, values[start : start + 3], strict=True):
            column.metric(label, value)


def status_panel(result: AnalysisResult) -> None:
    css_class = f"status-{result.status.lower()}"
    detail = "No configured limits exceeded."
    if result.events:
        detail = f"{len(result.events)} event(s) detected; review the event log below."
    st.markdown(
        f'<div class="status-box {css_class}">System status: {result.status} &nbsp;—&nbsp; {detail}</div>',
        unsafe_allow_html=True,
    )


def waveform_figure(result: AnalysisResult) -> go.Figure:
    data = result.measurements
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scattergl(x=data["timestamp"], y=data["voltage"], name="Voltage", line={"color": "#38bdf8", "width": 1.4}),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scattergl(x=data["timestamp"], y=data["current"], name="Current", line={"color": "#f59e0b", "width": 1.2}),
        secondary_y=True,
    )
    figure.update_xaxes(title_text="Time (s)", gridcolor="#25324a")
    figure.update_yaxes(title_text="Voltage (V)", secondary_y=False, gridcolor="#25324a")
    figure.update_yaxes(title_text="Current (A)", secondary_y=True, gridcolor="#25324a")
    figure.update_layout(template="plotly_dark", height=430, margin={"l": 20, "r": 20, "t": 35, "b": 20}, legend={"orientation": "h"})
    return figure


def harmonic_figure(result: AnalysisResult) -> go.Figure:
    maximum_order = min(15, len(result.voltage_harmonics.components))
    voltage = result.voltage_harmonics.components.iloc[:maximum_order]
    current = result.current_harmonics.components.iloc[:maximum_order]
    figure = go.Figure()
    figure.add_bar(x=voltage["order"], y=voltage["percent_of_fundamental"], name="Voltage", marker_color="#38bdf8")
    figure.add_bar(x=current["order"], y=current["percent_of_fundamental"], name="Current", marker_color="#f59e0b")
    figure.update_layout(
        barmode="group", template="plotly_dark", height=420,
        xaxis={"title": "Harmonic order", "dtick": 1, "gridcolor": "#25324a"},
        yaxis={"title": "% of fundamental RMS", "gridcolor": "#25324a"},
        margin={"l": 20, "r": 20, "t": 35, "b": 20}, legend={"orientation": "h"},
    )
    return figure


def spectrum_figure(result: AnalysisResult) -> go.Figure:
    analysis = result.voltage_harmonics
    limit = analysis.spectrum_frequency_hz <= min(1_000.0, result.sample_rate_hz / 2.0)
    figure = go.Figure(
        go.Scatter(
            x=analysis.spectrum_frequency_hz[limit],
            y=analysis.spectrum_peak_amplitude[limit],
            line={"color": "#38bdf8", "width": 1.4},
            name="Voltage spectrum",
        )
    )
    figure.update_layout(
        template="plotly_dark", height=360,
        xaxis={"title": "Frequency (Hz)", "gridcolor": "#25324a"},
        yaxis={"title": "Peak amplitude (V)", "gridcolor": "#25324a"},
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
    )
    return figure


def event_table(result: AnalysisResult) -> None:
    if not result.events:
        st.success("Voltage, frequency, and THD are within the configured demonstration limits.")
        return
    display = result.event_table.rename(
        columns={
            "event_type": "Event",
            "start_time_s": "Start (s)",
            "end_time_s": "End (s)",
            "duration_s": "Duration (s)",
            "magnitude": "Magnitude",
            "unit": "Unit",
            "severity": "Severity",
            "description": "Details",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def select_data() -> pd.DataFrame:
    st.sidebar.header("Data source")
    source = st.sidebar.radio("Input", ("Built-in scenario", "Upload CSV"))
    if source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("timestamp, voltage, current", type="csv")
        if uploaded is None:
            st.info("Upload a CSV file to begin, or select a built-in scenario.")
            st.stop()
        data, _ = load_csv(uploaded)
        return data
    scenario = st.sidebar.selectbox(
        "Scenario",
        options=list(SCENARIO_NAMES),
        format_func=SCENARIO_NAMES.get,
    )
    return generate_scenario(scenario)


def main() -> None:
    st.set_page_config(page_title="Power Quality Analyzer", page_icon="⚡", layout="wide")
    apply_theme()
    st.title("Power Quality Analyzer")
    st.markdown('<p class="caption">Single-phase waveform, power, harmonic, and event analysis</p>', unsafe_allow_html=True)
    try:
        measurements = select_data()
        result = analyze_measurements(measurements, AnalysisConfig())
    except (DataValidationError, ValueError) as error:
        st.error(f"Input data is invalid: {error}")
        st.stop()

    st.subheader("Overview")
    metric_cards(result)
    st.subheader("Status")
    status_panel(result)

    waveform_tab, harmonics_tab, events_tab, data_tab = st.tabs(
        ("Waveforms", "Harmonics", "Events", "Measurement data")
    )
    with waveform_tab:
        st.plotly_chart(waveform_figure(result), use_container_width=True)
    with harmonics_tab:
        left, right = st.columns((1.2, 1.0))
        with left:
            st.plotly_chart(harmonic_figure(result), use_container_width=True)
        with right:
            st.plotly_chart(spectrum_figure(result), use_container_width=True)
        st.caption("Bars use RMS harmonic fits; the spectrum is a Hann-windowed, coherent-gain-corrected one-sided FFT.")
    with events_tab:
        event_table(result)
    with data_tab:
        st.dataframe(result.measurements.head(2_000), use_container_width=True, hide_index=True)
        st.caption(f"Sample rate: {result.sample_rate_hz:.1f} Hz · Display limited to the first 2,000 rows.")

    st.sidebar.divider()
    st.sidebar.caption("Educational software — not a certified power-quality instrument.")


if __name__ == "__main__":
    main()
