"""CSV loading and validation for electrical measurements."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

from .signal_processing import infer_sample_rate

REQUIRED_COLUMNS = ("timestamp", "voltage", "current")
MINIMUM_SAMPLES = 64


class DataValidationError(ValueError):
    """Raised when an input measurement file cannot be safely analyzed."""


def validate_measurements(data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Validate, normalize, and return measurements plus inferred sample rate."""

    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(missing)}.")
    if len(data) < MINIMUM_SAMPLES:
        raise DataValidationError(f"At least {MINIMUM_SAMPLES} samples are required.")

    normalized = data.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized.isna().any().any():
        invalid_columns = normalized.columns[normalized.isna().any()].tolist()
        raise DataValidationError(
            "Non-numeric or missing values found in: " + ", ".join(invalid_columns) + "."
        )
    if not np.isfinite(normalized.to_numpy(dtype=float)).all():
        raise DataValidationError("The file contains infinite values.")

    try:
        sample_rate_hz = infer_sample_rate(normalized["timestamp"].to_numpy())
    except ValueError as error:
        raise DataValidationError(str(error)) from error
    duration_s = float(normalized["timestamp"].iloc[-1] - normalized["timestamp"].iloc[0])
    if duration_s <= 0:
        raise DataValidationError("Measurement duration must be positive.")
    return normalized, sample_rate_hz


def load_csv(source: str | Path | BinaryIO | TextIO) -> tuple[pd.DataFrame, float]:
    """Load a CSV source and validate the expected measurement schema."""

    try:
        data = pd.read_csv(source)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise DataValidationError(f"Could not read CSV: {error}") from error
    return validate_measurements(data)

