from io import StringIO

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataValidationError, load_csv, validate_measurements


def test_valid_csv_is_loaded_and_sample_rate_inferred() -> None:
    time = np.arange(100) / 1_000
    source = StringIO(pd.DataFrame({"timestamp": time, "voltage": 1, "current": 2}).to_csv(index=False))

    data, sample_rate = load_csv(source)

    assert len(data) == 100
    assert sample_rate == pytest.approx(1_000)


def test_missing_column_has_actionable_error() -> None:
    data = pd.DataFrame({"timestamp": np.arange(100), "voltage": np.ones(100)})

    with pytest.raises(DataValidationError, match="current"):
        validate_measurements(data)


def test_irregular_timestamp_is_rejected() -> None:
    timestamp = np.arange(100, dtype=float) / 1_000
    timestamp[50:] += 0.001
    data = pd.DataFrame({"timestamp": timestamp, "voltage": 1, "current": 2})

    with pytest.raises(DataValidationError, match="irregular"):
        validate_measurements(data)

