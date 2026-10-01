import pytest
import pandas as pd

from anchored_forecaster.data import _load_category_frame, parse_duration


def test_paper_durations():
    assert [parse_duration(value) for value in ("2D", "1D", "3D", "5D", "7D")] == [48, 24, 72, 120, 168]


def test_unknown_duration_is_rejected():
    with pytest.raises(ValueError):
        parse_duration("3W")


def test_station_timestamp_values_must_match(tmp_path):
    category = tmp_path / "WATER" / "S_5"
    for station, timestamps in {
        "a": ["2010-01-01 00:00:00", "2010-01-01 01:00:00"],
        "b": ["2010-01-01 00:00:00", "2010-01-01 02:00:00"],
    }.items():
        station_dir = category / station
        station_dir.mkdir(parents=True)
        pd.DataFrame({
            "TIMESTAMP": timestamps,
            "INTERPOLATED_VALUE": [1.0, 2.0],
            "CONFIDENCE": [1.0, 1.0],
        }).to_csv(station_dir / "values.csv", index=False)
        (station_dir / "location.json").write_text('{"X COORD": 0, "Y COORD": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="hourly phase grid|inconsistent timestamps"):
        _load_category_frame(
            tmp_path, "S_5", "WATER",
            ("2010-01-01 00:00:00", "2010-01-01 03:00:00"),
        )
