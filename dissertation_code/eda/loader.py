"""Load and clean the LaSDPC IoT dataset, restricted to temperature + RH channels.

Device type codes (per datasets/iot-dataset/README.md):
  1  = temperature (deg C)
  2  = relative humidity (% RH)
  10 = energy / CO2-occupancy proxy   -> interface-only, excluded from modelling
  11 = light intensity                -> contextual-only, excluded from modelling
"""

from pathlib import Path

import pandas as pd

DEVICE_TYPE_TEMPERATURE = 1
DEVICE_TYPE_HUMIDITY = 2

DEVICE_TYPE_LABELS = {
    1: "temperature",
    2: "relative_humidity",
    10: "energy_co2_proxy",
    11: "light_intensity",
}

MODEL_DEVICE_TYPES = (DEVICE_TYPE_TEMPERATURE, DEVICE_TYPE_HUMIDITY)


def load_raw(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date_time"] = pd.to_datetime(df["date_time"], format="mixed", utc=False)
    df["sensor_value"] = pd.to_numeric(df["sensor_value"], errors="coerce")
    df["device_type_label"] = df["id_device_type"].map(DEVICE_TYPE_LABELS)
    return df


def filter_temp_rh(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the temperature + RH channels used for modelling (project scope constraint)."""
    return df[df["id_device_type"].isin(MODEL_DEVICE_TYPES)].copy()


def drop_missing_sensor_values(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    n_before = len(df)
    cleaned = df.dropna(subset=["sensor_value"])
    return cleaned, n_before - len(cleaned)
