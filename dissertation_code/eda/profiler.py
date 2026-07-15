"""Per-zone baseline environmental profiles and data-quality checks for the LaSDPC slice."""

import pandas as pd

from dissertation_code.eda.loader import DEVICE_TYPE_HUMIDITY

# Comfort-zone bounds used elsewhere in the project (KB SS9 / Donkers et al.):
# temperature 20-26 degC comfort band; RH >75% sustained is the MVP humidity uncertainty trigger.
RH_HIGH_THRESHOLD = 75.0


def per_zone_profile(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (environment, device_type): count, range, mean, std."""
    grouped = df.groupby(["id_enviroment", "device_type_label"])["sensor_value"]
    profile = grouped.agg(["count", "min", "mean", "max", "std"]).reset_index()
    return profile.sort_values(["id_enviroment", "device_type_label"]).reset_index(
        drop=True
    )


def zone_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Which zones have both temperature and RH coverage (MVP needs >=2 such zones)."""
    pivot = pd.crosstab(df["id_enviroment"], df["device_type_label"])
    for col in ("temperature", "relative_humidity"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["has_temp_and_rh"] = (pivot["temperature"] > 0) & (
        pivot["relative_humidity"] > 0
    )
    return pivot


def high_humidity_episodes(
    df: pd.DataFrame, threshold: float = RH_HIGH_THRESHOLD
) -> pd.DataFrame:
    """Readings above the RH uncertainty threshold, per zone -- candidate active-learning query moments."""
    rh = df[df["id_device_type"] == DEVICE_TYPE_HUMIDITY]
    flagged = rh[rh["sensor_value"] > threshold]
    return (
        flagged.groupby("id_enviroment")["sensor_value"]
        .agg(["count", "mean", "max"])
        .rename(columns={"count": "n_readings_above_threshold"})
    )


def data_quality_report(
    df_raw: pd.DataFrame, df_clean: pd.DataFrame, n_dropped: int
) -> dict:
    """Sprint-1 dataset audit report (KB SS14 / SS15 risk register item)."""
    span = df_raw["date_time"].max() - df_raw["date_time"].min()
    return {
        "total_rows_raw": len(df_raw),
        "total_rows_temp_rh": len(df_clean),
        "rows_dropped_missing_value": n_dropped,
        "date_range_start": str(df_raw["date_time"].min()),
        "date_range_end": str(df_raw["date_time"].max()),
        "date_range_span": str(span),
        "n_environments": df_raw["id_enviroment"].nunique(),
        "n_devices": df_raw["id_device"].nunique(),
        "device_types_present": sorted(df_raw["id_device_type"].unique().tolist()),
    }
