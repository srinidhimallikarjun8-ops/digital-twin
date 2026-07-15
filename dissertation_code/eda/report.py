"""Sprint 1 EDA entry point: load -> clean -> profile -> write report + plots.

Produces the Sprint 1 deliverable (KB SS14): annotated dataset summary + data-quality report,
restricted to temperature + relative humidity per the project's modelling scope.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from dissertation_code.eda.loader import (
    drop_missing_sensor_values,
    filter_temp_rh,
    load_raw,
)
from dissertation_code.eda.profiler import (
    data_quality_report,
    high_humidity_episodes,
    per_zone_profile,
    zone_coverage,
)

DEFAULT_DATASET_PATH = Path("datasets/iot-dataset/Dataset_slice.csv")
DEFAULT_OUTPUT_DIR = Path("dissertation_code/eda/output")


def run(
    dataset_path: Path = DEFAULT_DATASET_PATH, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw(dataset_path)
    df_temp_rh = filter_temp_rh(df_raw)
    df_clean, n_dropped = drop_missing_sensor_values(df_temp_rh)

    quality = data_quality_report(df_raw, df_clean, n_dropped)
    profile = per_zone_profile(df_clean)
    coverage = zone_coverage(df_clean)
    high_rh = high_humidity_episodes(df_clean)

    profile.to_csv(output_dir / "per_zone_profile.csv", index=False)
    coverage.to_csv(output_dir / "zone_coverage.csv")
    high_rh.to_csv(output_dir / "high_humidity_episodes.csv")
    with open(output_dir / "data_quality_report.json", "w") as f:
        json.dump(quality, f, indent=2)

    _plot_zone_timeseries(df_clean, output_dir)

    return {
        "quality": quality,
        "profile": profile,
        "coverage": coverage,
        "high_rh": high_rh,
    }


def _plot_zone_timeseries(df_clean, output_dir: Path) -> None:
    for device_type_label, ylabel in (
        ("temperature", "Temperature (C)"),
        ("relative_humidity", "RH (%)"),
    ):
        subset = df_clean[df_clean["device_type_label"] == device_type_label]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        for (zone, device), group in subset.groupby(["id_enviroment", "id_device"]):
            group = group.sort_values("date_time")
            ax.plot(
                group["date_time"],
                group["sensor_value"],
                label=f"zone {zone} (device {device})",
                linewidth=0.8,
            )
        ax.set_xlabel("time")
        ax.set_ylabel(ylabel)
        ax.set_title(f"LaSDPC {device_type_label} by zone")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{device_type_label}_by_zone.png", dpi=120)
        plt.close(fig)


def print_summary(results: dict) -> None:
    print("=== Data Quality Report ===")
    print(json.dumps(results["quality"], indent=2))
    print()
    print("=== Per-Zone Profile (temperature + RH) ===")
    print(results["profile"].to_string(index=False))
    print()
    print("=== Zone Coverage (temp + RH availability) ===")
    print(results["coverage"].to_string())
    print()
    print("=== High-Humidity Episodes (>75% RH) ===")
    if results["high_rh"].empty:
        print("none found")
    else:
        print(results["high_rh"].to_string())
