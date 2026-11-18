from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import EDA_RESULTS_DIR


def plot_pm25_distribution(dataframe: pd.DataFrame, output_path: Union[str, Path] = EDA_RESULTS_DIR / "pm25_distribution.png") -> Path:
    """Plot PM2.5 distribution for the combined station dataset."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.histplot(dataframe["PM2.5"].dropna(), bins=40, kde=True)
    plt.title("PM2.5 Distribution")
    plt.xlabel("PM2.5 (?g/m?)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def plot_missing_values(
    dataframe: pd.DataFrame,
    output_path: Union[str, Path] = EDA_RESULTS_DIR / "missing_values.png",
    reference_data: pd.DataFrame | None = None,
) -> Path:
    """Plot the missing-value percentage for key columns."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    source_data = dataframe if reference_data is None else reference_data
    missing_pct = (source_data.isna().mean() * 100).sort_values(ascending=False)
    missing_pct = missing_pct[missing_pct > 0]

    if missing_pct.empty:
        raise ValueError("No missing values found to plot.")

    plt.figure(figsize=(10, 6))
    sns.barplot(x=missing_pct.values, y=missing_pct.index, orient="h")
    plt.title("Missing Values by Feature")
    plt.xlabel("Missing (%)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def plot_station_pm25_summary(dataframe: pd.DataFrame, output_path: Union[str, Path] = EDA_RESULTS_DIR / "pm25_by_station.png") -> Path:
    """Compare PM2.5 distribution across monitoring stations."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=dataframe, x="station", y="PM2.5")
    plt.title("PM2.5 by Station")
    plt.xlabel("Station")
    plt.ylabel("PM2.5 (?g/m?)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def plot_monthly_trend(dataframe: pd.DataFrame, output_path: Union[str, Path] = EDA_RESULTS_DIR / "monthly_pm25_trend.png") -> Path:
    """Plot monthly average PM2.5 values across the study period."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    monthly = dataframe.assign(month_label=dataframe["datetime"].dt.to_period("M").astype(str)).groupby("month_label")["PM2.5"].mean().reset_index()
    monthly["month_label"] = pd.to_datetime(monthly["month_label"])

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly, x="month_label", y="PM2.5")
    plt.title("Monthly PM2.5 Trend")
    plt.xlabel("Month")
    plt.ylabel("Average PM2.5")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def plot_correlation_heatmap(dataframe: pd.DataFrame, output_path: Union[str, Path] = EDA_RESULTS_DIR / "correlation_heatmap.png") -> Path:
    """Create a correlation heatmap for pollutants and key weather variables."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    selected = dataframe[["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]].dropna()
    corr = selected.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Pollution and Weather Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def run_eda_visualizations(
    dataframe: pd.DataFrame,
    results_dir: Union[str, Path] = EDA_RESULTS_DIR,
    raw_data: pd.DataFrame | None = None,
) -> dict:
    """Generate the set of EDA plots required for the current project stage."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    reference_data = raw_data if raw_data is not None else dataframe
    paths = {
        "pm25_distribution": plot_pm25_distribution(dataframe, results_dir / "pm25_distribution.png"),
        "missing_values": plot_missing_values(reference_data, results_dir / "missing_values.png"),
        "pm25_by_station": plot_station_pm25_summary(dataframe, results_dir / "pm25_by_station.png"),
        "monthly_pm25_trend": plot_monthly_trend(dataframe, results_dir / "monthly_pm25_trend.png"),
        "correlation_heatmap": plot_correlation_heatmap(dataframe, results_dir / "correlation_heatmap.png"),
    }
    return paths
