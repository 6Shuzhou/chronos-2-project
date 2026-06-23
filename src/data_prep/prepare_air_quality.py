"""
Prepare Air Quality Data in India (2015-2020) for Chronos-2 experiments.

Target: AQI (Air Quality Index)
Frequency: daily
Prediction length: 30 days
Split: 80/10/10 chronological
"""

import pandas as pd
import numpy as np
from pathlib import Path


def prepare_air_quality(
    input_path: str = "/root/datasets/station_day.csv",
    output_dir: str = "/root/datasets/air_quality",
    target_col: str = "AQI",
    prediction_length: int = 30,
    min_length_ratio: float = 2.5,
):
    """Prepare air quality dataset for forecasting experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    df = pd.read_csv(input_path, parse_dates=["Date"])
    print(f"Raw shape: {df.shape}")
    print(f"Stations: {df['StationId'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Keep only target column
    df = df[["StationId", "Date", target_col]].copy()
    df = df.rename(columns={"StationId": "item_id", "Date": "timestamp", target_col: "target"})
    
    # Process each station
    processed = []
    min_length = int(prediction_length * min_length_ratio)
    
    for station_id, group in df.groupby("item_id"):
        group = group.sort_values("timestamp").set_index("timestamp")
        
        # Resample target to daily frequency (handle duplicate timestamps by mean)
        target_series = group["target"].resample("D").mean()
        
        # Forward fill then backward fill
        target_series = target_series.ffill().bfill()
        
        # Drop if still missing or too short
        if target_series.isna().any() or len(target_series) < min_length:
            continue
        
        # Reconstruct dataframe
        station_df = pd.DataFrame({
            "timestamp": target_series.index,
            "item_id": station_id,
            "target": target_series.values,
        })
        
        processed.append(station_df)
    
    full_df = pd.concat(processed, ignore_index=True)
    print(f"\nAfter preprocessing: {full_df.shape}")
    print(f"Kept stations: {full_df['item_id'].nunique()}")
    print(f"Date range: {full_df['timestamp'].min()} to {full_df['timestamp'].max()}")
    print(f"Missing target values: {full_df['target'].isna().sum()}")
    
    # Determine global split dates (chronological across all stations)
    global_min = full_df["timestamp"].min()
    global_max = full_df["timestamp"].max()
    total_days = (global_max - global_min).days + 1
    
    train_end = global_min + pd.Timedelta(days=int(total_days * 0.8))
    val_end = global_min + pd.Timedelta(days=int(total_days * 0.9))
    
    print(f"\nSplit dates: train <= {train_end.date()}, val <= {val_end.date()}, test <= {global_max.date()}")
    
    # Split each series
    train_list, val_list, test_list = [], [], []
    
    for station_id, group in full_df.groupby("item_id"):
        group = group.sort_values("timestamp")
        
        train = group[group["timestamp"] <= train_end]
        val = group[(group["timestamp"] > train_end) & (group["timestamp"] <= val_end)]
        test = group[group["timestamp"] > val_end]
        
        # Require sufficient length in each split
        if len(train) < prediction_length + 1 or len(val) < prediction_length or len(test) < prediction_length:
            continue
        
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)
    
    train_df = pd.concat(train_list, ignore_index=True)
    val_df = pd.concat(val_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)
    
    # Save
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "val").mkdir(exist_ok=True)
    (output_dir / "test").mkdir(exist_ok=True)
    
    train_df[["item_id", "timestamp", "target"]].to_csv(output_dir / "train" / "data.csv", index=False)
    val_df[["item_id", "timestamp", "target"]].to_csv(output_dir / "val" / "data.csv", index=False)
    test_df[["item_id", "timestamp", "target"]].to_csv(output_dir / "test" / "data.csv", index=False)
    
    print(f"\nTrain: {train_df.shape}, stations: {train_df['item_id'].nunique()}")
    print(f"Val:   {val_df.shape}, stations: {val_df['item_id'].nunique()}")
    print(f"Test:  {test_df.shape}, stations: {test_df['item_id'].nunique()}")
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_air_quality()
