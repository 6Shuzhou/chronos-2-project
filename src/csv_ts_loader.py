"""
Generic CSV time series loader for pre-split datasets.
Assumes train/val/test CSVs with columns: item_id, timestamp, target
"""

import os
import pandas as pd
from typing import Tuple
from pathlib import Path
from autogluon.timeseries import TimeSeriesDataFrame


class CSVTimeseriesLoader:
    """Load pre-split time series datasets stored as CSV."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
    
    def load(self) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame]:
        train_path = self.data_dir / "train" / "data.csv"
        val_path = self.data_dir / "val" / "data.csv"
        test_path = self.data_dir / "test" / "data.csv"
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        # Handle both 'timestamp' and 'date' column names
        time_col = "timestamp" if "timestamp" in train_df.columns else "date"
        
        train_df[time_col] = pd.to_datetime(train_df[time_col])
        val_df[time_col] = pd.to_datetime(val_df[time_col])
        test_df[time_col] = pd.to_datetime(test_df[time_col])
        
        # Resample to daily frequency and forward-fill missing dates (weekends/holidays)
        def resample_and_fill(df):
            dfs = []
            for item_id in df["item_id"].unique():
                item = df[df["item_id"] == item_id].copy()
                # Remove duplicate dates (keep last)
                item = item.drop_duplicates(subset=[time_col], keep="last")
                item = item.set_index(time_col).sort_index()
                item = item.resample("D").ffill().bfill()
                item["item_id"] = item_id
                item = item.reset_index()
                item = item.rename(columns={time_col: "timestamp"})
                dfs.append(item)
            return pd.concat(dfs, ignore_index=True)
        
        train_df = resample_and_fill(train_df)
        val_df = resample_and_fill(val_df)
        test_df = resample_and_fill(test_df)
        
        train_ts = TimeSeriesDataFrame.from_data_frame(
            train_df, id_column="item_id", timestamp_column="timestamp"
        )
        val_ts = TimeSeriesDataFrame.from_data_frame(
            val_df, id_column="item_id", timestamp_column="timestamp"
        )
        # Test must contain full history (train + val + test) for correct evaluation,
        # matching GluonTS/Monash convention where test includes all historical context
        full_test_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        test_ts = TimeSeriesDataFrame.from_data_frame(
            full_test_df, id_column="item_id", timestamp_column="timestamp"
        )
        
        print(f"Loaded {self.data_dir.name}: train={train_ts.shape}, val={val_ts.shape}, test={test_ts.shape}, items={train_ts.num_items}")
        return train_ts, val_ts, test_ts
