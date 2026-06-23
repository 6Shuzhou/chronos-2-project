"""
Monash Time Series Forecasting datasets loader.
Supports loading datasets saved in /root/datasets/monash_tsf/ as TimeSeriesDataFrame.
"""

import os
import pandas as pd
from typing import Tuple
from pathlib import Path
from autogluon.timeseries import TimeSeriesDataFrame


class MonashDatasetLoader:
    """Loader for Monash TSF datasets stored as CSV."""
    
    def __init__(self, data_dir: str = "/root/datasets/monash_tsf"):
        self.data_dir = Path(data_dir)
    
    def load(self, dataset_name: str) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """
        Load a Monash dataset.
        
        Args:
            dataset_name: Name of dataset (e.g., 'hospital', 'covid_deaths')
        
        Returns:
            Tuple of (train_data, test_data) as TimeSeriesDataFrames
        """
        train_path = self.data_dir / dataset_name / "train" / "data.csv"
        test_path = self.data_dir / dataset_name / "test" / "data.csv"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Train data not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Parse timestamps
        train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
        test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
        
        train_ts = TimeSeriesDataFrame.from_data_frame(
            train_df,
            id_column="item_id",
            timestamp_column="timestamp",
        )
        test_ts = TimeSeriesDataFrame.from_data_frame(
            test_df,
            id_column="item_id",
            timestamp_column="timestamp",
        )
        
        print(f"Loaded {dataset_name}: train={train_ts.shape}, test={test_ts.shape}, items={train_ts.num_items}")
        return train_ts, test_ts
    
    def split_train_val(
        self,
        train_ts: TimeSeriesDataFrame,
        prediction_length: int,
        val_ratio: float = 0.2,
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """
        Split training data into train/validation for fine-tuning.
        Validation set contains the last prediction_length * (1/val_ratio) timesteps.
        """
        train_items = []
        val_items = []
        
        for item_id in train_ts.item_ids:
            item_df = train_ts.loc[item_id].copy().reset_index()
            n = len(item_df)
            
            # AutoGluon needs both train and val to be longer than prediction_length
            # so it can reserve a validation window internally
            if n <= 2 * prediction_length:
                continue
            
            val_size = max(prediction_length + 1, int(n * val_ratio))
            train_size = n - val_size
            
            if train_size < prediction_length:
                continue
            
            train_split = item_df.iloc[:train_size].copy()
            val_split = item_df.iloc[train_size:].copy()
            
            train_split["item_id"] = item_id
            val_split["item_id"] = item_id
            
            cols = ["item_id", "timestamp", "target"]
            train_split = train_split[cols]
            val_split = val_split[cols]
            
            train_items.append(train_split)
            val_items.append(val_split)
        
        train_out = TimeSeriesDataFrame(pd.concat(train_items, ignore_index=True))
        val_out = TimeSeriesDataFrame(pd.concat(val_items, ignore_index=True))
        
        return train_out, val_out


def prepare_monash_dataset(
    dataset_name: str,
    prediction_length: int = 24,
    val_ratio: float = 0.2,
    data_dir: str = "/root/datasets/monash_tsf",
) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame]:
    """
    Convenience function to prepare a Monash dataset.
    
    Returns:
        Tuple of (train, val, test) TimeSeriesDataFrames.
        Note: val is carved out from the end of train.
    """
    loader = MonashDatasetLoader(data_dir=data_dir)
    train_full, test = loader.load(dataset_name)
    train, val = loader.split_train_val(train_full, prediction_length, val_ratio)
    return train, val, test


if __name__ == "__main__":
    for name in ["hospital", "covid_deaths", "fred_md", "car_parts_without_missing"]:
        print(f"\nTesting {name}")
        train, val, test = prepare_monash_dataset(name, prediction_length=12)
        print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
