"""
ETT Dataset Loader for Chronos-2
Supports ETTh1, ETTh2, ETTm1, ETTm2 datasets
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from autogluon.timeseries import TimeSeriesDataFrame


class ETTDatasetLoader:
    """Loader for ETT (Electricity Transformer Temperature) datasets."""
    
    DATASET_PATHS = {
        "ETTh1": "ETTh1.csv",
        "ETTh2": "ETTh2.csv",
        "ETTm1": "ETTm1.csv",
        "ETTm2": "ETTm2.csv",
    }
    
    FREQUENCIES = {
        "ETTh1": "H",
        "ETTh2": "H",
        "ETTm1": "15min",
        "ETTm2": "15min",
    }
    
    def __init__(self, data_dir: str = "/root/datasets/ett-dataset"):
        self.data_dir = Path(data_dir)
        
    def load(self, dataset_name: str = "ETTh1") -> pd.DataFrame:
        """Load raw ETT dataset as pandas DataFrame."""
        if dataset_name not in self.DATASET_PATHS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASET_PATHS.keys())}")
        
        filepath = self.data_dir / self.DATASET_PATHS[dataset_name]
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found at {filepath}. Please download ETT datasets first.")
        
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {dataset_name}: {df.shape}")
        return df
    
    def to_timeseries_dataframe(
        self,
        dataset_name: str = "ETTh1",
        target_columns: Optional[list] = None,
        use_univariate: bool = True
    ) -> TimeSeriesDataFrame:
        """
        Convert ETT data to AutoGluon TimeSeriesDataFrame format.
        
        Args:
            dataset_name: Name of ETT dataset
            target_columns: List of target columns. If None, uses all feature columns
            use_univariate: If True, treats each feature as separate time series
        
        Returns:
            TimeSeriesDataFrame
        """
        df = self.load(dataset_name)
        
        if target_columns is None:
            target_columns = [col for col in df.columns if col != 'date']
        
        if use_univariate:
            # Each column becomes a separate time series
            frames = []
            for col in target_columns:
                temp_df = df[['date', col]].copy()
                temp_df['item_id'] = col
                temp_df = temp_df.rename(columns={'date': 'timestamp', col: 'target'})
                frames.append(temp_df)
            
            combined = pd.concat(frames, ignore_index=True)
            ts_df = TimeSeriesDataFrame.from_data_frame(
                combined,
                id_column="item_id",
                timestamp_column="timestamp"
            )
        else:
            # Multivariate: single time series with multiple targets
            # For Chronos, we'll use one target and treat others as known covariates if needed
            df = df.rename(columns={'date': 'timestamp'})
            df['item_id'] = 'series_1'
            # Use first target column as main target
            main_target = target_columns[0]
            df = df.rename(columns={main_target: 'target'})
            ts_df = TimeSeriesDataFrame.from_data_frame(
                df,
                id_column="item_id",
                timestamp_column="timestamp"
            )
        
        print(f"TimeSeriesDataFrame shape: {ts_df.shape}, items: {ts_df.num_items}")
        return ts_df
    
    def split_data(
        self,
        ts_df: TimeSeriesDataFrame,
        prediction_length: int = 96,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """
        Split time series data into train/val/test sets.
        
        Args:
            ts_df: Input TimeSeriesDataFrame
            prediction_length: Forecast horizon
            train_ratio: Ratio for training
            val_ratio: Ratio for validation
        
        Returns:
            Tuple of (train, val, test) TimeSeriesDataFrames
        """
        train_items = []
        val_items = []
        test_items = []
        
        for item_id in ts_df.item_ids:
            item_df = ts_df.loc[item_id].copy().reset_index()
            n = len(item_df)
            
            # Calculate split points
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            # Ensure minimum length
            if train_end < prediction_length:
                raise ValueError(f"Train split too small for item {item_id}")
            if val_end - train_end < prediction_length:
                val_end = train_end + prediction_length
            if n - val_end < prediction_length:
                val_end = n - prediction_length
            
            train_df = item_df.iloc[:train_end].copy()
            val_df = item_df.iloc[train_end:val_end].copy()
            test_df = item_df.iloc[val_end:].copy()
            
            # Ensure item_id column exists
            train_df['item_id'] = item_id
            val_df['item_id'] = item_id
            test_df['item_id'] = item_id
            
            # Reorder columns: item_id, timestamp, target
            cols = ['item_id', 'timestamp', 'target']
            train_df = train_df[cols]
            val_df = val_df[cols]
            test_df = test_df[cols]
            
            train_items.append(train_df)
            val_items.append(val_df)
            test_items.append(test_df)
        
        train = TimeSeriesDataFrame(pd.concat(train_items, ignore_index=True))
        val = TimeSeriesDataFrame(pd.concat(val_items, ignore_index=True))
        test = TimeSeriesDataFrame(pd.concat(test_items, ignore_index=True))
        
        print(f"Split sizes - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
        return train, val, test
    
    def get_dataset_info(self, dataset_name: str = "ETTh1") -> dict:
        """Get information about the dataset."""
        df = self.load(dataset_name)
        info = {
            "name": dataset_name,
            "shape": df.shape,
            "columns": list(df.columns),
            "frequency": self.FREQUENCIES.get(dataset_name, "unknown"),
            "date_range": (df['date'].min(), df['date'].max()),
            "num_timesteps": len(df),
        }
        return info


def prepare_ett_for_chronos(
    dataset_name: str = "ETTh1",
    prediction_length: int = 96,
    data_dir: str = "/root/datasets/ett-dataset"
) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame]:
    """
    Convenience function to prepare ETT dataset for Chronos-2.
    
    Returns:
        Tuple of (train, val, test) TimeSeriesDataFrames
    """
    loader = ETTDatasetLoader(data_dir=data_dir)
    ts_df = loader.to_timeseries_dataframe(dataset_name, use_univariate=True)
    train, val, test = loader.split_data(ts_df, prediction_length=prediction_length)
    return train, val, test


if __name__ == "__main__":
    # Test ETT loader
    for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        print(f"\n{'='*50}")
        print(f"Testing {ds}")
        print(f"{'='*50}")
        try:
            loader = ETTDatasetLoader()
            info = loader.get_dataset_info(ds)
            print(f"Info: {info}")
            
            train, val, test = prepare_ett_for_chronos(ds, prediction_length=96)
            print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
        except Exception as e:
            print(f"Error loading {ds}: {e}")
