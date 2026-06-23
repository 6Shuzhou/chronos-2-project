"""
Evaluation utilities for Chronos-2 forecasting results.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from autogluon.timeseries import TimeSeriesDataFrame


def calculate_metrics(
    y_true: TimeSeriesDataFrame,
    y_pred: TimeSeriesDataFrame,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate forecasting metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values (mean forecast)
        metrics: List of metric names
    
    Returns:
        Dictionary of metric names to values
    """
    if metrics is None:
        metrics = ["MAE", "RMSE", "MAPE", "sMAPE"]
    
    # Align predictions with ground truth
    # Extract mean prediction
    if "mean" in y_pred.columns:
        pred_values = y_pred["mean"].values
    else:
        # Use first quantile if mean not available
        pred_values = y_pred.iloc[:, 0].values
    
    true_values = y_true["target"].values
    
    # Ensure same length
    min_len = min(len(pred_values), len(true_values))
    pred_values = pred_values[:min_len]
    true_values = true_values[:min_len]
    
    results = {}
    
    # Remove NaN values
    mask = ~(np.isnan(true_values) | np.isnan(pred_values))
    y_t = true_values[mask]
    y_p = pred_values[mask]
    
    if len(y_t) == 0:
        return {m: float('nan') for m in metrics}
    
    for metric in metrics:
        metric_upper = metric.upper()
        if metric_upper == "MAE":
            results[metric] = np.mean(np.abs(y_t - y_p))
        elif metric_upper == "RMSE":
            results[metric] = np.sqrt(np.mean((y_t - y_p) ** 2))
        elif metric_upper == "MAPE":
            mask = y_t != 0
            results[metric] = np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100
        elif metric_upper == "SMAPE":
            denom = (np.abs(y_t) + np.abs(y_p)) / 2
            mask = denom != 0
            results[metric] = np.mean(np.abs(y_t[mask] - y_p[mask]) / denom[mask]) * 100
        elif metric_upper == "MSE":
            results[metric] = np.mean((y_t - y_p) ** 2)
        elif metric_upper == "R2":
            ss_res = np.sum((y_t - y_p) ** 2)
            ss_tot = np.sum((y_t - np.mean(y_t)) ** 2)
            results[metric] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return results


def plot_forecast(
    train_data: TimeSeriesDataFrame,
    test_data: TimeSeriesDataFrame,
    predictions: TimeSeriesDataFrame,
    item_id: Optional[str] = None,
    save_path: Optional[str] = None,
    title: str = "Chronos-2 Forecast"
) -> plt.Figure:
    """
    Plot forecast against actual values.
    
    Args:
        train_data: Training data
        test_data: Test data (ground truth)
        predictions: Model predictions
        item_id: Specific item to plot (if multi-series)
        save_path: Path to save figure
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Select item if multi-series
    if item_id is not None:
        train_series = train_data.loc[item_id]
        test_series = test_data.loc[item_id]
        pred_series = predictions.loc[item_id] if item_id in predictions.item_ids else predictions
    else:
        # Use first item
        first_item = train_data.item_ids[0]
        train_series = train_data.loc[first_item]
        test_series = test_data.loc[first_item]
        pred_series = predictions.loc[first_item] if first_item in predictions.item_ids else predictions
    
    # Plot training data
    ax.plot(train_series.index, train_series["target"], label="Train", color="blue", alpha=0.7)
    
    # Plot test data (ground truth)
    ax.plot(test_series.index, test_series["target"], label="Actual", color="green", linewidth=2)
    
    # Plot predictions
    if "mean" in pred_series.columns:
        ax.plot(pred_series.index, pred_series["mean"], label="Forecast", color="red", linestyle="--", linewidth=2)
        
        # Plot prediction intervals if available
        if "0.1" in pred_series.columns and "0.9" in pred_series.columns:
            ax.fill_between(
                pred_series.index,
                pred_series["0.1"],
                pred_series["0.9"],
                alpha=0.2,
                color="red",
                label="80% Confidence"
            )
    else:
        ax.plot(pred_series.index, pred_series.iloc[:, 0], label="Forecast", color="red", linestyle="--")
    
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def plot_multiple_forecasts(
    train_data: TimeSeriesDataFrame,
    test_data: TimeSeriesDataFrame,
    predictions_dict: Dict[str, TimeSeriesDataFrame],
    item_id: Optional[str] = None,
    save_path: Optional[str] = None,
    max_items: int = 4
) -> plt.Figure:
    """
    Plot forecasts from multiple models for comparison.
    
    Args:
        train_data: Training data
        test_data: Test data
        predictions_dict: Dictionary of model_name -> predictions
        item_id: Item ID to plot
        save_path: Path to save figure
        max_items: Maximum number of subplots
    
    Returns:
        Matplotlib figure
    """
    n_models = len(predictions_dict)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        
        if item_id is not None:
            train_series = train_data.loc[item_id]
            test_series = test_data.loc[item_id]
            pred_series = predictions.loc[item_id] if item_id in predictions.item_ids else predictions
        else:
            first_item = train_data.item_ids[0]
            train_series = train_data.loc[first_item]
            test_series = test_data.loc[first_item]
            pred_series = predictions.loc[first_item] if first_item in predictions.item_ids else predictions
        
        ax.plot(train_series.index[-100:], train_series["target"].iloc[-100:], label="Train", color="blue", alpha=0.5)
        ax.plot(test_series.index, test_series["target"], label="Actual", color="green", linewidth=2)
        
        if "mean" in pred_series.columns:
            ax.plot(pred_series.index, pred_series["mean"], label="Forecast", color="red", linestyle="--")
        else:
            ax.plot(pred_series.index, pred_series.iloc[:, 0], label="Forecast", color="red", linestyle="--")
        
        ax.set_title(f"{model_name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    return fig


def compare_results(
    results_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create comparison table of results from multiple experiments.
    
    Args:
        results_dict: Dictionary of experiment_name -> metrics
        save_path: Path to save CSV
    
    Returns:
        Comparison DataFrame
    """
    df = pd.DataFrame(results_dict).T
    print("\nResults Comparison:")
    print(df.to_string())
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"\nComparison saved to: {save_path}")
    
    return df


if __name__ == "__main__":
    print("Evaluation utilities loaded. Import this module to use.")
