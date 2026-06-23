"""
Chronos-2 experiments using AutoGluon TimeSeries native API.
Follows the official tutorial: https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html

Supports:
- Zero-shot forecasting with Chronos-2
- Fine-tuning (transfer learning) with Chronos-2
- Evaluation and visualization

Note: Models are NOT saved to disk to save space. Only results are preserved.
"""

import os
import shutil
import json
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def get_temp_model_dir() -> str:
    """Get a temporary directory for AutoGluon models that will be cleaned up."""
    return "/tmp/chronos2_autogluon_models"


def cleanup_model_dir(path: str):
    """Remove AutoGluon model directory to save space."""
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Cleaned up model directory: {path}")


def run_zero_shot(
    train_data: TimeSeriesDataFrame,
    test_data: TimeSeriesDataFrame,
    model_path: str = "/root/chronos-2-model",
    prediction_length: int = 96,
    time_limit: int = 300,
    results_dir: str = "/root/chronos-2-project/results",
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Run Chronos-2 zero-shot forecasting experiment.
    
    Args:
        train_data: Training time series data
        test_data: Test time series data
        model_path: Path to local Chronos-2 model or HuggingFace repo ID
        prediction_length: Forecast horizon
        time_limit: Time limit in seconds for fitting
        results_dir: Directory to save results
        plot: Whether to generate forecast plots
    
    Returns:
        Dictionary with scores and predictions
    """
    os.makedirs(results_dir, exist_ok=True)
    model_dir = get_temp_model_dir()
    
    print(f"\n{'='*60}")
    print("Chronos-2 Zero-Shot Forecasting")
    print(f"{'='*60}")
    print(f"Prediction length: {prediction_length}")
    print(f"Model path: {model_path}")
    print(f"Time limit: {time_limit}s")
    
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=model_dir,
    )
    
    start_time = time.time()
    predictor.fit(
        train_data=train_data,
        hyperparameters={
            "Chronos2": {
                "model_path": model_path,
            }
        },
        time_limit=time_limit,
        enable_ensemble=False,
    )
    fit_time = time.time() - start_time
    
    # Predictions
    predictions = predictor.predict(train_data, model="Chronos2")
    
    # Evaluate
    scores = predictor.evaluate(test_data, model="Chronos2")
    
    # Save results
    results = {
        "mode": "zero_shot",
        "model": "Chronos2",
        "model_path": model_path,
        "prediction_length": prediction_length,
        "fit_time": fit_time,
        "scores": scores,
    }
    
    results_path = Path(results_dir) / "zero_shot_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    pred_path = Path(results_dir) / "zero_shot_predictions.csv"
    predictions.to_csv(pred_path)
    
    # Plot
    if plot:
        plots_dir = Path(results_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        for item_id in train_data.item_ids[:3]:  # Plot first 3 items
            fig, ax = plt.subplots(figsize=(12, 5))
            
            train_item = train_data.loc[item_id]
            test_item = test_data.loc[item_id]
            pred_item = predictions.loc[item_id]
            
            ax.plot(train_item.index[-200:], train_item["target"].iloc[-200:], label="Train", color="blue", alpha=0.7)
            ax.plot(test_item.index, test_item["target"], label="Actual", color="green", linewidth=2)
            ax.plot(pred_item.index, pred_item["mean"], label="Forecast", color="red", linestyle="--", linewidth=2)
            
            # Add quantile bands if available
            q_cols = [c for c in pred_item.columns if c not in ["mean"]]
            if len(q_cols) >= 2:
                low_q = min(q_cols, key=lambda x: float(x))
                high_q = max(q_cols, key=lambda x: float(x))
                ax.fill_between(
                    pred_item.index,
                    pred_item[low_q],
                    pred_item[high_q],
                    alpha=0.2, color="red", label=f"{low_q}-{high_q}"
                )
            
            ax.set_title(f"Chronos-2 Zero-Shot: {item_id}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / f"zero_shot_{item_id}.png", dpi=200, bbox_inches='tight')
            plt.close()
    
    # Cleanup model to save space
    cleanup_model_dir(model_dir)
    
    print(f"\n{'='*60}")
    print("Zero-Shot Results:")
    print(f"{'='*60}")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nResults saved to: {results_dir}")
    
    return results


def run_finetune(
    train_data: TimeSeriesDataFrame,
    val_data: Optional[TimeSeriesDataFrame],
    test_data: TimeSeriesDataFrame,
    model_path: str = "/root/chronos-2-model",
    prediction_length: int = 96,
    time_limit: int = 600,
    fine_tune_mode: str = "lora",  # "lora" or "full"
    fine_tune_lr: float = 1e-3,
    fine_tune_steps: int = 1000,
    results_dir: str = "/root/chronos-2-project/results",
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Run Chronos-2 fine-tuning (transfer learning) experiment.
    
    By default uses LoRA adapter for efficient fine-tuning with minimal disk footprint.
    
    Args:
        train_data: Training time series data
        val_data: Validation data (optional)
        test_data: Test time series data
        model_path: Path to local Chronos-2 model
        prediction_length: Forecast horizon
        time_limit: Time limit in seconds
        fine_tune_mode: "lora" (default, efficient) or "full"
        fine_tune_lr: Learning rate for fine-tuning
        fine_tune_steps: Number of fine-tuning steps
        results_dir: Directory to save results
        plot: Whether to generate plots
    
    Returns:
        Dictionary with scores and predictions
    """
    os.makedirs(results_dir, exist_ok=True)
    model_dir = get_temp_model_dir() + "_finetuned"
    
    print(f"\n{'='*60}")
    print("Chronos-2 Fine-Tuning (Transfer Learning)")
    print(f"{'='*60}")
    print(f"Prediction length: {prediction_length}")
    print(f"Model path: {model_path}")
    print(f"Fine-tune mode: {fine_tune_mode}")
    print(f"Learning rate: {fine_tune_lr}")
    print(f"Steps: {fine_tune_steps}")
    print(f"Time limit: {time_limit}s")
    
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=model_dir,
    )
    
    fit_kwargs = {
        "train_data": train_data,
        "hyperparameters": {
            "Chronos2": {
                "model_path": model_path,
                "fine_tune": True,
                "fine_tune_mode": fine_tune_mode,
                "fine_tune_lr": fine_tune_lr,
                "fine_tune_steps": fine_tune_steps,
            }
        },
        "time_limit": time_limit,
        "enable_ensemble": False,
    }
    
    if val_data is not None:
        fit_kwargs["tuning_data"] = val_data
    
    start_time = time.time()
    predictor.fit(**fit_kwargs)
    fit_time = time.time() - start_time
    
    # Predictions
    predictions = predictor.predict(train_data, model="Chronos2")
    
    # Evaluate
    scores = predictor.evaluate(test_data, model="Chronos2")
    
    # Save results
    results = {
        "mode": "fine_tuned",
        "model": "Chronos2",
        "model_path": model_path,
        "prediction_length": prediction_length,
        "fine_tune_mode": fine_tune_mode,
        "fine_tune_lr": fine_tune_lr,
        "fine_tune_steps": fine_tune_steps,
        "fit_time": fit_time,
        "scores": scores,
    }
    
    results_path = Path(results_dir) / "finetune_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    pred_path = Path(results_dir) / "finetune_predictions.csv"
    predictions.to_csv(pred_path)
    
    # Plot
    if plot:
        plots_dir = Path(results_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        for item_id in train_data.item_ids[:3]:
            fig, ax = plt.subplots(figsize=(12, 5))
            
            train_item = train_data.loc[item_id]
            test_item = test_data.loc[item_id]
            pred_item = predictions.loc[item_id]
            
            ax.plot(train_item.index[-200:], train_item["target"].iloc[-200:], label="Train", color="blue", alpha=0.7)
            ax.plot(test_item.index, test_item["target"], label="Actual", color="green", linewidth=2)
            ax.plot(pred_item.index, pred_item["mean"], label="Forecast (Fine-tuned)", color="red", linestyle="--", linewidth=2)
            
            q_cols = [c for c in pred_item.columns if c not in ["mean"]]
            if len(q_cols) >= 2:
                low_q = min(q_cols, key=lambda x: float(x))
                high_q = max(q_cols, key=lambda x: float(x))
                ax.fill_between(
                    pred_item.index,
                    pred_item[low_q],
                    pred_item[high_q],
                    alpha=0.2, color="red", label=f"{low_q}-{high_q}"
                )
            
            ax.set_title(f"Chronos-2 Fine-Tuned: {item_id}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / f"finetuned_{item_id}.png", dpi=200, bbox_inches='tight')
            plt.close()
    
    # Cleanup model to save space
    cleanup_model_dir(model_dir)
    
    print(f"\n{'='*60}")
    print("Fine-Tuning Results:")
    print(f"{'='*60}")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nResults saved to: {results_dir}")
    
    return results


def run_compare(
    train_data: TimeSeriesDataFrame,
    val_data: Optional[TimeSeriesDataFrame],
    test_data: TimeSeriesDataFrame,
    model_path: str = "/root/chronos-2-model",
    prediction_length: int = 96,
    time_limit: int = 900,
    results_dir: str = "/root/chronos-2-project/results",
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Compare zero-shot vs fine-tuned Chronos-2 in a single run.
    Trains both models and produces a comparison table.
    
    Args:
        train_data: Training data
        val_data: Validation data (optional)
        test_data: Test data
        model_path: Path to local Chronos-2 model
        prediction_length: Forecast horizon
        time_limit: Total time limit (shared between both models)
        results_dir: Directory to save results
        plot: Whether to generate plots
    
    Returns:
        Combined results dictionary
    """
    os.makedirs(results_dir, exist_ok=True)
    model_dir = get_temp_model_dir() + "_compare"
    
    print(f"\n{'='*60}")
    print("Chronos-2: Zero-Shot vs Fine-Tuned Comparison")
    print(f"{'='*60}")
    
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=model_dir,
    )
    
    # Train both zero-shot and fine-tuned models
    fit_kwargs = {
        "train_data": train_data,
        "hyperparameters": {
            "Chronos2": [
                # Zero-shot model
                {
                    "model_path": model_path,
                    "ag_args": {"name_suffix": "ZeroShot"},
                },
                # Fine-tuned model with LoRA (efficient, small footprint)
                {
                    "model_path": model_path,
                    "fine_tune": True,
                    "fine_tune_mode": "lora",
                    "ag_args": {"name_suffix": "FineTuned"},
                },
            ]
        },
        "time_limit": time_limit,
        "enable_ensemble": False,
    }
    
    if val_data is not None:
        fit_kwargs["tuning_data"] = val_data
    
    start_time = time.time()
    predictor.fit(**fit_kwargs)
    fit_time = time.time() - start_time
    
    # Evaluate both models
    leaderboard = predictor.leaderboard(test_data)
    
    # Get individual scores
    zs_scores = predictor.evaluate(test_data, model="Chronos2ZeroShot")
    ft_scores = predictor.evaluate(test_data, model="Chronos2FineTuned")
    
    # Save results
    results = {
        "mode": "compare",
        "model": "Chronos2",
        "model_path": model_path,
        "prediction_length": prediction_length,
        "fit_time": fit_time,
        "zero_shot_scores": zs_scores,
        "fine_tuned_scores": ft_scores,
        "leaderboard": leaderboard.to_dict(),
    }
    
    results_path = Path(results_dir) / "compare_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    leaderboard.to_csv(Path(results_dir) / "leaderboard.csv")
    
    # Plot comparison
    if plot:
        plots_dir = Path(results_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        for item_id in train_data.item_ids[:2]:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            
            train_item = train_data.loc[item_id]
            test_item = test_data.loc[item_id]
            
            zs_pred = predictor.predict(train_data, model="Chronos2ZeroShot").loc[item_id]
            ft_pred = predictor.predict(train_data, model="Chronos2FineTuned").loc[item_id]
            
            for ax, pred, title in zip(axes, [zs_pred, ft_pred], ["Zero-Shot", "Fine-Tuned"]):
                ax.plot(train_item.index[-200:], train_item["target"].iloc[-200:], label="Train", color="blue", alpha=0.7)
                ax.plot(test_item.index, test_item["target"], label="Actual", color="green", linewidth=2)
                ax.plot(pred.index, pred["mean"], label="Forecast", color="red", linestyle="--", linewidth=2)
                ax.set_title(f"{title}: {item_id}")
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / f"compare_{item_id}.png", dpi=200, bbox_inches='tight')
            plt.close()
    
    # Cleanup model to save space
    cleanup_model_dir(model_dir)
    
    print(f"\n{'='*60}")
    print("Comparison Results:")
    print(f"{'='*60}")
    print(f"\nZero-Shot:")
    for metric, value in zs_scores.items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nFine-Tuned:")
    for metric, value in ft_scores.items():
        print(f"  {metric}: {value:.4f}")
    print(f"\nResults saved to: {results_dir}")
    
    return results


if __name__ == "__main__":
    from ett_loader import prepare_ett_for_chronos
    
    print("Testing Chronos-2 experiments on ETTh1...")
    train, val, test = prepare_ett_for_chronos("ETTh1", prediction_length=24, data_dir="/root/datasets/ett-dataset")
    
    # Use only 2 items for quick test
    train_small = train.loc[["OT", "HUFL"]]
    val_small = val.loc[["OT", "HUFL"]]
    test_small = test.loc[["OT", "HUFL"]]
    
    # Test zero-shot
    print("\n>>> Testing Zero-Shot...")
    zs_results = run_zero_shot(
        train_small, test_small,
        prediction_length=24,
        time_limit=60,
        results_dir="/root/chronos-2-project/results/test",
        plot=True,
    )
    
    print("\nChronos-2 experiment test complete!")
