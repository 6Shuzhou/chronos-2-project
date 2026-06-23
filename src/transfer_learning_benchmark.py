"""
Chronos-2 Transfer Learning Benchmark
Compares Zero-shot vs LoRA Fine-tuning vs Full Fine-tuning
across multiple time series datasets.

Datasets:
- ETT: ETTh1, ETTh2, ETTm1, ETTm2
- Monash: hospital, covid_deaths, fred_md, car_parts_without_missing
"""

import os
import sys
import json
import time
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ett_loader import prepare_ett_for_chronos
from monash_loader import prepare_monash_dataset
from csv_ts_loader import CSVTimeseriesLoader


warnings.filterwarnings('ignore')


@dataclass
class DatasetConfig:
    name: str
    prediction_length: int
    family: str  # "ett" or "monash"


# Dataset configurations for transfer learning research
DATASET_CONFIGS = {
    "hospital": DatasetConfig("hospital", 12, "monash"),
    "covid_deaths": DatasetConfig("covid_deaths", 30, "monash"),
    "fred_md": DatasetConfig("fred_md", 12, "monash"),
    "car_parts_without_missing": DatasetConfig("car_parts_without_missing", 12, "monash"),
    "sp500": DatasetConfig("sp500", 30, "csv"),
    "crypto": DatasetConfig("crypto", 30, "csv"),
    "air_quality": DatasetConfig("air_quality", 30, "csv"),
}


def load_dataset(config: DatasetConfig) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame]:
    """Load train/val/test for a dataset."""
    if config.family == "ett":
        return prepare_ett_for_chronos(
            config.name,
            prediction_length=config.prediction_length,
            data_dir="/root/datasets/ett-dataset",
        )
    elif config.family == "monash":
        return prepare_monash_dataset(
            config.name,
            prediction_length=config.prediction_length,
            data_dir="/root/datasets/monash_tsf",
        )
    elif config.family == "csv":
        loader = CSVTimeseriesLoader(f"/root/datasets/{config.name}")
        return loader.load()
    else:
        raise ValueError(f"Unknown family: {config.family}")


def get_temp_model_dir(dataset: str, method: str) -> str:
    """Get temporary model directory."""
    return f"/tmp/chronos2_benchmark_{dataset}_{method}"


def cleanup_model_dir(path: str):
    """Remove model directory to save space."""
    if os.path.exists(path):
        shutil.rmtree(path)


def run_experiment(
    dataset_name: str,
    method: str,  # "zero_shot", "lora", "full"
    train_data: TimeSeriesDataFrame,
    val_data: Optional[TimeSeriesDataFrame],
    test_data: TimeSeriesDataFrame,
    prediction_length: int,
    model_path: str = "/root/chronos-2-model",
) -> Dict:
    """
    Run a single experiment for a dataset and method.
    
    Returns:
        Dictionary with metrics and timing info.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Method: {method}")
    print(f"{'='*60}")
    
    model_dir = get_temp_model_dir(dataset_name, method)
    cleanup_model_dir(model_dir)
    
    # Build hyperparameters based on method
    chronos2_params = {"model_path": model_path}
    if method == "lora":
        chronos2_params.update({
            "fine_tune": True,
            "fine_tune_mode": "lora",
        })
    elif method == "full":
        chronos2_params.update({
            "fine_tune": True,
            "fine_tune_mode": "full",
        })
    
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=model_dir,
        eval_metric="MASE",
    )
    
    fit_kwargs = {
        "train_data": train_data,
        "hyperparameters": {"Chronos2": chronos2_params},
        "enable_ensemble": False,
    }
    if val_data is not None:
        fit_kwargs["tuning_data"] = val_data
    
    start_fit = time.time()
    predictor.fit(**fit_kwargs)
    fit_time = time.time() - start_fit
    
    # Predict
    start_pred = time.time()
    predictions = predictor.predict(train_data, model="Chronos2")
    pred_time = time.time() - start_pred
    
    # Evaluate
    scores = predictor.evaluate(test_data, model="Chronos2")
    
    # Get model info
    leaderboard = predictor.leaderboard(silent=True)
    
    cleanup_model_dir(model_dir)
    
    result = {
        "dataset": dataset_name,
        "method": method,
        "prediction_length": prediction_length,
        "fit_time": fit_time,
        "pred_time": pred_time,
        "num_series": train_data.num_items,
        "train_size": len(train_data),
        "test_size": len(test_data),
    }
    result.update({f"score_{k}": v for k, v in scores.items()})
    
    print(f"Scores: {scores}")
    print(f"Fit time: {fit_time:.2f}s, Pred time: {pred_time:.2f}s")
    
    return result


def run_benchmark(
    datasets: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    model_path: str = "/root/chronos-2-model",
    output_dir: str = "/root/chronos-2-project/results/transfer_learning_benchmark",
) -> pd.DataFrame:
    """
    Run full benchmark across datasets and methods.
    
    Args:
        datasets: List of dataset names to evaluate
        methods: List of methods ["zero_shot", "lora", "full"]
        model_path: Path to local Chronos-2 model
        output_dir: Directory to save results
    
    Returns:
        DataFrame with all results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())
    if methods is None:
        methods = ["zero_shot", "lora", "full"]
    
    all_results = []
    
    for dataset_name in datasets:
        config = DATASET_CONFIGS[dataset_name]
        print(f"\n\n{'#'*70}")
        print(f"# Processing dataset: {dataset_name}")
        print(f"{'#'*70}")
        
        train, val, test = load_dataset(config)
        
        for method in methods:
            try:
                result = run_experiment(
                    dataset_name=dataset_name,
                    method=method,
                    train_data=train,
                    val_data=val,
                    test_data=test,
                    prediction_length=config.prediction_length,
                    model_path=model_path,
                )
                all_results.append(result)
                
            except Exception as e:
                print(f"ERROR on {dataset_name}/{method}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "dataset": dataset_name,
                    "method": method,
                    "error": str(e),
                })
    
    # Final results - merge with existing to avoid overwriting
    results_df = pd.DataFrame(all_results)
    csv_path = Path(output_dir) / "results.csv"
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        results_df = pd.concat([old_df, results_df], ignore_index=True)
        # Drop duplicates based on dataset + method, keeping latest
        results_df = results_df.drop_duplicates(subset=["dataset", "method"], keep="last")
    results_df.to_csv(csv_path, index=False)
    
    # Save full JSON (append mode)
    json_path = Path(output_dir) / "results.json"
    if json_path.exists():
        with open(json_path) as f:
            old_results = json.load(f)
        all_results = old_results + all_results
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("Benchmark Complete!")
    print(f"{'='*60}")
    print(results_df.to_string())
    
    return results_df


def plot_results(results_df: pd.DataFrame, output_dir: str):
    """Generate comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter successful results
    if "error" in results_df.columns:
        df = results_df[results_df["error"].isna()].copy()
    else:
        df = results_df.copy()
    
    if len(df) == 0:
        print("No successful results to plot.")
        return
    
    # 1. MASE comparison bar plot
    if "score_MASE" in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        datasets = sorted(df["dataset"].unique())
        methods = ["zero_shot", "lora", "full"]
        x = np.arange(len(datasets))
        width = 0.25
        
        for i, method in enumerate(methods):
            method_df = df[df["method"] == method].set_index("dataset")
            values = [method_df.loc[d, "score_MASE"] if d in method_df.index else np.nan for d in datasets]
            # MASE in autogluon is negative (higher is better convention), flip sign
            values = [-v if not np.isnan(v) else v for v in values]
            ax.bar(x + i * width, values, width, label=method)
        
        ax.set_xlabel("Dataset")
        ax.set_ylabel("MASE (lower is better)")
        ax.set_title("Chronos-2 Transfer Learning: MASE Comparison")
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "mase_comparison.png", dpi=200, bbox_inches='tight')
        plt.close()
    
    # 2. Fit time comparison
    if "fit_time" in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for method in ["zero_shot", "lora", "full"]:
            method_df = df[df["method"] == method]
            if len(method_df) > 0:
                ax.scatter(method_df["train_size"], method_df["fit_time"], label=method, s=100, alpha=0.7)
        
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("Fit Time (seconds)")
        ax.set_title("Training Time vs Dataset Size")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "fit_time_comparison.png", dpi=200, bbox_inches='tight')
        plt.close()
    
    # 3. Pivot table
    if "score_MASE" in df.columns:
        pivot = df.pivot_table(
            index="dataset",
            columns="method",
            values="score_MASE",
            aggfunc="first",
        )
        pivot = -pivot  # Flip sign so lower is better
        pivot.to_csv(Path(output_dir) / "mase_pivot.csv")
        
        # Save formatted markdown table
        with open(Path(output_dir) / "mase_pivot.md", "w") as f:
            f.write("# Chronos-2 Transfer Learning Benchmark\n\n")
            f.write("## MASE (lower is better)\n\n")
            f.write(pivot.round(4).to_markdown())
            f.write("\n\n")
            
            if "fit_time" in df.columns:
                time_pivot = df.pivot_table(
                    index="dataset",
                    columns="method",
                    values="fit_time",
                    aggfunc="first",
                )
                f.write("## Fit Time (seconds)\n\n")
                f.write(time_pivot.round(2).to_markdown())
                f.write("\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Datasets to benchmark")
    parser.add_argument("--methods", nargs="+", default=None,
                        choices=["zero_shot", "lora", "full"],
                        help="Methods to compare")
    parser.add_argument("--output_dir", default="/root/chronos-2-project/results/transfer_learning_benchmark",
                        help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()
    
    results_df = run_benchmark(
        datasets=args.datasets,
        methods=args.methods,
        output_dir=args.output_dir,
    )
    
    if args.plot:
        plot_results(results_df, args.output_dir)
