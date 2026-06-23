"""Generate case-study predictions and plots for positive/negative transfer."""
import os
import sys
import shutil
import time
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transfer_learning_benchmark import load_dataset, DATASET_CONFIGS, cleanup_model_dir
from autogluon.timeseries import TimeSeriesPredictor

warnings.filterwarnings('ignore')

RESULTS_DIR = Path('/root/chronos-2-project/results')
PLOTS_DIR = RESULTS_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)
PRED_DIR = RESULTS_DIR / 'case_study'
PRED_DIR.mkdir(exist_ok=True)

MODEL_PATH = "/root/chronos-2-model"

CASES = {
    "covid_deaths": {
        "label": "Positive transfer: COVID-19 deaths",
        "item_id": None,  # will pick series with largest positive delta
    },
    "fred_md": {
        "label": "Negative transfer: FRED-MD",
        "item_id": None,  # will pick series with largest negative delta
    },
}


def run_and_save_predictions(dataset_name: str, method: str, method_label: str):
    """Run zero-shot or LoRA-ES and save per-series predictions."""
    config = DATASET_CONFIGS[dataset_name]
    train, val, test = load_dataset(config)
    
    model_dir = f"/tmp/chronos2_case_{dataset_name}_{method}"
    cleanup_model_dir(model_dir)
    
    chronos2_params = {"model_path": MODEL_PATH}
    if method == "lora_es":
        chronos2_params.update({
            "fine_tune": True,
            "fine_tune_mode": "lora",
            "fine_tune_lr": 1e-4,
            "fine_tune_steps": 1000,
            "eval_during_fine_tune": True,
        })
    
    predictor = TimeSeriesPredictor(
        prediction_length=config.prediction_length,
        path=model_dir,
        eval_metric="MASE",
    )
    
    fit_kwargs = {
        "train_data": train,
        "hyperparameters": {"Chronos2": chronos2_params},
        "enable_ensemble": False,
    }
    if val is not None:
        fit_kwargs["tuning_data"] = val
    
    print(f"[{dataset_name}/{method}] Fitting...")
    start = time.time()
    predictor.fit(**fit_kwargs)
    print(f"[{dataset_name}/{method}] Fit done in {time.time()-start:.1f}s")
    
    predictions = predictor.predict(train, model="Chronos2")
    
    # Save predictions
    pred_file = PRED_DIR / f"{dataset_name}_{method}_predictions.csv"
    predictions.to_csv(pred_file)
    print(f"Saved {pred_file}")
    
    cleanup_model_dir(model_dir)
    return predictions


def select_series(dataset_name: str, method: str):
    """Select the series with largest absolute transfer gain for plotting."""
    config = DATASET_CONFIGS[dataset_name]
    train, val, test = load_dataset(config)
    
    zs = pd.read_csv(PRED_DIR / f"{dataset_name}_zero_shot_predictions.csv")
    es = pd.read_csv(PRED_DIR / f"{dataset_name}_lora_es_predictions.csv")
    
    # Align with test set
    test_df = test.reset_index()
    test_df = test_df.rename(columns={"target": "actual"})
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])
    zs["timestamp"] = pd.to_datetime(zs["timestamp"])
    es["timestamp"] = pd.to_datetime(es["timestamp"])
    
    # Compute per-series MASE-like error (MAE)
    def series_mae(pred_df, value_col="mean"):
        merged = test_df.merge(pred_df[["item_id", "timestamp", value_col]],
                               on=["item_id", "timestamp"], how="inner")
        merged["abs_err"] = (merged["actual"] - merged[value_col]).abs()
        return merged.groupby("item_id")["abs_err"].mean()
    
    mae_zs = series_mae(zs)
    mae_es = series_mae(es)
    
    delta = (mae_zs - mae_es) / mae_zs  # positive = ES better
    
    if method == "best_positive":
        item_id = delta.idxmax()
    else:  # best_negative
        item_id = delta.idxmin()
    
    print(f"[{dataset_name}] Selected series '{item_id}' with delta={delta.loc[item_id]:.3f}")
    return item_id


def get_case_data(dataset_name: str, item_id: str):
    """Return history, actual, zero-shot and LoRA-ES series for one item."""
    config = DATASET_CONFIGS[dataset_name]
    train, val, test = load_dataset(config)

    train_series = train.xs(item_id, level="item_id")["target"]
    test_series = test.xs(item_id, level="item_id")["target"]

    zs = pd.read_csv(PRED_DIR / f"{dataset_name}_zero_shot_predictions.csv")
    es = pd.read_csv(PRED_DIR / f"{dataset_name}_lora_es_predictions.csv")

    zs_series = zs[zs["item_id"] == item_id].set_index("timestamp")["mean"]
    es_series = es[es["item_id"] == item_id].set_index("timestamp")["mean"]

    train_series.index = pd.to_datetime(train_series.index)
    test_series.index = pd.to_datetime(test_series.index)
    zs_series.index = pd.to_datetime(zs_series.index)
    es_series.index = pd.to_datetime(es_series.index)

    # The test TimeSeriesDataFrame contains full history (train+val+test).
    # Restrict the ground-truth to the prediction horizon only.
    pred_index = zs_series.index
    actual_series = test_series.loc[pred_index]

    # History is the tail of the training series immediately before the horizon.
    hist = train_series.tail(3 * config.prediction_length)
    return hist, actual_series, zs_series, es_series


def plot_case_study(output_path: Path):
    """Combined 1x2 figure with positive and negative transfer examples."""
    covid_item = select_series("covid_deaths", "best_positive")
    fred_item = select_series("fred_md", "best_negative")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=False)

    for ax, (dataset_name, item_id, title) in zip(
        axes,
        [
            ("covid_deaths", covid_item, "(a) Positive transfer: COVID-19 deaths"),
            ("fred_md", fred_item, "(b) Negative transfer: FRED-MD"),
        ],
    ):
        hist, actual, zs, es = get_case_data(dataset_name, item_id)

        ax.plot(hist.index, hist.values, label="History", color="black", linewidth=1.5)
        ax.plot(actual.index, actual.values, label="Actual", color="black", linewidth=2, linestyle="--")
        ax.plot(zs.index, zs.values, label="ICL (zero-shot)", color="#3498db", linewidth=2)
        ax.plot(es.index, es.values, label="LoRA-ES", color="#9b59b6", linewidth=2)

        ax.axvline(x=hist.index[-1], color="gray", linestyle=":", alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot {output_path}")


def main():
    for dataset_name in CASES.keys():
        # Run if predictions don't exist
        for method in ["zero_shot", "lora_es"]:
            pred_file = PRED_DIR / f"{dataset_name}_{method}_predictions.csv"
            if not pred_file.exists():
                run_and_save_predictions(dataset_name, method, method)
            else:
                print(f"Using cached {pred_file}")

    plot_case_study(PLOTS_DIR / "figure5_case_study.png")

    print("\nCase study plot generated:")
    print("  - results/plots/figure5_case_study.png")


if __name__ == "__main__":
    main()
