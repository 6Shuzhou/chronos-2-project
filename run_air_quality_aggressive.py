"""Run aggressive LoRA for air_quality and append to aggressive_all_results.csv"""
import sys
sys.path.insert(0, '/root/chronos-2-project/src')
from transfer_learning_benchmark import load_dataset, DATASET_CONFIGS, cleanup_model_dir
from autogluon.timeseries import TimeSeriesPredictor
import time
import pandas as pd
from pathlib import Path

aggressive_params = {
    "model_path": "/root/chronos-2-model",
    "fine_tune": True,
    "fine_tune_mode": "lora",
    "fine_tune_lr": 1e-4,
    "fine_tune_steps": 1000,
    "eval_during_fine_tune": True,
}

ds_name = 'air_quality'
config = DATASET_CONFIGS[ds_name]
train, val, test = load_dataset(config)

model_dir = f"/tmp/chronos2_aggr_all_{ds_name}"
cleanup_model_dir(model_dir)

predictor = TimeSeriesPredictor(
    prediction_length=config.prediction_length,
    path=model_dir,
    eval_metric="MASE",
)

start = time.time()
predictor.fit(
    train_data=train,
    tuning_data=val,
    hyperparameters={"Chronos2": aggressive_params},
    enable_ensemble=False,
)
fit_time = time.time() - start

scores = predictor.evaluate(test, model="Chronos2")
mase = -scores["MASE"]

print(f"[{ds_name}] Aggressive LoRA MASE={mase:.4f}, time={fit_time:.1f}s")

cleanup_model_dir(model_dir)

# Append to existing results
output_path = Path("/root/chronos-2-project/results/aggressive_all_results.csv")
existing = pd.read_csv(output_path)
new_row = pd.DataFrame([{
    "dataset": ds_name,
    "method": "aggressive_lora",
    "prediction_length": config.prediction_length,
    "fit_time": fit_time,
    "score_MASE": mase,
}])
updated = pd.concat([existing, new_row], ignore_index=True)
updated.to_csv(output_path, index=False)
print("\nUpdated aggressive_all_results.csv")
print(updated.to_string())
