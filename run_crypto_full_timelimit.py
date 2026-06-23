import sys
sys.path.insert(0, '/root/chronos-2-project/src')
from transfer_learning_benchmark import load_dataset, DATASET_CONFIGS, cleanup_model_dir
import shutil

config = DATASET_CONFIGS["crypto"]
train, val, test = load_dataset(config)

# Run full FT with 5 min time limit
model_dir = "/tmp/chronos2_benchmark_crypto_full"
cleanup_model_dir(model_dir)

from autogluon.timeseries import TimeSeriesPredictor
import time

predictor = TimeSeriesPredictor(
    prediction_length=config.prediction_length,
    path=model_dir,
    eval_metric="MASE",
)

start_fit = time.time()
predictor.fit(
    train_data=train,
    tuning_data=val,
    hyperparameters={"Chronos2": {"model_path": "/root/chronos-2-model", "fine_tune": True, "fine_tune_mode": "full"}},
    enable_ensemble=False,
    time_limit=300,  # 5 minutes max
)
fit_time = time.time() - start_fit

start_pred = time.time()
predictions = predictor.predict(train, model="Chronos2")
pred_time = time.time() - start_pred

scores = predictor.evaluate(test, model="Chronos2")
print(f"Scores: {scores}")
print(f"Fit time: {fit_time:.2f}s, Pred time: {pred_time:.2f}s")

cleanup_model_dir(model_dir)

# Append to results
import pandas as pd
from pathlib import Path

csv_path = Path("/root/chronos-2-project/results/transfer_learning_benchmark/results.csv")
results_df = pd.read_csv(csv_path)

new_row = {
    "dataset": "crypto",
    "method": "full",
    "prediction_length": config.prediction_length,
    "fit_time": fit_time,
    "pred_time": pred_time,
    "num_series": train.num_items,
    "train_size": len(train),
    "test_size": len(test),
    "score_MASE": scores.get("MASE"),
    "error": None,
}
results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
results_df = results_df.drop_duplicates(subset=["dataset", "method"], keep="last")
results_df.to_csv(csv_path, index=False)
print("Saved to results.csv")
