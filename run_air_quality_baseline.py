"""Run baseline models for air_quality and append to baseline_results.csv"""
import sys
sys.path.insert(0, '/root/chronos-2-project/src')
from transfer_learning_benchmark import load_dataset, DATASET_CONFIGS, cleanup_model_dir
from autogluon.timeseries import TimeSeriesPredictor
import time
import pandas as pd
from pathlib import Path

baselines = [
    ("SeasonalNaive", {}, None),
    ("DeepAR", {}, 60),
    ("PatchTST", {}, 60),
]

ds_name = 'air_quality'
config = DATASET_CONFIGS[ds_name]
train, val, test = load_dataset(config)

results = []

for model_name, model_kwargs, time_limit in baselines:
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name} | Model: {model_name}")
    print(f"{'='*60}")
    
    model_dir = f"/tmp/baseline_{ds_name}_{model_name}"
    cleanup_model_dir(model_dir)
    
    try:
        predictor = TimeSeriesPredictor(
            prediction_length=config.prediction_length,
            path=model_dir,
            eval_metric="MASE",
        )
        
        fit_kwargs = {
            "train_data": train,
            "hyperparameters": {model_name: model_kwargs},
            "enable_ensemble": False,
        }
        if time_limit is not None:
            fit_kwargs["time_limit"] = time_limit
        
        start = time.time()
        predictor.fit(**fit_kwargs)
        fit_time = time.time() - start
        
        scores = predictor.evaluate(test, model=model_name)
        mase = -scores["MASE"]
        
        print(f"Result: MASE={mase:.4f}, time={fit_time:.1f}s")
        results.append({
            "dataset": ds_name,
            "method": model_name,
            "prediction_length": config.prediction_length,
            "fit_time": fit_time,
            "score_MASE": mase,
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({
            "dataset": ds_name,
            "method": model_name,
            "prediction_length": config.prediction_length,
            "score_MASE": None,
            "error": str(e),
        })
    finally:
        cleanup_model_dir(model_dir)

# Append to existing results
output_path = Path("/root/chronos-2-project/results/baseline_results.csv")
existing = pd.read_csv(output_path)
new_df = pd.DataFrame(results)
updated = pd.concat([existing, new_df], ignore_index=True)
updated.to_csv(output_path, index=False)
print("\nUpdated baseline_results.csv")
print(updated.to_string())
