"""精简版激进微调对比实验"""
import sys
sys.path.insert(0, '/root/chronos-2-project/src')
from transfer_learning_benchmark import load_dataset, DATASET_CONFIGS, cleanup_model_dir
from autogluon.timeseries import TimeSeriesPredictor
import time
import pandas as pd
from pathlib import Path

datasets = ['covid_deaths', 'crypto']
configs = [
    {"name": "default", "params": {
        "model_path": "/root/chronos-2-model",
        "fine_tune": True, "fine_tune_mode": "lora",
        "fine_tune_lr": 1e-5, "fine_tune_steps": 1000, "eval_during_fine_tune": False,
    }},
    {"name": "lr_1e-4", "params": {
        "model_path": "/root/chronos-2-model",
        "fine_tune": True, "fine_tune_mode": "lora",
        "fine_tune_lr": 1e-4, "fine_tune_steps": 1000, "eval_during_fine_tune": True,
    }},
    {"name": "lr_5e-5_steps_2000", "params": {
        "model_path": "/root/chronos-2-model",
        "fine_tune": True, "fine_tune_mode": "lora",
        "fine_tune_lr": 5e-5, "fine_tune_steps": 2000, "eval_during_fine_tune": True,
    }},
]

results = []
for ds_name in datasets:
    config = DATASET_CONFIGS[ds_name]
    train, val, test = load_dataset(config)
    
    for cfg in configs:
        model_dir = f"/tmp/chronos2_aggr_{ds_name}_{cfg['name']}"
        cleanup_model_dir(model_dir)
        
        predictor = TimeSeriesPredictor(
            prediction_length=config.prediction_length, path=model_dir, eval_metric="MASE")
        
        start = time.time()
        predictor.fit(train_data=train, tuning_data=val,
                      hyperparameters={"Chronos2": cfg['params']}, enable_ensemble=False)
        fit_time = time.time() - start
        
        scores = predictor.evaluate(test, model="Chronos2")
        mase = -scores["MASE"]
        
        print(f"[{ds_name}/{cfg['name']}] MASE={mase:.4f}, time={fit_time:.1f}s")
        results.append({"dataset": ds_name, "config": cfg['name'], "MASE": mase, "fit_time": fit_time})
        cleanup_model_dir(model_dir)

pd.DataFrame(results).to_csv("/root/chronos-2-project/results/aggressive_finetune_results.csv", index=False)
print("\n=== Results ===")
print(pd.DataFrame(results).to_string())
