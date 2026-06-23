"""用最优激进配置重新跑全部6个数据集的LoRA"""
import sys
sys.path.insert(0, '/root/chronos-2-project/src')
from transfer_learning_benchmark import load_dataset, DATASET_CONFIGS, cleanup_model_dir
from autogluon.timeseries import TimeSeriesPredictor
import time
import pandas as pd
from pathlib import Path

# 激进LoRA配置
aggressive_params = {
    "model_path": "/root/chronos-2-model",
    "fine_tune": True,
    "fine_tune_mode": "lora",
    "fine_tune_lr": 1e-4,
    "fine_tune_steps": 1000,
    "eval_during_fine_tune": True,
}

datasets = ['hospital', 'fred_md', 'car_parts_without_missing', 'covid_deaths', 'sp500', 'crypto']
results = []

for ds_name in datasets:
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
    results.append({
        "dataset": ds_name,
        "method": "aggressive_lora",
        "prediction_length": config.prediction_length,
        "fit_time": fit_time,
        "score_MASE": mase,
    })
    cleanup_model_dir(model_dir)

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv("/root/chronos-2-project/results/aggressive_all_results.csv", index=False)
print("\n=== Aggressive LoRA Results ===")
print(results_df.to_string())

# 同时打印与默认LoRA的对比
print("\n=== 对比默认LoRA ===")
default = pd.read_csv("/root/chronos-2-project/results/transfer_learning_benchmark/results.csv")
default_lora = default[default['method'] == 'lora'][['dataset', 'score_MASE']].copy()
default_lora['score_MASE'] = -default_lora['score_MASE']
default_lora.columns = ['dataset', 'default_MASE']

merged = results_df[['dataset', 'score_MASE']].merge(default_lora, on='dataset')
merged['improvement'] = (merged['default_MASE'] - merged['score_MASE']) / merged['default_MASE'] * 100
print(merged.to_string())
