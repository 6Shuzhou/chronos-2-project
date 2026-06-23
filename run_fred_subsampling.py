"""FRED-MD subsampling experiment to validate negative transfer at different scales."""
import sys
sys.path.insert(0, '/root/chronos-2-project/src')
from monash_loader import prepare_monash_dataset
from autogluon.timeseries import TimeSeriesPredictor
import shutil
import pandas as pd
import numpy as np

# Load full fred_md data
train_full, val_full, test_full = prepare_monash_dataset('fred_md', prediction_length=12)

all_items = train_full.index.levels[0].tolist()
print(f"Total items: {len(all_items)}")

# Subsample sizes (fred_md has 107 items)
sizes = [50, 100, len(all_items)]
results = []

for size in sizes:
    print(f"\n{'='*60}")
    print(f"Subsampling {size} items from fred_md")
    print(f"{'='*60}")
    
    np.random.seed(42)
    selected = np.random.choice(all_items, size=size, replace=False)
    
    train = train_full.loc[selected]
    val = val_full.loc[selected]
    test = test_full.loc[selected]
    
    # Zero-shot
    model_dir = f"/tmp/fred_sub_{size}_zs"
    shutil.rmtree(model_dir, ignore_errors=True)
    
    pred = TimeSeriesPredictor(prediction_length=12, path=model_dir, eval_metric="MASE")
    pred.fit(train_data=train, tuning_data=val,
             hyperparameters={"Chronos2": {"model_path": "/root/chronos-2-model"}},
             enable_ensemble=False)
    zs_scores = pred.evaluate(test, model="Chronos2")
    zs_mase = -zs_scores["MASE"]
    shutil.rmtree(model_dir, ignore_errors=True)
    
    # LoRA-ES
    model_dir = f"/tmp/fred_sub_{size}_lora"
    shutil.rmtree(model_dir, ignore_errors=True)
    
    pred = TimeSeriesPredictor(prediction_length=12, path=model_dir, eval_metric="MASE")
    pred.fit(train_data=train, tuning_data=val,
             hyperparameters={"Chronos2": {
                 "model_path": "/root/chronos-2-model",
                 "fine_tune": True, "fine_tune_mode": "lora",
                 "fine_tune_lr": 1e-4, "fine_tune_steps": 1000,
                 "eval_during_fine_tune": True,
             }},
             enable_ensemble=False)
    lora_scores = pred.evaluate(test, model="Chronos2")
    lora_mase = -lora_scores["MASE"]
    shutil.rmtree(model_dir, ignore_errors=True)
    
    gain = (zs_mase - lora_mase) / zs_mase * 100
    
    print(f"Size={size:4d}: ZS={zs_mase:.4f}, LoRA-ES={lora_mase:.4f}, Gain={gain:+.1f}%")
    results.append({
        'size': size,
        'zero_shot_mase': zs_mase,
        'lora_mase': lora_mase,
        'gain_pct': gain,
    })

results_df = pd.DataFrame(results)
results_df.to_csv('/root/chronos-2-project/results/fred_subsampling_results.csv', index=False)
print("\n=== FRED-MD Subsampling Results ===")
print(results_df.to_string())
