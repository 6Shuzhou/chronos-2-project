"""
Subsampling experiment: crypto dataset at different scales.
Tests how dataset size affects LoRA fine-tuning benefit.
"""
import sys
sys.path.insert(0, '/root/chronos-2-project/src')
from csv_ts_loader import CSVTimeseriesLoader
from autogluon.timeseries import TimeSeriesPredictor
import shutil
import time
import pandas as pd
import numpy as np

# Load full crypto data
loader = CSVTimeseriesLoader('/root/datasets/crypto')
train_full, val_full, test_full = loader.load()

all_items = train_full.index.levels[0].tolist()
print(f"Total items: {len(all_items)}")

# Subsample sizes
sizes = [50, 100, 250, 500, 1000, len(all_items)]
results = []

for size in sizes:
    print(f"\n{'='*60}")
    print(f"Subsampling {size} items")
    print(f"{'='*60}")
    
    np.random.seed(42)
    selected = np.random.choice(all_items, size=size, replace=False)
    
    train = train_full.loc[selected]
    val = val_full.loc[selected]
    test = test_full.loc[selected]
    
    # Zero-shot
    model_dir = f"/tmp/subsample_{size}_zs"
    shutil.rmtree(model_dir, ignore_errors=True)
    
    pred = TimeSeriesPredictor(prediction_length=30, path=model_dir, eval_metric="MASE")
    pred.fit(train_data=train, tuning_data=val,
             hyperparameters={"Chronos2": {"model_path": "/root/chronos-2-model"}},
             enable_ensemble=False)
    zs_scores = pred.evaluate(test, model="Chronos2")
    zs_mase = -zs_scores["MASE"]
    shutil.rmtree(model_dir, ignore_errors=True)
    
    # LoRA
    model_dir = f"/tmp/subsample_{size}_lora"
    shutil.rmtree(model_dir, ignore_errors=True)
    
    pred = TimeSeriesPredictor(prediction_length=30, path=model_dir, eval_metric="MASE")
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
    
    print(f"Size={size:4d}: ZS={zs_mase:.4f}, LoRA={lora_mase:.4f}, Gain={gain:+.1f}%")
    results.append({
        'size': size,
        'zero_shot_mase': zs_mase,
        'lora_mase': lora_mase,
        'gain_pct': gain,
    })

results_df = pd.DataFrame(results)
results_df.to_csv('/root/chronos-2-project/results/subsampling_results.csv', index=False)
print("\n=== Subsampling Results ===")
print(results_df.to_string())

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(results_df['size'], results_df['zero_shot_mase'], 'o-', label='Zero-shot', linewidth=2)
axes[0].plot(results_df['size'], results_df['lora_mase'], 's-', label='LoRA', linewidth=2)
axes[0].set_xlabel('Number of Series')
axes[0].set_ylabel('MASE (lower is better)')
axes[0].set_title('Performance vs. Dataset Scale')
axes[0].set_xscale('log')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(results_df['size'].astype(str), results_df['gain_pct'], color=['#3498db']*5 + ['#e74c3c'], width=0.6)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Number of Series')
axes[1].set_ylabel('LoRA Gain (%)')
axes[1].set_title('Fine-tuning Benefit vs. Dataset Scale')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/root/chronos-2-project/results/subsampling_curve.png', dpi=200, bbox_inches='tight')
plt.close()
print("\nPlot saved to: /root/chronos-2-project/results/subsampling_curve.png")
