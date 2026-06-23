"""Generate figures for the paper."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path('/root/chronos-2-project/results')
PLOTS_DIR = RESULTS_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

# Load main results
main = pd.read_csv(RESULTS_DIR / 'transfer_learning_benchmark' / 'results.csv')
main['MASE'] = -main['score_MASE']
main_pivot = main.pivot(index='dataset', columns='method', values='MASE')
main_pivot = main_pivot.rename(index={'car_parts_without_missing': 'car_parts'})

# Load aggressive results and merge
agg = pd.read_csv(RESULTS_DIR / 'aggressive_all_results.csv')
agg['MASE'] = agg['score_MASE']
agg['dataset'] = agg['dataset'].replace({'car_parts_without_missing': 'car_parts'})
agg_pivot = agg.set_index('dataset')['MASE']
main_pivot['lora_es'] = agg_pivot

# Reorder columns
main_pivot = main_pivot[['zero_shot', 'lora', 'full', 'lora_es']]
main_pivot.columns = ['ICL (ZS)', 'LoRA', 'Full FT', 'LoRA-ES']

# Order datasets for plotting
dataset_order = ['hospital', 'fred_md', 'air_quality', 'car_parts', 'sp500', 'crypto', 'covid_deaths']
main_pivot = main_pivot.reindex(dataset_order)

# Clean names
main_pivot.index = ['hospital', 'fred_md', 'air_quality', 'car_parts', 'sp500', 'crypto', 'covid_deaths']

# ============================================================
# Figure 1: Main Results - MASE by dataset and method
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(main_pivot))
width = 0.2
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

for i, col in enumerate(main_pivot.columns):
    ax.bar(x + i * width, main_pivot[col], width, label=col, color=colors[i])

ax.set_xlabel('Dataset')
ax.set_ylabel('MASE (lower is better)')
ax.set_title('Chronos-2 Performance Across Datasets and Adaptation Strategies')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(main_pivot.index, rotation=30, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'figure1_main_results.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 2: Transfer Gain Delta (LoRA-ES vs ICL)
# ============================================================
main_pivot['Delta'] = (main_pivot['ICL (ZS)'] - main_pivot['LoRA-ES']) / main_pivot['ICL (ZS)'] * 100
delta = main_pivot['Delta']

fig, ax = plt.subplots(figsize=(10, 5))
colors_delta = ['#e74c3c' if v < 0 else '#2ecc71' for v in delta]
bars = ax.barh(range(len(delta)), delta, color=colors_delta, alpha=0.8)
ax.set_yticks(range(len(delta)))
ax.set_yticklabels(delta.index)
ax.set_xlabel('Transfer Gain Δ (%)')
ax.set_title('LoRA-ES Transfer Gain Relative to Zero-Shot ICL')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, delta)):
    if val >= 0:
        ax.text(val + 0.8, i, f'{val:+.1f}%',
                va='center', ha='left', fontsize=9, color='black')
    else:
        # Place label inside the negative bar to avoid y-axis overlap
        ax.text(val + 1.0, i, f'{val:+.1f}%',
                va='center', ha='left', fontsize=9, color='white')

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'figure2_transfer_gain.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 3: Baseline Comparison
# ============================================================
baseline = pd.read_csv(RESULTS_DIR / 'baseline_results.csv')
baseline['MASE'] = baseline['score_MASE']
baseline_pivot = baseline.pivot(index='dataset', columns='method', values='MASE')
baseline_pivot = baseline_pivot.reindex(dataset_order)
baseline_pivot.index = ['hospital', 'fred_md', 'air_quality', 'car_parts', 'sp500', 'crypto', 'covid_deaths']

# Add Chronos-2 ICL
chronos_icl = main_pivot['ICL (ZS)']
baseline_pivot['Chronos-2 ICL'] = chronos_icl
baseline_pivot = baseline_pivot[['Chronos-2 ICL', 'SeasonalNaive', 'DeepAR', 'PatchTST']]

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(baseline_pivot))
width = 0.2
colors_base = ['#3498db', '#95a5a6', '#e67e22', '#f1c40f']

for i, col in enumerate(baseline_pivot.columns):
    ax.bar(x + i * width, baseline_pivot[col], width, label=col, color=colors_base[i])

ax.set_xlabel('Dataset')
ax.set_ylabel('MASE (lower is better)')
ax.set_title('Chronos-2 ICL vs. Traditional Forecasting Methods')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(baseline_pivot.index, rotation=30, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'figure3_baselines.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 4: Subsampling Trends
# ============================================================
crypto_sub = pd.read_csv(RESULTS_DIR / 'subsampling_results.csv')
hosp_sub = pd.read_csv(RESULTS_DIR / 'hospital_subsampling_results.csv')
fred_sub = pd.read_csv(RESULTS_DIR / 'fred_subsampling_results.csv')
air_sub = pd.read_csv(RESULTS_DIR / 'air_quality_subsampling_results.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: positive transfer (crypto)
ax = axes[0]
ax.plot(crypto_sub['size'], crypto_sub['gain_pct'], 'o-', color='#2ecc71', linewidth=2, markersize=8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Number of Series')
ax.set_ylabel('Transfer Gain Δ (%)')
ax.set_title('Positive Transfer: Cryptocurrency')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# Right: negative transfer (hospital, fred, air_quality)
ax = axes[1]
ax.plot(hosp_sub['size'], hosp_sub['gain_pct'], 's-', color='#e74c3c', label='hospital', linewidth=2, markersize=8)
ax.plot(fred_sub['size'], fred_sub['gain_pct'], '^-', color='#c0392b', label='fred_md', linewidth=2, markersize=8)
ax.plot(air_sub['size'], air_sub['gain_pct'], 'd-', color='#e67e22', label='air_quality', linewidth=2, markersize=8)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Number of Series')
ax.set_ylabel('Transfer Gain Δ (%)')
ax.set_title('Negative Transfer: Structured Domains')
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(PLOTS_DIR / 'figure4_subsampling.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Figures saved to {PLOTS_DIR}")
print("Generated:")
for f in sorted(PLOTS_DIR.glob('figure*.png')):
    print(f"  - {f.name}")
