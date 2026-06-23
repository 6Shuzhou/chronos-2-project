"""
Covariate mechanism study: Does temporal covariates improve Chronos-2?
Compares: no covariates vs. with temporal covariates × ZS vs LoRA
"""
import sys
sys.path.insert(0, '/root/chronos-2-project/src')
from transfer_learning_benchmark import load_dataset, DATASET_CONFIGS, cleanup_model_dir
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import shutil
import time
import pandas as pd
from pathlib import Path


def add_temporal_covariates(ts_df: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
    """Add temporal covariates (dayofweek, month, quarter, is_weekend, dayofyear)."""
    df = ts_df.reset_index()
    time_col = 'timestamp'
    df['dayofweek'] = df[time_col].dt.dayofweek.astype(float)
    df['month'] = df[time_col].dt.month.astype(float)
    df['quarter'] = df[time_col].dt.quarter.astype(float)
    df['is_weekend'] = (df[time_col].dt.dayofweek >= 5).astype(float)
    df['dayofyear'] = df[time_col].dt.dayofyear.astype(float)
    return TimeSeriesDataFrame.from_data_frame(df, id_column='item_id', timestamp_column=time_col)


def run_single_experiment(
    dataset_name: str,
    train_data: TimeSeriesDataFrame,
    val_data: TimeSeriesDataFrame,
    test_data: TimeSeriesDataFrame,
    method: str,
    use_covariates: bool,
    prediction_length: int,
) -> dict:
    """Run one experiment configuration."""
    suffix = "cov" if use_covariates else "nocov"
    model_dir = f"/tmp/chronos2_cov_{dataset_name}_{method}_{suffix}"
    cleanup_model_dir(model_dir)
    
    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=model_dir,
        eval_metric="MASE",
    )
    
    # Build hyperparameters
    chronos_params = {"model_path": "/root/chronos-2-model"}
    if method == "lora":
        chronos_params.update({
            "fine_tune": True,
            "fine_tune_mode": "lora",
            "fine_tune_lr": 1e-4,
            "fine_tune_steps": 1000,
            "eval_during_fine_tune": True,
        })
    
    fit_kwargs = {
        "train_data": train_data,
        "tuning_data": val_data,
        "hyperparameters": {"Chronos2": chronos_params},
        "enable_ensemble": False,
    }
    
    start = time.time()
    predictor.fit(**fit_kwargs)
    fit_time = time.time() - start
    
    # For prediction, we need known_covariates from test data
    # Extract covariate columns from test
    cov_cols = [c for c in test_data.columns if c != 'target']
    if use_covariates and len(cov_cols) > 0:
        known_cov = test_data[cov_cols]
        predictions = predictor.predict(train_data, known_covariates=known_cov, model="Chronos2")
    else:
        predictions = predictor.predict(train_data, model="Chronos2")
    
    scores = predictor.evaluate(test_data, model="Chronos2")
    mase = -scores["MASE"]
    
    cleanup_model_dir(model_dir)
    
    print(f"  [{dataset_name}/{method}/{suffix}] MASE={mase:.4f}, time={fit_time:.1f}s")
    
    return {
        "dataset": dataset_name,
        "method": method,
        "use_covariates": use_covariates,
        "MASE": mase,
        "fit_time": fit_time,
    }


def main():
    datasets = ['hospital', 'fred_md', 'car_parts_without_missing', 'covid_deaths', 'sp500', 'crypto']
    results = []
    
    for ds_name in datasets:
        config = DATASET_CONFIGS[ds_name]
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        
        # Load base data
        train, val, test = load_dataset(config)
        
        # Add covariates
        train_cov = add_temporal_covariates(train)
        val_cov = add_temporal_covariates(val)
        test_cov = add_temporal_covariates(test)
        
        for method in ['zero_shot', 'lora']:
            # Without covariates
            result_nocov = run_single_experiment(
                ds_name, train, val, test, method, False, config.prediction_length
            )
            results.append(result_nocov)
            
            # With covariates
            result_cov = run_single_experiment(
                ds_name, train_cov, val_cov, test_cov, method, True, config.prediction_length
            )
            results.append(result_cov)
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = Path("/root/chronos-2-project/results/covariate_experiment_results.csv")
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("Covariate Experiment Summary")
    print("="*60)
    
    pivot = results_df.pivot_table(
        index='dataset', columns=['method', 'use_covariates'], values='MASE', aggfunc='first'
    )
    print(pivot.round(4).to_string())
    
    print("\nCovariate benefit (% improvement over no-covariates):")
    for ds in datasets:
        for method in ['zero_shot', 'lora']:
            ds_data = results_df[(results_df['dataset']==ds) & (results_df['method']==method)]
            nocov = ds_data[ds_data['use_covariates']==False]['MASE'].values[0]
            cov = ds_data[ds_data['use_covariates']==True]['MASE'].values[0]
            benefit = (nocov - cov) / nocov * 100
            print(f"  {ds:25s} {method:10s}: {benefit:+.2f}%")


if __name__ == "__main__":
    main()
