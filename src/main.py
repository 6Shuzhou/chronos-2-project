#!/usr/bin/env python3
"""
Chronos-2 Time Series Forecasting - Main Entry Point
Follows AutoGluon official tutorial: https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html

Supports zero-shot inference and fine-tuning (transfer learning) on ETT datasets.
Models are NOT saved to disk to save space - only results are preserved.

Usage:
    # Zero-shot prediction on ETTh1
    python main.py --mode zero_shot --dataset ETTh1 --prediction_length 96

    # Fine-tuning on ETTh1
    python main.py --mode finetune --dataset ETTh1 --prediction_length 96 --time_limit 600

    # Compare zero-shot vs fine-tuned
    python main.py --mode compare --dataset ETTh1 --prediction_length 96 --time_limit 900
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ett_loader import prepare_ett_for_chronos
from chronos2_experiment import run_zero_shot, run_finetune, run_compare


def parse_args():
    parser = argparse.ArgumentParser(description="Chronos-2 Time Series Forecasting")

    parser.add_argument("--mode", type=str, default="zero_shot",
                        choices=["zero_shot", "finetune", "compare"],
                        help="Execution mode")
    parser.add_argument("--dataset", type=str, default="ETTh1",
                        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
                        help="ETT dataset name")
    parser.add_argument("--prediction_length", type=int, default=96,
                        help="Forecast horizon")
    parser.add_argument("--model_path", type=str,
                        default="/root/chronos-2-model",
                        help="Path to local chronos-2 model or HuggingFace repo ID")
    parser.add_argument("--time_limit", type=int, default=300,
                        help="Time limit in seconds for fitting")
    parser.add_argument("--fine_tune_mode", type=str, default="lora",
                        choices=["lora", "full"],
                        help="Fine-tuning mode: lora (efficient, default) or full")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-3,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--fine_tune_steps", type=int, default=1000,
                        help="Number of fine-tuning steps")
    parser.add_argument("--data_dir", type=str,
                        default="/root/datasets/ett-dataset",
                        help="Directory containing ETT datasets")
    parser.add_argument("--results_dir", type=str,
                        default="/root/chronos-2-project/results",
                        help="Output directory for results")
    parser.add_argument("--plot", action="store_true",
                        help="Generate forecast plots")
    parser.add_argument("--no_plot", dest="plot", action="store_false",
                        help="Disable forecast plots")
    parser.set_defaults(plot=True)

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("Chronos-2 Forecasting Project")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Prediction length: {args.prediction_length}")

    # Load dataset
    print("\nLoading dataset...")
    train, val, test = prepare_ett_for_chronos(
        dataset_name=args.dataset,
        prediction_length=args.prediction_length,
        data_dir=args.data_dir,
    )

    # Run experiment
    if args.mode == "zero_shot":
        results = run_zero_shot(
            train_data=train,
            test_data=test,
            model_path=args.model_path,
            prediction_length=args.prediction_length,
            time_limit=args.time_limit,
            results_dir=args.results_dir,
            plot=args.plot,
        )
    elif args.mode == "finetune":
        results = run_finetune(
            train_data=train,
            val_data=val,
            test_data=test,
            model_path=args.model_path,
            prediction_length=args.prediction_length,
            time_limit=args.time_limit,
            fine_tune_mode=args.fine_tune_mode,
            fine_tune_lr=args.fine_tune_lr,
            fine_tune_steps=args.fine_tune_steps,
            results_dir=args.results_dir,
            plot=args.plot,
        )
    elif args.mode == "compare":
        results = run_compare(
            train_data=train,
            val_data=val,
            test_data=test,
            model_path=args.model_path,
            prediction_length=args.prediction_length,
            time_limit=args.time_limit,
            results_dir=args.results_dir,
            plot=args.plot,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"\n{'='*60}")
    print("Experiment Complete!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
