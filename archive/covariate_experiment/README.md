# Covariate Experiment (Archived)

## Status
This experiment has been removed from the main paper due to methodological concerns.

## Why it was removed

The experiment tested whether basic temporal covariates (day-of-week, month, quarter, weekend indicator, day-of-year) improve Chronos-2's zero-shot or LoRA performance. However, the design has fundamental issues:

1. **Redundant information**: Chronos-2's pre-training already encodes temporal patterns from raw target values. Explicit temporal features provide little new information.

2. **Zero-shot unfairness**: The pre-trained model was never exposed to covariate inputs during pre-training. Zero-shot + covariates tests model unfamiliarity, not feature usefulness.

3. **Insufficient adaptation capacity**: Standard LoRA (rank=8, lr=1e-5/1e-4, 1000 steps) may not be sufficient to learn new covariate-target relationships.

4. **Weak feature engineering**: Cyclical variables (day-of-week, month, day-of-year) were treated as linear without proper encoding or normalization.

## Result summary (for reference only)

Temporal covariates did not provide consistent improvements across the 6 datasets. Most configurations showed slight degradation, with only car_parts LoRA showing a marginal improvement (~2.3%).

Because of the methodological limitations above, these results should not be interpreted as evidence that covariates are useless for Chronos-2, but rather that this particular experimental setup was not well-suited to answer that question.

## Files

- `covariate_experiment_results.csv`: Raw results from the 24 experimental configurations.
