# About the threshold confidence interval and associated rejection

## What is a threshold confidence interval?

Version 0.7.3 of Gpredomics introduced the concept of a confidence interval for the decision threshold.
When this feature is enabled, models can reject a sample when the predicted score falls inside a confidence interval
computed around their binary decision threshold:

```
Model #27 Ratio:Raw [k=51] [gen:1] [fit:0.799] AUC 0.923/0.902 | accuracy 0.911/0.864 | sensitivity 0.875/0.909 | specificity 0.957/0.818 | rejection rate 0.122/0.267 | G-means 0.886/0.862 
Class healthy < [Rejection zone - 95% CI: 0.120, 0.225, 0.270] < Class cirrhosis: (msp_0049 + msp_0313 + msp_0380 + msp_0385 + msp_0581 + msp_0602c + msp_0768 + msp_0881 + msp_0884 + msp_0977 + msp_1127 + msp_1453 + msp_1543 + msp_1789 + msp_1793 + msp_1799) / (msp_0017 + msp_0034 + msp_0047 + msp_0048 + msp_0062 + msp_0063 + msp_0088 + msp_0089 + msp_0144 + msp_0151 + msp_0194 + msp_0196 + msp_0198 + msp_0205 + msp_0236 + msp_0263 + msp_0306 + msp_0381 + msp_0450 + msp_0464 + msp_0468 + msp_0558 + msp_0572 + msp_0757 + msp_0763 + msp_0852 + msp_0898 + msp_0917 + msp_1021 + msp_1093 + msp_1139 + msp_1185 + msp_1275 + msp_1323 + msp_1881 + 1e-5)
```
There, the rejection zone is defined between ` 0.120` and `0.270` while `0.225` is the threshold obtained from the entire data set (i.e. without bootstrapping).

Concretely, for each epoch, the threshold of each model is estimated on the full dataset and on a number of
resampled datasets (the number of bootstrap iterations given by `threshold_ci_n_bootstrap`). This yields an empirical
distribution of the threshold and allows estimating how variable the threshold is; from that we derive the model's
uncertainty on individual samples. The confidence zone is controlled by `threshold_ci_alpha`: lowering alpha increases
the confidence level (1 − alpha) and therefore tends to widen the confidence interval (so decreasing alpha makes the
rejection zone larger).

Due to the necessity of computing `threshold_ci_n_bootstrap`, threshold confidence interval has cost of **O(B × N)** where 
B is the number of bootstrap iterations and N the sample size. However, note that even if Gpredomics does accept smaller values 
(the minimal enforced value is 100), it emits warnings when B < 1000 and again when B < 2000. In other words: while small
B is allowed for quick tests, production use should rely on ≥ 1000 (≥ 2000 is even safer). 

## Penalization

Once the confidence interval is estimated, its lower and upper bounds define a rejection zone. Any sample whose score
falls inside that zone is considered rejected and therefore excluded from performance metric computations. The
parameter `threshold_ci_penalty` is used directly as a selection pressure in the genetic algorithm: models with a low
rejection rate (more decisive models) are favored by evolution compared to models that reject many samples.

## When to use threshold confidence intervals

**Recommended scenarios:**
- **High-stakes clinical decisions** where false positives/negatives have severe consequences (e.g., cancer screening, 
  antimicrobial resistance prediction)
- **Regulatory compliance** where model uncertainty must be quantified and reported
- **Imbalanced datasets** where threshold estimation is inherently uncertain
- **Small sample sizes** (N < 200) where threshold variability is higher

**Not recommended when:**
- **Fast prototyping** is prioritized over uncertainty quantification (use B=0 to disable)
- **Very large datasets** (N > 10,000) where threshold is stable and CI computation becomes expensive
- **Streaming/real-time predictions** where abstention is not operationally feasible

## Statistical implementation

The implementation uses a stratified resampling strategy and a percentile-based confidence interval construction. Key
points:

- **Stratified sampling**: bootstrap / subsampling is stratified by class (positive / negative) so class proportions are
  preserved in each replicate.

- **Sampling modes**:
  - If `threshold_ci_frac_bootstrap == 1.0`: classic bootstrap (with replacement). Each bootstrap draw has the same
    nominal size as the corresponding class in the original data.
  - If `0 < threshold_ci_frac_bootstrap < 1.0`: subsampling without replacement; in each replicate a fraction
    `threshold_ci_frac_bootstrap` of the samples from each class is drawn.

- **Subsampling correction (Geyer rescaling)**: When threshold_ci_frac_bootstrap < 1.0, subsample-based variability 
    underestimates full-sample variability. To correct this bias, the code applies Geyer rescaling:
    
      - For each bootstrap replicate b, compute the deviation: δ_b = θ̂_b - θ̂ (where θ̂ is the threshold on full data)
      - Rescale by subsample size: δ*_b = √m · δ_b (where m = subsample size)
      - Compute quantiles on {δ*_b} to get q_lower and q_upper
      - De-pivot with full sample size: CI = [θ̂ - q_upper/√n, θ̂ - q_lower/√n] (where n = full sample size)

    This ensures the CI has the correct coverage even when using subsampling for computational efficiency.

- **Confidence interval building**: percentile method is used. For B bootstrap replicates and significance α the code
  selects quantile indices
  - lower_idx = ceil((α / 2) * (B − 1))
  - upper_idx = floor((1 − α / 2) * (B − 1))
  and uses the sorted bootstrap statistics at those indices (after Geyer rescaling if applicable) as CI bounds.

- **Rejection rate**: defined as the fraction of samples assigned the "undecided" label (i.e., whose score lies between
  the lower and upper bounds). This rejection rate is stored with the model and used to compute the GA penalty as
  fit := fit − threshold_ci_penalty * rejection_rate.

- **Reproducibility**: Bootstrap indices and labels are precomputed once per epoch using the RNG seed, then reused 
  for all individuals in the population. This ensures deterministic results across runs with the same seed.

- **Performance**: the code contains a faster path that precomputes bootstrap indices and labels once per dataset and
  reuses them for all individuals in a population for reproducibility and speed.

## References

- Bootstrap methodology: Efron, B. & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.
- Subsampling correction: Politis, D. N., Romano, J. P., & Wolf, M. (1999). Subsampling. Springer.

*Last updated: v0.7.3*