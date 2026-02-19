# Population

Population stands for a group of Individual

## Population diversity

### Niche concept

In the context of Gpredomics, the niche concept refers to the partitioning of the population into subgroups (niches) based on language and data type characteristics. This approach is inspired by ecological niches, where different species occupy distinct roles in an ecosystem.

Niches are used to maintain diversity within the population and to ensure that the evolutionary process explores a broader range of model structures. By selecting and evolving individuals within each niche, Gpredomics avoids premature convergence to a single model type and increases the chances of discovering high-performing, diverse solutions.

In the genetic algorithm (GA), the parameter `select_niche_pct` controls the proportion of top-performing individuals selected from each niche at every epoch (generation). Specifically, for each combination of language and data type present in the population, the algorithm selects the top `select_niche_pct` percent of individuals within that subgroup. These selected individuals are guaranteed to be included as parents for the next generation, ensuring that all niches are represented in the evolutionary process. This mechanism helps maintain diversity and prevents the loss of potentially valuable model types that might otherwise be outcompeted in the global population. The remaining parents are selected according to other criteria, such as elite or random selection, as defined by the GA parameters.

### Diversity filtration

Population diversity filtration is a mechanism designed to prevent the population from becoming too homogeneous, which can hinder the discovery of novel and high-performing models. In Gpredomics, this is controlled by the `forced_diversity_pct` parameter. At specified intervals (epochs), the population of parents is filtered to ensure that individuals are sufficiently different from each other, typically using a dissimilarity metric such as the signed Jaccard distance between feature sets.

For diversity filtration, the population is split into two main groups: (1) all binary, ternary, and pow2 models are grouped together, and (2) ratio models are treated separately. Diversity is enforced independently within each of these two groups, ensuring that both types of model structures maintain sufficient internal diversity.

When diversity filtration is triggered, individuals that are too similar to others (i.e., their dissimilarity is below the threshold set by `forced_diversity_pct`) are removed from the parent pool. This encourages the retention of a broader variety of model structures and feature combinations, reducing the risk of premature convergence and improving the robustness of the evolutionary search.

This process is especially important in later generations, where selection pressure can otherwise lead to a loss of diversity. The frequency of diversity filtration is controlled by the `forced_diversity_epochs` parameter, allowing users to balance exploration and exploitation according to their needs.

## Population subset

### Family of Best Models

The Family of Best Models (FBM) is a statistically defined subset of the population that contains all models whose performance is not significantly worse than the best observed model. Rather than focusing solely on the single best model, the FBM includes all individuals whose fit (e.g., AUC, accuracy) exceeds a threshold determined by a binomial confidence interval around the best model's score.

**Mathematical definition:**

Let $N$ be the number of individuals in the population, and let $f_1 \geq f_2 \geq \dots \geq f_N$ be their fit scores, sorted in decreasing order. The best model has score $f_1$. For a given confidence level $1-\alpha$, the lower bound of the binomial confidence interval for $f_1$ is computed as:

$$
	ext{lower\_bound} = \text{BinomialCI}(f_1, N, \alpha)
$$

All individuals with $f_i > \text{lower\_bound}$ are included in the FBM:

$$
	ext{FBM} = \{ i \mid f_i > \text{lower\_bound} \}
$$

This approach ensures that the FBM contains all models that are statistically indistinguishable from the best, given the sample size and the chosen $\alpha$. If the fit metric is not in $[0,1]$, a default top 5% selection is used instead.

The FBM is used in Gpredomics to guide feature selection, model interpretation, and to provide a robust set of candidate models for downstream analysis. The strictness and size of the FBM can be tuned via the `best_models_criterion` parameter (interpreted as $\alpha$).

Note that the composition of the FBM depends mainly on the course of the evolutionary process and the distribution of the fit values, which can both vary between runs. For more consistent comparisons across experiments, it is sometimes preferable to use a stable criterion such as selecting a fixed percentage of the best models.

### FBM confidence interval methods

The `fbm_ci_method` parameter controls which binomial confidence interval method is used to determine the lower bound around the best model's accuracy. Different methods provide different trade-offs between coverage probability (how often the true parameter lies within the interval) and interval width (which directly controls the size of the FBM).

Let $p = f_1$ (best model's accuracy treated as a binomial proportion), $n$ be the number of samples, $\alpha$ the significance level, and $z = \Phi^{-1}(1 - \alpha/2)$ the standard normal quantile.

#### Wald (`wald`)

The simplest normal-approximation interval:

$$\text{lower} = p - z \sqrt{\frac{p(1-p)}{n}}$$

Fast to compute but has well-known deficiencies: poor coverage when $p$ is near 0 or 1, and can produce bounds outside $[0, 1]$. Narrowest of all methods, resulting in the smallest FBM.

#### Wald with continuity correction (`wald_continuity`, alias `blaise`)

Adds a continuity correction of $\frac{1}{2n}$ to compensate for the discrete-to-continuous approximation:

$$\text{lower} = p - \left(\frac{1}{2n} + z \sqrt{\frac{p(1-p)}{n}}\right)$$

Slightly wider than Wald. Improves coverage but can still exceed $[0, 1]$.

#### Wilson score (`wilson`) — default

The Wilson score interval, derived by inverting the score test rather than the Wald test:

$$\text{lower} = \frac{p + \frac{z^2}{2n} - z\sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

**This is the default method.** It provides near-nominal coverage across all sample sizes and proportions, is always within $[0, 1]$, and is the method recommended by Brown, Cai & DasGupta (2001) in their comprehensive comparative study of binomial confidence intervals.

#### Agresti-Coull (`agresti_coull`)

Also known as the "add two successes and two failures" method. Constructs a Wald interval on the adjusted proportion $\tilde{p}$:

$$\tilde{n} = n + z^2, \quad \tilde{p} = \frac{np + z^2/2}{\tilde{n}}, \quad \text{lower} = \tilde{p} - z\sqrt{\frac{\tilde{p}(1-\tilde{p})}{\tilde{n}}}$$

Close to Wilson in coverage and width, with a simpler derivation. Recommended by Agresti & Coull (1998) as a practical alternative.

#### Clopper-Pearson (`clopper_pearson`)

The exact interval based on inverting the binomial test using Beta distribution quantiles:

$$\text{lower} = B^{-1}\!\left(\frac{\alpha}{2};\, x,\, n - x + 1\right)$$

where $x = \lfloor p \cdot n \rceil$ and $B^{-1}$ is the quantile function of the Beta distribution. This is the only method that guarantees at least the nominal coverage level, but at the cost of being conservative (over-covering). Produces the widest intervals and therefore the largest FBM.

### Comparison of methods

| Method | Width | Coverage | Bounds | Recommended for |
|--------|-------|----------|--------|-----------------|
| `wald` | Narrowest | Poor near 0/1 | Can exceed [0,1] | Legacy compatibility |
| `wald_continuity` | Narrow | Slightly better | Can exceed [0,1] | Discrete correction |
| **`wilson`** | **Moderate** | **Near-nominal** | **Always [0,1]** | **General use (default)** |
| `agresti_coull` | Moderate | Near-nominal | Can slightly exceed [0,1] | Alternative to Wilson |
| `clopper_pearson` | Widest | Conservative | Always [0,1] | When guaranteed coverage is needed |

### Configuration

The CI method is configured separately for each algorithm stage:

- **Voting** (jury selection): `voting.fbm_ci_method` — controls which models become jury experts
- **Cross-validation**: `cv.cv_fbm_ci_method` — controls FBM selection within CV folds
- **Beam search**: `beam.fbm_ci_method` — controls best-model selection between beam steps

All default to `wilson`. Existing param.yaml files without these fields are backward-compatible and will use the default.

### References

- Wilson, E.B. (1927). "Probable Inference, the Law of Succession, and Statistical Inference." *Journal of the American Statistical Association* 22(158):209–212. doi:[10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953)

- Agresti, A. & Coull, B.A. (1998). "Approximate is Better than 'Exact' for Interval Estimation of Binomial Proportions." *The American Statistician* 52(2):119–126. doi:[10.1080/00031305.1998.10480550](https://doi.org/10.1080/00031305.1998.10480550)

- Brown, L.D., Cai, T.T. & DasGupta, A. (2001). "Interval Estimation for a Binomial Proportion." *Statistical Science* 16(2):101–133. doi:[10.1214/ss/1009213286](https://doi.org/10.1214/ss/1009213286)

- Clopper, C.J. & Pearson, E.S. (1934). "The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial." *Biometrika* 26(4):404–413. doi:[10.2307/2331986](https://doi.org/10.2307/2331986)

*Last updated: v0.7.7*