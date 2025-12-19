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

*Last updated: v0.7.6*