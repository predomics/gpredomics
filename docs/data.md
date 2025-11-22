# Dealing with Gpredomics data

Gpredomics data must follow a specific .tsv format to be correctly interpreted by the rust binary.

## Format

The X and y files must be structured as follows: 

`X.tsv` : features by rows and samples by columns; first column contains feature names, subsequent columns contain numeric values per sample.
| feature |	sample_a | sample_b	| sample_c
| :-- | :-- | :-- | :-- 
| feature_1 | 0.10 | 0.20 | 0.30
| feature_2 | 0.00 | 0.05 | 0.10
| feature_3 | 0.90 | 0.80 | 0.70

`y.tsv`: two‑column TSV mapping sample to class; the first line (header) is ignored; classes: 0 (negative), 1 (positive), 2 (unknown, ignored in metrics).
| sample | class
| :-- | :-- 
| sample_a| 0
| sample_b| 1

Note: The X file can be transposed (samples as rows) if the `features_in_rows` parameter is set to false. 
Gpredomics automatically aligns samples between X and Y files based on their IDs.

It is possible to specify an external test set via the `Xtest` and `ytest` parameters. If no test set is provided, a split holdout can be generated automatically in Gpredomics according to the value of the `holdout_ratio` parameter. This split is stratified first by class, then by the `stratify_by` variable (if specified, see below) to ensure a good balanced representation in the training and test sets.

## Data annotations

### Feature annotations

It is possible to provide Gpredomics with specific annotations for features via a .tsv file whose path is specified in the `feature_annotations` parameter. This file may contain `prior_weight`, `feature_penalty`, and additional tags. It must be structured as follows: 

| feature |	prior_weight | feature_penalty | order | taste
| :-- | :-- | :-- | :-- | :-- 
| Penicillium_camemberti | 1 | 0.01| Eurotiales | yummy
| Rhizopus_stolonifer | 2 | 0.5 | Mucorales | yuck
| Penicillium_glaucum | 1 | 0.01 | Eurotiales | yummy


Tags columns (e.g., order, taste): Used only to enrich the final results display and help the user identify potential biological patterns.
`prior_weight`: Influences the generation of the initial population in the genetic algorithm. Features with higher weights have a higher probability of being selected in randomly generated individuals.
`feature_penalty`: Allows the addition of a custom "soft" penalty per variable. The penalization is obtained by calculating the weighted average (based on absolute feature coefficients) of the penalty values for all selected features. Specifically, a cost proportional to this average is subtracted from the model's fitness. If penalties are too powerful or not powerful enough, it is possible to modify the weight using the `user_feature_penalties_weight` parameter.

### Sample annotations

It is also possible to provide Gpredomics with annotations associated with samples (e.g., hospital, batch), allowing folds to be stratified according to both class and metadata. For more information, see the cv.md documentation.

*Last updated: v0.7.4*