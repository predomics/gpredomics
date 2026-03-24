# Technical implementation

Gpredomics has been fully documented since version 0.7.6. To view the technical documentation, simply enter:

```
cargo doc --open
```

Some additional technical documentation will be available in the next version.

## Key structural notes

### Individual struct

The `Individual` struct (in `src/individual.rs`) uses the following notable design choices:

- **`features: BTreeMap<usize, i8>`**: Features are stored in a `BTreeMap` rather than a `HashMap` to ensure deterministic iteration order. This guarantees reproducible hashing, scoring, and serialization regardless of insertion order. The keys are feature indices and values are coefficient signs.

- **`cls: ClassificationMetrics`**: Classification metrics (AUC, threshold, sensitivity, specificity, accuracy, F1, MCC, PPV, NPV, G-mean) are grouped into a dedicated `ClassificationMetrics` struct rather than being stored as individual fields. This keeps the `Individual` struct organized and makes it straightforward to pass metrics around as a unit.

### ACO module

The `src/aco.rs` module implements Ant Colony Optimization as a Max-Min Ant System (MMAS) variant. It follows the same `Population`-based interface as GA and Beam, constructing `Individual` models via probabilistic feature selection guided by a pheromone matrix. See [aco.md](aco.md) for algorithmic details.

## Release checklist

- Update `Cargo.toml` version.
- Update `README.md` version badge, citation block, and `Last updated`.
- Update `Last updated` in all changed md fifiles in docs.
- Update `CITATION.cff` version.

*Last updated: v0.8.3*
