# Proposal: two new data types — `tanh_prev` and `log1p`

**Status**: implemented. See `src/individual.rs` (`TANH_PREV_TYPE`, `LOG1P_TYPE`, `evaluate_tanh_prev`, `evaluate_log1p`), `src/param.rs` (epsilon validation in `get()`), and the accompanying tests. User-facing documentation in `docs/individual.md`.

## Context

Gpredomics currently offers four data types (`src/individual.rs`):

| Constant | String | Formula | Issue we want to address |
|---|---|---|---|
| `RAW_TYPE` | `raw` | `x` | no transform |
| `PREVALENCE_TYPE` | `prevalence`, `prev` | `1 if x > ε else 0` | hard step — loses all information about abundance magnitude, and a noisy reading near ε flips the feature on/off |
| `LOG_TYPE` | `log` | `ln(x/ε)` | uses an epsilon hack; `x = 0` gives `-∞` and dominates downstream sums; picking ε is a compromise |
| `ZSCORE_TYPE` | `zscore`, `z`, `standardized` | `(x - μ) / σ` | baseline data type for standardized pipelines |

We want to add two data types to the list, motivated by metagenomics practice where:
- Abundances are **log-normal**, not linear.
- Very low abundances (≤ 1e-5) are **noisy / uncertain** — they may or may not reflect a real signal.
- Abundances ≥ 1e-4 are typically **reliably detected**.
- Common "typical" abundances for present species are in the 0.1 range (rarely — 1–2 species per sample — but present in most samples).

## Type A — `tanh_prev`: smooth alternative to prevalence

### Formula

```
tanh(x / x_ref)
```

- `tanh(0) = 0`, `tanh(∞) = 1` — range [0, 1] for non-negative abundance data.
- `tanh(x_ref) ≈ 0.762` — at `x = x_ref`, the feature is "mostly present but not fully saturated".
- Monotonic, smooth, differentiable.

This replaces the hard 0/1 step of `prevalence` with a smooth transition, so a small change in abundance near the detection threshold no longer flips the feature on/off discontinuously.

### `x_ref` — global, not per-feature

**Decision**: `x_ref` is a **single global parameter**, not per-feature.

In principle a per-feature reference (like zscore's per-feature mean/std) could be estimated from training data. However, this is a poor fit for the typical use case:

1. Abundance distributions are **log-normal**, so linear statistics like Q3 or mean of non-zero values are unstable and not meaningful.
2. The biologically relevant thresholds (uncertain ≤ 1e-5, reliable ≥ 1e-4) are **features of the measurement technology**, not of individual taxa.
3. A single global `x_ref` gives the user direct, interpretable control.

### No hardcoded default — user must set `x_ref` explicitly

The appropriate `x_ref` depends strongly on the **profiler** (tools differ in sensitivity and normalization) and the **sequencing depth** (deeper sequencing lowers the detection floor). A hardcoded default would silently mis-scale the transform for anyone whose data doesn't match the implicit assumption.

Therefore: **when `data_type` contains `tanh_prev`, `data_type_epsilon` must be explicitly set in `param.yaml`**. If unset (i.e., still at its internal fallback), `validate()` returns an error with a message telling the user to set it to a value appropriate for their data.

The `param.yaml` shipped with gpredomics shows an example value (e.g., `1e-4` for a typical metagenomic profiler at standard depth) with a comment explaining what it represents, but this is documentation — not a default that silently applies.

### Transition behavior (for reference)

With `x_ref = 1e-4`:

| x | `x / x_ref` | `tanh(x/x_ref)` | interpretation |
|---|---|---|---|
| 0 | 0 | 0.000 | absent |
| 1e-5 | 0.1 | 0.100 | uncertain — leans absent |
| 5e-5 | 0.5 | 0.462 | transition |
| 1e-4 | 1.0 | 0.762 | present (canonical "reliably detected") |
| 5e-4 | 5.0 | 0.9999 | saturated |
| 1e-3+ | ≥ 10 | ≈ 1.000 | fully present |

A user who wants the transition shifted (e.g., center it at 5e-5 rather than 1e-4) simply changes `x_ref`.

### Name

`tanh_prev`. Accepted strings: `"tanh_prev"`, `"tanh"`.

## Type B — `log1p`: zero-safe alternative to log

### Rejected candidates first

The user suggested `x·log(x)` and `tanh(x)·log(x)`. Both are **non-monotonic**:
- `x·log(x)` is 0 at `x = 0` and at `x = 1`, negative on `(0, 1)`, positive above 1.
- `tanh(x)·log(x)` has the same qualitative shape.

Non-monotonicity is fatal for the linear/ratio model interpretation. If a feature has coefficient `+1`, the model assumes "higher x → higher score". With a non-monotonic transform, `x = 0.5` contributes negatively while `x = 2` contributes positively, making feature directionality meaningless.

### Formula

```
ln(1 + x / x_ref)
```

- `log1p(0) = 0` — no `-∞` problem; `x = 0` is the natural baseline.
- For `x << x_ref`: `log1p(x/x_ref) ≈ x/x_ref` — linear, reflecting that very small abundances are mostly noise.
- For `x >> x_ref`: `log1p(x/x_ref) ≈ ln(x/x_ref) = ln(x) - ln(x_ref)` — recovers log scaling for reliable abundances.
- Strictly monotonic, smooth, differentiable, well-defined on all of `[0, ∞)`.

`log1p` is the textbook solution to the zero-log problem. No epsilon hack, no diverging term, and the transition point `x = x_ref` separates the "noise regime" (linear) from the "signal regime" (log).

### `x_ref` — global, user-set, no hardcoded default

Yes, `log1p` should take an `x_ref` parameter. Without it, `log1p(x)` with `x` in the 1e-5 range is essentially zero everywhere (since `log1p(1e-5) ≈ 1e-5`), which loses all resolution for metagenomic abundances. With `x_ref`, you rescale to a sensible operating regime.

Like `tanh_prev`, the correct `x_ref` depends on the profiler and sequencing depth, so `log1p` also **requires the user to set `data_type_epsilon` explicitly**. `validate()` errors out if `data_type` contains `log1p` and the parameter is not set.

### Behavior for reference

With `x_ref = 1e-5`, compared to the current `LOG_TYPE` (`ln(x/ε)` with `ε = 1e-5`):

| x | current `log` | proposed `log1p` | difference |
|---|---|---|---|
| 0 | **−∞** (broken) | 0.000 | ✓ fixed |
| 1e-5 | 0.000 | 0.693 | small offset |
| 1e-4 | 2.303 | 2.398 | negligible |
| 1e-3 | 4.605 | 4.615 | negligible |
| 1e-2 | 6.908 | 6.908 | identical |
| 1e-1 | 9.210 | 9.210 | identical |

Above `x ≈ 10·x_ref` the two transforms are indistinguishable; below that, `log1p` smoothly goes to 0 instead of diverging.

### Name

`log1p`. Accepted strings: `"log1p"`.

### Alternatives considered

- **`asinh(x/x_ref) = ln(x/x_ref + √(1 + (x/x_ref)²))`** — similar shape, well-defined, but the "log of 2x" asymptote has no biological interpretation for non-negative abundance data, and it's less familiar to biologists than `log1p`.
- **Box-Cox `(x^λ - 1)/λ`** — family of transforms parameterized by λ; adds complexity without clear benefit over `log1p` for this use case.

`log1p` wins on simplicity, familiarity, and direct substitutability for the current `log` type.

## Implementation notes

### New constants (`src/individual.rs`)

```rust
pub const TANH_PREV_TYPE: u8 = 4;
pub const LOG1P_TYPE: u8 = 5;
```

### String parsing

Extend `data_type()`:

```rust
"tanh_prev" | "tanh" => TANH_PREV_TYPE,
"log1p"              => LOG1P_TYPE,
```

### Where `x_ref` lives

Both types use `self.epsilon` (already part of `Individual`) as their `x_ref`. No new field.

- For `LOG_TYPE` today, `epsilon` is used as `(x/epsilon).ln()` — same role.
- For `TANH_PREV_TYPE`, we use `tanh(x/epsilon)`.
- For `LOG1P_TYPE`, we use `(1.0 + x/epsilon).ln()` — i.e., `f64::ln_1p(x/epsilon)`.
- For `PREVALENCE_TYPE` today, `epsilon` is the threshold — same role again.

Reusing `epsilon` keeps the data structure unchanged and serialized experiments remain compatible. The parameter name `data_type_epsilon` in `param.yaml` already reads naturally as "reference value" for these new types.

### Epsilon handling

A single global `data_type_epsilon` is used for all data types, as today.

- `LOG_TYPE` / `PREVALENCE_TYPE` (existing) — current behavior unchanged; a default epsilon is kept for backward compatibility.
- `LOG1P_TYPE` / `TANH_PREV_TYPE` (new) — **no silent default is accepted**. `validate()` errors if `data_type` contains one of these types and the user has not explicitly set `data_type_epsilon` in `param.yaml`. Appropriate values depend on the profiler and sequencing depth and must come from the user.

Detecting "not explicitly set" requires changing the serde default for `data_type_epsilon` from `1e-5` to `Option<f64>` (or a sentinel like `NaN`), so the loader can distinguish "absent from YAML" from "user wrote `1e-5`". The simplest path: `#[serde(default)] pub data_type_epsilon: Option<f64>`, and downstream code unwraps with `1e-5` for `LOG_TYPE` / `PREVALENCE_TYPE` and errors for the new types.

When mixing `tanh_prev` with `log1p` in a single run (via the comma-separated `data_type` list), the user picks one global epsilon that works for their dataset. No per-type override is introduced in v1 — it would add complexity without a concrete use case.

### `evaluate_*` functions

Add `evaluate_tanh_prev` and `evaluate_log1p` in `src/individual.rs`, mirroring the existing `evaluate_raw` / `evaluate_prevalence` / `evaluate_log` structure. Each must handle the three language branches already present in those functions:

- `BINARY_LANG` / `TERNARY_LANG` / `POW2_LANG`: weighted sum of transformed values.
- `RATIO_LANG`: sum transformed values into positive/negative buckets, then divide.
- `MCMC_GENERIC_LANG`: sum transformed values, pass through logistic with betas.

All three branches are straightforward for both new transforms — there is no reason to restrict these types to a subset of languages.

### Display

The current display logic in `src/individual.rs` has a special case for `LOG_TYPE + RATIO_LANG` that renders the ratio as `ln(∏pos) - ln(∏neg)`, using the identity `ln(a)+ln(b) = ln(a·b)`. This identity **does not hold for `log1p` or `tanh_prev`** — they render as ordinary sums / ratios of sums, exactly like `RAW_TYPE`.

Feature formatting: wrap each feature name in the transform function for readability.
- `TANH_PREV_TYPE`: `tanh_prev(feature_name)`
- `LOG1P_TYPE`: `log1p(feature_name)`

Example (binary language, tanh_prev):

```
Class cirrhosis: score ≥ 0.42
score = (tanh_prev(Bacteroides_vulgatus) + tanh_prev(Prevotella_copri))
```

Example (ratio language, log1p):

```
Class cirrhosis: score ≥ 1.30
score = (log1p(feat_A) + log1p(feat_B)) / (log1p(feat_C) + log1p(feat_D))
```

No special Unicode glyphs — the function-call form is self-documenting and reads cleanly in both terminal and log output.

### Training-data stats — none needed

Unlike `ZSCORE_TYPE`, these transforms do **not** require training-data statistics. `x_ref` is a global parameter, so nothing needs to be computed on train data or propagated to test data. This keeps the implementation significantly simpler than `ZSCORE_TYPE`.

### Parameter validation

Add checks in `src/param.rs::validate`:
- `x_ref > 0` (required for both transforms — `tanh(x/0)` and `log1p(x/0)` are undefined).

### Unknown-params whitelist

Add `"log1p"`, `"tanh_prev"`, `"tanh"` as accepted values of `data_type` in any validation / documentation that enumerates data types.

## Summary of decisions

| Question | Decision |
|---|---|
| Name A | `tanh_prev` (also accepts `tanh`) |
| Name B | `log1p` |
| `x_ref` for `tanh_prev` | global (`data_type_epsilon`), **no default** — user must set it explicitly |
| `x_ref` for `log1p` | global (`data_type_epsilon`), **no default** — user must set it explicitly |
| Per-feature stats | **no** — global only, because abundances are log-normal and thresholds are instrument-level |
| Language branches | support all (binary/ternary/pow2/ratio/MCMC) — no restrictions |
| Storage field | reuse existing `epsilon` on `Individual` |
| Training stats | none required |

## Feature preselection — prefilter stays data-type-agnostic

The current prefilter (`feature_minimal_prevalence_pct` using a hard `> epsilon` threshold per sample) is kept unchanged for the new soft types. Rationale:

- The prefilter's role is to drop features that are **essentially zero everywhere** — noise-only columns. Applying `tanh_prev` or `log1p` to a feature whose raw value is below `epsilon` in every sample still yields near-zero contributions, so the feature would be uninformative regardless of the transform.
- The prefilter is a cheap, pre-training noise-floor cut. It does not encode the model's view of presence/absence.
- Users who want to admit borderline features can lower `feature_minimal_prevalence_pct` or `data_type_epsilon` — no type-specific logic needed.
- Adding data-type-specific branches to the prefilter adds code complexity for no demonstrated benefit.

This is documented behavior: prefiltering is identical for all data types.

## Unit tests

All new tests go in the existing `mod tests` block at the bottom of `src/individual.rs` (plus a few parser tests in `src/param.rs`), following the style already established by `test_evaluate_prevalence_weighted_score` and `test_evaluate_log_weighted_score`.

### Parsing and validation

- `test_tanh_prev_data_type_parsing` — `data_type("tanh_prev")` and `data_type("tanh")` both return `TANH_PREV_TYPE`.
- `test_log1p_data_type_parsing` — `data_type("log1p")` returns `LOG1P_TYPE`.
- `test_unknown_datatype_still_panics` — `data_type("tnh")` still panics, to guard against typos being silently accepted.
- `test_validate_tanh_prev_requires_epsilon` — in `param.rs`, a config with `data_type: tanh_prev` and no `data_type_epsilon` returns an `Err` from `validate()` with a message that names `tanh_prev` and `data_type_epsilon`.
- `test_validate_log1p_requires_epsilon` — same for `log1p`.
- `test_validate_mixed_types_require_epsilon` — a config with `data_type: raw,tanh_prev` still requires an explicit epsilon.
- `test_validate_log_keeps_default_epsilon` — `data_type: log` without explicit epsilon continues to work (backward compatibility).
- `test_validate_epsilon_positive` — `data_type_epsilon <= 0` is rejected for `tanh_prev` and `log1p` (both transforms are undefined at `x_ref = 0`).

### `evaluate_tanh_prev` — one test per language branch

All tests use `epsilon = 1e-4` and hand-computed expected values (so regressions are obvious at review time).

- `test_evaluate_tanh_prev_binary` — two positive features, values spanning the transition:
  - `x = 0` → `tanh(0) = 0`
  - `x = 1e-5` → `tanh(0.1) ≈ 0.09966...`
  - `x = 1e-4` → `tanh(1.0) ≈ 0.76159...`
  - `x = 1e-3` → `tanh(10) ≈ 0.99999...`
  - Asserts the sum with `approx_eq!` at ~1e-10 tolerance.
- `test_evaluate_tanh_prev_ternary` — one positive, one negative feature; score = `tanh(x₀/ε) − tanh(x₁/ε)`.
- `test_evaluate_tanh_prev_pow2` — coefficient magnitudes `2` and `-4`, score = `2·tanh(x₀/ε) − 4·tanh(x₁/ε)`.
- `test_evaluate_tanh_prev_ratio` — `RATIO_LANG` with two positive and two negative features; denominator has the standard `+ self.epsilon` guard.
- `test_evaluate_tanh_prev_mcmc_generic` — `MCMC_GENERIC_LANG` with mock `Betas { a, b, c }`, asserts `logistic(pos·a + neg·b + c)`.
- `test_evaluate_tanh_prev_zero_sample_len` — empty `X`, `sample_len = 0` returns an empty vector.
- `test_evaluate_tanh_prev_missing_values` — a sparse `X` with some `(sample, feature)` keys absent; the missing entries are treated as `0.0` (not 0 on the inside of `tanh`, but `tanh(0/ε) = 0`). Mirrors `test_evaluate_prevalence_missing_values`.
- `test_evaluate_tanh_prev_zero_is_zero` — sanity check: every sample with all features at exactly zero gives a score of zero.

### `evaluate_log1p` — one test per language branch

Same structure as tanh tests, with `epsilon = 1e-5` and expected values computed using `f64::ln_1p(x / epsilon)`:

- `test_evaluate_log1p_binary` — values across regimes:
  - `x = 0` → `ln(1) = 0` (the key test — no `-∞`)
  - `x = 1e-5` → `ln(2) ≈ 0.69314...`
  - `x = 1e-4` → `ln(11) ≈ 2.39789...`
  - `x = 1e-3` → `ln(101) ≈ 4.61512...`
- `test_evaluate_log1p_ternary` — one positive, one negative feature.
- `test_evaluate_log1p_pow2` — scaled coefficients.
- `test_evaluate_log1p_ratio` — verifies that `RATIO_LANG + LOG1P_TYPE` uses the plain ratio-of-sums path, NOT the `ln(∏pos) − ln(∏neg)` path used by `LOG_TYPE + RATIO_LANG`. This is the regression guard for the non-identity noted earlier in this doc.
- `test_evaluate_log1p_mcmc_generic` — with mock `Betas`.
- `test_evaluate_log1p_zero_sample_len` — empty input.
- `test_evaluate_log1p_missing_values` — sparse `X`.
- `test_evaluate_log1p_zero_is_zero` — every feature at `x = 0` gives a score of zero (this is the whole point of `log1p`; must not accidentally regress to `log(0) = -∞` or `log(ε)`).
- `test_evaluate_log1p_matches_log_for_large_x` — with `epsilon = 1e-5`, a feature value of `0.1` should give `log1p(1e4) ≈ ln(1e4)` to within `1e-4` relative tolerance — confirms the asymptotic equivalence documented in the behavior table.

### Integration with `evaluate()` dispatch

- `test_evaluate_dispatches_tanh_prev` — `Individual::evaluate(&data)` with `data_type = TANH_PREV_TYPE` routes to `evaluate_tanh_prev` (not `evaluate_zscore`, which has its own early return, and not `evaluate_from_features` before falling through to the wrong branch).
- `test_evaluate_dispatches_log1p` — same for `LOG1P_TYPE`.
- `test_evaluate_unknown_datatype_panics` — confirms the existing `panic!("Unknown data-type {}", other)` fallback still triggers for an out-of-range `data_type` value.

### Feature preselection (documented invariant)

- `test_prefilter_ignores_data_type` — a `Data` object with a known sparse feature (below the prevalence threshold) is filtered out identically whether `data_type` is `raw`, `tanh_prev`, or `log1p`. Documents and locks in the decision that the prefilter is data-type-agnostic.

### Display tests

- `test_display_tanh_prev_binary` — checks that `to_string()` produces something like `tanh_prev(feat0) + tanh_prev(feat1)` (matching literal substring, not the full formatted output, to stay robust to color codes and metric headers).
- `test_display_log1p_ratio` — checks that `RATIO_LANG + LOG1P_TYPE` renders as `(log1p(a) + log1p(b)) / (log1p(c) + log1p(d))`, i.e., the same structure as `RATIO_LANG + RAW_TYPE` with `log1p(...)` wrapping each name — NOT the `ln(∏) − ln(∏)` form used by `LOG_TYPE`.
- `test_display_tanh_prev_pow2_coefficient` — checks that `POW2_LANG` coefficients ≠ 1 still render as a leading factor: `2*tanh_prev(feat0)`.

### Floating-point tolerances

Use `approx::assert_relative_eq!` (or the existing `approx::assert_abs_diff_eq!` the codebase already uses for float comparisons) with `epsilon = 1e-10` for all `tanh_prev` assertions and `epsilon = 1e-12` for `log1p`. Exact `assert_eq!` on `f64` is avoided for these transforms because the intermediate `tanh` / `ln_1p` results are not guaranteed bit-identical across platforms.

### Target coverage

The goal is parity with the existing test coverage for `evaluate_prevalence` and `evaluate_log` (language branches × sample-len edge cases × missing-values × validation). At minimum, each new public function (`evaluate_tanh_prev`, `evaluate_log1p`) must have a test for every language branch it handles.

## All open questions resolved

| Question | Resolution |
|---|---|
| Display formatting | `tanh_prev(feature_name)` / `log1p(feature_name)` — function-call form, no special glyphs |
| Single vs. per-type epsilon | Single global `data_type_epsilon` for v1; revisit only if needed |
| Ratio + log1p display | Ordinary ratio of sums (same as `RATIO_LANG + RAW_TYPE`) — the log identity does not apply |
| Prefilter behavior | Data-type-agnostic — hard `> epsilon` prevalence filter applied uniformly |

The proposal is ready for implementation.
