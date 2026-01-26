use crate::data::PreselectionMethod;
use crate::experiment::ImportanceAggregation;
use crate::{beam::BeamMethod, voting::VotingMethod};
use log::warn;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::error::Error;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(non_camel_case_types)]

/// Fit function options for model evaluation
pub enum FitFunction {
    /// Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC)
    auc,
    /// True Positive Rate (TPR), also known as Sensitivity.
    sensitivity,
    /// True Negative Rate (TNR), also known as Specificity.
    specificity,
    /// Matthews Correlation Coefficient (MCC)
    mcc,
    /// Harmonic mean of precision and recall (F1 Score)
    f1_score,
    /// Negative Predictive Value (NPV)
    npv,
    /// Positive Predictive Value (PPV)
    ppv,
    /// Geometric Mean of sensitivity and specificity
    g_mean,
}

/// GPU memory management policies
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum GpuMemoryPolicy {
    /// Strict memory management policy.
    Strict,
    /// Adapt used memory according to available resources.
    Adaptive,
    /// Maximize used memory usage for optimal performance.
    Performance,
}

/// Main parameter structure encompassing all configuration sections
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Param {
    /// General parameters section
    #[serde(default)]
    pub general: General,

    /// Data loading and prefiltering parameters section
    #[serde(default)]
    pub data: Data,

    /// Cross-validation parameters section
    #[serde(default)]
    pub cv: CV,

    /// Genetic Algorithm parameters section
    #[serde(default)]
    pub ga: GA,

    /// Beam search parameters section
    #[serde(default)]
    pub beam: BEAM,

    /// MCMC parameters section
    #[serde(default)]
    pub mcmc: MCMC,

    /// Voting ensemble parameters section
    #[serde(default)]
    pub voting: Voting,

    /// Feature importance computation parameters section
    #[serde(default)]
    pub importance: Importance,

    /// GPU memory management parameters section
    #[serde(default)]
    pub gpu: GPU,

    /// Experimental features section
    #[serde(default)]
    pub experimental: Experimental,

    /// Internal tag for parameter source identification
    #[serde(skip)]
    pub tag: String,
}

/// General parameters
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct General {
    #[serde(default = "seed_default")]
    pub seed: u64,
    #[serde(default = "algorithm_default")]
    pub algo: String,
    #[serde(default = "language_default")]
    pub language: String,
    #[serde(default = "data_type_default")]
    pub data_type: String,
    #[serde(default = "data_type_epsilon_default")]
    pub data_type_epsilon: f64,
    #[serde(default = "one_default")]
    pub thread_number: usize,
    #[serde(default = "log_base_default")]
    pub log_base: String,
    #[serde(default = "log_suffix_default")]
    pub log_suffix: String,
    #[serde(default = "log_level_default")]
    pub log_level: String,
    #[serde(default = "fit_default")]
    pub fit: FitFunction,
    #[serde(default = "zero_default")]
    pub k_penalty: f64,
    #[serde(default = "zero_default")]
    pub fr_penalty: f64,
    #[serde(default = "zero_default")]
    pub bias_penalty: f64,
    #[serde(default = "zero_default")]
    pub threshold_ci_penalty: f64,
    #[serde(default = "zero_default")]
    pub threshold_ci_alpha: f64,
    #[serde(default = "uzero_default")]
    pub threshold_ci_n_bootstrap: usize,
    #[serde(default = "zero_default")]
    pub threshold_ci_frac_bootstrap: f64,
    #[serde(default = "zero_default")]
    pub user_penalties_weight: f64,
    #[serde(default = "n_model_to_display_default")]
    pub n_model_to_display: u32,
    #[serde(default = "false_default")]
    pub gpu: bool,
    #[serde(default = "false_default")]
    pub cv: bool,
    #[serde(default = "true_default")]
    pub display_colorful: bool,
    #[serde(default = "feature_keep_trace_default")]
    pub keep_trace: bool,
    #[serde(default = "save_experiment_default")]
    pub save_exp: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct Data {
    #[serde(default = "empty_string")]
    pub X: String,
    #[serde(default = "empty_string")]
    pub y: String,
    #[serde(default = "empty_string")]
    pub Xtest: String,
    #[serde(default = "empty_string")]
    pub ytest: String,
    #[serde(default = "true_default")] // for retrocompatibility
    pub features_in_rows: bool,
    #[serde(default = "holdout_ratio_default")]
    pub holdout_ratio: f64,
    #[serde(default = "uzero_default")]
    pub max_features_per_class: usize,
    #[serde(default = "feature_selection_method_default")]
    pub feature_selection_method: PreselectionMethod,
    #[serde(default = "feature_minimal_prevalence_pct_default")]
    pub feature_minimal_prevalence_pct: f64,
    #[serde(default = "feature_maximal_adj_pvalue_default")]
    pub feature_maximal_adj_pvalue: f64,
    #[serde(default = "zero_default")]
    pub feature_minimal_feature_value: f64,
    #[serde(default = "feature_minimal_log_abs_bayes_factor_default")]
    pub feature_minimal_log_abs_bayes_factor: f64,
    #[serde(default = "false_default")]
    pub inverse_classes: bool,
    #[serde(default = "uzero_default")]
    pub n_validation_samples: usize,
    #[serde(default = "class_names_default")]
    pub classes: Vec<String>,
    #[serde(default = "empty_string")]
    pub feature_annotations: String,
    #[serde(default = "empty_string")]
    pub sample_annotations: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct CV {
    #[serde(default = "folds_default")]
    pub inner_folds: usize,
    #[serde(default = "zero_default")]
    pub overfit_penalty: f64,
    #[serde(default = "uzero_default")]
    pub resampling_inner_folds_epochs: usize,
    #[serde(default = "folds_default")]
    pub outer_folds: usize,
    #[serde(default = "false_default")]
    pub fit_on_valid: bool,
    #[serde(default = "best_models_ci_alpha_default")]
    pub cv_best_models_ci_alpha: f64,
    #[serde(default = "empty_string")]
    pub stratify_by: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct Voting {
    #[serde(default = "false_default")]
    pub vote: bool,
    #[serde(default = "half_default")]
    pub min_perf: f64,
    #[serde(default = "diversity_voting_default")]
    pub min_diversity: f64,
    #[serde(default = "best_models_ci_alpha_default")]
    pub fbm_ci_alpha: f64,
    #[serde(default = "voting_default")]
    pub method: VotingMethod,
    #[serde(default = "half_default")]
    pub method_threshold: f64,
    #[serde(default = "zero_default")]
    pub threshold_windows_pct: f64,
    #[serde(default = "false_default")]
    pub complete_display: bool,
    #[serde(default = "false_default")]
    pub prune_before_voting: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct GA {
    #[serde(default = "pop_size_default")]
    pub population_size: u32,
    #[serde(default = "max_epochs_default")]
    pub max_epochs: usize,
    #[serde(default = "min_epochs_default")]
    pub min_epochs: usize,
    #[serde(default = "max_age_best_model_default")]
    pub max_age_best_model: usize,
    #[serde(default = "one_default")]
    #[serde(alias = "kmin")]
    pub k_min: usize,
    #[serde(default = "k_max_default")]
    #[serde(alias = "kmax")]
    pub k_max: usize,
    #[serde(default = "ga_elite_pct_default")]
    pub select_elite_pct: f64,
    #[serde(default = "zero_default")]
    pub select_niche_pct: f64,
    #[serde(default = "ga_random_pct_default")]
    pub select_random_pct: f64,
    #[serde(default = "ga_mut_children_pct_default")]
    pub mutated_children_pct: f64,
    #[serde(default = "ga_mut_features_pct_default")]
    pub mutated_features_pct: f64,
    #[serde(default = "ga_mut_non_null_pct_default")]
    pub mutation_non_null_chance_pct: f64,
    #[serde(default = "zero_default")]
    pub forced_diversity_pct: f64,
    #[serde(default = "uzero_default")]
    pub forced_diversity_epochs: usize,
    #[serde(default = "zero_default")]
    pub random_sampling_pct: f64,
    #[serde(default = "uzero_default")]
    pub random_sampling_epochs: usize,
    #[serde(default = "uzero_default")]
    pub n_epochs_before_global: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct BEAM {
    #[serde(default = "beam_method_default")]
    pub method: BeamMethod,
    #[serde(default = "one_default")]
    #[serde(alias = "kmin")]
    pub k_start: usize,
    #[serde(default = "k_max_default")]
    #[serde(alias = "kmax")]
    pub k_stop: usize,
    #[serde(default = "best_models_criterion_default")]
    pub best_models_criterion: f64,
    #[serde(default = "max_nb_of_models_default")]
    pub max_nb_of_models: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct MCMC {
    #[serde(default = "n_iter_default")]
    pub n_iter: usize,
    #[serde(default = "n_burn_default")]
    pub n_burn: usize,
    #[serde(default = "lambda_default")]
    pub lambda: f64,
    #[serde(default = "nmin_default")]
    pub nmin: u32,
    #[serde(default = "empty_string")]
    pub save_trace_outdir: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct GPU {
    #[serde(default = "memory_policy_default")]
    pub memory_policy: GpuMemoryPolicy,
    #[serde(default = "max_total_memory_mb_default")]
    pub max_total_memory_mb: u64,
    #[serde(default = "max_buffer_size_mb_default")]
    pub max_buffer_size_mb: u32,
    #[serde(default = "fallback_to_cpu_default")]
    pub fallback_to_cpu: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct Importance {
    #[serde(default = "false_default")]
    pub compute_importance: bool,
    #[serde(default = "n_permutations_mda_default")]
    pub n_permutations_mda: usize,
    #[serde(default = "false_default")]
    pub scaled_importance: bool,
    #[serde(default = "importance_aggregation_default")]
    pub importance_aggregation: ImportanceAggregation,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[allow(missing_docs)]
pub struct Experimental {}

// Default section definitions

impl Default for General {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for Data {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for Importance {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for Voting {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for CV {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for GA {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for BEAM {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for MCMC {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for Experimental {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Default for GPU {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap_or_else(|_| GPU {
            memory_policy: memory_policy_default(),
            max_total_memory_mb: max_total_memory_mb_default(),
            max_buffer_size_mb: max_buffer_size_mb_default(),
            fallback_to_cpu: fallback_to_cpu_default(),
        })
    }
}

impl Default for Param {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).unwrap()
    }
}

impl Param {
    /// Creates a new Param instance with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Saves param to a YAML file
    ///
    /// # Arguments
    ///
    /// * `param` - Reference to the Param struct to be saved.
    /// * `output_file` - Path to the output YAML file.
    ///
    /// # Returns
    ///
    /// * `Result<(), Box<dyn Error>>` - Ok if saving is successful, Err with an error otherwise.
    ///
    /// # Examples
    /// ```no_run
    /// # use gpredomics::param::Param;
    /// let params = Param::default();
    /// params.save("output_param.yaml")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        let yaml = serde_yaml::to_string(self)?;
        std::fs::write(path, yaml)?;
        Ok(())
    }
}

/// Checks for unknown parameters in the YAML file.
///
/// # Arguments
///
/// * `yaml_value` - The parsed YAML as a serde_yaml::Value.
///
/// # Returns
///
/// * `Result<(), String>` - Ok if no unknown parameters are found, Err with a message otherwise.
fn check_unknown_params(yaml_value: &serde_yaml::Value) -> Result<(), String> {
    // Define all valid parameter keys for each section
    let valid_general_keys: HashSet<&str> = [
        "seed",
        "algo",
        "language",
        "data_type",
        "epsilon",
        "thread_number",
        "log_base",
        "log_suffix",
        "log_level",
        "fit",
        "k_penalty",
        "fr_penalty",
        "bias_penalty",
        "threshold_ci_penalty",
        "threshold_ci_alpha",
        "threshold_ci_n_bootstrap",
        "threshold_ci_frac_bootstrap",
        "user_feature_penalties_weight",
        "user_penalties_weight",
        "n_model_to_display",
        "gpu",
        "cv",
        "display_colorful",
        "keep_trace",
        "save_exp",
        "data_type_epsilon",
    ]
    .iter()
    .cloned()
    .collect();

    let valid_data_keys: HashSet<&str> = [
        "X",
        "y",
        "Xtest",
        "ytest",
        "features_in_rows",
        "holdout_ratio",
        "max_features_per_class",
        "feature_selection_method",
        "feature_minimal_prevalence_pct",
        "feature_maximal_adj_pvalue",
        "feature_minimal_feature_value",
        "feature_minimal_log_abs_bayes_factor",
        "inverse_classes",
        "n_validation_samples",
        "classes",
        "feature_annotations",
        "sample_annotations",
    ]
    .iter()
    .cloned()
    .collect();

    let valid_cv_keys: HashSet<&str> = [
        "inner_folds",
        "overfit_penalty",
        "resampling_inner_folds_epochs",
        "outer_folds",
        "fit_on_valid",
        "cv_best_models_ci_alpha",
        "stratify_by",
    ]
    .iter()
    .cloned()
    .collect();

    let valid_voting_keys: HashSet<&str> = [
        "vote",
        "min_perf",
        "min_diversity",
        "fbm_ci_alpha",
        "method",
        "method_threshold",
        "threshold_windows_pct",
        "complete_display",
        "prune_before_voting",
    ]
    .iter()
    .cloned()
    .collect();

    let valid_ga_keys: HashSet<&str> = [
        "population_size",
        "max_epochs",
        "min_epochs",
        "max_age_best_model",
        "k_min",
        "kmin",
        "k_max",
        "kmax",
        "select_elite_pct",
        "select_niche_pct",
        "select_random_pct",
        "mutated_children_pct",
        "mutated_features_pct",
        "mutation_non_null_chance_pct",
        "forced_diversity_pct",
        "forced_diversity_epochs",
        "random_sampling_pct",
        "random_sampling_epochs",
        "n_epochs_before_global",
    ]
    .iter()
    .cloned()
    .collect();

    let valid_beam_keys: HashSet<&str> = [
        "method",
        "k_start",
        "kmin",
        "k_stop",
        "kmax",
        "best_models_criterion",
        "max_nb_of_models",
    ]
    .iter()
    .cloned()
    .collect();

    let valid_mcmc_keys: HashSet<&str> =
        ["n_iter", "n_burn", "lambda", "nmin", "save_trace_outdir"]
            .iter()
            .cloned()
            .collect();

    let valid_gpu_keys: HashSet<&str> = [
        "memory_policy",
        "max_total_memory_mb",
        "max_buffer_size_mb",
        "fallback_to_cpu",
    ]
    .iter()
    .cloned()
    .collect();

    let valid_importance_keys: HashSet<&str> = [
        "compute_importance",
        "n_permutations_mda",
        "scaled_importance",
        "importance_aggregation",
    ]
    .iter()
    .cloned()
    .collect();

    let valid_experimental_keys: HashSet<&str> = [].iter().cloned().collect();

    let valid_sections: HashSet<&str> = [
        "general",
        "data",
        "cv",
        "voting",
        "ga",
        "beam",
        "mcmc",
        "gpu",
        "importance",
        "experimental",
    ]
    .iter()
    .cloned()
    .collect();

    let mut unknown_params = Vec::new();

    // Check top-level sections
    if let Some(mapping) = yaml_value.as_mapping() {
        for (key, value) in mapping {
            if let Some(section_name) = key.as_str() {
                if !valid_sections.contains(section_name) {
                    unknown_params.push(format!("Unknown section: '{}'", section_name));
                    continue;
                }

                // Check keys within each section
                if let Some(section_mapping) = value.as_mapping() {
                    let valid_keys = match section_name {
                        "general" => &valid_general_keys,
                        "data" => &valid_data_keys,
                        "cv" => &valid_cv_keys,
                        "voting" => &valid_voting_keys,
                        "ga" => &valid_ga_keys,
                        "beam" => &valid_beam_keys,
                        "mcmc" => &valid_mcmc_keys,
                        "gpu" => &valid_gpu_keys,
                        "importance" => &valid_importance_keys,
                        "experimental" => &valid_experimental_keys,
                        _ => continue,
                    };

                    for param_key in section_mapping.keys() {
                        if let Some(param_name) = param_key.as_str() {
                            if !valid_keys.contains(param_name) {
                                unknown_params.push(format!(
                                    "'{}' in section '{}'",
                                    param_name, section_name
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    if !unknown_params.is_empty() {
        let error_msg = format!(
            "Unknown parameter: {}. These parameters will be ignored. Please check for typos or remove them.",
            unknown_params.join("; ")
        );
        return Err(error_msg);
    }

    Ok(())
}

/// Loads and validates parameters from a YAML file.
///
/// # Arguments
///
/// * `param_file` - Path to the YAML parameter file.
///
/// # Returns
///
/// * `Result<Param, Box<dyn Error>>` - Loaded and validated parameters or an error.
///
/// # Errors
///
/// * Returns an error if the file cannot be read or if validation fails.
///
/// # Examples
/// ```
/// # use gpredomics::param::get;
/// # let params = get("samples/tests/param.yaml".to_string())?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn get(param_file: String) -> Result<Param, Box<dyn Error>> {
    // Read the file content as a string
    let yaml_content = std::fs::read_to_string(&param_file)?;

    // First, parse as generic YAML to check for unknown parameters
    let yaml_value: serde_yaml::Value = serde_yaml::from_str(&yaml_content)?;

    // Check for unknown parameters
    if let Err(e) = check_unknown_params(&yaml_value) {
        return Err(e.into());
    }

    // Now deserialize into the Param struct from the same content
    let mut config: Param = serde_yaml::from_str(&yaml_content)?;

    let _ = validate(&mut config)?;

    Ok(config)
}

/// Validates the provided parameters and issues warnings for potential misconfigurations.
///
/// # Arguments
///
/// * `param` - Mutable reference to the Param struct to be validated.
///
/// # Returns
///
/// * `Result<(), String>` - Ok if validation passes, Err with a message if validation fails.
///
/// # Warnings
///
/// * Issues warnings for specific parameter combinations that may lead to suboptimal performance or data leakage.
///
/// # Examples
/// ```
/// # use gpredomics::param::{Param, validate};
/// let mut params = Param::default();
/// validate(&mut params)?;
/// # Ok::<(), String>(())
/// ```
pub fn validate(param: &mut Param) -> Result<(), String> {
    if param.general.log_base.len() > 0 {
        param.general.display_colorful = false;
    }

    if param.cv.fit_on_valid && param.data.Xtest == "".to_string() {
        warn!("fit_on_valid=true without independent test set risks data leakage!");
    }

    if param.general.gpu {
        warn!(
            "GPU acceleration enabled: computations use f32 precision for performance.\n  \
            CPU uses f64 precision. Minor numerical differences (order of 1e-7) are expected\n  \
            between CPU and GPU runs, which may lead to slightly different model rankings\n  \
            and final results. For perfect reproducibility, use the same backend (CPU or GPU)\n  \
            across all runs."
        );
    }

    if param.general.algo == "beam" && !param.general.keep_trace {
        warn!("keep_trace is deactivated: only the last models will be kept in memory during BEAM runs. \
        If you want to get the best models across all k size, please enable keep_trace.");
    }

    if param.general.threshold_ci_alpha > 0.0 && param.general.threshold_ci_alpha < 1.0 {
        validate_bootstrap(param)?
    };

    if (param.data.Xtest.is_empty() && !param.data.ytest.is_empty())
        || (!param.data.Xtest.is_empty() && param.data.ytest.is_empty())
    {
        return Err(format!("Both Xtest and ytest must be provided together.",));
    }

    validate_penalties(param)?;
    Ok(())
}

fn validate_bootstrap(param: &mut Param) -> Result<(), String> {
    if param.general.threshold_ci_n_bootstrap == 0 {
        return Ok(());
    }

    if param.general.threshold_ci_alpha <= 0.0 || param.general.threshold_ci_alpha >= 1.0 {
        return Err(format!(
            "Invalid threshold_ci_alpha={:.3}. Must be in range (0, 1).",
            param.general.threshold_ci_alpha
        ));
    }

    if param.general.threshold_ci_frac_bootstrap <= 0.0
        || param.general.threshold_ci_frac_bootstrap > 1.0
    {
        return Err(format!(
            "Invalid threshold_ci_frac_bootstrap={:.3}. Must be in range (0, 1).",
            param.general.threshold_ci_frac_bootstrap
        ));
    }

    const B_MIN: usize = 1000; // CI (Efron & Tibshirani 1993)
    const B_REC: usize = 2000; // Robustness (Rousselet et al. 2021)

    let B = param.general.threshold_ci_n_bootstrap;

    if B < B_MIN {
        warn!(
            "Bootstrap B={} < {} (Efron & Tibshirani 1993 minimum for percentile CI). \
                Quantile estimates may be unstable. Increase B or disable (set to 0).",
            B, B_MIN
        );
    } else if B < B_REC {
        warn!(
            "Bootstrap B={} < {} (Rousselet et al. 2021 recommendation). \
                Percentile CI may be too narrow for small samples. \
                Consider B ≥ {} for {}% CI stability.",
            B,
            B_REC,
            B_REC,
            (1.0 - param.general.threshold_ci_alpha) * 100.0
        );
    }

    Ok(())
}

fn validate_penalties(param: &mut Param) -> Result<(), String> {
    if param.general.k_penalty < 0.0 {
        return Err(format!(
            "Invalid k_penalty={:.3}. Must be >= 0.",
            param.general.k_penalty
        ));
    }

    if param.general.fr_penalty < 0.0 {
        return Err(format!(
            "Invalid fr_penalty={:.3}. Must be >= 0.",
            param.general.fr_penalty
        ));
    }

    if param.general.bias_penalty < 0.0 {
        return Err(format!(
            "Invalid bias_penalty={:.3}. Must be >= 0.",
            param.general.bias_penalty
        ));
    }

    if param.general.threshold_ci_penalty < 0.0 {
        return Err(format!(
            "Invalid threshold_ci_penalty={:.3}. Must be >= 0.",
            param.general.threshold_ci_penalty
        ));
    }

    if param.general.algo == "ga".to_string()
        && param.ga.random_sampling_pct > 0.0
        && param.cv.overfit_penalty > 0.0
    {
        return Err(format!("Randomized samples and overfit penalty cannot be used together. If you want to resample the folds of the cross-validation used to penalise overfitting, you can use the parameter random_sampling_epochs."));
    }

    Ok(())
}

/// Default value definitions

fn seed_default() -> u64 {
    4815162342
}
fn empty_string() -> String {
    "".to_string()
}
fn min_epochs_default() -> usize {
    10
}
fn max_epochs_default() -> usize {
    200
}
fn max_age_best_model_default() -> usize {
    10
}
fn algorithm_default() -> String {
    "ga".to_string()
}
fn feature_selection_method_default() -> PreselectionMethod {
    PreselectionMethod::wilcoxon
}
fn feature_minimal_prevalence_pct_default() -> f64 {
    10.0
}
fn feature_maximal_adj_pvalue_default() -> f64 {
    0.5
}
fn feature_minimal_log_abs_bayes_factor_default() -> f64 {
    2.0
}
fn language_default() -> String {
    "binary".to_string()
}
fn data_type_default() -> String {
    "raw".to_string()
}
fn data_type_epsilon_default() -> f64 {
    1e-5
}
fn feature_keep_trace_default() -> bool {
    true
}
fn save_experiment_default() -> String {
    "".to_string()
}
fn log_base_default() -> String {
    "".to_string()
}
fn log_suffix_default() -> String {
    "log".to_string()
}
fn log_level_default() -> String {
    "info".to_string()
}
fn folds_default() -> usize {
    5
}
fn n_permutations_mda_default() -> usize {
    100
}
fn importance_aggregation_default() -> ImportanceAggregation {
    ImportanceAggregation::mean
}
fn fit_default() -> FitFunction {
    FitFunction::auc
}
fn n_model_to_display_default() -> u32 {
    10
}
fn false_default() -> bool {
    false
}
fn true_default() -> bool {
    true
}
fn beam_method_default() -> BeamMethod {
    BeamMethod::LimitedExhaustive
}
fn best_models_ci_alpha_default() -> f64 {
    0.05
}
fn max_nb_of_models_default() -> usize {
    10000
}
fn class_names_default() -> Vec<String> {
    Vec::new()
}
fn memory_policy_default() -> GpuMemoryPolicy {
    GpuMemoryPolicy::Adaptive
}
fn max_total_memory_mb_default() -> u64 {
    256
}
fn max_buffer_size_mb_default() -> u32 {
    128
}
fn fallback_to_cpu_default() -> bool {
    true
}
fn n_iter_default() -> usize {
    10_000
}
fn n_burn_default() -> usize {
    5_000
}
fn lambda_default() -> f64 {
    0.001
}
fn nmin_default() -> u32 {
    10
}
fn zero_default() -> f64 {
    0.0
}
fn uzero_default() -> usize {
    0
}
fn half_default() -> f64 {
    0.5
}
fn one_default() -> usize {
    1
}
fn voting_default() -> VotingMethod {
    VotingMethod::Majority
}
fn diversity_voting_default() -> f64 {
    5.0
}
fn k_max_default() -> usize {
    200
}
fn pop_size_default() -> u32 {
    5000
}
fn ga_elite_pct_default() -> f64 {
    2.0
}
fn ga_random_pct_default() -> f64 {
    2.0
}
fn ga_mut_children_pct_default() -> f64 {
    80.0
}
fn ga_mut_features_pct_default() -> f64 {
    20.0
}
fn ga_mut_non_null_pct_default() -> f64 {
    20.0
}
fn holdout_ratio_default() -> f64 {
    0.2
}
fn best_models_criterion_default() -> f64 {
    10.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_unknown_params_valid() {
        let yaml = r#"
            general:
                seed: 42
                algo: ga
            data:
                X: "test.tsv"
                y: "test.tsv"
            "#;
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(check_unknown_params(&value).is_ok());
    }

    #[test]
    fn test_check_unknown_params_invalid_section() {
        let yaml = r#"
            general:
                seed: 42
            unknown_section:
                param: 123
            "#;
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let result = check_unknown_params(&value);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown_section"));
    }

    #[test]
    fn test_check_unknown_params_invalid_parameter() {
        let yaml = r#"
            general:
                seed: 42
                wrong_param: 999
            data:
                X: "test.tsv"
            "#;
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let result = check_unknown_params(&value);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("wrong_param"));
    }

    #[test]
    fn test_check_unknown_params_typo() {
        let yaml = r#"
            ga:
                populaton_size: 5000
            "#;
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        let result = check_unknown_params(&value);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("populaton_size"));
    }

    #[test]
    fn test_check_unknown_params_alias_accepted() {
        let yaml = r#"
            ga:
                kmin: 1
                kmax: 200
            beam:
                kmin: 1
                kmax: 100
            "#;
        let value: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        assert!(check_unknown_params(&value).is_ok());
    }
}
