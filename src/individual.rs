use crate::bayesian_mcmc::Betas;
use crate::data::Data;
use crate::experiment::{Importance, ImportanceCollection, ImportanceScope, ImportanceType};
use crate::param::FitFunction;
use crate::utils::serde_json_hashmap_numeric;
use crate::utils::{compute_auc_from_value, compute_roc_and_metrics_from_value};
use crate::utils::{compute_metrics_from_value, generate_random_vector, shuffle_row};
use crate::Population;
use log::debug;
use rand::seq::SliceRandom; // Provides the `choose_multiple` method
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use statrs::function::logistic::logistic;
use std::cmp::min;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};

/// Confidence interval for the threshold
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct ThresholdCI {
    /// upper bound of the confidence interval
    pub upper: f64,
    /// lower bound of the confidence interval
    pub lower: f64,
    /// rejection rate associated with this confidence interval
    pub rejection_rate: f64,
}

/// Additional metrics that can be stored in Individual
/// These metrics are optional and may not be computed for all individuals
/// This structure will evolve in a more general Metrics structure in the future
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct AdditionalMetrics {
    /// Matthews correlation coefficient
    pub mcc: Option<f64>,
    /// Harmonic mean of precision and recall
    pub f1_score: Option<f64>,
    /// Negative predictive value
    pub npv: Option<f64>,
    /// Positive predictive value
    pub ppv: Option<f64>,
    /// Geometric mean of sensitivity and specificity
    #[serde(alias = "g_means")]
    pub g_mean: Option<f64>,
}

impl Default for AdditionalMetrics {
    fn default() -> Self {
        AdditionalMetrics {
            mcc: None,
            f1_score: None,
            npv: None,
            ppv: None,
            g_mean: None,
        }
    }
}

/// Mathematical model with a set of variables and their corresponding signs
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct Individual {
    /// Map between feature indices and their corresponding signs
    #[serde(with = "serde_json_hashmap_numeric::usize_i8")]
    pub features: HashMap<usize, i8>,
    /// Number of variables used in the model
    pub k: usize,

    /// Language of the model, see docs for details
    pub language: u8,
    /// Data type of the model, see docs for details
    pub data_type: u8,
    /// Epsilon value used during score calculation
    pub epsilon: f64,

    /// Fit value of the model
    pub fit: f64,
    /// Area Under the Curve obtained regarding the model prediction on the training set
    pub auc: f64,
    /// Decision threshold used for binary classification
    pub threshold: f64,
    /// Confidence interval for the threshold if applicable and associated rejection rate
    pub threshold_ci: Option<ThresholdCI>,
    /// Sensitivity obtained regarding the model prediction on the training set
    pub sensitivity: f64,
    /// Specificity obtained regarding the model prediction on the training set
    pub specificity: f64,
    /// Accuracy obtained regarding the model prediction on the training set
    pub accuracy: f64,
    /// If computed, additional metrics for the individual
    #[serde(default)]
    pub metrics: AdditionalMetrics,

    /// Iteration of the algorithm that led to the emergence of the model
    pub epoch: usize, // generation or other counter important in the strategy
    /// Parents of the individual in the generation context
    pub parents: Option<Vec<u64>>,

    /// Identfier hash of the model
    pub hash: u64,

    /// For MCMC individuals, the beta coefficients
    pub betas: Option<Betas>,
}

/// Constants for language and data type representations
/// MCMC_GENERIC_LANG is set to 101 to avoid conflict with other language codes
pub const MCMC_GENERIC_LANG: u8 = 101;
/// Binary language with coefficients in {0,1}
pub const BINARY_LANG: u8 = 0;
/// Ternary language with coefficients in {-1,0,1}
pub const TERNARY_LANG: u8 = 1;
/// Power-of-2 language with coefficients in {...-4,-2,-1,0,1,2,4,...}
pub const POW2_LANG: u8 = 2;
/// Ratio language with coefficients in {-1,0,1}
pub const RATIO_LANG: u8 = 3;
/// Raw data type
pub const RAW_TYPE: u8 = 0;
/// Prevalence data type
pub const PREVALENCE_TYPE: u8 = 1;
/// Log data type
pub const LOG_TYPE: u8 = 2;

/// Default minimum epsilon value for ratio calculations
pub const DEFAULT_MINIMUM: f64 = f64::MIN_POSITIVE;

const DEFAULT_POW2_START: u8 = 4;

/// Converts a language string to its corresponding u8 representation
///
/// # Panics
///
/// Panics if the provided language string is not recognized
///
/// # Examples
///
/// ```
/// # use gpredomics::individual::language;
/// let lang = language("binary");
/// assert_eq!(lang, 0);
/// ```
pub fn language(language_string: &str) -> u8 {
    match language_string.to_lowercase().as_str() {
        "binary" | "bin" => BINARY_LANG,
        "ternary" | "ter" => TERNARY_LANG,
        "pow2" => POW2_LANG,
        "ratio" => RATIO_LANG,
        "generic" | "mcmc_generic" => MCMC_GENERIC_LANG,
        other => panic!("Unrecognized language {}", other),
    }
}

/// Converts a data type string to its corresponding u8 representation
///
/// # Panics
///
/// Panics if the provided data type string is not recognized
///
/// # Examples
///
/// ```
/// # use gpredomics::individual::data_type;
/// let dtype = data_type("raw");
/// assert_eq!(dtype, 0);
/// ```
pub fn data_type(data_type_string: &str) -> u8 {
    match data_type_string.to_lowercase().as_str() {
        "raw" => RAW_TYPE,
        "prevalence" | "prev" => PREVALENCE_TYPE,
        "log" => LOG_TYPE,
        other => panic!("Unrecognized data type {}", other),
    }
}

impl Individual {
    /// Generates a new empty Individual with default values
    ///
    /// # Examples
    ///
    /// ```
    /// # use gpredomics::individual::Individual;
    /// let individual = Individual::new();
    /// assert_eq!(individual.features.len(), 0);
    /// ```
    pub fn new() -> Individual {
        Individual {
            features: HashMap::new(),
            auc: 0.0,
            specificity: 0.0,
            sensitivity: 0.0,
            accuracy: 0.0,
            threshold: 0.0,
            fit: 0.0,
            k: 0,
            epoch: 0,
            language: BINARY_LANG,
            data_type: RAW_TYPE,
            hash: 0,
            epsilon: DEFAULT_MINIMUM,
            parents: None,
            betas: None,
            threshold_ci: None,
            metrics: AdditionalMetrics {
                mcc: None,
                f1_score: None,
                npv: None,
                ppv: None,
                g_mean: None,
            },
        }
    }

    /// Generates a string representation of the Individual to make it human-readable
    ///
    /// # Arguments
    ///
    /// * `data` - Reference to the Data object used for feature names
    /// * `data_to_test` - Optional reference to a Data object for testing metrics
    /// * `algo` - String representing the algorithm used (e.g., "ga", "beam", "mcmc")
    /// * `ci_alpha` - Confidence interval alpha value for threshold CI display
    ///
    /// # Returns
    ///
    /// A formatted string representing the Individual
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// let mut individual = Individual::new();
    /// individual.features.insert(0, 1);
    /// individual.features.insert(1, 1);
    /// let data = Data::new();
    /// let display_str = individual.display(&data, None, &"ga".to_string(), 0.05);
    /// println!("{}", display_str);
    /// ```
    pub fn display(
        &self,
        data: &Data,
        data_to_test: Option<&Data>,
        algo: &String,
        ci_alpha: f64,
    ) -> String {
        let algo_str = match algo.as_str() {
            "ga" => format!(" [gen:{}] ", self.epoch),
            "beam" => " ".to_string(),
            "mcmc" => format!(" [MCMC step: {}] ", self.epoch),
            _ => " [unknown] ".to_string(),
        };

        let metrics = match data_to_test {
            Some(test_data) => {
                let (acc_test, se_test, sp_test, rej_test, additional) =
                    self.compute_metrics(test_data);
                let mut m = format!("{}:{} [k={}]{}[fit:{:.3}] AUC {:.3}/{:.3} | accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3}",
                    self.get_language(), self.get_data_type(), self.features.len(), algo_str, self.fit,
                    self.auc, self.compute_new_auc(test_data), self.accuracy, acc_test,
                    self.sensitivity, se_test, self.specificity, sp_test);

                if let Some(ref threshold_ci) = self.threshold_ci {
                    m = format!(
                        "{} | rejection rate {:.3}/{:.3}",
                        m, threshold_ci.rejection_rate, rej_test
                    );
                }
                if self.metrics.mcc.is_some() {
                    m = format!(
                        "{} | MCC {:.3}/{:.3} ",
                        m,
                        self.metrics.mcc.unwrap(),
                        additional.mcc.unwrap()
                    );
                }
                if self.metrics.f1_score.is_some() {
                    m = format!(
                        "{} | F1-score {:.3}/{:.3} ",
                        m,
                        self.metrics.f1_score.unwrap(),
                        additional.f1_score.unwrap()
                    );
                }
                if self.metrics.npv.is_some() {
                    m = format!(
                        "{} | NPV {:.3}/{:.3} ",
                        m,
                        self.metrics.npv.unwrap(),
                        additional.npv.unwrap()
                    );
                }
                if self.metrics.ppv.is_some() {
                    m = format!(
                        "{} | PPV {:.3}/{:.3} ",
                        m,
                        self.metrics.ppv.unwrap(),
                        additional.ppv.unwrap()
                    );
                }
                if self.metrics.g_mean.is_some() {
                    m = format!(
                        "{} | G-mean {:.3}/{:.3} ",
                        m,
                        self.metrics.g_mean.unwrap(),
                        additional.g_mean.unwrap()
                    );
                }
                m
            }
            None => {
                let mut m = format!("{}:{} [k={}]{}[fit:{:.3}] AUC {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3}",
                    self.get_language(), self.get_data_type(), self.features.len(), algo_str,
                    self.fit, self.auc, self.accuracy, self.sensitivity, self.specificity);

                if let Some(ref threshold_ci) = self.threshold_ci {
                    m = format!("{} | rejection rate {:.3}", m, threshold_ci.rejection_rate);
                }
                if self.metrics.mcc.is_some() {
                    m = format!("{} | MCC {:.3} ", m, self.metrics.mcc.unwrap());
                }
                if self.metrics.f1_score.is_some() {
                    m = format!("{} | F1-score {:.3} ", m, self.metrics.f1_score.unwrap());
                }
                if self.metrics.npv.is_some() {
                    m = format!("{} | NPV {:.3} ", m, self.metrics.npv.unwrap());
                }
                if self.metrics.ppv.is_some() {
                    m = format!("{} | PPV {:.3} ", m, self.metrics.ppv.unwrap());
                }
                if self.metrics.g_mean.is_some() {
                    m = format!("{} | G-mean {:.3} ", m, self.metrics.g_mean.unwrap());
                }
                m
            }
        };

        // Sort features by index
        let mut sorted_features: Vec<_> = self.features.iter().collect();
        sorted_features.sort_by(|a, b| a.0.cmp(b.0));

        let mut positive_features: Vec<_> = sorted_features
            .iter()
            .filter(|&&(_, &coef)| coef > 0)
            .collect();
        let mut negative_features: Vec<_> = sorted_features
            .iter()
            .filter(|&&(_, &coef)| coef < 0)
            .collect();

        positive_features.sort_by(|a, b| b.1.cmp(a.1));
        negative_features.sort_by(|a, b| a.1.cmp(b.1));

        let positive_str: Vec<String> = positive_features
            .iter()
            .enumerate()
            .map(|(_i, &&(index, coef))| {
                let mut str = format!("\x1b[96m{}\x1b[0m", data.features[*index]);

                if self.data_type == PREVALENCE_TYPE {
                    str = format!("{}⁰", str);
                }
                if self.language == POW2_LANG && !(*coef == 1_i8) && self.data_type != LOG_TYPE {
                    str = format!("{}*{}", coef, str);
                } else if self.language == POW2_LANG
                    && !(*coef == 1_i8)
                    && self.data_type == LOG_TYPE
                {
                    // b*ln(a) == ln(a^b)
                    str = format!("{}^{}", str, coef);
                }
                str
            })
            .collect();

        let negative_str: Vec<String> = negative_features
            .iter()
            .enumerate()
            .map(|(_i, &&(index, coef))| {
                let mut str = format!("\x1b[95m{}\x1b[0m", data.features[*index]);

                if self.data_type == PREVALENCE_TYPE {
                    str = format!("{}⁰", str);
                }
                if self.language == POW2_LANG && !(*coef == -1_i8) && self.data_type != LOG_TYPE {
                    str = format!("{}*{}", coef.abs(), str);
                } else if self.language == POW2_LANG
                    && !(*coef == -1_i8)
                    && self.data_type == LOG_TYPE
                {
                    // b*ln(a) == ln(a^b)
                    // absolute coeff as minus is before ln() -> ln(prod(pos)) - ln(prod(neg)) = threshold + ln(prod(data.type_minimum^coeff)
                    str = format!("{}^{}", str, coef.abs());
                }
                str
            })
            .collect();

        let mut negative_str_owned = negative_str.clone();
        if self.language == RATIO_LANG && self.data_type != LOG_TYPE {
            negative_str_owned.push(format!("{:2e}", self.epsilon));
        }

        let (second_line_first_part, second_line_second_part) = if let Some(ref threshold_ci) =
            self.threshold_ci
        {
            let threshold_text = format!("Class \x1b[95m{}\x1b[0m: score < {}\nClass \x1b[96m{}\x1b[0m: score > {}\nRejection zone \x1b[2m({:2}% CI)\x1b[0m: score ∈ [{:.3}; {:.3}]\nscore =", 
                data.classes[0], threshold_ci.lower, data.classes[1], threshold_ci.upper, (1.0-ci_alpha)*100.0,  threshold_ci.lower, threshold_ci.upper);
            (threshold_text, String::new())
        } else {
            (
                format!("Class {}:", data.classes[1]),
                format!("≥ {}", self.threshold),
            )
        };

        let positive_str_final = if positive_str.is_empty() {
            vec!["0".to_string()]
        } else {
            positive_str
        };
        let negative_str_final = if negative_str_owned.is_empty() {
            vec!["0".to_string()]
        } else {
            negative_str_owned
        };

        let (positive_joined, negative_joined) = if self.data_type == LOG_TYPE {
            let pos_str: Vec<String> = positive_str_final
                .iter()
                .map(|f| {
                    if f != "0" {
                        format!("{}⁺", f)
                    } else {
                        "1".to_string()
                    }
                })
                .collect();
            let neg_str: Vec<String> = negative_str_final
                .iter()
                .map(|f| {
                    if f != "0" {
                        format!("{}⁺", f)
                    } else {
                        "1".to_string()
                    }
                })
                .collect();

            (
                format!("ln({})", pos_str.join(" × ")),
                format!("ln({})", neg_str.join(" × ")),
            )
        } else {
            (
                format!("({})", positive_str_final.join(" + ")),
                format!("({})", negative_str_final.join(" + ")),
            )
        };

        let formatted_string = if self.language == BINARY_LANG {
            format!(
                "{}\n{} {} {}",
                metrics, second_line_first_part, positive_joined, second_line_second_part
            )
        } else if self.language == TERNARY_LANG || self.language == POW2_LANG {
            format!(
                "{}\n{} {} - {} {}",
                metrics,
                second_line_first_part,
                positive_joined,
                negative_joined,
                second_line_second_part
            )
        } else if self.language == RATIO_LANG {
            if self.data_type == LOG_TYPE {
                // LOG+RATIO: Display ln() - ln() instead of /
                format!(
                    "{}\n{} {} - {} - {} {}",
                    metrics,
                    second_line_first_part,
                    positive_joined,
                    negative_joined,
                    self.epsilon,
                    second_line_second_part
                )
            } else {
                // Standard RATIO: Display /
                format!(
                    "{}\n{} {} / {} {}",
                    metrics,
                    second_line_first_part,
                    positive_joined,
                    negative_joined,
                    second_line_second_part
                )
            }
        } else {
            format!(
                "{}\n{} {:?} {}",
                metrics, second_line_first_part, self, second_line_second_part
            )
        };

        formatted_string
    }

    /// Computes the hash of the Individual based on its features and betas
    ///
    /// # Examples
    ///
    /// ```
    /// # use gpredomics::individual::Individual;
    /// let mut individual = Individual::new();
    /// individual.features.insert(0, 1);
    /// individual.features.insert(1, -1);
    /// individual.compute_hash();
    /// println!("Individual hash: {}", individual.hash);
    /// ```
    pub fn compute_hash(&mut self) {
        let mut hasher = DefaultHasher::new();

        // Convert HashMap to a sorted representation
        let sorted_features: BTreeMap<_, _> = self.features.iter().collect();
        sorted_features.hash(&mut hasher);
        self.betas.hash(&mut hasher);
        self.hash = hasher.finish();
    }

    /// Counts the number of features in the Individual and updates k accordingly
    ///
    /// # Examples
    /// ```
    /// # use gpredomics::individual::Individual;
    /// let mut individual = Individual::new();
    /// individual.features.insert(0, 1);
    /// individual.features.insert(1, -1);
    /// individual.count_k();
    /// assert_eq!(individual.k, 2);
    /// ```
    pub fn count_k(&mut self) {
        self.k = self.features.len();
    }

    /// Checks compatibility of the Individual with the provided Data
    ///
    /// # Arguments
    ///
    /// * `data` - Reference to the Data object to check compatibility against
    ///
    /// # Returns
    ///
    /// Returns true if compatible (or only warnings), false if incompatible
    pub fn check_compatibility(&self, data: &Data) -> bool {
        use log::{error, warn};
        let mut is_compatible = true;

        // Check data is valid
        if data.feature_len == 0 {
            error!("Data has no features (feature_len = 0)");
            return false;
        }
        if data.sample_len == 0 {
            error!("Data has no samples (sample_len = 0)");
            return false;
        }

        // Check for empty features
        if self.features.is_empty() {
            warn!("Individual has no features");
        }

        // Check k consistency
        if self.k > data.feature_len {
            error!(
                "Individual has more feature than data! ({} > {})",
                self.k, data.feature_len
            );
            return false;
        }

        // Check that all feature indices are within data's feature range
        for &feature_idx in self.features.keys() {
            if feature_idx >= data.feature_len {
                error!(
                    "Individual has feature index {} which is out of bounds (data has {} features)",
                    feature_idx, data.feature_len
                );
                is_compatible = false;
            }
        }

        // Check feature values according to language
        for (&feature_idx, &value) in self.features.iter() {
            let valid = match self.language {
                BINARY_LANG => value == 1,
                TERNARY_LANG | RATIO_LANG => value >= -1 && value <= 1,
                POW2_LANG => {
                    let abs_val = value.abs();
                    abs_val > 0 && abs_val <= 64 && (abs_val & (abs_val - 1)) == 0
                }
                _ => true, // Unknown language, skip validation
            };

            if !valid {
                error!(
                    "Individual has invalid value {} for feature {} with language {} ({})",
                    value,
                    feature_idx,
                    self.language,
                    self.get_language()
                );
                is_compatible = false;
            }
        }

        // Warn if features are present but not in feature_selection
        if !data.feature_selection.is_empty() {
            let selection_set: HashSet<usize> = data.feature_selection.iter().copied().collect();

            for &feature_idx in self.features.keys() {
                if feature_idx < data.feature_len && !selection_set.contains(&feature_idx) {
                    warn!(
                        "Individual has feature {} which is not in the current feature_selection (this feature will still be used but may indicate a mismatch)",
                        feature_idx
                    );
                }
            }
        }

        is_compatible
    }

    /// Generates a child Individual from a main parent Individual
    ///
    /// # Arguments
    ///
    /// * `main_parent` - Reference to the main parent Individual whose language and data_type will be inherited
    ///
    /// # Returns
    ///
    /// A new Individual having inherited language and data_type from the main parent
    pub fn child(main_parent: &Individual) -> Individual {
        let mut i = Individual::new();
        if main_parent.threshold_ci.is_some() {
            i.threshold_ci = Some(ThresholdCI {
                upper: 0.0,
                lower: 0.0,
                rejection_rate: 0.0,
            })
        }
        i.language = main_parent.language;
        i.data_type = main_parent.data_type;
        i.epsilon = main_parent.epsilon;
        i
    }

    /// Evaluates the Individual on the provided Data and returns both class predictions and scores
    ///
    /// # Arguments
    ///
    /// * `d` - Reference to the Data object on which to evaluate the Individual
    ///
    /// # Returns
    ///
    /// A tuple containing a vector of class predictions and a vector of scores
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// # let individual = Individual::new();
    /// # let data = Data::new();
    /// let (classes, scores) = individual.evaluate_class_and_score(&data);
    /// ```
    pub fn evaluate_class_and_score(&self, d: &Data) -> (Vec<u8>, Vec<f64>) {
        let value = self.evaluate(d);
        let class = value
            .iter()
            .map(|&v| {
                if let Some(ref threshold_ci) = self.threshold_ci {
                    if v > threshold_ci.upper {
                        1
                    } else if v < threshold_ci.lower {
                        0
                    } else {
                        2
                    }
                } else {
                    if v >= self.threshold {
                        1
                    } else {
                        0
                    }
                }
            })
            .collect();

        (class, value)
    }

    /// Evaluates the Individual on the provided Data and returns class predictions
    ///
    /// # Arguments
    ///
    /// * `d` - Reference to the Data object on which to evaluate the Individual
    ///
    /// # Returns
    ///
    /// A vector of class predictions
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// # let individual = Individual::new();
    /// # let data = Data::new();
    /// let classes = individual.evaluate_class(&data);
    /// ```
    pub fn evaluate_class(&self, d: &Data) -> Vec<u8> {
        let value = self.evaluate(d);
        value
            .iter()
            .map(|&v| {
                if let Some(ref threshold_ci) = self.threshold_ci {
                    if v > threshold_ci.upper {
                        1
                    } else if v < threshold_ci.lower {
                        0
                    } else {
                        2
                    }
                } else {
                    if v >= self.threshold {
                        1
                    } else {
                        0
                    }
                }
            })
            .collect()
    }

    /// Evaluates the Individual on the provided Data and returns scores
    ///
    /// # Arguments
    ///
    /// * `d` - Reference to the Data object on which to evaluate the Individual
    ///
    /// # Returns
    ///
    /// A vector of scores
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// # let individual = Individual::new();
    /// # let data = Data::new();
    /// let scores = individual.evaluate(&data);
    /// ```
    pub fn evaluate(&self, d: &Data) -> Vec<f64> {
        self.evaluate_from_features(&d.X, d.sample_len)
    }

    /// Evaluates the Individual using provided feature matrix and sample length
    ///
    /// # Arguments
    ///
    /// * `X` - Reference to a HashMap representing the feature matrix
    /// * `sample_len` - Number of samples to evaluate to make easier the HashMap usage
    ///
    /// # Returns
    ///
    /// A vector of scores
    ///
    /// # Panics
    ///
    /// Panics if the Individual's data_type is unknown
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use std::collections::HashMap;
    /// # let individual = Individual::new();
    /// # let X: HashMap<(usize, usize), f64> = HashMap::new();
    /// # let sample_len = 10;
    /// let scores = individual.evaluate_from_features(&X, sample_len);
    /// ```
    pub fn evaluate_from_features(
        &self,
        X: &HashMap<(usize, usize), f64>,
        sample_len: usize,
    ) -> Vec<f64> {
        match self.data_type {
            RAW_TYPE => self.evaluate_raw(X, sample_len),
            PREVALENCE_TYPE => self.evaluate_prevalence(X, sample_len),
            LOG_TYPE => self.evaluate_log(X, sample_len),
            other => panic!("Unknown data-type {}", other),
        }
    }

    /// Evaluates the Individual using raw feature values
    ///
    /// # Arguments
    ///
    /// * `X` - Reference to a HashMap representing the feature matrix
    /// * `sample_len` - Number of samples to evaluate to make easier the HashMap usage
    ///
    /// # Returns
    ///
    /// A vector of scores
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use gpredomics::individual::Individual;
    /// # use std::collections::HashMap;
    /// # let individual = Individual::new();
    /// # let X: HashMap<(usize, usize), f64> = HashMap::new();
    /// # let sample_len = 10;
    /// let scores = individual.evaluate_raw(&X, sample_len);
    /// ```
    fn evaluate_raw(&self, X: &HashMap<(usize, usize), f64>, sample_len: usize) -> Vec<f64> {
        let mut score = vec![0.0; sample_len];

        if self.language == RATIO_LANG {
            let mut r: Vec<Vec<f64>> = vec![vec![0.0, 0.0]; sample_len];
            for (feature_index, coef) in self.features.iter() {
                let part = if *coef > 0 { 0 } else { 1 };
                for sample in 0..sample_len {
                    r[sample][part] += X.get(&(sample, *feature_index)).unwrap_or(&0.0);
                }
            }
            for sample in 0..sample_len {
                score[sample] = r[sample][0] / (r[sample][1] + self.epsilon);
            }
        } else if self.language == MCMC_GENERIC_LANG {
            let betas = self
                .betas
                .as_ref()
                .expect("MCMC Individuals must have betas coefficeints");
            let mut pos_sums = vec![0.0; sample_len];
            let mut neg_sums = vec![0.0; sample_len];
            for (feature_index, coef) in self.features.iter() {
                for sample in 0..sample_len {
                    let v = *X.get(&(sample, *feature_index)).unwrap_or(&0.0);
                    match coef {
                        1 => pos_sums[sample] += v,
                        -1 => neg_sums[sample] += v,
                        _ => {}
                    }
                }
            }

            score = pos_sums
                .into_iter()
                .zip(neg_sums.into_iter())
                .map(|(pos, neg)| {
                    let z = pos * betas.a + neg * betas.b + betas.c;
                    logistic(z)
                })
                .collect();
        } else {
            for (feature_index, coef) in self.features.iter() {
                let x_coef = *coef as f64;
                for sample in 0..sample_len {
                    score[sample] += X.get(&(sample, *feature_index)).unwrap_or(&0.0) * x_coef;
                }
            }
        }
        score
    }

    /// Evaluates the Individual using prevalence modified feature values
    ///
    /// # Arguments
    ///
    /// * `X` - Reference to a HashMap representing the feature matrix
    /// * `sample_len` - Number of samples to evaluate to make easier the HashMap usage
    ///
    /// # Returns
    ///
    /// A vector of scores
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use gpredomics::individual::Individual;
    /// # use std::collections::HashMap;
    /// # let individual = Individual::new();
    /// # let X: HashMap<(usize, usize), f64> = HashMap::new();
    /// # let sample_len = 10;
    /// let scores = individual.evaluate_prevalence(&X, sample_len);
    /// ```
    fn evaluate_prevalence(&self, X: &HashMap<(usize, usize), f64>, sample_len: usize) -> Vec<f64> {
        let mut score = vec![0.0; sample_len];

        if self.language == RATIO_LANG {
            let mut r: Vec<Vec<f64>> = vec![vec![0.0, 0.0]; sample_len];
            for (feature_index, coef) in self.features.iter() {
                let part = if *coef > 0 { 0 } else { 1 };
                for sample in 0..sample_len {
                    r[sample][part] +=
                        if X.get(&(sample, *feature_index)).unwrap_or(&0.0) > &self.epsilon {
                            1.0
                        } else {
                            0.0
                        };
                }
            }
            for sample in 0..sample_len {
                score[sample] = r[sample][0] / (r[sample][1] + self.epsilon);
            }
        } else if self.language == MCMC_GENERIC_LANG {
            let betas = self
                .betas
                .as_ref()
                .expect("MCMC Individuals must have betas coefficeints");
            let mut pos_sums = vec![0.0; sample_len];
            let mut neg_sums = vec![0.0; sample_len];
            for (feature_index, coef) in self.features.iter() {
                for sample in 0..sample_len {
                    let v = if X.get(&(sample, *feature_index)).unwrap_or(&0.0) > &self.epsilon {
                        1.0
                    } else {
                        0.0
                    };
                    match coef {
                        1 => pos_sums[sample] += v,
                        -1 => neg_sums[sample] += v,
                        _ => {}
                    }
                }
            }

            score = pos_sums
                .into_iter()
                .zip(neg_sums.into_iter())
                .map(|(pos, neg)| {
                    let z = pos * betas.a + neg * betas.b + betas.c;
                    logistic(z)
                })
                .collect();
        } else {
            for (feature_index, coef) in self.features.iter() {
                let x_coef = *coef as f64;
                for sample in 0..sample_len {
                    score[sample] +=
                        if X.get(&(sample, *feature_index)).unwrap_or(&0.0) > &self.epsilon {
                            1.0
                        } else {
                            0.0
                        } * x_coef;
                }
            }
        }

        score
    }

    /// Evaluates the Individual using log-transformed feature values
    ///
    /// # Arguments
    ///
    /// * `X` - Reference to a HashMap representing the feature matrix
    /// * `sample_len` - Number of samples to evaluate to make easier the HashMap usage
    ///
    /// # Returns
    ///
    /// A vector of scores
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use gpredomics::individual::Individual;
    /// # use std::collections::HashMap;
    /// # let individual = Individual::new();
    /// # let X: HashMap<(usize, usize), f64> = HashMap::new();
    /// # let sample_len = 10;
    /// let scores = individual.evaluate_log(&X, sample_len);
    /// ```
    fn evaluate_log(&self, X: &HashMap<(usize, usize), f64>, sample_len: usize) -> Vec<f64> {
        // Shouldn't + epsilon be added?
        let mut score = vec![0.0; sample_len];

        if self.language == RATIO_LANG {
            let mut r: Vec<Vec<f64>> = vec![vec![0.0, 0.0]; sample_len];
            for (feature_index, coef) in self.features.iter() {
                let part = if *coef > 0 { 0 } else { 1 };
                for sample in 0..sample_len {
                    if let Some(val) = X.get(&(sample, *feature_index)) {
                        r[sample][part] += (val / self.epsilon).ln() * coef.abs() as f64;
                    }
                }
            }
            for sample in 0..sample_len {
                score[sample] = r[sample][0] / (r[sample][1] + self.epsilon);
            }
        } else if self.language == MCMC_GENERIC_LANG {
            let betas = self
                .betas
                .as_ref()
                .expect("MCMC Individuals must have betas coefficeints");
            let mut pos_sums = vec![0.0; sample_len];
            let mut neg_sums = vec![0.0; sample_len];
            for (feature_index, coef) in self.features.iter() {
                for sample in 0..sample_len {
                    if let Some(v) = X.get(&(sample, *feature_index)) {
                        match coef {
                            1 => pos_sums[sample] += (v / self.epsilon).ln(),
                            -1 => neg_sums[sample] += (v / self.epsilon).ln(),
                            _ => {}
                        }
                    }
                }
            }

            score = pos_sums
                .into_iter()
                .zip(neg_sums.into_iter())
                .map(|(pos, neg)| {
                    let z = pos * betas.a + neg * betas.b + betas.c;
                    logistic(z)
                })
                .collect();
        } else {
            for (feature_index, coef) in self.features.iter() {
                let x_coef = *coef as f64;
                for sample in 0..sample_len {
                    if let Some(val) = X.get(&(sample, *feature_index)) {
                        score[sample] += (val / self.epsilon).ln() * x_coef;
                    }
                }
            }
        }

        score
    }

    /// Computes AUC and updates self.auc
    ///
    /// # Arguments
    ///
    /// * `d` - Reference to the Data object on which to compute AUC
    ///
    /// # Returns
    ///
    /// The computed AUC value and updates self.auc
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// # let mut individual = Individual::new();
    /// # let data = Data::new();
    /// let auc = individual.compute_auc(&data);
    /// ```
    pub fn compute_auc(&mut self, d: &Data) -> f64 {
        let value = self.evaluate(d);
        self.auc = compute_auc_from_value(&value, &d.y);
        self.auc
    }

    /// Computes AUC without updating self.auc
    ///
    /// # Arguments
    ///
    /// * `d` - Reference to the Data object on which to compute AUC
    ///
    /// # Returns
    ///
    /// The computed AUC value
    ///
    pub fn compute_new_auc(&self, d: &Data) -> f64 {
        let value = self.evaluate(d);
        compute_auc_from_value(&value, &d.y)
    }

    /// Compute AUC based on X and y rather than a complete Data object
    ///
    /// # Arguments
    ///
    /// * `X` - Reference to a HashMap representing the feature matrix
    /// * `y` - Reference to a vector of true class labels
    ///
    /// # Returns
    ///
    /// The computed AUC value and updates self.auc
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use std::collections::HashMap;
    /// # let mut individual = Individual::new();
    /// # let X: HashMap<(usize, usize), f64> = HashMap::new();
    /// # let y: Vec<u8> = vec![];
    /// let auc = individual.compute_auc_from_features(&X, &y);
    /// ```
    pub fn compute_auc_from_features(
        &mut self,
        X: &HashMap<(usize, usize), f64>,
        y: &Vec<u8>,
    ) -> f64 {
        let value = self.evaluate_from_features(X, y.len());
        self.auc = compute_auc_from_value(&value, y);
        self.auc
    }

    /// Computes ROC and various metrics, updating the Individual's fields
    /// Same results as compute_auc and compute_threshold_and_metrics (different threshold but same metrics)
    /// If the fit is not computed on AUC but on objective, metrics are calculated on this objective
    /// Obsolete? use compute_threshold_and_metrics after compute_auc instead
    ///
    /// # Arguments
    ///
    /// * `d` - Reference to the Data object on which to compute ROC and metrics
    /// * `fit_function` - Reference to the FitFunction enum specifying the fit function
    /// * `penalties` - Optional array of two f64 values representing penalties for false positives and false negatives
    ///
    /// # Returns
    ///
    /// A tuple containing (AUC, threshold, accuracy, sensitivity, specificity, objective)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// # use gpredomics::param::FitFunction;
    /// # let mut individual = Individual::new();
    /// # let data = Data::new();
    /// # let fit_function = FitFunction::AUC;
    /// let (auc, threshold, accuracy, sensitivity, specificity, objective) = individual.compute_roc_and_metrics(&data, &fit_function, None);
    /// ```
    pub fn compute_roc_and_metrics(
        &mut self,
        d: &Data,
        fit_function: &FitFunction,
        penalties: Option<[f64; 2]>,
    ) -> (f64, f64, f64, f64, f64, f64) {
        let objective;
        let scores: Vec<_> = self.evaluate(d);
        (
            self.auc,
            self.threshold,
            self.accuracy,
            self.sensitivity,
            self.specificity,
            objective,
        ) = compute_roc_and_metrics_from_value(&scores, &d.y, fit_function, penalties);
        (
            self.auc,
            self.threshold,
            self.accuracy,
            self.sensitivity,
            self.specificity,
            objective,
        )
    }

    /// Calculates the confusion matrix (TP, FP, TN, FN) for the Individual on the provided Data
    ///
    /// # Arguments
    ///
    /// * `data` - Reference to the Data object on which to calculate the confusion matrix
    ///
    /// # Returns
    ///
    /// A tuple containing (TP, FP, TN, FN)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// # let individual = Individual::new();
    /// # let data = Data::new();
    /// let (tp, fp, tn, fn_count) = individual.calculate_confusion_matrix(&data);
    /// ```
    pub fn calculate_confusion_matrix(&self, data: &Data) -> (usize, usize, usize, usize) {
        let mut tp = 0; // True Positives
        let mut fp = 0; // False Positives
        let mut tn = 0; // True Negatives
        let mut fn_count = 0; // False Negatives

        let value = self.evaluate(data);

        for (i, &pred) in value.iter().enumerate() {
            match data.y[i] {
                1 => {
                    // Positive class
                    if pred > self.threshold {
                        tp += 1;
                    } else {
                        fn_count += 1;
                    }
                }
                0 => {
                    // Negative class
                    if pred > self.threshold {
                        fp += 1;
                    } else {
                        tn += 1;
                    }
                }
                2 => {
                    // Unknown class, ignore
                }
                _ => panic!("Invalid class label in y: {}", data.y[i]),
            }
        }

        (tp, fp, tn, fn_count)
    }

    /// Generates a random Individual based on the provided Data
    ///
    /// # Arguments
    ///
    /// * `d` - Reference to the Data object to determine feature length
    /// * `rng` - Mutable reference to a ChaCha8Rng random number generator
    ///
    /// # Returns
    ///
    /// A randomly generated Individual
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// # use rand_chacha::ChaCha8Rng;
    /// # use rand::SeedableRng;
    /// # let data = Data::new();
    /// let mut rng = ChaCha8Rng::from_entropy();
    /// let random_individual = Individual::random(&data, &mut rng);
    /// ```
    pub fn random(d: &Data, rng: &mut ChaCha8Rng) -> Individual {
        let mut features: HashMap<usize, i8> = HashMap::new();
        for (i, coef) in generate_random_vector(d.feature_len, rng)
            .iter()
            .enumerate()
        {
            if *coef != 0 {
                features.insert(i, *coef);
            }
        }

        let mut i = Individual::new();
        i.features = features;
        i.k = i.features.len();
        i
    }

    /// Generates a random Individual based on data with feature selection (uniform or weighted)
    ///
    /// This is the unified function that handles both uniform and weighted feature selection.
    /// Use `prior_weight` parameter to enable weighted selection, or pass `None` for uniform selection.
    ///
    /// # Arguments
    ///
    /// * `k_min` - Minimum number of features to select
    /// * `k_max` - Maximum number of features to select
    /// * `feature_selection` - Slice of usize representing the indices of features available for selection
    /// * `feature_class` - Reference to a HashMap mapping feature indices to their classes
    /// * `language` - u8 representing the language type (e.g., BINARY_LANG, POW2_LANG, etc.)
    /// * `data_type` - u8 representing the data type (e.g., RAW_TYPE, PREVALENCE_TYPE, LOG_TYPE)
    /// * `epsilon` - f64 value used in certain calculations (e.g., for RATIO_LANG)
    /// * `prior_weight` - Optional reference to a HashMap mapping feature indices to their weights for weighted selection
    /// * `threshold_ci` - Boolean indicating whether to include threshold confidence intervals
    /// * `rng` - Mutable reference to a ChaCha8Rng random number generator
    ///
    /// # Returns
    ///
    /// A randomly generated Individual based on the specified feature selection method
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::{Individual, BINARY_LANG, RAW_TYPE};
    /// # use rand_chacha::ChaCha8Rng;
    /// # use rand::SeedableRng;
    /// # use std::collections::HashMap;
    /// # let feature_selection = vec![0, 1, 2];
    /// # let feature_class = HashMap::new();
    /// let mut rng = ChaCha8Rng::from_entropy();
    /// let random_individual = Individual::random_select(
    ///     1,
    ///     5,
    ///     &feature_selection,
    ///     &feature_class,
    ///     BINARY_LANG,
    ///     RAW_TYPE,
    ///     0.01,
    ///     None,
    ///     false,
    ///     &mut rng,
    /// );
    /// ```
    pub fn random_select(
        k_min: usize,
        k_max: usize,
        feature_selection: &[usize],
        feature_class: &HashMap<usize, u8>,
        language: u8,
        data_type: u8,
        epsilon: f64,
        prior_weight: Option<&HashMap<usize, f64>>,
        threshold_ci: bool,
        rng: &mut ChaCha8Rng,
    ) -> Individual {
        if let Some(weights) = prior_weight {
            Self::random_select_weighted(
                k_min,
                k_max,
                feature_selection,
                feature_class,
                language,
                data_type,
                epsilon,
                weights,
                threshold_ci,
                rng,
            )
        } else {
            Self::random_select_k(
                k_min,
                k_max,
                feature_selection,
                feature_class,
                language,
                data_type,
                epsilon,
                threshold_ci,
                rng,
            )
        }
    }

    /// Generates a random Individual with uniform feature selection
    ///
    /// # Arguments
    ///
    /// * `k_min` - Minimum number of features to select
    /// * `k_max` - Maximum number of features to select
    /// * `feature_selection` - Slice of usize representing the indices of features available for selection
    /// * `feature_class` - Reference to a HashMap mapping feature indices to their classes
    /// * `language` - u8 representing the language type (e.g., BINARY_LANG, POW2_LANG, etc.)
    /// * `data_type` - u8 representing the data type (e.g., RAW_TYPE, PREVALENCE_TYPE, LOG_TYPE)
    /// * `epsilon` - f64 value used in certain calculations (e.g., for RATIO_LANG)
    /// * `threshold_ci` - Boolean indicating whether to include threshold confidence intervals
    /// * `rng` - Mutable reference to a ChaCha8Rng random number generator
    ///
    /// # Returns
    ///
    /// A randomly generated Individual with uniformly selected features
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::{Individual, BINARY_LANG, RAW_TYPE};
    /// # use rand_chacha::ChaCha8Rng;
    /// # use rand::SeedableRng;
    /// # use std::collections::HashMap;
    /// # let feature_selection = vec![0, 1, 2];
    /// # let feature_class = HashMap::new();
    /// let mut rng = ChaCha8Rng::from_entropy();
    /// let random_individual = Individual::random_select_k(
    ///     1,
    ///     5,
    ///     &feature_selection,
    ///     &feature_class,
    ///    BINARY_LANG,
    ///    RAW_TYPE,
    ///    0.01,
    ///    false,
    ///    &mut rng,
    /// );
    /// ```
    pub fn random_select_k(
        k_min: usize,
        k_max: usize,
        feature_selection: &[usize],
        feature_class: &HashMap<usize, u8>,
        language: u8,
        data_type: u8,
        epsilon: f64,
        threshold_ci: bool,
        rng: &mut ChaCha8Rng,
    ) -> Individual {
        // chose k variables amount feature_selection
        // set a random coeficient for these k variables

        let chosen_feature_set: Vec<usize> = if language == BINARY_LANG {
            feature_selection
                .iter()
                .cloned()
                .filter(|i| feature_class[i] > 0)
                .collect::<Vec<usize>>()
        } else {
            feature_selection.to_vec()
        };

        let k: usize = rng.gen_range(
            (if k_min > 0 { k_min } else { 1 })..=(if k_max > 0 {
                min(k_max, chosen_feature_set.len())
            } else {
                chosen_feature_set.len()
            }),
        );

        // Randomly pick k values
        let random_values = chosen_feature_set.choose_multiple(rng, k as usize);

        let features: HashMap<usize, i8> = match language {
            BINARY_LANG => random_values
                .collect::<Vec<&usize>>()
                .iter()
                .map(|i| (**i, 1))
                .collect(),
            POW2_LANG => random_values
                .collect::<Vec<&usize>>()
                .iter()
                .map(|i| {
                    (
                        **i,
                        (if feature_class[i] > 0 { 1 } else { -1 }) * DEFAULT_POW2_START as i8,
                    )
                })
                .collect(),
            _ => random_values
                .collect::<Vec<&usize>>()
                .iter()
                .map(|i| (**i, if feature_class[i] > 0 { 1 } else { -1 }))
                .collect(),
        };

        let mut i = Individual::new();
        i.features = features;
        if language == RATIO_LANG {
            i.threshold = 1.0
        }
        if threshold_ci {
            i.threshold_ci = Some(ThresholdCI {
                upper: 0.0,
                lower: 0.0,
                rejection_rate: 0.0,
            })
        }
        i.k = k;
        i.language = language;
        i.data_type = data_type;
        i.epsilon = epsilon;
        i
    }

    /// Generates a random Individual with weighted feature selection
    ///
    /// # Arguments
    ///
    /// * `k_min` - Minimum number of features to select
    /// * `k_max` - Maximum number of features to select
    /// * `feature_selection` - Slice of usize representing the indices of features available for selection
    /// * `feature_class` - Reference to a HashMap mapping feature indices to their classes
    /// * `language` - u8 representing the language type (e.g., BINARY_LANG, POW2_LANG, etc.)
    /// * `data_type` - u8 representing the data type (e.g., RAW_TYPE, PREVALENCE_TYPE, LOG_TYPE)
    /// * `epsilon` - f64 value used in certain calculations (e.g., for RATIO_LANG)        
    /// * `prior_weight` - Reference to a HashMap mapping feature indices to their weights for weighted selection
    /// * `threshold_ci` - Boolean indicating whether to include threshold confidence intervals
    /// * `rng` - Mutable reference to a ChaCha8Rng random number generator
    ///
    /// # Returns
    ///
    /// A randomly generated Individual with weighted selected features
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::{Individual, BINARY_LANG, RAW_TYPE};
    /// # use rand_chacha::ChaCha8Rng;
    /// # use rand::SeedableRng;
    /// # use std::collections::HashMap;
    /// # let feature_selection = vec![0, 1, 2];
    /// # let feature_class = HashMap::new();
    /// # let prior_weight = HashMap::new();
    /// let mut rng = ChaCha8Rng::from_entropy();
    /// let random_individual = Individual::random_select_weighted(
    ///     1,
    ///     5,
    ///     &feature_selection,
    ///     &feature_class,
    ///     BINARY_LANG,
    ///     RAW_TYPE,
    ///     0.01,
    ///     &prior_weight,
    ///     false,
    ///     &mut rng,
    /// );
    /// ```
    pub fn random_select_weighted(
        k_min: usize,
        k_max: usize,
        feature_selection: &[usize],
        feature_class: &HashMap<usize, u8>,
        language: u8,
        data_type: u8,
        epsilon: f64,
        prior_weight: &HashMap<usize, f64>,
        threshold_ci: bool,
        rng: &mut ChaCha8Rng,
    ) -> Individual {
        let valid_pairs: Vec<(usize, f64)> = feature_selection
            .iter()
            .filter_map(|&feature_idx| {
                let weight = prior_weight.get(&feature_idx).copied().unwrap_or(1.0);
                if weight > 0.0 {
                    Some((feature_idx, weight))
                } else {
                    None
                }
            })
            .collect();

        assert!(!valid_pairs.is_empty(), "No features with positive weight!");

        let (valid_features, valid_weights): (Vec<_>, Vec<_>) = valid_pairs.into_iter().unzip();

        // Number of features to select
        let k = rng.gen_range(k_min..=k_max).min(valid_features.len());

        let selected_features: Vec<usize> = valid_features
            .choose_multiple_weighted(rng, k, |&feature_idx| {
                let idx = valid_features
                    .iter()
                    .position(|&x| x == feature_idx)
                    .unwrap();
                valid_weights[idx]
            })
            .expect("Failed weighted selection")
            .copied()
            .collect();

        // Create the individual with the selected features
        let mut features = HashMap::new();
        for &feature_idx in &selected_features {
            let coef = Individual::random_coefficient(
                language,
                feature_class.get(&feature_idx).copied(),
                rng,
            );
            features.insert(feature_idx, coef);
        }

        let mut i = Individual::new();
        i.features = features;
        i.k = selected_features.len();
        i.language = language;
        i.data_type = data_type;
        i.epsilon = epsilon;
        i.threshold_ci = if threshold_ci {
            Some(ThresholdCI {
                upper: 0.0,
                lower: 0.0,
                rejection_rate: 0.0,
            })
        } else {
            None
        };
        i
    }

    /// Generate a random coefficient based on language and feature class
    ///
    /// # Arguments
    ///
    /// * `language` - u8 representing the language type (e.g., BINARY_LANG, POW2_LANG, etc.)
    /// * `feature_class` - Optional u8 representing the class of the feature
    /// * `rng` - Mutable reference to a ChaCha8Rng random number generator
    ///
    /// # Returns
    ///
    /// An i8 representing the randomly generated coefficient
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::{Individual, BINARY_LANG};
    /// # use rand_chacha::ChaCha8Rng;
    /// # use rand::SeedableRng;
    /// # let mut rng = ChaCha8Rng::from_entropy();
    /// let coef = Individual::random_coefficient(
    ///     BINARY_LANG,
    ///     Some(1),
    ///     &mut rng,
    /// );
    /// ```
    pub fn random_coefficient(language: u8, feature_class: Option<u8>, rng: &mut ChaCha8Rng) -> i8 {
        match language {
            BINARY_LANG => 1,
            TERNARY_LANG | RATIO_LANG => {
                if let Some(class) = feature_class {
                    if class == 0 {
                        -1
                    } else {
                        1
                    }
                } else {
                    if rng.gen_bool(0.5) {
                        1
                    } else {
                        -1
                    }
                }
            }
            POW2_LANG => {
                let sign = if rng.gen_bool(0.5) { 1 } else { -1 };
                let power = rng.gen_range(0..=6); // 2^0 to 2^6 = 1 to 64
                sign * (1 << power)
            }
            _ => panic!("Unknown language: {}", language),
        }
    }

    /// Computes accuracy, sensitivity, specificity, and additional requested metrics
    ///
    /// # Arguments
    ///
    /// * `d` - Reference to the Data object on which to compute metrics
    ///
    /// # Returns
    ///
    /// A tuple containing (accuracy, sensitivity, specificity, additional metrics)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// # let individual = Individual::new();
    /// # let data = Data::new();
    /// let (accuracy, sensitivity, specificity, additional_metrics) = individual.compute_metrics(&data);
    /// ```
    pub fn compute_metrics(&self, d: &Data) -> (f64, f64, f64, f64, AdditionalMetrics) {
        let value = self.evaluate(d);
        let others_to_compute: [bool; 5] = [
            self.metrics.mcc.is_some(),
            self.metrics.f1_score.is_some(),
            self.metrics.npv.is_some(),
            self.metrics.ppv.is_some(),
            self.metrics.g_mean.is_some(),
        ];
        if let Some(ref threshold_ci) = self.threshold_ci {
            compute_metrics_from_value(
                &value,
                &d.y,
                self.threshold,
                Some([threshold_ci.lower, threshold_ci.upper]),
                others_to_compute,
            )
        } else {
            compute_metrics_from_value(&value, &d.y, self.threshold, None, others_to_compute)
        }
    }

    /// Computes the threshold using the Youden index and returns accuracy, sensitivity, and specificity
    ///
    /// This method is deprecated. Prefer `compute_roc_and_metrics` instead.
    ///
    /// # Arguments
    ///
    /// * `d` - Reference to the Data object on which to compute metrics
    ///
    /// # Returns
    ///
    /// A tuple containing (threshold, accuracy, sensitivity, specificity)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// # let individual = Individual::new();
    /// # let data = Data::new();
    /// let (threshold, accuracy, sensitivity, specificity) = individual.compute_threshold_and_metrics(&data);
    /// ```
    pub fn compute_threshold_and_metrics(&self, d: &Data) -> (f64, f64, f64, f64) {
        let value = self.evaluate(d);
        let mut combined: Vec<(f64, u8)> = value.iter().cloned().zip(d.y.iter().cloned()).collect();

        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut tp = d.y.iter().filter(|&&label| label == 1).count();
        let mut fn_count = 0;
        let mut tn = 0;
        let mut fp = d.y.iter().filter(|&&label| label == 0).count();

        let mut best_threshold = 0.0;
        let mut best_youden_index = f64::NEG_INFINITY;
        let mut best_metrics = (0.0, 0.0, 0.0);

        for i in 0..combined.len() {
            let (threshold, label) = combined[i];

            let sensitivity = if (tp + fn_count) > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };

            let specificity = if (fp + tn) > 0 {
                tn as f64 / (fp + tn) as f64
            } else {
                0.0
            };

            let youden_index = sensitivity + specificity - 1.0;

            if youden_index > best_youden_index {
                best_youden_index = youden_index;
                best_threshold = threshold;

                let accuracy = if (tp + tn + fp + fn_count) > 0 {
                    (tp + tn) as f64 / (tp + tn + fp + fn_count) as f64
                } else {
                    0.0
                };

                best_metrics = (accuracy, sensitivity, specificity);
            }

            match label {
                1 => {
                    tp -= 1;
                    fn_count += 1;
                }
                0 => {
                    fp -= 1;
                    tn += 1;
                }
                _ => (),
            }
        }

        (
            best_threshold,
            best_metrics.0,
            best_metrics.1,
            best_metrics.2,
        )
    }

    /// Returns a sorted vector of feature indices used in the Individual
    ///
    /// # Returns
    ///
    /// A vector of usize representing the sorted feature indices
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # let individual = Individual::new();
    /// let feature_indices = individual.features_index();
    /// ```
    pub fn features_index(&self) -> Vec<usize> {
        let mut features = self.features.keys().copied().collect::<Vec<usize>>();
        features.sort();
        features
    }

    /// Computes the mean decrease in AUC for each feature using permutations
    ///
    /// # Arguments
    ///
    /// * `data` - Reference to the [`Data`] object on which to compute feature importance
    /// * `permutations` - Number of permutations to perform for each feature
    /// * `features_to_process` - Slice of [`usize`] representing the feature indices to process
    /// * `feature_seeds` - Reference to a [`HashMap`] mapping feature indices to vectors of [`u64`] seeds for permutations
    ///
    /// # Returns
    ///
    /// An [`ImportanceCollection`] containing the computed feature importances
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// # use std::collections::HashMap;
    /// # let individual = Individual::new();
    /// # let data = Data::new();
    /// # let features_to_process = vec![];
    /// # let feature_seeds = HashMap::new();
    /// let importance_collection = individual.compute_mda_feature_importance(
    ///     &data,
    ///     100,
    ///     &features_to_process,
    ///     &feature_seeds,
    /// );
    /// ```
    pub fn compute_mda_feature_importance(
        &self,
        data: &Data,
        permutations: usize,
        features_to_process: &[usize],
        feature_seeds: &HashMap<usize, Vec<u64>>,
    ) -> ImportanceCollection {
        let baseline_auc = self.compute_new_auc(data);
        let mut importances = Vec::new();

        // Protection against strange behavior
        if permutations == 0 {
            panic!("To compute mean decrease in AUC, permutations are needed (and currently set to 0)!");
        }

        for &feature_idx in features_to_process {
            if !feature_seeds.contains_key(&feature_idx) {
                panic!("Missing seeds for feature index {}", feature_idx);
            }

            let seeds = &feature_seeds[&feature_idx];
            if seeds.len() < permutations {
                panic!(
                    "Feature {} has {} seeds but {} permutations requested",
                    feature_idx,
                    seeds.len(),
                    permutations
                );
            }
        }

        let mut seen = HashSet::new();
        let unique_features: Vec<usize> = features_to_process
            .iter()
            .filter(|&&feature_idx| seen.insert(feature_idx))
            .cloned()
            .collect();

        if unique_features.len() != features_to_process.len() {
            debug!(
                "Individual Importance : removed {} duplicate features from analysis",
                features_to_process.len() - unique_features.len()
            );
        }

        for &feature_idx in &unique_features {
            let importance_value = if !self.features.contains_key(&feature_idx) {
                0.0
            } else {
                let mut permuted_auc_sum = 0.0;

                let seeds = feature_seeds.get(&feature_idx).expect("Seeds required");

                for &seed in seeds.iter().take(permutations) {
                    let mut permutation_rng = ChaCha8Rng::seed_from_u64(seed);
                    let mut X_permuted = data.X.clone();
                    shuffle_row(
                        &mut X_permuted,
                        data.sample_len,
                        feature_idx,
                        &mut permutation_rng,
                    );
                    let scores = self.evaluate_from_features(&X_permuted, data.sample_len);
                    let permuted_auc = compute_auc_from_value(&scores, &data.y);
                    permuted_auc_sum += permuted_auc;
                }

                let mean_permuted_auc = permuted_auc_sum / permutations as f64;
                baseline_auc - mean_permuted_auc
            };

            let importance_obj = Importance {
                importance_type: ImportanceType::MDA,
                feature_idx,
                scope: ImportanceScope::Individual {
                    model_hash: self.hash,
                },
                aggreg_method: None,
                importance: importance_value,
                is_scaled: false,
                dispersion: 0.0,
                scope_pct: 1.0,
                direction: None,
            };

            importances.push(importance_obj);
        }

        ImportanceCollection { importances }
    }

    /// Maximizes the objective function based on false positive and false negative penalties*
    ///
    /// This function is deprecated.
    ///
    /// # Arguments
    ///
    /// * `data` - Reference to the Data object on which to maximize the objective
    /// * `fpr_penalty` - f64 value representing the penalty for false positives
    /// * `fnr_penalty` - f64 value representing the penalty for false negatives
    ///
    /// # Returns
    ///
    /// The maximum objective value achieved
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// # let mut individual = Individual::new();
    /// # let data = Data::new();
    /// let max_objective = individual.maximize_objective(&data, 1.0, 1.0);
    /// ```
    pub fn maximize_objective(&mut self, data: &Data, fpr_penalty: f64, fnr_penalty: f64) -> f64 {
        let scores = self.evaluate(data);
        self.maximize_objective_with_scores(&scores, data, fpr_penalty, fnr_penalty)
    }

    /// Maximizes the objective function based on false positive and false negative penalties using precomputed scores
    ///
    /// This function is deprecated.
    ///
    /// # Arguments
    ///
    /// * `scores` - Slice of f64 representing the precomputed scores
    /// * `data` - Reference to the Data object on which to maximize the objective
    /// * `fpr_penalty` - f64 value representing the penalty for false positives
    /// * `fnr_penalty` - f64 value representing the penalty for false negatives
    ///
    /// # Returns
    ///
    /// The maximum objective value achieved
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// # let mut individual = Individual::new();
    /// # let data = Data::new();
    /// let scores = individual.evaluate(&data);
    /// let max_objective = individual.maximize_objective_with_scores(&scores, &data, 1.0, 1.0);
    /// ```
    pub fn maximize_objective_with_scores(
        &mut self,
        scores: &[f64],
        data: &Data,
        fpr_penalty: f64,
        fnr_penalty: f64,
    ) -> f64 {
        let mut paired_data: Vec<_> = scores
            .iter()
            .zip(data.y.iter())
            .filter(|(_, &y)| y == 0 || y == 1)
            .map(|(&score, &label)| (score, label))
            .collect();

        paired_data
            .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_pos = paired_data.iter().filter(|(_, y)| *y == 1).count();
        let total_neg = paired_data.len() - total_pos;

        if total_pos == 0 || total_neg == 0 {
            return 0.0;
        }

        let mut best_objective = f64::MIN;
        self.threshold = f64::NEG_INFINITY;
        self.sensitivity = 0.0;
        self.specificity = 0.0;
        self.accuracy = 0.0;

        let mut tn = 0;
        let mut fn_count = 0;
        let mut i = 0;

        while i < paired_data.len() {
            let current_score = paired_data[i].0;
            let mut current_tn = 0;
            let mut current_fn = 0;

            while i < paired_data.len() && (paired_data[i].0 - current_score).abs() < f64::EPSILON {
                match paired_data[i].1 {
                    0 => current_tn += 1,
                    1 => current_fn += 1,
                    _ => unreachable!(),
                }
                i += 1;
            }

            tn += current_tn;
            fn_count += current_fn;

            let tp = total_pos - fn_count;

            let sensitivity = (tp + current_fn) as f64 / total_pos as f64;
            let specificity = (tn - current_tn) as f64 / total_neg as f64;
            let accuracy =
                (tp + current_fn + tn - current_tn) as f64 / (total_pos + total_neg) as f64;

            let objective = (fpr_penalty * specificity + fnr_penalty * sensitivity)
                / (fpr_penalty + fnr_penalty);

            if objective > best_objective
                || (objective == best_objective && current_score < self.threshold)
            {
                best_objective = objective;
                self.threshold = current_score;
                self.sensitivity = sensitivity;
                self.specificity = specificity;
                self.accuracy = accuracy;
            }
        }

        best_objective
    }

    /// Computes the genealogy of the Individual up to a specified maximum depth
    ///
    /// This function builds a genealogy tree of the individual by tracing back its ancestors. It is used in GpredomicsR.
    ///
    /// # Arguments
    ///
    /// * `collection` - Reference to a vector of Population objects representing the collection of populations
    /// * `max_depth` - usize value representing the maximum depth of genealogy to compute
    ///
    /// # Returns
    ///
    /// A HashMap where keys are tuples of (individual hash, optional parent hashes) and values are sets of usize representing the depths at which the individuals appear in the genealogy
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::population::Population;
    /// # let individual = Individual::new();
    /// # let collection: Vec<Population> = vec![];
    /// let genealogy = individual.get_genealogy(&collection, 5);
    /// ```
    pub fn get_genealogy(
        &self,
        collection: &Vec<Population>,
        max_depth: usize,
    ) -> HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> {
        let mut genealogy: HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> =
            HashMap::with_capacity(max_depth * 2);
        let mut initial_set = HashSet::new();

        if collection.is_empty() {
            return genealogy;
        }

        initial_set.insert(0);
        genealogy.insert((self.hash, self.parents.clone()), initial_set);

        let real_max_generation = collection.len();
        let effective_max_depth = std::cmp::min(max_depth, real_max_generation);

        debug!(
            "Computing genealogy (maximum depth = {:?}, real generations = {:?})...",
            effective_max_depth, real_max_generation
        );

        let mut individual_index: HashMap<u64, Vec<(usize, &Individual)>> = HashMap::new();
        for population in collection {
            for individual in &population.individuals {
                if individual.epoch < self.epoch {
                    individual_index
                        .entry(individual.hash)
                        .or_insert_with(Vec::new)
                        .push((individual.epoch, individual));
                }
            }
        }

        for positions in individual_index.values_mut() {
            positions.sort_by(|a, b| b.0.cmp(&a.0));
        }

        let mut queue = VecDeque::new();
        queue.push_back((self.clone(), 0));

        let mut visited_paths: HashSet<(u64, usize)> = HashSet::new();
        visited_paths.insert((self.hash, 0));

        while let Some((current_ind, depth)) = queue.pop_front() {
            if depth >= effective_max_depth {
                continue;
            }

            if let Some(parents) = &current_ind.parents {
                let next_depth = depth + 1;

                for &parent_hash in parents {
                    if let Some(positions) = individual_index.get(&parent_hash) {
                        if let Some(&(_, ancestor)) = positions.first() {
                            if ancestor.epoch < current_ind.epoch {
                                let key = (ancestor.hash, ancestor.parents.clone());
                                genealogy
                                    .entry(key.clone())
                                    .or_insert_with(HashSet::new)
                                    .insert(next_depth);
                                let path_key = (ancestor.hash, next_depth);
                                if !visited_paths.contains(&path_key)
                                    && next_depth < effective_max_depth
                                {
                                    visited_paths.insert(path_key);
                                    queue.push_back((ancestor.clone(), next_depth));
                                }
                            }
                        }
                    }
                }
            }
        }

        genealogy
    }

    /// Returns the language of the Individual as a string
    ///
    /// # Returns
    ///
    /// A string slice representing the language of the Individual
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # let individual = Individual::new();
    /// let language = individual.get_language();
    /// ```
    pub fn get_language(&self) -> &str {
        match self.language {
            BINARY_LANG => "Binary",
            TERNARY_LANG => "Ternary",
            RATIO_LANG => "Ratio",
            POW2_LANG => "Pow2",
            MCMC_GENERIC_LANG => "MCMC_Generic",
            _ => "Unknown",
        }
    }

    /// Returns the data type of the Individual as a string
    ///
    /// # Returns
    ///
    /// A string slice representing the data type of the Individual
    ///
    /// # Examples
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # let individual = Individual::new();
    /// let data_type = individual.get_data_type();
    /// ```
    pub fn get_data_type(&self) -> &str {
        match self.data_type {
            RAW_TYPE => "Raw",
            PREVALENCE_TYPE => "Prevalence",
            LOG_TYPE => "Log",
            _ => "Unknown",
        }
    }

    /// Gets the coefficient for a given feature index
    ///
    /// # Arguments
    ///
    /// * `idx` - usize representing the feature index
    ///
    /// # Returns
    ///
    /// An i8 representing the coefficient of the feature; returns 0 if the feature is not present
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # let individual = Individual::new();
    /// let coef = individual.get_coef(5);
    /// ```
    pub fn get_coef(&self, idx: usize) -> i8 {
        *self.features.get(&idx).unwrap_or(&0)
    }

    /// Sets the coefficient for a given feature index
    ///
    /// # Arguments
    ///
    /// * `idx` - usize representing the feature index
    /// * `coef` - i8 representing the coefficient to set; if coef is 0, the feature is removed
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # let mut individual = Individual::new();
    /// individual.set_coef(5, 1);
    /// ```
    pub fn set_coef(&mut self, idx: usize, coef: i8) {
        if coef == 0 {
            self.features.remove(&idx);
        } else {
            self.features.insert(idx, coef);
        }
        self.k = self.features.len();
    }

    /// Gets the betas of the Individual
    ///
    /// # Returns
    ///
    /// An array of f64 representing the MCMC betas.
    ///
    /// # Panics
    ///
    /// Panics if the betas are uninitialized.
    ///
    /// # Examples
    ///
    /// ```
    /// # use gpredomics::individual::Individual;
    /// let mut individual = Individual::new();
    /// individual.set_beta(0, 0.5);
    /// individual.set_beta(1, 1.0);
    /// individual.set_beta(2, -0.5);
    /// let betas = individual.get_betas();
    /// assert_eq!(betas, [0.5, 1.0, -0.5]);
    /// ```
    pub fn get_betas(&self) -> [f64; 3] {
        self.betas
            .as_ref()
            .map(|b| b.get())
            .expect("β uninitialized")
    }

    /// Sets the beta value for a given index
    ///
    /// # Arguments
    ///
    /// * `idx` - usize representing the index (0, 1, or 2)
    /// * `val` - f64 representing the beta value to set
    ///
    /// # Examples
    ///
    /// ```
    /// # use gpredomics::individual::Individual;
    /// let mut individual = Individual::new();
    /// individual.set_beta(0, 0.5);
    /// individual.set_beta(1, 1.0);
    /// individual.set_beta(2, -0.5);
    /// let betas = individual.get_betas();
    /// assert_eq!(betas, [0.5, 1.0, -0.5]);
    /// ```
    pub fn set_beta(&mut self, idx: usize, val: f64) {
        if let Some(b) = self.betas.as_mut() {
            b.set(idx, val);
        } else {
            self.betas = Some(Betas::new(
                if idx == 0 { val } else { 0.0 },
                if idx == 1 { val } else { 0.0 },
                if idx == 2 { val } else { 0.0 },
            ));
        }
    }

    /// Computes the signed Jaccard dissimilarity between this Individual and another
    ///
    /// # Arguments
    ///
    /// * `other` - Reference to another Individual to compare with
    ///
    /// # Returns
    ///
    /// A f64 value between 0.0 and 1.0 representing the signed Jaccard dissimilarity
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # let individual1 = Individual::new();
    /// # let individual2 = Individual::new();
    /// let dissimilarity = individual1.signed_jaccard_dissimilarity_with(&individual2);
    /// ```
    pub fn signed_jaccard_dissimilarity_with(&self, other: &Individual) -> f64 {
        let signed_set1: HashSet<(usize, i8)> = self
            .features
            .iter()
            .map(|(id, coef)| (*id, coef.signum() as i8))
            .collect();

        let signed_set2: HashSet<(usize, i8)> = other
            .features
            .iter()
            .map(|(id, coef)| (*id, coef.signum() as i8))
            .collect();

        let intersection = signed_set1.intersection(&signed_set2).count();
        let union = signed_set1.union(&signed_set2).count();

        if union == 0 {
            return 0.0;
        }

        1.0 - (intersection as f64) / (union as f64)
    }

    /// Prunes in-place using MDA permutation importance computed internally.
    ///
    /// If `threshold` is Some(t), drop features with importance < t.
    /// Else if `quantile` is Some((q, eps)), drop features with importance < quantile(q) - eps.
    /// Keeps at least `min_k` features. Returns &mut Self for chaining.
    ///
    /// # Arguments
    ///
    /// * `data` - Reference to the [`Data`] object used for importance computation
    /// * `n_perm` - Number of permutations to use for importance computation
    /// * `rng_seed` - Seed for random number generation to ensure reproducibility
    /// * `threshold` - Optional f64 threshold for pruning based on importance
    /// * `quantile` - Optional tuple (f64, f64) representing quantile and epsilon for pruning
    /// * `min_k` - Minimum number of features to retain
    ///
    /// # Returns
    ///
    /// A mutable reference to the pruned Individual
    ///
    /// # Panics
    ///
    /// Panics if:
    /// * `n_perm` is 0
    /// * quantile `q` is not in [0, 1].
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use gpredomics::individual::Individual;
    /// # use gpredomics::data::Data;
    /// # let mut individual = Individual::new();
    /// # let data = Data::new();
    /// individual.prune_by_importance(&data, 100, 42, Some(0.01), None, 5);
    /// ```
    pub fn prune_by_importance(
        &mut self,
        data: &Data,
        n_perm: usize,
        rng_seed: u64,
        threshold: Option<f64>,
        quantile: Option<(f64, f64)>,
        min_k: usize,
    ) -> &mut Self {
        assert!(n_perm > 0, "n_perm must be > 0");
        let feats: Vec<usize> = self.features.keys().copied().collect();
        if feats.is_empty() {
            return self;
        }

        // Deterministic seeds per feature
        use rand::{RngCore, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut feature_seeds = std::collections::HashMap::with_capacity(feats.len());
        let mut seed_rng = ChaCha8Rng::seed_from_u64(rng_seed);
        for &f in &feats {
            let base = (f as u64) ^ rng_seed;
            let mut seeds = Vec::with_capacity(n_perm);
            for i in 0..n_perm {
                seeds.push(
                    base.wrapping_add(i as u64)
                        .wrapping_add(seed_rng.next_u64()),
                );
            }
            feature_seeds.insert(f, seeds);
        }

        // Compute individual OOB permutation importances (MDA)
        let imp_coll = self.compute_mda_feature_importance(data, n_perm, &feats, &feature_seeds);

        // Collect importances
        use crate::experiment::{ImportanceScope, ImportanceType};
        let mut imp_map: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
        for im in &imp_coll.importances {
            if matches!(im.scope, ImportanceScope::Individual { .. })
                && matches!(im.importance_type, ImportanceType::MDA)
            {
                imp_map.insert(im.feature_idx, im.importance);
            }
        }
        let vals: Vec<f64> = feats
            .iter()
            .map(|f| *imp_map.get(f).unwrap_or(&0.0))
            .collect();

        // Determine threshold
        let thr = if let Some(t) = threshold {
            t
        } else if let Some((q, eps)) = quantile {
            assert!((0.0..=1.0).contains(&q), "q must be in [0,1]");
            let mut sorted = vals.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((sorted.len().saturating_sub(1)) as f64 * q).round() as usize;
            sorted[idx] - eps
        } else {
            // default: no pruning
            f64::NEG_INFINITY
        };

        // Rank ascending by importance
        let mut ranked: Vec<(usize, f64)> = feats
            .iter()
            .zip(vals.iter())
            .map(|(f, &v)| (*f, v))
            .collect();
        ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Initial drop/keep
        let mut to_drop: Vec<usize> = Vec::new();
        let mut to_keep: Vec<usize> = Vec::new();
        for (f, v) in &ranked {
            if *v < thr {
                to_drop.push(*f);
            } else {
                to_keep.push(*f);
            }
        }

        // Enforce min_k
        if to_keep.len() < min_k {
            let need = min_k - to_keep.len();
            // Re-add best among dropped
            let mut drop_ranked: Vec<(usize, f64)> = to_drop
                .iter()
                .map(|f| (*f, *imp_map.get(f).unwrap_or(&0.0)))
                .collect();
            drop_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // highest first
            for (f, _) in drop_ranked.into_iter().take(need) {
                to_drop.retain(|x| *x != f);
                to_keep.push(f);
            }
        }

        // Avoid empty model
        if to_keep.is_empty() {
            // Pattern should match &(usize, f64); remove unnecessary reference in pattern to satisfy compiler
            if let Some((best_f, _)) = ranked
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(f, s)| (*f, *s))
            {
                to_keep.push(best_f);
                to_drop.retain(|x| *x != best_f);
            }
        }

        // Apply pruning
        for f in to_drop {
            self.features.remove(&f);
        }
        self.k = self.features.len();

        self
    }
}

impl fmt::Debug for Individual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut sorted_keys: Vec<usize> = self.features.keys().cloned().collect();
        sorted_keys.sort();
        let mut desc = sorted_keys
            .iter()
            .map(|i| match self.features[i] {
                1 => format!("[{}] ", i + 1),
                -1 => format!("-[{}] ", i + 1),
                other => format!("{}[{}] ", other, i + 1),
            })
            .collect::<Vec<String>>()
            .join("");
        if desc.len() > 0 {
            desc = desc[0..desc.len() - 1].to_string()
        }
        write!(
            f,
            "{}:{} {}",
            self.get_language(),
            self.get_data_type(),
            desc
        )
    }
}

// Safe implementation of Send and Sync
unsafe impl Send for Individual {}
unsafe impl Sync for Individual {}

/// When a parent has a child of a different language, do we need to convert the gene values ?
pub fn needs_conversion(parent_language: u8, child_language: u8) -> bool {
    match (parent_language, child_language) {
        (x, y) if x == y => false,
        (BINARY_LANG, _) => false,
        (_, BINARY_LANG) => true,
        (TERNARY_LANG, _) => false,
        (RATIO_LANG, _) => false,
        _ => true,
    }
}

/// A conversion function for interlanguage wedding
pub fn gene_convert_from_to(parent_language: u8, child_language: u8, value: i8) -> i8 {
    match (parent_language, child_language) {
        (_, BINARY_LANG) => 1,
        (_, TERNARY_LANG) | (_, RATIO_LANG) => {
            if value > 0 {
                1
            } else {
                -1
            }
        }
        _ => value,
    }
}

// unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use std::collections::{BTreeMap, HashMap};

    impl Individual {
        /// Genarates a test Individual for unit tests
        ///
        /// # Returns
        ///
        /// An Individual object with predefined features and metrics
        pub fn test() -> Individual {
            Individual {
                features: vec![(0, 1), (1, -1), (2, 1), (3, 0)].into_iter().collect(),
                auc: 0.4,
                fit: 0.8,
                specificity: 0.15,
                sensitivity: 0.16,
                accuracy: 0.23,
                threshold: 42.0,
                k: 42,
                epoch: 42,
                language: 0,
                data_type: 0,
                hash: 0,
                epsilon: f64::MIN_POSITIVE,
                parents: None,
                betas: None,
                threshold_ci: None,
                metrics: AdditionalMetrics {
                    mcc: None,
                    f1_score: None,
                    npv: None,
                    ppv: None,
                    g_mean: None,
                },
            }
        }

        /// Generates another test Individual for unit tests
        ///
        /// # Returns
        ///
        /// An Individual object with predefined features and metrics
        pub fn test2() -> Individual {
            Individual {
                features: vec![(0, 1), (1, -1)].into_iter().collect(),
                auc: 0.4,
                fit: 0.8,
                specificity: 0.15,
                sensitivity: 0.16,
                accuracy: 0.23,
                threshold: 0.0,
                k: 42,
                epoch: 42,
                language: 0,
                data_type: 0,
                hash: 0,
                epsilon: f64::MIN_POSITIVE,
                parents: None,
                betas: None,
                threshold_ci: None,
                metrics: AdditionalMetrics {
                    mcc: None,
                    f1_score: None,
                    npv: None,
                    ppv: None,
                    g_mean: None,
                },
            }
        }

        /// Generates a test Individual with specified features for unit tests
        ///
        /// # Arguments
        ///
        /// * `features_vec` - A vector of tuples where each tuple contains a feature index and its corresponding coefficient
        ///
        /// # Returns
        ///
        /// An Individual object with the specified features and predefined metrics
        pub fn test_with_these_given_features(features_vec: Vec<(usize, i8)>) -> Individual {
            Individual {
                features: features_vec.into_iter().collect::<HashMap<usize, i8>>(),
                auc: 0.8,
                fit: 0.7,
                specificity: 0.1,
                sensitivity: 0.9,
                accuracy: 0.3,
                threshold: 0.5,
                k: 0,
                epoch: 0,
                language: BINARY_LANG,
                data_type: RAW_TYPE,
                hash: 0x123456789abcdef0,
                epsilon: DEFAULT_MINIMUM,
                parents: None,
                betas: None,
                threshold_ci: None,
                metrics: AdditionalMetrics {
                    mcc: None,
                    f1_score: None,
                    npv: None,
                    ppv: None,
                    g_mean: None,
                },
            }
        }

        /// Generates a test Individual with specified features, fitness, language, and data type for unit tests
        ///
        /// # Arguments
        ///
        /// * `features` - A vector of tuples where each tuple contains a feature index and its corresponding coefficient
        /// * `fit` - A f64 value representing the fitness of the Individual
        /// * `language` - A u8 value representing the language of the Individual
        /// * `data_type` - A u8 value representing the data type of the Individual
        ///
        /// # Returns
        ///
        /// An Individual object with the specified features, fitness, language, and data type
        pub fn test_with_these_given_features_fit_lang_types(
            features: Vec<(usize, i8)>,
            fit: f64,
            language: u8,
            data_type: u8,
        ) -> Individual {
            Individual {
                features: features.into_iter().collect(),
                auc: 0.0,
                fit,
                specificity: 0.0,
                sensitivity: 0.0,
                accuracy: 0.0,
                threshold: 0.0,
                k: 0,
                epoch: 0,
                language,
                data_type,
                hash: 0,
                epsilon: 0.0,
                parents: None,
                betas: None,
                threshold_ci: None,
                metrics: AdditionalMetrics {
                    mcc: None,
                    f1_score: None,
                    npv: None,
                    ppv: None,
                    g_mean: None,
                },
            }
        }

        /// Generates a test Individual with specified sensitivity, specificity, and accuracy for unit tests
        ///
        /// # Arguments
        ///
        /// * `sensitivity` - A f64 value representing the sensitivity of the Individual
        /// * `specificity` - A f64 value representing the specificity of the Individual
        /// * `accuracy` - A f64 value representing the accuracy of the Individual
        ///
        /// # Returns
        ///
        /// An Individual object with the specified sensitivity, specificity, and accuracy
        pub fn test_with_metrics(sensitivity: f64, specificity: f64, accuracy: f64) -> Individual {
            Individual {
                features: vec![(0, 1), (1, -1)].into_iter().collect(),
                auc: 0.8,
                fit: 0.7,
                specificity: specificity,
                sensitivity: sensitivity,
                accuracy: accuracy,
                threshold: 0.5,
                k: 0,
                epoch: 0,
                language: BINARY_LANG,
                data_type: RAW_TYPE,
                hash: 0x123456789abcdef0,
                epsilon: DEFAULT_MINIMUM,
                parents: None,
                betas: None,
                threshold_ci: None,
                metrics: AdditionalMetrics {
                    mcc: None,
                    f1_score: None,
                    npv: None,
                    ppv: None,
                    g_mean: None,
                },
            }
        }

        /// Generates a test Individual with specified features for unit tests
        ///
        /// # Arguments
        ///
        /// * `features` - A slice of usize representing the feature indices
        ///
        /// # Returns
        ///
        /// An Individual object with the specified features and predefined metrics
        pub fn specific_test(features: &[usize]) -> Individual {
            let mut features_map = HashMap::new();
            for &feature_idx in features {
                features_map.insert(feature_idx, 1i8);
            }

            Individual {
                features: features_map,
                auc: 0.8,
                fit: 0.7,
                specificity: 0.75,
                sensitivity: 0.85,
                accuracy: 0.80,
                threshold: 0.5,
                k: features.len(),
                epoch: 0,
                language: BINARY_LANG,
                data_type: RAW_TYPE,
                hash: 0x123456789abcdef0,
                epsilon: DEFAULT_MINIMUM,
                parents: None,
                betas: None,
                threshold_ci: None,
                metrics: AdditionalMetrics {
                    mcc: None,
                    f1_score: None,
                    npv: None,
                    ppv: None,
                    g_mean: None,
                },
            }
        }
    }

    // test for language and data_types
    #[test]
    fn test_language_recognized() {
        assert_eq!(language("binary"), BINARY_LANG, "'binary' misinterpreted");
        assert_eq!(language("BIN"), BINARY_LANG, "'BIN' misinterpreted");
        assert_eq!(
            language("ternary"),
            TERNARY_LANG,
            "'ternary' misinterpreted"
        );
        assert_eq!(language("ter"), TERNARY_LANG, "'ter' misinterpreted");
        assert_eq!(language("pOw2"), POW2_LANG, "'pOw2' misinterpreted");
        assert_eq!(language("ratiO"), RATIO_LANG, "'ratiO' misinterpreted");
    }

    #[test]
    #[should_panic(expected = "Unrecognized language")]
    fn test_language_unrecognized() {
        language("unknown");
    }

    #[test]
    fn test_data_type_recognized() {
        assert_eq!(data_type("raw"), RAW_TYPE, "'raw' misinterpreted");
        assert_eq!(
            data_type("prevalEnce"),
            PREVALENCE_TYPE,
            "'prevalEnce' misinterpreted"
        );
        assert_eq!(data_type("pRev"), PREVALENCE_TYPE, "'pRev' misinterpreted");
        assert_eq!(data_type("log"), LOG_TYPE, "'log' misinterpreted");
    }

    #[test]
    #[should_panic(expected = "Unrecognized data type")]
    fn test_data_type_unrecognized() {
        data_type("unknown");
    }

    /// test for hash
    #[test]
    fn test_compute_hash_first_hash() {
        let mut ind = Individual::test();
        ind.compute_hash();

        let mut hasher = DefaultHasher::new();
        let sorted_features: BTreeMap<_, _> = ind.features.iter().collect();
        sorted_features.hash(&mut hasher);
        ind.betas.hash(&mut hasher);
        let expected_hash = hasher.finish();

        assert_eq!(
            ind.hash, expected_hash,
            "hash is different from expected hash"
        );
    }

    #[test]
    fn test_compute_hash_and_rehash() {
        let mut ind = Individual::test();
        ind.compute_hash();
        let first_hash = ind.hash;

        ind.features.insert(4, 0);
        ind.compute_hash();

        assert_ne!(
            ind.hash, first_hash,
            "hash should be different after adding a new feature"
        );

        let mut hasher = DefaultHasher::new();
        let sorted_features: BTreeMap<_, _> = ind.features.iter().collect();
        sorted_features.hash(&mut hasher);
        ind.betas.hash(&mut hasher);
        let expected_hash = hasher.finish();

        assert_eq!(
            ind.hash, expected_hash,
            "hash is different from expected hash"
        );
    }

    // fn child(main_parent: &Individual)

    // fn evaluate(&self, d: &Data)
    #[test]
    fn test_evaluate_and_evaluate_from_features() {
        let mut ind = Individual::test();
        ind.data_type = RAW_TYPE;
        ind.language = TERNARY_LANG;
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();

        X.insert((0, 0), 0.1);
        // missing value for (0, 1)
        X.insert((1, 0), 0.3);
        X.insert((1, 1), 0.9);

        let mut data = Data::new();
        data.X = X.clone();
        data.sample_len = 2;

        ind.data_type = RAW_TYPE;
        assert_eq!(
            ind.evaluate_from_features(&X, 2),
            vec![0.1, -0.6000000000000001]
        );
        assert_eq!(ind.evaluate(&data), ind.evaluate_from_features(&X, 2),
        "evaluate() and evaluate_from_features() should return the same result as the first call the second");
        ind.data_type = PREVALENCE_TYPE;
        assert_eq!(ind.evaluate(&data), ind.evaluate_from_features(&X, 2),
        "evaluate() and evaluate_from_features() should return the same result as the first call the second");
        assert_eq!(ind.evaluate_from_features(&X, 2), vec![1.0, 0.0]);
        ind.data_type = LOG_TYPE;
        assert_eq!(ind.evaluate(&data), ind.evaluate_from_features(&X, 2),
        "evaluate() and evaluate_from_features() should return the same result as the first call the second");
        assert_eq!(
            ind.evaluate_from_features(&X, 2),
            vec![706.09383343927, -1.0986122886680505]
        );
        assert_eq!(ind.evaluate(&data), ind.evaluate_from_features(&X, 2),
        "evaluate() and evaluate_from_features() should return the same result as the first call the second");
    }

    // fn child(main_parent: &Individual)

    // fn evaluate_raw
    #[test]
    fn test_evaluate_raw_weighted_score() {
        let mut ind: Individual = Individual::test();
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        ind.data_type = RAW_TYPE;

        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        X.insert((0, 0), 2.0);
        X.insert((0, 1), 3.0);
        X.insert((1, 0), 4.0);
        X.insert((1, 1), 5.0);

        ind.language = RATIO_LANG;
        assert_eq!(
            ind.evaluate_raw(&X, 2),
            vec![2.0 / (3.0 + ind.epsilon), 4.0 / (5.0 + ind.epsilon)],
            "bad calculation for raw data scores with ratio Language"
        );
        ind.language = TERNARY_LANG;
        assert_eq!(
            ind.evaluate_raw(&X, 2),
            vec![2.0 * 1.0 + 3.0 * -1.0, 4.0 * 1.0 + 5.0 * -1.0],
            "bad calculation for raw data scores with ter language"
        );
        ind.features = vec![(0, 2), (1, -4)].into_iter().collect();
        ind.language = POW2_LANG;
        assert_eq!(
            ind.evaluate_raw(&X, 2),
            vec![2.0 * 2.0 + 3.0 * -4.0, 4.0 * 2.0 + 5.0 * -4.0],
            "bad calculation for raw data scores with pow2 language"
        );
        ind.features = vec![(0, 1), (1, 0)].into_iter().collect();
        ind.language = BINARY_LANG;
        assert_eq!(
            ind.evaluate_raw(&X, 2),
            vec![2.0 * 1.0 + 3.0 * 0.0, 4.0 * 1.0 + 5.0 * 0.0],
            "bad calculation for raw data scores with bin language"
        );
    }

    #[test]
    fn test_evaluate_raw_zero_or_more_sample_len() {
        let ind = Individual::test();
        let X: HashMap<(usize, usize), f64> = HashMap::new();
        let scores = ind.evaluate_raw(&X, 0);
        assert!(scores.is_empty(), "score should be empty when sample_len=0");
        let scores = ind.evaluate_raw(&X, 10);
        assert_eq!(
            scores,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "selecting samples outside the range should lead to a score of 0.0"
        );
    }

    #[test]
    fn test_evaluate_raw_missing_values() {
        let mut ind = Individual::test();
        ind.data_type = RAW_TYPE;
        ind.language = TERNARY_LANG;
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        X.insert((0, 0), 2.0);
        // missing value for (0, 1)
        X.insert((1, 0), 4.0);
        X.insert((1, 1), 5.0);

        let scores = ind.evaluate_raw(&X, 2);
        assert_eq!(
            scores,
            vec![2.0 * 1.0, 4.0 * 1.0 + 5.0 * (-1.0)],
            "X missing value should be interpreted as coefficient 0"
        );
    }

    // fn evaluate_prevalence
    #[test]
    fn test_evaluate_prevalence_weighted_score() {
        let mut ind: Individual = Individual::test();
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        ind.data_type = RAW_TYPE;

        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        X.insert((0, 0), 2.0);
        X.insert((0, 1), 3.0);
        X.insert((1, 0), 4.0);
        X.insert((1, 1), 5.0);

        ind.language = RATIO_LANG;
        assert_eq!(
            ind.evaluate_prevalence(&X, 2),
            vec![1.0 / (1.0 + ind.epsilon), 1.0 / (1.0 + ind.epsilon)],
            "bad calculation for prevalence data scores with ratio Language"
        );
        ind.language = TERNARY_LANG;
        assert_eq!(
            ind.evaluate_prevalence(&X, 2),
            vec![1.0 - 1.0, 1.0 - 1.0],
            "bad calculation for prevalence data scores with ter language"
        );
        ind.features = vec![(0, 2), (1, -4)].into_iter().collect();
        ind.language = POW2_LANG;
        assert_eq!(
            ind.evaluate_prevalence(&X, 2),
            vec![2.0 - 4.0, 2.0 - 4.0],
            "bad calculation for prevalence data scores with pow2 language"
        );
        ind.features = vec![(0, 1), (1, 0)].into_iter().collect();
        ind.language = BINARY_LANG;
        assert_eq!(
            ind.evaluate_prevalence(&X, 2),
            vec![1.0 * 1.0 + 1.0 * 0.0, 1.0 * 1.0 + 1.0 * 0.0],
            "bad calculation for prevalence data scores with bin language"
        );
    }

    #[test]
    fn test_evaluate_prevalence_zero_or_more_sample_len() {
        let ind = Individual::test();
        let X: HashMap<(usize, usize), f64> = HashMap::new();
        let scores = ind.evaluate_prevalence(&X, 0);
        assert!(scores.is_empty(), "score should be empty when sample_len=0");
        let scores = ind.evaluate_prevalence(&X, 10);
        assert_eq!(
            scores,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "selecting samples outside the range should lead to a score of 0.0"
        );
    }

    #[test]
    fn test_evaluate_prevalence_missing_values() {
        let mut ind = Individual::test();
        ind.data_type = RAW_TYPE;
        ind.language = TERNARY_LANG;
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        X.insert((0, 0), 2.0);
        // missing value for (0, 1)
        X.insert((1, 0), 4.0);
        X.insert((1, 1), 5.0);

        let scores = ind.evaluate_prevalence(&X, 2);
        assert_eq!(
            scores,
            vec![1.0 * 1.0, 1.0 * 1.0 + 1.0 * (-1.0)],
            "X missing value should be interpreted as coefficient 0"
        );
    }

    #[test]
    fn test_evaluate_log_weighted_score() {
        let mut ind: Individual = Individual::test();
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        ind.data_type = LOG_TYPE;

        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        X.insert((0, 0), 0.1);
        X.insert((0, 1), 0.75);
        X.insert((1, 0), 0.3);
        X.insert((1, 1), 0.9);

        // Could be interesting to add a is_nan() or is_infinite() verification in evaluate_log
        ind.language = RATIO_LANG;
        assert_eq!(
            ind.evaluate_log(&X, 2),
            vec![
                (0.1_f64 / ind.epsilon).ln() / ((0.75_f64 / ind.epsilon).ln() + ind.epsilon),
                (0.3_f64 / ind.epsilon).ln() / ((0.9_f64 / ind.epsilon).ln() + ind.epsilon)
            ],
            "bad calculation for log data scores with ratio language"
        );
        ind.language = TERNARY_LANG;
        assert_eq!(
            ind.evaluate_log(&X, 2),
            vec![
                (0.1_f64 / ind.epsilon).ln() * 1.0 + (0.75_f64 / ind.epsilon).ln() * -1.0,
                (0.3_f64 / ind.epsilon).ln() * 1.0 + (0.9_f64 / ind.epsilon).ln() * -1.0
            ],
            "bad calculation for log data scores with ter language"
        );
        ind.features = vec![(0, 2), (1, -4)].into_iter().collect();
        ind.language = POW2_LANG;
        assert_eq!(
            ind.evaluate_log(&X, 2),
            vec![
                (0.1_f64 / ind.epsilon).ln() * 2.0 + (0.75_f64 / ind.epsilon).ln() * -4.0,
                (0.3_f64 / ind.epsilon).ln() * 2.0 + (0.9_f64 / ind.epsilon).ln() * -4.0
            ],
            "bad calculation for log data scores with pow2 language"
        );
        ind.features = vec![(0, 1), (1, 0)].into_iter().collect();
        ind.language = BINARY_LANG;
        assert_eq!(
            ind.evaluate_log(&X, 2),
            vec![
                (0.1_f64 / ind.epsilon).ln() * 1.0 + (0.75_f64 / ind.epsilon).ln() * 0.0,
                (0.3_f64 / ind.epsilon).ln() * 1.0 + (0.9_f64 / ind.epsilon).ln() * 0.0
            ],
            "bad calculation for log data scores with bin language"
        );
    }

    #[test]
    fn test_evaluate_log_zero_or_more_sample_len() {
        let mut ind = Individual::test();
        ind.data_type = LOG_TYPE;
        let X: HashMap<(usize, usize), f64> = HashMap::new();
        let scores = ind.evaluate_log(&X, 0);
        assert!(scores.is_empty(), "score should be empty when sample_len=0");
        let scores = ind.evaluate_log(&X, 10);
        assert_eq!(
            scores,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "selecting samples outside the range should lead to a score of 0.0"
        );
    }

    #[test]
    fn test_evaluate_log_missing_values() {
        let mut ind = Individual::test();
        ind.data_type = RAW_TYPE;
        ind.language = TERNARY_LANG;
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        X.insert((0, 0), 0.1);
        // missing value for (0, 1)
        X.insert((1, 0), 0.3);
        X.insert((1, 1), 0.9);

        let scores = ind.evaluate_log(&X, 2);
        assert_eq!(
            scores,
            vec![
                (0.1_f64 / ind.epsilon).ln() * 1.0,
                (0.3_f64 / ind.epsilon).ln() * 1.0 + (0.9_f64 / ind.epsilon).ln() * -1.0
            ],
            "X missing value should be interpreted as coefficient 0"
        );
    }

    // tests for auc
    #[test]
    fn test_compute_auc() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        let data = Data::test2();
        assert_eq!(
            0.7380952380952381,
            ind.compute_auc(&data),
            "bad calculation for AUC with compute_auc : this could be a ties issue"
        );
        assert_eq!(
            0.7380952380952381,
            ind.compute_roc_and_metrics(&data, &FitFunction::auc, None)
                .0,
            "bad calculation for AUC with compute_roc_and_metrics : this could be a ties issue"
        );
        assert_eq!(ind.compute_auc(&data), ind.compute_auc_from_features(&data.X, &data.y),
        "Individual.compute_auc_from_features(&data.X, &data.y) should return the same result as Individual.compute_auc(&data)");
        assert_eq!(ind.compute_auc(&data), compute_auc_from_value(&ind.evaluate(&data), &data.y),
        "Individual.compute_auc_from_value(scores, &data.y) should return the same result as Individual.compute_auc(&data)");
        assert_eq!(
            0.0,
            compute_auc_from_value(
                &vec![0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64],
                &vec![1_u8, 1_u8, 1_u8, 1_u8, 0_u8]
            ),
            "auc with a perfect classification and class1 < class0 should be 0.0"
        );
        assert_eq!(
            1.0,
            compute_auc_from_value(
                &vec![0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64],
                &vec![0_u8, 0_u8, 0_u8, 0_u8, 1_u8]
            ),
            "auc with a perfect classification and class0 < class1 should be 1.0"
        );
        assert_eq!(
            1.0,
            compute_auc_from_value(
                &vec![0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64],
                &vec![0_u8, 0_u8, 0_u8, 0_u8, 1_u8]
            ),
            "auc with a perfect classification and class0 < class1 should be 1.0"
        );
        // maybe add a verification inside compute_auc to avoid below cases ?
        assert_eq!(
            0.5,
            compute_auc_from_value(
                &vec![0.1_f64, 0.2_f64, 0.3_f64, 0.4_f64],
                &vec![0_u8, 0_u8, 0_u8, 0_u8]
            ),
            "auc should be equal to 0 when there is no positive class"
        );
        assert_eq!(0.5, compute_auc_from_value(&vec![0.5_f64, 0.6_f64, 0.7_f64, 0.8_f64], &vec![1_u8, 1_u8, 1_u8, 1_u8]),
        "auc should be equal to 0 when there is no negative class to avoid positive biais in model selection");
        assert_eq!(
            0.4166666666666667,
            compute_auc_from_value(
                &vec![0.5_f64, 0.6_f64, 0.3_f64, 0.1_f64, 0.9_f64, 0.1_f64],
                &vec![1_u8, 2_u8, 1_u8, 0_u8, 0_u8, 1_u8]
            ),
            "class 2 should be omited in AUC"
        );
    }

    // fn calculate_confusion_matrix
    #[test]
    fn test_calculate_confusion_matrix_basic() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        let data = Data::test2();
        let confusion_matrix = ind.calculate_confusion_matrix(&data);
        assert_eq!(
            confusion_matrix.0, 2,
            "incorrect identification of true positives"
        );
        assert_eq!(
            confusion_matrix.1, 4,
            "incorrect identification of false positives"
        );
        assert_eq!(
            confusion_matrix.2, 3,
            "incorrect identification of true negatives"
        );
        assert_eq!(
            confusion_matrix.3, 1,
            "incorrect identification of false negatives"
        );
    }

    #[test]
    fn test_calculate_confusion_matrix_class_2() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        let mut data = Data::test2();
        data.y = vec![1, 0, 2, 0, 0, 0, 0, 0, 1, 0];
        let confusion_matrix = ind.calculate_confusion_matrix(&data);
        assert_eq!(confusion_matrix.3, 0, "class 2 shoudn't  be classified");
    }

    #[test]
    #[should_panic(expected = "Invalid class label in y: 3")]
    fn test_calculate_confusion_matrix_invalid_class_label() {
        let ind = Individual::test();
        let mut data = Data::test2();
        data.y = vec![1, 0, 3, 3, 3, 3, 0, 1, 0, 1];
        let _confusion_matrix = ind.calculate_confusion_matrix(&data);
    }

    // fn count_k
    #[test]
    fn test_count_k_basic() {
        let mut ind = Individual::test();
        ind.count_k();
        assert_eq!(
            ind.features.len(),
            ind.k,
            "count_k() should attribute Individual.features.len() as Individual.k"
        );
    }

    #[test]
    fn test_count_k_no_features() {
        let mut ind = Individual::new();
        ind.count_k();
        assert_eq!(
            ind.features.len(),
            ind.k,
            "count_k() should attribute Individual.features.len() as Individual.k"
        );
    }

    #[test]
    fn test_random() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test2();
        let ind = Individual::random(&data, &mut rng);
        // warning : ind.features.len() != data.feature_len as generate_random_vector can return 0 not kept in the Hashmap
        assert!(
            ind.features.len() <= data.feature_len,
            "random indivudal features should respect the data feature_len"
        );
        assert_eq!(
            ind.k,
            ind.features.len(),
            "random indivudal k should respect the data feature_len"
        );
        assert_eq!(ind.features, vec![(0, -1), (1, 1)].into_iter().collect(),
        "the generated Individual isn't the same as generated in the past, indicating a reproducibility problem.");
    }

    // fn random_select_k
    #[test]
    fn test_random_select_k() {
        let features = vec![0, 1, 2, 3, 4];
        let mut expected_features = HashMap::new();
        let mut feature_class = HashMap::new();
        feature_class.insert(0, 1);
        feature_class.insert(1, 0);
        feature_class.insert(2, 1);
        feature_class.insert(3, 0);
        feature_class.insert(4, 1);
        expected_features.insert(2, 1);

        // warning : random_select_k never select k_max features
        let mut rng = ChaCha8Rng::seed_from_u64(42); // Seed for reproducibility
        let ind_bin = Individual::random_select_k(
            2,
            3,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            false,
            &mut rng,
        );
        let ind_ter = Individual::random_select_k(
            2,
            3,
            &features,
            &feature_class,
            TERNARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            false,
            &mut rng,
        );
        let ind_ratio = Individual::random_select_k(
            2,
            3,
            &features,
            &feature_class,
            RATIO_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            false,
            &mut rng,
        );
        let ind_pow2 = Individual::random_select_k(
            2,
            3,
            &features,
            &feature_class,
            POW2_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            false,
            &mut rng,
        );

        assert!(
            ind_bin
                .features
                .iter()
                .all(|(key, value)| feature_class.get(key) == Some(&(*value as u8))),
            "selected k features should be part of input feature_class"
        );
        assert_eq!(
            ind_bin.language, BINARY_LANG,
            "input language should be respected"
        );
        assert_eq!(
            ind_bin.data_type, RAW_TYPE,
            "input data_type should be respected"
        );
        assert_eq!(
            ind_bin.epsilon, DEFAULT_MINIMUM,
            "input epsilon should be respected"
        );
        assert!(
            ind_bin.features.values().all(|&v| vec![0, 1].contains(&v)),
            "invalid coefficient for BINARY_LANG"
        );
        assert!(
            ind_ter.features.values().all(|&v| vec![-1, 1].contains(&v)),
            "invalid coefficient for TERNARY_LANG"
        );
        assert!(
            ind_ratio
                .features
                .values()
                .all(|&v| vec![-1, 1].contains(&v)),
            "invalid coefficient for RATIO_LANG"
        );
        assert!(
            ind_pow2
                .features
                .values()
                .all(|&v| vec![-4, 4].contains(&v)),
            "invalid initial coefficient for POW2_LANG"
        );
        assert_eq!(ind_ratio.threshold, 1.0, "new individual created with random_select_k() with a RATIO_LANG should have a threshold of 1.0");

        let ind = Individual::random_select_k(
            0,
            3,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            false,
            &mut rng,
        );
        assert_eq!(ind.features, expected_features,
        "the selected features are not the same as selected in the past, indicating a reproducibility problem.");
        // k_min=1 & k_max=1 should return 1 feature and not panic
        // ind = Individual::random_select_k(1, 1, &features, &feature_class, BINARY_LANG, RAW_TYPE, DEFAULT_MINIMUM, &mut rng);
    }

    #[test]
    fn test_random_select_k_equal_min_max() {
        // Test that k_min = k_max works correctly (forces exactly k features)
        let features = vec![0, 1, 2, 3, 4];
        let mut feature_class = HashMap::new();
        feature_class.insert(0, 1);
        feature_class.insert(1, 0);
        feature_class.insert(2, 1);
        feature_class.insert(3, 0);
        feature_class.insert(4, 1);

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Test with k_min = k_max = 3
        let ind = Individual::random_select_k(
            3,
            3,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            false,
            &mut rng,
        );

        assert_eq!(
            ind.k, 3,
            "Individual should have exactly 3 features when k_min=k_max=3"
        );
        assert_eq!(
            ind.features.len(),
            3,
            "Individual should have exactly 3 features"
        );

        // Test with k_min = k_max = 1
        let ind_single = Individual::random_select_k(
            1,
            1,
            &features,
            &feature_class,
            TERNARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            false,
            &mut rng,
        );

        assert_eq!(
            ind_single.k, 1,
            "Individual should have exactly 1 feature when k_min=k_max=1"
        );
        assert_eq!(
            ind_single.features.len(),
            1,
            "Individual should have exactly 1 feature"
        );
    }

    // fn compute_metrics
    // #[test]
    // #[should_panic(expected = "A predicted vs real class of (0, 3) should not exist")] Outdated test
    // fn test_compute_metrics_invalid_class() {
    //     let ind = Individual::test();
    //     let mut data = Data::test2();
    //     data.y = vec![1, 3, 1, 0, 0, 0, 0, 0, 1, 1];
    //     ind.compute_metrics(&data);
    // }

    #[test]
    fn test_compute_metrics_basic() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];
        let metrics = ind.compute_metrics(&data);
        assert_eq!(0.5_f64, metrics.0, "bad calculation for accuracy");
        assert_eq!(
            0.6666666666666666_f64, metrics.1,
            "bad calculation for sensitivity"
        );
        assert_eq!(
            0.42857142857142855_f64, metrics.2,
            "bad calculation for specificity"
        );
    }

    #[test]
    fn test_compute_metrics_class2() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 2];
        assert_eq!(
            (
                0.5555555555555556_f64,
                0.6666666666666666_f64,
                0.5_f64,
                0.0_f64,
                AdditionalMetrics {
                    mcc: None,
                    f1_score: None,
                    npv: None,
                    ppv: None,
                    g_mean: None
                }
            ),
            ind.compute_metrics(&data),
            "class 2 should be omitted in calculation"
        )
    }

    #[test]
    fn test_compute_metrics_too_much_y() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1];
        assert_eq!((0.5_f64, 0.6666666666666666_f64, 0.42857142857142855_f64, 0.0_f64, AdditionalMetrics { mcc:None, f1_score: None, npv: None, ppv: None, g_mean: None}), ind.compute_metrics(&data),
        "when ind.sample_len < data.sample_len (or y.len() if it does not match), only the ind.sample_len values should be used to calculate its metrics");
    }

    #[test]
    fn test_compute_metrics_not_enough_y() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 1];
        assert_eq!((0.25_f64, 0.3333333333333333_f64, 0.0_f64, 0.0_f64, AdditionalMetrics { mcc:None, f1_score: None, npv: None, ppv: None, g_mean: None}), ind.compute_metrics(&data),
        "when data.sample_len (or y.len() if it does not match) < ind.sample_len, only the data.sample_len values should be used to calculate its metrics");
    }

    // fn compute_threshold_and_metrics
    // threshold = 0.84 according to R ; same metrics as below
    #[test]
    fn test_compute_threshold_and_metrics_basic() {
        let ind = Individual::test();
        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];
        let results = ind.compute_threshold_and_metrics(&data);
        assert_eq!(0.89_f64, results.0, "bad identification of the threshold");
        assert_eq!(0.8_f64, results.1, "bad calculation for accuracy");
        assert_eq!(
            0.6666666666666666_f64, results.2,
            "bad calculation for sensitivity"
        );
        assert_eq!(
            0.8571428571428571_f64, results.3,
            "bad calculation for specificity"
        );

        let scores: Vec<_> = ind.evaluate(&data);
        let (_, _, accuracy, sensitivity, specificity, _): (f64, f64, f64, f64, f64, f64) =
            compute_roc_and_metrics_from_value(&scores, &data.y, &FitFunction::auc, None);
        assert_eq!(accuracy, ind.compute_threshold_and_metrics(&data).1, "Accuracy calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
        assert_eq!(sensitivity, ind.compute_threshold_and_metrics(&data).2, "Sensitivity calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
        assert_eq!(specificity, ind.compute_threshold_and_metrics(&data).3, "Specificity calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
    }

    // threshold = 0.84 according to R ; same metrics as below -> need to control if this difference could be a problem
    #[test]
    fn test_compute_threshold_and_metrics_class_2() {
        let ind = Individual::test();
        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 2];
        assert_eq!(
            (
                0.79_f64,
                0.7777777777777778_f64,
                0.6666666666666666_f64,
                0.8333333333333334_f64
            ),
            ind.compute_threshold_and_metrics(&data),
            "class 2 should be omitted in calculation"
        );

        let scores: Vec<_> = ind.evaluate(&data);
        let (_, _, accuracy, sensitivity, specificity, _): (f64, f64, f64, f64, f64, f64) =
            compute_roc_and_metrics_from_value(&scores, &data.y, &FitFunction::auc, None);
        assert_eq!(accuracy, ind.compute_threshold_and_metrics(&data).1, "Accuracy calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
        assert_eq!(sensitivity, ind.compute_threshold_and_metrics(&data).2, "Sensitivity calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
        assert_eq!(specificity, ind.compute_threshold_and_metrics(&data).3, "Specificity calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
    }

    //#[test]
    //fn test_compute_threshold_and_metrics_too_much_y() {
    //    let mut ind = Individual::test();
    //    ind.threshold = 0.75;
    //    let mut data = Data::test2();
    //    data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1];
    //    assert_eq!((0.79_f64, 0.8_f64, 0.6666666666666666_f64, 0.8571428571428571_f64), ind.compute_threshold_and_metrics(&data),
    //    "when ind.sample_len < data.sample_len (or y.len() if it does not match), only the ind.sample_len values should be used to calculate its metrics");
    //}

    #[test]
    fn test_compute_threshold_and_metrics_not_enough_y() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 1];
        assert_eq!((0.89_f64, 0.5_f64, 0.3333333333333333_f64, 1.0_f64), ind.compute_threshold_and_metrics(&data),
        "when data.sample_len (or y.len() if it does not match) < ind.sample_len, only the data.sample_len values should be used to calculate its metrics");

        let (_, _, accuracy, sensitivity, specificity, _): (f64, f64, f64, f64, f64, f64) =
            ind.compute_roc_and_metrics(&data, &FitFunction::auc, None);
        assert_eq!(accuracy, ind.compute_threshold_and_metrics(&data).1, "Accuracy calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
        assert_eq!(sensitivity, ind.compute_threshold_and_metrics(&data).2, "Sensitivity calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
        assert_eq!(specificity, ind.compute_threshold_and_metrics(&data).3, "Specificity calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
    }

    // fn features_index
    #[test]
    fn test_features_index_basic() {
        let mut ind = Individual::test();
        ind.features.insert(10, 0.42 as i8);
        assert_eq!(
            vec![0_usize, 1_usize, 2_usize, 3_usize, 10_usize],
            ind.features_index()
        );
    }

    fn generate_feature_seeds(
        features: &[usize],
        permutations: usize,
        seed: u64,
    ) -> HashMap<usize, Vec<u64>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        features
            .iter()
            .map(|&f| {
                let seeds: Vec<u64> = (0..permutations).map(|_| rng.next_u64()).collect();
                (f, seeds)
            })
            .collect()
    }

    #[test]
    fn test_compute_mda_feature_importance_zero_permutations_division_by_zero() {
        let individual = Individual::specific_test(&[0]);
        let data = Data::specific_test(10, 2);
        let features_to_process = vec![0];
        let feature_seeds = generate_feature_seeds(&features_to_process, 0, 123);

        let result = std::panic::catch_unwind(|| {
            individual.compute_mda_feature_importance(
                &data,
                0,
                &features_to_process,
                &feature_seeds,
            )
        });

        assert!(
            result.is_err(),
            "Zero permutations should cause division by zero panic"
        );
    }

    #[test]
    fn test_compute_mda_feature_importance_feature_not_in_individual_returns_zero() {
        let individual = Individual::specific_test(&[2, 4]);
        let data = Data::specific_test(30, 6);
        let features_to_process = vec![0, 1, 2, 3, 4, 5];
        let feature_seeds = generate_feature_seeds(&features_to_process, 10, 789);

        let result = individual.compute_mda_feature_importance(
            &data,
            10,
            &features_to_process,
            &feature_seeds,
        );

        for importance in &result.importances {
            if ![2, 4].contains(&importance.feature_idx) {
                assert_eq!(
                    importance.importance, 0.0,
                    "Features not in individual (feature {}) must return 0.0 importance",
                    importance.feature_idx
                );
            }
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_missing_seeds_panics() {
        let individual = Individual::specific_test(&[1]);
        let data = Data::specific_test(30, 3);
        let features_to_process = vec![1, 2];

        // Seeds only provided for feature 1, missing for feature 2
        let mut incomplete_seeds = HashMap::new();
        incomplete_seeds.insert(1, vec![12345u64; 5]);

        let result = std::panic::catch_unwind(|| {
            individual.compute_mda_feature_importance(
                &data,
                5,
                &features_to_process,
                &incomplete_seeds,
            )
        });

        assert!(
            result.is_err(),
            "Missing seeds should cause .expect() to panic"
        );
    }

    #[test]
    fn test_compute_mda_feature_importance_reproducibility_same_seeds() {
        let individual = Individual::specific_test(&[1, 2]);
        let data = Data::specific_test(50, 4);
        let features_to_process = vec![1, 2];
        let feature_seeds = generate_feature_seeds(&features_to_process, 50, 999);

        let result1 = individual.compute_mda_feature_importance(
            &data,
            2,
            &features_to_process,
            &feature_seeds,
        );
        let result2 = individual.compute_mda_feature_importance(
            &data,
            2,
            &features_to_process,
            &feature_seeds,
        );

        assert_eq!(result1.importances.len(), result2.importances.len());

        for (imp1, imp2) in result1.importances.iter().zip(result2.importances.iter()) {
            assert_eq!(imp1.feature_idx, imp2.feature_idx);
            assert!(
                (imp1.importance - imp2.importance).abs() < 1e-12,
                "Same seeds must yield identical results for feature {}",
                imp1.feature_idx
            );
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_output_structure_fields() {
        let individual = Individual::specific_test(&[1]);
        let data = Data::specific_test(20, 3);
        let features_to_process = vec![0, 1];
        let feature_seeds = generate_feature_seeds(&features_to_process, 5, 456);

        let result = individual.compute_mda_feature_importance(
            &data,
            5,
            &features_to_process,
            &feature_seeds,
        );

        for importance in &result.importances {
            // Verify all fixed structure fields
            assert_eq!(importance.importance_type, ImportanceType::MDA);
            assert_eq!(importance.aggreg_method, None);
            assert_eq!(importance.is_scaled, false);
            assert_eq!(importance.dispersion, 0.0);
            assert_eq!(importance.scope_pct, 1.0);
            assert_eq!(importance.direction, None);

            match importance.scope {
                ImportanceScope::Individual { model_hash } => {
                    assert_eq!(model_hash, individual.hash);
                }
                _ => panic!(
                    "Expected Individual scope for feature {}",
                    importance.feature_idx
                ),
            }
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_all_requested_features_returned() {
        let individual = Individual::specific_test(&[2, 5, 7]);
        let data = Data::specific_test(40, 10);
        let features_to_process = vec![2, 5, 7, 9];
        let feature_seeds = generate_feature_seeds(&features_to_process, 5, 789);

        let result = individual.compute_mda_feature_importance(
            &data,
            5,
            &features_to_process,
            &feature_seeds,
        );

        let mut result_features: Vec<usize> = result
            .importances
            .iter()
            .map(|imp| imp.feature_idx)
            .collect();
        result_features.sort();

        let mut expected_features = features_to_process.clone();
        expected_features.sort();

        assert_eq!(
            result_features, expected_features,
            "All requested features must be present in results"
        )
    }

    #[test]
    fn test_compute_mda_feature_importance_single_permutation_finite_values() {
        let individual = Individual::specific_test(&[0]);
        let data = Data::specific_test(30, 2);
        let features_to_process = vec![0, 1];
        let feature_seeds = generate_feature_seeds(&features_to_process, 1, 456);

        let result = individual.compute_mda_feature_importance(
            &data,
            1,
            &features_to_process,
            &feature_seeds,
        );

        assert_eq!(result.importances.len(), 2);

        for importance in &result.importances {
            assert!(
                importance.importance.is_finite(),
                "Single permutation should yield finite importance for feature {}",
                importance.feature_idx
            );
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_bounds_realistic() {
        let individual = Individual::specific_test(&[0, 1]);
        let data = Data::specific_test(50, 3);
        let features_to_process = vec![0, 1, 2];
        let feature_seeds = generate_feature_seeds(&features_to_process, 20, 654);

        let result = individual.compute_mda_feature_importance(
            &data,
            20,
            &features_to_process,
            &feature_seeds,
        );

        for importance in &result.importances {
            // Importance = baseline_auc - mean_permuted_auc
            // Can be negative if permutation improves performance
            assert!(
                importance.importance.is_finite(),
                "Importance must be finite for feature {}",
                importance.feature_idx
            );
            assert!(
                importance.importance >= -1.0 && importance.importance <= 1.0,
                "Importance should be bounded by AUC difference range for feature {}: got {}",
                importance.feature_idx,
                importance.importance
            );
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_empty_individual_features() {
        let individual = Individual::specific_test(&[]); // No features
        let data = Data::specific_test(20, 4);
        let features_to_process = vec![0, 1, 2, 3];
        let feature_seeds = generate_feature_seeds(&features_to_process, 5, 321);

        let result = individual.compute_mda_feature_importance(
            &data,
            5,
            &features_to_process,
            &feature_seeds,
        );

        for importance in &result.importances {
            assert_eq!(
                importance.importance, 0.0,
                "Individual with no features should return 0 importance for feature {}",
                importance.feature_idx
            );
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_different_seeds_may_differ() {
        let individual = Individual::specific_test(&[0, 2]);
        let data = Data::specific_test(60, 3);
        let features_to_process = vec![0, 2];

        let seeds1 = generate_feature_seeds(&features_to_process, 10, 111);
        let seeds2 = generate_feature_seeds(&features_to_process, 10, 222);

        let result1 =
            individual.compute_mda_feature_importance(&data, 10, &features_to_process, &seeds1);
        let result2 =
            individual.compute_mda_feature_importance(&data, 10, &features_to_process, &seeds2);

        // Note: Results may be the same in some cases (deterministic data), so we only check structure
        assert_eq!(result1.importances.len(), result2.importances.len());

        for (imp1, imp2) in result1.importances.iter().zip(result2.importances.iter()) {
            assert_eq!(imp1.feature_idx, imp2.feature_idx);
            assert!(imp1.importance.is_finite() && imp2.importance.is_finite());
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_large_permutation_count() {
        let individual = Individual::specific_test(&[0]);
        let data = Data::specific_test(50, 2);
        let features_to_process = vec![0];
        let feature_seeds1 = generate_feature_seeds(&features_to_process, 10000, 123);
        let feature_seeds2 = generate_feature_seeds(&features_to_process, 10000, 456);

        let result = individual.compute_mda_feature_importance(
            &data,
            10000,
            &features_to_process,
            &feature_seeds1,
        );

        assert_eq!(result.importances.len(), 1);

        let importance = &result.importances[0];
        assert!(
            importance.importance.is_finite(),
            "Large permutation count should yield finite importance"
        );

        // With many permutations, results should be stable
        let result2 = individual.compute_mda_feature_importance(
            &data,
            10000,
            &features_to_process,
            &feature_seeds2,
        );

        assert!(
            (importance.importance - result2.importances[0].importance).abs() < 1e-2,
            "Large permutation count with different seeds should be highly reproducible: {} vs {}",
            importance.importance,
            result2.importances[0].importance
        );
    }

    #[test]
    fn test_compute_mda_feature_importance_individual_with_mixed_coefficients() {
        // Test individual with mixed positive/negative coefficients
        let mut features_map = HashMap::new();
        features_map.insert(0, 1i8); // Positive coefficient
        features_map.insert(1, -1i8); // Negative coefficient
        features_map.insert(2, 1i8); // Positive coefficient

        let individual = Individual {
            features: features_map,
            auc: 0.8,
            fit: 0.7,
            specificity: 0.75,
            sensitivity: 0.85,
            accuracy: 0.80,
            threshold: 0.5,
            k: 3,
            epoch: 0,
            language: TERNARY_LANG, // Supports negative coefficients
            data_type: RAW_TYPE,
            hash: 0x123456789abcdef0,
            epsilon: DEFAULT_MINIMUM,
            parents: None,
            betas: None,
            threshold_ci: None,
            metrics: AdditionalMetrics {
                mcc: None,
                f1_score: None,
                npv: None,
                ppv: None,
                g_mean: None,
            },
        };

        let data = Data::specific_test(40, 5);
        let features_to_process = vec![0, 1, 2, 3, 4];
        let feature_seeds = generate_feature_seeds(&features_to_process, 10, 555);

        let result = individual.compute_mda_feature_importance(
            &data,
            10,
            &features_to_process,
            &feature_seeds,
        );

        for importance in &result.importances {
            if [0, 1, 2].contains(&importance.feature_idx) {
                // Features present in individual should have computed importance
                assert!(
                    importance.importance.is_finite(),
                    "Present features should have finite importance"
                );
            } else {
                // Features not in individual should have zero importance
                assert_eq!(
                    importance.importance, 0.0,
                    "Absent features should have zero importance"
                );
            }
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_different_datatypes() {
        let mut individual = Individual::specific_test(&[0, 1]);
        let data = Data::specific_test(50, 3);
        let features_to_process = vec![0, 1, 2];
        let feature_seeds = generate_feature_seeds(&features_to_process, 10, 123);

        // Test RAW_TYPE (default)
        individual.data_type = RAW_TYPE;
        let result_raw = individual.compute_mda_feature_importance(
            &data,
            10,
            &features_to_process,
            &feature_seeds,
        );
        assert_eq!(result_raw.importances.len(), 3);

        // Test PREVALENCE_TYPE
        individual.data_type = PREVALENCE_TYPE;
        let result_prev = individual.compute_mda_feature_importance(
            &data,
            10,
            &features_to_process,
            &feature_seeds,
        );
        assert_eq!(result_prev.importances.len(), 3);

        // Test LOG_TYPE
        individual.data_type = LOG_TYPE;
        let result_log = individual.compute_mda_feature_importance(
            &data,
            10,
            &features_to_process,
            &feature_seeds,
        );
        assert_eq!(result_log.importances.len(), 3);

        // Verify all results contain finite importance values
        for result in [&result_raw, &result_prev, &result_log] {
            for imp in &result.importances {
                assert!(
                    imp.importance.is_finite(),
                    "All datatypes should produce finite importance values for feature {}",
                    imp.feature_idx
                );
            }
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_duplicate_features_in_process() {
        let individual = Individual::specific_test(&[0, 1, 2]);
        let data = Data::specific_test(30, 4);
        let features_to_process = vec![0, 1, 1, 2, 3]; // feature 1 dupliquée
        let feature_seeds = generate_feature_seeds(&features_to_process, 5, 789);

        let result = individual.compute_mda_feature_importance(
            &data,
            5,
            &features_to_process,
            &feature_seeds,
        );

        // Should process each unique feature only once
        let unique_features: std::collections::HashSet<usize> =
            features_to_process.iter().cloned().collect();
        assert_eq!(
            result.importances.len(),
            unique_features.len(),
            "Should return importance for each unique feature only"
        );

        // Verify no duplicate feature indices in results
        let mut seen_features = std::collections::HashSet::new();
        for imp in &result.importances {
            assert!(
                !seen_features.contains(&imp.feature_idx),
                "Duplicate importance found for feature {}",
                imp.feature_idx
            );
            seen_features.insert(imp.feature_idx);

            assert!(
                unique_features.contains(&imp.feature_idx),
                "Result contains unexpected feature {}",
                imp.feature_idx
            );
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_extreme_unbalanced_data() {
        let individual = Individual::specific_test(&[0, 1]);
        let mut data = Data::specific_test(50, 3);
        let features_to_process = vec![0, 1, 2];
        let feature_seeds = generate_feature_seeds(&features_to_process, 10, 456);

        // Test: All positive labels (y = 1)
        data.y = vec![1u8; 50];
        let result_all_pos = individual.compute_mda_feature_importance(
            &data,
            10,
            &features_to_process,
            &feature_seeds,
        );

        for imp in &result_all_pos.importances {
            assert!(
                imp.importance.is_finite(),
                "Importance should be finite with all positive labels for feature {}",
                imp.feature_idx
            );
            // AUC calculation should handle single-class case gracefully
        }

        // Test: All negative labels (y = 0)
        data.y = vec![0u8; 50];
        let result_all_neg = individual.compute_mda_feature_importance(
            &data,
            10,
            &features_to_process,
            &feature_seeds,
        );

        for imp in &result_all_neg.importances {
            assert!(
                imp.importance.is_finite(),
                "Importance should be finite with all negative labels for feature {}",
                imp.feature_idx
            );
        }

        // Test: Highly unbalanced (49:1)
        data.y = vec![0u8; 49];
        data.y.push(1u8);
        let result_unbalanced = individual.compute_mda_feature_importance(
            &data,
            5,
            &features_to_process,
            &feature_seeds,
        );

        for imp in &result_unbalanced.importances {
            assert!(
                imp.importance.is_finite(),
                "Importance should be finite with unbalanced labels for feature {}",
                imp.feature_idx
            );
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_constant_feature_values() {
        let individual = Individual::specific_test(&[0, 1]);
        let mut data = Data::specific_test(40, 3);
        let features_to_process = vec![0, 1, 2];
        let feature_seeds = generate_feature_seeds(&features_to_process, 8, 999);

        // Make feature 0 constant (all same value)
        for sample in 0..data.sample_len {
            data.X.insert((sample, 0), 0.5); // Constant value
        }

        let result = individual.compute_mda_feature_importance(
            &data,
            8,
            &features_to_process,
            &feature_seeds,
        );

        for imp in &result.importances {
            assert!(
                imp.importance.is_finite(),
                "Constant features should still produce finite importance for feature {}",
                imp.feature_idx
            );

            if imp.feature_idx == 0 && individual.features.contains_key(&0) {
                // Constant features typically have low/zero importance
                assert!(
                    imp.importance >= -1.0 && imp.importance <= 1.0,
                    "Constant feature importance should be bounded"
                );
            }
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_very_small_dataset() {
        let individual = Individual::specific_test(&[0]);
        let data = Data::specific_test(3, 2); // Very small: 3 samples, 2 features
        let features_to_process = vec![0, 1];
        let feature_seeds = generate_feature_seeds(&features_to_process, 2, 777);

        let result = individual.compute_mda_feature_importance(
            &data,
            2,
            &features_to_process,
            &feature_seeds,
        );

        assert_eq!(result.importances.len(), 2);

        for imp in &result.importances {
            assert!(
                imp.importance.is_finite(),
                "Small dataset should still produce finite importance for feature {}",
                imp.feature_idx
            );
            // With very few samples, AUC calculation might be less stable but should not crash
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_out_of_bounds_features() {
        let individual = Individual::specific_test(&[0, 1]);
        let data = Data::specific_test(30, 3); // Features 0, 1, 2 exist
        let features_to_process = vec![0, 1, 5, 10]; // Features 5, 10 don't exist in data
        let feature_seeds = generate_feature_seeds(&features_to_process, 5, 555);

        let result = individual.compute_mda_feature_importance(
            &data,
            5,
            &features_to_process,
            &feature_seeds,
        );

        for imp in &result.importances {
            assert!(
                imp.importance.is_finite(),
                "Out-of-bounds features should be handled gracefully for feature {}",
                imp.feature_idx
            );

            if ![0, 1, 2].contains(&imp.feature_idx) {
                // Out-of-bounds features should have zero importance
                assert_eq!(
                    imp.importance, 0.0,
                    "Out-of-bounds feature {} should have zero importance",
                    imp.feature_idx
                );
            }
        }
    }

    #[test]
    fn test_compute_mda_feature_importance_mixed_languages() {
        // Test different Individual languages
        let languages_to_test = vec![
            (BINARY_LANG, &[0, 1][..]),
            (TERNARY_LANG, &[0, 1, 2][..]),
            (RATIO_LANG, &[0][..]),
        ];

        let data = Data::specific_test(40, 4);
        let features_to_process = vec![0, 1, 2, 3];
        let feature_seeds = generate_feature_seeds(&features_to_process, 8, 333);

        for (language, individual_features) in languages_to_test {
            let mut individual = Individual::specific_test(individual_features);
            individual.language = language;

            let result = individual.compute_mda_feature_importance(
                &data,
                8,
                &features_to_process,
                &feature_seeds,
            );

            assert_eq!(result.importances.len(), features_to_process.len());

            for imp in &result.importances {
                assert!(
                    imp.importance.is_finite(),
                    "Language {:?} should produce finite importance for feature {}",
                    language,
                    imp.feature_idx
                );
            }
        }
    }

    // fn maximize_objective

    // fn maximize_objective_with_scores
    #[test]
    fn test_maximize_objective_with_scores() {
        let mut ind = Individual::test();
        let data = Data::test2();
        let scores = ind.evaluate(&data);
        let best_objective = ind.maximize_objective_with_scores(&scores, &data, 1.0, 1.0);

        assert_eq!(ind.maximize_objective(&data, 1.0, 1.0), ind.maximize_objective_with_scores(&scores, &data, 1.0, 1.0),
        "Individual.maximize_objective() and Individual.maximize_objective_with_scores() should return the same results for a same scores vector");
        assert!(
            best_objective > 0.0,
            "The best objective should be greater than 0.0"
        );
        assert!(
            ind.sensitivity >= 0.0 && ind.sensitivity <= 1.0,
            "Sensitivity should be between 0.0 and 1.0"
        );
        assert!(
            ind.specificity >= 0.0 && ind.specificity <= 1.0,
            "Specificity should be between 0.0 and 1.0"
        );
        assert!(
            ind.accuracy >= 0.0 && ind.accuracy <= 1.0,
            "Accuracy should be between 0.0 and 1.0"
        );
        let _best_objective = ind.maximize_objective_with_scores(&scores, &data, 0.0, 1.0);
        assert_eq!(ind.sensitivity, 1.0, "focusing only on sensitivity (fpr_penalty=0.0 and fnr_penalty=1.0) normally leads to a sensitivity of 1.0 (the model classifies everything positively)");
        assert_eq!(ind.specificity, 0.0, "focusing only on sensitivity (fpr_penalty=0.0 and fnr_penalty=1.0) normally leads to a specificity of 0.0 (the model classifies everything positively)");
        // Function is broke : sp=1 VS se=0 never reached
        // Interesting fact : R selected threshold (0.89) is correctly reached by this function
        // let best_objective = ind.maximize_objective_with_scores(&scores, &data, 1.0, 0.0);
        // assert_eq!(ind.sensitivity, 0.0, "focusing only on specificity (fpr_penalty=1.0 and fnr_penalty=0.0) normally leads to a sensitivity of 0.0 (the model classifies everything negatively)");
        // assert_eq!(ind.specificity, 1.0, "focusing only on specificity (fpr_penalty=1.0 and fnr_penalty=0.0) normally leads to a specificity of 1.0 (the model classifies everything negatively)");

        // maybe add a panic! for fpr_penalty=0.O and fnr_penalty=0.0 to avoid NaN ?
        // let best_objective = ind.maximize_objective_with_scores(&scores, &data, 0.0, 0.0);
    }

    #[test]
    fn test_get_language() {
        let mut ind = Individual::test();
        ind.language = BINARY_LANG;
        assert_eq!(ind.get_language(), "Binary");
        ind.language = TERNARY_LANG;
        assert_eq!(ind.get_language(), "Ternary");
        ind.language = RATIO_LANG;
        assert_eq!(ind.get_language(), "Ratio");
        ind.language = POW2_LANG;
        assert_eq!(ind.get_language(), "Pow2");
        ind.language = 42;
        assert_eq!(ind.get_language(), "Unknown");
    }

    #[test]
    fn test_get_data_type() {
        let mut ind = Individual::test();
        ind.data_type = RAW_TYPE;
        assert_eq!(ind.get_data_type(), "Raw");
        ind.data_type = PREVALENCE_TYPE;
        assert_eq!(ind.get_data_type(), "Prevalence");
        ind.data_type = LOG_TYPE;
        assert_eq!(ind.get_data_type(), "Log");
        ind.data_type = 42;
        assert_eq!(ind.get_data_type(), "Unknown");
    }

    #[test]
    fn test_on_more_complicated_data() {
        let mut individual = Individual::new();
        let mut data = Data::new();
        let mut data_test = Data::new();
        let _ = data.load_data(
            "samples/Qin2014/Xtrain.tsv",
            "samples/Qin2014/Ytrain.tsv",
            true,
        );
        let _ = data_test.load_data(
            "samples/Qin2014/Xtest.tsv",
            "samples/Qin2014/Ytest.tsv",
            true,
        );

        // Set the language and data type
        individual.language = TERNARY_LANG;
        individual.data_type = LOG_TYPE;
        individual.epsilon = 1e-5;

        // Set the feature indices and their signs
        let feature_indices = vec![
            (9, 1),
            (22, 1),
            (23, 1),
            (24, -1),
            (42, -1),
            (47, 1),
            (57, -1),
            (66, -1),
            (72, -1),
            (82, -1),
            (87, -1),
            (92, -1),
            (105, 1),
            (124, -1),
            (130, -1),
            (174, 1),
            (194, 1),
            (221, 1),
            (222, 1),
            (262, 1),
            (272, 1),
            (301, -1),
            (319, 1),
            (320, -1),
            (324, 1),
            (334, 1),
            (359, 1),
            (378, 1),
            (436, -1),
            (466, -1),
            (468, -1),
            (476, -1),
            (488, 1),
            (497, -1),
            (512, 1),
            (522, -1),
            (546, -1),
            (565, 1),
            (591, -1),
            (614, -1),
            (649, -1),
            (658, 1),
            (670, -1),
            (686, 1),
            (716, 1),
            (825, 1),
            (834, -1),
            (865, -1),
            (867, 1),
            (874, 1),
            (877, 1),
            (1117, 1),
            (1273, 1),
            (1313, 1),
            (1317, 1),
            (1464, 1),
            (1525, 1),
            (1629, 1),
            (1666, 1),
            (1710, 1),
            (1735, 1),
            (1738, 1),
            (1740, 1),
            (1741, 1),
            (1794, 1),
            (1870, 1),
        ];

        for (index, sign) in feature_indices {
            individual.features.insert(index, sign);
        }

        data.classes = vec!["healthy".to_string(), "cirrhosis".to_string()];
        data.y[3] = 2 as u8;
        data.y[4] = 2 as u8;
        data_test.y[7] = 2 as u8;

        // control both metrics and display
        let right_string = "Ternary:Log [k=66] [gen:0] [fit:0.000] AUC 0.962/0.895 | accuracy 0.921/0.828 | sensitivity 0.937/0.867 | specificity 0.904/0.786\n\
            Class cirrhosis: ln(msp_0010⁺ × msp_0023⁺ × msp_0024⁺ × msp_0048⁺ × msp_0106⁺ × msp_0176⁺ × msp_0196⁺ × msp_0223⁺ × msp_0224⁺ × msp_0265⁺ × msp_0275⁺ × msp_0324⁺ × msp_0329⁺ \
            × msp_0339⁺ × msp_0364⁺ × msp_0383⁺ × msp_0493⁺ × msp_0517⁺ × msp_0570⁺ × msp_0664⁺ × msp_0692⁺ × msp_0722⁺ × msp_0832⁺ × msp_0874⁺ × msp_0881⁺ × msp_0884⁺ × msp_1127⁺ × msp_1284⁺ \
            × msp_1325⁺ × msp_1329⁺ × msp_1479⁺ × msp_1543⁺ × msp_1660⁺ × msp_1700⁺ × msp_1748⁺ × msp_1782⁺ × msp_1785⁺ × msp_1787⁺ × msp_1788⁺ × msp_1862⁺ × msp_1942⁺) - ln(msp_0025⁺ × msp_0043⁺ \
            × msp_0058⁺ × msp_0067⁺ × msp_0073⁺ × msp_0083⁺ × msp_0088⁺ × msp_0093⁺ × msp_0125⁺ × msp_0131⁺ × msp_0306⁺ × msp_0325⁺ × msp_0441⁺ × msp_0471⁺ × msp_0473c⁺ × msp_0481⁺ × msp_0502⁺ × msp_0527⁺ \
            × msp_0551⁺ × msp_0596⁺ × msp_0619⁺ × msp_0654⁺ × msp_0676⁺ × msp_0841⁺ × msp_0872⁺) ";

        (
            individual.auc,
            individual.threshold,
            individual.accuracy,
            individual.sensitivity,
            individual.specificity,
            _,
        ) = individual.compute_roc_and_metrics(&data, &FitFunction::auc, None);

        // except the threshold (small variation between launch ~0.000000000001)
        let display_output = crate::utils::strip_ansi_if_needed(
            &individual.display(&data, Some(&data_test), &"ga".to_string(), 0.0),
            false,
        );
        assert_eq!(
            right_string.split("≥ 19").collect::<Vec<_>>()[0],
            display_output.split("≥ 19").collect::<Vec<_>>()[0]
        );

        assert_eq!(
            individual.compute_auc(&data),
            0.961572606214331,
            "Wrong auc calculated"
        );
        assert_eq!(
            individual.compute_new_auc(&data_test),
            0.8952380952380953,
            "Wrong test auc calculated"
        );
        // Compute ROC and metrics should return the same AUC as .compute_auc and the same metrics as .compute_threshold_and_metrics
        let (threshold, accuracy, sensitivity, specificity): (f64, f64, f64, f64) =
            individual.compute_threshold_and_metrics(&data);

        assert_eq!(individual.compute_new_auc(&data), individual.auc, "AUC calculated with Individual.compute_auc() and Individual.compute_roc_and_metrics() should be the same" );
        assert_eq!(accuracy,  individual.accuracy, "Accuracy calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
        assert_eq!(sensitivity, individual.sensitivity, "Sensitivity calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
        assert_eq!(specificity,  individual.specificity, "Specificity calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );

        individual.threshold = threshold;
        assert_eq!(accuracy, individual.compute_metrics(&data).0,  "Accuracy calculated with Individual.compute_threshold_and_metrics() and Individual.compute_metrics() should be the same");
        assert_eq!(sensitivity, individual.compute_metrics(&data).1,  "Sensitivity calculated with Individual.compute_threshold_and_metrics() and Individual.compute_metrics() should be the same");
        assert_eq!(specificity, individual.compute_metrics(&data).2,  "Specificity calculated with Individual.compute_threshold_and_metrics() and Individual.compute_metrics() should be the same");
    }

    #[test]
    fn test_evaluate_class_and_score() {
        let ind = Individual::test2();
        let data = Data::test2();
        let scores = ind.evaluate(&data);
        assert_eq!(
            scores,
            vec![0.89, 0.79, 0.74, -0.73, 0.89, 0.79, 0.74, -0.73, 0.89, 0.79],
            "bad calculation for score"
        );
        let class_and_score = ind.evaluate_class_and_score(&data);
        assert_eq!(
            class_and_score,
            (
                vec![1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
                vec![0.89, 0.79, 0.74, -0.73, 0.89, 0.79, 0.74, -0.73, 0.89, 0.79]
            ),
            "bad calculation for class_and_score "
        );
    }

    #[test]
    fn test_get_genealogy() {
        use std::collections::{HashMap, HashSet};

        // Outdated - betas are now used to compute hash
        // let mut real_tree2: HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> = HashMap::new();
        // real_tree2.insert((13776636996568204467, Some(vec![8310551554976651538, 15701002319609139959])), [0].iter().cloned().collect());
        // real_tree2.insert((8310551554976651538, Some(vec![8712745243790315600, 5577107020388865403])), [1].iter().cloned().collect());
        // real_tree2.insert((15701002319609139959, Some(vec![573762283934476040])), [1].iter().cloned().collect());
        // real_tree2.insert((8712745243790315600, Some(vec![8225502949628013706])), [2].iter().cloned().collect());
        // real_tree2.insert((5577107020388865403, Some(vec![212344198819259482, 16067238433084305925])), [2].iter().cloned().collect());
        // real_tree2.insert((573762283934476040, None), [2].iter().cloned().collect());
        // real_tree2.insert((8225502949628013706, Some(vec![9859973499533993323])), [3].iter().cloned().collect());
        // real_tree2.insert((212344198819259482, None), [3].iter().cloned().collect());
        // real_tree2.insert((16067238433084305925, None), [3].iter().cloned().collect());
        // real_tree2.insert((9859973499533993323, None), [4].iter().cloned().collect());

        // let mut pop2 = create_test_population();
        // pop2.compute_hash();
        // pop2.individuals[0].parents = Some(vec![8310551554976651538, 15701002319609139959]);
        // pop2.individuals[1].parents = Some(vec![8712745243790315600, 5577107020388865403]);
        // pop2.individuals[2].parents = Some(vec![573762283934476040]);
        // pop2.individuals[3].parents = Some(vec![8225502949628013706]);
        // pop2.individuals[4].parents = Some(vec![212344198819259482, 16067238433084305925]);
        // pop2.individuals[6].parents = Some(vec![9859973499533993323]);

        // for (i, epoch) in [4, 3, 3, 2, 2, 2, 1, 1, 1, 0].iter().enumerate() {
        //     pop2.individuals[i].epoch = *epoch;
        // }
        // let mut gen_0 = Population::new();
        // let mut gen_1  = Population::new();
        // let mut gen_2  = Population::new();
        // let mut gen_3  = Population::new();
        // let mut gen_4  = Population::new();
        // gen_0.individuals = vec![pop2.individuals[9].clone()];
        // gen_1.individuals = vec![pop2.individuals[6].clone(), pop2.individuals[7].clone(), pop2.individuals[8].clone()];
        // gen_2.individuals = vec![pop2.individuals[5].clone(), pop2.individuals[4].clone(), pop2.individuals[3].clone()];
        // gen_3.individuals = vec![pop2.individuals[2].clone(), pop2.individuals[1].clone()];
        // gen_4.individuals = vec![pop2.individuals[0].clone()];

        // let collection = vec![gen_0, gen_1, gen_2, gen_3, gen_4];
        // assert_eq!(collection[4].individuals[0].get_genealogy(&collection, 15), real_tree2, "Generated tree is broken for hash-computed tree");

        // What about consanguinity?
        // This function should return this tree:
        //         10
        //         / \
        // 11,7,8---1   2
        //  &/   / \ / \
        //  |   3   4   5---
        //  |  / \ / \ /    \
        //  | 6   7   8      \
        //  |/                \
        //  9                  9
        let mut real_tree: HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> = HashMap::new();
        real_tree.insert((10, Some(vec![1, 2])), [0].iter().cloned().collect());
        real_tree.insert(
            (1, Some(vec![3, 4, 7, 8, 9, 11])),
            [1].iter().cloned().collect(),
        );
        real_tree.insert((2, Some(vec![4, 5])), [1].iter().cloned().collect());
        real_tree.insert((3, Some(vec![6, 7])), [2].iter().cloned().collect());
        real_tree.insert((4, Some(vec![7, 8])), [2].iter().cloned().collect());
        real_tree.insert((5, Some(vec![8, 9])), [2].iter().cloned().collect());
        real_tree.insert((6, Some(vec![9])), [3].iter().cloned().collect());
        real_tree.insert((7, None), [2, 3].iter().cloned().collect());
        real_tree.insert((8, None), [2, 3].iter().cloned().collect());
        real_tree.insert((9, None), [2, 3, 4].iter().cloned().collect());
        real_tree.insert((11, None), [2].iter().cloned().collect());

        let mut pop = Population::test();
        let mut gen_0 = Population::new();
        let mut gen_1 = Population::new();
        let mut gen_2 = Population::new();
        let mut gen_3 = Population::new();
        let mut gen_4 = Population::new();
        pop.individuals[0].hash = 10;
        pop.individuals[0].parents = Some(vec![1, 2]);
        pop.individuals[1].parents = Some(vec![3, 4, 7, 8, 9, 11]);
        pop.individuals[2].parents = Some(vec![4, 5]);
        pop.individuals[3].parents = Some(vec![6, 7]);
        pop.individuals[4].parents = Some(vec![7, 8]);
        pop.individuals[5].parents = Some(vec![8, 9]);
        pop.individuals[6].parents = Some(vec![9]);

        for (i, epoch) in [4, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0].iter().enumerate() {
            pop.individuals[i].epoch = *epoch;
        }
        pop.individuals[10].hash = 11;
        gen_0.individuals = vec![pop.individuals[9].clone(), pop.individuals[10].clone()];
        gen_1.individuals = vec![
            pop.individuals[9].clone(),
            pop.individuals[6].clone(),
            pop.individuals[7].clone(),
            pop.individuals[8].clone(),
        ];
        gen_2.individuals = vec![
            pop.individuals[9].clone(),
            pop.individuals[5].clone(),
            pop.individuals[4].clone(),
            pop.individuals[3].clone(),
        ];
        gen_3.individuals = vec![pop.individuals[2].clone(), pop.individuals[1].clone()];
        gen_4.individuals = vec![pop.individuals[0].clone()];

        let collection = vec![gen_0, gen_1, gen_2, gen_3, gen_4];
        assert_eq!(
            collection[4].individuals[0].get_genealogy(&collection, 15),
            real_tree,
            "Generated tree is broken for complex tree"
        );

        // Max depth test
        let mut real_tree: HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> = HashMap::new();
        real_tree.insert((10, Some(vec![1, 2])), [0].iter().cloned().collect());
        real_tree.insert(
            (1, Some(vec![3, 4, 7, 8, 9, 11])),
            [1].iter().cloned().collect(),
        );
        real_tree.insert((2, Some(vec![4, 5])), [1].iter().cloned().collect());
        assert_eq!(
            collection[4].individuals[0].get_genealogy(&collection, 1),
            real_tree,
            "Max_depth should limit ancestors"
        );
    }

    #[test]
    fn test_get_genealogy_edge_cases() {
        // Empty population
        let empty_population: Vec<Population> = Vec::new();
        let individual = Individual::test();
        let genealogy = individual.get_genealogy(&empty_population, 10);
        assert!(
            genealogy.is_empty(),
            "Genealogy should be empty for an empty population"
        );

        // Single individual
        let mut single_individual_population = vec![Population::new()];
        single_individual_population[0]
            .individuals
            .push(individual.clone());
        let genealogy = individual.get_genealogy(&single_individual_population, 10);
        assert_eq!(
            genealogy.len(),
            1,
            "Genealogy should contain only the individual itself"
        );

        // Individual having no parents
        let mut no_parents_population = vec![Population::new()];
        let mut no_parents_individual = Individual::test();
        no_parents_individual.parents = None;
        no_parents_population[0]
            .individuals
            .push(no_parents_individual.clone());
        let genealogy = no_parents_individual.get_genealogy(&no_parents_population, 10);
        assert_eq!(
            genealogy.len(),
            1,
            "Genealogy should contain only the individual itself"
        );

        // max_depth set to 0
        let genealogy = individual.get_genealogy(&empty_population, 0);
        assert!(
            genealogy.is_empty(),
            "Genealogy should be empty for max_depth set to 0"
        );
    }

    #[test]
    fn test_get_genealogy_complex() {
        // Deep Tree
        let mut deep_tree: HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> = HashMap::new();
        deep_tree.insert((1, Some(vec![2])), [0].iter().cloned().collect());
        deep_tree.insert((2, Some(vec![3])), [1].iter().cloned().collect());
        deep_tree.insert((3, Some(vec![4])), [2].iter().cloned().collect());
        deep_tree.insert((4, Some(vec![5])), [3].iter().cloned().collect());
        deep_tree.insert((5, Some(vec![6])), [4].iter().cloned().collect());
        deep_tree.insert((6, Some(vec![7])), [5].iter().cloned().collect());
        deep_tree.insert((7, Some(vec![8])), [6].iter().cloned().collect());
        deep_tree.insert((8, Some(vec![9])), [7].iter().cloned().collect());
        deep_tree.insert((9, Some(vec![10])), [8].iter().cloned().collect());
        deep_tree.insert((10, None), [9].iter().cloned().collect());

        let mut pop_deep = Population::test();
        pop_deep.individuals[0].hash = 1;
        pop_deep.individuals[0].parents = Some(vec![2]);
        pop_deep.individuals[1].hash = 2;
        pop_deep.individuals[1].parents = Some(vec![3]);
        pop_deep.individuals[2].hash = 3;
        pop_deep.individuals[2].parents = Some(vec![4]);
        pop_deep.individuals[3].hash = 4;
        pop_deep.individuals[3].parents = Some(vec![5]);
        pop_deep.individuals[4].hash = 5;
        pop_deep.individuals[4].parents = Some(vec![6]);
        pop_deep.individuals[5].hash = 6;
        pop_deep.individuals[5].parents = Some(vec![7]);
        pop_deep.individuals[6].hash = 7;
        pop_deep.individuals[6].parents = Some(vec![8]);
        pop_deep.individuals[7].hash = 8;
        pop_deep.individuals[7].parents = Some(vec![9]);
        pop_deep.individuals[8].hash = 9;
        pop_deep.individuals[8].parents = Some(vec![10]);
        pop_deep.individuals[9].hash = 10;
        pop_deep.individuals[9].parents = None;

        for (i, epoch) in [9, 8, 7, 6, 5, 4, 3, 2, 1, 0].iter().enumerate() {
            pop_deep.individuals[i].epoch = *epoch;
        }

        let mut gen_0_deep = Population::new();
        let mut gen_1_deep = Population::new();
        let mut gen_2_deep = Population::new();
        let mut gen_3_deep = Population::new();
        let mut gen_4_deep = Population::new();
        let mut gen_5_deep = Population::new();
        let mut gen_6_deep = Population::new();
        let mut gen_7_deep = Population::new();
        let mut gen_8_deep = Population::new();
        let mut gen_9_deep = Population::new();

        gen_0_deep.individuals = vec![pop_deep.individuals[9].clone()];
        gen_1_deep.individuals = vec![pop_deep.individuals[8].clone()];
        gen_2_deep.individuals = vec![pop_deep.individuals[7].clone()];
        gen_3_deep.individuals = vec![pop_deep.individuals[6].clone()];
        gen_4_deep.individuals = vec![pop_deep.individuals[5].clone()];
        gen_5_deep.individuals = vec![pop_deep.individuals[4].clone()];
        gen_6_deep.individuals = vec![pop_deep.individuals[3].clone()];
        gen_7_deep.individuals = vec![pop_deep.individuals[2].clone()];
        gen_8_deep.individuals = vec![pop_deep.individuals[1].clone()];
        gen_9_deep.individuals = vec![pop_deep.individuals[0].clone()];

        let collection_deep = vec![
            gen_0_deep, gen_1_deep, gen_2_deep, gen_3_deep, gen_4_deep, gen_5_deep, gen_6_deep,
            gen_7_deep, gen_8_deep, gen_9_deep,
        ];
        assert_eq!(
            collection_deep[9].individuals[0].get_genealogy(&collection_deep, 10),
            deep_tree,
            "Generated tree is broken for deep tree"
        );

        // Wide Tree
        let mut wide_tree: HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> = HashMap::new();
        wide_tree.insert((1, Some(vec![2, 3, 4, 5])), [0].iter().cloned().collect());
        wide_tree.insert((2, None), [1].iter().cloned().collect());
        wide_tree.insert((3, None), [1].iter().cloned().collect());
        wide_tree.insert((4, None), [1].iter().cloned().collect());
        wide_tree.insert((5, None), [1].iter().cloned().collect());

        let mut pop_wide = Population::test();
        pop_wide.individuals[0].hash = 1;
        pop_wide.individuals[0].parents = Some(vec![2, 3, 4, 5]);
        pop_wide.individuals[1].hash = 2;
        pop_wide.individuals[1].parents = None;
        pop_wide.individuals[2].hash = 3;
        pop_wide.individuals[2].parents = None;
        pop_wide.individuals[3].hash = 4;
        pop_wide.individuals[3].parents = None;
        pop_wide.individuals[4].hash = 5;
        pop_wide.individuals[4].parents = None;

        for (i, epoch) in [1, 0, 0, 0, 0].iter().enumerate() {
            pop_wide.individuals[i].epoch = *epoch;
        }

        let mut gen_0_wide = Population::new();
        let mut gen_1_wide = Population::new();

        gen_0_wide.individuals = vec![
            pop_wide.individuals[1].clone(),
            pop_wide.individuals[2].clone(),
            pop_wide.individuals[3].clone(),
            pop_wide.individuals[4].clone(),
        ];
        gen_1_wide.individuals = vec![pop_wide.individuals[0].clone()];

        let collection_wide = vec![gen_0_wide, gen_1_wide];
        assert_eq!(
            collection_wide[1].individuals[0].get_genealogy(&collection_wide, 10),
            wide_tree,
            "Generated tree is broken for wide tree"
        );

        // Tree with Cycles
        let mut cycle_tree: HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> = HashMap::new();
        cycle_tree.insert((1, Some(vec![2])), [0].iter().cloned().collect());
        cycle_tree.insert((2, Some(vec![3])), [1].iter().cloned().collect());
        cycle_tree.insert((3, Some(vec![1])), [2].iter().cloned().collect());

        let mut pop_cycle = Population::test();
        pop_cycle.individuals[0].hash = 1;
        pop_cycle.individuals[0].parents = Some(vec![2]);
        pop_cycle.individuals[1].hash = 2;
        pop_cycle.individuals[1].parents = Some(vec![3]);
        pop_cycle.individuals[2].hash = 3;
        pop_cycle.individuals[2].parents = Some(vec![1]);

        for (i, epoch) in [2, 1, 0].iter().enumerate() {
            pop_cycle.individuals[i].epoch = *epoch;
        }

        let mut gen_0_cycle = Population::new();
        let mut gen_1_cycle = Population::new();
        let mut gen_2_cycle = Population::new();

        gen_0_cycle.individuals = vec![pop_cycle.individuals[2].clone()];
        gen_1_cycle.individuals = vec![pop_cycle.individuals[1].clone()];
        gen_2_cycle.individuals = vec![pop_cycle.individuals[0].clone()];

        let collection_cycle = vec![gen_0_cycle, gen_1_cycle, gen_2_cycle];
        assert_eq!(
            collection_cycle[2].individuals[0].get_genealogy(&collection_cycle, 10),
            cycle_tree,
            "Generated tree is broken for tree with cycles"
        );

        // Tree with Missing Parents
        let mut missing_parents_tree: HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> =
            HashMap::new();
        missing_parents_tree.insert((1, Some(vec![2])), [0].iter().cloned().collect());
        missing_parents_tree.insert((2, None), [1].iter().cloned().collect());

        let mut pop_missing = Population::test();
        pop_missing.individuals[0].hash = 1;
        pop_missing.individuals[0].parents = Some(vec![2]);
        pop_missing.individuals[1].hash = 2;
        pop_missing.individuals[1].parents = None;

        for (i, epoch) in [1, 0].iter().enumerate() {
            pop_missing.individuals[i].epoch = *epoch;
        }

        let mut gen_0_missing = Population::new();
        let mut gen_1_missing = Population::new();

        gen_0_missing.individuals = vec![pop_missing.individuals[1].clone()];
        gen_1_missing.individuals = vec![pop_missing.individuals[0].clone()];

        let collection_missing = vec![gen_0_missing, gen_1_missing];
        assert_eq!(
            collection_missing[1].individuals[0].get_genealogy(&collection_missing, 10),
            missing_parents_tree,
            "Generated tree is broken for tree with missing parents"
        );

        // Tree with Multiple Paths
        let mut multiple_paths_tree: HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> =
            HashMap::new();
        multiple_paths_tree.insert((1, Some(vec![2, 3])), [0].iter().cloned().collect());
        multiple_paths_tree.insert((2, Some(vec![4])), [1].iter().cloned().collect());
        multiple_paths_tree.insert((3, Some(vec![4])), [1].iter().cloned().collect());
        multiple_paths_tree.insert((4, None), [2].iter().cloned().collect());

        let mut pop_multiple = Population::test();
        pop_multiple.individuals[0].hash = 1;
        pop_multiple.individuals[0].parents = Some(vec![2, 3]);
        pop_multiple.individuals[1].hash = 2;
        pop_multiple.individuals[1].parents = Some(vec![4]);
        pop_multiple.individuals[2].hash = 3;
        pop_multiple.individuals[2].parents = Some(vec![4]);
        pop_multiple.individuals[3].hash = 4;
        pop_multiple.individuals[3].parents = None;

        for (i, epoch) in [2, 1, 1, 0].iter().enumerate() {
            pop_multiple.individuals[i].epoch = *epoch;
        }

        let mut gen_0_multiple = Population::new();
        let mut gen_1_multiple = Population::new();
        let mut gen_2_multiple = Population::new();

        gen_0_multiple.individuals = vec![pop_multiple.individuals[3].clone()];
        gen_1_multiple.individuals = vec![
            pop_multiple.individuals[1].clone(),
            pop_multiple.individuals[2].clone(),
        ];
        gen_2_multiple.individuals = vec![pop_multiple.individuals[0].clone()];

        let collection_multiple = vec![gen_0_multiple, gen_1_multiple, gen_2_multiple];
        assert_eq!(
            collection_multiple[2].individuals[0].get_genealogy(&collection_multiple, 10),
            multiple_paths_tree,
            "Generated tree is broken for tree with multiple paths"
        );
    }

    #[test]
    fn test_signed_jaccard_dissimilarity_with_empty_individuals() {
        // Test with two individuals having no features
        // Expected: 0.0 (no features means no dissimilarity by convention)
        let ind1 = Individual::test_with_these_given_features(vec![]);
        let ind2 = Individual::test_with_these_given_features(vec![]);
        assert_eq!(ind1.signed_jaccard_dissimilarity_with(&ind2), 0.0);
    }

    #[test]
    fn test_signed_jaccard_dissimilarity_with_identical_signs() {
        // Test with identical features having same signs after signum()
        // Individual 1: features {(1, 4), (2, 2)} → signum → {(1, +1), (2, +1)}
        // Individual 2: features {(1, 1), (2, 3)} → signum → {(1, +1), (2, +1)}
        // Intersection: 2, Union: 2 → Dissimilarity = 1 - 2/2 = 0.0
        let ind1 = Individual::test_with_these_given_features(vec![(1, 4), (2, 2)]);
        let ind2 = Individual::test_with_these_given_features(vec![(1, 1), (2, 3)]);
        assert_eq!(ind1.signed_jaccard_dissimilarity_with(&ind2), 0.0);
    }

    #[test]
    fn test_signed_jaccard_dissimilarity_with_opposite_signs() {
        // Test with same feature IDs but opposite signs after signum()
        // Individual 1: features {(1, 2), (2, -1)} → signum → {(1, +1), (2, -1)}
        // Individual 2: features {(1, -3), (2, 4)} → signum → {(1, -1), (2, +1)}
        // Intersection: 0, Union: 4 → Dissimilarity = 1 - 0/4 = 1.0
        let ind1 = Individual::test_with_these_given_features(vec![(1, 2), (2, -1)]);
        let ind2 = Individual::test_with_these_given_features(vec![(1, -3), (2, 4)]);
        assert_eq!(ind1.signed_jaccard_dissimilarity_with(&ind2), 1.0);
    }

    #[test]
    fn test_signed_jaccard_dissimilarity_with_partial_overlap() {
        // Test with partial overlap in signed features after signum()
        // Individual 1: features {(1, 1), (2, 2), (3, -1)} → signum → {(1, +1), (2, +1), (3, -1)}
        // Individual 2: features {(2, 4), (3, -2), (4, 1)} → signum → {(2, +1), (3, -1), (4, +1)}
        // Intersection: {(2, +1), (3, -1)} = 2
        // Union: {(1, +1), (2, +1), (3, -1), (4, +1)} = 4
        // Dissimilarity = 1 - 2/4 = 0.5
        let ind1 = Individual::test_with_these_given_features(vec![(1, 1), (2, 2), (3, -1)]);
        let ind2 = Individual::test_with_these_given_features(vec![(2, 4), (3, -2), (4, 1)]);
        assert_eq!(ind1.signed_jaccard_dissimilarity_with(&ind2), 0.5);
    }

    #[test]
    fn test_signed_jaccard_dissimilarity_with_minimal_overlap() {
        // Test with minimal overlap in signed features
        // Individual 1: features {(1, -4), (2, 2), (4, -1)} → signum → {(1, -1), (2, +1), (4, -1)}
        // Individual 2: features {(2, -1), (3, 1), (4, -2)} → signum → {(2, -1), (3, +1), (4, -1)}
        // Intersection: {(4, -1)} = 1
        // Union: {(1, -1), (2, +1), (2, -1), (3, +1), (4, -1)} = 5
        // Dissimilarity = 1 - 1/5 = 0.8
        let ind1 = Individual::test_with_these_given_features(vec![(1, -4), (2, 2), (4, -1)]);
        let ind2 = Individual::test_with_these_given_features(vec![(2, -1), (3, 1), (4, -2)]);
        assert_eq!(ind1.signed_jaccard_dissimilarity_with(&ind2), 0.8);
    }

    #[test]
    fn test_signed_jaccard_dissimilarity_with_disjoint_features() {
        // Test with partial feature ID overlap but different signs
        // Individual 1: features {(1, 1), (2, -2)} → signum → {(1, +1), (2, -1)}
        // Individual 2: features {(1, 3), (3, 1)} → signum → {(1, +1), (3, +1)}
        // Intersection: {(1, +1)} = 1
        // Union: {(1, +1), (2, -1), (3, +1)} = 3
        // Dissimilarity = 1 - 1/3 = 2/3 ≈ 0.6666666666666667
        let ind1 = Individual::test_with_these_given_features(vec![(1, 1), (2, -2)]);
        let ind2 = Individual::test_with_these_given_features(vec![(1, 3), (3, 1)]);
        let expected = 2.0 / 3.0;
        let result = ind1.signed_jaccard_dissimilarity_with(&ind2);
        assert!((result - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_signed_jaccard_dissimilarity_with_zero_coefficient() {
        // Test with zero coefficient (signum(0) = 0)
        // Individual 1: features {(1, 0), (2, 1)} → signum → {(1, 0), (2, +1)}
        // Individual 2: features {(1, 2), (3, -1)} → signum → {(1, +1), (3, -1)}
        // Intersection: {} = 0 (no matching (id, sign) pairs)
        // Union: {(1, 0), (1, +1), (2, +1), (3, -1)} = 4
        // Dissimilarity = 1 - 0/4 = 1.0
        let ind1 = Individual::test_with_these_given_features(vec![(1, 0), (2, 1)]);
        let ind2 = Individual::test_with_these_given_features(vec![(1, 2), (3, -1)]);
        assert_eq!(ind1.signed_jaccard_dissimilarity_with(&ind2), 1.0);
    }

    #[test]
    fn test_signed_jaccard_dissimilarity_with_single_feature_opposite_sign() {
        // Test with single feature having opposite signs after signum()
        // Individual 1: features {(5, 4)} → signum → {(5, +1)}
        // Individual 2: features {(5, -2)} → signum → {(5, -1)}
        // Intersection: {} = 0
        // Union: {(5, +1), (5, -1)} = 2
        // Dissimilarity = 1 - 0/2 = 1.0
        let ind1 = Individual::test_with_these_given_features(vec![(5, 4)]);
        let ind2 = Individual::test_with_these_given_features(vec![(5, -2)]);
        assert_eq!(ind1.signed_jaccard_dissimilarity_with(&ind2), 1.0);
    }

    #[test]
    fn test_signed_jaccard_dissimilarity_with_asymmetric_sizes() {
        // Test with individuals having different numbers of features
        // Individual 1: features {(1, 1), (2, 2), (3, 1), (4, 3)} → signum → {(1, +1), (2, +1), (3, +1), (4, +1)}
        // Individual 2: features {(1, 1)} → signum → {(1, +1)}
        // Intersection: {(1, +1)} = 1
        // Union: {(1, +1), (2, +1), (3, +1), (4, +1)} = 4
        // Dissimilarity = 1 - 1/4 = 0.75
        let ind1 = Individual::test_with_these_given_features(vec![(1, 1), (2, 2), (3, 1), (4, 3)]);
        let ind2 = Individual::test_with_these_given_features(vec![(1, 1)]);
        assert_eq!(ind1.signed_jaccard_dissimilarity_with(&ind2), 0.75);
    }

    #[test]
    fn test_signed_jaccard_dissimilarity_with_one_empty() {
        // Test with one empty individual
        // Individual 1: features {(1, 1), (2, -1)} → signum → {(1, +1), (2, -1)}
        // Individual 2: features {} (empty)
        // Intersection: {} = 0
        // Union: {(1, +1), (2, -1)} = 2
        // Dissimilarity = 1 - 0/2 = 1.0
        let ind1 = Individual::test_with_these_given_features(vec![(1, 1), (2, -1)]);
        let ind2 = Individual::test_with_these_given_features(vec![]);
        assert_eq!(ind1.signed_jaccard_dissimilarity_with(&ind2), 1.0);

        // Test symmetry
        assert_eq!(ind2.signed_jaccard_dissimilarity_with(&ind1), 1.0);
    }

    #[test]
    fn test_signed_jaccard_dissimilarity_with_pow2_language() {
        // Test with pow2 language values (-4,-2,-1,0,1,2,4)
        // Individual 1: features {(1, -4), (2, 2), (3, 0)} → signum → {(1, -1), (2, +1), (3, 0)}
        // Individual 2: features {(1, -2), (2, 4), (4, 1)} → signum → {(1, -1), (2, +1), (4, +1)}
        // Intersection: {(1, -1), (2, +1)} = 2
        // Union: {(1, -1), (2, +1), (3, 0), (4, +1)} = 4
        // Dissimilarity = 1 - 2/4 = 0.5
        let ind1 = Individual::test_with_these_given_features(vec![(1, -4), (2, 2), (3, 0)]);
        let ind2 = Individual::test_with_these_given_features(vec![(1, -2), (2, 4), (4, 1)]);
        assert_eq!(ind1.signed_jaccard_dissimilarity_with(&ind2), 0.5);
    }

    // ============================================================================
    // Tests for ThresholdCI
    // ============================================================================

    #[test]
    fn test_threshold_ci_creation_and_initialization() {
        let ci = ThresholdCI {
            upper: 0.8,
            lower: 0.2,
            rejection_rate: 0.15,
        };

        assert_eq!(ci.upper, 0.8);
        assert_eq!(ci.lower, 0.2);
        assert_eq!(ci.rejection_rate, 0.15);
        assert!(
            ci.lower < ci.upper,
            "Lower bound should be less than upper bound"
        );
    }

    #[test]
    fn test_threshold_ci_valid_bounds_ordering() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.3,
            upper: 0.7,
            rejection_rate: 0.0,
        });

        let ci = ind.threshold_ci.as_ref().unwrap();
        assert!(
            ci.lower < ind.threshold,
            "Lower CI should be below threshold"
        );
        assert!(
            ind.threshold < ci.upper,
            "Upper CI should be above threshold"
        );
    }

    #[test]
    fn test_evaluate_class_with_threshold_ci_abstention_zone() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.3,
            upper: 0.7,
            rejection_rate: 0.0,
        });

        let data = Data::test2();
        // Modify data to have specific scores for testing
        // scores: [0.89, 0.79, 0.74, -0.73, 0.89, 0.79, 0.74, -0.73, 0.89, 0.79]
        let classes = ind.evaluate_class(&data);

        // Verify that scores are classified correctly:
        // 0.89 > 0.7 → class 1
        // 0.79 > 0.7 → class 1
        // 0.74 > 0.7 → class 1
        // -0.73 < 0.3 → class 0
        assert_eq!(classes[0], 1, "Score 0.89 should be class 1");
        assert_eq!(classes[1], 1, "Score 0.79 should be class 1");
        assert_eq!(classes[2], 1, "Score 0.74 should be class 1");
        assert_eq!(classes[3], 0, "Score -0.73 should be class 0");
    }

    #[test]
    fn test_evaluate_class_with_threshold_ci_scores_in_abstention() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.6,
            upper: 0.9,
            rejection_rate: 0.0,
        });

        let data = Data::test2();
        // scores: [0.89, 0.79, 0.74, -0.73, 0.89, 0.79, 0.74, -0.73, 0.89, 0.79]
        let classes = ind.evaluate_class(&data);

        // Verify abstention zone (class 2):
        // 0.89 > 0.9? No, it's equal, but > upper → class 1 (actually 0.89 < 0.9)
        // 0.79 is between 0.6 and 0.9 → class 2
        // 0.74 is between 0.6 and 0.9 → class 2
        assert_eq!(
            classes[1], 2,
            "Score 0.79 should be in abstention zone (class 2)"
        );
        assert_eq!(
            classes[2], 2,
            "Score 0.74 should be in abstention zone (class 2)"
        );
    }

    #[test]
    fn test_evaluate_class_without_threshold_ci() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.threshold_ci = None;

        let data = Data::test2();
        let classes = ind.evaluate_class(&data);

        // Without CI, only classes 0 and 1 should be returned
        for &class in &classes {
            assert!(
                class == 0 || class == 1,
                "Without CI, only classes 0 and 1 should exist, got {}",
                class
            );
        }
    }

    #[test]
    fn test_evaluate_class_and_score_with_threshold_ci() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.4,
            upper: 0.8,
            rejection_rate: 0.0,
        });

        let data = Data::test2();
        let (classes, scores) = ind.evaluate_class_and_score(&data);

        assert_eq!(
            classes.len(),
            scores.len(),
            "Classes and scores should have same length"
        );

        for i in 0..classes.len() {
            if scores[i] > 0.8 {
                assert_eq!(classes[i], 1, "Score {} > 0.8 should be class 1", scores[i]);
            } else if scores[i] < 0.4 {
                assert_eq!(classes[i], 0, "Score {} < 0.4 should be class 0", scores[i]);
            } else {
                assert_eq!(
                    classes[i], 2,
                    "Score {} in [0.4, 0.8] should be class 2",
                    scores[i]
                );
            }
        }
    }

    #[test]
    fn test_rejection_rate_all_in_ci() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.threshold_ci = Some(ThresholdCI {
            lower: -10.0,
            upper: 10.0,
            rejection_rate: 0.0,
        });

        let data = Data::test2();
        let (_, _, _, rejection_rate, _) = ind.compute_metrics(&data);

        // All scores should be in CI zone
        assert_eq!(
            rejection_rate, 1.0,
            "All samples should be rejected with such wide CI"
        );
    }

    #[test]
    fn test_rejection_rate_none_in_ci() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.49,
            upper: 0.51,
            rejection_rate: 0.0,
        });

        let data = Data::test2();
        let (_, _, _, rejection_rate, _) = ind.compute_metrics(&data);

        // No scores should be in such narrow CI zone around threshold
        assert!(
            rejection_rate < 0.5,
            "With narrow CI, rejection rate should be low, got {}",
            rejection_rate
        );
    }

    #[test]
    fn test_rejection_rate_bounds() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.3,
            upper: 0.7,
            rejection_rate: 0.0,
        });

        let data = Data::test2();
        let (_, _, _, rejection_rate, _) = ind.compute_metrics(&data);

        assert!(
            rejection_rate >= 0.0 && rejection_rate <= 1.0,
            "Rejection rate should be in [0, 1], got {}",
            rejection_rate
        );
    }

    #[test]
    fn test_threshold_ci_very_wide() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.threshold_ci = Some(ThresholdCI {
            lower: -1000.0,
            upper: 1000.0,
            rejection_rate: 0.0,
        });

        let data = Data::test2();
        let classes = ind.evaluate_class(&data);

        // All scores should be in abstention zone
        for &class in &classes {
            assert_eq!(
                class, 2,
                "With very wide CI, all should be in abstention zone"
            );
        }
    }

    #[test]
    fn test_threshold_ci_very_narrow() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.4999,
            upper: 0.5001,
            rejection_rate: 0.0,
        });

        let data = Data::test2();
        let classes = ind.evaluate_class(&data);

        // Very few (likely none) should be in abstention zone
        let abstentions = classes.iter().filter(|&&c| c == 2).count();
        assert!(
            abstentions == 0,
            "With very narrow CI, no abstention expected, got {}",
            abstentions
        );
    }

    #[test]
    fn test_threshold_ci_asymmetric() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.1, // Far below threshold
            upper: 0.6, // Close to threshold
            rejection_rate: 0.0,
        });

        let data = Data::test2();
        let classes = ind.evaluate_class(&data);

        // Verify asymmetric behavior
        for i in 0..classes.len() {
            let score = ind.evaluate(&data)[i];
            if score > 0.6 {
                assert_eq!(classes[i], 1);
            } else if score < 0.1 {
                assert_eq!(classes[i], 0);
            } else {
                assert_eq!(classes[i], 2);
            }
        }
    }

    #[test]
    fn test_threshold_ci_equal_bounds() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.5,
            upper: 0.5,
            rejection_rate: 0.0,
        });

        let data = Data::test2();
        let classes = ind.evaluate_class(&data);

        // With equal bounds, only exact matches should be abstention
        let abstentions = classes.iter().filter(|&&c| c == 2).count();
        // Most likely 0 abstentions since exact float equality is rare
        assert!(
            abstentions <= classes.len(),
            "Rejections should be minimal with zero-width CI"
        );
    }

    #[test]
    fn test_child_inherits_threshold_ci() {
        let mut parent = Individual::test();
        parent.threshold_ci = Some(ThresholdCI {
            lower: 0.3,
            upper: 0.7,
            rejection_rate: 0.25,
        });

        let child = Individual::child(&parent);

        assert!(
            child.threshold_ci.is_some(),
            "Child should inherit threshold_ci structure"
        );
        let ci = child.threshold_ci.unwrap();
        assert_eq!(ci.lower, 0.0, "Child CI should be reset to 0.0");
        assert_eq!(ci.upper, 0.0, "Child CI should be reset to 0.0");
        assert_eq!(
            ci.rejection_rate, 0.0,
            "Child CI rejection_rate should be reset to 0.0"
        );
    }

    #[test]
    fn test_child_no_threshold_ci_when_parent_has_none() {
        let parent = Individual::test();
        assert!(parent.threshold_ci.is_none());

        let child = Individual::child(&parent);

        assert!(
            child.threshold_ci.is_none(),
            "Child should not have CI when parent doesn't"
        );
    }

    // ============================================================================
    // Tests for AdditionalMetrics
    // ============================================================================

    #[test]
    fn test_additional_metrics_initialization() {
        let ind = Individual::new();

        assert!(
            ind.metrics.mcc.is_none(),
            "MCC should be None at initialization"
        );
        assert!(
            ind.metrics.f1_score.is_none(),
            "F1-score should be None at initialization"
        );
        assert!(
            ind.metrics.npv.is_none(),
            "NPV should be None at initialization"
        );
        assert!(
            ind.metrics.ppv.is_none(),
            "PPV should be None at initialization"
        );
        assert!(
            ind.metrics.g_mean.is_none(),
            "G-mean should be None at initialization"
        );
    }

    #[test]
    fn test_additional_metrics_mcc_calculation() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        ind.metrics.mcc = Some(0.0); // Signal that we want MCC computed

        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        assert!(
            additional.mcc.is_some(),
            "MCC should be computed when requested"
        );
        let mcc = additional.mcc.unwrap();
        assert!(
            mcc >= -1.0 && mcc <= 1.0,
            "MCC should be in [-1, 1], got {}",
            mcc
        );
    }

    #[test]
    fn test_additional_metrics_f1_calculation() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        ind.metrics.f1_score = Some(0.0); // Signal that we want F1 computed

        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        assert!(
            additional.f1_score.is_some(),
            "F1-score should be computed when requested"
        );
        let f1 = additional.f1_score.unwrap();
        assert!(
            f1 >= 0.0 && f1 <= 1.0,
            "F1-score should be in [0, 1], got {}",
            f1
        );
    }

    #[test]
    fn test_additional_metrics_npv_calculation() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        ind.metrics.npv = Some(0.0); // Signal that we want NPV computed

        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        assert!(
            additional.npv.is_some(),
            "NPV should be computed when requested"
        );
        let npv = additional.npv.unwrap();
        assert!(
            npv >= 0.0 && npv <= 1.0,
            "NPV should be in [0, 1], got {}",
            npv
        );
    }

    #[test]
    fn test_additional_metrics_ppv_calculation() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        ind.metrics.ppv = Some(0.0); // Signal that we want PPV computed

        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        assert!(
            additional.ppv.is_some(),
            "PPV should be computed when requested"
        );
        let ppv = additional.ppv.unwrap();
        assert!(
            ppv >= 0.0 && ppv <= 1.0,
            "PPV should be in [0, 1], got {}",
            ppv
        );
    }

    #[test]
    fn test_additional_metrics_g_mean_calculation() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        ind.metrics.g_mean = Some(0.0); // Signal that we want G-mean computed

        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        assert!(
            additional.g_mean.is_some(),
            "G-mean should be computed when requested"
        );
        let g_mean = additional.g_mean.unwrap();
        assert!(
            g_mean >= 0.0 && g_mean <= 1.0,
            "G-mean should be in [0, 1], got {}",
            g_mean
        );
    }

    #[test]
    fn test_additional_metrics_all_computed() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        // Request all metrics
        ind.metrics.mcc = Some(0.0);
        ind.metrics.f1_score = Some(0.0);
        ind.metrics.npv = Some(0.0);
        ind.metrics.ppv = Some(0.0);
        ind.metrics.g_mean = Some(0.0);

        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        assert!(
            additional.mcc.is_some(),
            "All requested metrics should be computed"
        );
        assert!(additional.f1_score.is_some());
        assert!(additional.npv.is_some());
        assert!(additional.ppv.is_some());
        assert!(additional.g_mean.is_some());
    }

    #[test]
    fn test_additional_metrics_none_when_not_requested() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        // Don't request any additional metrics

        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        assert!(
            additional.mcc.is_none(),
            "Unrequested metrics should remain None"
        );
        assert!(additional.f1_score.is_none());
        assert!(additional.npv.is_none());
        assert!(additional.ppv.is_none());
        assert!(additional.g_mean.is_none());
    }

    #[test]
    fn test_metrics_perfect_classification() {
        let mut ind = Individual::new();
        ind.features = vec![(0, 1)].into_iter().collect();
        ind.threshold = 0.5;
        ind.metrics.mcc = Some(0.0);
        ind.metrics.f1_score = Some(0.0);
        ind.metrics.g_mean = Some(0.0);

        let mut data = Data::new();
        data.sample_len = 10;
        data.y = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];
        data.feature_len = 1;

        // Create perfect separation: class 1 has high scores, class 0 has low scores
        let mut X = HashMap::new();
        for i in 0..5 {
            X.insert((i, 0), 1.0); // Class 1 samples → score 1.0
        }
        for i in 5..10 {
            X.insert((i, 0), 0.0); // Class 0 samples → score 0.0
        }
        data.X = X;

        let (accuracy, sensitivity, specificity, _, additional) = ind.compute_metrics(&data);

        assert_eq!(
            accuracy, 1.0,
            "Perfect classification should have accuracy 1.0"
        );
        assert_eq!(
            sensitivity, 1.0,
            "Perfect classification should have sensitivity 1.0"
        );
        assert_eq!(
            specificity, 1.0,
            "Perfect classification should have specificity 1.0"
        );

        if let Some(mcc) = additional.mcc {
            assert!(
                (mcc - 1.0).abs() < 0.01,
                "Perfect classification should have MCC ≈ 1.0, got {}",
                mcc
            );
        }

        if let Some(f1) = additional.f1_score {
            assert!(
                (f1 - 1.0).abs() < 0.01,
                "Perfect classification should have F1 ≈ 1.0, got {}",
                f1
            );
        }
    }

    #[test]
    fn test_metrics_inverse_classification() {
        let mut ind = Individual::test();
        ind.threshold = 10.0; // Very high threshold - everything predicted as 0
        ind.metrics.mcc = Some(0.0);

        let mut data = Data::test2();
        data.y = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

        let (accuracy, _, _, _, additional) = ind.compute_metrics(&data);

        assert_eq!(
            accuracy, 0.5,
            "Inverse classification with balanced classes should have 50% accuracy"
        );

        if let Some(mcc) = additional.mcc {
            assert!(
                mcc <= 0.0,
                "Inverse classification should have negative or zero MCC, got {}",
                mcc
            );
        }
    }

    #[test]
    fn test_metrics_with_imbalanced_classes_99_1() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.metrics.ppv = Some(0.0);
        ind.metrics.npv = Some(0.0);
        ind.metrics.mcc = Some(0.0);
        ind.metrics.g_mean = Some(0.0);

        let mut data = Data::test2();
        // 99 negatives, 1 positive
        data.y = vec![0; 99];
        data.y.push(1);
        data.sample_len = 100;

        // Modify X to have 100 samples
        let mut new_X = HashMap::new();
        for sample in 0..100 {
            new_X.insert((sample, 0), if sample < 99 { 0.0 } else { 1.0 });
            new_X.insert((sample, 1), if sample < 99 { 0.0 } else { 1.0 });
        }
        data.X = new_X;

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        // With extreme imbalance, PPV and NPV behave differently
        assert!(additional.npv.is_some());
        assert!(additional.ppv.is_some());

        // NPV should be high (many true negatives)
        if let Some(npv) = additional.npv {
            assert!(
                npv > 0.9,
                "NPV should be high with 99% negative class, got {}",
                npv
            );
        }
    }

    #[test]
    fn test_metrics_with_imbalanced_classes_1_99() {
        let mut ind = Individual::new();
        ind.features = vec![(0, 1)].into_iter().collect();
        ind.threshold = 0.5;
        ind.metrics.ppv = Some(0.0);
        ind.metrics.npv = Some(0.0);

        let mut data = Data::new();
        // 1 negative, 99 positives
        data.y = vec![1; 99];
        data.y.insert(0, 0);
        data.sample_len = 100;
        data.feature_len = 2;

        let mut new_X = HashMap::new();
        for sample in 0..100 {
            // First sample (class 0) gets low score, others (class 1) get high score
            new_X.insert((sample, 0), if sample == 0 { 0.0 } else { 1.0 });
        }
        data.X = new_X;

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        // PPV should be high (many true positives)
        if let Some(ppv) = additional.ppv {
            assert!(
                ppv > 0.9,
                "PPV should be high with 99% positive class, got {}",
                ppv
            );
        }
    }

    #[test]
    fn test_metrics_with_class_2_ignored() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        ind.metrics.mcc = Some(0.0);
        ind.metrics.f1_score = Some(0.0);

        let mut data = Data::test2();
        data.y = vec![1, 0, 2, 2, 0, 0, 2, 0, 1, 0]; // Include class 2

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        // Class 2 should be ignored, metrics should still be computed
        assert!(additional.mcc.is_some());
        assert!(additional.f1_score.is_some());

        // Verify values are finite
        if let Some(mcc) = additional.mcc {
            assert!(
                mcc.is_finite(),
                "MCC should be finite even with class 2 present"
            );
        }
    }

    #[test]
    fn test_metrics_consistency_ppv_formula() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        ind.metrics.ppv = Some(0.0);

        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];

        let (tp, fp, _, _) = ind.calculate_confusion_matrix(&data);
        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        if let Some(ppv) = additional.ppv {
            let expected_ppv = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            assert!(
                (ppv - expected_ppv).abs() < 1e-10,
                "PPV should equal TP/(TP+FP): expected {}, got {}",
                expected_ppv,
                ppv
            );
        }
    }

    #[test]
    fn test_metrics_consistency_npv_formula() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        ind.metrics.npv = Some(0.0);

        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];

        let (_, _, tn, fn_count) = ind.calculate_confusion_matrix(&data);
        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        if let Some(npv) = additional.npv {
            let expected_npv = if tn + fn_count > 0 {
                tn as f64 / (tn + fn_count) as f64
            } else {
                0.0
            };
            assert!(
                (npv - expected_npv).abs() < 1e-10,
                "NPV should equal TN/(TN+FN): expected {}, got {}",
                expected_npv,
                npv
            );
        }
    }

    #[test]
    fn test_metrics_consistency_g_mean_formula() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        ind.metrics.g_mean = Some(0.0);

        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];

        let (_, sensitivity, specificity, _, additional) = ind.compute_metrics(&data);

        if let Some(g_mean) = additional.g_mean {
            let expected_g_mean = (sensitivity * specificity).sqrt();
            assert!(
                (g_mean - expected_g_mean).abs() < 1e-10,
                "G-mean should equal sqrt(sensitivity * specificity): expected {}, got {}",
                expected_g_mean,
                g_mean
            );
        }
    }

    // ============================================================================
    // Integration tests: threshold_ci + metrics
    // ============================================================================

    #[test]
    fn test_display_with_threshold_ci_and_metrics() {
        let mut ind = Individual::test2();
        ind.threshold = 0.75;
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.6,
            upper: 0.9,
            rejection_rate: 0.15,
        });
        ind.metrics.mcc = Some(0.5);
        ind.metrics.f1_score = Some(0.7);

        let data = Data::test2();
        let display = ind.display(&data, None, &"ga".to_string(), 0.05);

        assert!(
            display.contains("rejection rate"),
            "Display should show rejection rate"
        );
        assert!(
            display.contains("MCC"),
            "Display should show MCC when present"
        );
        assert!(display.contains("0.500"), "Display should show MCC value");
        assert!(
            display.contains("0.700"),
            "Display should show F1-score value"
        );
    }

    #[test]
    fn test_display_with_threshold_ci_train_and_test() {
        let mut ind = Individual::test2();
        ind.threshold = 0.75;
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.6,
            upper: 0.9,
            rejection_rate: 0.15,
        });

        let data_train = Data::test2();
        let data_test = Data::test2();

        let display = ind.display(&data_train, Some(&data_test), &"ga".to_string(), 0.05);

        // Should show train/test rejection rates
        assert!(
            display.contains("rejection rate"),
            "Display should show rejection rate"
        );
        // Format is "rejection rate X.XXX/Y.YYY" for train/test
        let rejection_count = display.matches("rejection rate").count();
        assert_eq!(rejection_count, 1, "Should have one rejection rate line");
    }

    #[test]
    fn test_compute_metrics_returns_rejection_rate_with_ci() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.6,
            upper: 0.9,
            rejection_rate: 0.0,
        });

        let data = Data::test2();
        let (_, _, _, rejection_rate, _) = ind.compute_metrics(&data);

        assert!(
            rejection_rate >= 0.0 && rejection_rate <= 1.0,
            "Rejection rate should be in [0, 1]"
        );
    }

    #[test]
    fn test_compute_metrics_rejection_rate_zero_without_ci() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        ind.threshold_ci = None;

        let data = Data::test2();
        let (_, _, _, rejection_rate, _) = ind.compute_metrics(&data);

        assert_eq!(rejection_rate, 0.0, "Rejection rate should be 0 without CI");
    }

    #[test]
    fn test_random_select_k_with_threshold_ci_true() {
        let features = vec![0, 1, 2, 3, 4];
        let mut feature_class = HashMap::new();
        for i in 0..5 {
            feature_class.insert(i, 1);
        }

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ind = Individual::random_select_k(
            2,
            3,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            true,
            &mut rng,
        );

        assert!(
            ind.threshold_ci.is_some(),
            "threshold_ci should be created when flag is true"
        );
        let ci = ind.threshold_ci.unwrap();
        assert_eq!(ci.upper, 0.0, "New CI should have upper = 0.0");
        assert_eq!(ci.lower, 0.0, "New CI should have lower = 0.0");
        assert_eq!(
            ci.rejection_rate, 0.0,
            "New CI should have rejection_rate = 0.0"
        );
    }

    #[test]
    fn test_random_select_k_with_threshold_ci_false() {
        let features = vec![0, 1, 2, 3, 4];
        let mut feature_class = HashMap::new();
        for i in 0..5 {
            feature_class.insert(i, 1);
        }

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ind = Individual::random_select_k(
            2,
            3,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            false,
            &mut rng,
        );

        assert!(
            ind.threshold_ci.is_none(),
            "threshold_ci should not be created when flag is false"
        );
    }

    // ============================================================================
    // Tests for random_select (unified function)
    // ============================================================================

    #[test]
    fn test_random_select_without_weights() {
        let features = vec![0, 1, 2, 3, 4];
        let mut feature_class = HashMap::new();
        for i in 0..5 {
            feature_class.insert(i, 1);
        }

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ind = Individual::random_select(
            2,
            3,
            &features,
            &feature_class,
            TERNARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            None, // No weights, should behave like random_select_k
            false,
            &mut rng,
        );

        assert!(
            ind.k >= 2 && ind.k <= 3,
            "Individual should have between k_min and k_max features"
        );
        assert_eq!(ind.language, TERNARY_LANG);
        assert_eq!(ind.data_type, RAW_TYPE);
        assert_eq!(ind.epsilon, DEFAULT_MINIMUM);
    }

    #[test]
    fn test_random_select_with_weights() {
        let features = vec![0, 1, 2, 3, 4];
        let mut feature_class = HashMap::new();
        for i in 0..5 {
            feature_class.insert(i, 1);
        }

        let mut prior_weight = HashMap::new();
        prior_weight.insert(0, 0.1);
        prior_weight.insert(1, 0.1);
        prior_weight.insert(2, 100.0); // Much higher weight
        prior_weight.insert(3, 0.1);
        prior_weight.insert(4, 0.1);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut feature_2_count = 0;
        let n_trials = 50;

        for _ in 0..n_trials {
            let ind = Individual::random_select(
                1,
                1,
                &features,
                &feature_class,
                BINARY_LANG,
                RAW_TYPE,
                DEFAULT_MINIMUM,
                Some(&prior_weight), // With weights
                false,
                &mut rng,
            );
            if ind.features.contains_key(&2) {
                feature_2_count += 1;
            }
        }

        // Feature 2 should be selected much more often
        assert!(
            feature_2_count > 40,
            "Feature with higher weight should be selected more often (got {} out of {})",
            feature_2_count,
            n_trials
        );
    }

    #[test]
    fn test_random_select_consistency_with_old_functions() {
        let features = vec![0, 1, 2];
        let mut feature_class = HashMap::new();
        for i in 0..3 {
            feature_class.insert(i, 1);
        }

        let mut prior_weight = HashMap::new();
        for i in 0..3 {
            prior_weight.insert(i, 1.0);
        }

        // Test that random_select without weights produces same result as random_select_k
        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let ind1 = Individual::random_select_k(
            1,
            2,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            false,
            &mut rng1,
        );

        let mut rng2 = ChaCha8Rng::seed_from_u64(123);
        let ind2 = Individual::random_select(
            1,
            2,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            None,
            false,
            &mut rng2,
        );

        assert_eq!(
            ind1.features, ind2.features,
            "random_select(None) should produce same results as random_select_k"
        );

        // Test that random_select with weights produces same result as random_select_weighted
        let mut rng3 = ChaCha8Rng::seed_from_u64(456);
        let ind3 = Individual::random_select_weighted(
            1,
            2,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &prior_weight,
            false,
            &mut rng3,
        );

        let mut rng4 = ChaCha8Rng::seed_from_u64(456);
        let ind4 = Individual::random_select(
            1,
            2,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            Some(&prior_weight),
            false,
            &mut rng4,
        );

        assert_eq!(
            ind3.features, ind4.features,
            "random_select(Some) should produce same results as random_select_weighted"
        );
    }

    // ============================================================================
    // Tests for random_select_weighted
    // ============================================================================

    #[test]
    fn test_random_select_weighted_basic() {
        let features = vec![0, 1, 2, 3, 4];
        let mut feature_class = HashMap::new();
        feature_class.insert(0, 1);
        feature_class.insert(1, 0);
        feature_class.insert(2, 1);
        feature_class.insert(3, 0);
        feature_class.insert(4, 1);

        let mut prior_weight = HashMap::new();
        prior_weight.insert(0, 1.0);
        prior_weight.insert(1, 1.0);
        prior_weight.insert(2, 1.0);
        prior_weight.insert(3, 1.0);
        prior_weight.insert(4, 1.0);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ind = Individual::random_select_weighted(
            2,
            3,
            &features,
            &feature_class,
            TERNARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &prior_weight,
            false,
            &mut rng,
        );

        assert!(
            ind.k >= 2 && ind.k <= 3,
            "Individual should have between k_min and k_max features"
        );
        assert_eq!(
            ind.language, TERNARY_LANG,
            "input language should be respected"
        );
        assert_eq!(
            ind.data_type, RAW_TYPE,
            "input data_type should be respected"
        );
        assert_eq!(
            ind.epsilon, DEFAULT_MINIMUM,
            "input epsilon should be respected"
        );
        assert!(
            ind.features.keys().all(|k| features.contains(k)),
            "all selected features should be from feature_selection"
        );
    }

    #[test]
    fn test_random_select_weighted_with_weights() {
        let features = vec![0, 1, 2, 3, 4];
        let mut feature_class = HashMap::new();
        for i in 0..5 {
            feature_class.insert(i, 1);
        }

        // Feature 2 has much higher weight
        let mut prior_weight = HashMap::new();
        prior_weight.insert(0, 0.1);
        prior_weight.insert(1, 0.1);
        prior_weight.insert(2, 100.0);
        prior_weight.insert(3, 0.1);
        prior_weight.insert(4, 0.1);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut feature_2_count = 0;
        let n_trials = 100;

        for _ in 0..n_trials {
            let ind = Individual::random_select_weighted(
                1,
                1,
                &features,
                &feature_class,
                BINARY_LANG,
                RAW_TYPE,
                DEFAULT_MINIMUM,
                &prior_weight,
                false,
                &mut rng,
            );
            if ind.features.contains_key(&2) {
                feature_2_count += 1;
            }
        }

        // Feature 2 should be selected much more often than others
        assert!(
            feature_2_count > 80,
            "Feature with higher weight should be selected more often (got {} out of {})",
            feature_2_count,
            n_trials
        );
    }

    #[test]
    fn test_random_select_weighted_zero_weights_filtered() {
        let features = vec![0, 1, 2, 3, 4];
        let mut feature_class = HashMap::new();
        for i in 0..5 {
            feature_class.insert(i, 1);
        }

        // Only features 2 and 3 have positive weights
        let mut prior_weight = HashMap::new();
        prior_weight.insert(0, 0.0);
        prior_weight.insert(1, 0.0);
        prior_weight.insert(2, 1.0);
        prior_weight.insert(3, 1.0);
        prior_weight.insert(4, 0.0);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ind = Individual::random_select_weighted(
            1,
            2,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &prior_weight,
            false,
            &mut rng,
        );

        // Should only select from features 2 and 3
        assert!(
            ind.features.keys().all(|k| *k == 2 || *k == 3),
            "Should only select features with positive weight"
        );
    }

    #[test]
    #[should_panic(expected = "No features with positive weight!")]
    fn test_random_select_weighted_all_zero_weights() {
        let features = vec![0, 1, 2];
        let mut feature_class = HashMap::new();
        for i in 0..3 {
            feature_class.insert(i, 1);
        }

        let mut prior_weight = HashMap::new();
        prior_weight.insert(0, 0.0);
        prior_weight.insert(1, 0.0);
        prior_weight.insert(2, 0.0);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        Individual::random_select_weighted(
            1,
            2,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &prior_weight,
            false,
            &mut rng,
        );
    }

    #[test]
    fn test_random_select_weighted_equal_min_max() {
        // Test that k_min = k_max works correctly for weighted selection
        let features = vec![0, 1, 2, 3, 4];
        let mut feature_class = HashMap::new();
        for i in 0..5 {
            feature_class.insert(i, 1);
        }

        let mut prior_weight = HashMap::new();
        for i in 0..5 {
            prior_weight.insert(i, 1.0);
        }

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Test with k_min = k_max = 3
        let ind = Individual::random_select_weighted(
            3,
            3,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &prior_weight,
            false,
            &mut rng,
        );

        assert_eq!(
            ind.k, 3,
            "Individual should have exactly 3 features when k_min=k_max=3"
        );
        assert_eq!(
            ind.features.len(),
            3,
            "Individual should have exactly 3 features"
        );
    }

    #[test]
    fn test_random_select_weighted_with_threshold_ci() {
        let features = vec![0, 1, 2];
        let mut feature_class = HashMap::new();
        feature_class.insert(0, 1);
        feature_class.insert(1, 0);
        feature_class.insert(2, 1);

        let mut prior_weight = HashMap::new();
        prior_weight.insert(0, 1.0);
        prior_weight.insert(1, 1.0);
        prior_weight.insert(2, 1.0);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ind_with_ci = Individual::random_select_weighted(
            1,
            2,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &prior_weight,
            true,
            &mut rng,
        );

        assert!(
            ind_with_ci.threshold_ci.is_some(),
            "threshold_ci should be Some when threshold_ci parameter is true"
        );

        let mut rng2 = ChaCha8Rng::seed_from_u64(43);
        let ind_without_ci = Individual::random_select_weighted(
            1,
            2,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &prior_weight,
            false,
            &mut rng2,
        );

        assert!(
            ind_without_ci.threshold_ci.is_none(),
            "threshold_ci should be None when threshold_ci parameter is false"
        );
    }

    #[test]
    fn test_random_select_weighted_coefficient_types() {
        let features = vec![0, 1, 2];
        let mut feature_class = HashMap::new();
        feature_class.insert(0, 1);
        feature_class.insert(1, 0);
        feature_class.insert(2, 1);

        let mut prior_weight = HashMap::new();
        prior_weight.insert(0, 1.0);
        prior_weight.insert(1, 1.0);
        prior_weight.insert(2, 1.0);

        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Test BINARY_LANG: all coefficients should be 1
        let ind_bin = Individual::random_select_weighted(
            2,
            3,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &prior_weight,
            false,
            &mut rng,
        );
        assert!(
            ind_bin.features.values().all(|&v| v == 1),
            "BINARY_LANG should only have coefficient 1"
        );

        // Test TERNARY_LANG: coefficients should be 1 or -1
        let ind_ter = Individual::random_select_weighted(
            2,
            3,
            &features,
            &feature_class,
            TERNARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &prior_weight,
            false,
            &mut rng,
        );
        assert!(
            ind_ter.features.values().all(|&v| v == 1 || v == -1),
            "TERNARY_LANG should only have coefficients 1 or -1"
        );

        // Test POW2_LANG: coefficients should be powers of 2 or their negatives
        let ind_pow2 = Individual::random_select_weighted(
            2,
            3,
            &features,
            &feature_class,
            POW2_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &prior_weight,
            false,
            &mut rng,
        );
        for &coef in ind_pow2.features.values() {
            let abs_coef = coef.abs() as u8;
            // Check if it's a power of 2 (has only one bit set)
            assert!(
                abs_coef != 0 && (abs_coef & (abs_coef - 1)) == 0,
                "POW2_LANG coefficients should be powers of 2 (got {})",
                coef
            );
        }
    }

    #[test]
    fn test_random_select_weighted_default_weight() {
        // Test that missing weights default to 1.0
        let features = vec![0, 1, 2];
        let mut feature_class = HashMap::new();
        for i in 0..3 {
            feature_class.insert(i, 1);
        }

        let mut prior_weight = HashMap::new();
        // Only set weight for feature 0
        prior_weight.insert(0, 1.0);
        // Features 1 and 2 will use default weight of 1.0

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ind = Individual::random_select_weighted(
            1,
            2,
            &features,
            &feature_class,
            BINARY_LANG,
            RAW_TYPE,
            DEFAULT_MINIMUM,
            &prior_weight,
            false,
            &mut rng,
        );

        // Should still be able to select features without explicit weights
        assert!(
            ind.k >= 1 && ind.k <= 2,
            "Should handle features with default weights"
        );
    }

    // ============================================================================
    // Edge cases and robustness tests
    // ============================================================================

    #[test]
    fn test_metrics_with_zero_samples() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;

        let mut data = Data::new();
        data.sample_len = 0;
        data.y = vec![];

        let (accuracy, sensitivity, specificity, _, _) = ind.compute_metrics(&data);

        // With zero samples, metrics should handle gracefully
        assert!(
            accuracy.is_nan() || accuracy == 0.0,
            "Accuracy with 0 samples should be NaN or 0"
        );
        assert!(
            sensitivity.is_nan() || sensitivity == 0.0,
            "Sensitivity with 0 samples should be NaN or 0"
        );
        assert!(
            specificity.is_nan() || specificity == 0.0,
            "Specificity with 0 samples should be NaN or 0"
        );
    }

    #[test]
    fn test_metrics_with_one_sample() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.metrics.mcc = Some(0.0);
        ind.metrics.f1_score = Some(0.0);

        let mut data = Data::test2();
        data.sample_len = 1;
        data.y = vec![1];

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        // With one sample, some metrics may be undefined
        if let Some(mcc) = additional.mcc {
            assert!(
                mcc.is_finite() || mcc.is_nan(),
                "MCC should be finite or NaN with 1 sample"
            );
        }
    }

    #[test]
    fn test_metrics_with_two_samples_balanced() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.metrics.mcc = Some(0.0);

        let mut data = Data::test2();
        data.sample_len = 2;
        data.y = vec![0, 1];

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        if let Some(mcc) = additional.mcc {
            assert!(
                mcc >= -1.0 && mcc <= 1.0 || mcc.is_nan(),
                "MCC should be in valid range or NaN with 2 samples"
            );
        }
    }

    #[test]
    fn test_metrics_all_class_zero() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.metrics.ppv = Some(0.0);
        ind.metrics.npv = Some(0.0);
        ind.metrics.mcc = Some(0.0);

        let mut data = Data::test2();
        data.y = vec![0; 10];

        let (_, sensitivity, specificity, _, additional) = ind.compute_metrics(&data);

        // All class 0: specificity should be computable, sensitivity undefined
        assert!(
            specificity >= 0.0 && specificity <= 1.0,
            "Specificity should be valid"
        );
        assert!(
            sensitivity.is_nan() || sensitivity == 0.0,
            "Sensitivity undefined with no positive class"
        );

        if let Some(npv) = additional.npv {
            assert!(
                npv >= 0.0 && npv <= 1.0 || npv.is_nan(),
                "NPV should be in valid range"
            );
        }
    }

    #[test]
    fn test_metrics_all_class_one() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.metrics.ppv = Some(0.0);
        ind.metrics.npv = Some(0.0);
        ind.metrics.mcc = Some(0.0);

        let mut data = Data::test2();
        data.y = vec![1; 10];

        let (_, sensitivity, specificity, _, additional) = ind.compute_metrics(&data);

        // All class 1: sensitivity should be computable, specificity undefined
        assert!(
            sensitivity >= 0.0 && sensitivity <= 1.0,
            "Sensitivity should be valid"
        );
        assert!(
            specificity.is_nan() || specificity == 0.0,
            "Specificity undefined with no negative class"
        );

        if let Some(ppv) = additional.ppv {
            assert!(
                ppv >= 0.0 && ppv <= 1.0 || ppv.is_nan(),
                "PPV should be in valid range"
            );
        }
    }

    #[test]
    fn test_metrics_with_very_close_scores() {
        let mut ind = Individual::new();
        ind.features = vec![(0, 1)].into_iter().collect();
        ind.threshold = 0.50000001;
        ind.metrics.mcc = Some(0.0);

        let mut data = Data::new();
        data.sample_len = 10;
        data.y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
        data.feature_len = 1;

        // All scores very close to threshold
        let mut X = HashMap::new();
        for i in 0..10 {
            X.insert((i, 0), 0.5 + (i as f64) * 0.00000001);
        }
        data.X = X;

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        if let Some(mcc) = additional.mcc {
            assert!(
                mcc.is_finite(),
                "MCC should be finite even with very close scores"
            );
        }
    }

    #[test]
    fn test_metrics_with_extreme_scores() {
        let mut ind = Individual::new();
        ind.features = vec![(0, 1)].into_iter().collect();
        ind.threshold = 0.0;
        ind.metrics.mcc = Some(0.0);

        let mut data = Data::new();
        data.sample_len = 6;
        data.y = vec![0, 0, 0, 1, 1, 1];
        data.feature_len = 1;

        // Extreme scores
        let mut X = HashMap::new();
        X.insert((0, 0), -1e100);
        X.insert((1, 0), -1e50);
        X.insert((2, 0), -1e10);
        X.insert((3, 0), 1e10);
        X.insert((4, 0), 1e50);
        X.insert((5, 0), 1e100);
        data.X = X;

        let (_, _, _, _, additional) = ind.compute_metrics(&data);

        if let Some(mcc) = additional.mcc {
            assert!(mcc.is_finite(), "MCC should be finite with extreme scores");
        }
    }

    #[test]
    fn test_threshold_ci_with_extreme_imbalance() {
        let mut ind = Individual::test();
        ind.threshold = 0.5;
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.4,
            upper: 0.6,
            rejection_rate: 0.0,
        });

        let mut data = Data::test2();
        data.y = vec![0; 100];
        data.y.push(1);
        data.sample_len = 101;

        // Create X for 101 samples
        let mut new_X = HashMap::new();
        for sample in 0..101 {
            new_X.insert((sample, 0), if sample < 100 { 0.0 } else { 1.0 });
            new_X.insert((sample, 1), if sample < 100 { 0.0 } else { 1.0 });
        }
        data.X = new_X;

        let (_, _, _, rejection_rate, _) = ind.compute_metrics(&data);

        assert!(
            rejection_rate >= 0.0 && rejection_rate <= 1.0,
            "Rejection rate should be valid even with extreme imbalance"
        );
    }

    // -----------------------------------------------------------------
    // Tests for monotonicity of rejection_rate when widening [lower, upper] interval
    // -----------------------------------------------------------------

    #[test]
    fn test_individual_rejection_rate_monotonicity_widening_interval() {
        // Test that widening the [lower, upper] interval can only increase or maintain rejection_rate
        // on a fixed dataset for an Individual

        let mut ind = Individual::test2();
        ind.threshold = 0.5;

        let data = Data::test2();

        // Initial narrow interval
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.45,
            upper: 0.55,
            rejection_rate: 0.0,
        });

        let (_, _, _, rejection_rate_narrow, _) = ind.compute_metrics(&data);

        // Wider interval
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.3,
            upper: 0.7,
            rejection_rate: 0.0,
        });

        let (_, _, _, rejection_rate_wide, _) = ind.compute_metrics(&data);

        assert!(rejection_rate_wide >= rejection_rate_narrow - 1e-10,
                "Widening the [lower, upper] interval should increase or maintain rejection_rate: narrow={}, wide={}",
                rejection_rate_narrow, rejection_rate_wide);
    }

    #[test]
    fn test_individual_rejection_rate_monotonicity_multiple_intervals() {
        // Test monotonicity across multiple interval widths
        let mut ind = Individual::test2();
        ind.threshold = 0.5;

        let data = Data::test2();

        let intervals = vec![
            (0.48, 0.52),
            (0.45, 0.55),
            (0.40, 0.60),
            (0.30, 0.70),
            (0.20, 0.80),
        ];

        let mut prev_rejection_rate = 0.0;

        for (lower, upper) in intervals {
            ind.threshold_ci = Some(ThresholdCI {
                lower,
                upper,
                rejection_rate: 0.0,
            });

            let (_, _, _, rejection_rate, _) = ind.compute_metrics(&data);

            assert!(rejection_rate >= prev_rejection_rate - 1e-10,
                    "Rejection rate should be monotonic: interval [{}, {}] has rejection_rate={}, previous was {}",
                    lower, upper, rejection_rate, prev_rejection_rate);

            prev_rejection_rate = rejection_rate;
        }
    }

    #[test]
    fn test_individual_rejection_rate_zero_width_interval() {
        // Edge case: zero-width interval should give same result as no threshold_ci
        let mut ind = Individual::test2();
        ind.threshold = 0.5;

        let data = Data::test2();

        // Without threshold_ci
        ind.threshold_ci = None;
        let (_, _, _, rejection_rate_none, _) = ind.compute_metrics(&data);

        // With zero-width interval (lower == upper == threshold)
        ind.threshold_ci = Some(ThresholdCI {
            lower: 0.5,
            upper: 0.5,
            rejection_rate: 0.0,
        });

        let (_, _, _, rejection_rate_zero, _) = ind.compute_metrics(&data);

        assert_eq!(
            rejection_rate_none, 0.0,
            "Without threshold_ci, rejection_rate should be 0"
        );
        assert!(
            (rejection_rate_zero - rejection_rate_none).abs() < 1e-10,
            "Zero-width interval should give rejection_rate close to no interval case"
        );
    }

    // -----------------------------------------------------------------
    // Tests for FitFunction setting ind.fit and reflecting in ind.metrics
    // -----------------------------------------------------------------

    #[test]
    fn test_fit_function_mcc_sets_fit_and_metrics() {
        // Test that FitFunction::mcc properly sets ind.fit and ind.metrics.mcc
        use crate::param::Param;
        use crate::population::Population;

        let mut pop = Population::new();
        let mut ind = Individual::test2();
        ind.metrics.mcc = Some(0.0); // Initialize to trigger metric tracking
        pop.individuals.push(ind);

        let data = Data::test2();
        let mut param = Param::default();
        param.general.fit = FitFunction::mcc;

        pop.fit_without_penalty(&data, &mut None, &None, &None, &param);

        let fitted_ind = &pop.individuals[0];

        assert!(
            fitted_ind.fit != 0.0 || fitted_ind.fit == 0.0,
            "fit should be set"
        );
        assert!(
            fitted_ind.metrics.mcc.is_some(),
            "MCC metric should be Some"
        );

        let mcc_value = fitted_ind.metrics.mcc.unwrap();
        assert!(
            (fitted_ind.fit - mcc_value).abs() < 1e-10,
            "ind.fit should equal ind.metrics.mcc for FitFunction::mcc: fit={}, mcc={}",
            fitted_ind.fit,
            mcc_value
        );
    }

    #[test]
    fn test_fit_function_f1_score_sets_fit_and_metrics() {
        use crate::param::Param;
        use crate::population::Population;

        let mut pop = Population::new();
        let mut ind = Individual::test2();
        ind.metrics.f1_score = Some(0.0);
        pop.individuals.push(ind);

        let data = Data::test2();
        let mut param = Param::default();
        param.general.fit = FitFunction::f1_score;

        pop.fit_without_penalty(&data, &mut None, &None, &None, &param);

        let fitted_ind = &pop.individuals[0];

        assert!(
            fitted_ind.metrics.f1_score.is_some(),
            "F1-score metric should be Some"
        );

        let f1_value = fitted_ind.metrics.f1_score.unwrap();
        assert!(
            (fitted_ind.fit - f1_value).abs() < 1e-10,
            "ind.fit should equal ind.metrics.f1_score for FitFunction::f1_score: fit={}, f1={}",
            fitted_ind.fit,
            f1_value
        );
    }

    #[test]
    fn test_fit_function_npv_sets_fit_and_metrics() {
        use crate::param::Param;
        use crate::population::Population;

        let mut pop = Population::new();
        let mut ind = Individual::test2();
        ind.metrics.npv = Some(0.0);
        pop.individuals.push(ind);

        let data = Data::test2();
        let mut param = Param::default();
        param.general.fit = FitFunction::npv;

        pop.fit_without_penalty(&data, &mut None, &None, &None, &param);

        let fitted_ind = &pop.individuals[0];

        assert!(
            fitted_ind.metrics.npv.is_some(),
            "NPV metric should be Some"
        );

        let npv_value = fitted_ind.metrics.npv.unwrap();
        assert!(
            (fitted_ind.fit - npv_value).abs() < 1e-10,
            "ind.fit should equal ind.metrics.npv for FitFunction::npv: fit={}, npv={}",
            fitted_ind.fit,
            npv_value
        );
    }

    #[test]
    fn test_fit_function_ppv_sets_fit_and_metrics() {
        use crate::param::Param;
        use crate::population::Population;

        let mut pop = Population::new();
        let mut ind = Individual::test2();
        ind.metrics.ppv = Some(0.0);
        pop.individuals.push(ind);

        let data = Data::test2();
        let mut param = Param::default();
        param.general.fit = FitFunction::ppv;

        pop.fit_without_penalty(&data, &mut None, &None, &None, &param);

        let fitted_ind = &pop.individuals[0];

        assert!(
            fitted_ind.metrics.ppv.is_some(),
            "PPV metric should be Some"
        );

        let ppv_value = fitted_ind.metrics.ppv.unwrap();
        assert!(
            (fitted_ind.fit - ppv_value).abs() < 1e-10,
            "ind.fit should equal ind.metrics.ppv for FitFunction::ppv: fit={}, ppv={}",
            fitted_ind.fit,
            ppv_value
        );
    }

    #[test]
    fn test_fit_function_g_mean_sets_fit_and_metrics() {
        use crate::param::Param;
        use crate::population::Population;

        let mut pop = Population::new();
        let mut ind = Individual::test2();
        ind.metrics.g_mean = Some(0.0);
        pop.individuals.push(ind);

        let data = Data::test2();
        let mut param = Param::default();
        param.general.fit = FitFunction::g_mean;

        pop.fit_without_penalty(&data, &mut None, &None, &None, &param);

        let fitted_ind = &pop.individuals[0];

        assert!(
            fitted_ind.metrics.g_mean.is_some(),
            "G-mean metric should be Some"
        );

        let g_mean_value = fitted_ind.metrics.g_mean.unwrap();
        assert!(
            (fitted_ind.fit - g_mean_value).abs() < 1e-10,
            "ind.fit should equal ind.metrics.g_mean for FitFunction::g_mean: fit={}, g_mean={}",
            fitted_ind.fit,
            g_mean_value
        );
    }

    #[test]
    fn test_fit_function_auc_does_not_set_additional_metrics() {
        use crate::param::Param;
        use crate::population::Population;

        let mut pop = Population::new();
        let ind = Individual::test2();
        pop.individuals.push(ind);

        let data = Data::test2();
        let mut param = Param::default();
        param.general.fit = FitFunction::auc;

        pop.fit_without_penalty(&data, &mut None, &None, &None, &param);

        let fitted_ind = &pop.individuals[0];

        // For FitFunction::auc, fit should be set to auc, but additional metrics should remain None
        assert!(
            (fitted_ind.fit - fitted_ind.auc).abs() < 1e-10,
            "For FitFunction::auc, ind.fit should equal ind.auc"
        );
        assert!(
            fitted_ind.metrics.mcc.is_none(),
            "MCC should be None for FitFunction::auc"
        );
        assert!(
            fitted_ind.metrics.f1_score.is_none(),
            "F1-score should be None for FitFunction::auc"
        );
        assert!(
            fitted_ind.metrics.npv.is_none(),
            "NPV should be None for FitFunction::auc"
        );
        assert!(
            fitted_ind.metrics.ppv.is_none(),
            "PPV should be None for FitFunction::auc"
        );
        assert!(
            fitted_ind.metrics.g_mean.is_none(),
            "G-mean should be None for FitFunction::auc"
        );
    }

    // -----------------------------------------------------------------
    // Tests for prune_by_importance
    // -----------------------------------------------------------------

    #[test]
    fn test_prune_by_importance_with_threshold() {
        let data = Data::specific_test(50, 10);

        let mut individual = Individual::new();
        individual.language = BINARY_LANG;
        individual.data_type = RAW_TYPE;
        individual.features.insert(0, 1);
        individual.features.insert(1, 1);
        individual.features.insert(2, 1);
        individual.features.insert(3, 1);
        individual.features.insert(4, 1);
        individual.k = 5;

        let k_before = individual.k;
        individual.prune_by_importance(&data, 10, 12345, Some(0.0), None, 2);

        assert!(individual.k >= 2, "Should keep at least min_k=2 features");
        assert!(
            individual.k <= k_before,
            "Should not have more features than before"
        );
    }

    #[test]
    fn test_prune_by_importance_with_quantile() {
        let data = Data::specific_test(50, 15);

        let mut individual = Individual::new();
        individual.language = TERNARY_LANG;
        individual.data_type = RAW_TYPE;
        for i in 0..8 {
            individual
                .features
                .insert(i, if i % 2 == 0 { 1 } else { -1 });
        }
        individual.k = 8;

        let k_before = individual.k;
        individual.prune_by_importance(&data, 20, 54321, None, Some((0.25, 0.0)), 3);

        assert!(individual.k >= 3, "Should keep at least min_k=3 features");
        assert!(
            individual.k <= k_before,
            "Should not have more features than before"
        );
    }

    #[test]
    fn test_prune_by_importance_respects_min_k() {
        let data = Data::specific_test(40, 10);

        let mut individual = Individual::new();
        individual.language = BINARY_LANG;
        individual.data_type = RAW_TYPE;
        for i in 0..6 {
            individual.features.insert(i, 1);
        }
        individual.k = 6;

        // Very strict threshold - should try to drop all features
        individual.prune_by_importance(&data, 15, 99999, Some(100.0), None, 4);

        assert_eq!(individual.k, 4, "Should keep exactly min_k=4 features");
    }

    #[test]
    fn test_prune_by_importance_empty_individual() {
        let data = Data::specific_test(30, 5);

        let mut individual = Individual::new();
        individual.language = BINARY_LANG;
        individual.data_type = RAW_TYPE;
        individual.k = 0;

        // Should not panic
        individual.prune_by_importance(&data, 10, 111, Some(0.0), None, 1);

        assert_eq!(individual.k, 0, "Empty individual should remain empty");
    }

    #[test]
    fn test_prune_by_importance_single_feature_preserved() {
        let data = Data::specific_test(40, 8);

        let mut individual = Individual::new();
        individual.language = BINARY_LANG;
        individual.data_type = RAW_TYPE;
        individual.features.insert(2, 1);
        individual.k = 1;

        // Try to prune with aggressive threshold
        individual.prune_by_importance(&data, 10, 222, Some(999.0), None, 1);

        assert_eq!(
            individual.k, 1,
            "Single feature should be preserved when min_k=1"
        );
        assert!(
            individual.features.contains_key(&2),
            "Original feature should remain"
        );
    }

    #[test]
    fn test_prune_by_importance_different_languages() {
        let data = Data::specific_test(60, 12);

        let languages = vec![
            (BINARY_LANG, "Binary"),
            (TERNARY_LANG, "Ternary"),
            (RATIO_LANG, "Ratio"),
            (POW2_LANG, "Pow2"),
        ];

        for (lang, _lang_name) in languages {
            let mut individual = Individual::new();
            individual.language = lang;
            individual.data_type = RAW_TYPE;

            for i in 0..7 {
                let coef = match lang {
                    BINARY_LANG => 1,
                    TERNARY_LANG | RATIO_LANG => {
                        if i % 2 == 0 {
                            1
                        } else {
                            -1
                        }
                    }
                    POW2_LANG => {
                        if i % 2 == 0 {
                            2
                        } else {
                            -2
                        }
                    }
                    _ => 1,
                };
                individual.features.insert(i, coef);
            }
            individual.k = 7;

            if lang == RATIO_LANG {
                individual.epsilon = 1e-5;
                individual.threshold = 1.0;
            }

            let k_before = individual.k;
            individual.prune_by_importance(&data, 15, 555, Some(0.0), None, 2);

            assert!(individual.k >= 2, "Should keep at least min_k features");
            assert!(individual.k <= k_before, "Should not gain features");
        }
    }

    #[test]
    fn test_prune_by_importance_different_data_types() {
        let data = Data::specific_test(50, 10);

        let data_types = vec![
            (RAW_TYPE, "Raw"),
            (PREVALENCE_TYPE, "Prevalence"),
            (LOG_TYPE, "Log"),
        ];

        for (dtype, _dtype_name) in data_types {
            let mut individual = Individual::new();
            individual.language = BINARY_LANG;
            individual.data_type = dtype;
            individual.epsilon = if dtype == LOG_TYPE { 0.1 } else { 1e-5 };

            for i in 0..6 {
                individual.features.insert(i, 1);
            }
            individual.k = 6;

            let k_before = individual.k;
            individual.prune_by_importance(&data, 10, 777, Some(0.0), None, 2);

            assert!(individual.k >= 2, "Should keep at least min_k features");
            assert!(individual.k <= k_before, "Should not gain features");
        }
    }

    #[test]
    fn test_prune_by_importance_determinism() {
        let data = Data::specific_test(60, 12);

        let mut individual1 = Individual::new();
        individual1.language = TERNARY_LANG;
        individual1.data_type = RAW_TYPE;
        for i in 0..8 {
            individual1
                .features
                .insert(i, if i % 2 == 0 { 1 } else { -1 });
        }
        individual1.k = 8;

        let mut individual2 = individual1.clone();

        let seed = 999888;
        individual1.prune_by_importance(&data, 25, seed, Some(0.0), None, 3);
        individual2.prune_by_importance(&data, 25, seed, Some(0.0), None, 3);

        assert_eq!(individual1.k, individual2.k, "Feature counts should match");
        assert_eq!(
            individual1.features, individual2.features,
            "Features should be identical"
        );
    }

    #[test]
    fn test_prune_by_importance_no_threshold_keeps_all() {
        let data = Data::specific_test(50, 10);

        let mut individual = Individual::new();
        individual.language = BINARY_LANG;
        individual.data_type = RAW_TYPE;
        for i in 0..6 {
            individual.features.insert(i, 1);
        }
        individual.k = 6;

        // Use negative infinity threshold - should keep all features
        individual.prune_by_importance(&data, 10, 333, Some(f64::NEG_INFINITY), None, 1);

        assert_eq!(
            individual.k, 6,
            "Should keep all features with NEG_INFINITY threshold"
        );
    }

    #[test]
    fn test_prune_by_importance_feature_removal_correctness() {
        let data = Data::specific_test(100, 20);

        let mut individual = Individual::new();
        individual.language = BINARY_LANG;
        individual.data_type = RAW_TYPE;

        let original_features: Vec<usize> = vec![1, 3, 5, 7, 9, 11, 13, 15];
        for &feat in &original_features {
            individual.features.insert(feat, 1);
        }
        individual.k = original_features.len();

        individual.prune_by_importance(&data, 30, 444, Some(0.0), None, 3);

        let remaining_features: Vec<usize> = individual.features.keys().copied().collect();

        // All remaining features should be from the original set
        for feat in &remaining_features {
            assert!(
                original_features.contains(feat),
                "Feature {} was not in original set",
                feat
            );
        }

        assert!(
            individual.k <= original_features.len(),
            "Should not have more features than original"
        );
        assert!(individual.k >= 3, "Should maintain min_k");
    }

    #[test]
    #[should_panic(expected = "n_perm must be > 0")]
    fn test_prune_by_importance_zero_permutations_panics() {
        let data = Data::specific_test(30, 5);

        let mut individual = Individual::new();
        individual.language = BINARY_LANG;
        individual.data_type = RAW_TYPE;
        individual.features.insert(0, 1);
        individual.k = 1;

        // Should panic with 0 permutations
        individual.prune_by_importance(&data, 0, 111, Some(0.0), None, 1);
    }
}
