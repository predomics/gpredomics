use crate::param::Param;
use crate::utils::{self, serde_json_hashmap_numeric};
use crate::ChaCha8Rng;
use fast_float::parse;
use fishers_exact::fishers_exact;
use log::{debug, info, warn};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use statrs::distribution::Normal; // For random shuffling
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[allow(non_camel_case_types)]
pub enum PreselectionMethod {
    wilcoxon,
    studentt,
    bayesian_fisher,
}

pub struct FeatureAnnotations {
    pub feature_tags: HashMap<usize, Vec<String>>,
    pub prior_weight: HashMap<usize, f64>,
    pub feature_penalty: HashMap<usize, f64>,
}

pub struct SampleAnnotations {
    pub sample_tags: HashMap<usize, Vec<String>>,
    pub samples_subclasses: HashMap<usize, u8>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct Data {
    #[serde(with = "serde_json_hashmap_numeric::tuple_usize_f64")]
    pub X: HashMap<(usize, usize), f64>, // Matrix for feature values
    pub y: Vec<u8>,            // Vector for target values
    pub features: Vec<String>, // Feature names (from the first column of X.tsv)
    pub samples: Vec<String>,
    #[serde(with = "serde_json_hashmap_numeric::usize_u8")]
    pub feature_class: HashMap<usize, u8>, // Sign for each feature
    pub feature_selection: Vec<usize>,
    pub feature_len: usize,
    pub sample_len: usize,
    pub classes: Vec<String>,
    #[serde(default)]
    pub feature_significance: HashMap<usize, f64>,
}

impl Data {
    /// Create a new `Data` instance with default values
    pub fn new() -> Data {
        Data {
            X: HashMap::new(),
            y: Vec::new(),
            features: Vec::new(),
            samples: Vec::new(),
            feature_class: HashMap::new(),
            feature_significance: HashMap::new(),
            feature_selection: Vec::new(),
            feature_len: 0,
            sample_len: 0,
            classes: Vec::new(),
        }
    }

    /// Check if another dataset is compatible with the current one
    pub fn check_compatibility(&self, other: &Data) -> bool {
        self.features == other.features
    }

    /// Load data from `X.tsv` and `y.tsv` files.
    pub fn load_data(
        &mut self,
        X_path: &str,
        y_path: &str,
        features_in_rows: bool,
    ) -> Result<(), Box<dyn Error>> {
        if features_in_rows {
            self.load_data_features_in_rows(X_path, y_path)
        } else {
            self.load_data_features_in_columns(X_path, y_path)
        }
    }

    /// Legacy format (current Gpredomics format): rows=features, columns=samples
    fn load_data_features_in_rows(
        &mut self,
        X_path: &str,
        y_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        #[inline]
        fn trim_line(line: &str) -> &str {
            line.trim_end_matches(['\n', '\r'])
        }

        info!("Loading files {} and {}...", X_path, y_path);

        let file_X = File::open(X_path)?;
        let mut reader_X = BufReader::with_capacity(8 * 1024 * 1024, file_X);

        // Read the first line to get sample names
        let mut first_line = String::new();
        reader_X.read_line(&mut first_line)?;
        let trimmed_first_line = trim_line(&first_line);
        self.samples = trimmed_first_line
            .split('\t')
            .skip(1)
            .map(String::from)
            .collect();

        let file_size = std::fs::metadata(X_path)?.len() as usize;
        let estimated_line_size = 20 + (self.samples.len() * 9);
        let estimated_features = file_size / estimated_line_size.max(50);

        let capacity = ((self.samples.len() * estimated_features) as f64 * 0.25) as usize;
        self.X.reserve(capacity.max(self.samples.len()));

        // Read the remaining lines for feature names and data
        for (feature, line) in reader_X.lines().enumerate() {
            let line = line?;
            let trimmed_line = trim_line(&line);
            let mut fields = trimmed_line.split('\t');

            // First field is the feature name
            if let Some(feature_name) = fields.next() {
                self.features.push(feature_name.to_string());
            }

            // Remaining fields are the feature values
            for (sample, value) in fields.enumerate() {
                if let Ok(num_val) = parse(value) {
                    if num_val != 0.0 {
                        self.X.insert((sample, feature), num_val);
                    }
                }
            }
        }

        // Open and read the y.tsv file
        let file_y = File::open(y_path)?;
        let reader_y = BufReader::new(file_y);

        // Parse y.tsv and store target values
        let mut y_map = HashMap::new();
        for line in reader_y.lines().skip(1) {
            let line = line?;
            let trimmed_line = trim_line(&line);
            let mut fields = trimmed_line.split('\t');

            // First field is the sample name
            if let Some(sample_name) = fields.next() {
                // Second field is the target value
                if let Some(value) = fields.next() {
                    let target: u8 = value.parse()?;
                    y_map.insert(sample_name.to_string(), target);
                }
            }
        }

        // Reorder `y` to match the order of `samples` from X.tsv
        self.y = self
            .samples
            .iter()
            .map(|sample_name| {
                *y_map.get(sample_name).unwrap_or_else(|| {
                    warn!(
                        "No y value available for {}. Setting y to 2 for this sample.",
                        sample_name
                    );
                    &2
                })
            })
            .collect();

        self.feature_len = self.features.len();
        self.sample_len = self.samples.len();

        Ok(())
    }

    /// Standard ML format: rows=samples, columns=features
    fn load_data_features_in_columns(
        &mut self,
        X_path: &str,
        y_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        #[inline]
        fn trim_line(line: &str) -> &str {
            line.trim_end_matches(['\n', '\r'])
        }

        info!("Loading files {} and {}...", X_path, y_path);

        let file_X = File::open(X_path)?;
        let mut reader_X = BufReader::with_capacity(8 * 1024 * 1024, file_X);

        // Read the first line to get feature names
        let mut first_line = String::new();
        reader_X.read_line(&mut first_line)?;
        let trimmed_first_line = trim_line(&first_line);
        self.features = trimmed_first_line
            .split('\t')
            .skip(1)
            .map(String::from)
            .collect();

        let file_size = std::fs::metadata(X_path)?.len() as usize;
        let estimated_line_size = 20 + (self.features.len() * 9);
        let estimated_samples = file_size / estimated_line_size.max(50);

        let capacity = ((self.features.len() * estimated_samples) as f64 * 0.25) as usize;
        self.X.reserve(capacity.max(self.features.len()));

        // Read the remaining lines: each line = one sample
        for (sample, line) in reader_X.lines().enumerate() {
            let line = line?;
            let trimmed_line = trim_line(&line);
            let mut fields = trimmed_line.split('\t');

            // First field is the sample name
            if let Some(sample_name) = fields.next() {
                self.samples.push(sample_name.to_string());
            }

            // Remaining fields are the feature values
            for (feature, value) in fields.enumerate() {
                if let Ok(num_val) = parse(value) {
                    if num_val != 0.0 {
                        self.X.insert((sample, feature), num_val);
                    }
                }
            }
        }

        // Open and read the y.tsv file
        let file_y = File::open(y_path)?;
        let reader_y = BufReader::new(file_y);

        // Parse y.tsv and store target values
        let mut y_map = HashMap::new();
        for line in reader_y.lines().skip(1) {
            let line = line?;
            let trimmed_line = trim_line(&line);
            let mut fields = trimmed_line.split('\t');

            // First field is the sample name
            if let Some(sample_name) = fields.next() {
                // Second field is the target value
                if let Some(value) = fields.next() {
                    let target: u8 = value.parse()?;
                    y_map.insert(sample_name.to_string(), target);
                }
            }
        }

        // Reorder `y` to match the order of `samples` from X.tsv
        self.y = self
            .samples
            .iter()
            .map(|sample_name| {
                *y_map.get(sample_name).unwrap_or_else(|| {
                    warn!(
                        "No y value available for {}. Setting y to 2 for this sample.",
                        sample_name
                    );
                    &2
                })
            })
            .collect();

        self.feature_len = self.features.len();
        self.sample_len = self.samples.len();

        Ok(())
    }

    pub fn set_classes(&mut self, classes: Vec<String>) {
        self.classes = classes;
    }

    pub fn inverse_classes(&mut self) {
        for label in &mut self.y {
            match *label {
                0 => *label = 1,
                1 => *label = 0,
                2 => *label = 2,
                _ => {
                    warn!("Unknown classes : {}. Passed.", *label);
                }
            }
        }

        self.classes.swap(0, 1);

        info!("Classes inverted");
    }

    /// for a given feature (chosen as the #j line of X ) answer 0 if the feature is more significantly associated with class 0, 1 with class 1, 2 otherwise
    /// using student T e.g. for normally ditributed features
    fn compare_classes_studentt(
        &self,
        j: usize,
        max_p_value: f64,
        min_prevalence: f64,
        min_mean_value: f64,
    ) -> (u8, f64) {
        // Separate values into two classes

        let mut count_0: usize = 0;
        let mut count_1: usize = 0;

        let class_0: Vec<f64> = (0..self.sample_len)
            .filter(|i| self.y[*i] == 0)
            .map(|i| {
                if self.X.contains_key(&(i, j)) {
                    count_0 += 1;
                    self.X[&(i, j)]
                } else {
                    0.0
                }
            })
            .collect();

        let class_1: Vec<f64> = (0..self.sample_len)
            .filter(|i| self.y[*i] == 1)
            .map(|i| {
                if self.X.contains_key(&(i, j)) {
                    count_1 += 1;
                    self.X[&(i, j)]
                } else {
                    0.0
                }
            })
            .collect();

        let n0 = class_0.len() as f64;
        let n1 = class_1.len() as f64;

        let prev0 = count_0 as f64 / n0;
        let prev1 = count_1 as f64 / n1;
        if prev0 < min_prevalence && prev1 < min_prevalence {
            return (2_u8, 2.0);
        }

        // Calculate means
        let mean_0 = class_0.iter().copied().sum::<f64>() / class_0.len() as f64;
        let mean_1 = class_1.iter().copied().sum::<f64>() / class_1.len() as f64;

        if mean_0 < min_mean_value && mean_1 < min_mean_value {
            return (2_u8, 2.0);
        }

        // Calculate t-statistic (simple, equal variance assumption)

        let var0 = class_0.iter().map(|x| (x - mean_0).powi(2)).sum::<f64>() / (n0 - 1.0);
        let var1 = class_1.iter().map(|x| (x - mean_1).powi(2)).sum::<f64>() / (n1 - 1.0);

        let pooled_std = (((n0 - 1.0) * var0 + (n1 - 1.0) * var1) / (n0 + n1 - 2.0)
            * (1.0 / n0 + 1.0 / n1))
            .sqrt();
        if pooled_std > 0.0 {
            let t_stat = (mean_0 - mean_1) / pooled_std;

            // Compute p-value
            let degrees_of_freedom = (n0 + n1 - 2.0).round();
            let t_dist = StudentsT::new(0.0, 1.0, degrees_of_freedom).unwrap();
            let cumulative = t_dist.cdf(t_stat.abs()); // CDF up to |t_stat|
            let upper_tail = 1.0 - cumulative; // Upper-tail area
            let p_value = 2.0 * upper_tail; // Two-tailed test

            // Interpretation
            if p_value < max_p_value {
                if mean_0 > mean_1 {
                    (0_u8, p_value)
                } else {
                    (1_u8, p_value)
                }
            } else {
                (2_u8, p_value)
            }
        } else {
            (2_u8, 2.0)
        }
    }

    /// Same as above but using Wilcoxon this time: for a given feature (chosen as the #j line of X ) answer 0 if the feature is more significantly associated with class 0, 1 with class 1, 2 otherwise
    /// using Wilcoxon e.g. for sparse/log normal features
    pub fn compare_classes_wilcoxon(
        &self,
        j: usize,
        max_p_value: f64,
        min_prevalence: f64,
        min_mean_value: f64,
    ) -> (u8, f64) {
        // Separate values into two classes
        let mut count_0: usize = 0;
        let mut count_1: usize = 0;

        let class_0: Vec<f64> = (0..self.sample_len)
            .filter(|i| self.y[*i] == 0)
            .map(|i| {
                if self.X.contains_key(&(i, j)) {
                    count_0 += 1;
                    self.X[&(i, j)]
                } else {
                    0.0
                }
            })
            .collect();

        let class_1: Vec<f64> = (0..self.sample_len)
            .filter(|i| self.y[*i] == 1)
            .map(|i| {
                if self.X.contains_key(&(i, j)) {
                    count_1 += 1;
                    self.X[&(i, j)]
                } else {
                    0.0
                }
            })
            .collect();

        let n0 = class_0.len() as f64;
        let n1 = class_1.len() as f64;

        if n0 == 0.0 || n1 == 0.0 {
            return (2_u8, 2.0);
        }

        let prev0 = count_0 as f64 / n0;
        let prev1 = count_1 as f64 / n1;

        if prev0 < min_prevalence && prev1 < min_prevalence {
            return (2_u8, 2.0);
        }

        // Calculate means
        let mean_0 = class_0.iter().copied().sum::<f64>() / class_0.len() as f64;
        let mean_1 = class_1.iter().copied().sum::<f64>() / class_1.len() as f64;

        if mean_0 < min_mean_value && mean_1 < min_mean_value {
            return (2_u8, 2.0);
        }

        // Combine both classes with their labels
        let mut combined: Vec<(f64, u8)> = class_0
            .iter()
            .map(|&value| (value, 0))
            .chain(class_1.iter().map(|&value| (value, 1)))
            .collect();

        // Sort combined values by value, breaking ties arbitrarily
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Assign ranks
        let mut ranks = vec![0.0; combined.len()];
        let mut i = 0;
        while i < combined.len() {
            let start = i;
            while i + 1 < combined.len() && combined[i].0 == combined[i + 1].0 {
                i += 1;
            }
            let rank = (start + i + 2) as f64 / 2.0;
            for j in start..=i {
                ranks[j] = rank;
            }
            i += 1;
        }

        // Ex-aequo tie correction
        let mut tie_correction = 0.0;
        let mut i = 0;
        while i < combined.len() {
            let start = i;
            while i + 1 < combined.len() && combined[i].0 == combined[i + 1].0 {
                i += 1;
            }
            let tied_count = i - start + 1;
            if tied_count > 1 {
                tie_correction += (tied_count.pow(3) - tied_count) as f64;
            }
            i += 1;
        }

        let std_u = ((n0 * n1 / 12.0)
            * ((n0 + n1 + 1.0) - tie_correction / ((n0 + n1) * (n0 + n1 - 1.0))))
            .sqrt();

        // Compute rank sums
        let rank_sum_0: f64 = combined
            .iter()
            .zip(ranks.iter())
            .filter(|((_, class), _)| *class == 0)
            .map(|(_, &rank)| rank)
            .sum();

        // Compute U statistic
        let u_stat = rank_sum_0 - (n0 * (n0 + 1.0)) / 2.0;

        // Compute p-value using normal approximation
        let mean_u = n0 * n1 / 2.0;
        let diff = u_stat - mean_u;
        let abs_diff_corrected = diff.abs() - 0.5; // Correction for continuous values
        let z = abs_diff_corrected / std_u * diff.signum();

        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - normal_dist.cdf(z.abs())); // Two-tailed p-value
                                                              // Interpretation
        if p_value < max_p_value {
            if mean_0 > mean_1 {
                (0_u8, p_value)
            } else {
                (1_u8, p_value)
            }
        } else {
            (2_u8, p_value)
        }
    }

    fn compare_classes_bayesian_fisher(
        &self,
        j: usize,
        min_absolute_log_factor: f64,
        min_prevalence: f64,
        min_mean_value: f64,
    ) -> (u8, f64) {
        let mut class_0_present: u32 = 0;
        let mut class_0_absent: u32 = 0;
        let mut class_1_present: u32 = 0;
        let mut class_1_absent: u32 = 0;

        let mut class_0_values: Vec<f64> = Vec::new();
        let mut class_1_values: Vec<f64> = Vec::new();

        for i in 0..self.sample_len {
            if self.y[i] == 0 {
                if self.X.contains_key(&(i, j)) && self.X[&(i, j)] >= 0.0 {
                    class_0_present += 1;
                    class_0_values.push(self.X[&(i, j)]);
                } else {
                    class_0_absent += 1;
                }
            } else if self.y[i] == 1 {
                if self.X.contains_key(&(i, j)) && self.X[&(i, j)] >= 0.0 {
                    class_1_present += 1;
                    class_1_values.push(self.X[&(i, j)]);
                } else {
                    class_1_absent += 1;
                }
            }
        }

        let prev0 = class_0_present as f64 / (class_0_present + class_0_absent) as f64;
        let prev1 = class_1_present as f64 / (class_1_present + class_1_absent) as f64;
        if prev0 < min_prevalence && prev1 < min_prevalence {
            return (2_u8, 2.0);
        }

        let mean_0 = class_0_values.iter().sum::<f64>() / class_0_values.len() as f64;
        let mean_1 = class_1_values.iter().sum::<f64>() / class_1_values.len() as f64;

        if mean_0 < min_mean_value && mean_1 < min_mean_value {
            return (2_u8, 0.0);
        }

        let fisher_test = fishers_exact(&[
            class_0_present,
            class_1_present,
            class_0_absent,
            class_1_absent,
        ])
        .unwrap();
        let bayes_factor = fisher_test.greater_pvalue / fisher_test.less_pvalue;

        if bayes_factor.log10().abs() >= min_absolute_log_factor {
            if bayes_factor < 1.0 {
                (0_u8, bayes_factor.log10().abs())
            } else {
                (1_u8, bayes_factor.log10().abs())
            }
        } else {
            (2_u8, 0.0)
        }
    }

    // Dissociate this function from select_features to allow its use in beam algorithm
    pub fn evaluate_features(
        &self,
        param: &Param,
    ) -> (Vec<(usize, u8, f64)>, Vec<(usize, u8, f64)>) {
        let mut results: Vec<(usize, u8, f64)> = (0..self.feature_len)
            .into_par_iter()
            .map(|j| {
                let (class, value) = match param.data.feature_selection_method {
                    PreselectionMethod::studentt => self.compare_classes_studentt(
                        j,
                        param.data.feature_maximal_adj_pvalue,
                        param.data.feature_minimal_prevalence_pct as f64 / 100.0,
                        param.data.feature_minimal_feature_value,
                    ),
                    PreselectionMethod::bayesian_fisher => self.compare_classes_bayesian_fisher(
                        j,
                        param.data.feature_minimal_log_abs_bayes_factor,
                        param.data.feature_minimal_prevalence_pct as f64 / 100.0,
                        param.data.feature_minimal_feature_value,
                    ),
                    _ => self.compare_classes_wilcoxon(
                        j,
                        param.data.feature_maximal_adj_pvalue,
                        param.data.feature_minimal_prevalence_pct as f64 / 100.0,
                        param.data.feature_minimal_feature_value,
                    ),
                };
                (j, class, value)
            })
            .collect();

        if param.data.feature_selection_method == PreselectionMethod::bayesian_fisher {
            results.sort_by(|a, b| match b.2.partial_cmp(&a.2) {
                Some(ordering) => ordering,
                None => std::cmp::Ordering::Equal,
            });
        } else {
            // Keep only tested features and ajust their p-values
            results.retain(|(_, _, value)| *value != 2.0);
            results = self.apply_fdr_correction(results, param.data.feature_maximal_adj_pvalue);

            results.sort_by(|a, b| match a.2.partial_cmp(&b.2) {
                Some(ordering) => ordering,
                None => std::cmp::Ordering::Equal,
            });
        }

        let mut class_0_features: Vec<(usize, u8, f64)> = results
            .iter()
            .cloned()
            .filter(|&(_, class, _)| class == 0)
            .collect();
        let mut class_1_features: Vec<(usize, u8, f64)> = results
            .iter()
            .cloned()
            .filter(|&(_, class, _)| class == 1)
            .collect();
        if param.data.max_features_per_class != 0
            && class_0_features.len() < param.data.max_features_per_class
        {
            warn!("Class {:?} has only {} significant features based on required threshold ! All features kept for this class.", self.classes[0], class_0_features.len());
        } else if param.data.max_features_per_class != 0
            && class_0_features.len() >= param.data.max_features_per_class
        {
            class_0_features.truncate(param.data.max_features_per_class);
        }
        if param.data.max_features_per_class != 0
            && class_1_features.len() < param.data.max_features_per_class
        {
            warn!("Class {:?} has only {} significant features based on required threshold ! All features kept for this class.", self.classes[1], class_1_features.len());
        } else if param.data.max_features_per_class != 0
            && class_1_features.len() >= param.data.max_features_per_class
        {
            class_1_features.truncate(param.data.max_features_per_class);
        }

        (class_0_features, class_1_features)
    }

    /// Fill feature_selection, e.g. a restriction of features based on param (notably pvalue as computed by either studentt or wilcoxon)
    pub fn select_features(&mut self, param: &Param) {
        info!("Selecting features...");

        self.feature_selection = Vec::new();
        self.feature_class = HashMap::new();

        let (class_0_features, class_1_features) = self.evaluate_features(param);

        for (j, class, value) in class_0_features
            .into_iter()
            .chain(class_1_features.into_iter())
        {
            self.feature_class.insert(j, class);
            self.feature_significance.insert(j, value);
            self.feature_selection.push(j);
        }

        assert!( self.feature_selection.len()>0, "No feature has been selected, please lower your selection criteria or improve the quality of your data!");

        info!("{} features selected", self.feature_selection.len());

        if self.feature_len > 10000 {
            warn!("Large dataset. Removing non-selected features from memory to speed up...");

            let mut new_x = HashMap::new();
            for (&(sample, feature), &value) in &self.X {
                if self.feature_selection.contains(&feature) {
                    new_x.insert((sample, feature), value);
                }
            }

            self.X = new_x;

            info!(
                "Non-kept features removed. Dataset compacted from {} to {} features.",
                self.feature_len,
                self.feature_selection.len()
            );
        }
    }

    /// Benjamini-Hochberg FDR correction (Implementation correcte)
    fn apply_fdr_correction(
        &self,
        mut results: Vec<(usize, u8, f64)>,
        fdr_alpha: f64,
    ) -> Vec<(usize, u8, f64)> {
        if results.is_empty() {
            return results;
        }

        assert!(
            fdr_alpha >= 0.0 && fdr_alpha <= 1.0,
            "FDR alpha {} is outside [0,1], clamping to valid range",
            fdr_alpha
        );

        results.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let n = results.len();
        let n_f = n as f64;
        let raw_nominal = results.iter().filter(|r| r.2 <= fdr_alpha).count();

        let mut threshold_idx: Option<usize> = None;
        for i in (0..n).rev() {
            let rank = (i + 1) as f64;
            let threshold = (rank / n_f) * fdr_alpha;
            if results[i].2 <= threshold {
                threshold_idx = Some(i);
                break;
            }
        }

        if let Some(idx) = threshold_idx {
            results.truncate(idx + 1);
        } else {
            results.clear();
        }

        let kept = results.len();
        debug!(
            "BH-FDR ajusted α={:.3}: kept {} / {} features",
            fdr_alpha, kept, raw_nominal
        );

        results
    }

    /// filter Data for some samples (represented by a Vector of indices)
    pub fn subset(&self, samples: Vec<usize>) -> Data {
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        for (new_sample, sample) in samples.iter().enumerate() {
            for feature in 0..self.feature_len {
                if self.X.contains_key(&(*sample, feature)) {
                    X.insert((new_sample, feature), self.X[&(*sample, feature)]);
                }
            }
        }

        Data {
            X: X,
            y: samples.iter().map(|i| self.y[*i]).collect(),
            features: self.features.clone(),
            samples: samples.iter().map(|i| self.samples[*i].clone()).collect(),
            feature_class: HashMap::new(),
            feature_significance: self.feature_significance.clone(),
            feature_selection: self.feature_selection.clone(), // Inherit feature selection from parent
            feature_len: self.feature_len,
            sample_len: samples.len(),
            classes: self.classes.clone(),
        }
    }

    pub fn clone_with_new_x(&self, X: HashMap<(usize, usize), f64>) -> Data {
        Data {
            X: X,
            y: self.y.clone(),
            features: self.features.clone(),
            samples: self.samples.clone(),
            feature_class: self.feature_class.clone(),
            feature_significance: self.feature_significance.clone(),
            feature_selection: self.feature_selection.clone(),
            feature_len: self.feature_len,
            sample_len: self.sample_len,
            classes: self.classes.clone(),
        }
    }

    pub fn add(&mut self, other: &Data) {
        self.samples.extend_from_slice(&other.samples);
        self.y.extend_from_slice(&other.y);
        for j in 0..self.feature_len {
            for i in 0..other.sample_len {
                if other.X.contains_key(&(i, j)) {
                    self.X.insert((i + self.sample_len, j), other.X[&(i, j)]);
                }
            }
        }
        self.sample_len += other.sample_len;
    }

    pub fn remove_class(&mut self, class_to_remove: u8) -> Data {
        let indices_to_keep: Vec<usize> = self
            .y
            .iter()
            .enumerate()
            .filter(|(_, &class)| class != class_to_remove)
            .map(|(index, _)| index)
            .collect();

        warn!("Removing class {:?} samples...", class_to_remove);

        self.subset(indices_to_keep)
    }

    pub fn random_subset(&self, n_samples: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
        // Use stratify_indices_by_class to separate positive and negative samples
        let (indices_class1, indices_class0) = utils::stratify_indices_by_class(&self.y);

        let total_len = self.sample_len;
        let n_class0 = indices_class0.len();
        let n_class1 = indices_class1.len();

        let mut n_samples_0 =
            ((n_samples as f64) * (n_class0 as f64) / (total_len as f64)).round() as usize;
        n_samples_0 = n_samples_0.min(n_class0);
        let mut n_samples_1 = n_samples.saturating_sub(n_samples_0).min(n_class1);

        let current_total = n_samples_0 + n_samples_1;
        let missing = n_samples.saturating_sub(current_total);
        if missing > 0 {
            let available_0 = n_class0 - n_samples_0;
            let available_1 = n_class1 - n_samples_1;
            if available_0 >= missing {
                n_samples_0 += missing;
            } else if available_1 >= missing {
                n_samples_1 += missing;
            } else {
                n_samples_0 += available_0;
                n_samples_1 += available_1;
            }
        }

        let mut rng = rng;
        let chosen_0 = if n_samples_0 > 0 {
            indices_class0
                .choose_multiple(&mut rng, n_samples_0)
                .cloned()
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let chosen_1 = if n_samples_1 > 0 {
            indices_class1
                .choose_multiple(&mut rng, n_samples_1)
                .cloned()
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let mut all_indices = [chosen_0, chosen_1].concat();
        all_indices.shuffle(&mut rng);
        all_indices.truncate(n_samples.min(total_len));

        all_indices
    }
}

/// Implement a custom Debug trait for Data
impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let _ = writeln!(
            f,
            "Features: {}   Samples: {}",
            self.feature_len, self.sample_len
        );

        let samples_string = self.samples.join("\t");
        let truncated_samples = if samples_string.len() > 100 {
            format!("{}...", &samples_string[..97])
        } else {
            samples_string
        };

        writeln!(f, "X:                  {}", truncated_samples)?;
        // Limit to the first 20 rows
        for j in (0..self.feature_len).take(20) {
            let feature = &self.features[j]; // Use the feature name from self.features
            let row_display: String = (0..self.sample_len)
                .map(|i| {
                    if self.X.contains_key(&(i, j)) {
                        format!("{:.2}", self.X[&(i, j)])
                    } else {
                        "".to_string()
                    }
                })
                .collect::<Vec<_>>()
                .join("\t");

            let truncated_row = if row_display.len() > 80 {
                format!("{}...", &row_display[..77])
            } else {
                row_display
            };

            writeln!(f, "{:<20} {}", feature, truncated_row)?;
        }

        writeln!(f, "\ny:")?;
        // Limit y to the first 20 entries
        for (i, sample) in self.y.iter().take(20).enumerate() {
            writeln!(f, "{}\t{:?}", self.samples[i], sample)?;
        }

        Ok(())
    }
}

impl fmt::Debug for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Reuse the Display formatter
        write!(f, "{}", self)
    }
}

// unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::param;
    use rand::Rng;
    use rand::SeedableRng;
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, HashMap};
    use std::fs;
    use std::path::PathBuf;

    impl Data {
        pub fn test() -> Data {
            let mut X: HashMap<(usize, usize), f64> = HashMap::new();
            let mut feature_class: HashMap<usize, u8> = HashMap::new();

            // Simulate data
            X.insert((0, 0), 0.9); // F0 C0
            X.insert((0, 1), 0.01); // F1 C0
            X.insert((1, 1), 0.91); // F1 C1
            X.insert((3, 0), 0.12); // F0 C1
            X.insert((3, 1), 0.75); // F1 C1
            X.insert((4, 0), 0.01); // F0 C1
            X.insert((5, 1), 0.9); // F1 C1
            feature_class.insert(0, 0);
            feature_class.insert(1, 1);
            Data {
                X,
                y: vec![0, 1, 0, 1, 1, 1],
                features: vec!["feature1".to_string(), "feature2".to_string()],
                samples: vec![
                    "sample1".to_string(),
                    "sample2".to_string(),
                    "sample3".to_string(),
                    "sample4".to_string(),
                    "sample5".to_string(),
                    "sample6".to_string(),
                ],
                feature_class,
                feature_significance: HashMap::new(),
                feature_selection: vec![0, 1],
                feature_len: 2,
                sample_len: 6,
                classes: vec!["a".to_string(), "b".to_string()],
            }
        }

        pub fn specific_test(num_samples: usize, num_features: usize) -> Data {
            let mut X = HashMap::new();
            let mut rng = ChaCha8Rng::seed_from_u64(12345);

            // Populate X with (sample, feature) -> value mapping
            for sample in 0..num_samples {
                for feature in 0..num_features {
                    X.insert((sample, feature), rng.gen_range(0.0..1.0));
                }
            }

            let y: Vec<u8> = (0..num_samples)
                .map(|_| if rng.gen::<f64>() > 0.5 { 1 } else { 0 })
                .collect();

            Data {
                X,
                y,
                sample_len: num_samples,
                feature_class: HashMap::new(),
                feature_significance: HashMap::new(),
                features: (0..num_features)
                    .map(|i| format!("feature_{}", i))
                    .collect(),
                samples: (0..num_samples).map(|i| format!("sample_{}", i)).collect(),
                feature_selection: (0..num_features).collect(),
                feature_len: num_features,
                classes: vec!["class_0".to_string(), "class_1".to_string()],
            }
        }

        pub fn test_with_these_features(feature_indices: &[usize]) -> Data {
            let mut data = Data::new();
            data.feature_len = feature_indices.len();
            data.sample_len = 5;

            for &feature_idx in feature_indices {
                data.feature_class.insert(feature_idx, 1u8);
                for sample_idx in 0..data.sample_len {
                    data.X
                        .insert((sample_idx, feature_idx), 0.5 + feature_idx as f64 * 0.1);
                }
            }

            data.y = vec![0, 1, 0, 1, 0];
            data.feature_selection = feature_indices.to_vec();
            data
        }

        pub fn test2() -> Data {
            let mut X: HashMap<(usize, usize), f64> = HashMap::new();
            let mut feature_class: HashMap<usize, u8> = HashMap::new();

            // Simulate data
            X.insert((0, 0), 0.9); // Sample 0, Feature 0
            X.insert((0, 1), 0.01); // Sample 0, Feature 1
            X.insert((1, 0), 0.91); // Sample 1, Feature 0
            X.insert((1, 1), 0.12); // Sample 1, Feature 1
            X.insert((2, 0), 0.75); // Sample 2, Feature 0
            X.insert((2, 1), 0.01); // Sample 2, Feature 1
            X.insert((3, 0), 0.19); // Sample 3, Feature 0
            X.insert((3, 1), 0.92); // Sample 3, Feature 1
            X.insert((4, 0), 0.9); // Sample 4, Feature 0
            X.insert((4, 1), 0.01); // Sample 4, Feature 1
            X.insert((5, 0), 0.91); // Sample 5, Feature 0
            X.insert((5, 1), 0.12); // Sample 5, Feature 1
            X.insert((6, 0), 0.75); // Sample 6, Feature 0
            X.insert((6, 1), 0.01); // Sample 6, Feature 1
            X.insert((7, 0), 0.19); // Sample 7, Feature 0
            X.insert((7, 1), 0.92); // Sample 7, Feature 1
            X.insert((8, 0), 0.9); // Sample 8, Feature 0
            X.insert((8, 1), 0.01); // Sample 8, Feature 1
            X.insert((9, 0), 0.91); // Sample 9, Feature 0
            X.insert((9, 1), 0.12); // Sample 9, Feature 1
            feature_class.insert(0, 0);
            feature_class.insert(1, 1);

            Data {
                X,
                y: vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0], // Vraies étiquettes
                features: vec!["feature1".to_string(), "feature2".to_string()],
                samples: vec![
                    "sample1".to_string(),
                    "sample2".to_string(),
                    "sample3".to_string(),
                    "sample4".to_string(),
                    "sample5".to_string(),
                    "sample6".to_string(),
                    "sample7".to_string(),
                    "sample8".to_string(),
                    "sample9".to_string(),
                    "sample10".to_string(),
                ],
                feature_class,
                feature_significance: HashMap::new(),
                feature_selection: vec![0, 1],
                feature_len: 2,
                sample_len: 10,
                classes: vec!["a".to_string(), "b".to_string()],
            }
        }

        pub fn test_disc_data() -> Data {
            let mut X: HashMap<(usize, usize), f64> = HashMap::new();
            let mut feature_class: HashMap<usize, u8> = HashMap::new();

            // Simulate data
            X.insert((0, 0), 0.9); // Sample 0, Feature 0
            X.insert((0, 1), 0.01); // Sample 0, Feature 1
            X.insert((1, 0), 0.91); // Sample 1, Feature 0
            X.insert((1, 1), 0.12); // Sample 1, Feature 1
            X.insert((2, 0), 0.75); // Sample 2, Feature 0
            X.insert((2, 1), 0.01); // Sample 2, Feature 1
            X.insert((3, 0), 0.19); // Sample 3, Feature 0
            X.insert((3, 1), 0.92); // Sample 3, Feature 1
            X.insert((4, 0), 0.9); // Sample 4, Feature 0
            X.insert((4, 1), 0.01); // Sample 4, Feature 1
            feature_class.insert(0, 0);
            feature_class.insert(1, 1);

            Data {
                X,
                y: vec![1, 0, 1, 0, 0], // Vraies étiquettes
                features: vec!["feature1".to_string(), "feature2".to_string()],
                samples: vec![
                    "sample1".to_string(),
                    "sample2".to_string(),
                    "sample3".to_string(),
                    "sample4".to_string(),
                    "sample5".to_string(),
                ],
                feature_class,
                feature_significance: HashMap::new(),
                feature_selection: vec![0, 1],
                feature_len: 2,
                sample_len: 5,
                classes: vec!["a".to_string(), "b".to_string()],
            }
        }

        pub fn test_valid_data() -> Data {
            let mut X: HashMap<(usize, usize), f64> = HashMap::new();
            let mut feature_class: HashMap<usize, u8> = HashMap::new();

            // Simulate data
            X.insert((5, 0), 0.91); // Sample 5, Feature 0
            X.insert((5, 1), 0.12); // Sample 5, Feature 1
            X.insert((6, 0), 0.75); // Sample 6, Feature 0
            X.insert((6, 1), 0.01); // Sample 6, Feature 1
            X.insert((7, 0), 0.19); // Sample 7, Feature 0
            X.insert((7, 1), 0.92); // Sample 7, Feature 1
            X.insert((8, 0), 0.9); // Sample 8, Feature 0
            X.insert((8, 1), 0.01); // Sample 8, Feature 1
            X.insert((9, 0), 0.91); // Sample 9, Feature 0
            X.insert((9, 1), 0.12); // Sample 9, Feature 1
            feature_class.insert(0, 0);
            feature_class.insert(1, 1);

            Data {
                X,
                y: vec![0, 0, 0, 1, 0], // Vraies étiquettes
                features: vec!["feature1".to_string(), "feature2".to_string()],
                samples: vec![
                    "sample6".to_string(),
                    "sample7".to_string(),
                    "sample8".to_string(),
                    "sample9".to_string(),
                    "sample10".to_string(),
                ],
                feature_class,
                feature_significance: HashMap::new(),
                feature_selection: vec![0, 1],
                feature_len: 2,
                sample_len: 5,
                classes: vec!["a".to_string(), "b".to_string()],
            }
        }
    }

    #[test]
    fn test_load_data() {
        let mut data_test = Data::new();
        let _err = data_test.load_data("./samples/tests/X.tsv", "./samples/tests/y.tsv", true);

        // Use the hashed test.X to make the code cleaner
        let mut sorted_X: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for (key, value) in data_test.X {
            sorted_X.insert(key, value);
        }
        let serialized = bincode::serialize(&sorted_X).unwrap();
        let mut hasher = Sha256::new();
        hasher.update(serialized);
        let hash = hasher.finalize();

        assert_eq!(format!("{:x}", hash), "adba327f62ffab0a8d43c1aa3a6c20e630783d3b103dd103f28b9e23ab51eb18", 
        "the test X hash isn't the same as generated in the past, indicating a reproducibility problem linked either to the load_data function or to the modification of ./tests/X.tsv");
        assert_eq!(data_test.y, [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        "the test y are not the same as generated in the past, indicating a reproducibility problem linked either to the load_data function or to the modification of ./tests/y.tsv");
        assert_eq!(data_test.features, ["msp_0001", "msp_0002", "msp_0003", "msp_0004", "msp_0005", "msp_0006", "msp_0007", "msp_0008", "msp_0009", "msp_0010"],
        "the test X features isn't the same as generated in the past, indicating a reproducibility problem linked either to the load_data function or to the modification of ./tests/X.tsv");
        assert_eq!(data_test.samples, ["LV15", "HV26", "HV17", "LV9", "HV14", "HV7", "LV7", "HV30", "HV29", "HV15", "LV16", "LV1", "LV11", "HV16", "HV6", "LV23", 
        "LV4", "LV22", "HV3", "LV13", "LV8", "LV24", "HV12", "HV13", "LV6", "HV9", "LV12", "HV23", "HV31", "LV14"],
        "the test X samples are not the same as generated in the past, indicating a reproducibility problem linked either to the load_data function or to the modification of ./tests/X.tsv");
        assert_eq!(data_test.feature_len, 10,
        "the test X feature_len isn't the same as generated in the past, indicating a reproducibility problem linked either to the load_data function or to the modification of ./tests/X.tsv");
        assert_eq!(data_test.sample_len, 30,
        "the test X sample_len isn't the same as generated in the past, indicating a reproducibility problem linked either to the load_data function or to the modification of ./tests/X.tsv");
    }

    #[test]
    fn test_compare_classes_studentt_test_class_0() {
        let data = Data::test();
        let result = data.compare_classes_studentt(0, 1.0, 0.0, 0.0);
        assert_eq!(
            result,
            (0_u8, 0.20893746598423224),
            "test feature 0 should be significantly associated with class 0"
        );
    }

    #[test]
    fn test_compare_classes_studentt_test_class_1() {
        let data = Data::test();
        let result = data.compare_classes_studentt(1, 1.0, 0.0, 0.0);
        assert_eq!(
            result,
            (1_u8, 0.12215172301873412),
            "test feature 1 should be significantly associated with class 1"
        );
    }

    #[test]
    fn test_compare_classes_studentt_test_class_2_low_mean() {
        let data = Data::test();
        let result = data.compare_classes_studentt(1, 1.0, 0.0, 0.95);
        assert_eq!(
            result,
            (2, 2.0),
            "test feature 1 should not be associated (class 2 instead of 1) : min_mean_value<0.95"
        );
    }

    #[test]
    fn test_compare_classes_studentt_test_class_2_low_prev() {
        let data = Data::test();
        let result = data.compare_classes_studentt(1, 1.0, 0.95, 0.0);
        assert_eq!(
            result,
            (2, 2.0),
            "test feature 1 should not be associated (class 2 instead of 1) : min_prevalence<0.95"
        );
    }

    #[test]
    fn test_compare_classes_studentt_test_class_2_high_pval() {
        let data = Data::test();
        let result = data.compare_classes_studentt(1, 0.00001, 0.0, 0.0);
        assert_eq!(
            result,
            (2, 0.12215172301873412),
            "test feature 1 should not be associated (class 2 instead of 1) : p_value>max_p_value"
        );
    }

    //#[test]
    //fn test_compare_classes_studentt_test_class_2_outside_range() {
    //    let data = create_test_data();
    //    let result = data.compare_classes_studentt(48152342, 1.0, 0.0, 0.0);
    //    assert_eq!(result, 2, "unexistent feature should not be associated (class 2)");
    //}

    // Same for Wilcoxon
    #[test]
    fn test_compare_classes_wilcoxon_class_0() {
        let data = Data::test();
        let result = data.compare_classes_wilcoxon(0, 1.0, 0.0, 0.0);
        assert_eq!(
            result,
            (0_u8, 0.8057327908484386),
            "test feature 0 should be significantly associated with class 0"
        );
    }

    #[test]
    fn test_compare_classes_wilcoxon_class_1() {
        let data = Data::test();
        let result = data.compare_classes_wilcoxon(1, 1.0, 0.0, 0.0);
        assert_eq!(
            result,
            (1_u8, 0.3475580367718807),
            "test feature 1 should be significantly associated with class 1"
        );
    }

    #[test]
    fn test_compare_classes_wilcoxon_class_2_low_mean() {
        let data = Data::test();
        let result = data.compare_classes_wilcoxon(1, 1.0, 0.0, 0.95);
        assert_eq!(
            result,
            (2, 2.0),
            "test feature 1 should not be associated (class 2 instead of 1) : min_mean_value<0.95"
        );
    }

    #[test]
    fn test_compare_classes_wilcoxon_class_2_low_prev() {
        let data = Data::test();
        let result = data.compare_classes_wilcoxon(1, 1.0, 0.95, 0.0);
        assert_eq!(
            result,
            (2, 2.0),
            "test feature 1 should not be associated (class 2 instead of 1) : min_prevalence<0.95"
        );
    }

    #[test]
    fn test_compare_classes_wilcoxon_class_2_high_pval() {
        let data = Data::test();
        let result = data.compare_classes_wilcoxon(1, 0.00001, 0.0, 0.0);
        assert_eq!(
            result,
            (2, 0.3475580367718807),
            "test feature 1 should not be associated (class 2 instead of 1) : p_value>max_p_value"
        );
    }

    //#[test]
    //fn test_compare_classes_wilcoxon_class_2_outside_range() {
    //    let data = create_test_data();
    //    let result = data.compare_classes_wilcoxon(48152342, 1.0, 0.0, 0.0);
    //    assert_eq!(result, 2, "unexistent feature should not be associated (class 2)");
    //}

    // tests for select_features
    #[test]
    fn test_select_features() {
        let mut data = Data::new();
        let _ = data.load_data(
            "samples/Qin2014/Xtrain.tsv",
            "samples/Qin2014/Ytrain.tsv",
            true,
        );
        let mut param = param::get("param.yaml".to_string()).unwrap();
        param.data.max_features_per_class = 0;

        // Test with Bayesian Fisher
        param.data.feature_selection_method = PreselectionMethod::bayesian_fisher;
        param.data.feature_minimal_log_abs_bayes_factor = 2.0;

        param.data.feature_minimal_prevalence_pct = 0.0;
        data.select_features(&param);
        assert_eq!(data.feature_selection.len(), 316, "The bayesian method should identify 316 significant features for feature_minimal_log_abs_bayes_factor=2");

        // Test reduced number
        param.data.max_features_per_class = 1;
        data.select_features(&param);
        assert_eq!(data.feature_selection.len(), 2, "Incorrect number of selected features : select only one feature per class should lead to 2 selected features");
        assert_eq!(data.feature_selection, vec![473, 1313], "Incorrect selected features : the result differs from the past and from the original script result.");

        // Test with Studentt
        param.data.feature_selection_method = PreselectionMethod::studentt;
        param.data.max_features_per_class = 0;
        param.data.feature_maximal_adj_pvalue = 0.05;
        param.data.feature_minimal_feature_value = 0.0001;
        param.data.feature_minimal_prevalence_pct = 0.0;

        data.select_features(&param);
        assert_eq!(data.feature_selection.len(), 112, "The student-based preselection method should identify 112 significant features for feature_maximal_adj_pvalue=0.05");

        // Test with Wilcoxon
        param.data.feature_selection_method = PreselectionMethod::wilcoxon;
        param.data.max_features_per_class = 0;
        param.data.feature_maximal_adj_pvalue = 0.05;
        param.data.feature_minimal_feature_value = 0.0001;
        param.data.feature_minimal_prevalence_pct = 0.0;

        data.select_features(&param);
        assert_eq!(data.feature_selection.len(), 303, "The wilcoxon-based preselection should identify 348 significant features for feature_maximal_adj_pvalue=0.05");
    }

    // tests for subset
    #[test]
    fn test_subset_indices() {
        let original_data = Data::test();
        let subset_data = original_data.subset(vec![0, 3]);

        assert_eq!(
            subset_data.X,
            HashMap::from([
                ((0, 0), 0.9),
                ((0, 1), 0.01),
                ((1, 0), 0.12),
                ((1, 1), 0.75)
            ]),
            "the subset X should be composed of the selected-samples X"
        );
        assert_eq!(
            subset_data.y,
            vec![0, 1],
            "the subset y should be composed of the selected-samples y"
        );
        assert_eq!(
            subset_data.samples,
            vec!["sample1".to_string(), "sample4".to_string()],
            "the subset samples names should be the selected-samples names"
        );
        assert_eq!(
            subset_data.feature_len, original_data.feature_len,
            "the subset feature_len should be the selected-samples feature_len"
        );
        assert_eq!(
            subset_data.sample_len, 2,
            "the subset sample_len should be the number of samples used to reduce the data"
        );
    }

    #[test]
    fn test_subset_empty_set() {
        let original_data = Data::test();
        let subset_data = original_data.subset(vec![]);
        let expected_X: HashMap<(usize, usize), f64> = HashMap::new();
        let expected_y: Vec<u8> = vec![];
        let expected_samples: Vec<String> = vec![];

        assert_eq!(
            subset_data.X, expected_X,
            "an empty subset should have empty X"
        );
        assert_eq!(
            subset_data.y, expected_y,
            "an empty subset should have empty y"
        );
        assert_eq!(
            subset_data.samples, expected_samples,
            "an empty subset shouldn't have samples"
        );
        assert_eq!(
            subset_data.feature_len, original_data.feature_len,
            "an empty subset should keep its reference to features"
        );
        assert_eq!(
            subset_data.sample_len, 0,
            "an empty subset should have 0 sample"
        );
    }

    // tests for clone_with_new_x
    #[test]
    fn test_clone_with_new_x_basic() {
        let original_data = Data::test();

        let new_x = HashMap::from([((0, 0), 0.5), ((1, 0), 0.8)]);
        let cloned_data = original_data.clone_with_new_x(new_x.clone());

        assert_eq!(cloned_data.X, new_x, "the clone must have the new X");
        assert_eq!(
            cloned_data.y, original_data.y,
            "the clone must have the same y as the reference"
        );
        assert_eq!(
            cloned_data.features, original_data.features,
            "the clone must have the same features as the reference"
        );
        assert_eq!(
            cloned_data.samples, original_data.samples,
            "the clone must have the same samples as the reference"
        );
        assert_eq!(
            cloned_data.feature_class, original_data.feature_class,
            "the clone must have the same feature_class as the reference"
        );
        assert_eq!(
            cloned_data.feature_selection, original_data.feature_selection,
            "the clone must have the same feature_selection as the reference"
        );
        assert_eq!(
            cloned_data.feature_len, original_data.feature_len,
            "the clone must have the same feature_len as the reference"
        );
        assert_eq!(
            cloned_data.sample_len, original_data.sample_len,
            "the clone must have the same sample_len as the reference"
        );
    }

    #[test]
    fn test_clone_with_new_x_empty() {
        let original_data = Data::test();

        let new_x: HashMap<(usize, usize), f64> = HashMap::new();
        let cloned_data = original_data.clone_with_new_x(new_x);

        assert!(
            cloned_data.X.is_empty(),
            "the clone must have the new X despite its emptiness"
        );
        assert_eq!(
            cloned_data.y, original_data.y,
            "the clone must have the same y as the reference"
        );
        assert_eq!(
            cloned_data.features, original_data.features,
            "the clone must have the same features as the reference"
        );
        assert_eq!(
            cloned_data.samples, original_data.samples,
            "the clone must have the same samples as the reference"
        );
        assert_eq!(
            cloned_data.feature_class, original_data.feature_class,
            "the clone must have the same feature_class as the reference"
        );
        assert_eq!(
            cloned_data.feature_selection, original_data.feature_selection,
            "the clone must have the same feature_selection as the reference"
        );
        assert_eq!(
            cloned_data.feature_len, original_data.feature_len,
            "the clone must have the same feature_len as the reference"
        );
        assert_eq!(
            cloned_data.sample_len, original_data.sample_len,
            "the clone must have the same sample_len as the reference"
        );
    }

    // tests for add
    #[test]
    fn test_add_basic() {
        let mut data1 = Data {
            X: HashMap::from([((0, 0), 0.5), ((1, 0), 0.8)]),
            y: vec![0, 1],
            features: vec!["feature1".to_string()],
            samples: vec!["sample1".to_string(), "sample2".to_string()],
            feature_class: HashMap::new(),
            feature_significance: HashMap::new(),
            feature_selection: Vec::new(),
            feature_len: 1,
            sample_len: 2,
            classes: vec!["a".to_string(), "b".to_string()],
        };

        let data2 = Data {
            X: HashMap::from([((0, 0), 0.3), ((1, 0), 0.6)]),
            y: vec![1, 0],
            features: vec!["feature1".to_string()],
            samples: vec!["sample3".to_string(), "sample4".to_string()],
            feature_class: HashMap::new(),
            feature_significance: HashMap::new(),
            feature_selection: Vec::new(),
            feature_len: 1,
            sample_len: 2,
            classes: vec!["a".to_string(), "b".to_string()],
        };

        data1.add(&data2);

        let expected_X: HashMap<(usize, usize), f64> =
            HashMap::from([((0, 0), 0.5), ((1, 0), 0.8), ((2, 0), 0.3), ((3, 0), 0.6)]);
        let expected_y = vec![0, 1, 1, 0];
        let expected_samples = vec![
            "sample1".to_string(),
            "sample2".to_string(),
            "sample3".to_string(),
            "sample4".to_string(),
        ];

        assert_eq!(
            data1.X, expected_X,
            "the combination of two Data must notably result in the combinaition of their X"
        );
        assert_eq!(
            data1.y, expected_y,
            "the combination of two Data must notably result in the combinaition of their y"
        );
        assert_eq!(
            data1.samples, expected_samples,
            "the combination of two Data must notably result in the combinaition of their samples"
        );
        assert_eq!(
            data1.sample_len, 4,
            "the combination of two Data must notably result in the sum of their sample_len"
        );
    }

    #[test]
    fn test_add_empty_data() {
        let original_data = Data::test();
        let mut cumulated_data = original_data.clone();
        let empty_data = Data::new();

        cumulated_data.add(&empty_data);

        assert_eq!(
            original_data.X, cumulated_data.X,
            "the combination of a Data with an empty Data must contain the X of the non-empty Data"
        );
        assert_eq!(
            original_data.y, cumulated_data.y,
            "the combination of a Data with an empty Data must contain the y of the non-empty Data"
        );
        assert_eq!(original_data.features, cumulated_data.features, "the combination of a Data with an empty Data must contain the features of the non-empty Data");
        assert_eq!(original_data.feature_class, cumulated_data.feature_class, "the combination of a Data with an empty Data must contain the feature_class of the non-empty Data");
        assert_eq!(original_data.feature_len, cumulated_data.feature_len, "the combination of a Data with an empty Data must contain the feature_len of the non-empty Data");
        assert_eq!(original_data.samples, cumulated_data.samples, "the combination of a Data with an empty Data must contain the samples of the non-empty Data");
        assert_eq!(original_data.sample_len, cumulated_data.sample_len, "the combination of a Data with an empty Data must contain the sample_len of the non-empty Data");
    }

    fn create_test_files(suffix: &str) -> (PathBuf, PathBuf) {
        let temp_dir = std::env::temp_dir();
        let x_path = temp_dir.join(format!("gpredomics_test_X_{}.tsv", suffix));
        let y_path = temp_dir.join(format!("gpredomics_test_y_{}.tsv", suffix));
        (x_path, y_path)
    }

    /// Helper to cleanup test files
    fn cleanup_test_files(x_path: &PathBuf, y_path: &PathBuf) {
        let _ = fs::remove_file(x_path);
        let _ = fs::remove_file(y_path);
    }
    // a few ways to make add() more robust:
    // fn test_add_same_samples() {} -> case where data1 contains samples 1 & 2 and data2 samples 1 & 3
    // fn test_add_different_features() {} -> case where data1 contains feature X but data2 doesn't

    #[test]
    fn test_data_compatibility() {
        let mut data_test = Data::test();
        let data_test2 = Data::test();
        assert!(
            data_test.check_compatibility(&data_test2),
            "two identical data should be compatible"
        );

        data_test.features[1] = "some other name".to_string();
        assert!(
            !data_test.check_compatibility(&data_test2),
            "two data with different features should not be compatible"
        );
    }

    #[test]
    fn test_load_standard_format_qin2014() {
        let (x_path, y_path) = create_test_files("standard_001");

        let x_content = "SampleID\tFeature1\tFeature2\tFeature3\nSample1\t0.5\t1.0\t0.2\nSample2\t0.0\t2.0\t0.0\nSample3\t0.3\t1.5\t0.8\n";
        fs::write(&x_path, x_content).unwrap();

        let y_content = "SampleID\tLabel\nSample1\t0\nSample2\t1\nSample3\t1\n";
        fs::write(&y_path, y_content).unwrap();

        let mut data = Data::new();
        data.load_data(x_path.to_str().unwrap(), y_path.to_str().unwrap(), false)
            .unwrap();

        assert_eq!(data.sample_len, 3);
        assert_eq!(data.feature_len, 3);
        assert_eq!(data.samples, vec!["Sample1", "Sample2", "Sample3"]);
        assert_eq!(data.features, vec!["Feature1", "Feature2", "Feature3"]);
        assert_eq!(data.y, vec![0, 1, 1]);
        assert_eq!(data.X.get(&(0, 0)), Some(&0.5));
        assert_eq!(data.X.get(&(1, 0)), None);
        assert_eq!(data.X.get(&(2, 1)), Some(&1.5));

        cleanup_test_files(&x_path, &y_path);
    }

    #[test]
    fn test_qin2014_legacy_vs_standard_equivalence() {
        let (x_legacy, y_legacy) = create_test_files("legacy_equiv");
        let (x_standard, y_standard) = create_test_files("standard_equiv");

        let x_legacy_content = "FeatureID\tSample1\tSample2\tSample3\nFeature1\t0.5\t0.0\t0.3\nFeature2\t1.0\t2.0\t1.5\nFeature3\t0.2\t0.0\t0.8\n";
        fs::write(&x_legacy, x_legacy_content).unwrap();

        let x_standard_content = "SampleID\tFeature1\tFeature2\tFeature3\nSample1\t0.5\t1.0\t0.2\nSample2\t0.0\t2.0\t0.0\nSample3\t0.3\t1.5\t0.8\n";
        fs::write(&x_standard, x_standard_content).unwrap();

        let y_content = "SampleID\tLabel\nSample1\t0\nSample2\t1\nSample3\t1\n";
        fs::write(&y_legacy, y_content).unwrap();
        fs::write(&y_standard, y_content).unwrap();

        let mut data_legacy = Data::new();
        data_legacy
            .load_data(x_legacy.to_str().unwrap(), y_legacy.to_str().unwrap(), true)
            .unwrap();

        let mut data_standard = Data::new();
        data_standard
            .load_data(
                x_standard.to_str().unwrap(),
                y_standard.to_str().unwrap(),
                false,
            )
            .unwrap();

        assert_eq!(data_legacy.X, data_standard.X);
        assert_eq!(data_legacy.y, data_standard.y);
        assert_eq!(data_legacy.features, data_standard.features);
        assert_eq!(data_legacy.samples, data_standard.samples);
        assert_eq!(data_legacy.feature_len, data_standard.feature_len);
        assert_eq!(data_legacy.sample_len, data_standard.sample_len);

        cleanup_test_files(&x_legacy, &y_legacy);
        cleanup_test_files(&x_standard, &y_standard);
    }

    #[test]
    fn test_sparse_zeros_not_stored() {
        let (x_path, y_path) = create_test_files("sparse_001");

        let x_content = "FeatureID\tSample1\tSample2\tSample3\nFeature1\t0.5\t0.0\t0.3\nFeature2\t1.0\t2.0\t1.5\nFeature3\t0.2\t0.0\t0.8\n";
        fs::write(&x_path, x_content).unwrap();

        let y_content = "SampleID\tLabel\nSample1\t0\nSample2\t1\nSample3\t1\n";
        fs::write(&y_path, y_content).unwrap();

        let mut data = Data::new();
        data.load_data(x_path.to_str().unwrap(), y_path.to_str().unwrap(), true)
            .unwrap();

        assert_eq!(data.X.get(&(1, 0)), None);
        assert_eq!(data.X.get(&(1, 2)), None);
        assert_eq!(data.X.get(&(0, 0)), Some(&0.5));
        assert_eq!(data.X.len(), 7);

        cleanup_test_files(&x_path, &y_path);
    }

    #[test]
    fn test_y_reordering_with_unordered_input() {
        let (x_path, y_path) = create_test_files("reorder_001");

        let x_content = "FeatureID\tSample1\tSample2\tSample3\nFeature1\t0.5\t0.0\t0.3\n";
        fs::write(&x_path, x_content).unwrap();

        let y_content = "SampleID\tLabel\nSample3\t1\nSample1\t0\nSample2\t1\n";
        fs::write(&y_path, y_content).unwrap();

        let mut data = Data::new();
        data.load_data(x_path.to_str().unwrap(), y_path.to_str().unwrap(), true)
            .unwrap();

        assert_eq!(data.y, vec![0, 1, 1]);
        assert_eq!(data.samples, vec!["Sample1", "Sample2", "Sample3"]);

        cleanup_test_files(&x_path, &y_path);
    }

    #[test]
    fn test_consistency_multiple_loads() {
        let (x_path, y_path) = create_test_files("consistency_001");

        let x_content = "FeatureID\tSample1\tSample2\tSample3\nFeature1\t0.5\t0.0\t0.3\nFeature2\t1.0\t2.0\t1.5\n";
        fs::write(&x_path, x_content).unwrap();

        let y_content = "SampleID\tLabel\nSample1\t0\nSample2\t1\nSample3\t1\n";
        fs::write(&y_path, y_content).unwrap();

        let mut data1 = Data::new();
        data1
            .load_data(x_path.to_str().unwrap(), y_path.to_str().unwrap(), true)
            .unwrap();

        let mut data2 = Data::new();
        data2
            .load_data(x_path.to_str().unwrap(), y_path.to_str().unwrap(), true)
            .unwrap();

        assert_eq!(data1, data2);

        cleanup_test_files(&x_path, &y_path);
    }

    #[test]
    fn test_default_features_in_rows_false() {
        let param = Param::new();
        assert_eq!(param.data.features_in_rows, false);
    }

    #[test]
    fn test_fdr_correction_empty() {
        let data = Data::new();
        let results = vec![];
        let corrected = data.apply_fdr_correction(results, 0.05);
        assert_eq!(corrected.len(), 0);
    }

    #[test]
    fn test_fdr_correction_single_feature() {
        let data = Data::new();
        let results = vec![(0, 0u8, 0.03)];
        let corrected = data.apply_fdr_correction(results, 0.05);
        // Single feature: threshold = (1/1) * 0.05 = 0.05
        // p=0.03 <= 0.05 → kept
        assert_eq!(corrected.len(), 1);
        assert_eq!(corrected[0].2, 0.03);
    }

    #[test]
    fn test_fdr_correction_all_pass() {
        let data = Data::new();
        let results = vec![(0, 0u8, 0.001), (1, 0u8, 0.002), (2, 0u8, 0.003)];
        let corrected = data.apply_fdr_correction(results, 0.05);
        // Rank 1: threshold = (1/3) * 0.05 = 0.0167, p=0.001 <= 0.0167 ✓
        // Rank 2: threshold = (2/3) * 0.05 = 0.0333, p=0.002 <= 0.0333 ✓
        // Rank 3: threshold = (3/3) * 0.05 = 0.05, p=0.003 <= 0.05 ✓
        assert_eq!(corrected.len(), 3);
    }

    #[test]
    fn test_fdr_correction_partial_rejection() {
        let data = Data::new();
        let results = vec![
            (0, 0u8, 0.001), // Rank 1: threshold = (1/5)*0.05 = 0.01
            (1, 0u8, 0.005), // Rank 2: threshold = (2/5)*0.05 = 0.02
            (2, 0u8, 0.015), // Rank 3: threshold = (3/5)*0.05 = 0.03
            (3, 0u8, 0.04),  // Rank 4: threshold = (4/5)*0.05 = 0.04
            (4, 0u8, 0.10),  // Rank 5: threshold = (5/5)*0.05 = 0.05
        ];
        let corrected = data.apply_fdr_correction(results, 0.05);
        assert_eq!(corrected.len(), 4); // Rank i = Rank 4 -> Threshold = 0.04 -> only 0.10 rejected
    }

    #[test]
    fn test_fdr_correction_realistic_omics() {
        // Simulation: 100 features, 5 significant (true positives)
        // + 95 noise features with random p-values
        let data = Data::new();
        let mut results = vec![];

        // Add 5 truly significant features (low p-values)
        for i in 0..5 {
            results.push((i, 0u8, 0.001 + (i as f64) * 0.0001));
        }

        // Add 95 noise features with uniform random p-values (0.05 to 1.0)
        for i in 5..100 {
            let p_value = 0.05 + ((i as f64) * 0.0095); // 0.05 to 0.95
            results.push((i, 0u8, p_value));
        }

        let corrected = data.apply_fdr_correction(results.clone(), 0.05);

        // Beyond rank ~5: threshold grows, but p-values are too high

        // We expect to keep only the truly significant ones
        assert!(
            corrected.len() <= 10,
            "Should be conservative with alpha=0.05"
        );
        assert!(corrected.len() >= 1, "Should keep at least top features");

        // Verify all kept features have low p-values
        for (_, _, p_val) in &corrected {
            assert!(*p_val <= 0.01, "Kept features should have p-value <= 0.01");
        }
    }

    #[test]
    fn test_fdr_correction_sorted_independently() {
        let data = Data::new();
        // Input NOT sorted
        let results = vec![
            (0, 0u8, 0.10),
            (1, 0u8, 0.001),
            (2, 0u8, 0.05),
            (3, 0u8, 0.02),
        ];
        let corrected = data.apply_fdr_correction(results, 0.05);

        // Should sort: 0.001, 0.02, 0.05, 0.10
        // Rank 1: 0.001 <= (1/4)*0.05 = 0.0125 ✓
        // Rank 2: 0.02 <= (2/4)*0.05 = 0.025 ✓
        // Rank 3: 0.05 <= (3/4)*0.05 = 0.0375 ✗
        // Rank 4: 0.10 <= (4/4)*0.05 = 0.05 ✗

        assert_eq!(corrected.len(), 2);
    }

    #[test]
    fn test_fdr_correction_alpha_levels() {
        let data = Data::new();
        let results = vec![
            (0, 0u8, 0.001),
            (1, 0u8, 0.01),
            (2, 0u8, 0.05),
            (3, 0u8, 0.10),
        ];

        // Alpha = 0.05
        let corrected_05 = data.apply_fdr_correction(results.clone(), 0.05);

        // Alpha = 0.10 (more permissive)
        let corrected_10 = data.apply_fdr_correction(results.clone(), 0.10);

        // More permissive alpha should keep more features
        assert!(corrected_10.len() >= corrected_05.len());
    }

    #[test]
    fn test_fdr_correction_preserves_feature_ids() {
        let data = Data::new();
        let results = vec![(42, 0u8, 0.001), (100, 0u8, 0.02), (7, 0u8, 0.10)];
        let corrected = data.apply_fdr_correction(results, 0.05);

        // Feature IDs should be preserved
        let ids: Vec<usize> = corrected.iter().map(|(id, _, _)| *id).collect();
        assert!(ids.contains(&42));
    }

    #[test]
    fn test_fdr_correction_alpha_zero() {
        // BH with alpha=0: thresholds (i/n)*0 = 0, so only p==0 would pass.
        // Here we use p-values strictly > 0 to validate 0 discoveries.
        let data = Data::new();
        let results = vec![
            (0, 0u8, 0.001),
            (1, 0u8, 0.02),
            (2, 0u8, 0.50),
            (3, 0u8, 0.99),
        ];
        let corrected = data.apply_fdr_correction(results, 0.0);
        assert_eq!(
            corrected.len(),
            0,
            "With alpha=0 & p>0, every features should be discarded"
        );
    }

    #[test]
    fn test_fdr_correction_alpha_one() {
        // BH with alpha=1: the largest threshold is (n/n)*1 = 1,
        // so all p-values in [0,1] are retained.
        let data = Data::new();
        let results = vec![
            (0, 0u8, 0.001),
            (1, 0u8, 0.20),
            (2, 0u8, 0.70),
            (3, 0u8, 1.00),
        ];
        let corrected = data.apply_fdr_correction(results.clone(), 1.0);
        assert_eq!(
            corrected.len(),
            results.len(),
            "With alpha=1 & p∈[0,1], every features should be selected"
        );
    }
}
