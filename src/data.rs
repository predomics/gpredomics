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
use std::path::Path;

/// Methods available for feature preselection
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[allow(non_camel_case_types)]
pub enum PreselectionMethod {
    /// Wilcoxon rank-sum test
    wilcoxon,
    /// Student's t-test
    studentt,
    /// Bayesian Fisher's exact test
    bayesian_fisher,
}

/// Feature annotations structure for storing additional metadata related to features.
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct FeatureAnnotations {
    /// Names of the tag columns in the annotation file
    pub tag_column_names: Vec<String>,
    /// Mapping from feature index to its associated tags
    pub feature_tags: HashMap<usize, Vec<String>>,
    /// Mapping from feature index a prior weight influencing its selection during [`Population`] generation.
    pub prior_weight: HashMap<usize, f64>,
    /// Mapping from feature index to its penalty value influencing the calculation of fitness score.
    pub feature_penalty: HashMap<usize, f64>,
}

/// Sample annotations structure for storing additional metadata related to samples.
#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct SampleAnnotations {
    /// Names of the tag columns in the annotation file
    pub tag_column_names: Vec<String>,
    /// Mapping from sample index to its associated tags
    pub sample_tags: HashMap<usize, Vec<String>>,
}

/// Main data structure for storing feature matrix, target values, and related metadata.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct Data {
    /// Feature matrix and associated coefficients
    #[serde(with = "serde_json_hashmap_numeric::tuple_usize_f64")]
    pub X: HashMap<(usize, usize), f64>,
    /// Feature names
    pub features: Vec<String>,
    /// Features count
    pub feature_len: usize,
    /// Feature annotations if provided
    #[serde(default)]
    pub feature_annotations: Option<FeatureAnnotations>,

    /// Features selected after statistical testing
    pub feature_selection: Vec<usize>,
    /// Features associated class signs after statistical testing
    #[serde(with = "serde_json_hashmap_numeric::usize_u8")]
    pub feature_class: HashMap<usize, u8>,
    /// Features associated significance p-values/bayes factor after statistical testing
    #[serde(default)]
    pub feature_significance: HashMap<usize, f64>,

    /// Sample real classes
    pub y: Vec<u8>,
    /// Sample names
    pub samples: Vec<String>,
    /// Samples count
    pub sample_len: usize,
    /// Sample annotations if provided
    #[serde(default)]
    pub sample_annotations: Option<SampleAnnotations>,

    /// Class labels
    pub classes: Vec<String>,
}

impl Data {
    /// Creates a new `Data` instance.
    ///
    /// # Returns
    ///
    /// A new `Data` instance with initialized empty fields.
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
            feature_annotations: None,
            sample_annotations: None,
        }
    }

    /// Checks if another dataset is compatible with the current one
    ///
    /// This method compares the features, feature length, and classes of two `Data` instances in order to determine compatibility
    /// before proceeding with operations that require matching datasets like computing new predictions.
    ///
    /// # Arguments
    ///
    /// * `other` - Another `Data` instance to compare with
    ///
    /// # Returns
    ///
    /// `true` if compatible, `false` otherwise
    pub fn check_compatibility(&self, other: &Data) -> bool {
        self.features == other.features
            && self.feature_len == other.feature_len
            && self.classes == other.classes
    }

    /// Loads data from `X.tsv` and `y.tsv` files.
    ///
    /// # Arguments
    ///
    /// * `X_path` - Path to the feature matrix file
    /// * `y_path` - Path to the target values file
    /// * `features_in_rows` - Boolean indicating if features are in rows (true) or columns (false)
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error if loading fails
    ///
    /// # Errors
    ///
    /// * Returns error if file reading or parsing fails
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

    /// Detects the delimiter used in a file based on extension
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file
    ///
    /// # Returns
    ///
    /// `Ok(char)` with the detected delimiter, or an error if detection fails
    fn detect_delimiter(path: &str) -> Result<char, Box<dyn Error>> {
        let path_lower = path.to_lowercase();

        // Check file extension for tab-delimited formats
        if path_lower.ends_with(".txt")
            || path_lower.ends_with(".tsv")
            || path_lower.ends_with(".tab")
        {
            return Ok('\t');
        }

        // For .csv files, detect between comma and semicolon
        if path_lower.ends_with(".csv") {
            let file = File::open(path)?;
            let mut reader = BufReader::new(file);
            let mut first_line = String::new();
            reader.read_line(&mut first_line)?;

            let comma_count = first_line.matches(',').count();
            let semicolon_count = first_line.matches(';').count();

            if comma_count == 0 && semicolon_count == 0 {
                return Err(format!(
                    "Incompatible file format: no valid delimiter found in CSV file '{}'",
                    path
                )
                .into());
            }

            if semicolon_count > comma_count {
                return Ok(';');
            } else {
                return Ok(',');
            }
        }

        // Unknown extension: fail immediately
        Err(format!("Incompatible file format: unknown file extension for '{}'. Supported formats: .txt, .tsv, .tab, .csv", path).into())
    }

    /// Load y values from y file and reorder them to match the sample order
    fn load_y_data(
        &mut self,
        y_path: &str,
        trim_line: impl Fn(&str) -> &str,
    ) -> Result<(), Box<dyn Error>> {
        // Detect delimiter for y file
        let delimiter_y = Self::detect_delimiter(y_path)?;
        debug!("Detected delimiter for y: {:?}", delimiter_y);

        // Open and read the y file
        let file_y = File::open(y_path)?;
        let reader_y = BufReader::new(file_y);

        // Parse y file and store target values
        let mut y_map = HashMap::new();
        for line in reader_y.lines().skip(1) {
            let line = line?;
            let trimmed_line = trim_line(&line);
            let mut fields = trimmed_line.split(delimiter_y);

            // First field is the sample name
            if let Some(sample_name) = fields.next() {
                // Second field is the target value
                if let Some(value) = fields.next() {
                    let target: u8 = value.parse()?;
                    y_map.insert(sample_name.to_string(), target);
                }
            }
        }

        // Reorder `y` to match the order of `samples` from X file
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

        Ok(())
    }

    /// Loads data following Predomics Legacy format (current Gpredomics format): rows=features, columns=samples
    ///
    /// # Arguments
    ///
    /// * `X_path` - Path to the feature matrix file
    /// * `y_path` - Path to the target values file
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error if loading fails
    ///
    /// # Errors
    ///
    /// * Returns error if file reading or parsing fails
    fn load_data_features_in_rows(
        &mut self,
        X_path: &str,
        y_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        #[inline]
        fn trim_line(line: &str) -> &str {
            line.trim_end_matches(['\n', '\r'])
        }

        // Check if files exist
        if !Path::new(X_path).exists() {
            return Err(format!("File does not exist: {}", X_path).into());
        }
        if !Path::new(y_path).exists() {
            return Err(format!("File does not exist: {}", y_path).into());
        }

        info!("Loading files {} and {}...", X_path, y_path);

        // Detect delimiter for X file
        let delimiter_X = Self::detect_delimiter(X_path)?;
        debug!("Detected delimiter for X: {:?}", delimiter_X);

        let file_X = File::open(X_path)?;
        let mut reader_X = BufReader::with_capacity(8 * 1024 * 1024, file_X);

        // Read the first line to get sample names
        let mut first_line = String::new();
        reader_X.read_line(&mut first_line)?;
        let trimmed_first_line = trim_line(&first_line);
        self.samples = trimmed_first_line
            .split(delimiter_X)
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
            let mut fields = trimmed_line.split(delimiter_X);

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

        // Load y data
        self.load_y_data(y_path, trim_line)?;

        self.feature_len = self.features.len();
        self.sample_len = self.samples.len();

        // Validate that we have sufficient data (likely parsing error if < 3)
        if self.feature_len < 3 {
            return Err(format!("Failed to read data: only {} feature(s) found. This likely indicates a parsing error. Please check the file format and delimiter.", self.feature_len).into());
        }
        if self.sample_len < 3 {
            return Err(format!("Failed to read data: only {} sample(s) found. This likely indicates a parsing error. Please check the file format and delimiter.", self.sample_len).into());
        }

        Ok(())
    }

    /// Loads data following standard ML format: rows=samples, columns=features
    ///
    /// # Arguments
    ///
    /// * `X_path` - Path to the feature matrix file
    /// * `y_path` - Path to the target values file
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or an error if loading fails
    ///
    /// # Errors
    ///
    /// * Returns error if file reading or parsing fails
    fn load_data_features_in_columns(
        &mut self,
        X_path: &str,
        y_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        #[inline]
        fn trim_line(line: &str) -> &str {
            line.trim_end_matches(['\n', '\r'])
        }

        // Check if files exist
        if !Path::new(X_path).exists() {
            return Err(format!("File does not exist: {}", X_path).into());
        }
        if !Path::new(y_path).exists() {
            return Err(format!("File does not exist: {}", y_path).into());
        }

        info!("Loading files {} and {}...", X_path, y_path);

        // Detect delimiter for X file
        let delimiter_X = Self::detect_delimiter(X_path)?;
        debug!("Detected delimiter for X: {:?}", delimiter_X);

        let file_X = File::open(X_path)?;
        let mut reader_X = BufReader::with_capacity(8 * 1024 * 1024, file_X);

        // Read the first line to get feature names
        let mut first_line = String::new();
        reader_X.read_line(&mut first_line)?;
        let trimmed_first_line = trim_line(&first_line);
        self.features = trimmed_first_line
            .split(delimiter_X)
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
            let mut fields = trimmed_line.split(delimiter_X);

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

        // Load y data
        self.load_y_data(y_path, trim_line)?;

        self.feature_len = self.features.len();
        self.sample_len = self.samples.len();

        // Validate that we have sufficient data (likely parsing error if < 3)
        if self.feature_len < 3 {
            return Err(format!("Failed to read data: only {} feature(s) found. This likely indicates a parsing error. Please check the file format and delimiter.", self.feature_len).into());
        }
        if self.sample_len < 3 {
            return Err(format!("Failed to read data: only {} sample(s) found. This likely indicates a parsing error. Please check the file format and delimiter.", self.sample_len).into());
        }

        Ok(())
    }

    /// Sets class labels
    ///
    /// # Arguments
    ///
    /// * `classes` - A vector of class labels to set
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use gpredomics::data::Data;
    /// let mut data = Data::new();
    /// data.load_data("./samples/Qin2014/Xtrain.tsv", "./samples/Qin2014/Ytrain.tsv", false).unwrap();
    /// data.set_classes(vec!["class1".to_string(), "class2".to_string()]);
    /// ```
    pub fn set_classes(&mut self, classes: Vec<String>) {
        self.classes = classes;
    }

    /// Inverts class labels 0 and 1 in the dataset
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use gpredomics::data::Data;
    /// let mut data = Data::new();
    /// data.load_data("./samples/Qin2014/Xtrain.tsv", "./samples/Qin2014/Ytrain.tsv", false).unwrap();
    /// data.inverse_classes();
    /// ```
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

    /// Loads a feature annotation file containing metadata for features.
    ///
    /// # File structure
    ///
    /// * feature_name (required)
    /// * prior_weight (optional, f64)
    /// * feature_penalty (optional, f64 or list of f64 separated by ',')
    /// * other columns (optional, stored in feature_tags)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the feature annotation file
    ///
    /// # Returns
    ///
    /// `Ok(FeatureAnnotations)` if successful, or an error if loading fails
    pub fn load_feature_annotation<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<FeatureAnnotations, Box<dyn Error>> {
        let mut fw_c = 0;
        let mut fp_c = 0;
        let path_buf = path.as_ref().to_path_buf();
        let file = File::open(&path_buf)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let header = match lines.next() {
            Some(Ok(line)) => line,
            _ => return Err("Empty annotation file".into()),
        };
        let columns: Vec<&str> = header.trim_end().split('\t').collect();
        let mut idx_feature = None;
        let mut idx_prior = None;
        let mut idx_penalty = None;
        let mut extra_idxs = vec![];
        for (i, col) in columns.iter().enumerate() {
            match *col {
                "feature" | "feature_name" => idx_feature = Some(i),
                "prior_weight" => idx_prior = Some(i),
                "feature_penalty" => idx_penalty = Some(i),
                _ => extra_idxs.push((i, col.to_string())),
            }
        }
        let idx_feature = idx_feature.ok_or("No feature/feature_name column found")?;
        let tag_column_names: Vec<String> =
            extra_idxs.iter().map(|(_, name)| name.clone()).collect();
        let mut feature_tags: HashMap<usize, Vec<String>> = HashMap::new();
        let mut prior_weight: HashMap<usize, f64> = HashMap::new();
        let mut feature_penalty: HashMap<usize, f64> = HashMap::new();
        // Reopen the file to read all lines after the header
        let file2 = File::open(&path_buf)?;
        let reader2 = BufReader::new(file2);
        let all_lines: Vec<String> = reader2.lines().skip(1).filter_map(Result::ok).collect();
        let mut annotation_features = Vec::new();
        for line in &all_lines {
            let fields: Vec<&str> = line.trim_end().split('\t').collect();
            if fields.len() > idx_feature {
                annotation_features.push(fields[idx_feature].to_string());
            }
        }

        // Parsing as before
        for line in &all_lines {
            let fields: Vec<&str> = line.trim_end().split('\t').collect();
            if fields.len() <= idx_feature {
                continue;
            }
            let feature_name = fields[idx_feature];
            if !self.features.contains(&feature_name.to_string()) {
                log::debug!(
                    "Feature '{}' from annotation not found in data.features: {:?}",
                    feature_name,
                    self.features
                );
            }
            let feature_idx = self.features.iter().position(|f| f == feature_name);
            let feature_idx = match feature_idx {
                Some(idx) => idx,
                None => {
                    log::warn!(
                        "Feature '{}' from annotation not found in data.features",
                        feature_name
                    );
                    continue; // unknown features
                }
            };
            // Prior_weight
            if let Some(idx) = idx_prior {
                if let Some(val) = fields.get(idx) {
                    if !val.is_empty() {
                        if let Ok(v) = val.parse::<f64>() {
                            prior_weight.insert(feature_idx, v);
                            fw_c += 1;
                        }
                    }
                }
            }
            // Feature_penalty
            if let Some(idx) = idx_penalty {
                if let Some(val) = fields.get(idx) {
                    if !val.is_empty() {
                        if let Ok(v) = val.parse::<f64>() {
                            feature_penalty.insert(feature_idx, v);
                            fp_c += 1;
                        }
                    }
                }
            }
            // Feature_tags
            let mut tags = Vec::new();
            for (i, _name) in &extra_idxs {
                if let Some(val) = fields.get(*i) {
                    tags.push(val.to_string());
                } else {
                    tags.push(String::new());
                }
            }
            if !tags.is_empty() {
                feature_tags.insert(feature_idx, tags);
            }
        }

        if fw_c != self.feature_len {
            warn!("Not all features have prior_weight defined in the annotation file ({}/{}). Missing values will default to 1.0.", fw_c, self.feature_len);
        }

        if fp_c != self.feature_len {
            warn!("Not all features have feature_penalty defined in the annotation file ({}/{}). Missing values will default to no penalty.", fp_c, self.feature_len);
        }

        Ok(FeatureAnnotations {
            feature_tags,
            tag_column_names,
            prior_weight,
            feature_penalty,
        })
    }

    /// Loads a sample annotation file containing metadata for samples.
    ///
    /// # File structure
    ///
    /// * sample or sample_name (required)
    /// * other columns (optional, stored in sample_tags)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the sample annotation file
    ///
    /// # Returns
    ///
    /// `Ok(SampleAnnotations)` if successful, or an error if loading fails
    pub fn load_sample_annotation<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<SampleAnnotations, Box<dyn Error>> {
        let path_buf = path.as_ref().to_path_buf();
        let file = File::open(&path_buf)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Header line
        let header = match lines.next() {
            Some(Ok(line)) => line,
            _ => return Err("Empty sample annotation file".into()),
        };

        let columns: Vec<&str> = header.trim_end().split('\t').collect();
        let mut idx_sample = None;
        let mut extra_idxs: Vec<(usize, String)> = Vec::new();

        // Detection of the sample column and additional columns
        for (i, col) in columns.iter().enumerate() {
            match *col {
                "sample" | "sample_name" => idx_sample = Some(i),
                _ => extra_idxs.push((i, col.to_string())),
            }
        }

        let idx_sample = idx_sample.ok_or("No sample/sample_name column found")?;

        let mut sample_tags: HashMap<usize, Vec<String>> = HashMap::new();

        // Parsing annotation lines
        for line_res in lines {
            let line = match line_res {
                Ok(l) => l,
                Err(e) => {
                    log::warn!(
                        "Error reading line in sample annotation file {}: {}",
                        path_buf.display(),
                        e
                    );
                    continue;
                }
            };

            let fields: Vec<&str> = line.trim_end().split('\t').collect();
            if fields.len() <= idx_sample {
                continue;
            }

            let sample_name = fields[idx_sample];

            // Attach to an index of self.samples
            if !self.samples.contains(&sample_name.to_string()) {
                log::debug!(
                    "Sample '{}' from annotation not found in data.samples: {:?}",
                    sample_name,
                    self.samples
                );
            }

            let sample_idx = match self.samples.iter().position(|s| s == sample_name) {
                Some(idx) => idx,
                None => {
                    log::warn!(
                        "Sample '{}' from annotation not found in data.samples",
                        sample_name
                    );
                    continue; // ignore unknown samples
                }
            };

            // Retrieval of tags (all columns except sample)
            let mut tags = Vec::with_capacity(extra_idxs.len());
            for i in &extra_idxs {
                let val = fields.get(i.0).map(|s| s.to_string()).unwrap_or_default();
                tags.push(val);
            }

            if !tags.is_empty() {
                sample_tags.insert(sample_idx, tags);
            }
        }

        let tag_column_names: Vec<String> =
            extra_idxs.iter().map(|(_, name)| name.clone()).collect();
        Ok(SampleAnnotations {
            tag_column_names,
            sample_tags,
        })
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

    /// Evaluate all features and return two lists of significant features for each class
    ///
    /// # Arguments
    ///
    /// * `param` - Reference to the Param struct containing feature selection parameters
    ///
    /// # Returns
    ///
    /// A tuple containing two vectors:
    /// * First vector: significant features for class 0 as (feature_index, class, significance_value)
    /// * Second vector: significant features for class 1 as (feature_index, class, significance_value)
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

    /// Fills feature_selection, e.g. a restriction of features based on param (notably pvalue as computed by either studentt or wilcoxon)
    ///
    /// # Arguments
    ///
    /// * `param` - Reference to the Param struct containing feature selection parameters
    ///
    /// # Examples
    ///
    /// ```
    /// # use gpredomics::data::Data;
    /// # use gpredomics::param::Param;
    /// let mut data = Data::new();
    /// data.load_data("./samples/Qin2014/Xtrain.tsv", "./samples/Qin2014/Ytrain.tsv", true).unwrap();
    /// let mut param = Param::default();
    /// data.select_features(&param);
    /// ```
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

    /// Applies Benjamini-Hochberg FDR correction for multiple hypothesis testing.
    ///
    /// # Arguments
    ///
    /// * `results` - Vector of tuples containing (feature_index, class, p_value)
    /// * `fdr_alpha` - Desired FDR alpha level
    ///
    /// # Returns
    ///
    /// Filtered vector of tuples after FDR correction
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

    /// Filters Data to keep only some samples (represented by a Vector of indices)
    ///
    /// # Arguments
    ///
    /// * `samples` - Vector of sample indices to keep
    ///
    /// # Returns
    ///
    /// A new Data instance containing only the specified samples
    ///
    /// # Examples
    ///
    /// ```
    /// # use gpredomics::data::Data;
    /// let mut data = Data::new();
    /// data.load_data("./samples/Qin2014/Xtrain.tsv", "./samples/Qin2014/Ytrain.tsv", false).unwrap();
    /// let subset_data = data.subset(vec![0, 2, 4, 6]);
    /// ```
    pub fn subset(&self, samples: Vec<usize>) -> Data {
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        for (new_sample, sample) in samples.iter().enumerate() {
            for feature in 0..self.feature_len {
                if self.X.contains_key(&(*sample, feature)) {
                    X.insert((new_sample, feature), self.X[&(*sample, feature)]);
                }
            }
        }

        let sample_annotations = self.sample_annotations.as_ref().map(|sa| {
            let mut new_sample_tags: HashMap<usize, Vec<String>> = HashMap::new();

            for (new_sample, old_sample) in samples.iter().enumerate() {
                if let Some(tags) = sa.sample_tags.get(old_sample) {
                    new_sample_tags.insert(new_sample, tags.clone());
                }
            }

            SampleAnnotations {
                tag_column_names: sa.tag_column_names.clone(),
                sample_tags: new_sample_tags,
            }
        });

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
            feature_annotations: self.feature_annotations.clone(),
            sample_annotations: sample_annotations,
        }
    }

    /// Clones the Data object with a new feature matrix X.
    ///
    /// # Arguments
    ///
    /// * `X` - New feature matrix as a HashMap
    ///
    /// # Returns
    ///
    /// A new Data instance with the updated feature matrix.
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
            feature_annotations: self.feature_annotations.clone(),
            sample_annotations: self.sample_annotations.clone(),
        }
    }

    /// Adds another Data object to the current one by appending samples and merging feature matrices.
    ///
    /// # Arguments
    ///
    /// * `other` - Reference to the Data object to be added
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use gpredomics::data::Data;
    /// let mut data1 = Data::new();
    /// data1.load_data("./samples/Qin2014/X1.tsv", "./samples/Qin2014/y1.tsv", false).unwrap();
    /// let mut data2 = Data::new();
    /// data2.load_data("./samples/Qin2014/X2.tsv", "./samples/Qin2014/y2.tsv", false).unwrap();
    /// data1.add(&data2);
    /// ```
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

    /// Removes all samples of a specified class from the Data object.
    ///
    /// This method is useful to remove unknown samples (label 2) from data.
    /// # Arguments
    ///
    /// * `class_to_remove` - The class label to be removed
    ///
    /// # Returns
    ///
    /// A new Data instance with the specified class samples removed.
    ///
    /// # Examples
    ///
    /// ```
    /// # use gpredomics::data::Data;
    /// let mut data = Data::new();
    /// data.load_data("./samples/Qin2014/Xtrain.tsv", "./samples/Qin2014/Ytrain.tsv", false).unwrap();
    /// let filtered_data = data.remove_class(2);
    /// ```
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

    /// Generates a random subset of samples while maintaining class proportions.
    ///
    /// # Arguments
    ///
    /// * `n_samples` - Number of samples to include in the subset
    /// * `rng` - Mutable reference to a random number generator for reproducibility
    ///
    /// # Returns
    ///
    /// A vector of sample indices representing the random subset.  
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

    /// Stratified train/test split with secondary stratification by annotation.
    ///
    /// # Arguments
    ///
    /// * `data` - complete dataset
    /// * `test_ratio` - fraction of data for testing in (0,1)
    /// * `rng` - mutable random generator for reproducibility
    /// * `stratify_by` - name of the annotation column for double stratification (optional)
    ///
    /// # Returns
    ///
    /// A tuple containing the training and testing Data subsets.
    pub fn train_test_split(
        &self,
        test_ratio: f64,
        rng: &mut ChaCha8Rng,
        stratify_by: Option<&str>,
    ) -> (Data, Data) {
        assert!(
            test_ratio > 0.0 && test_ratio < 1.0,
            "test_ratio must be in (0,1)"
        );

        if stratify_by.is_none() {
            // Easy case: simple stratification by class only
            let (mut indices_class1, mut indices_class0) =
                utils::stratify_indices_by_class(&self.y);
            indices_class1.shuffle(rng);
            indices_class0.shuffle(rng);

            let n_test_1 = ((indices_class1.len() as f64) * test_ratio).round() as usize;
            let n_test_0 = ((indices_class0.len() as f64) * test_ratio).round() as usize;

            let (test_1, train_1) = indices_class1.split_at(n_test_1);
            let (test_0, train_0) = indices_class0.split_at(n_test_0);

            let mut train_indices = Vec::with_capacity(train_0.len() + train_1.len());
            train_indices.extend_from_slice(train_0);
            train_indices.extend_from_slice(train_1);

            let mut test_indices = Vec::with_capacity(test_0.len() + test_1.len());
            test_indices.extend_from_slice(test_0);
            test_indices.extend_from_slice(test_1);

            return (self.subset(train_indices), self.subset(test_indices));
        }

        // Case double stratification by annotation
        let stratify_col = stratify_by.unwrap();

        let annot = self
            .sample_annotations
            .as_ref()
            .expect("Sample annotations are required for stratified split by annotation");

        // Find the index of the annotation column
        let col_idx = annot
            .tag_column_names
            .iter()
            .position(|c| c == stratify_col)
            .expect(&format!(
                "Stratification column '{}' not found in sample annotations",
                stratify_col
            ));

        // Partition indices by class
        let (indices_class1, indices_class0) = utils::stratify_indices_by_class(&self.y);

        // Function to split stratified by annotation within each class
        let split_by_annotation = |indices: &[usize],
                                   rng: &mut ChaCha8Rng,
                                   ratio: f64|
         -> (Vec<usize>, Vec<usize>) {
            // Retrieve the annotation column for these indices
            let annotations_for_indices: Vec<_> = indices
                .iter()
                .map(|&i| {
                    let tags = annot.sample_tags.get(&i).unwrap_or_else(|| {
                        panic!(
                            "Sample index {} has no entry in sample_annotations.sample_tags while using double stratification on column '{}'. \
                            All samples must be annotated.",
                            i, stratify_col
                        );
                    });

                    tags.get(col_idx).unwrap_or_else(|| {
                        panic!(
                            "Sample index {} has incomplete annotations for column '{}': expected at least {} columns, found {}.",
                            i,
                            stratify_col,
                            col_idx + 1,
                            tags.len()
                        );
                    }).clone()
                })
                .collect();

            // Group indices by annotation level
            let mut levels_map: HashMap<String, Vec<usize>> = HashMap::new();
            for (idx, annotation_level) in annotations_for_indices.iter().enumerate() {
                levels_map
                    .entry(annotation_level.clone())
                    .or_insert_with(Vec::new)
                    .push(indices[idx]);
            }

            let mut train_part = Vec::new();
            let mut test_part = Vec::new();

            // For each annotation level, shuffle and split according to test_ratio
            for (_level, mut idxs) in levels_map.into_iter() {
                idxs.shuffle(rng);
                let len: usize = idxs.len();
                let n_test: usize = ((len as f64) * ratio).round() as usize;
                let n_test = n_test.min(len); // upper bound
                let (test_slice, train_slice) = idxs.split_at(n_test);
                train_part.extend_from_slice(train_slice);
                test_part.extend_from_slice(test_slice);
            }

            (train_part, test_part)
        };

        // Split within each class
        let (train_1, test_1) = split_by_annotation(&indices_class1, rng, test_ratio);
        let (train_0, test_0) = split_by_annotation(&indices_class0, rng, test_ratio);

        // Merge train and test indices from both classes
        let mut train_indices = Vec::with_capacity(train_0.len() + train_1.len());
        train_indices.extend_from_slice(&train_0);
        train_indices.extend_from_slice(&train_1);

        let mut test_indices = Vec::with_capacity(test_0.len() + test_1.len());
        test_indices.extend_from_slice(&test_0);
        test_indices.extend_from_slice(&test_1);

        (self.subset(train_indices), self.subset(test_indices))
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
        /// Generates a simple test dataset for unit testing.
        ///
        /// # Returns
        ///
        /// A Data instance with predefined values for testing.
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
                feature_annotations: None,
                sample_annotations: None,
            }
        }

        /// Generates a specific test dataset with random values for unit testing.
        ///
        /// # Arguments
        ///
        /// * `num_samples` - Number of samples in the dataset
        /// * `num_features` - Number of features in the dataset
        ///
        /// # Returns
        ///
        /// A Data instance with random values for testing.
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
                feature_annotations: None,
                sample_annotations: None,
            }
        }

        /// Generates a specific test where with significant features for unit testing.
        ///
        /// # Arguments
        ///
        /// * `num_samples` - Number of samples in the dataset
        /// * `num_features` - Number of features in the dataset
        ///
        /// # Returns
        ///
        /// A Data instance with significant features for testing.
        pub fn specific_significant_test(num_samples: usize, num_features: usize) -> Data {
            let mut X = HashMap::new();
            let mut rng = ChaCha8Rng::seed_from_u64(12345);

            // Populate X with (sample, feature) -> value mapping
            for sample in 0..num_samples {
                for feature in 0..num_features {
                    X.insert((sample, feature), rng.gen_range(0.0..1.0));
                }
            }

            let y: Vec<u8> = (0..num_samples)
                .map(|sample| {
                    if X.get(&(sample, 0)).cloned().unwrap_or(0.0) > 0.5 {
                        1
                    } else {
                        0
                    }
                })
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
                feature_annotations: None,
                sample_annotations: None,
            }
        }

        /// Generates a test dataset with specified features for unit testing.
        ///
        /// # Arguments
        ///
        /// * `feature_indices` - Slice of feature indices to include in the dataset
        ///
        /// # Returns
        ///
        /// A Data instance with the specified features for testing.
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

        /// Generates another specific test dataset for unit testing.
        ///     
        /// # Returns
        ///
        /// A Data instance with predefined values for testing.
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
                feature_annotations: None,
                sample_annotations: None,
            }
        }

        /// Generates a test dataset with discovery data for unit testing.
        ///    
        /// # Returns
        ///
        /// A Data instance with predefined discovery data for testing.
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
                feature_annotations: None,
                sample_annotations: None,
            }
        }

        /// Generates a test dataset with validation data for unit testing.
        ///
        /// # Returns
        ///
        /// A Data instance with predefined validation data for testing.
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
                feature_annotations: None,
                sample_annotations: None,
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
            feature_annotations: None,
            sample_annotations: None,
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
            feature_annotations: None,
            sample_annotations: None,
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

        let x_content = "FeatureID\tSample1\tSample2\tSample3\nFeature1\t0.5\t0.0\t0.3\nFeature2\t0.2\t0.4\t0.6\nFeature3\t0.7\t0.8\t0.9\n";
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

        let x_content = "FeatureID\tSample1\tSample2\tSample3\nFeature1\t0.5\t0.0\t0.3\nFeature2\t1.0\t2.0\t1.5\nFeature3\t0.7\t0.9\t0.8\n";
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
    fn test_default_features_in_rows_true() {
        let param = Param::new();
        assert_eq!(param.data.features_in_rows, true);
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

    #[test]
    fn test_load_feature_annotation_basic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create test data with known features
        let mut data = Data::new();
        data.features = vec![
            "feature1".to_string(),
            "feature2".to_string(),
            "feature3".to_string(),
        ];
        data.feature_len = 3;

        // Create temporary annotation file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            "feature_name\tprior_weight\tfeature_penalty\ttag1\ttag2"
        )
        .unwrap();
        writeln!(temp_file, "feature1\t1.5\t0.1\tgood\tA").unwrap();
        writeln!(temp_file, "feature2\t2.0\t0.2\tbetter\tB").unwrap();
        writeln!(temp_file, "feature3\t0.5\t0.3\tbest\tC").unwrap();
        temp_file.flush().unwrap();

        let annotations = data.load_feature_annotation(temp_file.path()).unwrap();

        // Verify tag_column_names
        assert_eq!(annotations.tag_column_names, vec!["tag1", "tag2"]);

        // Verify prior_weight
        assert_eq!(annotations.prior_weight.get(&0), Some(&1.5));
        assert_eq!(annotations.prior_weight.get(&1), Some(&2.0));
        assert_eq!(annotations.prior_weight.get(&2), Some(&0.5));

        // Verify feature_penalty
        assert_eq!(annotations.feature_penalty.get(&0), Some(&0.1));
        assert_eq!(annotations.feature_penalty.get(&1), Some(&0.2));
        assert_eq!(annotations.feature_penalty.get(&2), Some(&0.3));

        // Verify feature_tags
        assert_eq!(
            annotations.feature_tags.get(&0),
            Some(&vec!["good".to_string(), "A".to_string()])
        );
        assert_eq!(
            annotations.feature_tags.get(&1),
            Some(&vec!["better".to_string(), "B".to_string()])
        );
        assert_eq!(
            annotations.feature_tags.get(&2),
            Some(&vec!["best".to_string(), "C".to_string()])
        );
    }

    #[test]
    fn test_load_feature_annotation_partial_data() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut data = Data::new();
        data.features = vec!["feature1".to_string(), "feature2".to_string()];
        data.feature_len = 2;

        // Create annotation file with missing values
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "feature_name\tprior_weight\ttag1").unwrap();
        writeln!(temp_file, "feature1\t1.5\ttagA").unwrap();
        writeln!(temp_file, "feature2\t\ttagB").unwrap(); // Missing prior_weight
        temp_file.flush().unwrap();

        let annotations = data.load_feature_annotation(temp_file.path()).unwrap();

        // feature1 should have prior_weight
        assert_eq!(annotations.prior_weight.get(&0), Some(&1.5));

        // feature2 should NOT have prior_weight (empty string)
        assert_eq!(annotations.prior_weight.get(&1), None);

        // Both should have tags
        assert_eq!(
            annotations.feature_tags.get(&0),
            Some(&vec!["tagA".to_string()])
        );
        assert_eq!(
            annotations.feature_tags.get(&1),
            Some(&vec!["tagB".to_string()])
        );
    }

    #[test]
    fn test_load_feature_annotation_unknown_features() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut data = Data::new();
        data.features = vec!["feature1".to_string(), "feature2".to_string()];
        data.feature_len = 2;

        // Create annotation file with unknown feature
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "feature_name\tprior_weight").unwrap();
        writeln!(temp_file, "feature1\t1.5").unwrap();
        writeln!(temp_file, "unknown_feature\t2.0").unwrap(); // Unknown feature
        writeln!(temp_file, "feature2\t0.5").unwrap();
        temp_file.flush().unwrap();

        let annotations = data.load_feature_annotation(temp_file.path()).unwrap();

        // Only known features should be present
        assert_eq!(annotations.prior_weight.len(), 2);
        assert_eq!(annotations.prior_weight.get(&0), Some(&1.5));
        assert_eq!(annotations.prior_weight.get(&1), Some(&0.5));
    }

    #[test]
    fn test_load_feature_annotation_alternative_column_names() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut data = Data::new();
        data.features = vec!["feature1".to_string()];
        data.feature_len = 1;

        // Test with "feature" instead of "feature_name"
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "feature\tprior_weight").unwrap();
        writeln!(temp_file, "feature1\t1.5").unwrap();
        temp_file.flush().unwrap();

        let annotations = data.load_feature_annotation(temp_file.path()).unwrap();
        assert_eq!(annotations.prior_weight.get(&0), Some(&1.5));
    }

    #[test]
    fn test_load_feature_annotation_missing_feature_column() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut data = Data::new();
        data.features = vec!["feature1".to_string()];
        data.feature_len = 1;

        // Create annotation file without feature/feature_name column
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "prior_weight\ttag1").unwrap();
        writeln!(temp_file, "1.5\ttagA").unwrap();
        temp_file.flush().unwrap();

        let result = data.load_feature_annotation(temp_file.path());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No feature/feature_name column found"));
    }

    #[test]
    fn test_load_sample_annotation_basic() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create test data with known samples
        let mut data = Data::new();
        data.samples = vec![
            "sample1".to_string(),
            "sample2".to_string(),
            "sample3".to_string(),
        ];
        data.sample_len = 3;

        // Create temporary annotation file
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "sample_name\tage\tgender\tcondition").unwrap();
        writeln!(temp_file, "sample1\t25\tM\thealthy").unwrap();
        writeln!(temp_file, "sample2\t30\tF\tdiseased").unwrap();
        writeln!(temp_file, "sample3\t45\tM\thealthy").unwrap();
        temp_file.flush().unwrap();

        let annotations = data.load_sample_annotation(temp_file.path()).unwrap();

        // Verify tag_column_names
        assert_eq!(
            annotations.tag_column_names,
            vec!["age", "gender", "condition"]
        );

        // Verify sample_tags
        assert_eq!(
            annotations.sample_tags.get(&0),
            Some(&vec![
                "25".to_string(),
                "M".to_string(),
                "healthy".to_string()
            ])
        );
        assert_eq!(
            annotations.sample_tags.get(&1),
            Some(&vec![
                "30".to_string(),
                "F".to_string(),
                "diseased".to_string()
            ])
        );
        assert_eq!(
            annotations.sample_tags.get(&2),
            Some(&vec![
                "45".to_string(),
                "M".to_string(),
                "healthy".to_string()
            ])
        );
    }

    #[test]
    fn test_load_sample_annotation_alternative_column_name() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut data = Data::new();
        data.samples = vec!["sample1".to_string()];
        data.sample_len = 1;

        // Test with "sample" instead of "sample_name"
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "sample\tage").unwrap();
        writeln!(temp_file, "sample1\t25").unwrap();
        temp_file.flush().unwrap();

        let annotations = data.load_sample_annotation(temp_file.path()).unwrap();
        assert_eq!(
            annotations.sample_tags.get(&0),
            Some(&vec!["25".to_string()])
        );
    }

    #[test]
    fn test_load_sample_annotation_unknown_samples() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut data = Data::new();
        data.samples = vec!["sample1".to_string(), "sample2".to_string()];
        data.sample_len = 2;

        // Create annotation file with unknown sample
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "sample_name\tage").unwrap();
        writeln!(temp_file, "sample1\t25").unwrap();
        writeln!(temp_file, "unknown_sample\t30").unwrap(); // Unknown sample
        writeln!(temp_file, "sample2\t35").unwrap();
        temp_file.flush().unwrap();

        let annotations = data.load_sample_annotation(temp_file.path()).unwrap();

        // Only known samples should be present
        assert_eq!(annotations.sample_tags.len(), 2);
        assert_eq!(
            annotations.sample_tags.get(&0),
            Some(&vec!["25".to_string()])
        );
        assert_eq!(
            annotations.sample_tags.get(&1),
            Some(&vec!["35".to_string()])
        );
    }

    #[test]
    fn test_load_sample_annotation_partial_data() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut data = Data::new();
        data.samples = vec!["sample1".to_string(), "sample2".to_string()];
        data.sample_len = 2;

        // Create annotation file with missing values
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "sample_name\tage\tgender").unwrap();
        writeln!(temp_file, "sample1\t25\tM").unwrap();
        writeln!(temp_file, "sample2\t\t").unwrap(); // Missing values
        temp_file.flush().unwrap();

        let annotations = data.load_sample_annotation(temp_file.path()).unwrap();

        // sample2 should have empty strings for missing values
        assert_eq!(
            annotations.sample_tags.get(&1),
            Some(&vec!["".to_string(), "".to_string()])
        );
    }

    #[test]
    fn test_load_sample_annotation_missing_sample_column() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut data = Data::new();
        data.samples = vec!["sample1".to_string()];
        data.sample_len = 1;

        // Create annotation file without sample/sample_name column
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "age\tgender").unwrap();
        writeln!(temp_file, "25\tM").unwrap();
        temp_file.flush().unwrap();

        let result = data.load_sample_annotation(temp_file.path());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No sample/sample_name column found"));
    }

    #[test]
    fn test_load_sample_annotation_empty_file() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut data = Data::new();
        data.samples = vec!["sample1".to_string()];
        data.sample_len = 1;

        // Create empty annotation file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.flush().unwrap();

        let result = data.load_sample_annotation(temp_file.path());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Empty sample annotation file"));
    }

    #[test]
    fn test_feature_annotations_struct() {
        // Test creating FeatureAnnotations manually
        let mut feature_tags = HashMap::new();
        feature_tags.insert(0, vec!["tag1".to_string(), "tag2".to_string()]);
        feature_tags.insert(1, vec!["tag3".to_string(), "tag4".to_string()]);

        let mut prior_weight = HashMap::new();
        prior_weight.insert(0, 1.5);
        prior_weight.insert(1, 2.0);

        let mut feature_penalty = HashMap::new();
        feature_penalty.insert(0, 0.1);
        feature_penalty.insert(1, 0.2);

        let annotations = FeatureAnnotations {
            tag_column_names: vec!["col1".to_string(), "col2".to_string()],
            feature_tags,
            prior_weight,
            feature_penalty,
        };

        assert_eq!(annotations.tag_column_names.len(), 2);
        assert_eq!(annotations.feature_tags.len(), 2);
        assert_eq!(annotations.prior_weight.len(), 2);
        assert_eq!(annotations.feature_penalty.len(), 2);
    }

    #[test]
    fn test_sample_annotations_struct() {
        // Test creating SampleAnnotations manually
        let mut sample_tags = HashMap::new();
        sample_tags.insert(0, vec!["25".to_string(), "M".to_string()]);
        sample_tags.insert(1, vec!["30".to_string(), "F".to_string()]);

        let annotations = SampleAnnotations {
            tag_column_names: vec!["age".to_string(), "gender".to_string()],
            sample_tags,
        };

        assert_eq!(annotations.tag_column_names.len(), 2);
        assert_eq!(annotations.sample_tags.len(), 2);
        assert_eq!(
            annotations.sample_tags.get(&0),
            Some(&vec!["25".to_string(), "M".to_string()])
        );
    }

    #[test]
    fn test_data_with_feature_annotations() {
        // Test Data struct integration with FeatureAnnotations
        let mut data = Data::test();

        let mut prior_weight = HashMap::new();
        prior_weight.insert(0, 1.5);

        let feature_annotations = FeatureAnnotations {
            tag_column_names: vec!["category".to_string()],
            feature_tags: HashMap::new(),
            prior_weight,
            feature_penalty: HashMap::new(),
        };

        data.feature_annotations = Some(feature_annotations);

        assert!(data.feature_annotations.is_some());
        let annotations = data.feature_annotations.unwrap();
        assert_eq!(annotations.prior_weight.get(&0), Some(&1.5));
    }

    #[test]
    fn test_data_with_sample_annotations() {
        // Test Data struct integration with SampleAnnotations
        let mut data = Data::test();

        let mut sample_tags = HashMap::new();
        sample_tags.insert(0, vec!["healthy".to_string()]);

        let sample_annotations = SampleAnnotations {
            tag_column_names: vec!["condition".to_string()],
            sample_tags,
        };

        data.sample_annotations = Some(sample_annotations);

        assert!(data.sample_annotations.is_some());
        let annotations = data.sample_annotations.unwrap();
        assert_eq!(
            annotations.sample_tags.get(&0),
            Some(&vec!["healthy".to_string()])
        );
    }

    #[test]
    fn test_train_test_split_preserves_total_size() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();

        let test_ratio = 0.3;
        let total = data.sample_len;

        let (train, test) = data.train_test_split(test_ratio, &mut rng, None);

        assert_eq!(
            train.sample_len + test.sample_len,
            total,
            "Total number of samples must be preserved after split"
        );
    }

    #[test]
    fn test_train_test_split_preserves_class_distribution() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::specific_test(100, 20); // 100 samples, 20 features

        let test_ratio = 0.25;
        let (train, test) = data.train_test_split(test_ratio, &mut rng, None);

        let orig_class0 = data.y.iter().filter(|&&y| y == 0).count();
        let orig_class1 = data.y.iter().filter(|&&y| y == 1).count();

        let train_class0 = train.y.iter().filter(|&&y| y == 0).count();
        let train_class1 = train.y.iter().filter(|&&y| y == 1).count();

        let test_class0 = test.y.iter().filter(|&&y| y == 0).count();
        let test_class1 = test.y.iter().filter(|&&y| y == 1).count();

        assert_eq!(
            train_class0 + test_class0,
            orig_class0,
            "Total number of class 0 samples must be preserved across train and test"
        );
        assert_eq!(
            train_class1 + test_class1,
            orig_class1,
            "Total number of class 1 samples must be preserved across train and test"
        );

        // Optional: check approximate ratio preservation with some tolerance
        let orig_ratio0 = orig_class0 as f64 / data.sample_len as f64;
        let train_ratio0 = train_class0 as f64 / train.sample_len as f64;
        let test_ratio0 = test_class0 as f64 / test.sample_len as f64;

        assert!(
            (train_ratio0 - orig_ratio0).abs() < 0.1,
            "Train class 0 ratio deviates too much from original (orig={:.3}, train={:.3})",
            orig_ratio0,
            train_ratio0
        );
        assert!(
            (test_ratio0 - orig_ratio0).abs() < 0.1,
            "Test class 0 ratio deviates too much from original (orig={:.3}, test={:.3})",
            orig_ratio0,
            test_ratio0
        );
    }

    #[test]
    fn test_train_test_split_reproducibility() {
        let data = Data::specific_test(50, 10);
        let test_ratio = 0.2;

        let mut rng1 = ChaCha8Rng::seed_from_u64(123);
        let mut rng2 = ChaCha8Rng::seed_from_u64(123);

        let (train1, test1) = data.train_test_split(test_ratio, &mut rng1, None);
        let (train2, test2) = data.train_test_split(test_ratio, &mut rng2, None);

        assert_eq!(
            train1.samples, train2.samples,
            "Train splits should be identical for the same seed"
        );
        assert_eq!(
            test1.samples, test2.samples,
            "Test splits should be identical for the same seed"
        );
    }

    #[test]
    fn test_train_test_split_no_overlap() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();
        let test_ratio = 0.4;

        let (train, test) = data.train_test_split(test_ratio, &mut rng, None);

        for s in &train.samples {
            assert!(
                !test.samples.contains(s),
                "Sample '{}' should not appear in both train and test sets",
                s
            );
        }
    }

    #[test]
    fn test_train_test_split_with_stratify_by_keeps_annotation_levels() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::specific_test(60, 10);

        // Build batch annotations: even index = "A", odd index = "B"
        let values: Vec<String> = (0..data.sample_len)
            .map(|i| {
                if i % 2 == 0 {
                    "A".to_string()
                } else {
                    "B".to_string()
                }
            })
            .collect();

        let annotations = create_sample_annotations(data.sample_len, "batch", values);
        data.sample_annotations = Some(annotations);

        let test_ratio = 0.3;
        let (train, test) = data.train_test_split(test_ratio, &mut rng, Some("batch"));

        // Count levels in original data
        let annot = data.sample_annotations.as_ref().unwrap();
        let col_idx = annot
            .tag_column_names
            .iter()
            .position(|c| c == "batch")
            .unwrap();

        let mut total_a = 0;
        let mut total_b = 0;
        for i in 0..data.sample_len {
            let v = &annot.sample_tags[&i][col_idx];
            if v == "A" {
                total_a += 1;
            } else {
                total_b += 1;
            }
        }

        // Count in train + test
        let count_levels = |d: &Data| {
            let a = d
                .sample_annotations
                .as_ref()
                .unwrap()
                .sample_tags
                .iter()
                .filter(|(_, tags)| tags[col_idx] == "A")
                .count();
            let b = d
                .sample_annotations
                .as_ref()
                .unwrap()
                .sample_tags
                .iter()
                .filter(|(_, tags)| tags[col_idx] == "B")
                .count();
            (a, b)
        };

        let (train_a, train_b) = count_levels(&train);
        let (test_a, test_b) = count_levels(&test);

        assert_eq!(
            train_a + test_a,
            total_a,
            "Total count of batch=A should be preserved across train and test"
        );
        assert_eq!(
            train_b + test_b,
            total_b,
            "Total count of batch=B should be preserved across train and test"
        );
    }

    #[test]
    fn test_train_test_split_preserves_class_by_annotation_counts() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Build a controlled dataset: 80 samples, 4 groups of 20:
        // (class 0, batch A), (class 0, batch B), (class 1, batch A), (class 1, batch B)
        let total_samples = 80;
        let num_features = 10;
        let mut data = Data::specific_test(total_samples, num_features);

        // Overwrite y to make class-balanced
        data.y = (0..total_samples)
            .map(|i| if i < total_samples / 2 { 0 } else { 1 })
            .collect();

        // Batch annotation alternating A/B
        let annotation_values: Vec<String> = (0..total_samples)
            .map(|i| {
                if i % 2 == 0 {
                    "A".to_string()
                } else {
                    "B".to_string()
                }
            })
            .collect();

        let annotations = create_sample_annotations(total_samples, "batch", annotation_values);
        data.sample_annotations = Some(annotations);

        let test_ratio = 0.25;
        let (train, test) = data.train_test_split(test_ratio, &mut rng, Some("batch"));

        let annot = data.sample_annotations.as_ref().unwrap();
        let col_idx = annot
            .tag_column_names
            .iter()
            .position(|c| c == "batch")
            .unwrap();

        // Original counts for (class, batch)
        let mut orig_counts = std::collections::HashMap::new();
        for i in 0..data.sample_len {
            let c = data.y[i];
            let b = &annot.sample_tags[&i][col_idx];
            let key = format!("class{}_batch{}", c, b);
            *orig_counts.entry(key).or_insert(0) += 1;
        }

        // Helper to count in a split
        let count_in_split = |d: &Data| {
            let annot = d.sample_annotations.as_ref().unwrap();
            let mut counts = std::collections::HashMap::new();
            for i in 0..d.sample_len {
                let c = d.y[i];
                let b = &annot.sample_tags[&i][col_idx];
                let key = format!("class{}_batch{}", c, b);
                *counts.entry(key).or_insert(0) += 1;
            }
            counts
        };

        let train_counts = count_in_split(&train);
        let test_counts = count_in_split(&test);

        // Sum train+test and compare to original
        let mut merged_counts = std::collections::HashMap::new();
        for (k, v) in train_counts.iter().chain(test_counts.iter()) {
            *merged_counts.entry(k.clone()).or_insert(0) += *v;
        }

        assert_eq!(
            merged_counts, orig_counts,
            "Joint distribution of (class, batch) should be preserved across train and test"
        );
    }

    #[test]
    #[should_panic(expected = "Sample annotations are required for stratified split by annotation")]
    fn test_train_test_split_panics_without_annotations_when_stratify_by_is_set() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();

        let _ = data.train_test_split(0.3, &mut rng, Some("batch"));
    }

    #[test]
    #[should_panic(expected = "Stratification column 'nonexistent' not found")]
    fn test_train_test_split_panics_with_wrong_annotation_column() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::test();

        let annotations = create_sample_annotations(
            data.sample_len,
            "batch",
            vec!["A".to_string(); data.sample_len],
        );
        data.sample_annotations = Some(annotations);

        let _ = data.train_test_split(0.3, &mut rng, Some("nonexistent"));
    }

    #[test]
    fn test_train_test_split_preserves_feature_metadata() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = Data::test();

        let (train, test) = data.train_test_split(0.5, &mut rng, None);

        assert_eq!(
            train.features, data.features,
            "Train set must preserve original feature names"
        );
        assert_eq!(
            test.features, data.features,
            "Test set must preserve original feature names"
        );
        assert_eq!(
            train.feature_len, data.feature_len,
            "Train set must preserve feature_len"
        );
        assert_eq!(
            test.feature_len, data.feature_len,
            "Test set must preserve feature_len"
        );
    }

    pub fn create_sample_annotations(
        sample_len: usize,
        stratify_column_name: &str,
        values: Vec<String>,
    ) -> SampleAnnotations {
        assert_eq!(sample_len, values.len(), "Length mismatch");

        let mut sample_tags: HashMap<usize, Vec<String>> = HashMap::new();

        for i in 0..sample_len {
            sample_tags.insert(i, vec![values[i].clone()]);
        }

        SampleAnnotations {
            tag_column_names: vec![stratify_column_name.to_string()],
            sample_tags,
        }
    }

    #[test]
    fn test_subset_correctly_remaps_sample_annotations() {
        // Create a dataset with sample annotations
        let mut data = Data::specific_test(10, 5); // 10 samples, 5 features

        // Assign sample annotations with distinct values for each sample
        let annotation_values: Vec<String> = (0..10).map(|i| format!("Sample_{}", i)).collect();

        let annotations = create_sample_annotations(10, "sample_id", annotation_values.clone());
        data.sample_annotations = Some(annotations);

        // Select a subset of samples: indices [1, 3, 7, 9]
        let selected_indices = vec![1, 3, 7, 9];
        let subset = data.subset(selected_indices.clone());

        // Verify that subset has correct sample_len
        assert_eq!(
            subset.sample_len,
            selected_indices.len(),
            "Subset should have {} samples",
            selected_indices.len()
        );

        // Verify that sample_annotations are present in the subset
        assert!(
            subset.sample_annotations.is_some(),
            "Subset should preserve sample_annotations"
        );

        let subset_annot = subset.sample_annotations.as_ref().unwrap();

        // Verify column names are preserved
        assert_eq!(
            subset_annot.tag_column_names,
            vec!["sample_id".to_string()],
            "Annotation column names should be preserved"
        );

        // Verify that annotations are correctly remapped to new indices
        for (new_idx, &old_idx) in selected_indices.iter().enumerate() {
            let expected_value = format!("Sample_{}", old_idx);
            let actual_value = &subset_annot.sample_tags[&new_idx][0];

            assert_eq!(
                actual_value, &expected_value,
                "Annotation for new index {} should correspond to old index {} (expected '{}', got '{}')",
                new_idx, old_idx, expected_value, actual_value
            );
        }

        // Verify that the number of annotation entries matches subset size
        assert_eq!(
            subset_annot.sample_tags.len(),
            selected_indices.len(),
            "Number of annotation entries should match subset size"
        );

        // Verify that all new indices [0..n) are present in sample_tags
        for i in 0..subset.sample_len {
            assert!(
                subset_annot.sample_tags.contains_key(&i),
                "Annotation for new index {} should exist",
                i
            );
        }
    }

    #[test]
    fn test_subset_with_multiple_annotation_columns() {
        let mut data = Data::specific_test(8, 3); // 8 samples, 3 features

        // Create annotations with multiple columns
        let mut sample_tags: HashMap<usize, Vec<String>> = HashMap::new();
        for i in 0..8 {
            sample_tags.insert(
                i,
                vec![
                    format!("Batch_{}", i % 2),  // Column 0: batch
                    format!("Center_{}", i % 3), // Column 1: center
                    format!("Cohort_{}", i / 4), // Column 2: cohort
                ],
            );
        }

        let annotations = SampleAnnotations {
            tag_column_names: vec![
                "batch".to_string(),
                "center".to_string(),
                "cohort".to_string(),
            ],
            sample_tags,
        };
        data.sample_annotations = Some(annotations);

        // Select subset: [0, 2, 5, 7]
        let selected = vec![0, 2, 5, 7];
        let subset = data.subset(selected.clone());

        let subset_annot = subset.sample_annotations.as_ref().unwrap();

        // Verify all column names are preserved
        assert_eq!(
            subset_annot.tag_column_names.len(),
            3,
            "All annotation columns should be preserved"
        );

        // Verify annotations for each selected sample
        let expected_annotations = vec![
            vec!["Batch_0", "Center_0", "Cohort_0"], // old index 0 -> new index 0
            vec!["Batch_0", "Center_2", "Cohort_0"], // old index 2 -> new index 1
            vec!["Batch_1", "Center_2", "Cohort_1"], // old index 5 -> new index 2
            vec!["Batch_1", "Center_1", "Cohort_1"], // old index 7 -> new index 3
        ];

        for (new_idx, expected) in expected_annotations.iter().enumerate() {
            let actual = &subset_annot.sample_tags[&new_idx];
            assert_eq!(
                actual, expected,
                "Annotations at new index {} should match expected values",
                new_idx
            );
        }
    }

    #[test]
    fn test_subset_without_sample_annotations() {
        let data = Data::specific_test(10, 5);

        // Ensure no sample annotations
        assert!(data.sample_annotations.is_none());

        let subset = data.subset(vec![0, 2, 4]);

        // Verify that subset also has no annotations
        assert!(
            subset.sample_annotations.is_none(),
            "Subset should have no annotations when original data has none"
        );

        assert_eq!(subset.sample_len, 3);
    }

    #[test]
    fn test_subset_empty_selection_preserves_annotation_structure() {
        let mut data = Data::specific_test(5, 3);

        let annotations = create_sample_annotations(5, "group", vec!["A".to_string(); 5]);
        data.sample_annotations = Some(annotations);

        // Empty subset
        let subset = data.subset(vec![]);

        assert_eq!(subset.sample_len, 0);

        // Annotations structure should exist but be empty
        let subset_annot = subset.sample_annotations.as_ref().unwrap();
        assert_eq!(
            subset_annot.tag_column_names,
            vec!["group".to_string()],
            "Column names should be preserved even with empty subset"
        );
        assert_eq!(
            subset_annot.sample_tags.len(),
            0,
            "sample_tags should be empty for empty subset"
        );
    }

    #[test]
    fn test_subset_preserves_annotation_after_train_test_split() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut data = Data::specific_test(20, 5);

        // Add annotations
        let annotation_values: Vec<String> = (0..20)
            .map(|i| {
                if i < 10 {
                    "GroupA".to_string()
                } else {
                    "GroupB".to_string()
                }
            })
            .collect();

        let annotations = create_sample_annotations(20, "treatment", annotation_values);
        data.sample_annotations = Some(annotations);

        // Perform train/test split with stratification
        let (train, test) = data.train_test_split(0.3, &mut rng, Some("treatment"));

        // Both train and test should have sample_annotations
        assert!(
            train.sample_annotations.is_some(),
            "Train set should have annotations"
        );
        assert!(
            test.sample_annotations.is_some(),
            "Test set should have annotations"
        );

        let train_annot = train.sample_annotations.as_ref().unwrap();
        let test_annot = test.sample_annotations.as_ref().unwrap();

        // Verify that indices are correctly remapped in both splits
        for i in 0..train.sample_len {
            assert!(
                train_annot.sample_tags.contains_key(&i),
                "Train annotation should have key for index {}",
                i
            );
        }

        for i in 0..test.sample_len {
            assert!(
                test_annot.sample_tags.contains_key(&i),
                "Test annotation should have key for index {}",
                i
            );
        }

        // Verify that all annotations are either "GroupA" or "GroupB"
        for tags in train_annot.sample_tags.values() {
            let val = &tags[0];
            assert!(
                val == "GroupA" || val == "GroupB",
                "Train annotation should be GroupA or GroupB, got {}",
                val
            );
        }

        for tags in test_annot.sample_tags.values() {
            let val = &tags[0];
            assert!(
                val == "GroupA" || val == "GroupB",
                "Test annotation should be GroupA or GroupB, got {}",
                val
            );
        }
    }

    #[test]
    #[should_panic(expected = "Sample annotations are required for stratified split by annotation")]
    fn test_traintestsplit_panic_on_missing_annotations() {
        let mut data = Data::test_with_these_features(&[0, 1, 2, 3]);
        data.sample_len = 10;
        data.y = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        data.sample_annotations = None;

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let _ = data.train_test_split(0.3, &mut rng, Some("batch"));
    }

    #[test]
    #[should_panic(expected = "Sample index")]
    fn test_traintestsplit_panic_on_incomplete_annotation_line() {
        let mut data = Data::test_with_these_features(&[0, 1, 2, 3]);
        data.sample_len = 3;
        data.y = vec![0, 1, 1];
        let mut sample_tags = HashMap::new();
        sample_tags.insert(0, vec!["control".to_string()]);
        sample_tags.insert(1, vec!["treatment".to_string(), "batch1".to_string()]);
        sample_tags.insert(2, vec!["treatment".to_string()]);

        data.sample_annotations = Some(SampleAnnotations {
            tag_column_names: vec!["group".to_string(), "batch".to_string(), "time".to_string()],
            sample_tags,
        });

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let _ = data.train_test_split(0.3, &mut rng, Some("time"));
    }

    // Tests for delimiter detection
    #[test]
    fn test_detect_delimiter_txt_extension() {
        use std::io::Write;
        let temp_file = "test_temp_file.txt";
        let mut file = File::create(temp_file).unwrap();
        writeln!(file, "col1\tcol2\tcol3").unwrap();

        let result = Data::detect_delimiter(temp_file);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), '\t', "TXT files should use tab delimiter");

        std::fs::remove_file(temp_file).unwrap();
    }

    #[test]
    fn test_detect_delimiter_tsv_extension() {
        use std::io::Write;
        let temp_file = "test_temp_file.tsv";
        let mut file = File::create(temp_file).unwrap();
        writeln!(file, "col1\tcol2\tcol3").unwrap();

        let result = Data::detect_delimiter(temp_file);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), '\t', "TSV files should use tab delimiter");

        std::fs::remove_file(temp_file).unwrap();
    }

    #[test]
    fn test_detect_delimiter_tab_extension() {
        use std::io::Write;
        let temp_file = "test_temp_file.tab";
        let mut file = File::create(temp_file).unwrap();
        writeln!(file, "col1\tcol2\tcol3").unwrap();

        let result = Data::detect_delimiter(temp_file);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), '\t', "TAB files should use tab delimiter");

        std::fs::remove_file(temp_file).unwrap();
    }

    #[test]
    fn test_detect_delimiter_csv_with_comma() {
        use std::io::Write;
        let temp_file = "test_temp_file_comma.csv";
        {
            let mut file = File::create(temp_file).unwrap();
            writeln!(file, "col1,col2,col3").unwrap();
            file.flush().unwrap();
        } // File is closed here

        let result = Data::detect_delimiter(temp_file);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ',',
            "CSV files with commas should detect comma delimiter"
        );

        std::fs::remove_file(temp_file).unwrap();
    }

    #[test]
    fn test_detect_delimiter_csv_with_semicolon() {
        use std::io::Write;
        let temp_file = "test_temp_file_semicolon.csv";
        {
            let mut file = File::create(temp_file).unwrap();
            writeln!(file, "col1;col2;col3").unwrap();
            file.flush().unwrap();
        } // File is closed here

        let result = Data::detect_delimiter(temp_file);
        assert!(result.is_ok());
        assert_eq!(
            result.unwrap(),
            ';',
            "CSV files with semicolons should detect semicolon delimiter"
        );

        std::fs::remove_file(temp_file).unwrap();
    }

    #[test]
    fn test_detect_delimiter_csv_without_delimiter() {
        use std::io::Write;
        let temp_file = "test_temp_file.csv";
        let mut file = File::create(temp_file).unwrap();
        writeln!(file, "col1 col2 col3").unwrap();

        let result = Data::detect_delimiter(temp_file);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no valid delimiter found"));

        std::fs::remove_file(temp_file).unwrap();
    }

    #[test]
    fn test_detect_delimiter_unknown_extension() {
        use std::io::Write;
        let temp_file = "test_temp_file.dat";
        let mut file = File::create(temp_file).unwrap();
        writeln!(file, "col1\tcol2\tcol3").unwrap();

        let result = Data::detect_delimiter(temp_file);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unknown file extension"));

        std::fs::remove_file(temp_file).unwrap();
    }

    // Tests for file existence check
    #[test]
    fn test_load_data_nonexistent_x_file() {
        let mut data = Data::new();
        let result = data.load_data("nonexistent_X.tsv", "samples/tests/y.tsv", true);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("File does not exist: nonexistent_X.tsv"));
    }

    #[test]
    fn test_load_data_nonexistent_y_file() {
        let mut data = Data::new();
        let result = data.load_data("samples/tests/X.tsv", "nonexistent_y.tsv", true);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("File does not exist: nonexistent_y.tsv"));
    }

    // Tests for minimum data requirements
    #[test]
    fn test_load_data_insufficient_features() {
        use std::io::Write;

        // Create temp files with only 2 features
        let temp_x = "test_temp_X_insufficient.tsv";
        let temp_y = "test_temp_y_insufficient.tsv";

        let mut file_x = File::create(temp_x).unwrap();
        writeln!(file_x, "\tsample1\tsample2\tsample3\tsample4").unwrap();
        writeln!(file_x, "feature1\t0.5\t0.6\t0.7\t0.8").unwrap();
        writeln!(file_x, "feature2\t0.1\t0.2\t0.3\t0.4").unwrap();

        let mut file_y = File::create(temp_y).unwrap();
        writeln!(file_y, "sample\tclass").unwrap();
        writeln!(file_y, "sample1\t0").unwrap();
        writeln!(file_y, "sample2\t1").unwrap();
        writeln!(file_y, "sample3\t0").unwrap();
        writeln!(file_y, "sample4\t1").unwrap();

        let mut data = Data::new();
        let result = data.load_data(temp_x, temp_y, true);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("only 2 feature(s) found"));

        std::fs::remove_file(temp_x).unwrap();
        std::fs::remove_file(temp_y).unwrap();
    }

    #[test]
    fn test_load_data_insufficient_samples() {
        use std::io::Write;

        // Create temp files with only 2 samples
        let temp_x = "test_temp_X_insufficient2.tsv";
        let temp_y = "test_temp_y_insufficient2.tsv";

        let mut file_x = File::create(temp_x).unwrap();
        writeln!(file_x, "\tsample1\tsample2").unwrap();
        writeln!(file_x, "feature1\t0.5\t0.6").unwrap();
        writeln!(file_x, "feature2\t0.1\t0.2").unwrap();
        writeln!(file_x, "feature3\t0.3\t0.4").unwrap();
        writeln!(file_x, "feature4\t0.7\t0.8").unwrap();

        let mut file_y = File::create(temp_y).unwrap();
        writeln!(file_y, "sample\tclass").unwrap();
        writeln!(file_y, "sample1\t0").unwrap();
        writeln!(file_y, "sample2\t1").unwrap();

        let mut data = Data::new();
        let result = data.load_data(temp_x, temp_y, true);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("only 2 sample(s) found"));

        std::fs::remove_file(temp_x).unwrap();
        std::fs::remove_file(temp_y).unwrap();
    }

    #[test]
    fn test_load_data_valid_minimum() {
        use std::io::Write;

        // Create temp files with exactly 3 features and 3 samples (minimum valid)
        let temp_x = "test_temp_X_valid.tsv";
        let temp_y = "test_temp_y_valid.tsv";

        let mut file_x = File::create(temp_x).unwrap();
        writeln!(file_x, "\tsample1\tsample2\tsample3").unwrap();
        writeln!(file_x, "feature1\t0.5\t0.6\t0.7").unwrap();
        writeln!(file_x, "feature2\t0.1\t0.2\t0.3").unwrap();
        writeln!(file_x, "feature3\t0.3\t0.4\t0.5").unwrap();

        let mut file_y = File::create(temp_y).unwrap();
        writeln!(file_y, "sample\tclass").unwrap();
        writeln!(file_y, "sample1\t0").unwrap();
        writeln!(file_y, "sample2\t1").unwrap();
        writeln!(file_y, "sample3\t0").unwrap();

        let mut data = Data::new();
        let result = data.load_data(temp_x, temp_y, true);

        assert!(result.is_ok());
        assert_eq!(data.feature_len, 3);
        assert_eq!(data.sample_len, 3);

        std::fs::remove_file(temp_x).unwrap();
        std::fs::remove_file(temp_y).unwrap();
    }
}
