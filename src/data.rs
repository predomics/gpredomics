use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::fmt;
use crate::param::Param;
use crate::utils::{compare_classes_studentt, compare_classes_wilcoxon };

pub struct Data {
    pub X: Vec<Vec<f64>>,         // Matrix for feature values
    pub y: Vec<u8>,              // Vector for target values
    pub features: Vec<String>,    // Feature names (from the first column of X.tsv)
    pub samples: Vec<String>,     // Sample names (from the first row of X.tsv)
    pub univariate_order: Vec<u32>,     // Order of univariate features
    pub feature_class_sign: HashMap<usize, u8>, // Sign for each feature
    pub feature_selection: Vec<usize>,
    pub feature_len: usize,
    pub sample_len: usize
}

impl Data {
    /// Create a new `Data` instance with default values
    pub fn new() -> Data {
        Data {
            X: Vec::new(),
            y: Vec::new(),
            features: Vec::new(),
            samples: Vec::new(),
            univariate_order: Vec::new(),
            feature_class_sign: HashMap::new(),
            feature_selection: Vec::new(),
            feature_len: 0,
            sample_len: 0
        }
    }

    /// Load data from `X.tsv` and `y.tsv` files.
    pub fn load_data(&mut self, X_path: &str, y_path: &str) -> Result<(), Box<dyn Error>> {
        println!("Loading files {} and {}...", X_path, y_path);
        // Open and read the X.tsv file
        let file_X = File::open(X_path)?;
        let mut reader_X = BufReader::new(file_X);

        // Read the first line to get sample names
        let mut first_line = String::new();
        reader_X.read_line(&mut first_line)?;
        self.samples = first_line.trim_end().split('\t').skip(1).map(String::from).collect();

        // Read the remaining lines for feature names and data
        for line in reader_X.lines() {
            let line = line?;
            let mut fields = line.trim_end().split('\t');

            // First field is the feature name
            if let Some(feature_name) = fields.next() {
                self.features.push(feature_name.to_string());
            }

            // Remaining fields are the feature values
            let row: Vec<f64> = fields
                .map(|value| value.parse::<f64>().unwrap_or(0.0))
                .collect();
            self.X.push(row);
        }

        // Open and read the y.tsv file
        let file_y = File::open(y_path)?;
        let reader_y = BufReader::new(file_y);

        // Parse y.tsv and store target values
        let mut y_map = HashMap::new();
        for line in reader_y.lines().skip(1) {
            let line = line?;
            let mut fields = line.trim_end().split('\t');
            
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
            .map(|sample_name| *y_map.get(sample_name).unwrap_or(&0))
            .collect();

        self.feature_len = self.features.len();
        self.sample_len = self.samples.len();

        Ok(())
    }

    /*pub fn compute_feature_stats(&self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // Number of features
        let num_features = self.features.len();
        
        // Number of classes (assumed to be 0, 1, 2)
        let num_classes = 3;

        // Initialize accumulators
        let mut sums = vec![vec![0.0; num_features]; num_classes];
        let mut counts = vec![vec![0; num_features]; num_classes];
        let mut non_null_counts = vec![vec![0; num_features]; num_classes];

        // Iterate over rows in X and their corresponding class in y
        for (row, &class) in self.X.iter().zip(self.y.iter()) {
            assert!(class < num_classes as u8, "Invalid class label in y");

            for (j, &value) in row.iter().enumerate() {
                if value != 0.0 {
                    sums[class as usize][j] += value;
                    counts[class as usize][j] += 1;
                }
                non_null_counts[class as usize][j] += 1;
            }
        }

        // Calculate averages and prevalences
        let averages: Vec<Vec<f64>> = (0..num_classes)
            .map(|class| {
                (0..num_features)
                    .map(|j| {
                        if counts[class][j] > 0 {
                            sums[class][j] / counts[class][j] as f64
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        let prevalences: Vec<Vec<f64>> = (0..num_classes)
            .map(|class| {
                (0..num_features)
                    .map(|j| non_null_counts[class][j] as f64 / y.len() as f64)
                    .collect()
            })
            .collect();

        (averages, prevalences)
    }*/

    pub fn select_features(&mut self, param:&Param) {
        self.feature_selection = Vec::new();
        self.feature_class_sign = HashMap::new();

        let method=if param.data.pvalue_method=="studentt" { compare_classes_studentt } 
            else { compare_classes_wilcoxon };

        for (i,row) in self.X.iter().enumerate() {
            match method(row, &(self.y), param.data.feature_maximal_pvalue, 
                        param.data.feature_minimal_prevalence_pct as f64/100.0, param.data.feature_minimal_feature_value) {
                0 => {self.feature_selection.push(i); self.feature_class_sign.insert(i, 0);},
                1 => {self.feature_selection.push(i); self.feature_class_sign.insert(i, 1);},
                _ => {}
            }
        }
    }

    pub fn subset(&self, indices: Vec<usize>) -> Data {
        Data {
            X: self.X.iter().map(|row| { 
                indices.iter().map(|i| {row[*i]}).collect()
                }).collect(),
            y: indices.iter().map(|i| {self.y[*i]}).collect(),
            features: self.features.clone(),
            samples: indices.iter().map(|i| {self.samples[*i].clone()}).collect(),
            univariate_order: Vec::new(),
            feature_class_sign: HashMap::new(),
            feature_selection: Vec::new(),
            feature_len: self.feature_len,
            sample_len: indices.len()
        }
    }

    pub fn clone(&self) -> Data {
        Data {
            X: self.X.clone(),
            y: self.y.clone(),
            features: self.features.clone(),
            samples: self.samples.clone(),
            univariate_order: Vec::new(),
            feature_class_sign: self.feature_class_sign.clone(),
            feature_selection: self.feature_selection.clone(),
            feature_len: self.feature_len,
            sample_len: self.sample_len
        }
    }

    pub fn clone_with_new_x(&self, X:Vec<Vec<f64>>) -> Data {
        Data {
            X: X,
            y: self.y.clone(),
            features: self.features.clone(),
            samples: self.samples.clone(),
            univariate_order: Vec::new(),
            feature_class_sign: self.feature_class_sign.clone(),
            feature_selection: self.feature_selection.clone(),
            feature_len: self.feature_len,
            sample_len: self.sample_len
        }
    }

    pub fn add(&mut self, other: &Data) {
        self.samples.extend_from_slice(&other.samples);
        self.y.extend_from_slice(&other.y);
        for (i,row) in self.X.iter_mut().enumerate() {
            row.extend_from_slice(&other.X[i]);
        }
    }

}

/// Implement a custom Debug trait for Data
impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let samples_string = self.samples.join("\t");
        let truncated_samples = if samples_string.len() > 100 {
            format!("{}...", &samples_string[..97])
        } else {
            samples_string
        };

        writeln!(f, "X:                  {}",truncated_samples)?;
        // Limit to the first 20 rows
        for (i, row) in self.X.iter().enumerate().take(20) {
            let feature = &self.features[i]; // Use the feature name from self.features
            let row_display: String = row
                .iter()
                .map(|value| format!("{:.2}", value))
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
        for (i,sample) in self.y.iter().take(20).enumerate() {
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