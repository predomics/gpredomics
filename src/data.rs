use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::fmt;

pub struct Data {
    pub X: Vec<Vec<f32>>,         // Matrix for feature values
    pub y: Vec<u8>,              // Vector for target values
    pub features: Vec<String>,    // Feature names (from the first column of X.tsv)
    pub samples: Vec<String>,     // Sample names (from the first row of X.tsv)
    pub univariate_order: Vec<u32>,     // Order of univariate features
    pub feature_class_sign: HashMap<u32, u8>, // Sign for each feature
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
        }
    }

    /// Load data from `X.tsv` and `y.tsv` files.
    pub fn load_data(&mut self, X_path: &str, y_path: &str) -> Result<(), Box<dyn Error>> {
        println!("Loading file...");
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
            let row: Vec<f32> = fields
                .map(|value| value.parse::<f32>().unwrap_or(0.0))
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

        Ok(())
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