use std::fs::File;
use std::io::BufReader;
use std::collections::HashMap;

use polars::prelude::*; /// DataFrame, Series


pub struct Data {
    pub X: DataFrame,
    pub y: Series,
    univariate_order: Vec<u32>,
    /// feature_names: HashMap<u32, String>,
    feature_class_sign: HashMap<u32,u8>, // 0 for negative, 1 for positive

}


impl Data {
    /// Create a new Data struct with default values.
    pub fn new() -> Data {
        Data {
            X: DataFrame::default(),
            y: Series::default(),
            univariate_order: Vec::new(),
            feature_class_sign: HashMap::new(),
        }
    }

    /// Load data from CSV files into the Data struct.
    pub fn load_data(&mut self, X_path: &str, y_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file_X = File::open(X_path)?;
        let file_y = File::open(y_path)?;

        let reader_X = BufReader::new(file_X);
        let reader_y = BufReader::new(file_y);

        self.X = CsvReader::new(reader_X)
            .has_header(true)
            .with_delimiter(b'\t')
            .finish()?;
        self.y = CsvReader::new(reader_y)
            .has_header(true)
            .with_delimiter(b'\t')
            .finish()?
            .select_at_idx(1)
            .unwrap().clone();

        Ok(())
    }


 //   pub fn evaluate_feature_order(&mut self) -> Result<()> {
 //       self.funivariate_order = self.X.get_rows().iter().enumerate();
 //   }

    
}