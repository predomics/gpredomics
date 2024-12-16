use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::fmt;
use crate::param::Param;
use statrs::distribution::{ContinuousCDF, StudentsT};
use statrs::distribution::Normal;// For random shuffling
pub struct Data {
    pub X: HashMap<(usize,usize),f64>,         // Matrix for feature values
    pub y: Vec<u8>,              // Vector for target values
    pub features: Vec<String>,    // Feature names (from the first column of X.tsv)
    pub samples: Vec<String>,
    pub feature_class_sign: HashMap<usize, u8>, // Sign for each feature
    pub feature_selection: Vec<usize>,
    pub feature_len: usize,
    pub sample_len: usize
}

impl Data {
    /// Create a new `Data` instance with default values
    pub fn new() -> Data {
        Data {
            X: HashMap::new(),
            y: Vec::new(),
            features: Vec::new(),
            samples: Vec::new(),
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
        let trimmed_first_line= first_line.strip_suffix('\n')
            .or_else(|| first_line.strip_suffix("\r\n"))
            .unwrap_or(&first_line);
        self.samples = trimmed_first_line.split('\t').skip(1).map(String::from).collect();

        // Read the remaining lines for feature names and data
        for (j,line) in reader_X.lines().enumerate() {
            let line = line?;
            let trimmed_line= line.strip_suffix('\n')
                .or_else(|| line.strip_suffix("\r\n"))
                .unwrap_or(&line);
            let mut fields = trimmed_line.split('\t');

            // First field is the feature name
            if let Some(feature_name) = fields.next() {
                self.features.push(feature_name.to_string());
            }

            // Remaining fields are the feature values
            for (i,value) in fields.enumerate() {
                if let Ok(num_val)=value.parse::<f64>() {
                    if num_val!=0.0 {
                        self.X.insert((i,j),num_val);
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
            let trimmed_line= line.strip_suffix('\n')
                .or_else(|| line.strip_suffix("\r\n"))
                .unwrap_or(&line);
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
            .map(|sample_name| *y_map.get(sample_name).unwrap_or(&0))
            .collect();

        self.feature_len = self.features.len();
        self.sample_len = self.samples.len();

        println!("Features: {}   Samples: {}",self.feature_len, self.sample_len);

        Ok(())
    }

    /// for a given feature (chosen as the #j line of X ) answer 0 if the feature is more significantly associated with class 0, 1 with class 1, 2 otherwise 
    /// using student T e.g. for normally ditributed features
    fn compare_classes_studentt(&self, j: usize, max_p_value: f64, min_prevalence: f64, min_mean_value: f64) -> u8 {
        // Separate values into two classes

        let mut count_0: usize=0;
        let mut count_1: usize=0;

        let class_0: Vec<f64> = (0..self.sample_len)
            .filter(|i| {self.y[*i] == 0})
            .map(|i| {if self.X.contains_key(&(i,j)) {count_0+=1; self.X[&(i,j)]} else {0.0}}).collect();
    
        let class_1: Vec<f64> = (0..self.sample_len)
            .filter(|i| {self.y[*i] == 1})
            .map(|i| {if self.X.contains_key(&(i,j)) {count_1+=1; self.X[&(i,j)]} else {0.0}}).collect();
    
        let n0 = class_0.len() as f64;
        let n1 = class_1.len() as f64;
        //println!("prev0: {}-{}",n0,count_0);

        let prev0 = count_0 as f64 / n0;
        let prev1 = count_1 as f64 / n1;

        if prev0<min_prevalence && prev1<min_prevalence { return 2 }

        // Calculate means
        let mean_0 = class_0.iter().copied().sum::<f64>() / class_0.len() as f64;
        let mean_1 = class_1.iter().copied().sum::<f64>() / class_1.len() as f64;
    
        if mean_0<min_mean_value && mean_1<min_mean_value { return 2 }


    
        // Calculate t-statistic (simple, equal variance assumption)

        let var0 = class_0.iter().map(|x| (x - mean_0).powi(2)).sum::<f64>() / (n0 - 1.0);
        let var1 = class_1.iter().map(|x| (x - mean_1).powi(2)).sum::<f64>() / (n1 - 1.0);

        let pooled_std = ((var0 / n0) + (var1 / n1)).sqrt();
        if pooled_std > 0.0 {
            let t_stat = (mean_0 - mean_1) / pooled_std;
    
            // Compute p-value
            let degrees_of_freedom = (n0 + n1 - 2.0).round();
            let t_dist = StudentsT::new(0.0, 1.0, degrees_of_freedom).unwrap();
            //println!("t_stat {} n0 {} n1 {} var0 {} var1 {} prev0 {} prev1 {}",t_stat,n0,n1,var0,var1,prev0,prev1);
            let cumulative = t_dist.cdf(t_stat.abs()); // CDF up to |t_stat|
            let upper_tail = 1.0 - cumulative;         // Upper-tail area
            let p_value = 2.0 * upper_tail;       // Two-tailed test
    
            // Interpretation
            if (p_value < max_p_value) {
                if mean_0 > mean_1 {
                    0
                } else {
                    1
                }
            } else {
                2
            }
        }
        else {2}
    }
    

    /// Same as above but using Wilcoxon this time: for a given feature (chosen as the #j line of X ) answer 0 if the feature is more significantly associated with class 0, 1 with class 1, 2 otherwise 
    /// using Wilcoxon e.g. for sparse/log normal features
    pub fn compare_classes_wilcoxon(&self, j:usize, max_p_value: f64, min_prevalence: f64, min_mean_value: f64) -> u8 {
        // Separate values into two classes
        let mut count_0: usize=0;
        let mut count_1: usize=0;

        let class_0: Vec<f64> = (0..self.sample_len)
            .filter(|i| {self.y[*i] == 0})
            .map(|i| {if self.X.contains_key(&(i,j)) {count_0+=1; self.X[&(i,j)]} else {0.0}}).collect();
    
        let class_1: Vec<f64> = (0..self.sample_len)
            .filter(|i| {self.y[*i] == 1})
            .map(|i| {if self.X.contains_key(&(i,j)) {count_1+=1; self.X[&(i,j)]} else {0.0}}).collect();
    
        let n0 = class_0.len() as f64;
        let n1 = class_1.len() as f64;

        if n0==0.0 || n1==0.0 { return 2 }

        let prev0 = count_0 as f64 / n0;
        let prev1 = count_1 as f64 / n1;

        if prev0<min_prevalence && prev1<min_prevalence { return 2 }

        // Calculate means
        let mean_0 = class_0.iter().copied().sum::<f64>() / class_0.len() as f64;
        let mean_1 = class_1.iter().copied().sum::<f64>() / class_1.len() as f64;
    
        if mean_0<min_mean_value && mean_1<min_mean_value { return 2 }
    

    
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
            let rank = (start + i + 1) as f64 / 2.0;
            for j in start..=i {
                ranks[j] = rank;
            }
            i += 1;
        }
    
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
        let std_u = ((n0 * n1 * (n0 + n1 + 1.0)) / 12.0).sqrt();
        let z = (u_stat - mean_u) / std_u;
    
        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - normal_dist.cdf(z.abs())); // Two-tailed p-value
    
        // Interpretation
        if p_value < max_p_value {
            if rank_sum_0 > (n0 * n1 / 2.0) {
                0
            } else {
                1
            }
        } else {
            2
        }
    }
    

    /// Fill feature_selection, e.g. a restriction of features based on param (notably pvalue as computed by either studentt or wilcoxon)
    pub fn select_features(&mut self, param:&Param) {
        self.feature_selection = Vec::new();
        self.feature_class_sign = HashMap::new();

        if param.data.pvalue_method=="studentt" { 
            for j in 0..self.feature_len {
                match self.compare_classes_studentt(j,param.data.feature_maximal_pvalue,
                    param.data.feature_minimal_prevalence_pct as f64/100.0, param.data.feature_minimal_feature_value) {
                    0 => {self.feature_selection.push(j); self.feature_class_sign.insert(j, 0);},
                    1 => {self.feature_selection.push(j); self.feature_class_sign.insert(j, 1);},
                    _ => {}
                }
            }
        } 
        else { 
            for j in 0..self.feature_len {
                match self.compare_classes_wilcoxon(j,param.data.feature_maximal_pvalue,
                    param.data.feature_minimal_prevalence_pct as f64/100.0, param.data.feature_minimal_feature_value) {
                    0 => {self.feature_selection.push(j); self.feature_class_sign.insert(j, 0);},
                    1 => {self.feature_selection.push(j); self.feature_class_sign.insert(j, 1);},
                    _ => {}
                }
            }
        };

        
        
    }

    /// filter Data for some samples (represented by a Vector of indices)
    pub fn subset(&self, indices: Vec<usize>) -> Data {
        let mut X: HashMap<(usize,usize),f64> = HashMap::new();
        for i in indices.iter() {
            for j in 0..self.feature_len {
                if self.X.contains_key(&(*i,j)) {
                    X.insert((*i,j), self.X[&(*i,j)]);
                }
            }
        }

        Data {
            X: X,
            y: indices.iter().map(|i| {self.y[*i]}).collect(),
            features: self.features.clone(),
            samples: indices.iter().map(|i| {self.samples[*i].clone()}).collect(),
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
            feature_class_sign: self.feature_class_sign.clone(),
            feature_selection: self.feature_selection.clone(),
            feature_len: self.feature_len,
            sample_len: self.sample_len
        }
    }

    pub fn clone_with_new_x(&self, X:HashMap<(usize, usize), f64>) -> Data {
        Data {
            X: X,
            y: self.y.clone(),
            features: self.features.clone(),
            samples: self.samples.clone(),
            feature_class_sign: self.feature_class_sign.clone(),
            feature_selection: self.feature_selection.clone(),
            feature_len: self.feature_len,
            sample_len: self.sample_len
        }
    }

    pub fn add(&mut self, other: &Data) {
        self.samples.extend_from_slice(&other.samples);
        self.y.extend_from_slice(&other.y);
        for j in 0..self.feature_len {
            for i in 0..other.sample_len {
                if other.X.contains_key(&(i,j)) {
                    self.X.insert((i+self.sample_len,j), other.X[&(i,j)]);
                }
            }
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
        for j in (0..self.feature_len).take(20) {
            let feature = &self.features[j]; // Use the feature name from self.features 
            let row_display: String = (0..self.sample_len)
                .map(|i| {if self.X.contains_key(&(i,j)) {format!("{:.2}",self.X[&(i,j)])} else {"".to_string()}})
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