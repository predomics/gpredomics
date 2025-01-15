use crate::utils::{generate_random_vector,shuffle_row};
use crate::data::Data;
use rand::seq::SliceRandom; // Provides the `choose_multiple` method
use std::collections::{HashMap,BTreeMap};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::fmt;
use std::cmp::min;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Clone)]
pub struct Individual {
    pub features: HashMap<usize,i8>, /// a vector of feature indices with their corresponding signs
    pub auc: f64, // accuracy of the model
    pub fit: f64, // fit value of the model
    pub specificity: f64,
    pub sensitivity: f64,
    pub accuracy: f64,
    pub threshold: f64,
    pub k: usize, // nb of variables used
    pub epoch: usize, // generation or other counter important in the strategy 
    pub language: u8, // binary (0,1), ternary (-1,0,1), pow2 (-4,-2,-1,0,1,2,4), ratio (-1,-1,-1,81)
    pub data_type: u8, // abundance (raw), prevalence (0,1), log
    pub hash: u64,
    pub data_type_minimum: f64
}

pub const BINARY_LANG :u8 = 0;
pub const TERNARY_LANG :u8 = 1;
pub const POW2_LANG :u8 = 2;
pub const RATIO_LANG :u8 = 3;
pub const RAW_TYPE :u8 = 0;
pub const PREVALENCE_TYPE :u8 = 1;
pub const LOG_TYPE :u8 = 2;
pub const DEFAULT_MINIMUM :f64 = f64::MIN_POSITIVE;

const DEFAULT_POW2_START :u8 = 4;

pub fn language(language_string: &str) -> u8 {
    match language_string.to_lowercase().as_str() {
        "binary"|"bin" => BINARY_LANG,
        "ternary"|"ter" => TERNARY_LANG,
        "pow2" => POW2_LANG,
        "ratio" => RATIO_LANG,
        other => panic!("Unrecognized language {}", other)
    }
}

pub fn data_type(data_type_string: &str) -> u8 {
    match data_type_string.to_lowercase().as_str() {
        "raw" => RAW_TYPE,
        "prevalence"|"prev" => PREVALENCE_TYPE,
        "log" => LOG_TYPE,
        other => panic!("Unrecognized data type {}", other)
    }
}


impl Individual {
    /// Provides a help message describing the `Individual` struct and its fields.
    pub fn help() -> &'static str {
        "
        Individual Struct:
        -----------------
        Represents an individual entity with a set of attributes or features.

        Fields:
        - feature_indices: HashMap<u32,u8>
            A map between feature indices (u32) and their corresponding signs (u8).
            This represents the features present in the individual, with their signs indicating 
            the direction of the relationship with the target variable.

        - feature_names: Vec<String>
            A vector containing the names of the features present in the individual.
            This provides a human-readable representation of the features.

        - fit_method: String
            A string representing the method used to evaluate the fitness of the individual.
            This could be 'AUC', 'accuracy', or any other evaluation metric.

        - accuracy: f64
            A floating-point number representing the accuracy of the model represented by the individual.
            This value indicates how well the model performs on the given data.
        "
    }

    /// a generic creator for Individual
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
            data_type_minimum: DEFAULT_MINIMUM
        }
    }

    pub fn compute_hash(&mut self) {
        let mut hasher = DefaultHasher::new();
        
        // Convert HashMap to a sorted representation
        let sorted_features: BTreeMap<_, _> = self.features.iter().collect();
        sorted_features.hash(&mut hasher); // Hash the sorted features
        
        self.hash = hasher.finish();
    }

    /// a specific creator in generation context
    /// the "main" parent is the one that gives its language and data_type (the "other" parent contributes only in genes)
    pub fn child(main_parent: &Individual) -> Individual {
        let mut i=Individual::new();
        i.language = main_parent.language;
        i.data_type = main_parent.data_type;
        i.data_type_minimum = main_parent.data_type_minimum;
        i
    }

    pub fn evaluate(&self, d: &Data) -> Vec<f64> {
        self.evaluate_from_features(&d.X, d.sample_len)
    }

    pub fn evaluate_from_features(&self, X: &HashMap<(usize,usize),f64>, sample_len: usize) -> Vec<f64> {
        match self.data_type {
            RAW_TYPE => self.evaluate_raw(X, sample_len),
            PREVALENCE_TYPE => self.evaluate_prevalence(X, sample_len),
            LOG_TYPE => self.evaluate_log(X, sample_len),
            other => panic!("Unknown data-type {}",other)
        }
    }

    fn evaluate_raw(&self, X: &HashMap<(usize,usize),f64>, sample_len: usize) -> Vec<f64> {
        let mut score=vec![0.0; sample_len];

        if self.language == RATIO_LANG {
            let mut r: Vec<Vec<f64>> = vec![vec![0.0,0.0]; sample_len];
            for (feature_index,coef) in self.features.iter() {
                let part = if *coef>0 {0} else {1};
                for sample in 0..sample_len {
                    r[sample][part] += X.get(&(sample,*feature_index)).unwrap_or(&0.0);
                }
            }
            for sample in 0..sample_len {
                score[sample]=r[sample][0]/r[sample][1].max(self.data_type_minimum);
            }
        } else {
            for (feature_index,coef) in self.features.iter() {
                let x_coef = *coef as f64;
                for sample in 0..sample_len {
                        score[sample] += X.get(&(sample,*feature_index)).unwrap_or(&0.0) * x_coef;
                }
            }
        }

        score
    }

    fn evaluate_prevalence(&self, X: &HashMap<(usize,usize),f64>, sample_len: usize) -> Vec<f64> {
        let mut score=vec![0.0; sample_len];

        if self.language == RATIO_LANG {
            let mut r: Vec<Vec<f64>> = vec![vec![0.0,0.0]; sample_len];
            for (feature_index,coef) in self.features.iter() {
                let part = if *coef>0 {0} else {1};
                for sample in 0..sample_len {
                    r[sample][part] += if X.get(&(sample,*feature_index)).unwrap_or(&0.0)>&self.data_type_minimum {1.0} else {0.0};
                }
            }
            for sample in 0..sample_len {
                score[sample]=r[sample][0]/r[sample][1].max(self.data_type_minimum);
            }
        } else {
            for (feature_index,coef) in self.features.iter() {
                let x_coef = *coef as f64;
                for sample in 0..sample_len {
                        score[sample] += if X.get(&(sample,*feature_index)).unwrap_or(&0.0)>&self.data_type_minimum {1.0} else {0.0} * x_coef;
                }
            }
        }
        
        score
    }

    fn evaluate_log(&self, X: &HashMap<(usize,usize),f64>, sample_len: usize) -> Vec<f64> {
        let mut score=vec![0.0; sample_len];

        if self.language == RATIO_LANG {
            let mut r: Vec<Vec<f64>> = vec![vec![0.0,0.0]; sample_len];
            for (feature_index,coef) in self.features.iter() {
                let part = if *coef>0 {0} else {1};
                for sample in 0..sample_len {
                    r[sample][part] += X.get(&(sample,*feature_index)).unwrap_or(&0.0).max(self.data_type_minimum).ln()
                }
            }
            for sample in 0..sample_len {
                score[sample]=r[sample][0]/r[sample][1].max(self.data_type_minimum);
            }
        } else {
            for (feature_index,coef) in self.features.iter() {
                let x_coef = *coef as f64;
                for sample in 0..sample_len {
                        score[sample] += X.get(&(sample,*feature_index)).unwrap_or(&0.0)
                                .max(self.data_type_minimum).ln() * x_coef;
                }
            }
        }

        score
    }


    /// Compute AUC based on the target vector y
    pub fn compute_auc(&mut self, d: &Data) -> f64 {
        let value = self.evaluate(d);
        self.compute_auc_from_value(value, &d.y)
    }

    /// Compute AUC based on X and y rather than a complete Data object
    pub fn compute_auc_from_features(&mut self, X: &HashMap<(usize,usize),f64>, sample_len: usize, y: &Vec<u8>) -> f64 {
        let value = self.evaluate_from_features(X, sample_len);
        self.compute_auc_from_value(value, y)
    }

    /// Compute AUC based on the target vector y
    fn compute_auc_from_value(&mut self, value: Vec<f64>, y: &Vec<u8>) -> f64 {
        let mut thresholds: Vec<(usize,&f64)> = value.iter().enumerate().collect::<Vec<(usize,&f64)>>();

        thresholds.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut auc = 0.0;
        let mut roc_points = Vec::new();


        let mut tp:usize = 0;
        let mut fp:usize = 0;
        let mut tn:usize = 0;
        let mut fn_count:usize=0;

        for y_val in y.iter() {
            if *y_val == 0 {fp += 1}
            else if *y_val == 1 {tp += 1}
        }
        
        for (i,_) in thresholds {
            //let (tp, fp, tn, fn_count) = self.calculate_confusion_matrix(&value, &d.y, threshold);

            let tpr = if (tp + fn_count) > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };

            let fpr = if (fp + tn) > 0 {
                fp as f64 / (fp + tn) as f64
            } else {
                0.0
            };

            roc_points.push((fpr, tpr));
            
            if y[i] == 1 { tp-=1; fn_count+=1 }
            else if y[i] == 0 { tn+=1; fp-=1 }
        }

        let tpr = if (tp + fn_count) > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };

        let fpr = if (fp + tn) > 0 {
            fp as f64 / (fp + tn) as f64
        } else {
            0.0
        };

        roc_points.push((fpr, tpr));

        // Sort points by FPR to ensure proper order for AUC calculation
        roc_points.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap().then(a.1.partial_cmp(&b.1).unwrap())
        });

        // Compute AUC using trapezoidal rule
        for i in 1..roc_points.len() {
            let (prev_fpr, prev_tpr) = roc_points[i - 1];
            let (fpr, tpr) = roc_points[i];

            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
        }
        self.auc = auc;
        auc
    } 

    /// Calculate the confusion matrix at a given threshold
    pub fn calculate_confusion_matrix(&self,data: &Data) -> (usize, usize, usize, usize) {
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

    pub fn count_k(&mut self) {
        self.k = self.features.len();
    }

    /// completely random individual, not very usefull
    pub fn random(d: &Data, rng: &mut ChaCha8Rng) -> Individual {

        let mut features: HashMap<usize,i8> = HashMap::new(); 
        for (i,coef) in generate_random_vector(d.feature_len, rng).iter().enumerate() {
            if *coef!=0 {features.insert(i, *coef);}
        }
        
        let mut i = Individual::new();
        i.features = features;
        i.k = i.features.len();
        i
    }

    /// randomly generated individual amoung the selected features
    pub fn random_select_k(kmin: usize, kmax:usize, feature_selection: &Vec<usize>, feature_class: &HashMap<usize,u8>, 
                            language: u8, data_type: u8, data_type_minimum: f64, rng: &mut ChaCha8Rng) -> Individual {
        // chose k variables amount feature_selection
        // set a random coeficient for these k variables
    
        let chosen_feature_set: &Vec<usize> = if language==BINARY_LANG {
            &feature_selection.iter()
                    .cloned()
                    .filter(|i| {feature_class[i]>0})
                    .collect::<Vec<usize>>()
        } else {
            feature_selection
        };

        let k: usize= rng.gen_range((if kmin>0 {kmin} else {1})..(if kmax>0 {min(kmax, chosen_feature_set.len())} else {chosen_feature_set.len()}));

        // Randomly pick k values
        let random_values = chosen_feature_set.choose_multiple(rng, k as usize);

        let features: HashMap<usize,i8> = match language {
            BINARY_LANG => random_values.collect::<Vec<&usize>>().iter().map(|i| {(**i,1)}).collect(),
            POW2_LANG => random_values.collect::<Vec<&usize>>().iter()
                                    .map(|i| {(**i,
                                        (if feature_class[i]>0 {1} else {-1}) * DEFAULT_POW2_START as i8)}).collect(),
            _ => random_values.collect::<Vec<&usize>>().iter()
                            .map(|i| {(**i,if feature_class[i]>0 {1} else {-1})}).collect()
        };
        
        let mut i = Individual::new();
        i.features = features;
        if language==RATIO_LANG { i.threshold = 1.0}
        i.k = k;
        i.language = language;
        i.data_type = data_type;
        i.data_type_minimum = data_type_minimum;
        i

    }
    

    // write a function fit_model that takes in the data and computes all the following fields

    // write a function evaluate_contingency_table that takes in the data and evaluates the contingency table of the model

    /// a function that compute accuracy,precision and sensitivity, fixing the threshold using Youden index 
    /// return (threshold, accuracy, sensitivity, specificity)
    pub fn compute_threshold_and_metrics(&self, d: &Data) -> (f64, f64, f64, f64) {
        let value = self.evaluate(d); // Predicted probabilities
        let mut combined: Vec<(f64, u8)> = value.iter().cloned().zip(d.y.iter().cloned()).collect();
        
        // Sort by predicted probabilities
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Initialize confusion matrix
        let mut tp = d.y.iter().filter(|&&label| label == 1).count();
        let mut fn_count = 0;
        let mut tn = 0;
        let mut fp = d.y.iter().filter(|&&label| label == 0).count();

        let mut best_threshold = 0.0;
        let mut best_youden_index = f64::NEG_INFINITY;
        let mut best_metrics = (0.0, 0.0, 0.0); // (Accuracy, Sensitivity, Specificity)

        for i in 0..combined.len() {
            let (threshold, label) = combined[i];

            // Update confusion matrix based on current threshold
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

            // Calculate metrics
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
        }

        // Add a small offset to the threshold for precision handling
        (best_threshold, best_metrics.0, best_metrics.1, best_metrics.2)
    }


    /// return the index of features used in the individual
    fn features_index(&self) -> Vec<usize> {
        let mut features = self.features.keys().copied().collect::<Vec<usize>>();
        features.sort();
        features
    }

    /// Compute OOB feature importance by doing N permutations on samples on a feature (for each feature)
    /// uses mean decreased AUC
    pub fn compute_oob_feature_importance(&mut self, data: &Data, permutations: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
        let model_features = self.features_index();
        let mut importances = vec![0.0; model_features.len()]; // One importance value per feature
        let baseline_auc = self.compute_auc(data); // Baseline AUC

        for (i,feature_idx) in model_features.iter().enumerate() {
            let mut permuted_auc_sum = 0.0;

            for _ in 0..permutations {
                // Clone the feature matrix and shuffle the current feature row
                let mut X_permuted = data.X.clone();
                //X_permuted[*feature_idx].shuffle(rng);
                shuffle_row(&mut X_permuted, data.sample_len, *feature_idx, rng);

                // Recompute AUC with the permuted feature
                let permuted_auc = self.compute_auc_from_features(&X_permuted, data.sample_len, &data.y);
                permuted_auc_sum += permuted_auc;
            }

            // Compute the average AUC with permutations
            let mean_permuted_auc = permuted_auc_sum / permutations as f64;

            // Importance: how much the AUC drops due to shuffling
            importances[i] = baseline_auc - mean_permuted_auc;
        }

        importances
    }

    pub fn maximize_objective(&mut self, data: &Data, fpr_penalty: f64, fnr_penalty: f64) -> f64 {
        let scores = self.evaluate(data);
    
        // Step 2: Extract unique thresholds from scores
        let mut thresholds: Vec<f64> = scores.clone();
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        thresholds.dedup();

        let mut best_objective: f64 = f64::MIN;

        // Step 3: Compute metrics for each threshold
        for &threshold in thresholds.iter() {
            let mut tp = 0; // True Positives
            let mut tn = 0; // True Negatives
            let mut fp = 0; // False Positives
            let mut fn_count = 0; // False Negatives
            
            for (i, &score) in scores.iter().enumerate() {
                let is_positive = data.y[i]; // Assume binary labels: 1 for positive, 0 for negative
                if score >= threshold {
                    if is_positive == 1 {
                        tp += 1;
                    } else {
                        fp += 1;
                    }
                } else {
                    if is_positive == 1 {
                        fn_count += 1;
                    } else {
                        tn += 1;
                    }
                }
            }

            // Compute metrics
            let sensitivity = if tp + fn_count > 0 {
                tp as f64 / (tp + fn_count) as f64
            } else {
                0.0
            };
            let specificity = if tn + fp > 0 {
                tn as f64 / (tn + fp) as f64
            } else {
                0.0
            };
            // Compute the objective: specificity - fnr_penalty * FNR
            let objective = fpr_penalty * specificity - fnr_penalty * (1.0 - sensitivity);

            // Update best metrics if objective improves
            if objective > best_objective {
                best_objective = objective;
                self.sensitivity = sensitivity;
                self.specificity = specificity;
                self.threshold = threshold;
                self.accuracy = (tp+tn) as f64 / (tp+tn+fp+fn_count) as f64;
            }
        }
        best_objective
    }

    pub fn get_language(&self) -> &str {
        match self.language {
            BINARY_LANG => "Binary",
            TERNARY_LANG => "Ternary",
            RATIO_LANG => "Ratio",
            POW2_LANG => "Pow2",
            _ => "Unknown"
        }
    }

    pub fn get_data_type(&self) -> &str {
        match self.data_type {
            RAW_TYPE => "Raw",
            PREVALENCE_TYPE => "Prevalence",
            LOG_TYPE => "Log",
            _ => "Unknown"
        }
    }

}

impl fmt::Debug for Individual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut sorted_keys: Vec<usize> = self.features.keys().cloned().collect();
        sorted_keys.sort();
        let mut desc = sorted_keys.iter()
                .map(|i| {
                    match self.features[i] {
                        1 => format!("[{}] ",i+1),
                        -1 => format!("-[{}] ",i+1),
                        other => format!("{}[{}] ",other,i+1)
                    }
                }).collect::<Vec<String>>().join("");
        if desc.len()>0 { desc=desc[0..desc.len()-1].to_string() }
        write!(f, "{}:{} {}", self.get_language(), self.get_data_type(), desc)
    }
}

// Safe implementation of Send and Sync
unsafe impl Send for Individual {}
unsafe impl Sync for Individual {}

/// When a parent has a child of a different language, do we need to convert the gene values ?
pub fn needs_conversion(parent_language: u8, child_language: u8) -> bool {
    match (parent_language, child_language) {
        (x, y) if x==y => false,
        (BINARY_LANG,_) => false,
        (_,BINARY_LANG) => true,
        (TERNARY_LANG,_) => false,
        (RATIO_LANG,_) => false,
        _ => true
    }
}

/// A conversion function for interlanguage wedding
pub fn gene_convert_from_to(parent_language: u8, child_language: u8, value: i8) -> i8 {
    match (parent_language, child_language) {
        (_, BINARY_LANG) => 1,
        (_, TERNARY_LANG)|(_, RATIO_LANG) => if value>0 {1} else {-1},
        _ => value
    }
}