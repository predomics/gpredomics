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

    pub fn display(&self, data: &Data, data_to_test: Option<&Data>, algo: &String, level: usize, beautiful: bool) -> String {
        let algo_str;
        if algo == "ga" {
            algo_str = format!(" [gen:{}] ", self.epoch);
        } else {
            algo_str = format!(" ");
        }

        let metrics;
        match data_to_test {
            Some(test_data) => { 
                let (_, acc_test, se_test, sp_test) = self.compute_threshold_and_metrics(test_data);
                if beautiful == true {
                    metrics = format!("{}:{} [k={}]{}[fit:{:.3}] AUC {:.3}/{:.3} | accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3}",
                                  self.get_language(), self.get_data_type(), self.features.len(), algo_str, self.fit, self.auc, self.compute_new_auc(test_data), self.accuracy, acc_test, 
                                  self.sensitivity, se_test, self.specificity, sp_test)
                } else {
                    metrics = format!("{}:{} k={}]{}[fit:{:.3}] AUC {:.3}/{:.3} | accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3}",
                                  self.get_language(), self.get_data_type(), self.features.len(), algo_str, self.fit, self.auc, self.compute_new_auc(test_data), self.accuracy, acc_test, 
                                  self.sensitivity, se_test, self.specificity, sp_test)
                }
            },
    
            None => {
                if beautiful == true {
                    metrics = format!("{}:{} [k={}]{}[fit:{:.3}] AUC {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3}",
                    self.get_language(), self.get_data_type(), self.features.len(), algo_str, self.fit, self.auc, self.accuracy, self.sensitivity, self.specificity)
                } else {
                    metrics = format!("{}:{} [k={}]{}[fit:{:.3}] AUC {:.3} | accuracy {:.3} | sensitivity {:.3} | specificity {:.3}",
                    self.get_language(), self.get_data_type(), self.features.len(), algo_str, self.fit, self.auc, self.accuracy, self.sensitivity, self.specificity)
                }
            }
                
        }
        

        // Sort features by index
        let mut sorted_features: Vec<_> = self.features.iter().collect();
        sorted_features.sort_by(|a, b| a.0.cmp(b.0));
    
        let mut positive_features: Vec<_> = sorted_features.iter().filter(|&&(_, &coef)| coef > 0).collect();
        let mut negative_features: Vec<_> = sorted_features.iter().filter(|&&(_, &coef)| coef < 0).collect();
    
        positive_features.sort_by(|a, b| b.1.cmp(a.1));
        negative_features.sort_by(|a, b| b.1.cmp(a.1));
    
        let mut positive_str: Vec<String> = positive_features.iter().enumerate().map(|(i, &&(index, coef))| {
            let mut str;
            if level == 0 && beautiful == true {
                str = format!("\x1b[96mF_POS_{}\x1b[0m", i)
            } else if level == 0 && beautiful == false {
                str = format!("F_POS_{}", i)
            } else if level == 1 && beautiful == true {
                str = format!("\x1b[96m[{}]\x1b[0m", index.to_string())
            } else if level == 1 && beautiful == false {
                str = format!("[{}]", index.to_string())
            } else if level == 2 && beautiful == true {
                str = format!("\x1b[96m{}\x1b[0m", data.features[*index])
            } else {
                str = data.features[*index].clone()
            }
            if self.language == POW2_LANG && !(*coef == 1_i8) && self.language != LOG_TYPE {
                str = format!("{}*{}", coef, str);
            } else if self.language == POW2_LANG && !(*coef == 1_i8) && self.language == LOG_TYPE {
                // b*ln(a) == ln(a^b)
                str = format!("{}^{}", str, coef);
            }
            str
        }).collect();
    
        let mut negative_str: Vec<String> = negative_features.iter().enumerate().map(|(i, &&(index, coef))| {
            let mut str;
            if level == 0 && beautiful == true {
                    str = format!("\x1b[95mF_NEG_{}\x1b[0m", i)
                } else if level == 0 && beautiful == false {
                    str = format!("F_NEG_{}", i)
                } else if level == 1 && beautiful == true {
                    str = format!("\x1b[95m[{}]\x1b[0m", index.to_string())
                } else if level == 1 && beautiful == false {
                    str = format!("[{}]", index.to_string())
                } else if level == 2 && beautiful == true {
                    str = format!("\x1b[95m{}\x1b[0m", data.features[*index])
                } else {
                    str = data.features[*index].clone()
                }
            if self.language == POW2_LANG && !(*coef == -1_i8) && self.language != LOG_TYPE {
                str = format!("{}*{}", coef, str);
            } else if self.language == POW2_LANG && !(*coef == -1_i8) && self.language == LOG_TYPE {
                // b*ln(a) == ln(a^b)
                // absolute coeff as minus is before ln() -> ln(prod(pos)) - ln(prod(neg)) = threshold + ln(prod(data.type_minimum^coeff)
                str = format!("{}^{}", str, coef.abs());
            }
            str
        }).collect();
    
        if self.language == RATIO_LANG {
            negative_str.push("1e-12".to_string());
        }

        let threshold;
        let positive_str_joined;
        let negative_str_joined;
        if self.data_type == LOG_TYPE && self.language != RATIO_LANG {
            // Calculate the product of data_type_minimum raised to the power of each coefficient
            let product: f64 = self.features.values().map(|&coef| self.data_type_minimum.powi(coef as i32)).product();
            threshold = format!("{} (+ {})", self.threshold, product.ln());
            positive_str_joined = format!("ln({})", positive_str.join(" * "));
            negative_str_joined = format!("ln({})", negative_str.join(" * "));
        } else {
            threshold = format!("{}", self.threshold);
            positive_str_joined = format!("({})", positive_str.join(" + "));
            negative_str_joined = format!("({})", negative_str.join(" + "));
        }

        if positive_str.len() == 0 {
            positive_str.push("0".to_string());
        }
        if negative_str.len() == 0 {
            negative_str.push("0".to_string());
        }
        let predicted_class;
        if beautiful == true {
            predicted_class = format!("{}", data.classes[1]);
        } else {
            predicted_class = format!("{}", data.classes[1]);
        }
    
        let formatted_string;
        if self.language == BINARY_LANG && (level == 0 || level == 1 || level == 2) {
            formatted_string = format!("{}\nClass {} <======> {} > {}", metrics, predicted_class, positive_str_joined, threshold)
        } else if (self.language == TERNARY_LANG || self.language == POW2_LANG) && (level == 0 || level == 1 || level == 2) {
            formatted_string = format!("{}\nClass {} <======> {} - {} > {}", metrics, predicted_class, positive_str_joined, negative_str_joined, threshold)
        } else if self.language == RATIO_LANG && (level == 0 || level == 1 || level == 2) {
            formatted_string = format!("{}\nClass {} <======> {} / {} > {}", metrics, predicted_class, positive_str_joined, negative_str_joined, threshold)
        } else {
            formatted_string = format!("{}\nClass {} <======> {:?} > {}", metrics, predicted_class, self, threshold);
        };
    
        formatted_string
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
                score[sample]=r[sample][0]/(r[sample][1]+1e-12);
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
                score[sample]=r[sample][0]/(r[sample][1]+1e-12);
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
                    if let Some(val)=X.get(&(sample,*feature_index)) {
                        r[sample][part] += (val/self.data_type_minimum).ln() * coef.abs() as f64;
                    }
                }
            }
            for sample in 0..sample_len {
                score[sample]=r[sample][0]/(r[sample][1]+1e-12);
            }
        } else {
            for (feature_index,coef) in self.features.iter() {
                let x_coef = *coef as f64;
                for sample in 0..sample_len {
                    if let Some(val)=X.get(&(sample,*feature_index)) {
                        score[sample] += (val/self.data_type_minimum).ln() * x_coef ;
                    }
                }
            }
        }

        score
    }


    /// Compute AUC based on the target vector y
    pub fn compute_auc(&mut self, d: &Data) -> f64 {
        let value = self.evaluate(d);
        self.auc = self.compute_auc_from_value(&value, &d.y);
        self.auc
    }

    // Compute AUC without changing self.auc
    pub fn compute_new_auc(&self, d: &Data) -> f64 {
        let value = self.evaluate(d);
        self.compute_auc_from_value(&value, &d.y)
    }

    /// Compute AUC based on X and y rather than a complete Data object
    pub fn compute_auc_from_features(&mut self, X: &HashMap<(usize,usize),f64>, sample_len: usize, y: &Vec<u8>) -> f64 {
        let value = self.evaluate_from_features(X, sample_len);
        self.auc = self.compute_auc_from_value(&value, y);
        self.auc
    }

    /// Compute AUC for binary class using Mann-Whitney U algorithm O(n log n)
    pub fn compute_auc_from_value(&self, value: &[f64], y: &Vec<u8>) -> f64 {
        assert_eq!(value.len(), y.len());
        
        // Count positive and negative examples
        let n = value.len();
        let n1 = y.iter().filter(|&&label| label == 1).count();
        let n2 = n - n1;
        
        if n1 == 0 || n2 == 0 {
            return 0.5;
        }
        
        // Create pairs of (score, label) and sort by score (descending)
        let mut data: Vec<(f64, u8)> = value.iter()
            .zip(y.iter())
            .map(|(&v, &y)| (v, y))
            .collect();
        
        data.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // Calculate U efficiently with a single pass through the sorted array
        let mut u = 0.0;
        let mut pos_so_far = 0;
        let mut i = 0;
        
        while i < n {
            let score = data[i].0;
            
            let mut pos_equal = 0;
            let mut neg_equal = 0;
            
            while i < n && data[i].0 == score {
                if data[i].1 == 1 {
                    pos_equal += 1;
                } else {
                    neg_equal += 1;
                }
                i += 1;
            }
            
            // For each negative with this score:
            // - Add the number of positives with a higher score
            // - Add 0.5 for each positive with the same score
            if neg_equal > 0 {
                u += neg_equal as f64 * pos_so_far as f64; // Positives with higher scores
                u += 0.5 * neg_equal as f64 * pos_equal as f64; // Positives with equal scores
            }
            
            // Update the counter of positives seen so far
            pos_so_far += pos_equal;
        }
        
        u / (n1 as f64 * n2 as f64)
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
    
    /// a function that compute accuracy,precision and sensitivity
    /// return (threshold, accuracy, sensitivity, specificity)
    pub fn compute_metrics(&self, d: &Data) -> (f64, f64, f64) {
        let value = self.evaluate(d); // Predicted probabilities
        let predicted_and_real_class: Vec<(u8, u8)> = value.iter().cloned()
                .map(|x| {if x>=self.threshold {1} else {0}}).zip(d.y.iter().cloned()).collect();
        
        // Initialize confusion matrix
        let mut tp = 0;
        let mut fn_count = 0;
        let mut tn = 0;
        let mut fp = 0;
        for predicted_and_real in predicted_and_real_class.into_iter() {
            match predicted_and_real {
                (_,2) => {},
                (1,1) => tp+=1,
                (1,0) => fp+=1,
                (0,0) => tn+=1,
                (0,1) => fn_count+=1,
                other => panic!("A predicted vs real class of {:?} should not exist",other)
            }
        }

        let sensitivity = tp as f64 / (tp + fn_count) as f64;
        let specificity = tn as f64 / (fp + tn) as f64;
        let accuracy = (tp + tn) as f64 / (tp + tn + fp + fn_count) as f64;

        
        (accuracy,sensitivity,specificity)
    }


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
    
        self.maximize_objective_with_scores(&scores, data, fpr_penalty, fnr_penalty)
    }

    pub fn maximize_objective_with_scores(&mut self, scores: &[f64], data: &Data, fpr_penalty: f64, fnr_penalty: f64) -> f64 {
        // Step 2: Extract unique thresholds from scores
        let mut thresholds: Vec<f64> = scores.iter().cloned().collect();
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
            let objective = (fpr_penalty * specificity + fnr_penalty * sensitivity)/(fnr_penalty+fpr_penalty);

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

// unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, HashMap};
    use rand::prelude::*;

    fn create_test_individual() -> Individual {
        Individual  {features: vec![(0, 1), (1, -1), (2, 1), (3, 0)].into_iter().collect(), auc: 0.4, fit: 0.8, 
        specificity: 0.15, sensitivity:0.16, accuracy: 0.23, threshold: 42.0, k: 42, epoch:42,  language: 0, data_type: 0, hash: 0, 
        data_type_minimum: f64::MIN_POSITIVE}
    }

    fn create_test_data() -> Data {
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
        X.insert((4, 0), 0.9);  // Sample 4, Feature 0
        X.insert((4, 1), 0.01); // Sample 4, Feature 1
        X.insert((5, 0), 0.91); // Sample 5, Feature 0
        X.insert((5, 1), 0.12); // Sample 5, Feature 1
        X.insert((6, 0), 0.75); // Sample 6, Feature 0
        X.insert((6, 1), 0.01); // Sample 6, Feature 1
        X.insert((7, 0), 0.19); // Sample 7, Feature 0
        X.insert((7, 1), 0.92); // Sample 7, Feature 1
        X.insert((8, 0), 0.9);  // Sample 8, Feature 0
        X.insert((8, 1), 0.01); // Sample 8, Feature 1
        X.insert((9, 0), 0.91); // Sample 9, Feature 0
        X.insert((9, 1), 0.12); // Sample 9, Feature 1
        feature_class.insert(0, 0);
        feature_class.insert(1, 1);

        Data {
            X,
            y: vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0], // Vraies étiquettes
            features: vec!["feature1".to_string(), "feature2".to_string()],
            samples: vec!["sample1".to_string(), "sample2".to_string(), "sample3".to_string(),
            "sample4".to_string(), "sample5".to_string(), "sample6".to_string(), "sample7".to_string(), 
            "sample8".to_string(), "sample9".to_string(), "sample10".to_string()],
            feature_class,
            feature_selection: vec![0, 1],
            feature_len: 2,
            sample_len: 10,
            classes: vec!["a".to_string(),"b".to_string()]
        }
    }

    // test for language and data_types
    #[test]
    fn test_language_recognized() {
        assert_eq!(language("binary"), BINARY_LANG, "'binary' misinterpreted");
        assert_eq!(language("BIN"), BINARY_LANG, "'BIN' misinterpreted");
        assert_eq!(language("ternary"), TERNARY_LANG, "'ternary' misinterpreted");
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
        assert_eq!(data_type("prevalEnce"), PREVALENCE_TYPE, "'prevalEnce' misinterpreted");
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
        let mut ind = create_test_individual();
        ind.compute_hash();

        let mut hasher = DefaultHasher::new();
        let sorted_features: BTreeMap<_, _> = ind.features.iter().collect();
        sorted_features.hash(&mut hasher);
        let expected_hash = hasher.finish();

        assert_eq!(ind.hash, expected_hash, "hash is different from expected hash");
    }

    #[test]
    fn test_compute_hash_and_rehash() {
        let mut ind = create_test_individual();
        ind.compute_hash();
        let first_hash = ind.hash;

        ind.features.insert(4, 0);
        ind.compute_hash();

        assert_ne!(ind.hash, first_hash, "hash should be different after adding a new feature");

        let mut hasher = DefaultHasher::new();
        let sorted_features: BTreeMap<_, _> = ind.features.iter().collect();
        sorted_features.hash(&mut hasher);
        let expected_hash = hasher.finish();

        assert_eq!(ind.hash, expected_hash, "hash is different from expected hash");
    }

    // fn child(main_parent: &Individual)

    // fn evaluate(&self, d: &Data)
    #[test]
    fn test_evaluate_and_evaluate_from_features() {
        let mut ind = create_test_individual();
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
        assert_eq!(ind.evaluate_from_features(&X, 2), vec![0.1, -0.6000000000000001]);
        assert_eq!(ind.evaluate(&data), ind.evaluate_from_features(&X, 2), 
        "evaluate() and evaluate_from_features() should return the same result as the first call the second");
        ind.data_type = PREVALENCE_TYPE;
        assert_eq!(ind.evaluate(&data), ind.evaluate_from_features(&X, 2), 
        "evaluate() and evaluate_from_features() should return the same result as the first call the second");
        assert_eq!(ind.evaluate_from_features(&X, 2), vec![1.0, 0.0]);
        ind.data_type = LOG_TYPE;
        assert_eq!(ind.evaluate(&data), ind.evaluate_from_features(&X, 2), 
        "evaluate() and evaluate_from_features() should return the same result as the first call the second");
        assert_eq!(ind.evaluate_from_features(&X, 2), vec![706.09383343927, -1.0986122886680505]);
        assert_eq!(ind.evaluate(&data), ind.evaluate_from_features(&X, 2), 
        "evaluate() and evaluate_from_features() should return the same result as the first call the second");
    }

    // fn child(main_parent: &Individual)
    
    // fn evaluate_raw
    #[test]
    fn test_evaluate_raw_weighted_score() {
        let mut ind: Individual = create_test_individual();
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        ind.data_type = RAW_TYPE;
        
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        X.insert((0, 0), 2.0);
        X.insert((0, 1), 3.0);
        X.insert((1, 0), 4.0);
        X.insert((1, 1), 5.0);

        ind.language = RATIO_LANG;
        assert_eq!(ind.evaluate_raw(&X, 2), vec![2.0 / (3.0+1e-12), 4.0 / (5.0+1e-12)],
                                "bad calculation for raw data scores with ratio Language");
        ind.language = TERNARY_LANG;
        assert_eq!(ind.evaluate_raw(&X, 2), vec![2.0 * 1.0 + 3.0 * -1.0, 4.0 * 1.0 + 5.0 * -1.0],
                                "bad calculation for raw data scores with ter language");
        ind.features = vec![(0, 2), (1, -4)].into_iter().collect();
        ind.language = POW2_LANG;
        assert_eq!(ind.evaluate_raw(&X, 2), vec![2.0 * 2.0 + 3.0 * -4.0, 4.0 * 2.0 + 5.0 * -4.0],
                                "bad calculation for raw data scores with pow2 language");
        ind.features = vec![(0, 1), (1, 0)].into_iter().collect();
        ind.language = BINARY_LANG;
        assert_eq!(ind.evaluate_raw(&X, 2), vec![2.0 * 1.0 + 3.0 * 0.0, 4.0 * 1.0 + 5.0 * 0.0],
                                "bad calculation for raw data scores with bin language");
        
    }

    #[test]
    fn test_evaluate_raw_zero_or_more_sample_len() {
        let ind = create_test_individual();
        let X: HashMap<(usize, usize), f64> = HashMap::new();
        let scores = ind.evaluate_raw(&X, 0);
        assert!(scores.is_empty(), "score should be empty when sample_len=0");
        let scores = ind.evaluate_raw(&X, 10);
        assert_eq!(scores, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "selecting samples outside the range should lead to a score of 0.0");
    
    }

    #[test]
    fn test_evaluate_raw_missing_values() {
        let mut ind = create_test_individual();
        ind.data_type = RAW_TYPE;
        ind.language = TERNARY_LANG;
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        X.insert((0, 0), 2.0);
        // missing value for (0, 1)
        X.insert((1, 0), 4.0);
        X.insert((1, 1), 5.0);

        let scores = ind.evaluate_raw(&X, 2);
        assert_eq!(scores, vec![2.0 * 1.0, 4.0 * 1.0 + 5.0 * (-1.0)], "X missing value should be interpreted as coefficient 0");
        }

    // fn evaluate_prevalence
    #[test]
    fn test_evaluate_prevalence_weighted_score() {
        let mut ind: Individual = create_test_individual();
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        ind.data_type = RAW_TYPE;
        
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        X.insert((0, 0), 2.0);
        X.insert((0, 1), 3.0);
        X.insert((1, 0), 4.0);
        X.insert((1, 1), 5.0);

        ind.language = RATIO_LANG;
        assert_eq!(ind.evaluate_prevalence(&X, 2), vec![1.0 / (1.0+1e-12), 1.0 / (1.0+1e-12)],
                                "bad calculation for prevalence data scores with ratio Language");
        ind.language = TERNARY_LANG;
        assert_eq!(ind.evaluate_prevalence(&X, 2), vec![1.0 - 1.0, 1.0 - 1.0],
                                "bad calculation for prevalence data scores with ter language");
        ind.features = vec![(0, 2), (1, -4)].into_iter().collect();
        ind.language = POW2_LANG;
        assert_eq!(ind.evaluate_prevalence(&X, 2), vec![2.0 - 4.0, 2.0 - 4.0],
                                "bad calculation for prevalence data scores with pow2 language");
        ind.features = vec![(0, 1), (1, 0)].into_iter().collect();
        ind.language = BINARY_LANG;
        assert_eq!(ind.evaluate_prevalence(&X, 2), vec![1.0 * 1.0 + 1.0 * 0.0, 1.0 * 1.0 + 1.0 * 0.0],
                                "bad calculation for prevalence data scores with bin language");
        
    }

    #[test]
    fn test_evaluate_prevalence_zero_or_more_sample_len() {
        let ind = create_test_individual();
        let X: HashMap<(usize, usize), f64> = HashMap::new();
        let scores = ind.evaluate_prevalence(&X, 0);
        assert!(scores.is_empty(), "score should be empty when sample_len=0");
        let scores = ind.evaluate_prevalence(&X, 10);
        assert_eq!(scores, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "selecting samples outside the range should lead to a score of 0.0");
    }

    #[test]
    fn test_evaluate_prevalence_missing_values() {
        let mut ind = create_test_individual();
        ind.data_type = RAW_TYPE;
        ind.language = TERNARY_LANG;
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        X.insert((0, 0), 2.0);
        // missing value for (0, 1)
        X.insert((1, 0), 4.0);
        X.insert((1, 1), 5.0);

        let scores = ind.evaluate_prevalence(&X, 2);
        assert_eq!(scores, vec![1.0 * 1.0, 1.0 * 1.0 + 1.0 * (-1.0)], "X missing value should be interpreted as coefficient 0");
        }

    #[test]
    fn test_evaluate_log_weighted_score() {
        let mut ind: Individual = create_test_individual();
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        ind.data_type = LOG_TYPE;
        
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        X.insert((0, 0), 0.1);
        X.insert((0, 1), 0.75);
        X.insert((1, 0), 0.3);
        X.insert((1, 1), 0.9);

        // Could be interesting to add a is_nan() or is_infinite() verification in evaluate_log        
        ind.language = RATIO_LANG;
        assert_eq!(ind.evaluate_log(&X, 2), vec![(0.1_f64 / ind.data_type_minimum).ln() / ((0.75_f64 / ind.data_type_minimum).ln() +1e-12 ), (0.3_f64 / ind.data_type_minimum).ln() / ((0.9_f64 / ind.data_type_minimum).ln() + 1e-12)],
                                "bad calculation for log data scores with ratio language");
        ind.language = TERNARY_LANG;
        assert_eq!(ind.evaluate_log(&X, 2), vec![(0.1_f64 / ind.data_type_minimum).ln() * 1.0 +(0.75_f64 / ind.data_type_minimum).ln() * -1.0, (0.3_f64 / ind.data_type_minimum).ln() * 1.0 + (0.9_f64 / ind.data_type_minimum).ln() * -1.0],
                                "bad calculation for log data scores with ter language");
        ind.features = vec![(0, 2), (1, -4)].into_iter().collect();
        ind.language = POW2_LANG;
        assert_eq!(ind.evaluate_log(&X, 2), vec![(0.1_f64 / ind.data_type_minimum).ln() * 2.0 + (0.75_f64 / ind.data_type_minimum).ln() * -4.0, (0.3_f64 / ind.data_type_minimum).ln() * 2.0 + (0.9_f64 / ind.data_type_minimum).ln() * -4.0],
                                "bad calculation for log data scores with pow2 language");
                                ind.features = vec![(0, 1), (1, 0)].into_iter().collect();
        ind.language = BINARY_LANG;
        assert_eq!(ind.evaluate_log(&X, 2), vec![(0.1_f64 / ind.data_type_minimum).ln() * 1.0 + (0.75_f64 / ind.data_type_minimum).ln() * 0.0, (0.3_f64 / ind.data_type_minimum).ln() * 1.0 + (0.9_f64 / ind.data_type_minimum).ln() * 0.0],
                                "bad calculation for log data scores with bin language");         
    }

    #[test]
    fn test_evaluate_log_zero_or_more_sample_len() {
        let mut ind = create_test_individual();
        ind.data_type = LOG_TYPE;
        let X: HashMap<(usize, usize), f64> = HashMap::new();
        let scores = ind.evaluate_log(&X, 0);
        assert!(scores.is_empty(), "score should be empty when sample_len=0");
        let scores = ind.evaluate_log(&X, 10);
        assert_eq!(scores, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "selecting samples outside the range should lead to a score of 0.0");
    }

    #[test]
    fn test_evaluate_log_missing_values() {
        let mut ind = create_test_individual();
        ind.data_type = RAW_TYPE;
        ind.language = TERNARY_LANG;
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        X.insert((0, 0), 0.1);
        // missing value for (0, 1)
        X.insert((1, 0), 0.3);
        X.insert((1, 1), 0.9);

        let scores = ind.evaluate_log(&X, 2);
        assert_eq!(scores, vec![(0.1_f64 / ind.data_type_minimum).ln() * 1.0, 
        (0.3_f64 / ind.data_type_minimum).ln() * 1.0 + (0.9_f64 / ind.data_type_minimum).ln() * -1.0], 
        "X missing value should be interpreted as coefficient 0");
        }

    // tests for auc
    #[test]
    fn test_compute_auc() {
        let mut ind = create_test_individual();
        ind.threshold = 0.75;
        let data = create_test_data();
        assert_eq!(0.7380952380952381, ind.compute_auc(&data), "bad calculation for AUC");
        assert_eq!(ind.compute_auc(&data), ind.compute_auc_from_features(&data.X, data.sample_len, &data.y),
        "Individual.compute_auc_from_features(&data.X, &data.sample_len, &data.y) should return the same result as Individual.compute_auc(&data)");
        assert_eq!(ind.compute_auc(&data), ind.compute_auc_from_value(&ind.evaluate(&data), &data.y),
        "Individual.compute_auc_from_value(scores, &data.y) should return the same result as Individual.compute_auc(&data)");
        assert_eq!(0.0, ind.compute_auc_from_value(&vec![0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64], &vec![1_u8, 1_u8, 1_u8, 1_u8, 0_u8]),
        "auc with a perfect classification and class1 < class0 should be 0.0");
        assert_eq!(1.0, ind.compute_auc_from_value(&vec![0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64], &vec![0_u8, 0_u8, 0_u8, 0_u8, 1_u8]),
        "auc with a perfect classification and class0 < class1 should be 1.0");
        assert_eq!(1.0, ind.compute_auc_from_value(&vec![0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64], &vec![0_u8, 0_u8, 0_u8, 0_u8, 1_u8]),
        "auc with a perfect classification and class0 < class1 should be 1.0");
        // maybe add a verification inside compute_auc to avoid below cases ?
        assert_eq!(0.5, ind.compute_auc_from_value(&vec![0.1_f64, 0.2_f64, 0.3_f64, 0.4_f64], &vec![0_u8, 0_u8, 0_u8, 0_u8]),
        "auc should be equal to 0 when there is no positive class");
        assert_eq!(0.5, ind.compute_auc_from_value(&vec![0.5_f64, 0.6_f64, 0.7_f64, 0.8_f64], &vec![1_u8, 1_u8, 1_u8, 1_u8]),
        "auc should be equal to 0 when there is no negative class to avoid positive biais in model selection");
    }

    // fn calculate_confusion_matrix
    #[test]
    fn test_calculate_confusion_matrix_basic() {
        let mut ind = create_test_individual();
        ind.threshold = 0.75;
        let data = create_test_data();
        let confusion_matrix = ind.calculate_confusion_matrix(&data);
        assert_eq!(confusion_matrix.0, 2, "incorrect identification of true positives");
        assert_eq!(confusion_matrix.1, 4, "incorrect identification of false positives");
        assert_eq!(confusion_matrix.2, 3, "incorrect identification of true negatives");
        assert_eq!(confusion_matrix.3, 1, "incorrect identification of false negatives");
    }

    #[test]
    fn test_calculate_confusion_matrix_class_2() {
        let mut ind = create_test_individual();
        ind.threshold = 0.75;
        let mut data = create_test_data();
        data.y = vec![1, 0, 2, 0, 0, 0, 0, 0, 1, 0];
        let confusion_matrix = ind.calculate_confusion_matrix(&data);
        assert_eq!(confusion_matrix.3, 0, "class 2 shoudn't  be classified");
    }

    #[test]
    #[should_panic(expected = "Invalid class label in y: 3")]
    fn test_calculate_confusion_matrix_invalid_class_label() {
        let ind = create_test_individual();
        let mut data = create_test_data();
        data.y = vec![1, 0, 3, 3, 3, 3, 0, 1, 0, 1];
        let confusion_matrix = ind.calculate_confusion_matrix(&data);
        println!("{:?}", confusion_matrix);
    }

    // fn count_k
    #[test]
    fn test_count_k_basic() {
        let mut ind = create_test_individual();
        ind.count_k();
        assert_eq!(ind.features.len(), ind.k, "count_k() should attribute Individual.features.len() as Individual.k");
    }

    #[test]
    fn test_count_k_no_features() {
        let mut ind = Individual::new();
        ind.count_k();
        assert_eq!(ind.features.len(), ind.k, "count_k() should attribute Individual.features.len() as Individual.k");
    }

    #[test]
    fn test_random() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let data = create_test_data();
        let ind = Individual::random(&data, &mut rng);
        // warning : ind.features.len() != data.feature_len as generate_random_vector can return 0 not kept in the Hashmap
        assert!(ind.features.len() <= data.feature_len, "random indivudal features should respect the data feature_len");
        assert_eq!(ind.k, ind.features.len(), "random indivudal k should respect the data feature_len");
        assert_eq!(ind.features, vec![(0, -1), (1, 1)].into_iter().collect(),
        "the generated Individual isn't the same as generated in the past, indicating a reproducibility problem.");
    }

    // fn random_select_k
    #[test]
    fn test_random_select_k() {
        let features = vec![0, 1, 2, 3, 4];
        let mut expected_features = HashMap::new();
        let mut feature_class= HashMap::new();
        feature_class.insert(0, 1);
        feature_class.insert(1, 0);
        feature_class.insert(2, 1);
        feature_class.insert(3, 0);
        feature_class.insert(4, 1);
        expected_features.insert(2, 1);

        // warning : random_select_k never select kmax features
        let mut rng = ChaCha8Rng::seed_from_u64(42); // Seed for reproducibility
        let ind_bin = Individual::random_select_k(2, 3, &features, &feature_class, BINARY_LANG, RAW_TYPE, DEFAULT_MINIMUM, &mut rng);
        let ind_ter = Individual::random_select_k(2, 3, &features, &feature_class, TERNARY_LANG, RAW_TYPE, DEFAULT_MINIMUM, &mut rng);
        let ind_ratio = Individual::random_select_k(2, 3, &features, &feature_class, RATIO_LANG, RAW_TYPE, DEFAULT_MINIMUM, &mut rng);
        let ind_pow2 = Individual::random_select_k(2, 3, &features, &feature_class, POW2_LANG, RAW_TYPE, DEFAULT_MINIMUM, &mut rng);
        
        assert!(ind_bin.features.iter().all(|(key, value)| feature_class.get(key) == Some(&(*value as u8))), "selected k features should be part of input feature_class");
        assert_eq!(ind_bin.language, BINARY_LANG, "input language should be respected");
        assert_eq!(ind_bin.data_type, RAW_TYPE, "input data_type should be respected");
        assert_eq!(ind_bin.data_type_minimum, DEFAULT_MINIMUM, "input data_type_minimum should be respected"); 
        assert!(ind_bin.features.values().all(|&v| vec![0, 1].contains(&v)), "invalid coefficient for BINARY_LANG");
        assert!(ind_ter.features.values().all(|&v| vec![-1, 1].contains(&v)), "invalid coefficient for TERNARY_LANG");
        assert!(ind_ratio.features.values().all(|&v| vec![-1, 1].contains(&v)), "invalid coefficient for RATIO_LANG");
        assert!(ind_pow2.features.values().all(|&v| vec![-4, 4].contains(&v)), "invalid initial coefficient for POW2_LANG");
        assert_eq!(ind_ratio.threshold, 1.0, "new individual created with random_select_k() with a RATIO_LANG should have a threshold of 1.0");

        
        let ind = Individual::random_select_k(0, 3, &features, &feature_class, BINARY_LANG, RAW_TYPE, DEFAULT_MINIMUM, &mut rng);
        assert_eq!(ind.features, expected_features, 
        "the selected features are not the same as selected in the past, indicating a reproducibility problem.");
        // kmin=1 & kmax=1 should return 1 feature and not panic 
        // ind = Individual::random_select_k(1, 1, &features, &feature_class, BINARY_LANG, RAW_TYPE, DEFAULT_MINIMUM, &mut rng);
        
    }

    // fn compute_metrics
    #[test]
    #[should_panic(expected = "A predicted vs real class of (0, 3) should not exist")]
    fn test_compute_metrics_invalid_class() {
        let ind = create_test_individual();
        let mut data = create_test_data();
        data.y = vec![1, 3, 1, 0, 0, 0, 0, 0, 1, 1];
        ind.compute_metrics(&data);
    }

    #[test]
    fn test_compute_metrics_basic() {
        let mut ind = create_test_individual();
        ind.threshold = 0.75;
        let mut data = create_test_data();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];
        let metrics = ind.compute_metrics(&data);
        assert_eq!(0.5_f64, metrics.0, "bad calculation for accuracy");
        assert_eq!(0.6666666666666666_f64, metrics.1, "bad calculation for sensitivity");
        assert_eq!(0.42857142857142855_f64, metrics.2, "bad calculation for specificity");
    }

    #[test]
    fn test_compute_metrics_class2() {
        let mut ind = create_test_individual();
        ind.threshold = 0.75;
        let mut data = create_test_data();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 2];
        assert_eq!((0.5555555555555556_f64, 0.6666666666666666_f64, 0.5_f64), ind.compute_metrics(&data),
        "class 2 should be omitted in calculation")
    }

    #[test]
    fn test_compute_metrics_too_much_y() {
        let mut ind = create_test_individual();
        ind.threshold = 0.75;
        let mut data = create_test_data();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1];
        assert_eq!((0.5_f64, 0.6666666666666666_f64, 0.42857142857142855_f64), ind.compute_metrics(&data),
        "when ind.sample_len < data.sample_len (or y.len() if it does not match), only the ind.sample_len values should be used to calculate its metrics");
    }

    #[test]
    fn test_compute_metrics_not_enough_y() {
        let mut ind = create_test_individual();
        ind.threshold = 0.75;
        let mut data = create_test_data();
        data.y = vec![1, 0, 1, 1];
        assert_eq!((0.25_f64, 0.3333333333333333_f64, 0.0_f64), ind.compute_metrics(&data),
        "when data.sample_len (or y.len() if it does not match) < ind.sample_len, only the data.sample_len values should be used to calculate its metrics");
    }

    // fn compute_threshold_and_metrics
    // threshold = 0.84 according to R ; same metrics as below
    #[test]
    fn test_compute_threshold_and_metrics_basic() {
        let ind = create_test_individual();
        let mut data = create_test_data();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0];
        let results = ind.compute_threshold_and_metrics(&data);
        assert_eq!(0.79_f64, results.0, "bad identification of the threshold");
        assert_eq!(0.8_f64, results.1, "bad calculation for accuracy");
        assert_eq!(0.6666666666666666_f64, results.2, "bad calculation for sensitivity");
        assert_eq!(0.8571428571428571_f64, results.3, "bad calculation for specificity");
    }

    // threshold = 0.84 according to R ; same metrics as below -> need to control if this difference could be a problem
    #[test]
    fn test_compute_threshold_and_metrics_class_2() {
        let ind = create_test_individual();
        let mut data = create_test_data();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 2];
        assert_eq!((0.79_f64, 0.7777777777777778_f64, 0.6666666666666666_f64, 0.8333333333333334_f64), ind.compute_threshold_and_metrics(&data),
        "class 2 should be omitted in calculation");
    }

    //#[test]
    //fn test_compute_threshold_and_metrics_too_much_y() {
    //    let mut ind = create_test_individual();
    //    ind.threshold = 0.75;
    //    let mut data = create_test_data();
    //    data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1];
    //    assert_eq!((0.79_f64, 0.8_f64, 0.6666666666666666_f64, 0.8571428571428571_f64), ind.compute_threshold_and_metrics(&data),
    //    "when ind.sample_len < data.sample_len (or y.len() if it does not match), only the ind.sample_len values should be used to calculate its metrics");
    //}

    #[test]
    fn test_compute_threshold_and_metrics_not_enough_y() {
        let mut ind = create_test_individual();
        ind.threshold = 0.75;
        let mut data = create_test_data();
        data.y = vec![1, 0, 1, 1];
        assert_eq!((0.79_f64, 0.5_f64, 0.3333333333333333_f64, 1.0_f64), ind.compute_threshold_and_metrics(&data),
        "when data.sample_len (or y.len() if it does not match) < ind.sample_len, only the data.sample_len values should be used to calculate its metrics");
    }

    // fn features_index
    #[test]
    fn test_features_index_basic() {
        let mut ind = create_test_individual();
        ind.features.insert(10, 0.42 as i8);
        assert_eq!(vec![0_usize, 1_usize, 2_usize, 3_usize, 10_usize], ind.features_index());
    }

    // fn compute_oob_feature_importance
    #[test]
    fn test_compute_oob_feature_importance_basic() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut ind = create_test_individual();
        let data = create_test_data();
        let importances = ind.compute_oob_feature_importance(&data, 2, &mut rng);
        assert_eq!(importances.len(), ind.features.len(),
        "the number of importances should be the same as the number of features Individual.features.len()");
        for importance in importances.clone() {
            assert!(importance >= 0.0, "importance can not be negative");
        }
        assert_eq!(importances, vec![0.09523809523809534, 0.3928571428571429, 0.0, 0.0],
        "the calculated importances are not the same as calculated in the past for a same seed, indicating a reproducibility problem");
    }

    // fn maximize_objective

    // fn maximize_objective_with_scores
    #[test]
    fn test_maximize_objective_with_scores() {
        let mut ind = create_test_individual();
        let data = create_test_data();
        let scores = ind.evaluate(&data);
        let best_objective = ind.maximize_objective_with_scores(&scores, &data, 1.0, 1.0);

        assert_eq!(ind.maximize_objective(&data, 1.0, 1.0), ind.maximize_objective_with_scores(&scores, &data, 1.0, 1.0),
        "Individual.maximize_objective() and Individual.maximize_objective_with_scores() should return the same results for a same scores vector");
        assert!(best_objective > 0.0, "The best objective should be greater than 0.0");
        assert!(ind.sensitivity >= 0.0 && ind.sensitivity <= 1.0, "Sensitivity should be between 0.0 and 1.0");
        assert!(ind.specificity >= 0.0 && ind.specificity <= 1.0, "Specificity should be between 0.0 and 1.0");
        assert!(ind.accuracy >= 0.0 && ind.accuracy <= 1.0, "Accuracy should be between 0.0 and 1.0");
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
        let mut ind = create_test_individual();
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
        let mut ind = create_test_individual();
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
    fn test_auc_on_more_complicated_data() {
        let mut individual = Individual::new();
        let mut data = Data::new();
        let _ = data.load_data("samples/Qin2014/Xtrain.tsv", "samples/Qin2014/Ytrain.tsv");

        // Set the language and data type
        individual.language = POW2_LANG;
        individual.data_type = LOG_TYPE;
        individual.data_type_minimum = 1e-5;

        // Set the feature indices and their signs
        let feature_indices = vec![
            (9, 1), (22, 1), (23, 1), (24, -1), (42, -1), (47, 1), (57, -1), (66, -1), (72, -1),
            (82, -1), (87, -1), (92, -1), (105, 1), (124, -1), (130, -1), (174, 1), (194, 1),
            (221, 1), (222, 1), (262, 1), (272, 1), (301, -1), (319, 1), (320, -1), (324, 1),
            (334, 1), (359, 1), (378, 2), (436, -1), (466, -1), (468, -1), (476, -1), (488, 1),
            (497, -1), (512, 1), (522, -1), (546, -1), (565, 1), (591, -1), (614, -1), (649, -1),
            (658, 1), (670, -1), (686, 1), (716, 1), (825, 1), (834, -1), (865, -1), (867, 1),
            (874, 1), (877, 1), (1117, 2), (1273, 1), (1313, 1), (1317, 1), (1464, 1), (1525, 1),
            (1629, 1), (1666, 1), (1710, 1), (1735, 1), (1738, 1), (1740, 1), (1741, 1), (1794, 1),
            (1870, 1)
        ];

        for (index, sign) in feature_indices {
            individual.features.insert(index, sign);
        }

        assert_eq!(individual.compute_auc(&data), 0.9641038380325425);
    }
}