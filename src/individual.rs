use serde::{Serialize, Deserialize};
use crate::bayesian_mcmc::Betas;
use crate::utils::{compute_metrics_from_value, generate_random_vector, shuffle_row};
use crate::data::Data;
use rand::seq::SliceRandom; // Provides the `choose_multiple` method
use std::collections::{HashMap,BTreeMap,HashSet, VecDeque};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use std::fmt;
use std::cmp::min;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use crate::Population;
use crate::utils::{compute_auc_from_value, compute_roc_and_metrics_from_value, compute_mcc_and_metrics_from_value};
use crate::experiment::{Importance, ImportanceCollection, ImportanceScope, ImportanceType};
use rand::SeedableRng;
use log::{debug, warn};
use statrs::function::logistic::logistic;
use crate::utils::serde_json_hashmap_numeric;

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
pub struct ThresholdCI {
    pub upper: f64,
    pub lower:  f64,
    pub rejection_rate: f64,
}

#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct Individual {
    #[serde(with = "serde_json_hashmap_numeric::usize_i8")]
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
    pub epsilon: f64,
    pub parents: Option<Vec<u64>>,
    pub betas: Option<Betas>,
    pub threshold_ci: Option<ThresholdCI>,
}

pub const MCMC_GENERIC_LANG :u8 = 101;
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
        "generic"|"mcmc_generic" => MCMC_GENERIC_LANG,
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
            epsilon: DEFAULT_MINIMUM,
            parents: None,
            betas: None,
            threshold_ci: None
        }
    }

    pub fn display(&self, data: &Data, data_to_test: Option<&Data>, algo: &String, level: usize, beautiful: bool, ci_alpha: f64) -> String {
        let algo_str;
        if algo == "ga" {
            algo_str = format!(" [gen:{}] ", self.epoch);
        } else if algo =="beam"{
            algo_str = format!(" ");
        } else if algo == "mcmc" {
            algo_str = format!(" [MCMC step: {}] ", self.epoch);
        } else {
            algo_str = format!(" [unknwon] ")
        }

        let mut metrics;
        match data_to_test {
            Some(test_data) => { 
                let (acc_test, se_test, sp_test, rej_test) = self.compute_metrics(test_data);
                if beautiful == true {
                    metrics = format!("{}:{} [k={}]{}[fit:{:.3}] AUC {:.3}/{:.3} | accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3}",
                                  self.get_language(), self.get_data_type(), self.features.len(), algo_str, self.fit, self.auc, self.compute_new_auc(test_data), self.accuracy, acc_test, 
                                  self.sensitivity, se_test, self.specificity, sp_test)
                } else {
                    metrics = format!("{}:{} [k={}]{}[fit:{:.3}] AUC {:.3}/{:.3} | accuracy {:.3}/{:.3} | sensitivity {:.3}/{:.3} | specificity {:.3}/{:.3}",
                                  self.get_language(), self.get_data_type(), self.features.len(), algo_str, self.fit, self.auc, self.compute_new_auc(test_data), self.accuracy, acc_test, 
                                  self.sensitivity, se_test, self.specificity, sp_test)
                }
                if let Some(ref threshold_ci) = self.threshold_ci {
                    metrics = format!("{} | rejection rate {:.3}/{:.3}", metrics, threshold_ci.rejection_rate, rej_test)
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
                if let Some(ref threshold_ci) = self.threshold_ci {
                    metrics = format!("{} | rejection rate {:3}", metrics, threshold_ci.rejection_rate)
                }       
            }
                
        }
        
        // Sort features by index
        let mut sorted_features: Vec<_> = self.features.iter().collect();
        sorted_features.sort_by(|a, b| a.0.cmp(b.0));
    
        let mut positive_features: Vec<_> = sorted_features.iter().filter(|&&(_, &coef)| coef > 0).collect();
        let mut negative_features: Vec<_> = sorted_features.iter().filter(|&&(_, &coef)| coef < 0).collect();
    
        positive_features.sort_by(|a, b| b.1.cmp(a.1));
        negative_features.sort_by(|a, b| a.1.cmp(b.1));
    
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
            if self.data_type == PREVALENCE_TYPE {
                str = format!("{}⁰", str);
            }
            if self.language == POW2_LANG && !(*coef == 1_i8) && self.data_type != LOG_TYPE {
                str = format!("{}*{}", coef, str);
            } else if self.language == POW2_LANG && !(*coef == 1_i8) && self.data_type == LOG_TYPE {
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
            if self.data_type == PREVALENCE_TYPE {
                str = format!("{}⁰", str);
            }
            if self.language == POW2_LANG && !(*coef == -1_i8) && self.data_type != LOG_TYPE {
                str = format!("{}*{}", coef.abs(), str);
            } else if self.language == POW2_LANG && !(*coef == -1_i8) && self.data_type == LOG_TYPE {
                // b*ln(a) == ln(a^b)
                // absolute coeff as minus is before ln() -> ln(prod(pos)) - ln(prod(neg)) = threshold + ln(prod(data.type_minimum^coeff)
                str = format!("{}^{}", str, coef.abs());
            }
            str
        }).collect();
    
        if self.language == RATIO_LANG {
            negative_str.push(format!("{:2e}", self.epsilon));
        }

        let mut threshold;
        let positive_str_joined;
        let negative_str_joined;
        if self.data_type == LOG_TYPE && self.language != RATIO_LANG {
            // Calculate the product of epsilon raised to the power of each coefficient
            let product: f64 = self.features.values().map(|&coef| self.epsilon.powi(coef as i32)).product();
            threshold = format!("{} (+ {})", self.threshold, product.ln());
            positive_str_joined = format!("ln({})", positive_str.join(" * "));
            negative_str_joined = format!("ln({})", negative_str.join(" * "));
        } else {
            threshold = format!("{}", self.threshold);
            positive_str_joined = format!("({})", positive_str.join(" + "));
            negative_str_joined = format!("({})", negative_str.join(" + "));
        }

        let second_line_first_part; 
        let second_line_second_part; 
        
        if let Some(ref threshold_ci) = self.threshold_ci {
            threshold = format!("Class {} < \x1b[2m[Abstention zone - {:2}% CI: {:.3}, {:.3}, {:.3}]\x1b[0m < Class {}:", data.classes[0], (1.0-ci_alpha)*100.0, threshold_ci.lower,  self.threshold, threshold_ci.upper, data.classes[1]); 
            second_line_first_part = threshold;
            second_line_second_part = "".to_string()
        } else {
            threshold = format!("≥ {}", threshold);
            second_line_first_part = format!("Class {}:", data.classes[1]);
            second_line_second_part = threshold;
        }

        if positive_str.len() == 0 {
            positive_str.push("0".to_string());
        }
        if negative_str.len() == 0 {
            negative_str.push("0".to_string());
        }

    
        let formatted_string;
        if self.language == BINARY_LANG && (level == 0 || level == 1 || level == 2) {
            formatted_string = format!("{}\n{} {} {}", metrics, second_line_first_part, positive_str_joined, second_line_second_part)
        } else if (self.language == TERNARY_LANG || self.language == POW2_LANG) && (level == 0 || level == 1 || level == 2) {
            formatted_string = format!("{}\n{} {} - {} {}", metrics, second_line_first_part, positive_str_joined, negative_str_joined, second_line_second_part)
        } else if self.language == RATIO_LANG && (level == 0 || level == 1 || level == 2) {
            formatted_string = format!("{}\n{} {} / {} {}", metrics, second_line_first_part, positive_str_joined, negative_str_joined, second_line_second_part)
        } else {
            formatted_string = format!("{}\n{} {:?} {}", metrics, second_line_first_part, self, second_line_second_part);
        };
    
        formatted_string
    }

    pub fn compute_hash(&mut self) {
        let mut hasher = DefaultHasher::new();
        
        // Convert HashMap to a sorted representation
        let sorted_features: BTreeMap<_, _> = self.features.iter().collect();
        sorted_features.hash(&mut hasher);
        self.betas.hash(&mut hasher);
        self.hash = hasher.finish();
    }

    /// a specific creator in generation context
    /// the "main" parent is the one that gives its language and data_type (the "other" parent contributes only in genes)
    pub fn child(main_parent: &Individual) -> Individual {
        let mut i=Individual::new();
        if main_parent.threshold_ci.is_some() { i.threshold_ci = Some(ThresholdCI { upper: 0.0, lower: 0.0, rejection_rate: 0.0 }) }
        i.language = main_parent.language;
        i.data_type = main_parent.data_type;
        i.epsilon = main_parent.epsilon;
        i
    }

    pub fn evaluate_class_and_score(&self, d: &Data) -> (Vec<u8>, Vec<f64>) {
        let value = self.evaluate(d);
        let class = value.iter().map(|&v| {
            if let Some(ref threshold_ci) = self.threshold_ci {
                if v > threshold_ci.upper {
                    1
                } else if v < threshold_ci.lower {
                    0
                } else {
                    2  
                }
            } else {
                if v >= self.threshold { 1 } else { 0 }
            }
        }).collect();

        (class, value)
    }

    pub fn evaluate_class(&self, d: &Data) -> Vec<u8> {
        let value = self.evaluate(d);
        value.iter().map(|&v| {
            if let Some(ref threshold_ci) = self.threshold_ci {
                if v > threshold_ci.upper {
                    1
                } else if v < threshold_ci.lower {
                    0
                } else {
                    2  
                }
            } else {
                if v >= self.threshold { 1 } else { 0 }
            }
        }).collect()
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
                score[sample]=r[sample][0]/(r[sample][1]+self.epsilon);
            }
        } else if self.language == MCMC_GENERIC_LANG {
            let betas = self.betas.as_ref().expect("MCMC Individuals must have betas coefficeints");
            let mut pos_sums = vec![0.0; sample_len];
            let mut neg_sums = vec![0.0; sample_len];
            for (feature_index,coef) in self.features.iter() {
                for sample in 0..sample_len {
                    let v = *X.get(&(sample, *feature_index)).unwrap_or(&0.0);
                    match coef {
                        1 => pos_sums[sample] += v,
                        -1 => neg_sums[sample] += v,
                        _ => {}
                    } 
                }
            }

            score = pos_sums.into_iter().zip(neg_sums.into_iter()).map(|(pos, neg)| {
                        let z = pos * betas.a + neg * betas.b + betas.c;
                        logistic(z)
                        }).collect();
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
                    r[sample][part] += if X.get(&(sample,*feature_index)).unwrap_or(&0.0)>&self.epsilon {1.0} else {0.0};
                }
            }
            for sample in 0..sample_len {
                score[sample]=r[sample][0]/(r[sample][1]+self.epsilon);
            }
        } else if self.language == MCMC_GENERIC_LANG {
            let betas = self.betas.as_ref().expect("MCMC Individuals must have betas coefficeints");
            let mut pos_sums = vec![0.0; sample_len];
            let mut neg_sums = vec![0.0; sample_len];
            for (feature_index,coef) in self.features.iter() {
                for sample in 0..sample_len {
                    let v = if X.get(&(sample,*feature_index)).unwrap_or(&0.0)>&self.epsilon {1.0} else {0.0};
                    match coef {
                        1 => pos_sums[sample] += v,
                        -1 => neg_sums[sample] += v,
                        _ => {}
                    } 
                }
            }

            score = pos_sums.into_iter().zip(neg_sums.into_iter()).map(|(pos, neg)| {
                        let z = pos * betas.a + neg * betas.b + betas.c;
                        logistic(z)
                        }).collect();
        } else {
            for (feature_index,coef) in self.features.iter() {
                let x_coef = *coef as f64;
                for sample in 0..sample_len {
                        score[sample] += if X.get(&(sample,*feature_index)).unwrap_or(&0.0)>&self.epsilon {1.0} else {0.0} * x_coef;
                }
            }
        }
        
        score
    }

    fn evaluate_log(&self, X: &HashMap<(usize,usize),f64>, sample_len: usize) -> Vec<f64> {
        // Shouldn't + epsilon be added? 
        let mut score=vec![0.0; sample_len];

        if self.language == RATIO_LANG {
            let mut r: Vec<Vec<f64>> = vec![vec![0.0,0.0]; sample_len];
            for (feature_index,coef) in self.features.iter() {
                let part = if *coef>0 {0} else {1};
                for sample in 0..sample_len {
                    if let Some(val)=X.get(&(sample,*feature_index)) {
                        r[sample][part] += (val/self.epsilon).ln() * coef.abs() as f64;
                    }
                }
            }
            for sample in 0..sample_len {
                score[sample]=r[sample][0]/(r[sample][1]+self.epsilon);
            }
        } else if self.language == MCMC_GENERIC_LANG {
            let betas = self.betas.as_ref().expect("MCMC Individuals must have betas coefficeints");
            let mut pos_sums = vec![0.0; sample_len];
            let mut neg_sums = vec![0.0; sample_len];
            for (feature_index,coef) in self.features.iter() {
                for sample in 0..sample_len {
                    if let Some(v)=X.get(&(sample,*feature_index)) {
                        match coef {
                            1 => pos_sums[sample] += (v/self.epsilon).ln(),
                            -1 => neg_sums[sample] += (v/self.epsilon).ln(),
                            _ => {}
                        } 
                    }
                }
            }

            score = pos_sums.into_iter().zip(neg_sums.into_iter()).map(|(pos, neg)| {
                        let z = pos * betas.a + neg * betas.b + betas.c;
                        logistic(z)
                        }).collect();
                    
        } else {
            for (feature_index,coef) in self.features.iter() {
                let x_coef = *coef as f64;
                for sample in 0..sample_len {
                    if let Some(val)=X.get(&(sample,*feature_index)) {
                        score[sample] += (val/self.epsilon).ln() * x_coef ;
                    }
                }
            }
        }

        score
    }

    /// Compute AUC based on the target vector y
    pub fn compute_auc(&mut self, d: &Data) -> f64 {
        let value = self.evaluate(d);
        self.auc = compute_auc_from_value(&value, &d.y);
        self.auc
    }

    // Compute AUC without changing self.auc
    pub fn compute_new_auc(&self, d: &Data) -> f64 {
        let value = self.evaluate(d);
        compute_auc_from_value(&value, &d.y)
    }

    /// Compute AUC based on X and y rather than a complete Data object
    pub fn compute_auc_from_features(&mut self, X: &HashMap<(usize,usize),f64>, sample_len: usize, y: &Vec<u8>) -> f64 {
        let value = self.evaluate_from_features(X, sample_len);
        self.auc = compute_auc_from_value(&value, y);
        self.auc
    }

    
    // For GpredomicsR, compute metrics and AUC using the same data to be quicker
    // Same results as compute_auc and compute_threshold_and_metrics (different threshold but same metrics)
    // If the fit is not computed on AUC but on objective, metrics are calculated on this objective
    pub fn compute_roc_and_metrics(&mut self, d: &Data, penalties: Option<&[f64]>) -> (f64, f64, f64, f64, f64, f64) {
        let objective;
        let scores: Vec<_> = self.evaluate(d);
        (self.auc, self.threshold, self.accuracy, self.sensitivity, self.specificity, objective) = compute_roc_and_metrics_from_value(&scores, &d.y, penalties);
        (self.auc, self.threshold, self.accuracy, self.sensitivity, self.specificity, objective)        
    }

    pub fn compute_mcc_and_metrics(&mut self, d: &Data) -> (f64, f64, f64, f64, f64) {
        let scores: Vec<_> = self.evaluate(d);
        let mcc;
        (mcc, self.threshold, self.accuracy, self.sensitivity, self.specificity, _) = compute_mcc_and_metrics_from_value(&scores, &d.y, None);
        (mcc, self.threshold, self.accuracy, self.sensitivity, self.specificity)        
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
                            language: u8, data_type: u8, epsilon: f64, threshold_ci: bool, rng: &mut ChaCha8Rng) -> Individual {
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
        if language==RATIO_LANG { i.threshold = 1.0 }
        if threshold_ci {i.threshold_ci = Some(ThresholdCI { upper: 0.0, lower: 0.0, rejection_rate: 0.0 }) }
        i.k = k;
        i.language = language;
        i.data_type = data_type;
        i.epsilon = epsilon;
        i

    }
    
    /// a function that compute accuracy,precision and sensitivity
    /// return (accuracy, sensitivity, specificity)
    pub fn compute_metrics(&self, d: &Data) -> (f64, f64, f64, f64) {
        let value = self.evaluate(d);
        if let Some(ref threshold_ci) = self.threshold_ci {
            compute_metrics_from_value(&value, &d.y, self.threshold, Some([threshold_ci.lower, threshold_ci.upper]))
        } else {
            compute_metrics_from_value(&value, &d.y, self.threshold, None)
        }
    }

    /// a function that compute accuracy,precision and sensitivity, fixing the threshold using Youden index 
    /// return (threshold, accuracy, sensitivity, specificity)
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
    
        (best_threshold, best_metrics.0, best_metrics.1, best_metrics.2)
    }    

    /// return the index of features used in the individual
    pub fn features_index(&self) -> Vec<usize> {
        let mut features = self.features.keys().copied().collect::<Vec<usize>>();
        features.sort();
        features
    }

    /// Compute OOB feature importance by doing N permutations on samples on a feature (for each feature)
    /// uses mean decreased AUC
    pub fn compute_oob_feature_importance(&self, data: &Data, permutations: usize, features_to_process: &[usize], feature_seeds: &HashMap<usize, Vec<u64>>) -> ImportanceCollection {
        let baseline_auc = self.compute_new_auc(data);
        let mut importances = Vec::new();

        // Protection against strange behavior
        if permutations == 0 {
            panic!("compute_oob_feature_importance: To compute OOB calculation, permutations are needed (and currently set to 0)!");
        }

        for &feature_idx in features_to_process {
            if !feature_seeds.contains_key(&feature_idx) {
                panic!("compute_oob_feature_importance: Missing seeds for feature index {}", feature_idx);
            }
            
            let seeds = &feature_seeds[&feature_idx];
            if seeds.len() < permutations {
                panic!("compute_oob_feature_importance: Feature {} has {} seeds but {} permutations requested", feature_idx, seeds.len(), permutations);
            }
        }

        let mut seen = HashSet::new();
        let unique_features: Vec<usize> = features_to_process
            .iter()
            .filter(|&&feature_idx| seen.insert(feature_idx))
            .cloned()
            .collect();
        
        if unique_features.len() != features_to_process.len() {
            debug!("Individual Importance : removed {} duplicate features from analysis", features_to_process.len() - unique_features.len());
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
                    shuffle_row(&mut X_permuted, data.sample_len, feature_idx, &mut permutation_rng);
                    let scores        = self.evaluate_from_features(&X_permuted, data.sample_len); 
                    let permuted_auc  = compute_auc_from_value(&scores, &data.y);                 
                    permuted_auc_sum += permuted_auc;
                }
                
                let mean_permuted_auc = permuted_auc_sum / permutations as f64;
                baseline_auc - mean_permuted_auc
            };
            
            let importance_obj = Importance {
                importance_type: ImportanceType::MDA,
                feature_idx,
                scope: ImportanceScope::Individual { model_hash: self.hash },
                aggreg_method: None, 
                importance: importance_value,
                is_scaled: false,
                dispersion: 0.0, 
                scope_pct: 1.0, 
                direction : None
            };
            
            importances.push(importance_obj);
        }
        
        ImportanceCollection { importances }
    }

    pub fn maximize_objective(&mut self, data: &Data, fpr_penalty: f64, fnr_penalty: f64) -> f64 {
        let scores = self.evaluate(data);
        self.maximize_objective_with_scores(&scores, data, fpr_penalty, fnr_penalty)
    }

    pub fn maximize_objective_with_scores(&mut self, scores: &[f64], data: &Data, fpr_penalty: f64, fnr_penalty: f64) -> f64 {
        let mut paired_data: Vec<_> = scores.iter()
            .zip(data.y.iter())
            .filter(|(_, &y)| y == 0 || y == 1)
            .map(|(&score, &label)| (score, label))
            .collect();
        
        paired_data.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
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
                    _ => unreachable!()
                }
                i += 1;
            }
            
            tn += current_tn;
            fn_count += current_fn;
            
            let tp = total_pos - fn_count;

            let sensitivity = (tp + current_fn) as f64 / total_pos as f64;
            let specificity = (tn - current_tn) as f64 / total_neg as f64;
            let accuracy = (tp + current_fn + tn - current_tn) as f64 / (total_pos + total_neg) as f64;
            
            let objective = (fpr_penalty * specificity + fnr_penalty * sensitivity) / (fpr_penalty + fnr_penalty);
            
            if objective > best_objective || (objective == best_objective && current_score < self.threshold) {
                best_objective = objective;
                self.threshold = current_score;
                self.sensitivity = sensitivity;
                self.specificity = specificity;
                self.accuracy = accuracy;
            }
        }
        
        best_objective
    }
    

    pub fn get_genealogy(&self, collection: &Vec<Population>, max_depth: usize) -> HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> {
        let mut genealogy: HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> = HashMap::with_capacity(max_depth * 2);
        let mut initial_set = HashSet::new();

        if collection.is_empty() {
            return genealogy
        }

        initial_set.insert(0);
        genealogy.insert((self.hash, self.parents.clone()), initial_set);
        
        let real_max_generation = collection.len();
        let effective_max_depth = std::cmp::min(max_depth, real_max_generation);
        
        debug!("Computing genealogy (maximum depth = {:?}, real generations = {:?})...", 
            effective_max_depth, real_max_generation);
        
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
                                genealogy.entry(key.clone()).or_insert_with(HashSet::new).insert(next_depth);
                                let path_key = (ancestor.hash, next_depth);
                                if !visited_paths.contains(&path_key) && next_depth < effective_max_depth {
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
    
    pub fn get_language(&self) -> &str {
        match self.language {
            BINARY_LANG => "Binary",
            TERNARY_LANG => "Ternary",
            RATIO_LANG => "Ratio",
            POW2_LANG => "Pow2",
            MCMC_GENERIC_LANG => "MCMC_Generic",
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

    pub fn get_coef(&self, idx: usize) -> i8 {
        *self.features.get(&idx).unwrap_or(&0)
    }

    pub fn set_coef(&mut self, idx: usize, coef: i8) {
        if coef == 0 {
            self.features.remove(&idx);
        } else {
            self.features.insert(idx, coef);
        }
        self.k = self.features.len();
    }

    pub fn get_betas(&self) -> [f64;3] {
        self.betas.as_ref().map(|b| b.get()).expect("β uninitialized")
    }

    pub fn set_beta(&mut self, idx: usize, val: f64) {
        if let Some(b) = self.betas.as_mut() {
            b.set(idx, val);
        } else {
            self.betas = Some(Betas::new(
                if idx==0 {val} else {0.0},
                if idx==1 {val} else {0.0},
                if idx==2 {val} else {0.0},
            ));
        }
    }

    pub fn signed_jaccard_dissimilarity_with(&self, other: &Individual) -> f64 {
        let signed_set1: HashSet<(usize, i8)> = self.features.iter()
            .map(|(id, coef)| (*id, coef.signum() as i8))
            .collect();
        
        let signed_set2: HashSet<(usize, i8)> = other.features.iter()
            .map(|(id, coef)| (*id, coef.signum() as i8))
            .collect();
        
        let intersection = signed_set1.intersection(&signed_set2).count();
        let union = signed_set1.union(&signed_set2).count();
        
        if union == 0 {
            return 0.0;
        }
        
        1.0 - (intersection as f64) / (union as f64)
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

    impl Individual {
        pub fn test() -> Individual {
            Individual  {features: vec![(0, 1), (1, -1), (2, 1), (3, 0)].into_iter().collect(), auc: 0.4, fit: 0.8, 
            specificity: 0.15, sensitivity:0.16, accuracy: 0.23, threshold: 42.0, k: 42, epoch:42,  language: 0, data_type: 0, hash: 0, 
            epsilon: f64::MIN_POSITIVE, parents: None, betas: None}
        }

        pub fn test2() -> Individual {
            Individual  {features: vec![(0, 1), (1, -1)].into_iter().collect(), auc: 0.4, fit: 0.8, 
            specificity: 0.15, sensitivity:0.16, accuracy: 0.23, threshold: 0.0, k: 42, epoch:42,  language: 0, data_type: 0, hash: 0, 
            epsilon: f64::MIN_POSITIVE, parents: None, betas: None}
        }

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
            }
        }

        pub fn test_with_these_given_features_fit_lang_types(features: Vec<(usize, i8)>, fit: f64, language: u8, data_type: u8,) -> Individual {
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
            }
        }

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
            }
        }

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
            }
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
        let mut ind = Individual::test();
        ind.compute_hash();

        let mut hasher = DefaultHasher::new();
        let sorted_features: BTreeMap<_, _> = ind.features.iter().collect();
        sorted_features.hash(&mut hasher);
        ind.betas.hash(&mut hasher);
        let expected_hash = hasher.finish();

        assert_eq!(ind.hash, expected_hash, "hash is different from expected hash");
    }

    #[test]
    fn test_compute_hash_and_rehash() {
        let mut ind = Individual::test();
        ind.compute_hash();
        let first_hash = ind.hash;

        ind.features.insert(4, 0);
        ind.compute_hash();

        assert_ne!(ind.hash, first_hash, "hash should be different after adding a new feature");

        let mut hasher = DefaultHasher::new();
        let sorted_features: BTreeMap<_, _> = ind.features.iter().collect();
        sorted_features.hash(&mut hasher);
        ind.betas.hash(&mut hasher);
        let expected_hash = hasher.finish();

        assert_eq!(ind.hash, expected_hash, "hash is different from expected hash");
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
        let mut ind: Individual = Individual::test();
        ind.features = vec![(0, 1), (1, -1)].into_iter().collect();
        ind.data_type = RAW_TYPE;
        
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();
        X.insert((0, 0), 2.0);
        X.insert((0, 1), 3.0);
        X.insert((1, 0), 4.0);
        X.insert((1, 1), 5.0);

        ind.language = RATIO_LANG;
        assert_eq!(ind.evaluate_raw(&X, 2), vec![2.0 / (3.0+ ind.epsilon), 4.0 / (5.0+ ind.epsilon)],
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
        let ind = Individual::test();
        let X: HashMap<(usize, usize), f64> = HashMap::new();
        let scores = ind.evaluate_raw(&X, 0);
        assert!(scores.is_empty(), "score should be empty when sample_len=0");
        let scores = ind.evaluate_raw(&X, 10);
        assert_eq!(scores, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "selecting samples outside the range should lead to a score of 0.0");
    
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
        assert_eq!(scores, vec![2.0 * 1.0, 4.0 * 1.0 + 5.0 * (-1.0)], "X missing value should be interpreted as coefficient 0");
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
        assert_eq!(ind.evaluate_prevalence(&X, 2), vec![1.0 / (1.0+ ind.epsilon), 1.0 / (1.0+ ind.epsilon)],
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
        let ind = Individual::test();
        let X: HashMap<(usize, usize), f64> = HashMap::new();
        let scores = ind.evaluate_prevalence(&X, 0);
        assert!(scores.is_empty(), "score should be empty when sample_len=0");
        let scores = ind.evaluate_prevalence(&X, 10);
        assert_eq!(scores, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "selecting samples outside the range should lead to a score of 0.0");
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
        assert_eq!(scores, vec![1.0 * 1.0, 1.0 * 1.0 + 1.0 * (-1.0)], "X missing value should be interpreted as coefficient 0");
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
        assert_eq!(ind.evaluate_log(&X, 2), vec![(0.1_f64 / ind.epsilon).ln() / ((0.75_f64 / ind.epsilon).ln() + ind.epsilon ), (0.3_f64 / ind.epsilon).ln() / ((0.9_f64 / ind.epsilon).ln() + ind.epsilon)],
                                "bad calculation for log data scores with ratio language");
        ind.language = TERNARY_LANG;
        assert_eq!(ind.evaluate_log(&X, 2), vec![(0.1_f64 / ind.epsilon).ln() * 1.0 +(0.75_f64 / ind.epsilon).ln() * -1.0, (0.3_f64 / ind.epsilon).ln() * 1.0 + (0.9_f64 / ind.epsilon).ln() * -1.0],
                                "bad calculation for log data scores with ter language");
        ind.features = vec![(0, 2), (1, -4)].into_iter().collect();
        ind.language = POW2_LANG;
        assert_eq!(ind.evaluate_log(&X, 2), vec![(0.1_f64 / ind.epsilon).ln() * 2.0 + (0.75_f64 / ind.epsilon).ln() * -4.0, (0.3_f64 / ind.epsilon).ln() * 2.0 + (0.9_f64 / ind.epsilon).ln() * -4.0],
                                "bad calculation for log data scores with pow2 language");
                                ind.features = vec![(0, 1), (1, 0)].into_iter().collect();
        ind.language = BINARY_LANG;
        assert_eq!(ind.evaluate_log(&X, 2), vec![(0.1_f64 / ind.epsilon).ln() * 1.0 + (0.75_f64 / ind.epsilon).ln() * 0.0, (0.3_f64 / ind.epsilon).ln() * 1.0 + (0.9_f64 / ind.epsilon).ln() * 0.0],
                                "bad calculation for log data scores with bin language");         
    }

    #[test]
    fn test_evaluate_log_zero_or_more_sample_len() {
        let mut ind = Individual::test();
        ind.data_type = LOG_TYPE;
        let X: HashMap<(usize, usize), f64> = HashMap::new();
        let scores = ind.evaluate_log(&X, 0);
        assert!(scores.is_empty(), "score should be empty when sample_len=0");
        let scores = ind.evaluate_log(&X, 10);
        assert_eq!(scores, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "selecting samples outside the range should lead to a score of 0.0");
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
        assert_eq!(scores, vec![(0.1_f64 / ind.epsilon).ln() * 1.0, 
        (0.3_f64 / ind.epsilon).ln() * 1.0 + (0.9_f64 / ind.epsilon).ln() * -1.0], 
        "X missing value should be interpreted as coefficient 0");
        }

    // tests for auc
    #[test]
    fn test_compute_auc() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        let data = Data::test2();
        assert_eq!(0.7380952380952381, ind.compute_auc(&data), "bad calculation for AUC with compute_auc : this could be a ties issue");
        assert_eq!(0.7380952380952381, ind.compute_roc_and_metrics(&data, None).0, "bad calculation for AUC with compute_roc_and_metrics : this could be a ties issue");
        assert_eq!(ind.compute_auc(&data), ind.compute_auc_from_features(&data.X, data.sample_len, &data.y),
        "Individual.compute_auc_from_features(&data.X, &data.sample_len, &data.y) should return the same result as Individual.compute_auc(&data)");
        assert_eq!(ind.compute_auc(&data), compute_auc_from_value(&ind.evaluate(&data), &data.y),
        "Individual.compute_auc_from_value(scores, &data.y) should return the same result as Individual.compute_auc(&data)");
        assert_eq!(0.0, compute_auc_from_value(&vec![0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64], &vec![1_u8, 1_u8, 1_u8, 1_u8, 0_u8]),
        "auc with a perfect classification and class1 < class0 should be 0.0");
        assert_eq!(1.0, compute_auc_from_value(&vec![0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64], &vec![0_u8, 0_u8, 0_u8, 0_u8, 1_u8]),
        "auc with a perfect classification and class0 < class1 should be 1.0");
        assert_eq!(1.0, compute_auc_from_value(&vec![0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64], &vec![0_u8, 0_u8, 0_u8, 0_u8, 1_u8]),
        "auc with a perfect classification and class0 < class1 should be 1.0");
        // maybe add a verification inside compute_auc to avoid below cases ?
        assert_eq!(0.5, compute_auc_from_value(&vec![0.1_f64, 0.2_f64, 0.3_f64, 0.4_f64], &vec![0_u8, 0_u8, 0_u8, 0_u8]),
        "auc should be equal to 0 when there is no positive class");
        assert_eq!(0.5, compute_auc_from_value(&vec![0.5_f64, 0.6_f64, 0.7_f64, 0.8_f64], &vec![1_u8, 1_u8, 1_u8, 1_u8]),
        "auc should be equal to 0 when there is no negative class to avoid positive biais in model selection");
        assert_eq!(0.4166666666666667, compute_auc_from_value(&vec![0.5_f64, 0.6_f64, 0.3_f64, 0.1_f64, 0.9_f64, 0.1_f64], &vec![1_u8, 2_u8, 1_u8, 0_u8, 0_u8, 1_u8]),
        "class 2 should be omited in AUC");
    }

    // fn calculate_confusion_matrix
    #[test]
    fn test_calculate_confusion_matrix_basic() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        let data = Data::test2();
        let confusion_matrix = ind.calculate_confusion_matrix(&data);
        assert_eq!(confusion_matrix.0, 2, "incorrect identification of true positives");
        assert_eq!(confusion_matrix.1, 4, "incorrect identification of false positives");
        assert_eq!(confusion_matrix.2, 3, "incorrect identification of true negatives");
        assert_eq!(confusion_matrix.3, 1, "incorrect identification of false negatives");
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
        let data = Data::test2();
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
        assert_eq!(ind_bin.epsilon, DEFAULT_MINIMUM, "input epsilon should be respected"); 
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
        assert_eq!(0.6666666666666666_f64, metrics.1, "bad calculation for sensitivity");
        assert_eq!(0.42857142857142855_f64, metrics.2, "bad calculation for specificity");
    }

    #[test]
    fn test_compute_metrics_class2() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 2];
        assert_eq!((0.5555555555555556_f64, 0.6666666666666666_f64, 0.5_f64), ind.compute_metrics(&data),
        "class 2 should be omitted in calculation")
    }

    #[test]
    fn test_compute_metrics_too_much_y() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1];
        assert_eq!((0.5_f64, 0.6666666666666666_f64, 0.42857142857142855_f64), ind.compute_metrics(&data),
        "when ind.sample_len < data.sample_len (or y.len() if it does not match), only the ind.sample_len values should be used to calculate its metrics");
    }

    #[test]
    fn test_compute_metrics_not_enough_y() {
        let mut ind = Individual::test();
        ind.threshold = 0.75;
        let mut data = Data::test2();
        data.y = vec![1, 0, 1, 1];
        assert_eq!((0.25_f64, 0.3333333333333333_f64, 0.0_f64), ind.compute_metrics(&data),
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
        assert_eq!(0.6666666666666666_f64, results.2, "bad calculation for sensitivity");
        assert_eq!(0.8571428571428571_f64, results.3, "bad calculation for specificity");

        let scores: Vec<_> = ind.evaluate(&data);
        let (_, _, accuracy, sensitivity, specificity, _): (f64, f64, f64, f64, f64, f64)= compute_roc_and_metrics_from_value(&scores, &data.y, None);
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
        assert_eq!((0.79_f64, 0.7777777777777778_f64, 0.6666666666666666_f64, 0.8333333333333334_f64), ind.compute_threshold_and_metrics(&data), "class 2 should be omitted in calculation");
        
        let scores: Vec<_> = ind.evaluate(&data);
        let (_, _, accuracy, sensitivity, specificity, _): (f64, f64, f64, f64, f64, f64)= compute_roc_and_metrics_from_value(&scores, &data.y, None);
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
        
        let (_, _, accuracy, sensitivity, specificity, _): (f64, f64, f64, f64, f64, f64)= ind.compute_roc_and_metrics(&data, None);
        assert_eq!(accuracy, ind.compute_threshold_and_metrics(&data).1, "Accuracy calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
        assert_eq!(sensitivity, ind.compute_threshold_and_metrics(&data).2, "Sensitivity calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
        assert_eq!(specificity, ind.compute_threshold_and_metrics(&data).3, "Specificity calculated with Individual.compute_threshold_and_metrics() and Individual.compute_roc_and_metrics() should be the same" );
    }

    // fn features_index
    #[test]
    fn test_features_index_basic() {
        let mut ind = Individual::test();
        ind.features.insert(10, 0.42 as i8);
        assert_eq!(vec![0_usize, 1_usize, 2_usize, 3_usize, 10_usize], ind.features_index());
    }

    fn generate_feature_seeds(features: &[usize], permutations: usize, seed: u64) -> HashMap<usize, Vec<u64>> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        features.iter()
            .map(|&f| {
                let seeds: Vec<u64> = (0..permutations)
                    .map(|_| rng.next_u64())
                    .collect();
                (f, seeds)
            })
            .collect()
    }

    #[test]
    fn test_compute_oob_feature_importance_zero_permutations_division_by_zero() {
        let individual = Individual::specific_test(&[0]);
        let data = Data::specific_test(10, 2);
        let features_to_process = vec![0];
        let feature_seeds = generate_feature_seeds(&features_to_process, 0, 123);

        let result = std::panic::catch_unwind(|| {
            individual.compute_oob_feature_importance(&data, 0, &features_to_process, &feature_seeds)
        });
        
        assert!(result.is_err(), "Zero permutations should cause division by zero panic");
    }

    #[test]
    fn test_compute_oob_feature_importance_feature_not_in_individual_returns_zero() {
        let individual = Individual::specific_test(&[2, 4]); 
        let data = Data::specific_test(30, 6);
        let features_to_process = vec![0, 1, 2, 3, 4, 5];
        let feature_seeds = generate_feature_seeds(&features_to_process, 10, 789);

        let result = individual.compute_oob_feature_importance(&data, 10, &features_to_process, &feature_seeds);

        for importance in &result.importances {
            if ![2, 4].contains(&importance.feature_idx) {
                assert_eq!(importance.importance, 0.0, 
                    "Features not in individual (feature {}) must return 0.0 importance", 
                    importance.feature_idx);
            }
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_missing_seeds_panics() {
        let individual = Individual::specific_test(&[1]);
        let data = Data::specific_test(30, 3);
        let features_to_process = vec![1, 2];

        // Seeds only provided for feature 1, missing for feature 2
        let mut incomplete_seeds = HashMap::new();
        incomplete_seeds.insert(1, vec![12345u64; 5]);

        let result = std::panic::catch_unwind(|| {
            individual.compute_oob_feature_importance(&data, 5, &features_to_process, &incomplete_seeds)
        });
        
        assert!(result.is_err(), "Missing seeds should cause .expect() to panic");
    }

    #[test]
    fn test_compute_oob_feature_importance_reproducibility_same_seeds() {
        let individual = Individual::specific_test(&[1, 2]);
        let data = Data::specific_test(50, 4);
        let features_to_process = vec![1, 2];
        let feature_seeds = generate_feature_seeds(&features_to_process, 50, 999);

        let result1 = individual.compute_oob_feature_importance(&data, 2, &features_to_process, &feature_seeds);
        let result2 = individual.compute_oob_feature_importance(&data, 2, &features_to_process, &feature_seeds);
        
        assert_eq!(result1.importances.len(), result2.importances.len());

        for (imp1, imp2) in result1.importances.iter().zip(result2.importances.iter()) {
            assert_eq!(imp1.feature_idx, imp2.feature_idx);
            assert!((imp1.importance - imp2.importance).abs() < 1e-12, 
                "Same seeds must yield identical results for feature {}", imp1.feature_idx);
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_output_structure_fields() {
        let individual = Individual::specific_test(&[1]);
        let data = Data::specific_test(20, 3);
        let features_to_process = vec![0, 1];
        let feature_seeds = generate_feature_seeds(&features_to_process, 5, 456);

        let result = individual.compute_oob_feature_importance(&data, 5, &features_to_process, &feature_seeds);

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
                },
                _ => panic!("Expected Individual scope for feature {}", importance.feature_idx),
            }
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_all_requested_features_returned() {
        let individual = Individual::specific_test(&[2, 5, 7]);
        let data = Data::specific_test(40, 10);
        let features_to_process = vec![2, 5, 7, 9];
        let feature_seeds = generate_feature_seeds(&features_to_process, 5, 789);

        let result = individual.compute_oob_feature_importance(&data, 5, &features_to_process, &feature_seeds);

        let mut result_features: Vec<usize> = result.importances.iter()
            .map(|imp| imp.feature_idx)
            .collect();
        result_features.sort();

        let mut expected_features = features_to_process.clone();
        expected_features.sort();

        assert_eq!(result_features, expected_features, "All requested features must be present in results")}

    #[test]
    fn test_compute_oob_feature_importance_single_permutation_finite_values() {
        let individual = Individual::specific_test(&[0]);
        let data = Data::specific_test(30, 2);
        let features_to_process = vec![0, 1];
        let feature_seeds = generate_feature_seeds(&features_to_process, 1, 456);

        let result = individual.compute_oob_feature_importance(&data, 1, &features_to_process, &feature_seeds);

        assert_eq!(result.importances.len(), 2);
        
        for importance in &result.importances {
            assert!(importance.importance.is_finite(), 
                "Single permutation should yield finite importance for feature {}", 
                importance.feature_idx);
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_bounds_realistic() {
        let individual = Individual::specific_test(&[0, 1]);
        let data = Data::specific_test(50, 3);
        let features_to_process = vec![0, 1, 2];
        let feature_seeds = generate_feature_seeds(&features_to_process, 20, 654);

        let result = individual.compute_oob_feature_importance(&data, 20, &features_to_process, &feature_seeds);

        for importance in &result.importances {
            // Importance = baseline_auc - mean_permuted_auc
            // Can be negative if permutation improves performance
            assert!(importance.importance.is_finite(), 
                "Importance must be finite for feature {}", importance.feature_idx);
            assert!(importance.importance >= -1.0 && importance.importance <= 1.0, 
                "Importance should be bounded by AUC difference range for feature {}: got {}", 
                importance.feature_idx, importance.importance);
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_empty_individual_features() {
        let individual = Individual::specific_test(&[]); // No features
        let data = Data::specific_test(20, 4);
        let features_to_process = vec![0, 1, 2, 3];
        let feature_seeds = generate_feature_seeds(&features_to_process, 5, 321);

        let result = individual.compute_oob_feature_importance(&data, 5, &features_to_process, &feature_seeds);

        for importance in &result.importances {
            assert_eq!(importance.importance, 0.0, 
                "Individual with no features should return 0 importance for feature {}", 
                importance.feature_idx);
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_different_seeds_may_differ() {
        let individual = Individual::specific_test(&[0, 2]);
        let data = Data::specific_test(60, 3);
        let features_to_process = vec![0, 2];

        let seeds1 = generate_feature_seeds(&features_to_process, 10, 111);
        let seeds2 = generate_feature_seeds(&features_to_process, 10, 222);

        let result1 = individual.compute_oob_feature_importance(&data, 10, &features_to_process, &seeds1);
        let result2 = individual.compute_oob_feature_importance(&data, 10, &features_to_process, &seeds2);

        // Note: Results may be the same in some cases (deterministic data), so we only check structure
        assert_eq!(result1.importances.len(), result2.importances.len());
        
        for (imp1, imp2) in result1.importances.iter().zip(result2.importances.iter()) {
            assert_eq!(imp1.feature_idx, imp2.feature_idx);
            assert!(imp1.importance.is_finite() && imp2.importance.is_finite());
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_large_permutation_count() {
        let individual = Individual::specific_test(&[0]);
        let data = Data::specific_test(50, 2);
        let features_to_process = vec![0];
        let feature_seeds1 = generate_feature_seeds(&features_to_process, 10000, 123);
        let feature_seeds2 = generate_feature_seeds(&features_to_process, 10000, 456);

        let result = individual.compute_oob_feature_importance(&data, 10000, &features_to_process, &feature_seeds1);

        assert_eq!(result.importances.len(), 1);
        
        let importance = &result.importances[0];
        assert!(importance.importance.is_finite(), 
            "Large permutation count should yield finite importance");
        
        // With many permutations, results should be stable 
        let result2 = individual.compute_oob_feature_importance(&data, 10000, &features_to_process, &feature_seeds2);
        
        assert!((importance.importance - result2.importances[0].importance).abs() < 1e-2,
            "Large permutation count with different seeds should be highly reproducible: {} vs {}", importance.importance, result2.importances[0].importance);
    }

    #[test]
    fn test_compute_oob_feature_importance_individual_with_mixed_coefficients() {
        // Test individual with mixed positive/negative coefficients
        let mut features_map = HashMap::new();
        features_map.insert(0, 1i8);   // Positive coefficient
        features_map.insert(1, -1i8);  // Negative coefficient
        features_map.insert(2, 1i8);   // Positive coefficient
        
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
        };
        
        let data = Data::specific_test(40, 5);
        let features_to_process = vec![0, 1, 2, 3, 4];
        let feature_seeds = generate_feature_seeds(&features_to_process, 10, 555);

        let result = individual.compute_oob_feature_importance(&data, 10, &features_to_process, &feature_seeds);

        for importance in &result.importances {
            if [0, 1, 2].contains(&importance.feature_idx) {
                // Features present in individual should have computed importance
                assert!(importance.importance.is_finite(), 
                    "Present features should have finite importance");
            } else {
                // Features not in individual should have zero importance
                assert_eq!(importance.importance, 0.0, 
                    "Absent features should have zero importance");
            }
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_different_datatypes() {
        let mut individual = Individual::specific_test(&[0, 1]);
        let data = Data::specific_test(50, 3);
        let features_to_process = vec![0, 1, 2];
        let feature_seeds = generate_feature_seeds(&features_to_process, 10, 123);

        // Test RAW_TYPE (default)
        individual.data_type = RAW_TYPE;
        let result_raw = individual.compute_oob_feature_importance(&data, 10, &features_to_process, &feature_seeds);
        assert_eq!(result_raw.importances.len(), 3);
        
        // Test PREVALENCE_TYPE
        individual.data_type = PREVALENCE_TYPE;
        let result_prev = individual.compute_oob_feature_importance(&data, 10, &features_to_process, &feature_seeds);
        assert_eq!(result_prev.importances.len(), 3);

        // Test LOG_TYPE
        individual.data_type = LOG_TYPE;
        let result_log = individual.compute_oob_feature_importance(&data, 10, &features_to_process, &feature_seeds);
        assert_eq!(result_log.importances.len(), 3);

        // Verify all results contain finite importance values
        for result in [&result_raw, &result_prev, &result_log] {
            for imp in &result.importances {
                assert!(imp.importance.is_finite(), 
                    "All datatypes should produce finite importance values for feature {}", 
                    imp.feature_idx);
            }
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_duplicate_features_in_process() {
        let individual = Individual::specific_test(&[0, 1, 2]);
        let data = Data::specific_test(30, 4);
        let features_to_process = vec![0, 1, 1, 2, 3]; // feature 1 dupliquée
        let feature_seeds = generate_feature_seeds(&features_to_process, 5, 789);

        let result = individual.compute_oob_feature_importance(&data, 5, &features_to_process, &feature_seeds);

        // Should process each unique feature only once
        let unique_features: std::collections::HashSet<usize> = features_to_process.iter().cloned().collect();
        assert_eq!(result.importances.len(), unique_features.len(), 
            "Should return importance for each unique feature only");

        // Verify no duplicate feature indices in results
        let mut seen_features = std::collections::HashSet::new();
        for imp in &result.importances {
            assert!(!seen_features.contains(&imp.feature_idx), 
                "Duplicate importance found for feature {}", imp.feature_idx);
            seen_features.insert(imp.feature_idx);
            
            assert!(unique_features.contains(&imp.feature_idx), 
                "Result contains unexpected feature {}", imp.feature_idx);
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_extreme_unbalanced_data() {
        let individual = Individual::specific_test(&[0, 1]);
        let mut data = Data::specific_test(50, 3);
        let features_to_process = vec![0, 1, 2];
        let feature_seeds = generate_feature_seeds(&features_to_process, 10, 456);

        // Test: All positive labels (y = 1)
        data.y = vec![1u8; 50];
        let result_all_pos = individual.compute_oob_feature_importance(&data, 10, &features_to_process, &feature_seeds);

        for imp in &result_all_pos.importances {
            assert!(imp.importance.is_finite(), 
                "Importance should be finite with all positive labels for feature {}", 
                imp.feature_idx);
            // AUC calculation should handle single-class case gracefully
        }

        // Test: All negative labels (y = 0)  
        data.y = vec![0u8; 50];
        let result_all_neg = individual.compute_oob_feature_importance(&data, 10, &features_to_process, &feature_seeds);

        for imp in &result_all_neg.importances {
            assert!(imp.importance.is_finite(), 
                "Importance should be finite with all negative labels for feature {}", 
                imp.feature_idx);
        }

        // Test: Highly unbalanced (49:1)
        data.y = vec![0u8; 49];
        data.y.push(1u8);
        let result_unbalanced = individual.compute_oob_feature_importance(&data, 5, &features_to_process, &feature_seeds);

        for imp in &result_unbalanced.importances {
            assert!(imp.importance.is_finite(), 
                "Importance should be finite with unbalanced labels for feature {}", 
                imp.feature_idx);
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_constant_feature_values() {
        let individual = Individual::specific_test(&[0, 1]);
        let mut data = Data::specific_test(40, 3);
        let features_to_process = vec![0, 1, 2];
        let feature_seeds = generate_feature_seeds(&features_to_process, 8, 999);

        // Make feature 0 constant (all same value)
        for sample in 0..data.sample_len {
            data.X.insert((sample, 0), 0.5); // Constant value
        }

        let result = individual.compute_oob_feature_importance(&data, 8, &features_to_process, &feature_seeds);

        for imp in &result.importances {
            assert!(imp.importance.is_finite(), 
                "Constant features should still produce finite importance for feature {}", 
                imp.feature_idx);
            
            if imp.feature_idx == 0 && individual.features.contains_key(&0) {
                // Constant features typically have low/zero importance
                assert!(imp.importance >= -1.0 && imp.importance <= 1.0,
                    "Constant feature importance should be bounded");
            }
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_very_small_dataset() {
        let individual = Individual::specific_test(&[0]);
        let data = Data::specific_test(3, 2); // Very small: 3 samples, 2 features
        let features_to_process = vec![0, 1];
        let feature_seeds = generate_feature_seeds(&features_to_process, 2, 777);

        let result = individual.compute_oob_feature_importance(&data, 2, &features_to_process, &feature_seeds);

        assert_eq!(result.importances.len(), 2);
        
        for imp in &result.importances {
            assert!(imp.importance.is_finite(), 
                "Small dataset should still produce finite importance for feature {}", 
                imp.feature_idx);
            // With very few samples, AUC calculation might be less stable but should not crash
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_out_of_bounds_features() {
        let individual = Individual::specific_test(&[0, 1]);
        let data = Data::specific_test(30, 3); // Features 0, 1, 2 exist
        let features_to_process = vec![0, 1, 5, 10]; // Features 5, 10 don't exist in data
        let feature_seeds = generate_feature_seeds(&features_to_process, 5, 555);

        let result = individual.compute_oob_feature_importance(&data, 5, &features_to_process, &feature_seeds);

        for imp in &result.importances {
            assert!(imp.importance.is_finite(), 
                "Out-of-bounds features should be handled gracefully for feature {}", 
                imp.feature_idx);
                
            if ![0, 1, 2].contains(&imp.feature_idx) {
                // Out-of-bounds features should have zero importance
                assert_eq!(imp.importance, 0.0,
                    "Out-of-bounds feature {} should have zero importance", 
                    imp.feature_idx);
            }
        }
    }

    #[test]
    fn test_compute_oob_feature_importance_mixed_languages() {
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

            let result = individual.compute_oob_feature_importance(&data, 8, &features_to_process, &feature_seeds);

            assert_eq!(result.importances.len(), features_to_process.len());
            
            for imp in &result.importances {
                assert!(imp.importance.is_finite(), 
                    "Language {:?} should produce finite importance for feature {}", 
                    language, imp.feature_idx);
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
        let _ = data.load_data("samples/Qin2014/Xtrain.tsv", "samples/Qin2014/Ytrain.tsv");
        let _ = data_test.load_data("samples/Qin2014/Xtest.tsv", "samples/Qin2014/Ytest.tsv");

        // Set the language and data type 
        individual.language = TERNARY_LANG;
        individual.data_type = LOG_TYPE;
        individual.epsilon = 1e-5;

        // Set the feature indices and their signs
        let feature_indices = vec![
            (9, 1), (22, 1), (23, 1), (24, -1), (42, -1), (47, 1), (57, -1), (66, -1), (72, -1),
            (82, -1), (87, -1), (92, -1), (105, 1), (124, -1), (130, -1), (174, 1), (194, 1),
            (221, 1), (222, 1), (262, 1), (272, 1), (301, -1), (319, 1), (320, -1), (324, 1),
            (334, 1), (359, 1), (378, 1), (436, -1), (466, -1), (468, -1), (476, -1), (488, 1),
            (497, -1), (512, 1), (522, -1), (546, -1), (565, 1), (591, -1), (614, -1), (649, -1),
            (658, 1), (670, -1), (686, 1), (716, 1), (825, 1), (834, -1), (865, -1), (867, 1),
            (874, 1), (877, 1), (1117, 1), (1273, 1), (1313, 1), (1317, 1), (1464, 1), (1525, 1),
            (1629, 1), (1666, 1), (1710, 1), (1735, 1), (1738, 1), (1740, 1), (1741, 1), (1794, 1),
            (1870, 1)
        ];

        for (index, sign) in feature_indices {
            individual.features.insert(index, sign);
        }

        data.classes = vec!["healthy".to_string(), "cirrhosis".to_string()];
        data.y[3] = 2 as u8;
        data.y[4] = 2 as u8;
        data_test.y[7] = 2 as u8;

        // control both metrics and display
        let right_string = concat!("Ternary:Log [k=66] [gen:0] [fit:0.000] AUC 0.962/0.895 | accuracy 0.921/0.828 | sensitivity 0.937/0.867 | specificity 0.904/0.786\n",
                            "Class cirrhosis: ln(msp_0010 * msp_0023 * msp_0024 * msp_0048 * msp_0106 * msp_0176 * msp_0196 * msp_0223 * msp_0224 * msp_0265",
                            " * msp_0275 * msp_0324 * msp_0329 * msp_0339 * msp_0364 * msp_0383 * msp_0493 * msp_0517 * msp_0570 * msp_0664 * msp_0692 * msp_0722",
                            " * msp_0832 * msp_0874 * msp_0881 * msp_0884 * msp_1127 * msp_1284 * msp_1325 * msp_1329 * msp_1479 * msp_1543 * msp_1660 * msp_1700",
                            " * msp_1748 * msp_1782 * msp_1785 * msp_1787 * msp_1788 * msp_1862 * msp_1942) - ln(msp_0025 * msp_0043 * msp_0058 * msp_0067 * msp_0073",
                            " * msp_0083 * msp_0088 * msp_0093 * msp_0125 * msp_0131 * msp_0306 * msp_0325 * msp_0441 * msp_0471 * msp_0473c * msp_0481 * msp_0502", 
                            " * msp_0527 * msp_0551 * msp_0596 * msp_0619 * msp_0654 * msp_0676 * msp_0841 * msp_0872) ≥ 19.398124045367666 (+ -184.20680743952366)");

        (individual.auc, individual.threshold, individual.accuracy, individual.sensitivity, individual.specificity, _) = individual.compute_roc_and_metrics(&data, None);

        // except the threshold (small variation between launch ~0.000000000001)
        assert_eq!(right_string.split("≥ 19").collect::<Vec<_>>()[0], individual.display(&data, Some(&data_test), &"ga".to_string(), 2, false).split("≥ 19").collect::<Vec<_>>()[0]);

        assert_eq!(individual.compute_auc(&data), 0.961572606214331, "Wrong auc calculated");
        assert_eq!(individual.compute_new_auc(&data_test), 0.8952380952380953, "Wrong test auc calculated");
        // Compute ROC and metrics should return the same AUC as .compute_auc and the same metrics as .compute_threshold_and_metrics
        let (threshold, accuracy, sensitivity, specificity): (f64, f64, f64, f64)= individual.compute_threshold_and_metrics(&data);
       
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
        assert_eq!(scores, vec![0.89, 0.79, 0.74, -0.73, 0.89, 0.79, 0.74, -0.73, 0.89, 0.79], "bad calculation for score");
        let class_and_score = ind.evaluate_class_and_score(&data);
        assert_eq!(class_and_score, (vec![1, 1, 1, 0, 1, 1, 1, 0, 1, 1],vec![0.89, 0.79, 0.74, -0.73, 0.89, 0.79, 0.74, -0.73, 0.89, 0.79]), "bad calculation for class_and_score ");
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
        real_tree.insert((1, Some(vec![3, 4, 7, 8, 9, 11])), [1].iter().cloned().collect());
        real_tree.insert((2, Some(vec![4, 5])), [1].iter().cloned().collect());
        real_tree.insert((3, Some(vec![6, 7])), [2].iter().cloned().collect());
        real_tree.insert((4, Some(vec![7, 8])), [2].iter().cloned().collect());
        real_tree.insert((5, Some(vec![8, 9])), [2].iter().cloned().collect());
        real_tree.insert((6, Some(vec![9])), [3].iter().cloned().collect());
        real_tree.insert((7, None), [2, 3].iter().cloned().collect()); 
        real_tree.insert((8, None), [2, 3].iter().cloned().collect()); 
        real_tree.insert((9, None), [2, 3, 4].iter().cloned().collect());
        real_tree.insert((11, None), [2].iter().cloned().collect());
        
        let mut pop= Population::test();
        let mut gen_0 = Population::new();
        let mut gen_1  = Population::new();
        let mut gen_2  = Population::new();
        let mut gen_3  = Population::new();
        let mut gen_4  = Population::new();
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
        gen_1.individuals = vec![pop.individuals[9].clone(), pop.individuals[6].clone(), pop.individuals[7].clone(), pop.individuals[8].clone()];
        gen_2.individuals = vec![pop.individuals[9].clone(), pop.individuals[5].clone(), pop.individuals[4].clone(), pop.individuals[3].clone()];
        gen_3.individuals = vec![pop.individuals[2].clone(), pop.individuals[1].clone()]; 
        gen_4.individuals = vec![pop.individuals[0].clone()]; 
    
        let collection = vec![gen_0, gen_1, gen_2, gen_3, gen_4];
        assert_eq!(collection[4].individuals[0].get_genealogy(&collection, 15), real_tree, "Generated tree is broken for complex tree");
    
        // Max depth test
        let mut real_tree: HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> = HashMap::new();
        real_tree.insert((10, Some(vec![1, 2])), [0].iter().cloned().collect());
        real_tree.insert((1, Some(vec![3, 4, 7, 8, 9, 11])), [1].iter().cloned().collect());
        real_tree.insert((2, Some(vec![4, 5])), [1].iter().cloned().collect());
        assert_eq!(collection[4].individuals[0].get_genealogy(&collection, 1), real_tree, "Max_depth should limit ancestors");
    }

    #[test]
    fn test_get_genealogy_edge_cases() {
        // Empty population
        let empty_population: Vec<Population> = Vec::new();
        let individual = Individual::test();
        let genealogy = individual.get_genealogy(&empty_population, 10);
        assert!(genealogy.is_empty(), "Genealogy should be empty for an empty population");
    
        // Single individual
        let mut single_individual_population = vec![Population::new()];
        single_individual_population[0].individuals.push(individual.clone());
        let genealogy = individual.get_genealogy(&single_individual_population, 10);
        assert_eq!(genealogy.len(), 1, "Genealogy should contain only the individual itself");
    
        // Individual having no parents
        let mut no_parents_population = vec![Population::new()];
        let mut no_parents_individual = Individual::test();
        no_parents_individual.parents = None;
        no_parents_population[0].individuals.push(no_parents_individual.clone());
        let genealogy = no_parents_individual.get_genealogy(&no_parents_population, 10);
        assert_eq!(genealogy.len(), 1, "Genealogy should contain only the individual itself");
    
        // max_depth set to 0
        let genealogy = individual.get_genealogy(&empty_population, 0);
        assert!(genealogy.is_empty(), "Genealogy should be empty for max_depth set to 0");
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
    
        let collection_deep = vec![gen_0_deep, gen_1_deep, gen_2_deep, gen_3_deep, gen_4_deep, gen_5_deep, gen_6_deep, gen_7_deep, gen_8_deep, gen_9_deep];
        assert_eq!(collection_deep[9].individuals[0].get_genealogy(&collection_deep, 10), deep_tree, "Generated tree is broken for deep tree");
    
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
    
        gen_0_wide.individuals = vec![pop_wide.individuals[1].clone(), pop_wide.individuals[2].clone(), pop_wide.individuals[3].clone(), pop_wide.individuals[4].clone()];
        gen_1_wide.individuals = vec![pop_wide.individuals[0].clone()];
    
        let collection_wide = vec![gen_0_wide, gen_1_wide];
        assert_eq!(collection_wide[1].individuals[0].get_genealogy(&collection_wide, 10), wide_tree, "Generated tree is broken for wide tree");
    
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
        assert_eq!(collection_cycle[2].individuals[0].get_genealogy(&collection_cycle, 10), cycle_tree, "Generated tree is broken for tree with cycles");
    
        // Tree with Missing Parents
        let mut missing_parents_tree: HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> = HashMap::new();
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
        assert_eq!(collection_missing[1].individuals[0].get_genealogy(&collection_missing, 10), missing_parents_tree, "Generated tree is broken for tree with missing parents");
    
        // Tree with Multiple Paths
        let mut multiple_paths_tree: HashMap<(u64, Option<Vec<u64>>), HashSet<usize>> = HashMap::new();
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
        gen_1_multiple.individuals = vec![pop_multiple.individuals[1].clone(), pop_multiple.individuals[2].clone()];
        gen_2_multiple.individuals = vec![pop_multiple.individuals[0].clone()];
    
        let collection_multiple = vec![gen_0_multiple, gen_1_multiple, gen_2_multiple];
        assert_eq!(collection_multiple[2].individuals[0].get_genealogy(&collection_multiple, 10), multiple_paths_tree, "Generated tree is broken for tree with multiple paths");
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
    
}