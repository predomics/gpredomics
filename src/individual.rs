use crate::utils::generate_random_vector;
use crate::data::Data;
use rand::{rngs::ThreadRng, seq::SliceRandom}; // Provides the `choose_multiple` method
use std::collections::{HashMap, HashSet};
use rand::Rng;
use rand_chacha::ChaCha8Rng;

#[derive(Debug)]
pub struct Individual {
    pub features: Vec<i8>, /// a vector of feature indices with their corresponding signs
    //pub feature_names: Vec<string>, /// a vector of feature indices
    pub fit_method: String, // AUC, accuracy, etc.
    pub auc: f64, // accuracy of the model
    pub k: u32 // nb of variables used
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

    pub fn new() -> Individual {
        Individual {
            features: Vec::new(),
            fit_method: String::from("AUC"),
            auc: 0.0,
            k: 0,
        }
    }

    pub fn clone(&self) -> Individual {
        Individual {
            features: self.features.clone(),
            fit_method: self.fit_method.clone(),
            auc: self.auc,
            k: self.k
        }
    }

    pub fn evaluate(&self, d: &Data) -> Vec<f64> {
        let mut value=vec![0.0; d.samples.len()];
        for (i,row) in d.X.iter().enumerate() {
            for (j,x) in row.iter().enumerate() {
                value[j]+=self.features[i] as f64*x;
            }
        }
        value
    }

    /// Compute AUC based on the target vector y
    pub fn compute_auc(&mut self, d: &Data) -> f64 {
        let value = self.evaluate(d);
        let mut thresholds: Vec<f64> = (&value).clone();
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        thresholds.insert(0, thresholds[0] - 1.0); // Add a threshold below the minimum
        thresholds.push(thresholds.last().unwrap() + 1.0); // Add a threshold above the maximum


        let mut auc = 0.0;
        let mut roc_points = Vec::new();

        for &threshold in &thresholds {
            let (tp, fp, tn, fn_count) = self.calculate_confusion_matrix(&value, &d.y, threshold);

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
        }

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
    fn calculate_confusion_matrix(&self, value: &Vec<f64>, y: &[u8], threshold: f64) -> (usize, usize, usize, usize) {
        let mut tp = 0; // True Positives
        let mut fp = 0; // False Positives
        let mut tn = 0; // True Negatives
        let mut fn_count = 0; // False Negatives

        for (i, &pred) in value.iter().enumerate() {
            match y[i] {
                1 => {
                    // Positive class
                    if pred > threshold {
                        tp += 1;
                    } else {
                        fn_count += 1;
                    }
                }
                0 => {
                    // Negative class
                    if pred > threshold {
                        fp += 1;
                    } else {
                        tn += 1;
                    }
                }
                2 => {
                    // Unknown class, ignore
                }
                _ => panic!("Invalid class label in y: {}", y[i]),
            }
        }


        (tp, fp, tn, fn_count)
    }   

    pub fn count_k(&mut self) {
        self.k = (self.features).iter().map(|x| {if *x==0 {0} else {1}}).sum();
    }

    /// completely random individual, not very usefull
    pub fn random(d: &Data, rng: &mut ChaCha8Rng) -> Individual {

        let features = generate_random_vector(d.feature_len, rng);
        let k = (&features).iter().map(|x| {if *x==0 {0} else {1}}).sum();

        Individual {
            features: features,
            fit_method: String::from("AUC"),
            auc: 0.0,
            k: k
        }
    }

    /// 
    pub fn random_select_k(reference_size: usize, feature_selection: &Vec<u32>, kmin: u32, kmax: u32, feature_sign: &HashMap<u32,u8>, rng: &mut ChaCha8Rng) -> Individual {
        // chose k variables amount feature_selection
        // set a random coeficient for these k variables
    
    
        let k: u32=rng.gen_range(kmin..(kmax+1));

        // Randomly pick k values
        let random_values = feature_selection.choose_multiple(rng, k as usize);
        let mut chosen_feature_sign: HashMap<u32,u8>=HashMap::new();
        for i in random_values {
            chosen_feature_sign.insert(*i, feature_sign[i]);
        }
        
        let mut features:Vec<i8> = Vec::new();
        // Generate a vector of random values: 1, 0, or -1
        for i in 0..(reference_size as u32) {
            if chosen_feature_sign.contains_key(&i) {
                if chosen_feature_sign[&i]==0 {
                    features.push(-1);
                }
                else {
                    features.push(1);
                }
            }
            else {
                features.push(0);
            }
        }
        //println!("Model {:?}",features);
        Individual {
            features: features,
            fit_method: String::from("AUC"),
            auc: 0.0,
            k: k
        }

    }
    

    // write a function fit_model that takes in the data and computes all the following fields

    // write a function evaluate_contingency_table that takes in the data and evaluates the contingency table of the model

    // write a function evaluate_accuracy that takes in the data and evaluates the accuracy of the model

    // write a function evaluate_auc that takes in the data and evaluates the AUC of the model


}
