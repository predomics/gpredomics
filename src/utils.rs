use std::collections::HashMap;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::seq::SliceRandom;
use statrs::distribution::{Normal, ContinuousCDF};
use crate::data::Data;
use crate::experiment::ImportanceAggregation;

/// a macro to declare simple Vec<String>
#[macro_export]
macro_rules! string_vec {
    ($($x:expr),*) => {
        vec![$($x.into()),*]
    };
}

pub fn generate_random_vector(reference_size: usize, rng: &mut ChaCha8Rng) -> Vec<i8> {
    // chose k variables amount feature_selection
    // set a random coeficient for these k variables
    // Generate a vector of random values: 1, 0, or -1
    (0..reference_size).map(|_| { rng.gen_range(-1..2) }).collect()
}

/// a function used essentially in CV that split randomly a Vec<T> into p Vec<T> of approximatively the same size
pub fn split_into_balanced_random_chunks<T: std::clone::Clone>(vec: Vec<T>, p: usize, rng: &mut ChaCha8Rng) -> Vec<Vec<T>> {
    // Step 1: Shuffle the original vector
    let mut shuffled = vec;
    shuffled.shuffle(rng);

    // Step 2: Determine sizes for balanced chunks
    let n = shuffled.len();
    let base_size = n / p; // Minimum size for each chunk
    let extra_elements = n % p; // Remaining elements to distribute

    // Step 3: Create chunks with balanced sizes
    let mut chunks = Vec::new();
    let mut start = 0;

    for i in 0..p {
        let chunk_size = base_size + if i < extra_elements { 1 } else { 0 }; // Add one extra element to the first `extra_elements` chunks
        let end = start + chunk_size;
        chunks.push(shuffled[start..end].to_vec());
        start = end;
    }

    chunks
}

/// shuffle a feature
pub fn shuffle_row(X: &mut HashMap<(usize, usize), f64>, sample_len: usize, feature: usize, rng: &mut ChaCha8Rng) {
    // Extract all the column indices and values for the given row
    let feature_values: Vec<f64> = (0..sample_len)
        .filter_map(|i| X.remove(&(i, feature)))
        .collect();

    // Shuffle the column indices
    //feature_values.shuffle(rng);
    let new_samples: Vec<usize> = (0..sample_len).collect::<Vec<usize>>()
                        .choose_multiple(rng, feature_values.len()).copied().collect();


    // Update the matrix with shuffled values
    for (value,new_sample) in feature_values.iter().zip(new_samples.iter()) {
        X.insert((*new_sample, feature), *value);
    }
}

// Statistical functions

pub fn conf_inter_binomial(accuracy: f64, n: usize, alpha: f64) -> (f64, f64, f64) {
    assert!(n > 0, "confInterBinomial: Sample size (n) must be greater than zero.");
    assert!(accuracy >= 0.0 && accuracy <= 1.0, "confInterBinomial: accuracy should not be lower than 0 or greater than 1");
    assert!(alpha >= 0.0 && alpha <= 1.0, "confInterBinomial: alpha should not be lower than 0 or greater than 1");

    let normal = Normal::new(0.0, 1.0).unwrap_or_else(|e| panic!("confInterBinomial : normal distribution creation failed: {}", e));
    let z_value = -normal.inverse_cdf(alpha / 2.0);
    let std_error = ((accuracy * (1.0 - accuracy)) / n as f64).sqrt();
    
    let ci_range = z_value * std_error;
    let lower_bound = 0.0f64.max(accuracy - ci_range); 
    let upper_bound = 1.0f64.min(accuracy + ci_range);  

    (lower_bound, accuracy, upper_bound)
}

/// Compute AUC for binary class using Mann-Whitney U algorithm O(n log n)
pub fn compute_auc_from_value(value: &[f64], y: &Vec<u8>) -> f64 {
    let mut data: Vec<(f64, u8)> = value.iter()
        .zip(y.iter())
        .filter(|(_, &label)| label == 0 || label == 1)
        .map(|(&v, &y)| (v, y))
        .collect();

    let n = data.len();
    let n1 = data.iter().filter(|(_, label)| *label == 1).count();
    let n0 = n - n1;

    if n1 == 0 || n0 == 0 {
        return 0.5;
    }

    data.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

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

        if neg_equal > 0 {
            u += neg_equal as f64 * pos_so_far as f64;
            u += 0.5 * neg_equal as f64 * pos_equal as f64;
        }

        pos_so_far += pos_equal;
    }

    u / (n1 as f64 * n0 as f64)
}

pub fn compute_metrics_from_classes(predicted: &Vec<u8>, y: &Vec<u8>, ) -> (f64, f64, f64) {
        let mut tp = 0;
        let mut fn_count = 0;
        let mut tn = 0;
        let mut fp = 0;

        for (&pred, &real) in predicted.iter().zip(y.iter()) {
            if real == 2 {
                continue;
            }
            match (pred, real) {
                (1, 1) => tp += 1,
                (1, 0) => fp += 1,
                (0, 0) => tn += 1,
                (0, 1) => fn_count += 1,
                _ => {} //warn!("A predicted vs real class of {:?} should not exist", other),
            }
        }

        let sensitivity = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };
        let specificity = if fp + tn > 0 {
            tn as f64 / (fp + tn) as f64
        } else {
            0.0
        };
        let accuracy = if tp + tn + fp + fn_count > 0 {
            (tp + tn) as f64 / (tp + tn + fp + fn_count) as f64
        } else {
            0.0
        };

        (accuracy, sensitivity, specificity)
    }

pub fn compute_roc_and_metrics_from_value(value: &[f64], y: &Vec<u8>, penalties: Option<&[f64]>) -> (f64, f64, f64, f64, f64, f64) {
    let mut best_objective: f64 = f64::MIN;

    let mut data: Vec<_> = value.iter()
        .zip(y.iter())
        .filter(|(_, &y)| y == 0 || y == 1)
        .map(|(&v, &y)| (v, y))
        .collect();

    data.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos = data.iter().filter(|(_, y)| *y == 1).count();
    let total_neg = data.len() - total_pos;

    if total_pos == 0 || total_neg == 0 {
        return (0.5, f64::NAN, 0.0, 0.0, 0.0, f64::MIN);
    }

    let (mut auc, mut tn, mut fn_count) = (0.0, 0, 0);
    let mut best_threshold = f64::NEG_INFINITY;
    let (mut best_acc, mut best_sens, mut best_spec) = (0.0, 0.0, 0.0);

    let mut i = 0;
    while i < data.len() {
        let current_score = data[i].0;
        let (mut current_tn, mut current_fn) = (0, 0);

        while i < data.len() && (data[i].0 - current_score).abs() < f64::EPSILON {
            match data[i].1 {
                0 => current_tn += 1,
                1 => current_fn += 1,
                _ => unreachable!()
            }
            i += 1;
        }

        let remaining_pos_before = total_pos - fn_count;
        auc += current_tn as f64 * (remaining_pos_before - current_fn) as f64;
        auc += 0.5 * (current_tn * current_fn) as f64;

        tn += current_tn;
        fn_count += current_fn;

        let tp = total_pos - fn_count;

        // Include scores equal to the threshold as positive
        let sensitivity = (tp + current_fn) as f64 / total_pos as f64;
        let specificity = (tn - current_tn) as f64 / total_neg as f64;
        let accuracy = (tp + current_fn + tn - current_tn) as f64 / (total_pos + total_neg) as f64;

        if let Some(p) = penalties {
            if p.len() >= 2 {
                let objective = (p[0] * specificity + p[1] * sensitivity)/(p[0] + p[1]);
                if objective > best_objective || (objective == best_objective && current_score < best_threshold) {
                    best_objective = objective;
                    best_threshold = current_score;
                    best_acc = accuracy;
                    best_sens = sensitivity;
                    best_spec = specificity;
                }
            }
        } else {
            // Objective is Youden Maxima
            let objective = sensitivity + specificity - 1.0;
            if objective > best_objective || (objective == best_objective && current_score < best_threshold) {
                best_objective = objective;
                best_threshold = current_score;
                best_acc = accuracy;
                best_sens = sensitivity;
                best_spec = specificity;
            }
        }
    }

    let auc = if total_pos * total_neg > 0 {
        auc / (total_pos * total_neg) as f64
    } else {
        0.5
    };

    (auc, best_threshold, best_acc, best_sens, best_spec, best_objective)
}

pub fn compute_mcc_and_metrics_from_value(value: &[f64], y: &Vec<u8>) -> (f64, f64, f64, f64, f64) {
    let mut best_mcc: f64 = f64::MIN;

    let mut data: Vec<_> = value.iter()
        .zip(y.iter())
        .filter(|(_, &y)| y == 0 || y == 1)
        .map(|(&v, &y)| (v, y))
        .collect();

    data.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos = data.iter().filter(|(_, y)| *y == 1).count();
    let total_neg = data.len() - total_pos;

    if total_pos == 0 || total_neg == 0 {
        return (0.0, f64::NAN, 0.0, 0.0, 0.0);
    }

    let (mut tn, mut fn_count) = (0, 0);
    let mut best_threshold = f64::NEG_INFINITY;
    let (mut best_acc, mut best_sens, mut best_spec) = (0.0, 0.0, 0.0);

    let mut i = 0;
    while i < data.len() {
        let current_score = data[i].0;
        let (mut current_tn, mut current_fn) = (0, 0);

        while i < data.len() && (data[i].0 - current_score).abs() < f64::EPSILON {
            match data[i].1 {
                0 => current_tn += 1,
                1 => current_fn += 1,
                _ => unreachable!()
            }
            i += 1;
        }

        tn += current_tn;
        fn_count += current_fn;

        let tp = total_pos - fn_count;
        let fp = total_neg - tn;
        
        let numerator = (tp as f64 * tn as f64) - (fp as f64 * fn_count as f64);
        let denominator = ((tp + fp) as f64 * (tp + fn_count) as f64 * 
                          (tn + fp) as f64 * (tn + fn_count) as f64).sqrt();
        
        let mcc = if denominator == 0.0 { 0.0 } else { numerator / denominator };

        if mcc > best_mcc || (mcc == best_mcc && current_score < best_threshold) {
            best_mcc = mcc;
            best_threshold = current_score;
            best_acc = (tp + tn) as f64 / (total_pos + total_neg) as f64;
            best_sens = tp as f64 / total_pos as f64;
            best_spec = tn as f64 / total_neg as f64;
        }
    }

    (best_mcc, best_threshold, best_acc, best_sens, best_spec)
}


pub fn mean_and_std(values: &[f64]) -> (f64, f64) {
    let mut n = 0.0;
    let (mut mean, mut m2) = (0.0, 0.0);            // Welford
    for &x in values {
        n += 1.0;
        let delta = x - mean;
        mean += delta / n;
        m2   += delta * (x - mean);
    }
    (mean, (m2 / n).sqrt())
}

pub fn median(values: &mut [f64]) -> f64 {
    let mid = values.len() / 2;

    if values.is_empty() {
        return f64::NAN;
    }
    
    if values.iter().any(|&x| x.is_nan()) {
        return f64::NAN;
    }

    values.select_nth_unstable_by(mid, |a,b| a.partial_cmp(b).unwrap());
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        let max_low = *values[..mid].iter().max_by(|a,b| a.partial_cmp(b).unwrap()).unwrap();
        (max_low + values[mid]) / 2.0
    }
}

pub fn mad(values: &[f64]) -> f64 {
    let mut dev: Vec<f64> = {
        let mut buf = values.to_vec();
        let med = median(&mut buf);
        values.iter().map(|&v| (v - med).abs()).collect()
    };
    1.4826 * median(&mut dev)                
}

// Graphical functions
pub fn display_feature_importance_terminal(
    data: &Data,
    final_importances: &HashMap<usize, (f64, f64)>,
    nb_features: usize,
    aggregation_method: &ImportanceAggregation
) -> String {
    const GRAPH_WIDTH: usize = 80;
    const LEFT_MARGIN: usize = 30;
    const VALUE_AREA_WIDTH: usize = GRAPH_WIDTH - LEFT_MARGIN - 2;
    
    let mut importance_vec: Vec<(&usize, &(f64, f64))> = final_importances.iter().collect();
    importance_vec.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap_or(std::cmp::Ordering::Equal));
    
    let importance_vec = importance_vec.into_iter().take(nb_features).collect::<Vec<_>>();
    
    if importance_vec.is_empty() {
        return String::from("No features to display.");
    }
    
    let min_with_std = importance_vec.iter()
        .map(|(_, (imp, std))| imp - std)
        .fold(f64::MAX, |a, b| a.min(b));

    let max_with_std = importance_vec.iter()
        .map(|&(_, (imp, std))| imp + std)
        .fold(f64::MIN, f64::max);
    
    let scale_min = round_down_nicely(min_with_std);
    let scale_max = round_up_nicely(max_with_std);
    let scale_range = scale_max - scale_min;

    let scale_factor = if scale_range > 0.0 {
        VALUE_AREA_WIDTH as f64 / scale_range
    } else {
        VALUE_AREA_WIDTH as f64 / 1.0
    };
    
    let num_ticks = 7;

    let tick_positions: Vec<usize> = (0..num_ticks)
        .map(|i| i * VALUE_AREA_WIDTH / (num_ticks - 1))
        .collect();

    let mut result = String::new();
    
    result.push_str(match aggregation_method {
        ImportanceAggregation::median => "Feature importance using median aggregation method\n",
        ImportanceAggregation::mean => "Feature importance using mean aggregation method\n",
    });

    result.push_str(match aggregation_method {
        ImportanceAggregation::median => "Legend: • = importance value, <- - -> = confidence interval (±MAD)\n\n",
        ImportanceAggregation::mean => "Legend: • = importance value, <- - -> = confidence interval (±std dev)\n\n",
    });
    
    let header_line = format!("{:<LEFT_MARGIN$}|{:^VALUE_AREA_WIDTH$}|", "Feature", "Feature importance");
    result.push_str(&"-".repeat(LEFT_MARGIN));
    result.push_str("|-");
    result.push_str(&"-".repeat(VALUE_AREA_WIDTH));
    result.push_str("|\n");
    result.push_str(&header_line);
    result.push_str("\n");
    result.push_str(&"-".repeat(LEFT_MARGIN));
    result.push_str("|-");
    result.push_str(&"-".repeat(VALUE_AREA_WIDTH));
    result.push_str("|\n");
    
    for (i, (feature_idx, (importance, std_dev))) in importance_vec.iter().enumerate() {
        let feature_name = if data.features.len() > **feature_idx {
            &data.features[**feature_idx]
        } else {
            "Unknown"
        };
        
        let display_name = format!("#{} {}", i + 1, feature_name);
        let truncated_name = if display_name.len() > LEFT_MARGIN - 2 {
            format!("{}...", &display_name[0..LEFT_MARGIN - 5])
        } else {
            display_name
        };
        
        let normalized_importance = importance - scale_min;
        let normalized_min = (importance - std_dev - scale_min).max(0.0);
        let normalized_max = importance + std_dev - scale_min;
        
        let center_pos = (normalized_importance * scale_factor).round() as usize;
        let start_pos = (normalized_min * scale_factor).round() as usize;
        let end_pos = (normalized_max * scale_factor).round() as usize;
        let end_pos = std::cmp::min(end_pos, VALUE_AREA_WIDTH - 1);
        
        let left_margin = LEFT_MARGIN; 
        let mut line = format!("{:<left_margin$}|", truncated_name);
        
        for i in 0..VALUE_AREA_WIDTH {
            if i == center_pos {
                line.push('•');
            } else if i == start_pos {
                line.push('<');
            } else if i == end_pos {
                line.push('>');
            } else if i > start_pos && i < end_pos {
                line.push('-');
            } else {
                line.push(' ');
            }
        }
        
        line.push('|');
        result.push_str(&line);
        result.push('\n');
    }

    let mut marker_line = "-".repeat(LEFT_MARGIN);
    marker_line.push('|');
    
    for i in 0..VALUE_AREA_WIDTH {
        if tick_positions.contains(&i) {
            if i==0 {
                marker_line.push('-');
            } else {
                marker_line.push('|');
            }
            
        } else {
            marker_line.push('-');
        }
    }
    
    marker_line.push('|');
    result.push_str(&marker_line);
    result.push('\n');
    
    let mut scale_line = " ".repeat(LEFT_MARGIN + 1);

    for (i, &tick_pos) in tick_positions.iter().enumerate() {
        let tick_value = scale_min + i as f64 * (scale_range / (num_ticks - 1) as f64);
        let value_str = format_tick_value(tick_value);
        
        let label_width = value_str.len();
        let label_start = tick_pos.saturating_sub(label_width / 2);
        
        while scale_line.len() < LEFT_MARGIN + 1 + label_start {
            scale_line.push(' ');
        }

        scale_line.push_str(&value_str);
    }
    
    result.push_str(&scale_line);
    result.push_str("\n\n");
    
    result
}

fn format_tick_value(value: f64) -> String {
    let abs_value = value.abs();
    
    if abs_value == 0.0 {
        return "0".to_string();
    } else if abs_value < 0.001 || abs_value >= 10000.0 {
        return format!("{:.1e}", value);
    } else if abs_value < 0.01 {
        return format!("{:.4}", value);
    } else if abs_value < 0.1 {
        return format!("{:.3}", value);
    } else if abs_value < 1.0 {
        return format!("{:.2}", value);
    } else if abs_value < 10.0 {
        if (value - value.round()).abs() < 1e-10 {
            return format!("{:.0}", value);
        } else {
            return format!("{:.1}", value);
        }
    } else {
        if (value - value.round()).abs() < 1e-10 {
            return format!("{:.0}", value);
        } else {
            return format!("{:.1}", value);
        }
    }
}

fn round_up_nicely(value: f64) -> f64 {
    if value == 0.0 {
        return 0.1; 
    }
    
    if value < 0.0 {
        return -round_down_nicely(value.abs());
    }

    if value.abs() < 1e-10 {
        return if value >= 0.0 { 1e-10 } else { -1e-10 };
    }
    
    let magnitude = value.log10().floor();
    let power_of_ten = 10.0_f64.powf(magnitude);

    if value <= 1.0 * power_of_ten {
        1.0 * power_of_ten
    } else if value <= 2.0 * power_of_ten {
        2.0 * power_of_ten
    } else if value <= 5.0 * power_of_ten {
        5.0 * power_of_ten
    } else {
        10.0 * power_of_ten
    }
}

fn round_down_nicely(value: f64) -> f64 {
    if value < 0.0 {
        return -round_up_nicely(value.abs());
    }
    
    if value < 1e-10 {
        return 0.0; 
    }
    
    let magnitude = value.log10().floor();
    let power_of_ten = 10.0_f64.powf(magnitude);
    
    if value >= 5.0 * power_of_ten {
        5.0 * power_of_ten
    } else if value >= 2.0 * power_of_ten {
        2.0 * power_of_ten
    } else if value >= 1.0 * power_of_ten {
        1.0 * power_of_ten
    } else {
        0.5 * power_of_ten
    }
}

// Serde functions for JSONize HashMaps
pub mod serde_json_hashmap_numeric {
    use serde::{Serialize, Deserialize, Serializer, Deserializer};
    use std::collections::HashMap;
    
    // ===== General =====
    
    /// HashMap<usize, T>
    pub fn serialize_usize<S, T>(
        map: &HashMap<usize, T>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize + Clone,
    {
        let map_as_string: HashMap<String, T> = map.iter()
            .map(|(&k, v)| (k.to_string(), v.clone()))
            .collect();
        map_as_string.serialize(serializer)
    }
    
    /// HashMap<usize, T>
    pub fn deserialize_usize<'de, D, T>(
        deserializer: D,
    ) -> Result<HashMap<usize, T>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        let map_as_string: HashMap<String, T> = HashMap::deserialize(deserializer)?;
        let mut map = HashMap::new();
        for (k, v) in map_as_string {
            if let Ok(idx) = k.parse() {
                map.insert(idx, v);
            }
        }
        Ok(map)
    }
    
    /// HashMap<(usize, usize), T>
    pub fn serialize_tuple_usize<S, T>(
        map: &HashMap<(usize, usize), T>,
        serializer: S,
        ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize + Clone,
    {
        let map_as_string: Result<HashMap<String, T>, _> = map.iter()
            .map(|(&(i, j), v)| {
                Ok((format!("{},{}", i, j), v.clone()))
            })
            .collect();
        
        map_as_string?.serialize(serializer)
    }

    
    /// HashMap<(usize, usize), T>
    pub fn deserialize_tuple_usize<'de, D, T>(
        deserializer: D,
    ) -> Result<HashMap<(usize, usize), T>, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        let map_as_string: HashMap<String, T> = HashMap::deserialize(deserializer)?;
        let mut map = HashMap::new();
        
        for (k, v) in map_as_string {
            let parts: Vec<&str> = k.split(',').collect();
            if parts.len() == 2 {
                match (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    (Ok(i), Ok(j)) => {
                        map.insert((i, j), v);
                    }
                    _ => {
                        return Err(serde::de::Error::custom("Invalid tuple key format"));
                    }
                }
            }
        }
        Ok(map)
    }
    
    // ===== Specialized modules  =====
    
    /// HashMap<usize, i8> (Individual.features)
    pub mod usize_i8 {
        use super::*;
        
        pub fn serialize<S>(
            map: &HashMap<usize, i8>,
            serializer: S,
        ) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serialize_usize(map, serializer)
        }
        
        pub fn deserialize<'de, D>(
            deserializer: D,
        ) -> Result<HashMap<usize, i8>, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserialize_usize(deserializer)
        }
    }
    
    /// HashMap<usize, u8> (Data.featureclass)
    pub mod usize_u8 {
        use super::*;
        
        pub fn serialize<S>(
            map: &HashMap<usize, u8>,
            serializer: S,
        ) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serialize_usize(map, serializer)
        }
        
        pub fn deserialize<'de, D>(
            deserializer: D,
        ) -> Result<HashMap<usize, u8>, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserialize_usize(deserializer)
        }
    }
    
    /// HashMap<(usize, usize), f64> (Data.X)
    pub mod tuple_usize_f64 {
        use super::*;
        
        pub fn serialize<S>(
            map: &HashMap<(usize, usize), f64>,
            serializer: S,
        ) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serialize_tuple_usize(map, serializer)
        }
        
        pub fn deserialize<'de, D>(
            deserializer: D,
        ) -> Result<HashMap<(usize, usize), f64>, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserialize_tuple_usize(deserializer)
        }
    }

}

// unit tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::panic;
    use rand::SeedableRng;
    
    // tests for generate_random_vector
    #[test]
    fn test_generate_random_vector_vector_size() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let size = 10;
        let vector = generate_random_vector(size, &mut rng);
        assert_eq!(vector.len(), size, "the generated vector does not match the input size");
    }

    #[test]
    fn test_generate_random_vector_random_values() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let size = 100;
        let vector = generate_random_vector(size, &mut rng);

        for &value in &vector {
            assert!(value == -1 || value == 0 || value == 1, "the generated vector contains value.s outside [-1 ; 1]");
        }
    }

    #[test]
    fn test_generate_random_vector_empty_vector() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let size = 0;
        let vector = generate_random_vector(size, &mut rng);
        assert!(vector.is_empty(), "the generated vector should be empty for an input size of 0");
    }

    #[test]
    fn test_generate_random_vector_deterministic_output_and_reproductibility() {
        let mut rng1 = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let size = 10;

        let vector1 = generate_random_vector(size, &mut rng1);
        let vector2 = generate_random_vector(size, &mut rng2);

        assert_eq!(vector1, vector2, "the same seed generated two different vectors");
        assert_eq!(vector1, vec![-1, 1, -1, 1, 1, 0, 0, 0, 1, -1], "the generated vector isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
    }

    // tests for split_into_balanced_random_chunks
    #[test]
    fn test_split_into_balanced_random_chunks_split_remainder_division() {
        let chunks = split_into_balanced_random_chunks(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3, &mut ChaCha8Rng::seed_from_u64(42));
        assert_eq!(chunks.len(), 3, "the count of chunks does not match the input");
        assert_eq!(chunks[0].len(), 4, "the first chunk must have one more value when this is a remainder division");
        assert_eq!(chunks[1].len(), 3, "the count of value per chunck is not respected");
        assert_eq!(chunks[2].len(), 3, "the count of value per chunck is not respected");
    }

    #[test]
    fn test_split_into_balanced_random_chunks_split_remainderless_division() {
        let chunks = split_into_balanced_random_chunks(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], 3, &mut ChaCha8Rng::seed_from_u64(42));
        assert_eq!(chunks.len(), 3, "the count of chunks does not match the input");
        assert_eq!(chunks[0].len(), 3, "the first chunk must have the same number of value when this is a remainderless division");
        assert_eq!(chunks[1].len(), 3, "the count of value per chunck is not respected");
        assert_eq!(chunks[2].len(), 3, "the count of value per chunck is not respected");
    }

    #[test]
    fn test_split_into_balanced_random_chunks_split_into_single_chunk() {
        let chunks = split_into_balanced_random_chunks(vec![1, 2, 3, 4, 5], 1, &mut ChaCha8Rng::seed_from_u64(42));
        assert_eq!(chunks.len(), 1, "the count of chunks does not match the input");
        assert_eq!(chunks[0].len(), 5, "when splitted in one part, the chunk must be equal to the input vector");
    }

    #[test]
    fn test_split_into_balanced_random_chunks_split_empty_vectors() {
        let vec: Vec<i32> = vec![];
        let chunks = split_into_balanced_random_chunks(vec, 3, &mut ChaCha8Rng::seed_from_u64(42));
        assert_eq!(chunks.len(), 3, "the count of chunks does not match the input");
        for chunk in chunks {
            assert!(chunk.is_empty(), "empty vector should to the formation of empty chunks");
        }
    }

    #[test]
    fn test_split_into_balanced_random_chunks_split_more_chunks_than_elements() {
        let vec = vec![1, 2, 3];
        let chunks = split_into_balanced_random_chunks(vec.clone(), 5, &mut ChaCha8Rng::seed_from_u64(42));

        // vecs 1, 2, 3 should contain one value, vecs 4 and 5 should be empty
        assert_eq!(chunks.len(), 5, "the count of chunks does not match the input");
        assert_eq!(chunks.iter().filter(|chunk| !chunk.is_empty()).count(), vec.len(), "the chunks exceeding the values count must be empty");
    }

    #[test]
    fn test_split_into_balanced_random_chunks_deterministic_split_and_reproductibility() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut rng2 = ChaCha8Rng::seed_from_u64(42);
        let vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let p = 4;

        let chunks1 = split_into_balanced_random_chunks(vec.clone(), p, &mut rng);
        let chunks2 = split_into_balanced_random_chunks(vec.clone(), p, &mut rng2);

        assert_eq!(chunks1, chunks2, "the same seed generated two different chunks");
        assert_eq!(chunks1, vec![vec![1, 4, 6], vec![8, 10, 5], vec![9, 2], vec![7, 3]], "the generated chunks isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
    }

    // tests for shuffle_row
    #[test]
    fn test_shuffle_row_preserves_values_and_reorder() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();

        X.insert((0, 0), 1.0);
        X.insert((1, 0), 2.0);
        X.insert((2, 0), 3.0);
        X.insert((3, 0), 4.0);

        let sample_len = 4;
        let feature = 0;

        shuffle_row(&mut X, sample_len, feature, &mut rng);

        let values: Vec<f64> = (0..sample_len).filter_map(|i| X.get(&(i, feature)).copied()).collect();
        assert_eq!(values.len(), 4, "HashMap must contain the same number of values after shuffle");
        assert!(values.contains(&1.0), "the shuffle must conserve HashMap values");
        assert!(values.contains(&2.0), "the shuffle must conserve HashMap values");
        assert!(values.contains(&3.0), "the shuffle must conserve HashMap values");
        assert!(values.contains(&4.0), "the shuffle must conserve HashMap values");
        assert_ne!(values, vec![1.0, 2.0, 3.0, 4.0], "the shuffle must conserve HashMap values");
    }

    #[test]
    fn test_shuffle_row_empty_column() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut X: HashMap<(usize, usize), f64> = HashMap::new();

        let sample_len = 4;
        let feature = 0;

        shuffle_row(&mut X, sample_len, feature, &mut rng);

        for i in 0..sample_len {
            assert!(!X.contains_key(&(i, feature)), "the shuffle of an empty HashMap should also be empty");
        }
    }

    #[test]
    fn test_shuffle_row_deterministic_shuffle_and_reproductibility() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut X1: HashMap<(usize, usize), f64> = HashMap::new();
        
        X1.insert((0, 0), 1.0);
        X1.insert((1, 0), 2.0);
        X1.insert((2, 0), 3.0);
        X1.insert((3, 0), 4.0);

        let mut X2 = X1.clone();
        let sample_len = 4;
        let feature = 0;

        shuffle_row(&mut X1, sample_len, feature, &mut ChaCha8Rng::seed_from_u64(42));
        shuffle_row(&mut X2, sample_len, feature, &mut rng);
        
        assert_eq!(X1, X2, "the same seed generated two different chunks");
        assert_eq!(X1.get(&(0, 0)), Some(&2.0), "the generated HeatMap isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
        assert_eq!(X1.get(&(2, 0)), Some(&1.0), "the generated HeatMap isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
        assert_eq!(X1.get(&(1, 0)), Some(&3.0), "the generated HeatMap isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
        assert_eq!(X1.get(&(3, 0)), Some(&4.0), "the generated HeatMap isn't the same as generated in the past, indicating a reproducibility problem probably linked to the seed interpretation");
    }   

    #[test]
    fn test_conf_inter_binomial(){
        // assert_eq(Rust function results  == R function results)
        assert_eq!(conf_inter_binomial(0.0, 50, 0.05), (0_f64, 0_f64, 0_f64));
        assert_eq!(conf_inter_binomial(0.76, 50, 0.05), (0.6416207713410322_f64, 0.76_f64, 0.8783792286589678_f64));
        assert_eq!(conf_inter_binomial(1.0, 50, 0.05), (1_f64, 1_f64, 1_f64));

        // control panic! to avoid statistical issues due to invalid input
        let resultErrZeroSample = panic::catch_unwind(|| { conf_inter_binomial(0.76, 0, 0.05) });
        assert!(resultErrZeroSample.is_err(), "function should panic! when there is no sample");

        let resultErrInf = panic::catch_unwind(|| { conf_inter_binomial(-0.3, 50, 0.05) });
        assert!(resultErrInf.is_err(), "function should panic! for an accuracy lower than 0");

        let resultErrSup = panic::catch_unwind(|| { conf_inter_binomial(1.3, 50, 0.05) });
        assert!(resultErrSup.is_err(), "function should panic! for an accuracy greater than 1");
    }

    #[test]
    fn test_compute_auc_from_value_perfect_classification() {
        let value = vec![0.1, 0.2, 0.8, 0.9];
        let y = vec![0, 0, 1, 1];
        let auc = compute_auc_from_value(&value, &y);
        assert_eq!(auc, 1.0, "Perfect classification should yield AUC = 1.0");
    }

    #[test]
    fn test_compute_auc_from_value_random_classification() {
        let value = vec![0.5, 0.5, 0.5, 0.5];
        let y = vec![0, 1, 0, 1];
        let auc = compute_auc_from_value(&value, &y);
        assert_eq!(auc, 0.5, "Random classification should yield AUC = 0.5");
    }

    #[test]
    fn test_compute_auc_from_value_empty_vectors() {
        let value = vec![];
        let y = vec![];
        let auc = compute_auc_from_value(&value, &y);
        assert_eq!(auc, 0.5, "Empty vectors should yield AUC = 0.5");
    }

    #[test]
    fn test_compute_auc_from_value_single_class_only() {
        let value = vec![0.1, 0.2, 0.3, 0.4];
        let y = vec![0, 0, 0, 0];
        let auc = compute_auc_from_value(&value, &y);
        assert_eq!(auc, 0.5, "Single class only should yield AUC = 0.5");
    }

    #[test]
    fn test_compute_auc_from_value_ties_handling() {
        let value = vec![0.5, 0.5, 0.5, 0.5];
        let y = vec![0, 0, 1, 1];
        let auc = compute_auc_from_value(&value, &y);
        assert_eq!(auc, 0.5, "Ties should be handled correctly");
    }

    #[test]
    fn test_compute_auc_from_value_infinite_values() {
        let value = vec![f64::NEG_INFINITY, 0.5, f64::INFINITY];
        let y = vec![0, 1, 1];
        let auc = compute_auc_from_value(&value, &y);
        assert!(auc >= 0.0 && auc <= 1.0, "Should handle infinite values gracefully");
    }

    #[test]
    fn test_compute_auc_large_dataset() {
        let n = 10000;
        let value: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<u8> = (0..n).map(|i| if i < n/2 { 0 } else { 1 }).collect();
        let auc = compute_auc_from_value(&value, &y);
        assert!((auc - 1.0).abs() < 1e-10, "Large sorted dataset should yield perfect AUC");
    }

    #[test]
    fn test_compute_metrics_from_classes_perfect_predictions() {
        let predicted = vec![0, 1, 0, 1];
        let y = vec![0, 1, 0, 1];
        let (accuracy, sensitivity, specificity) = compute_metrics_from_classes(&predicted, &y);
        assert_eq!(accuracy, 1.0, "Perfect predictions should yield 100% accuracy");
        assert_eq!(sensitivity, 1.0, "Perfect predictions should yield 100% sensitivity");
        assert_eq!(specificity, 1.0, "Perfect predictions should yield 100% specificity");
    }

    #[test]
    fn test_compute_metrics_from_classes_all_wrong_predictions() {
        let predicted = vec![1, 0, 1, 0];
        let y = vec![0, 1, 0, 1];
        let (accuracy, sensitivity, specificity) = compute_metrics_from_classes(&predicted, &y);
        assert_eq!(accuracy, 0.0, "All wrong predictions should yield 0% accuracy");
        assert_eq!(sensitivity, 0.0, "All wrong predictions should yield 0% sensitivity");
        assert_eq!(specificity, 0.0, "All wrong predictions should yield 0% specificity");
    }

    #[test]
    fn test_compute_metrics_from_classes_mixed_predictions() {
        let predicted = vec![0, 1, 0, 0];
        let y = vec![0, 1, 1, 0];
        let (accuracy, sensitivity, specificity) = compute_metrics_from_classes(&predicted, &y);
        assert_eq!(accuracy, 0.75, "Mixed predictions should yield expected accuracy");
        assert_eq!(sensitivity, 0.5, "Mixed predictions should yield expected sensitivity");
        assert_eq!(specificity, 1.0, "Mixed predictions should yield expected specificity");
    }

    #[test]
    fn test_compute_metrics_from_classes_class_2_ignored() {
        let predicted = vec![0, 1, 0, 1];
        let y = vec![0, 1, 2, 1];
        let (accuracy, _, _) = compute_metrics_from_classes(&predicted, &y);
        assert_eq!(accuracy, 1.0, "Class 2 should be ignored in calculations");
    }

    #[test]
    fn test_compute_metrics_from_classes_empty_vectors() {
        let predicted = vec![];
        let y = vec![];
        let (accuracy, sensitivity, specificity) = compute_metrics_from_classes(&predicted, &y);
        assert_eq!(accuracy, 0.0, "Empty vectors should yield 0 metrics");
        assert_eq!(sensitivity, 0.0, "Empty vectors should yield 0 metrics");
        assert_eq!(specificity, 0.0, "Empty vectors should yield 0 metrics");
    }

    #[test]
    fn test_compute_metrics_extreme_imbalance() {
        let predicted = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        let (accuracy, sensitivity, specificity) = compute_metrics_from_classes(&predicted, &y);
        assert_eq!(accuracy, 1.0, "Perfect predictions should yield 100% accuracy even with imbalance");
        assert_eq!(sensitivity, 1.0, "Perfect predictions should yield 100% sensitivity");
        assert_eq!(specificity, 1.0, "Perfect predictions should yield 100% specificity");
    }

    #[test]
    fn test_compute_roc_and_metrics_from_value_basic_case() {
        let value = vec![0.1, 0.4, 0.6, 0.9];
        let y = vec![0, 0, 1, 1];
        let (auc, threshold, accuracy, sensitivity, specificity, _) = 
            compute_roc_and_metrics_from_value(&value, &y, None);
        assert!(auc >= 0.0 && auc <= 1.0, "AUC should be between 0 and 1");
        assert!(accuracy >= 0.0 && accuracy <= 1.0, "Accuracy should be between 0 and 1");
        assert!(sensitivity >= 0.0 && sensitivity <= 1.0, "Sensitivity should be between 0 and 1");
        assert!(specificity >= 0.0 && specificity <= 1.0, "Specificity should be between 0 and 1");
        assert!(!threshold.is_nan(), "Threshold should not be NaN");
    }

    #[test]
    fn test_compute_roc_and_metrics_from_value_with_penalties() {
        let value = vec![0.1, 0.4, 0.6, 0.9];
        let y = vec![0, 0, 1, 1];
        let penalties = Some(vec![2.0, 1.0]);
        let (_, _, _, _, _, objective) = 
            compute_roc_and_metrics_from_value(&value, &y, penalties.as_deref());
        assert!(objective >= 0.0, "Objective should be non-negative");
    }

    #[test]
    fn test_compute_roc_and_metrics_from_value_without_penalties() {
        let value = vec![0.1, 0.4, 0.6, 0.9];
        let y = vec![0, 0, 1, 1];
        let (_, _, _, sensitivity, specificity, objective) = 
            compute_roc_and_metrics_from_value(&value, &y, None);
        let expected_youden = sensitivity + specificity - 1.0;
        assert!((objective - expected_youden).abs() < 1e-10, "Without penalties, objective should be Youden index");
    }

    #[test]
    fn test_compute_roc_and_metrics_from_value_single_class_only() {
        let value = vec![0.1, 0.2, 0.3, 0.4];
        let y = vec![0, 0, 0, 0];
        let (auc, threshold, _, _, _, _) = 
            compute_roc_and_metrics_from_value(&value, &y, None);
        assert_eq!(auc, 0.5, "Single class should yield AUC = 0.5");
        assert!(threshold.is_nan(), "Single class should yield NaN threshold");
    }

    #[test]
    fn test_compute_mcc_and_metrics_from_value_perfect_classification() {
        let value = vec![0.1, 0.2, 0.8, 0.9];
        let y = vec![0, 0, 1, 1];
        let (mcc, _, accuracy, sensitivity, specificity) = 
            compute_mcc_and_metrics_from_value(&value, &y);
        assert_eq!(mcc, 1.0, "Perfect classification should yield MCC = 1.0");
        assert_eq!(accuracy, 1.0, "Perfect classification should yield accuracy = 1.0");
        assert_eq!(sensitivity, 1.0, "Perfect classification should yield sensitivity = 1.0");
        assert_eq!(specificity, 1.0, "Perfect classification should yield specificity = 1.0");
    }

    #[test]
    fn test_compute_mcc_and_metrics_from_value_random_classification() {
        let value = vec![0.5, 0.5, 0.5, 0.5];
        let y = vec![0, 1, 0, 1];
        let (mcc, _, _, _, _) = compute_mcc_and_metrics_from_value(&value, &y);
        assert!(mcc.abs() < 0.1, "Random classification should yield MCC ≈ 0");
    }

    #[test]
    fn test_compute_mcc_and_metrics_from_value_single_class_only() {
        let value = vec![0.1, 0.2, 0.3, 0.4];
        let y = vec![0, 0, 0, 0];
        let (mcc, threshold, _, _, _) = compute_mcc_and_metrics_from_value(&value, &y);
        assert_eq!(mcc, 0.0, "Single class should yield MCC = 0.0");
        assert!(threshold.is_nan(), "Single class should yield NaN threshold");
    }

    #[test]
    fn test_compute_mcc_zero_division_case() {
        let value = vec![0.5, 0.5, 0.5, 0.5];
        let y = vec![1, 1, 1, 1];
        let (mcc, _, _, _, _) = compute_mcc_and_metrics_from_value(&value, &y);
        assert_eq!(mcc, 0.0, "MCC with single class should be 0.0");
    }

    #[test]
    fn test_compute_mcc_and_metrics_from_value_empty_input() {
        let value = vec![];
        let y = vec![];
        let (mcc, threshold, _, _, _) = 
            compute_mcc_and_metrics_from_value(&value, &y);
        assert_eq!(mcc, 0.0, "Empty input should yield MCC = 0.0");
        assert!(threshold.is_nan(), "Empty input should yield NaN threshold");
    }

    #[test]
    fn test_mean_and_std_normal_values() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std) = mean_and_std(&values);
        assert_eq!(mean, 3.0, "Mean of 1-5 should be 3.0");
        assert!((std - 2.0f64.sqrt()).abs() < 1e-10, "Standard deviation should be √2");
    }

    #[test]
    fn test_mean_and_std_single_value() {
        let values = vec![42.0];
        let (mean, std) = mean_and_std(&values);
        assert_eq!(mean, 42.0, "Mean of single value should be that value");
        assert_eq!(std, 0.0, "Standard deviation of single value should be 0");
    }

    #[test]
    fn test_mean_and_std_empty_slice() {
        let values: Vec<f64> = vec![];
        let (mean, std) = mean_and_std(&values);
        assert_eq!(mean, 0.0, "Welford algorithm starts with mean = 0.0");
        assert!(std.is_nan(), "Standard deviation of empty slice should be NaN");
    }

    #[test]
    fn test_mean_and_std_identical_values() {
        let values = vec![5.0, 5.0, 5.0, 5.0];
        let (mean, std) = mean_and_std(&values);
        assert_eq!(mean, 5.0, "Mean of identical values should be that value");
        assert_eq!(std, 0.0, "Standard deviation of identical values should be 0");
    }

    #[test]
    fn test_mean_and_std_negative_values() {
        let values = vec![-1.0, -2.0, -3.0];
        let (mean, std) = mean_and_std(&values);
        assert_eq!(mean, -2.0, "Mean should handle negative values");
        assert!(std > 0.0, "Standard deviation should be positive for varying values");
    }

    #[test]
    fn test_mean_and_std_with_nan() {
        let values = vec![1.0, f64::NAN, 3.0];
        let (mean, std) = mean_and_std(&values);
        assert!(mean.is_nan(), "Welford with NaN should propagate NaN to mean");
        assert!(std.is_nan(), "Welford with NaN should propagate NaN to std");
    }

    #[test]
    fn test_mean_and_std_large_dataset() {
        let values: Vec<f64> = (0..100000).map(|i| i as f64).collect();
        let (mean, std) = mean_and_std(&values);
        let expected_mean = 49999.5;
        assert!((mean - expected_mean).abs() < 1e-6, "Large dataset mean should be accurate");
        assert!(std > 0.0, "Large dataset std should be positive");
    }

    #[test]
    fn test_median_odd_length() {
        let mut values = vec![1.0, 3.0, 2.0];
        let result = median(&mut values);
        assert_eq!(result, 2.0, "Median of [1,2,3] should be 2");
    }

    #[test]
    fn test_median_even_length() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let result = median(&mut values);
        assert_eq!(result, 2.5, "Median of [1,2,3,4] should be 2.5");
    }

    #[test]
    fn test_median_single_value() {
        let mut values = vec![42.0];
        let result = median(&mut values);
        assert_eq!(result, 42.0, "Median of single value should be that value");
    }

    #[test]
    fn test_median_two_values() {
        let mut values = vec![1.0, 3.0];
        let result = median(&mut values);
        assert_eq!(result, 2.0, "Median of two values should be their average");
    }

    #[test]
    fn test_median_unsorted_values() {
        let mut values = vec![5.0, 1.0, 9.0, 3.0, 7.0];
        let result = median(&mut values);
        assert_eq!(result, 5.0, "Median should work on unsorted data");
    }

    #[test]
    fn test_median_with_nan() {
        let mut values = vec![1.0, f64::NAN, 3.0];
        let result = median(&mut values);
        assert!(result.is_nan(), "Median with NaN should return NaN");
    }

    #[test]
    fn test_mad_normal_distribution() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = mad(&values);
        assert!(result > 0.0, "MAD should be positive for varying values");
    }

    #[test]
    fn test_mad_constant_values() {
        let values = vec![5.0, 5.0, 5.0, 5.0];
        let result = mad(&values);
        assert_eq!(result, 0.0, "MAD of constant values should be 0");
    }

    #[test]
    fn test_mad_single_value() {
        let values = vec![42.0];
        let result = mad(&values);
        assert_eq!(result, 0.0, "MAD of single value should be 0");
    }

    #[test]
    fn test_mad_two_values() {
        let values = vec![1.0, 3.0];
        let result = mad(&values);
        assert!(result > 0.0, "MAD of two different values should be positive");
    }

    #[test]
    fn test_mad_median_consistency() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut values_for_median = values.clone();
        let med = median(&mut values_for_median);
        let mad_result = mad(&values);
        
        assert_eq!(med, 3.0, "Median should be 3.0");
        assert!(mad_result > 0.0, "MAD should be positive for varying values");
        
        let expected_mad = 1.4826;
        assert!((mad_result - expected_mad).abs() < 0.001, "MAD should match expected calculation");
    }
    #[test]
    fn test_display_feature_importance_terminal_basic_output() {
        let mut data = Data::new();
        data.features = vec!["feature1".to_string(), "feature2".to_string()];
        
        let mut importance_map = HashMap::new();
        importance_map.insert(0, (0.8, 0.1));
        importance_map.insert(1, (0.6, 0.05));
        
        let result = display_feature_importance_terminal(&data, &importance_map, 2, &ImportanceAggregation::median);
        assert!(result.contains("Feature importance"), "Output should contain title");
        assert!(result.contains("feature1"), "Output should contain feature names");
        assert!(result.contains("•"), "Output should contain importance markers");
    }

    #[test]
    fn test_display_feature_importance_terminal_empty_features() {
        let data = Data::new();
        let importance_map = HashMap::new();
        
        let result = display_feature_importance_terminal(&data, &importance_map, 5, &ImportanceAggregation::mean);
        assert_eq!(result, "No features to display.", "Empty features should return appropriate message");
    }

    #[test]
    fn test_display_feature_importance_terminal_median_aggregation() {
        let mut data = Data::new();
        data.features = vec!["test_feature".to_string()];
        
        let mut importance_map = HashMap::new();
        importance_map.insert(0, (0.5, 0.1));
        
        let result = display_feature_importance_terminal(&data, &importance_map, 1, &ImportanceAggregation::median);
        assert!(result.contains("median aggregation"), "Should indicate median aggregation method");
        assert!(result.contains("±MAD"), "Should show MAD for median aggregation");
    }

    #[test]
    fn test_display_feature_importance_negative_importance() {
        let mut data = Data::new();
        data.features = vec!["negative_feature".to_string()];
        
        let mut importance_map = HashMap::new();
        importance_map.insert(0, (-0.5, 0.1));
        
        let result = display_feature_importance_terminal(&data, &importance_map, 1, &ImportanceAggregation::mean);
        assert!(result.contains("negative_feature"), "Should handle negative importance");
        assert!(result.contains("•"), "Should still display marker for negative values");
    }

    #[test]
    fn test_display_feature_importance_very_large_names() {
        let mut data = Data::new();
        let long_name = "a".repeat(100);
        data.features = vec![long_name.clone()];
        
        let mut importance_map = HashMap::new();
        importance_map.insert(0, (0.5, 0.1));
        
        let result = display_feature_importance_terminal(&data, &importance_map, 1, &ImportanceAggregation::mean);
        assert!(result.contains("..."), "Should truncate very long feature names");
        assert!(!result.contains(&long_name), "Should not contain full long name");
    }

    #[test]
    fn test_format_tick_value_zero() {
        assert_eq!(format_tick_value(0.0), "0");
    }

    #[test]
    fn test_format_tick_value_small_positive() {
        assert_eq!(format_tick_value(0.005), "0.0050");
        assert_eq!(format_tick_value(0.05), "0.050");
        assert_eq!(format_tick_value(0.5), "0.50");
    }

    #[test]
    fn test_format_tick_value_small_negative() {
        assert_eq!(format_tick_value(-0.005), "-0.0050");
        assert_eq!(format_tick_value(-0.5), "-0.50");
    }

    #[test]
    fn test_format_tick_value_large_values() {
        assert!(format_tick_value(15000.0).contains("e"), "Large values should use scientific notation");
        assert!(format_tick_value(-20000.0).contains("e"), "Large negative values should use scientific notation");
    }

    #[test]
    fn test_format_tick_value_scientific_notation() {
        assert!(format_tick_value(0.0005).contains("e"), "Very small values should use scientific notation");
        assert!(format_tick_value(-0.0001).contains("e"), "Very small negative values should use scientific notation");
    }

    #[test]
    fn test_format_tick_value_integer_values() {
        assert_eq!(format_tick_value(5.0), "5");
        assert_eq!(format_tick_value(42.0), "42");
    }

    #[test]
    fn test_format_tick_value_edge_cases() {
        assert_eq!(format_tick_value(f64::MIN_POSITIVE), "2.2e-308");
        assert_eq!(format_tick_value(-f64::MIN_POSITIVE), "-2.2e-308");
        assert_eq!(format_tick_value(1.0000001), "1.0");
        assert_eq!(format_tick_value(0.99999), "1.00");
    }

    #[test]
    fn test_round_up_nicely_positive_values() {
        assert_eq!(round_up_nicely(1.3), 2.0);
        assert_eq!(round_up_nicely(2.5), 5.0);
        assert_eq!(round_up_nicely(7.0), 10.0);
    }

    #[test]
    fn test_round_up_nicely_negative_values() {
        assert_eq!(round_up_nicely(-1.3), -1.0);
        assert!(round_up_nicely(-2.5) > -5.0);
    }

    #[test]
    fn test_round_up_nicely_zero() {
        assert_eq!(round_up_nicely(0.0), 0.1);
    }

    #[test]
    fn test_round_up_nicely_very_small_values() {
        let result = round_up_nicely(0.0003);
        assert!(result > 0.0003, "Should round up small positive values");
    }

    #[test]
    fn test_round_up_nicely_powers_of_ten() {
        assert_eq!(round_up_nicely(1.0), 1.0);
        assert_eq!(round_up_nicely(10.0), 10.0);
        assert_eq!(round_up_nicely(100.0), 100.0);
    }

    #[test]
    fn test_round_up_nicely_extreme_values() {
        assert!(round_up_nicely(f64::MIN_POSITIVE) > 0.0);
        assert!(round_up_nicely(f64::EPSILON) > 0.0);
        assert_eq!(round_up_nicely(f64::EPSILON), 1e-10);
    }

    #[test]
    fn test_round_down_nicely_positive_values() {
        assert_eq!(round_down_nicely(1.3), 1.0);
        assert_eq!(round_down_nicely(2.5), 2.0);
        assert_eq!(round_down_nicely(7.0), 5.0);
    }

    #[test]
    fn test_round_down_nicely_negative_values() {
        assert!(round_down_nicely(-1.3) < -1.3);
        assert!(round_down_nicely(-2.5) < -2.5);
    }

    #[test]
    fn test_round_down_nicely_very_small_values() {
        assert_eq!(round_down_nicely(1e-12), 0.0);
    }

    #[test]
    fn test_round_down_nicely_powers_of_ten() {
        assert_eq!(round_down_nicely(1.0), 1.0);
        assert_eq!(round_down_nicely(10.0), 10.0);
        assert_eq!(round_down_nicely(100.0), 100.0);
    }

    #[test]
    fn test_round_functions_with_extreme_values() {
        assert!(round_up_nicely(f64::MIN_POSITIVE) > 0.0);
        assert!(round_down_nicely(f64::MAX) > 0.0);
        assert_eq!(round_up_nicely(f64::EPSILON), 1e-10);
        assert_eq!(round_down_nicely(f64::EPSILON), 0.0);
    }

    #[test]
    fn test_serialize_empty_hashmap() {
        let map: HashMap<usize, i8> = HashMap::new();
        let serialized = serde_json::to_string(&map).unwrap();
        let deserialized: HashMap<usize, i8> = serde_json::from_str(&serialized).unwrap();
        assert!(deserialized.is_empty(), "Empty map serialization should work");
    }

    #[test]
    fn test_serialize_large_keys() {
        let mut map = HashMap::new();
        map.insert(usize::MAX, 42_i8);
        map.insert(0, -42_i8);
        
        let serialized = serde_json::to_string(&map).unwrap();
        let deserialized: HashMap<usize, i8> = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(map, deserialized, "Large keys should serialize correctly");
    }

    #[test]
    fn test_serialize_tuple_edge_cases() {
        use crate::utils::serde_json_hashmap_numeric::{serialize_tuple_usize, deserialize_tuple_usize};
        use serde_json::Value;
        
        let mut original_map = HashMap::new();
        original_map.insert((0, 0), 0.0_f64);
        original_map.insert((usize::MAX, usize::MAX), f64::MAX);
        original_map.insert((1, 2), f64::MIN_POSITIVE);
        

        let serialized_value = serialize_tuple_usize(&original_map, serde_json::value::Serializer).unwrap();
        let serialized_string = serde_json::to_string_pretty(&serialized_value).unwrap();
        
        assert!(serialized_string.contains("\"0,0\""));
        assert!(serialized_string.contains(&format!("\"{},{}\"", usize::MAX, usize::MAX)));
        assert!(serialized_string.contains("\"1,2\""));
        
        let value: Value = serde_json::from_str(&serialized_string).unwrap();
        let deserialized = deserialize_tuple_usize(value).unwrap();
        
        assert_eq!(original_map, deserialized, "Tuple key serialization roundtrip should work correctly");
    }

    #[test]
    fn test_roundtrip_serialization_usize_i8() {
        let mut original_map = HashMap::new();
        original_map.insert(1, 1_i8);
        original_map.insert(2, -1_i8);
        
        let serialized = serde_json::to_string(&original_map).unwrap();
        let deserialized: HashMap<usize, i8> = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(original_map, deserialized, "Roundtrip serialization should preserve data");
    }

    #[test]
    fn test_roundtrip_serialization_usize_u8() {
        let mut original_map = HashMap::new();
        original_map.insert(1, 255_u8);
        original_map.insert(2, 0_u8);
        
        let serialized = serde_json::to_string(&original_map).unwrap();
        let deserialized: HashMap<usize, u8> = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(original_map, deserialized, "u8 roundtrip serialization should work");
    }

    #[test]
    fn test_roundtrip_serialization_tuple_usize_f64() {
        use crate::utils::serde_json_hashmap_numeric::{serialize_tuple_usize, deserialize_tuple_usize};
        
        let mut original_map = HashMap::new();
        original_map.insert((0, 1), 1.5_f64);
        original_map.insert((2, 3), -2.7_f64);
        
        // Serialize HashMap<String, f64>
        let string_map: HashMap<String, f64> = original_map.iter()
            .map(|(&(i, j), &v)| (format!("{},{}", i, j), v))
            .collect();
        
        let serialized = serde_json::to_string(&string_map).unwrap();
        let string_map_deserialized: HashMap<String, f64> = serde_json::from_str(&serialized).unwrap();
        
        // Deserialize HashMap<(usize, usize), f64>
        let mut deserialized = HashMap::new();
        for (k, v) in string_map_deserialized {
            let parts: Vec<&str> = k.split(',').collect();
            if parts.len() == 2 {
                if let (Ok(i), Ok(j)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    deserialized.insert((i, j), v);
                }
            }
        }
        
        assert_eq!(original_map, deserialized, "Tuple key serialization should work correctly");
    }

    #[test]
    #[should_panic(expected = "confInterBinomial: Sample size (n) must be greater than zero.")]
    fn test_conf_inter_binomial_zero_samples() {
        conf_inter_binomial(0.5, 0, 0.05);
    }

    #[test]
    #[should_panic(expected = "confInterBinomial: accuracy should not be lower than 0 or greater than 1")]
    fn test_conf_inter_binomial_invalid_accuracy() {
        conf_inter_binomial(1.5, 50, 0.05);
    }

    #[test]
    #[should_panic(expected = "confInterBinomial: alpha should not be lower than 0 or greater than 1")]
    fn test_conf_inter_binomial_invalid_alpha() {
        conf_inter_binomial(0.5, 50, 1.5);
    }

    #[test]
    fn test_auc_roc_mcc_consistency() {
        let value = vec![0.1, 0.4, 0.6, 0.9];
        let y = vec![0, 0, 1, 1];
        
        let auc1 = compute_auc_from_value(&value, &y);
        let (auc2, _, _, _, _, _) = compute_roc_and_metrics_from_value(&value, &y, None);
        
        assert!((auc1 - auc2).abs() < 1e-10, "AUC should be consistent between functions");
    }

    #[test]
    fn test_compute_roc_large_dataset() {
        let n = 5000;
        let value: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) + 0.001 * (i % 10) as f64).collect();
        let y: Vec<u8> = (0..n).map(|i| if i < n/2 { 0 } else { 1 }).collect();
        let (auc, _, _, _, _, _) = compute_roc_and_metrics_from_value(&value, &y, None);
        assert!(auc > 0.9, "Large dataset with clear separation should yield high AUC");
    }

    #[test]
    fn test_compute_auc_manual_example_1() {
        // Simple case: 4 samples with clear classification
        let value = vec![0.1, 0.3, 0.7, 0.9];
        let y = vec![0, 0, 1, 1];
        
        // Manual AUC calculation:
        // Pairs (negative, positive): (0.1,0.7), (0.1,0.9), (0.3,0.7), (0.3,0.9)
        // All 4 pairs are correctly ordered (negative < positive)
        // AUC = 4/4 = 1.0
        
        let auc = compute_auc_from_value(&value, &y);
        assert!((auc - 1.0).abs() < 1e-10, "AUC should be 1.0 for perfect separation");
    }

    #[test]
    fn test_compute_auc_manual_example_2() {
        // Partially inverted classification
        let value = vec![0.8, 0.6, 0.4, 0.2];
        let y = vec![0, 0, 1, 1];
        
        // Manual AUC calculation:
        // Pairs (negative, positive): (0.8,0.4), (0.8,0.2), (0.6,0.4), (0.6,0.2)
        // No pair correctly ordered (all negative > positive)
        // AUC = 0/4 = 0.0
        
        let auc = compute_auc_from_value(&value, &y);
        assert!((auc - 0.0).abs() < 1e-10, "AUC should be 0.0 for completely inverse classification");
    }

    #[test]
    fn test_compute_auc_manual_example_3() {
        // Medium classification performance
        let value = vec![0.2, 0.6, 0.4, 0.8];
        let y = vec![0, 0, 1, 1];
        
        // Manual AUC calculation:
        // Pairs (negative, positive): (0.2,0.4), (0.2,0.8), (0.6,0.4), (0.6,0.8)
        // Correct pairs: (0.2,0.4), (0.2,0.8), (0.6,0.8) = 3/4
        // AUC = 3/4 = 0.75
        
        let auc = compute_auc_from_value(&value, &y);
        assert!((auc - 0.75).abs() < 1e-10, "AUC should be 0.75");
    }

    #[test]
    fn test_compute_auc_manual_example_4() {
        // Case with ties handling
        let value = vec![0.5, 0.5, 0.3, 0.7];
        let y = vec![0, 1, 0, 1];
        
        // Manual AUC calculation with ties:
        // Pairs (negative, positive): (0.5,0.5), (0.5,0.7), (0.3,0.5), (0.3,0.7)
        // (0.5,0.5) counts as 0.5, (0.5,0.7) = 1, (0.3,0.5) = 1, (0.3,0.7) = 1
        // AUC = (0.5 + 1 + 1 + 1) / 4 = 3.5/4 = 0.875
        
        let auc = compute_auc_from_value(&value, &y);
        assert!((auc - 0.875).abs() < 1e-10, "AUC should be 0.875 with ties");
    }

    #[test]
    fn test_compute_auc_manual_example_5() {
        // Imbalanced dataset
        let value = vec![0.1, 0.2, 0.9];
        let y = vec![0, 0, 1];
        
        // Manual AUC calculation:
        // Pairs (negative, positive): (0.1,0.9), (0.2,0.9)
        // All correct: 2/2 = 1.0
        
        let auc = compute_auc_from_value(&value, &y);
        assert!((auc - 1.0).abs() < 1e-10, "AUC should be 1.0 for imbalanced but perfect classification");
    }

    #[test]
    fn test_compute_mcc_manual_example_1() {
        // Perfect balanced classification
        let value = vec![0.1, 0.2, 0.8, 0.9];
        let y = vec![0, 0, 1, 1];
        
        // With optimal threshold at 0.5: TP=2, TN=2, FP=0, FN=0
        // MCC = (2*2 - 0*0) / sqrt((2+0)*(2+0)*(2+0)*(2+0)) = 4/4 = 1.0
        
        let (mcc, _, _, _, _) = compute_mcc_and_metrics_from_value(&value, &y);
        assert!((mcc - 1.0).abs() < 1e-10, "MCC should be 1.0 for perfect classification");
    }

    #[test]
    fn test_compute_mcc_manual_example_2() {
        // Imperfect but balanced classification
        let value = vec![0.5, 0.5, 0.5, 0.5];
        let y = vec![0, 1, 0, 1];
        
        // With optimal threshold at 0.5: TP=1, TN=1, FP=1, FN=1
        // MCC = (1*1 - 1*1) / sqrt((1+1)*(1+1)*(1+1)*(1+1)) = 0/4 = 0.0
        
        let (mcc, _, _, _, _) = compute_mcc_and_metrics_from_value(&value, &y);
        assert!(mcc.abs() < 1e-10, "MCC should be ~0.0 for random-like classification but is {}", mcc);
    }

    #[test]
    fn test_compute_mcc_manual_example_3() {
        // Classification with class bias
        let value = vec![0.2, 0.3, 0.7, 0.8];
        let y = vec![0, 0, 0, 1];
        
        // With optimal threshold at ~0.75: TP=1, TN=3, FP=0, FN=0
        // MCC = (1*3 - 0*0) / sqrt((1+0)*(1+0)*(3+0)*(3+0)) = 3/sqrt(9) = 3/3 = 1.0
        
        let (mcc, _, _, _, _) = compute_mcc_and_metrics_from_value(&value, &y);
        assert!((mcc - 1.0).abs() < 1e-10, "MCC should be 1.0 for perfect imbalanced classification");
    }

    #[test]
    fn test_compute_mcc_manual_example_4() {
        // Intermediate case with manual calculation
        let value = vec![0.1, 0.4, 0.6, 0.9];
        let y = vec![0, 0, 1, 1];
        
        // With optimal threshold at 0.5: TP=2, TN=2, FP=0, FN=0
        // MCC = (2*2 - 0*0) / sqrt((2+0)*(2+0)*(2+0)*(2+0)) = 4/4 = 1.0
        
        let (mcc, _, _, _, _) = compute_mcc_and_metrics_from_value(&value, &y);
        assert!((mcc - 1.0).abs() < 1e-10, "MCC should be 1.0");
    }

    #[test]
    fn test_compute_mcc_manual_example_5() {
        // Classification with symmetric errors
        let value = vec![0.2, 0.6, 0.4, 0.8];
        let y = vec![0, 0, 1, 1];
        
        // Analysis of possible thresholds:
        // Threshold 0.5: TP=1, TN=1, FP=1, FN=1 → MCC = 0
        // Threshold 0.3: TP=2, TN=1, FP=1, FN=0 → MCC = (2*1-1*0)/sqrt(3*2*2*1) = 2/sqrt(12) ≈ 0.577
        
        let (mcc, _, _, _, _) = compute_mcc_and_metrics_from_value(&value, &y);
        assert!(mcc > 0.5 && mcc < 0.6, "MCC should be approximately 0.577");
    }

    #[test]
    fn test_auc_mcc_relationship_manual_verification() {
        // Verification on a case where we can calculate both manually
        let value = vec![0.1, 0.3, 0.7, 0.9];
        let y = vec![0, 0, 1, 1];
        
        let auc = compute_auc_from_value(&value, &y);
        let (mcc, _, _, _, _) = compute_mcc_and_metrics_from_value(&value, &y);
        
        // Expected values calculated manually
        assert!((auc - 1.0).abs() < 1e-10, "AUC should be 1.0");
        assert!((mcc - 1.0).abs() < 1e-10, "MCC should be 1.0");
        
        // For perfect classification, AUC = MCC = 1.0
        assert!((auc - mcc).abs() < 1e-10, "For perfect classification, AUC should equal MCC");
    }

    #[test]
    fn test_compute_auc_manual_complex_case() {
        // More complex case with multiple score values
        let value = vec![0.1, 0.3, 0.4, 0.6, 0.7, 0.9];
        let y = vec![0, 0, 1, 0, 1, 1];
        
        // Manual calculation:
        // Negatives: 0.1, 0.3, 0.6 (indices 0, 1, 3)
        // Positives: 0.4, 0.7, 0.9 (indices 2, 4, 5)
        // Pairs to evaluate: (0.1,0.4), (0.1,0.7), (0.1,0.9), (0.3,0.4), (0.3,0.7), (0.3,0.9), (0.6,0.4), (0.6,0.7), (0.6,0.9)
        // Correct pairs: (0.1,0.4), (0.1,0.7), (0.1,0.9), (0.3,0.4), (0.3,0.7), (0.3,0.9), (0.6,0.7), (0.6,0.9) = 8/9
        // AUC = 8/9 ≈ 0.8889
        
        let auc = compute_auc_from_value(&value, &y);
        assert!((auc - 8.0/9.0).abs() < 1e-10, "AUC should be 8/9 for this configuration");
    }

    #[test]
    fn test_compute_mcc_manual_complex_case() {
        // Complex case with known MCC calculation
        let value = vec![0.1, 0.3, 0.6, 0.8];
        let y = vec![0, 1, 1, 0];
        
        // With threshold 0.45: predictions [0, 0, 1, 1], actual [0, 1, 1, 0]
        // TP=1 (index 2), TN=1 (index 0), FP=1 (index 3), FN=1 (index 1)
        // MCC = (1*1 - 1*1) / sqrt((1+1)*(1+1)*(1+1)*(1+1)) = 0/4 = 0.0
        // But algorithm will find optimal threshold
        
        let (mcc, threshold, _, _, _) = compute_mcc_and_metrics_from_value(&value, &y);
        
        // The optimal threshold should give better than random performance
        assert!(mcc >= 0.0, "MCC should be non-negative for this case");
        assert!(!threshold.is_nan(), "Threshold should not be NaN");
    }

    #[test]
    fn test_compute_auc_edge_case_with_duplicate_scores() {
        // Edge case: multiple samples with same score
        let value = vec![0.2, 0.2, 0.8, 0.8];
        let y = vec![0, 1, 0, 1];
        
        // Manual calculation with ties:
        // Pairs: (0.2,0.8), (0.2,0.8) → both count as 1.0
        // Pairs: (0.2,0.8), (0.2,0.8) → both count as 1.0
        // But we also have ties: (0.2,0.2) and (0.8,0.8) each count as 0.5
        // Total correct comparisons considering ties
        // Expected AUC = 0.5 (since classification is essentially random due to ties)
        
        let auc = compute_auc_from_value(&value, &y);
        assert!((auc - 0.5).abs() < 1e-10, "AUC should be 0.5 for this tie scenario");
    }

}
