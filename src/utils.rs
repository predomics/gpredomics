use std::collections::HashMap;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::seq::SliceRandom;
use statrs::distribution::{Normal, ContinuousCDF};
use crate::data::Data;
use crate::param::ImportanceAggregation;

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

pub fn cliff_delta_global(baselines: &[f64], permuted: &mut [f64]) -> (i64,u64) {
    permuted.sort_by(|x,y| x.partial_cmp(y).unwrap());
    let m      = permuted.len() as u64;
    let mut gt = 0_u64;           // # baseline > perm
    let mut lt = 0_u64;           // # baseline < perm

    for &a in baselines {
        // nb d’éléments strictement < a
        let k  = permuted.partition_point(|&b| b < a) as u64;
        // nb d’éléments strictement > a
        let g  = m - permuted.partition_point(|&b| b <= a) as u64;
        gt += k;
        lt += g;
    }
    let diff  = gt as i64 - lt as i64;      // peut être négatif
    let total = (baselines.len() as u64) * m;
    (diff, total)
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
        ImportanceAggregation::Median => "Feature importance using median aggregation method\n",
        ImportanceAggregation::Mean => "Feature importance using mean aggregation method\n",
    });

    result.push_str(match aggregation_method {
        ImportanceAggregation::Median => "Legend: • = importance value, <- - -> = confidence interval (±MAD)\n\n",
        ImportanceAggregation::Mean => "Legend: • = importance value, <- - -> = confidence interval (±std dev)\n\n",
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
            marker_line.push('|');
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
        return format!("{:.1}", value);
    } else {
        if value == value.round() {
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

// Serde functions
// Serde functions pour la sérialisation JSON des HashMap avec clés non-string
pub mod serde_json_hashmap_numeric {
    use serde::{Serialize, Deserialize, Serializer, Deserializer};
    use std::collections::HashMap;
    
    // ===== FONCTIONS GÉNÉRIQUES =====
    
    /// Sérialisation pour HashMap<usize, T>
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
    
    /// Désérialisation pour HashMap<usize, T>
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
    
    /// Sérialisation pour HashMap<u32, T>
    pub fn serialize_u32<S, T>(
        map: &HashMap<u32, T>,
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
    
    /// Désérialisation pour HashMap<u32, T>
    pub fn deserialize_u32<'de, D, T>(
        deserializer: D,
    ) -> Result<HashMap<u32, T>, D::Error>
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
    
    /// Sérialisation pour HashMap<(usize, usize), T>
    pub fn serialize_tuple_usize<S, T>(
        map: &HashMap<(usize, usize), T>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
        T: Serialize + Clone,
    {
        let map_as_string: HashMap<String, T> = map.iter()
            .map(|(&(i, j), v)| (format!("{},{}", i, j), v.clone()))
            .collect();
        map_as_string.serialize(serializer)
    }
    
    /// Désérialisation pour HashMap<(usize, usize), T>
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
                if let (Ok(i), Ok(j)) = (parts[0].parse(), parts[1].parse()) {
                    map.insert((i, j), v);
                }
            }
        }
        Ok(map)
    }
    
    // ===== MODULES SPÉCIALISÉS =====
    
    /// Module pour HashMap<usize, i8> (Individual.features)
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
    
    /// Module pour HashMap<usize, u8> (Data.featureclass)
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
    
    /// Module pour HashMap<u32, String> (Population.featurenames)
    pub mod u32_string {
        use super::*;
        
        pub fn serialize<S>(
            map: &HashMap<u32, String>,
            serializer: S,
        ) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serialize_u32(map, serializer)
        }
        
        pub fn deserialize<'de, D>(
            deserializer: D,
        ) -> Result<HashMap<u32, String>, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserialize_u32(deserializer)
        }
    }
    
    /// Module pour HashMap<(usize, usize), f64> (Data.X)
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
    
    /// Module pour HashMap<usize, (f64, f64, f64)> (MCMC.featureprob)
    pub mod usize_tuple3_f64 {
        use super::*;
        
        pub fn serialize<S>(
            map: &HashMap<usize, (f64, f64, f64)>,
            serializer: S,
        ) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serialize_usize(map, serializer)
        }
        
        pub fn deserialize<'de, D>(
            deserializer: D,
        ) -> Result<HashMap<usize, (f64, f64, f64)>, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserialize_usize(deserializer)
        }
    }
    
    /// Module pour HashMap<usize, (f64, f64)> (MCMC.modelstats)
    pub mod usize_tuple2_f64 {
        use super::*;
        
        pub fn serialize<S>(
            map: &HashMap<usize, (f64, f64)>,
            serializer: S,
        ) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            serialize_usize(map, serializer)
        }
        
        pub fn deserialize<'de, D>(
            deserializer: D,
        ) -> Result<HashMap<usize, (f64, f64)>, D::Error>
        where
            D: Deserializer<'de>,
        {
            deserialize_usize(deserializer)
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

}
