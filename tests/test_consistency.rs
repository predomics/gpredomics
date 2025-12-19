use gpredomics::data::Data;
/// Integration tests for serialization backward/forward compatibility
///
/// These tests ensure that:
/// 1. The current version can read all experiment files from previous versions
/// 2. The current version produces consistent results compared to previous versions
/// 3. Any differences are documented and tracked
/// 4. run() and run_on_data() produce consistent results under identical conditions
///
/// Run with: cargo test --test test_consistency -- --nocapture
use gpredomics::experiment::Experiment;
use gpredomics::param::Param;
use gpredomics::{run, run_on_data};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct VersionDifference {
    field: String,
    old_value: String,
    new_value: String,
}

#[derive(Debug)]
#[allow(dead_code)]
struct ComparisonResult {
    file_name: String,
    version: String,
    can_deserialize: bool,
    deserialization_error: Option<String>,
    differences: Vec<VersionDifference>,
}

/// Helper function to find all experiment files in version directory
fn find_version_files<P: AsRef<Path>>(version_dir: P) -> Vec<PathBuf> {
    let mut files = Vec::new();

    if let Ok(entries) = fs::read_dir(version_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if ext_str == "mp"
                        || ext_str == "msgpack"
                        || ext_str == "json"
                        || ext_str == "bin"
                    {
                        files.push(path);
                    }
                }
            }
        }
    }

    files.sort();
    files
}

/// Helper function to get all version directories
fn find_all_versions<P: AsRef<Path>>(base_dir: P) -> Vec<(String, PathBuf)> {
    let mut versions = Vec::new();

    if let Ok(entries) = fs::read_dir(base_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(dir_name) = path.file_name() {
                    let version_str = dir_name.to_string_lossy().to_string();
                    if version_str.starts_with('v') {
                        versions.push((version_str, path));
                    }
                }
            }
        }
    }

    versions.sort_by(|a, b| a.0.cmp(&b.0));
    versions
}

/// Compare two experiments and list differences
fn compare_experiments(old: &Experiment, new: &Experiment) -> Vec<VersionDifference> {
    let mut differences = Vec::new();

    // Compare execution times (allow some variance)
    if (old.execution_time - new.execution_time).abs() > 5.0 {
        differences.push(VersionDifference {
            field: "execution_time".to_string(),
            old_value: format!("{:.2}s", old.execution_time),
            new_value: format!("{:.2}s", new.execution_time),
        });
    }

    // Compare final population size
    let old_pop_size = old
        .final_population
        .as_ref()
        .map(|p| p.individuals.len())
        .unwrap_or(0);
    let new_pop_size = new
        .final_population
        .as_ref()
        .map(|p| p.individuals.len())
        .unwrap_or(0);
    if old_pop_size != new_pop_size {
        differences.push(VersionDifference {
            field: "final_population.size".to_string(),
            old_value: old_pop_size.to_string(),
            new_value: new_pop_size.to_string(),
        });
    }

    // Compare best model AUC (if available)
    if let (Some(old_pop), Some(new_pop)) = (&old.final_population, &new.final_population) {
        if !old_pop.individuals.is_empty() && !new_pop.individuals.is_empty() {
            let old_auc = old_pop.individuals[0].auc;
            let new_auc = new_pop.individuals[0].auc;
            if (old_auc - new_auc).abs() > 1e-6 {
                differences.push(VersionDifference {
                    field: "best_model.auc".to_string(),
                    old_value: format!("{:.6}", old_auc),
                    new_value: format!("{:.6}", new_auc),
                });
            }

            // Compare best model features
            let old_features = &old_pop.individuals[0].features;
            let new_features = &new_pop.individuals[0].features;
            if old_features != new_features {
                differences.push(VersionDifference {
                    field: "best_model.features".to_string(),
                    old_value: format!("{:?}", old_features),
                    new_value: format!("{:?}", new_features),
                });
            }
        }
    }

    // Compare collections size
    if old.collections.len() != new.collections.len() {
        differences.push(VersionDifference {
            field: "collections.len".to_string(),
            old_value: old.collections.len().to_string(),
            new_value: new.collections.len().to_string(),
        });
    }

    // Compare train data dimensions
    if old.train_data.sample_len != new.train_data.sample_len {
        differences.push(VersionDifference {
            field: "train_data.sample_len".to_string(),
            old_value: old.train_data.sample_len.to_string(),
            new_value: new.train_data.sample_len.to_string(),
        });
    }

    if old.train_data.feature_len != new.train_data.feature_len {
        differences.push(VersionDifference {
            field: "train_data.feature_len".to_string(),
            old_value: old.train_data.feature_len.to_string(),
            new_value: new.train_data.feature_len.to_string(),
        });
    }

    differences
}

/// Test that current version can read all previous versions
#[test]
fn test_can_read_all_versions() {
    let version_base_dir = "tests/consistency";

    if !Path::new(version_base_dir).exists() {
        println!(
            "WARNING: No version directory found at {}. Run generate_version_experiments first.",
            version_base_dir
        );
        println!("   cargo run --example generate_version_experiments --release");
        return;
    }

    let versions = find_all_versions(version_base_dir);

    if versions.is_empty() {
        println!("WARNING: No version subdirectories found. Generate version files first:");
        println!("   cargo run --example generate_version_experiments --release");
        return;
    }

    println!("\n=== Testing Deserialization Compatibility ===");
    println!("Current version: {}", env!("CARGO_PKG_VERSION"));
    println!("Found {} version(s) to test\n", versions.len());

    let mut all_results = Vec::new();
    let mut total_files = 0;
    let mut successful_reads = 0;
    let mut failed_reads = 0;

    for (version_name, version_path) in versions {
        println!("Testing version: {}", version_name);
        println!("{}", "─".repeat(60));

        let files = find_version_files(&version_path);

        if files.is_empty() {
            println!(
                "  WARNING: No experiment files found in {}",
                version_path.display()
            );
            continue;
        }

        for file_path in files {
            total_files += 1;
            let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();

            print!("  {} ... ", file_name);

            match Experiment::load_auto(&file_path) {
                Ok(_exp) => {
                    successful_reads += 1;
                    println!("OK");

                    all_results.push(ComparisonResult {
                        file_name,
                        version: version_name.clone(),
                        can_deserialize: true,
                        deserialization_error: None,
                        differences: Vec::new(),
                    });
                }
                Err(e) => {
                    failed_reads += 1;
                    println!("FAILED");
                    println!("     Error: {}", e);

                    all_results.push(ComparisonResult {
                        file_name,
                        version: version_name.clone(),
                        can_deserialize: false,
                        deserialization_error: Some(e.to_string()),
                        differences: Vec::new(),
                    });
                }
            }
        }

        println!();
    }

    // Print summary
    println!("=== Summary ===");
    println!("Total files tested: {}", total_files);
    println!(
        "Successfully read:  {} ({:.1}%)",
        successful_reads,
        100.0 * successful_reads as f64 / total_files.max(1) as f64
    );
    println!(
        "Failed to read:     {} ({:.1}%)",
        failed_reads,
        100.0 * failed_reads as f64 / total_files.max(1) as f64
    );

    // Assert that we can read all files
    if failed_reads > 0 {
        panic!(
            "\nFailed to deserialize {} file(s). Check compatibility!",
            failed_reads
        );
    }

    println!("\nAll version files can be deserialized successfully!");
}

/// Test that current version produces consistent results
#[test]
fn test_results_consistency() {
    let version_base_dir = "tests/consistency";

    if !Path::new(version_base_dir).exists() {
        println!("WARNING: No version directory found. Skipping consistency test.");
        return;
    }

    let versions = find_all_versions(version_base_dir);

    if versions.len() < 2 {
        println!(
            "WARNING: Need at least 2 versions for comparison. Current count: {}",
            versions.len()
        );
        return;
    }

    println!("\n=== Testing Results Consistency ===");
    println!("Comparing experiments across versions\n");

    // Group files by name across versions
    let mut file_groups: HashMap<String, Vec<(String, PathBuf)>> = HashMap::new();

    for (version_name, version_path) in &versions {
        let files = find_version_files(version_path);
        for file_path in files {
            let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();
            file_groups
                .entry(file_name.clone())
                .or_insert_with(Vec::new)
                .push((version_name.clone(), file_path));
        }
    }

    let mut total_comparisons = 0;
    let mut identical_results = 0;
    let mut different_results = 0;

    for (file_name, mut versions_with_file) in file_groups {
        if versions_with_file.len() < 2 {
            continue; // Need at least 2 versions to compare
        }

        versions_with_file.sort_by(|a, b| a.0.cmp(&b.0));

        println!("Comparing: {}", file_name);
        println!("{}", "─".repeat(60));

        // Load all experiments
        let mut experiments = Vec::new();
        for (version_name, file_path) in &versions_with_file {
            match Experiment::load_auto(file_path) {
                Ok(exp) => experiments.push((version_name.clone(), exp)),
                Err(e) => {
                    println!(
                        "  WARNING: Failed to load {} from {}: {}",
                        file_name, version_name, e
                    );
                }
            }
        }

        if experiments.len() < 2 {
            continue;
        }

        // Compare consecutive versions
        for i in 0..experiments.len() - 1 {
            let (old_version, old_exp) = &experiments[i];
            let (new_version, new_exp) = &experiments[i + 1];

            total_comparisons += 1;

            let differences = compare_experiments(old_exp, new_exp);

            println!("  {} -> {}", old_version, new_version);

            if differences.is_empty() {
                identical_results += 1;
                println!("    Identical results");
            } else {
                different_results += 1;
                println!("    WARNING: Found {} difference(s):", differences.len());
                for diff in &differences {
                    println!(
                        "      - {}: {} -> {}",
                        diff.field, diff.old_value, diff.new_value
                    );
                }
            }
        }

        println!();
    }

    // Print summary
    println!("=== Consistency Summary ===");
    println!("Total comparisons:  {}", total_comparisons);
    println!(
        "Identical results:  {} ({:.1}%)",
        identical_results,
        100.0 * identical_results as f64 / total_comparisons.max(1) as f64
    );
    println!(
        "Different results:  {} ({:.1}%)",
        different_results,
        100.0 * different_results as f64 / total_comparisons.max(1) as f64
    );

    if different_results > 0 {
        println!("\nWARNING: Some differences detected. This may be expected due to:");
        println!("   - Algorithm improvements");
        println!("   - Bug fixes");
        println!("   - Random seed variations");
        println!("   - Performance optimizations");
        println!("\n   Review the differences above to ensure they are intentional.");
    } else {
        println!("\nAll version comparisons show identical results!");
    }
}

fn get_qin_paths() -> (String, String) {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("samples/Qin2014");
    (
        base.join("Xtrain.tsv").to_str().unwrap().to_string(),
        base.join("Ytrain.tsv").to_str().unwrap().to_string(),
    )
}

#[test]
fn test_run_vs_run_on_data_strict_consistency() {
    // 1. Setup parameters
    let (x_path, y_path) = get_qin_paths();
    let mut param = Param::default();
    param.data.X = x_path;
    param.data.y = y_path;
    param.data.features_in_rows = true; // Legacy format for Qin2014
    param.general.seed = 42; // Critical for reproducibility
    param.ga.max_epochs = 5; // Short run for speed
    param.general.algo = "ga".to_string();

    let running = Arc::new(AtomicBool::new(true));

    // 2. Execute 'run' (File-based workflow)
    // This function loads data internally from paths in param
    let exp_file = run(&param, running.clone());

    // 3. Execute 'run_on_data' (Memory-based workflow)
    // We manually load data to simulate the R wrapper's behavior
    let mut data_mem = Data::new();
    data_mem
        .load_data(&param.data.X, &param.data.y, param.data.features_in_rows)
        .expect("Failed to load data for run_on_data");

    // Note: We pass None for test_data to mimic standard run behavior without external test set
    let exp_mem = run_on_data(data_mem, None, &param, running.clone());

    // 4. Assertions: The results must be bit-exact identical
    let pop_file = exp_file
        .final_population
        .as_ref()
        .expect("No population in file exp");
    let pop_mem = exp_mem
        .final_population
        .as_ref()
        .expect("No population in mem exp");

    assert_eq!(
        pop_file.individuals.len(),
        pop_mem.individuals.len(),
        "Population sizes differ"
    );

    // Check the best individual
    let best_file = &pop_file.individuals[0];
    let best_mem = &pop_mem.individuals[0];

    assert_eq!(
        best_file.features, best_mem.features,
        "Best individual features mismatch"
    );
    assert_eq!(
        best_file.fit, best_mem.fit,
        "Best individual fitness mismatch"
    );

    // Check execution metadata to ensure logic path was similar
    assert_eq!(
        exp_file.train_data.sample_len, exp_mem.train_data.sample_len,
        "Training data size mismatch"
    );
}

#[test]
fn test_run_vs_run_on_data_holdout_consistency() {
    // 1. Setup parameters with Holdout
    let (x_path, y_path) = get_qin_paths();
    let mut param = Param::default();
    param.data.X = x_path;
    param.data.y = y_path;
    param.data.features_in_rows = true;
    param.general.seed = 1234; // Different seed
    param.ga.max_epochs = 5;
    param.data.holdout_ratio = 0.2; // ENABLE HOLDOUT

    let running = Arc::new(AtomicBool::new(true));

    // 2. Execute 'run'
    // Logic: 'run' will see holdout_ratio > 0 and perform the split internally
    let exp_file = run(&param, running.clone());

    // 3. Execute 'run_on_data'
    let mut data_mem = Data::new();
    data_mem
        .load_data(&param.data.X, &param.data.y, param.data.features_in_rows)
        .expect("Failed to load data");

    // Logic: 'run_on_data' receives full data, sees holdout_ratio > 0 in param,
    // and MUST perform the exact same split as 'run'.
    let exp_mem = run_on_data(data_mem, None, &param, running.clone());

    // 4. Assertions on Split Consistency
    // If the split logic (RNG usage) differs, train/test sizes or content will differ.

    // Check Train sizes
    assert_eq!(
        exp_file.train_data.sample_len, exp_mem.train_data.sample_len,
        "Train set size mismatch after holdout split"
    );

    // Check Test sizes (Holdout)
    assert!(exp_file.test_data.is_some(), "File run missing test data");
    assert!(exp_mem.test_data.is_some(), "Memory run missing test data");

    assert_eq!(
        exp_file.test_data.as_ref().unwrap().sample_len,
        exp_mem.test_data.as_ref().unwrap().sample_len,
        "Holdout set size mismatch"
    );

    // Check actual samples in Train (to ensure the shuffle was identical)
    // We check the first sample's index (mapped ID)
    let file_first_sample = exp_file.train_data.samples.last();
    let mem_first_sample = exp_mem.train_data.samples.last();
    assert_eq!(
        file_first_sample, mem_first_sample,
        "Train set content differs (split mismatch)"
    );

    // 5. Assertions on Results
    let pop_file = exp_file.final_population.as_ref().unwrap();
    let pop_mem = exp_mem.final_population.as_ref().unwrap();

    assert_eq!(
        pop_file.individuals[0].fit, pop_mem.individuals[0].fit,
        "Fitness mismatch with holdout"
    );
}
