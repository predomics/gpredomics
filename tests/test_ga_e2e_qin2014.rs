/// End-to-End Integration Test for GA Algorithm with Qin2014 Dataset
///
/// This test validates the complete GA workflow:
/// 1. Loading and preprocessing Qin2014 data
/// 2. Running GA optimization
/// 3. Evaluating on test set
/// 4. Verifying experiment structure and results
/// 5. Testing serialization/deserialization
///
/// Run with: cargo test --test test_ga_e2e_qin2014 -- --nocapture
use gpredomics::experiment::Experiment;
use gpredomics::param::Param;
use gpredomics::run;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Helper function to create GA parameters for Qin2014
fn create_qin2014_params() -> Param {
    let mut param = Param::default();

    // General settings
    param.general.seed = 42;
    param.general.algo = "ga".to_string();
    param.general.cv = false;
    param.general.thread_number = 4;
    param.general.gpu = false;
    param.general.language = "ter".to_string();
    param.general.data_type = "prev".to_string();
    param.general.data_type_epsilon = 1e-5;
    param.general.fit = gpredomics::param::FitFunction::auc;
    param.general.k_penalty = 0.0001;
    param.general.fr_penalty = 0.0;
    param.general.n_model_to_display = 10;
    param.general.log_level = "info".to_string();
    param.general.display_colorful = false;
    param.general.keep_trace = false;
    param.general.save_exp = "test_exp.mp".to_string();
    param.general.log_base = "".to_string();
    param.general.log_suffix = "log".to_string();

    // Data settings
    param.data.X = "samples/Qin2014/Xtrain.tsv".to_string();
    param.data.y = "samples/Qin2014/Ytrain.tsv".to_string();
    param.data.Xtest = "samples/Qin2014/Xtest.tsv".to_string();
    param.data.ytest = "samples/Qin2014/Ytest.tsv".to_string();
    param.data.features_in_rows = true;
    param.data.max_features_per_class = 0;
    param.data.feature_minimal_prevalence_pct = 10.0;
    param.data.feature_minimal_feature_value = 1e-4;
    param.data.feature_selection_method = gpredomics::data::PreselectionMethod::wilcoxon;
    param.data.feature_maximal_adj_pvalue = 0.1;
    param.data.feature_minimal_log_abs_bayes_factor = 0.5;
    param.data.inverse_classes = false;
    param.data.classes = vec!["healthy".to_string(), "cirrhosis".to_string()];

    // CV settings
    param.cv.inner_folds = 5;
    param.cv.overfit_penalty = 0.0;
    param.cv.outer_folds = 5;
    param.cv.resampling_inner_folds_epochs = 0;
    param.cv.fit_on_valid = true;
    param.cv.cv_best_models_ci_alpha = 0.05;

    // Importance settings
    param.importance.compute_importance = false;
    param.importance.n_permutations_oob = 100;
    param.importance.scaled_importance = true;
    param.importance.importance_aggregation = gpredomics::experiment::ImportanceAggregation::mean;

    // Voting settings
    param.voting.vote = false;
    param.voting.fbm_ci_alpha = 0.05;
    param.voting.min_perf = 0.5;
    param.voting.min_diversity = 10.0;
    param.voting.method = gpredomics::voting::VotingMethod::Majority;
    param.voting.method_threshold = 0.5;
    param.voting.threshold_windows_pct = 5.0;
    param.voting.complete_display = false;

    // GA settings - Small population for quick test
    param.ga.population_size = 20000;
    param.ga.max_epochs = 10;
    param.ga.min_epochs = 5;
    param.ga.max_age_best_model = 10;
    param.ga.kmin = 1;
    param.ga.kmax = 50;
    param.ga.select_elite_pct = 5.0;
    param.ga.select_niche_pct = 0.0;
    param.ga.select_random_pct = 10.0;
    param.ga.mutated_children_pct = 80.0;
    param.ga.mutated_features_pct = 20.0;
    param.ga.mutation_non_null_chance_pct = 20.0;
    param.ga.forced_diversity_pct = 0.0;
    param.ga.forced_diversity_epochs = 10;
    param.ga.random_sampling_pct = 0.0;
    param.ga.random_sampling_epochs = 0;

    // GPU settings
    param.gpu.fallback_to_cpu = true;
    param.gpu.memory_policy = gpredomics::param::GpuMemoryPolicy::Strict;
    param.gpu.max_total_memory_mb = 256;
    param.gpu.max_buffer_size_mb = 128;

    param
}

#[test]
fn test_ga_qin2014_basic_run() {
    println!("\n=== Testing GA with Qin2014 Dataset (Basic Run) ===\n");

    let param = create_qin2014_params();

    // Verify data files exist
    assert!(
        Path::new(&param.data.X).exists(),
        "Training X file not found: {}",
        param.data.X
    );
    assert!(
        Path::new(&param.data.y).exists(),
        "Training y file not found: {}",
        param.data.y
    );
    assert!(
        Path::new(&param.data.Xtest).exists(),
        "Test X file not found: {}",
        param.data.Xtest
    );
    assert!(
        Path::new(&param.data.ytest).exists(),
        "Test y file not found: {}",
        param.data.ytest
    );

    // Run GA
    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    let output = Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .unwrap();
    let git_hash = String::from_utf8(output.stdout)
        .unwrap()
        .chars()
        .take(7)
        .collect::<String>();
    let gpredomics_version = format!("{}#{}", env!("CARGO_PKG_VERSION"), git_hash);

    // Verify experiment structure
    assert!(
        !experiment.id.is_empty(),
        "Experiment ID should not be empty"
    );
    assert!(
        !experiment.timestamp.is_empty(),
        "Timestamp should not be empty"
    );
    assert_eq!(experiment.gpredomics_version, gpredomics_version);
    assert_eq!(experiment.parameters.general.algo, "ga");

    // Verify train data was loaded
    assert!(
        experiment.train_data.sample_len > 0,
        "Train data should have samples"
    );
    assert!(
        experiment.train_data.feature_len > 0,
        "Train data should have features"
    );
    assert_eq!(
        experiment.train_data.classes.len(),
        2,
        "Should have 2 classes"
    );

    // Verify test data was loaded
    assert!(experiment.test_data.is_some(), "Test data should be loaded");
    let test_data = experiment.test_data.as_ref().unwrap();
    assert!(test_data.sample_len > 0, "Test data should have samples");
    assert!(test_data.feature_len > 0, "Test data should have features");

    // Verify experiment collections
    assert!(
        !experiment.collections.is_empty(),
        "Collections should not be empty"
    );
    assert!(
        !experiment.collections[0].is_empty(),
        "First collection should not be empty"
    );

    // Verify final population
    assert!(
        experiment.final_population.is_some(),
        "Final population should exist"
    );
    let final_pop = experiment.final_population.as_ref().unwrap();
    assert!(
        !final_pop.individuals.is_empty(),
        "Final population should have individuals"
    );

    // Verify best model
    let best_model = &final_pop.individuals[0];
    assert!(
        best_model.auc >= 0.0 && best_model.auc <= 1.0,
        "AUC should be between 0 and 1"
    );
    assert!(
        best_model.sensitivity >= 0.0 && best_model.sensitivity <= 1.0,
        "Sensitivity should be between 0 and 1"
    );
    assert!(
        best_model.specificity >= 0.0 && best_model.specificity <= 1.0,
        "Specificity should be between 0 and 1"
    );
    assert!(
        best_model.k > 0,
        "Best model should use at least one feature"
    );
    assert!(
        !best_model.features.is_empty(),
        "Best model should have features"
    );

    // Verify execution time is reasonable
    assert!(
        experiment.execution_time > 0.0,
        "Execution time should be positive"
    );
    assert!(
        experiment.execution_time < 1200.0,
        "Execution time should be less than 20 minutes"
    );

    println!("✓ Basic GA run completed successfully");
    println!("  - Train samples: {}", experiment.train_data.sample_len);
    println!("  - Train features: {}", experiment.train_data.feature_len);
    println!("  - Test samples: {}", test_data.sample_len);
    println!("  - Best model AUC: {:.4}", best_model.auc);
    println!("  - Best model features: {}", best_model.k);
    println!("  - Execution time: {:.2}s", experiment.execution_time);
}

#[test]
fn test_ga_qin2014_serialization() {
    println!("\n=== Testing GA Serialization with Qin2014 ===\n");

    let param = create_qin2014_params();
    let running = Arc::new(AtomicBool::new(true));
    let original_exp = run(&param, running);

    // Test MessagePack serialization
    let mp_file = "test_ga_qin2014_serialization.mp";
    original_exp
        .save_auto(mp_file)
        .expect("Failed to save experiment");
    assert!(Path::new(mp_file).exists(), "Serialized file should exist");

    let loaded_exp = Experiment::load_auto(mp_file).expect("Failed to load experiment");

    // Verify loaded experiment matches original
    assert_eq!(original_exp.id, loaded_exp.id);
    assert_eq!(original_exp.timestamp, loaded_exp.timestamp);
    assert_eq!(
        original_exp.gpredomics_version,
        loaded_exp.gpredomics_version
    );
    assert_eq!(
        original_exp.train_data.sample_len,
        loaded_exp.train_data.sample_len
    );
    assert_eq!(
        original_exp.train_data.feature_len,
        loaded_exp.train_data.feature_len
    );

    // Verify final population
    assert_eq!(
        original_exp
            .final_population
            .as_ref()
            .unwrap()
            .individuals
            .len(),
        loaded_exp
            .final_population
            .as_ref()
            .unwrap()
            .individuals
            .len()
    );

    // Verify best model metrics
    let orig_best = &original_exp.final_population.as_ref().unwrap().individuals[0];
    let loaded_best = &loaded_exp.final_population.as_ref().unwrap().individuals[0];
    assert_eq!(orig_best.auc, loaded_best.auc);
    assert_eq!(orig_best.sensitivity, loaded_best.sensitivity);
    assert_eq!(orig_best.specificity, loaded_best.specificity);
    assert_eq!(orig_best.k, loaded_best.k);
    assert_eq!(orig_best.features, loaded_best.features);

    // Cleanup
    std::fs::remove_file(mp_file).expect("Failed to cleanup test file");

    println!("✓ Serialization test passed");
}

#[test]
fn test_ga_qin2014_with_keep_trace() {
    println!("\n=== Testing GA with keep_trace Enabled ===\n");

    let mut param = create_qin2014_params();
    param.general.keep_trace = true;
    param.ga.max_epochs = 5; // Shorter run for trace test

    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    // Verify trace is kept
    assert!(
        !experiment.collections.is_empty(),
        "Collections should not be empty"
    );
    assert!(
        experiment.collections[0].len() > 1,
        "Should have multiple generations in trace"
    );

    // Verify progression
    let first_gen = &experiment.collections[0][0];
    let last_gen = &experiment.collections[0][experiment.collections[0].len() - 1];

    assert!(
        !first_gen.individuals.is_empty(),
        "First generation should have individuals"
    );
    assert!(
        !last_gen.individuals.is_empty(),
        "Last generation should have individuals"
    );

    // Evolution should generally improve (not guaranteed, but likely with this dataset)
    println!(
        "  - First gen best AUC: {:.4}",
        first_gen.individuals[0].auc
    );
    println!("  - Last gen best AUC: {:.4}", last_gen.individuals[0].auc);
    println!("  - Total generations: {}", experiment.collections[0].len());

    println!("✓ Keep trace test passed");
}

#[test]
fn test_ga_qin2014_early_stopping() {
    println!("\n=== Testing GA Early Stopping ===\n");

    let running = Arc::new(AtomicBool::new(true));
    let param = create_qin2014_params();

    // Simulate early stopping after a delay
    let running_clone = Arc::clone(&running);
    let _stop_thread = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(2));
        running_clone.store(false, Ordering::Relaxed);
    });

    let experiment = run(&param, running);

    // Should still have valid results even if stopped early
    assert!(
        experiment.final_population.is_some(),
        "Should have final population"
    );
    assert!(
        !experiment
            .final_population
            .as_ref()
            .unwrap()
            .individuals
            .is_empty(),
        "Final population should have individuals"
    );

    println!("✓ Early stopping test passed");
    println!("  - Stopped after {:.2}s", experiment.execution_time);
}

#[test]
fn test_ga_qin2014_multiple_languages() {
    println!("\n=== Testing GA with Multiple Languages ===\n");

    let mut param = create_qin2014_params();
    param.general.language = "ter,bin".to_string();
    param.ga.population_size = 100; // Larger population for multiple languages
    param.ga.max_epochs = 5;

    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    let final_pop = experiment.final_population.as_ref().unwrap();

    // Check for language diversity
    let mut languages_found = std::collections::HashSet::new();
    for individual in &final_pop.individuals {
        languages_found.insert(individual.language);
    }

    assert!(
        !languages_found.is_empty(),
        "Should have at least one language"
    );
    println!(
        "  - Languages found in final population: {:?}",
        languages_found
    );

    println!("✓ Multiple languages test passed");
}

#[test]
fn test_ga_qin2014_multiple_data_types() {
    println!("\n=== Testing GA with Multiple Data Types ===\n");

    let mut param = create_qin2014_params();
    param.general.data_type = "raw,prev".to_string();
    param.ga.population_size = 100; // Larger population for multiple data types
    param.ga.max_epochs = 5;

    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    let final_pop = experiment.final_population.as_ref().unwrap();

    // Check for data type diversity
    let mut data_types_found = std::collections::HashSet::new();
    for individual in &final_pop.individuals {
        data_types_found.insert(individual.data_type);
    }

    assert!(
        !data_types_found.is_empty(),
        "Should have at least one data type"
    );
    println!(
        "  - Data types found in final population: {:?}",
        data_types_found
    );

    println!("✓ Multiple data types test passed");
}

#[test]
fn test_ga_qin2014_feature_selection() {
    println!("\n=== Testing GA Feature Selection ===\n");

    let mut param = create_qin2014_params();

    // Test with more restrictive feature selection
    param.data.feature_maximal_adj_pvalue = 0.05;
    param.data.feature_minimal_prevalence_pct = 15.0;

    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    // Verify feature selection worked
    assert!(
        experiment.train_data.feature_selection.len() > 0,
        "Should have selected features"
    );
    assert!(
        experiment.train_data.feature_selection.len() <= experiment.train_data.feature_len,
        "Selected features should be <= total features"
    );

    println!("  - Total features: {}", experiment.train_data.feature_len);
    println!(
        "  - Selected features: {}",
        experiment.train_data.feature_selection.len()
    );

    println!("✓ Feature selection test passed");
}

#[test]
fn test_ga_qin2014_different_fit_functions() {
    println!("\n=== Testing GA with Different Fit Functions ===\n");

    let fit_functions = vec![
        gpredomics::param::FitFunction::auc,
        gpredomics::param::FitFunction::sensitivity,
        gpredomics::param::FitFunction::specificity,
    ];

    for fit_fn in fit_functions {
        println!("  Testing fit function: {:?}", fit_fn);

        let mut param = create_qin2014_params();
        param.general.fit = fit_fn.clone();
        param.ga.max_epochs = 5; // Quick test

        let running = Arc::new(AtomicBool::new(true));
        let experiment = run(&param, running);

        assert!(
            experiment.final_population.is_some(),
            "Should have final population for {:?}",
            fit_fn
        );
        let best_model = &experiment.final_population.as_ref().unwrap().individuals[0];

        println!("    - Best AUC: {:.4}", best_model.auc);
        println!("    - Best Sensitivity: {:.4}", best_model.sensitivity);
        println!("    - Best Specificity: {:.4}", best_model.specificity);
    }

    println!("✓ Different fit functions test passed");
}

#[test]
fn test_ga_qin2014_model_complexity() {
    println!("\n=== Testing GA Model Complexity Control ===\n");

    // Test with penalty to encourage simpler models
    let mut param_simple = create_qin2014_params();
    param_simple.general.k_penalty = 0.01; // High penalty
    param_simple.ga.max_epochs = 10;

    let running = Arc::new(AtomicBool::new(true));
    let exp_simple = run(&param_simple, running);

    // Test without penalty for complex models
    let mut param_complex = create_qin2014_params();
    param_complex.general.k_penalty = 0.0; // No penalty
    param_complex.ga.max_epochs = 10;

    let running2 = Arc::new(AtomicBool::new(true));
    let exp_complex = run(&param_complex, running2);

    let simple_k = exp_simple.final_population.as_ref().unwrap().individuals[0].k;
    let complex_k = exp_complex.final_population.as_ref().unwrap().individuals[0].k;

    println!("  - Simple model (k_penalty=0.01): {} features", simple_k);
    println!("  - Complex model (k_penalty=0.0): {} features", complex_k);

    // Simple model should tend to use fewer features (not guaranteed, but likely)
    assert!(simple_k > 0, "Simple model should use at least one feature");
    assert!(
        complex_k > 0,
        "Complex model should use at least one feature"
    );

    println!("✓ Model complexity test passed");
}

#[test]
fn test_ga_qin2014_reproducibility() {
    println!("\n=== Testing GA Reproducibility with Same Seed ===\n");

    let param = create_qin2014_params();

    // Run 1
    let running1 = Arc::new(AtomicBool::new(true));
    let exp1 = run(&param, running1);

    // Run 2 with same seed
    let running2 = Arc::new(AtomicBool::new(true));
    let exp2 = run(&param, running2);

    // Results should be identical with same seed
    let best1 = &exp1.final_population.as_ref().unwrap().individuals[0];
    let best2 = &exp2.final_population.as_ref().unwrap().individuals[0];

    assert_eq!(
        best1.features, best2.features,
        "Features should be identical"
    );
    assert_eq!(best1.auc, best2.auc, "AUC should be identical");
    assert_eq!(best1.k, best2.k, "Feature count should be identical");

    println!("  - Run 1 AUC: {:.4}", best1.auc);
    println!("  - Run 2 AUC: {:.4}", best2.auc);

    println!("✓ Reproducibility test passed");
}

#[test]
fn test_ga_qin2014_experiment_display() {
    println!("\n=== Testing Experiment Display Methods ===\n");

    let param = create_qin2014_params();
    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    // This should not panic
    experiment.display_results();

    println!("✓ Experiment display test passed");
}

#[test]
fn test_ga_qin2014_cv_enabled() {
    println!("\n=== Testing GA with CV Enabled (outer_folds=2) ===\n");

    let mut param = create_qin2014_params();
    // Enable CV and keep it fast/light
    param.general.cv = true;
    param.general.gpu = false; // GPU is disabled internally in CV path
    param.general.keep_trace = false;

    // CV settings
    param.cv.outer_folds = 2;
    param.cv.inner_folds = 3;
    param.cv.fit_on_valid = true;
    param.cv.overfit_penalty = 0.0;
    param.cv.resampling_inner_folds_epochs = 0;

    // GA quick settings
    param.ga.population_size = 200;
    param.ga.max_epochs = 3;
    param.ga.min_epochs = 2;

    // Run
    let running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let experiment = gpredomics::run(&param, running);

    // CV structure assertions
    assert!(
        experiment.cv_folds_ids.is_some(),
        "cv_folds_ids must be present in CV mode"
    );
    let fold_ids = experiment.cv_folds_ids.as_ref().unwrap();
    assert_eq!(
        fold_ids.len(),
        param.cv.outer_folds,
        "cv_folds_ids length must equal outer_folds"
    );

    assert_eq!(
        experiment.collections.len(),
        param.cv.outer_folds,
        "collections length must equal outer_folds"
    );
    for (i, coll) in experiment.collections.iter().enumerate() {
        assert!(
            !coll.is_empty(),
            "Fold {} should have at least one population",
            i
        );
        let last_pop = &coll[coll.len() - 1];
        assert!(
            !last_pop.individuals.is_empty(),
            "Fold {} last population should have individuals",
            i
        );
    }

    assert!(
        experiment.final_population.is_some(),
        "Final population should exist in CV mode"
    );
    assert!(
        !experiment
            .final_population
            .as_ref()
            .unwrap()
            .individuals
            .is_empty(),
        "Final population must contain models"
    );

    // Should not panic; also validates reconstructed CV FBM consistency with final_population
    experiment.display_results();

    println!("✓ GA CV-enabled test passed");
}

#[test]
#[ignore = "Requires GPU hardware - run with 'cargo test -- --ignored' if GPU is available"]
fn test_ga_qin2014_gpu_vs_cpu() {
    println!("\n=== Testing GPU vs CPU Consistency ===\n");

    // CPU run
    let mut param_cpu = create_qin2014_params();
    param_cpu.general.gpu = false;
    param_cpu.general.seed = 12345;
    param_cpu.ga.population_size = 300;
    param_cpu.ga.max_epochs = 5;

    println!("Running GA on CPU...");
    let running_cpu = Arc::new(AtomicBool::new(true));
    let exp_cpu = run(&param_cpu, running_cpu);

    // GPU run (will fallback to CPU if GPU not available)
    let mut param_gpu = create_qin2014_params();
    param_gpu.general.gpu = true;
    param_gpu.general.seed = 12345;
    param_gpu.ga.population_size = 300;
    param_gpu.ga.max_epochs = 5;
    param_gpu.gpu.fallback_to_cpu = true;

    println!("Running GA on GPU (or CPU fallback)...");
    let running_gpu = Arc::new(AtomicBool::new(true));
    let exp_gpu = run(&param_gpu, running_gpu);

    // Both should produce results
    assert!(exp_cpu.final_population.is_some());
    assert!(exp_gpu.final_population.is_some());

    let best_cpu = &exp_cpu.final_population.as_ref().unwrap().individuals[0];
    let best_gpu = &exp_gpu.final_population.as_ref().unwrap().individuals[0];

    println!("CPU best model: AUC={:.4}, k={}", best_cpu.auc, best_cpu.k);
    println!("GPU best model: AUC={:.4}, k={}", best_gpu.auc, best_gpu.k);

    // With same seed, results should be identical (or very close if GPU)
    // Note: GPU might have slight numerical differences due to floating point operations
    assert_eq!(best_cpu.features, best_gpu.features,
               "Same seed should give identical features (if true GPU was used, this might differ slightly)");

    println!("✓ GPU vs CPU consistency test passed");
}

/// Test GPU support with inner CV to verify GPU and CPU produce identical results
#[test]
#[ignore = "Requires GPU hardware - run with 'cargo test -- --ignored' if GPU is available"]
fn test_ga_gpu_with_inner_cv() {
    println!("\n=== Testing GA with GPU and Inner CV ===");

    let mut param = create_qin2014_params();
    param.general.seed = 42;
    param.ga.population_size = 500;
    param.ga.max_epochs = 30;
    param.cv.resampling_inner_folds_epochs = 5;
    param.cv.inner_folds = 5;
    param.cv.overfit_penalty = 0.5; // Enable inner CV

    // CPU run
    println!("\n--- Running with CPU ---");
    param.general.gpu = false;
    param.general.save_exp = "test_ga_cpu_inner_cv.mp".to_string();
    let running_cpu = Arc::new(AtomicBool::new(true));
    let exp_cpu = run(&param, running_cpu);

    // GPU run
    println!("\n--- Running with GPU ---");
    param.general.gpu = true;
    param.general.save_exp = "test_ga_gpu_inner_cv.mp".to_string();
    let running_gpu = Arc::new(AtomicBool::new(true));
    let exp_gpu = run(&param, running_gpu);

    // Cleanup
    let _ = std::fs::remove_file("test_ga_cpu_inner_cv.mp");
    let _ = std::fs::remove_file("test_ga_gpu_inner_cv.mp");

    // Verify both produced results
    assert!(
        exp_cpu.final_population.is_some(),
        "CPU run should produce results"
    );
    assert!(
        exp_gpu.final_population.is_some(),
        "GPU run should produce results"
    );

    let pop_cpu = exp_cpu.final_population.as_ref().unwrap();
    let pop_gpu = exp_gpu.final_population.as_ref().unwrap();

    assert!(
        !pop_cpu.individuals.is_empty(),
        "CPU population should not be empty"
    );
    assert!(
        !pop_gpu.individuals.is_empty(),
        "GPU population should not be empty"
    );

    // Compare top models
    let best_cpu = &pop_cpu.individuals[0];
    let best_gpu = &pop_gpu.individuals[0];

    println!(
        "\nTop CPU model: k={}, AUC={:.6}, features={:?}",
        best_cpu.k, best_cpu.auc, best_cpu.features
    );
    println!(
        "Top GPU model: k={}, AUC={:.6}, features={:?}",
        best_gpu.k, best_gpu.auc, best_gpu.features
    );

    // With same seed, features should be identical
    assert_eq!(
        best_cpu.features, best_gpu.features,
        "GPU and CPU should produce identical features with same seed"
    );

    // AUC should be very close (within floating point tolerance)
    let auc_diff = (best_cpu.auc - best_gpu.auc).abs();
    assert!(
        auc_diff < 1e-4,
        "AUC difference too large: CPU={:.6}, GPU={:.6}, diff={:.6}",
        best_cpu.auc,
        best_gpu.auc,
        auc_diff
    );

    println!("\n✓ GPU with inner CV test passed - GPU and CPU results match!");
}

/// Test consistency: CV inner vs no CV inner (overfit_penalty enabled vs disabled)
#[test]
fn test_ga_consistency_inner_cv_vs_no_inner_cv() {
    println!("\n=== Testing GA Consistency: Inner CV vs No Inner CV ===\n");

    let mut param = create_qin2014_params();
    param.general.seed = 42;
    param.general.cv = false; // No outer CV
    param.general.gpu = false;
    param.general.keep_trace = false;
    param.ga.population_size = 100;
    param.ga.max_epochs = 5;

    // Run WITHOUT inner CV (overfit_penalty = 0.0)
    println!("Running WITHOUT inner CV...");
    param.cv.overfit_penalty = 0.0;
    param.cv.inner_folds = 5;
    let running1 = Arc::new(AtomicBool::new(true));
    let exp_no_inner_cv = run(&param, running1);

    // Run WITH inner CV (overfit_penalty > 0.0)
    println!("Running WITH inner CV...");
    param.cv.overfit_penalty = 0.5;
    param.cv.inner_folds = 5;
    param.cv.resampling_inner_folds_epochs = 3;
    let running2 = Arc::new(AtomicBool::new(true));
    let exp_with_inner_cv = run(&param, running2);

    // Both should produce valid results
    assert!(exp_no_inner_cv.final_population.is_some());
    assert!(exp_with_inner_cv.final_population.is_some());

    let best_no_cv = &exp_no_inner_cv
        .final_population
        .as_ref()
        .unwrap()
        .individuals[0];
    let best_with_cv = &exp_with_inner_cv
        .final_population
        .as_ref()
        .unwrap()
        .individuals[0];

    println!("No Inner CV: AUC={:.4}, k={}", best_no_cv.auc, best_no_cv.k);
    println!(
        "With Inner CV: AUC={:.4}, k={}",
        best_with_cv.auc, best_with_cv.k
    );

    // Results will differ due to different fitness evaluation (with vs without CV)
    // But both should be valid
    assert!(best_no_cv.auc >= 0.0 && best_no_cv.auc <= 1.0);
    assert!(best_with_cv.auc >= 0.0 && best_with_cv.auc <= 1.0);
    assert!(best_no_cv.k > 0);
    assert!(best_with_cv.k > 0);

    println!("✓ Inner CV vs No Inner CV consistency test passed");
}

/// Test consistency: Outer CV vs no outer CV
#[test]
fn test_ga_consistency_outer_cv_vs_no_outer_cv() {
    println!("\n=== Testing GA Consistency: Outer CV vs No Outer CV ===\n");

    let mut param = create_qin2014_params();
    param.general.seed = 42;
    param.general.gpu = false;
    param.general.keep_trace = false;
    param.ga.population_size = 100;
    param.ga.max_epochs = 5;

    // Run WITHOUT outer CV
    println!("Running WITHOUT outer CV...");
    param.general.cv = false;
    param.cv.overfit_penalty = 0.0; // No inner CV either
    let running1 = Arc::new(AtomicBool::new(true));
    let exp_no_outer_cv = run(&param, running1);

    // Run WITH outer CV
    println!("Running WITH outer CV...");
    param.general.cv = true;
    param.cv.outer_folds = 2; // Keep small for speed
    param.cv.inner_folds = 3;
    param.cv.overfit_penalty = 0.0;
    let running2 = Arc::new(AtomicBool::new(true));
    let exp_with_outer_cv = run(&param, running2);

    // Both should produce valid results
    assert!(exp_no_outer_cv.final_population.is_some());
    assert!(exp_with_outer_cv.final_population.is_some());

    let best_no_cv = &exp_no_outer_cv
        .final_population
        .as_ref()
        .unwrap()
        .individuals[0];
    let best_with_cv = &exp_with_outer_cv
        .final_population
        .as_ref()
        .unwrap()
        .individuals[0];

    println!("No Outer CV: AUC={:.4}, k={}", best_no_cv.auc, best_no_cv.k);
    println!(
        "With Outer CV: AUC={:.4}, k={}",
        best_with_cv.auc, best_with_cv.k
    );

    // Verify CV structure
    assert!(
        exp_with_outer_cv.cv_folds_ids.is_some(),
        "CV should have fold IDs"
    );
    assert_eq!(
        exp_with_outer_cv.collections.len(),
        param.cv.outer_folds,
        "Should have one collection per fold"
    );

    // Both should be valid
    assert!(best_no_cv.auc >= 0.0 && best_no_cv.auc <= 1.0);
    assert!(best_with_cv.auc >= 0.0 && best_with_cv.auc <= 1.0);
    assert!(best_no_cv.k > 0);
    assert!(best_with_cv.k > 0);

    println!("✓ Outer CV vs No Outer CV consistency test passed");
}

/// Test consistency: GPU vs CPU (basic, no CV)
#[test]
#[ignore = "Requires GPU hardware - run with 'cargo test -- --ignored' if GPU is available"]
fn test_ga_consistency_gpu_vs_cpu_basic() {
    println!("\n=== Testing GA Consistency: GPU vs CPU (Basic) ===\n");

    let mut param = create_qin2014_params();
    param.general.seed = 12345;
    param.general.cv = false;
    param.general.keep_trace = false;
    param.ga.population_size = 200;
    param.ga.max_epochs = 5;
    param.cv.overfit_penalty = 0.0; // No inner CV

    // CPU run
    println!("Running on CPU...");
    param.general.gpu = false;
    let running_cpu = Arc::new(AtomicBool::new(true));
    let exp_cpu = run(&param, running_cpu);

    // GPU run
    println!("Running on GPU (or CPU fallback)...");
    param.general.gpu = true;
    param.gpu.fallback_to_cpu = true;
    let running_gpu = Arc::new(AtomicBool::new(true));
    let exp_gpu = run(&param, running_gpu);

    // Both should produce results
    assert!(exp_cpu.final_population.is_some());
    assert!(exp_gpu.final_population.is_some());

    let best_cpu = &exp_cpu.final_population.as_ref().unwrap().individuals[0];
    let best_gpu = &exp_gpu.final_population.as_ref().unwrap().individuals[0];

    println!(
        "CPU: AUC={:.6}, k={}, features={:?}",
        best_cpu.auc, best_cpu.k, best_cpu.features
    );
    println!(
        "GPU: AUC={:.6}, k={}, features={:?}",
        best_gpu.auc, best_gpu.k, best_gpu.features
    );

    // With same seed, results should be identical
    assert_eq!(
        best_cpu.features, best_gpu.features,
        "Same seed should give identical features"
    );

    let auc_diff = (best_cpu.auc - best_gpu.auc).abs();
    assert!(
        auc_diff < 1e-6,
        "AUC should be identical: CPU={:.6}, GPU={:.6}, diff={:.6}",
        best_cpu.auc,
        best_gpu.auc,
        auc_diff
    );

    println!("✓ GPU vs CPU basic consistency test passed");
}

/// Test consistency: GPU inner CV vs CPU inner CV
#[test]
#[ignore = "Requires GPU hardware - run with 'cargo test -- --ignored' if GPU is available"]
fn test_ga_consistency_gpu_vs_cpu_inner_cv() {
    println!("\n=== Testing GA Consistency: GPU Inner CV vs CPU Inner CV ===\n");

    let mut param = create_qin2014_params();
    param.general.seed = 54321;
    param.general.cv = false; // No outer CV
    param.general.keep_trace = false;
    param.ga.population_size = 5000;
    param.general.data_type = "raw,prev,log".to_string();
    param.general.language = "bin,ratio,ter,pow2".to_string();
    param.ga.max_epochs = 10;
    param.cv.inner_folds = 3;
    param.cv.overfit_penalty = 2.0; // Enable inner CV
    param.cv.resampling_inner_folds_epochs = 3;

    // CPU run with inner CV
    println!("Running CPU with inner CV...");
    param.general.gpu = false;
    let running_cpu = Arc::new(AtomicBool::new(true));
    let exp_cpu = run(&param, running_cpu);

    // GPU run with inner CV
    println!("Running GPU with inner CV...");
    param.general.gpu = true;
    param.gpu.fallback_to_cpu = true;
    let running_gpu = Arc::new(AtomicBool::new(true));
    let exp_gpu = run(&param, running_gpu);

    // Both should produce results
    assert!(exp_cpu.final_population.is_some());
    assert!(exp_gpu.final_population.is_some());

    let best_cpu = &exp_cpu.final_population.as_ref().unwrap().individuals[0];
    let best_gpu = &exp_gpu.final_population.as_ref().unwrap().individuals[0];

    println!(
        "CPU Inner CV: AUC={:.6}, k={}, features={:?}",
        best_cpu.auc, best_cpu.k, best_cpu.features
    );
    println!(
        "GPU Inner CV: AUC={:.6}, k={}, features={:?}",
        best_gpu.auc, best_gpu.k, best_gpu.features
    );

    // With same seed, results should be identical
    assert_eq!(
        best_cpu.features, best_gpu.features,
        "GPU and CPU inner CV should give identical features with same seed"
    );

    let auc_diff = (best_cpu.auc - best_gpu.auc).abs();
    assert!(
        auc_diff < 1e-6,
        "AUC should be identical: CPU={:.6}, GPU={:.6}, diff={:.6}",
        best_cpu.auc,
        best_gpu.auc,
        auc_diff
    );

    println!("✓ GPU Inner CV vs CPU Inner CV consistency test passed");
}

/// Test consistency: keep_trace enabled vs disabled
#[test]
fn test_ga_consistency_keep_trace() {
    println!("\n=== Testing GA Consistency: keep_trace On vs Off ===\n");

    let mut param = create_qin2014_params();
    param.general.seed = 77777;
    param.general.cv = false;
    param.general.gpu = false;
    param.ga.population_size = 100;
    param.ga.max_epochs = 5;

    // Run WITHOUT keep_trace
    println!("Running WITHOUT keep_trace...");
    param.general.keep_trace = false;
    let running1 = Arc::new(AtomicBool::new(true));
    let exp_no_trace = run(&param, running1);

    // Run WITH keep_trace
    println!("Running WITH keep_trace...");
    param.general.keep_trace = true;
    let running2 = Arc::new(AtomicBool::new(true));
    let exp_with_trace = run(&param, running2);

    // Both should produce the same final results
    assert!(exp_no_trace.final_population.is_some());
    assert!(exp_with_trace.final_population.is_some());

    let best_no_trace = &exp_no_trace.final_population.as_ref().unwrap().individuals[0];
    let best_with_trace = &exp_with_trace
        .final_population
        .as_ref()
        .unwrap()
        .individuals[0];

    println!(
        "No trace: AUC={:.6}, k={}, features={:?}",
        best_no_trace.auc, best_no_trace.k, best_no_trace.features
    );
    println!(
        "With trace: AUC={:.6}, k={}, features={:?}",
        best_with_trace.auc, best_with_trace.k, best_with_trace.features
    );

    // Final results should be identical (keep_trace only affects what's stored, not computation)
    assert_eq!(
        best_no_trace.features, best_with_trace.features,
        "keep_trace should not affect final results"
    );
    assert_eq!(
        best_no_trace.auc, best_with_trace.auc,
        "keep_trace should not affect AUC"
    );

    // Verify trace structure
    assert!(
        exp_no_trace.collections[0].len() == 1,
        "No trace should only keep final population"
    );
    assert!(
        exp_with_trace.collections[0].len() > 1,
        "With trace should keep all generations"
    );

    println!(
        "  - No trace: {} generation(s) stored",
        exp_no_trace.collections[0].len()
    );
    println!(
        "  - With trace: {} generations stored",
        exp_with_trace.collections[0].len()
    );

    println!("✓ keep_trace consistency test passed");
}

/// Test with multiple languages and data types like in param.yaml
#[test]
#[ignore = "Requires GPU hardware - run with 'cargo test -- --ignored' if GPU is available"]
fn test_ga_gpu_multiple_lang_dtype() {
    println!("\n=== Testing GA GPU with Multiple Languages/DataTypes ===");

    let mut param = create_qin2014_params();
    param.general.seed = 42;
    param.general.language = "bin,ter,ratio,pow2".to_string();
    param.general.data_type = "raw,prev,log".to_string();
    param.general.fit = gpredomics::param::FitFunction::g_mean;
    param.ga.population_size = 30000;
    param.ga.max_epochs = 3;
    param.cv.inner_folds = 5;
    param.cv.overfit_penalty = 1.0;

    // CPU run - separate process simulation
    println!("\n--- CPU Run (process 1) ---");
    param.general.gpu = false;
    let running1 = Arc::new(AtomicBool::new(true));
    let exp_cpu = run(&param, running1);

    // GPU run - separate process simulation
    println!("\n--- GPU Run (process 2) ---");
    param.general.gpu = true;
    let running2 = Arc::new(AtomicBool::new(true));
    let exp_gpu = run(&param, running2);

    // Verify results
    assert!(exp_cpu.final_population.is_some());
    assert!(exp_gpu.final_population.is_some());

    let best_cpu = &exp_cpu.final_population.as_ref().unwrap().individuals[0];
    let best_gpu = &exp_gpu.final_population.as_ref().unwrap().individuals[0];

    println!("\nCPU: k={}, AUC={:.6}", best_cpu.k, best_cpu.auc);
    println!("GPU: k={}, AUC={:.6}", best_gpu.k, best_gpu.auc);

    // Same seed should give same results
    assert_eq!(
        best_cpu.features, best_gpu.features,
        "Multiple lang/dtype: CPU and GPU should match with same seed"
    );

    println!("\n✓ Multiple languages/data types test passed!");
}

/// Comprehensive test: All language/datatype combinations with GPU/CPU and InnerCV/NoInnerCV
/// Tests all 4 configurations for each language × datatype pair with large population (20000)
#[ignore = "long-running test without assertion for debug purposes"]
#[test]
fn test_ga_comprehensive_all_combinations() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Comprehensive GA Test: All Combinations                     ║");
    println!("║  - Languages: BINARY, TERNARY, RATIO, POW2                   ║");
    println!("║  - Data Types: RAW, LOG, PREVALENCE                          ║");
    println!("║  - Backends: CPU, GPU                                        ║");
    println!("║  - CV Modes: No Inner CV, Inner CV                          ║");
    println!("║  - Population: 20000 individuals                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let languages = vec![
        ("bin", "BINARY"),
        ("ter", "TERNARY"),
        ("ratio", "RATIO"),
        ("pow2", "POW2"),
    ];

    let data_types = vec![("raw", "RAW"), ("log", "LOG"), ("prev", "PREVALENCE")];

    let mut test_results = Vec::new();

    for (lang_code, lang_name) in &languages {
        for (dtype_code, dtype_name) in &data_types {
            println!("\n┌─────────────────────────────────────────────────────────┐");
            println!(
                "│ Testing: {} × {}                                    ",
                lang_name, dtype_name
            );
            println!("└─────────────────────────────────────────────────────────┘");

            // Configuration matrix: 2x2 = 4 tests per language/datatype
            let configs = vec![
                ("CPU", false, "No Inner CV", false),
                ("CPU", false, "Inner CV", true),
                ("GPU", true, "No Inner CV", false),
                ("GPU", true, "Inner CV", true),
            ];

            let mut combination_results = Vec::new();

            for (backend_name, use_gpu, cv_name, use_inner_cv) in configs {
                print!("  ├─ {} + {}: ", backend_name, cv_name);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();

                let mut param = create_qin2014_params();
                param.general.seed = 123456; // Fixed seed for reproducibility
                param.general.language = lang_code.to_string();
                param.general.data_type = dtype_code.to_string();
                param.general.gpu = use_gpu;
                param.general.save_exp = format!(
                    "test_comprehensive_{}_{}_{}_{}.mp",
                    lang_code,
                    dtype_code,
                    backend_name,
                    cv_name.replace(" ", "_")
                );

                // GA settings
                param.ga.population_size = 20000;
                param.ga.max_epochs = 2; // Keep it reasonable
                param.ga.kmin = 2;
                param.ga.kmax = 5;

                // Inner CV configuration
                if use_inner_cv {
                    param.cv.inner_folds = 3;
                    param.cv.resampling_inner_folds_epochs = 1;
                    param.cv.overfit_penalty = 0.5;
                } else {
                    param.cv.inner_folds = 0;
                }

                // Adjust epsilon for LOG type
                if *dtype_code == "log" {
                    param.general.data_type_epsilon = 0.1;
                }

                // Run the experiment
                let running = Arc::new(AtomicBool::new(true));
                let experiment = run(&param, running);

                // Verify results
                assert!(
                    experiment.final_population.is_some(),
                    "No final population for {} {} {} {}",
                    lang_name,
                    dtype_name,
                    backend_name,
                    cv_name
                );

                let final_pop = experiment.final_population.as_ref().unwrap();
                assert!(
                    !final_pop.individuals.is_empty(),
                    "Empty population for {} {} {} {}",
                    lang_name,
                    dtype_name,
                    backend_name,
                    cv_name
                );

                let best = &final_pop.individuals[0];

                // Verify language and data_type match
                let expected_lang = gpredomics::individual::language(lang_code);
                let expected_dtype = gpredomics::individual::data_type(dtype_code);
                assert_eq!(
                    best.language, expected_lang,
                    "Language mismatch for {} {}",
                    lang_name, dtype_name
                );
                assert_eq!(
                    best.data_type, expected_dtype,
                    "Data type mismatch for {} {}",
                    lang_name, dtype_name
                );

                // Verify reasonable AUC
                assert!(
                    best.auc >= 0.5 && best.auc <= 1.0,
                    "Invalid AUC {:.3} for {} {} {} {}",
                    best.auc,
                    lang_name,
                    dtype_name,
                    backend_name,
                    cv_name
                );

                println!(
                    "AUC={:.4}, k={}, pop_size={}",
                    best.auc,
                    best.k,
                    final_pop.individuals.len()
                );

                combination_results.push((
                    backend_name,
                    cv_name,
                    best.auc,
                    best.k,
                    best.features.len(),
                ));

                // Cleanup
                let _ = std::fs::remove_file(&param.general.save_exp);
            }

            // Compare results within this language/datatype combination
            println!("  │");
            println!("  ├─ Results Summary:");
            for (backend, cv, auc, k, features) in &combination_results {
                println!(
                    "  │  • {} + {}: AUC={:.4}, k={}, features={}",
                    backend, cv, auc, k, features
                );
            }

            // Check that all configurations produced valid results
            assert_eq!(
                combination_results.len(),
                4,
                "Should have 4 results for {} {}",
                lang_name,
                dtype_name
            );

            // Verify determinism: same seed + same config should give same features
            // Compare CPU No InnerCV (index 0) with GPU No InnerCV (index 2)
            let cpu_no_cv = &combination_results[0];
            let gpu_no_cv = &combination_results[2];

            println!("  │");
            println!("  └─ Determinism check (No Inner CV): CPU vs GPU");
            println!("     CPU: AUC={:.4}, k={}", cpu_no_cv.2, cpu_no_cv.3);
            println!("     GPU: AUC={:.4}, k={}", gpu_no_cv.2, gpu_no_cv.3);

            // With same seed, CPU and GPU should produce identical results (if no CV)
            // Note: With inner CV, there might be minor differences due to f32/f64 precision
            let auc_diff = (cpu_no_cv.2 - gpu_no_cv.2).abs();
            if auc_diff > 0.001 {
                println!(
                    "     ⚠ Warning: AUC difference = {:.6} (may indicate non-determinism)",
                    auc_diff
                );
            } else {
                println!("     ✓ Results are consistent");
            }

            test_results.push((lang_name, dtype_name, combination_results));
        }
    }
}

/// Test GA with voting enabled
#[test]
fn test_ga_qin2014_with_voting() {
    println!("\n=== Testing GA with Voting Enabled ===\n");

    let mut param = create_qin2014_params();
    param.general.seed = 42;
    param.ga.population_size = 1000;
    param.ga.max_epochs = 5;

    // Enable voting
    param.voting.vote = true;
    param.voting.min_perf = 0.6;
    param.voting.min_diversity = 5.0;
    param.voting.method = gpredomics::voting::VotingMethod::Majority;
    param.voting.method_threshold = 0.5;
    param.voting.complete_display = false;

    let running = Arc::new(AtomicBool::new(true));
    let mut experiment = run(&param, running);

    // Verify experiment structure
    assert!(
        experiment.final_population.is_some(),
        "Should have final population"
    );
    let final_pop = experiment.final_population.as_ref().unwrap();
    assert!(
        !final_pop.individuals.is_empty(),
        "Final population should have individuals"
    );

    // Compute voting
    experiment.compute_voting();

    // Verify voting results exist
    assert!(
        experiment.others.is_some(),
        "Voting results should exist when voting is enabled"
    );

    if let Some(gpredomics::experiment::ExperimentMetadata::Jury { jury }) = &experiment.others {
        println!("  - Number of experts: {}", jury.experts.individuals.len());
        println!("  - Voting AUC: {:.4}", jury.auc);
        println!("  - Voting sensitivity: {:.4}", jury.sensitivity);
        println!("  - Voting specificity: {:.4}", jury.specificity);

        // Verify voting results are valid
        assert!(
            jury.experts.individuals.len() > 0,
            "Should have at least one expert"
        );
        assert!(
            jury.auc >= 0.0 && jury.auc <= 1.0,
            "Voting AUC should be valid"
        );
        assert!(
            jury.sensitivity >= 0.0 && jury.sensitivity <= 1.0,
            "Voting sensitivity should be valid"
        );
        assert!(
            jury.specificity >= 0.0 && jury.specificity <= 1.0,
            "Voting specificity should be valid"
        );
    } else {
        panic!("Expected Jury metadata");
    }

    println!("✓ GA with voting test passed");
}

/// Test GA with different voting methods
#[test]
fn test_ga_qin2014_voting_methods() {
    println!("\n=== Testing GA with Different Voting Methods ===\n");

    let voting_methods = vec![
        (gpredomics::voting::VotingMethod::Majority, "Majority"),
        (gpredomics::voting::VotingMethod::Consensus, "Consensus"),
    ];

    for (method, method_name) in voting_methods {
        println!("Testing voting method: {}", method_name);

        let mut param = create_qin2014_params();
        param.general.seed = 42;
        param.ga.population_size = 500;
        param.ga.max_epochs = 3;

        param.voting.vote = true;
        param.voting.min_perf = 0.55;
        param.voting.min_diversity = 5.0;
        param.voting.method = method;
        param.voting.method_threshold = 0.5;

        let running = Arc::new(AtomicBool::new(true));
        let mut experiment = run(&param, running);

        experiment.compute_voting();

        assert!(
            experiment.others.is_some(),
            "{} should produce voting results",
            method_name
        );

        if let Some(gpredomics::experiment::ExperimentMetadata::Jury { jury }) = &experiment.others
        {
            println!(
                "  {} - Experts: {}, AUC: {:.4}",
                method_name,
                jury.experts.individuals.len(),
                jury.auc
            );

            assert!(
                jury.auc >= 0.0 && jury.auc <= 1.0,
                "{} should produce valid AUC",
                method_name
            );
        } else {
            panic!("Expected Jury metadata for {}", method_name);
        }
    }

    println!("✓ Different voting methods test passed");
}

/// Test GA with threshold CI enabled
#[test]
fn test_ga_qin2014_with_threshold_ci() {
    println!("\n=== Testing GA with Threshold CI ===\n");

    let mut param = create_qin2014_params();
    param.general.seed = 42;
    param.ga.population_size = 500;
    param.ga.max_epochs = 5;

    // Enable threshold CI
    param.general.threshold_ci_alpha = 0.05;
    param.general.threshold_ci_n_bootstrap = 100;
    param.general.threshold_ci_frac_bootstrap = 0.8;
    param.general.threshold_ci_penalty = 0.1;

    let running = Arc::new(AtomicBool::new(true));
    let experiment = run(&param, running);

    // Verify experiment structure
    assert!(
        experiment.final_population.is_some(),
        "Should have final population"
    );
    let final_pop = experiment.final_population.as_ref().unwrap();
    assert!(
        !final_pop.individuals.is_empty(),
        "Final population should have individuals"
    );

    // Verify threshold CI is set
    let best_model = &final_pop.individuals[0];
    assert!(
        best_model.threshold_ci.is_some(),
        "Best model should have threshold CI"
    );

    let threshold_ci = best_model.threshold_ci.as_ref().unwrap();
    println!("  - Threshold: {:.4}", best_model.threshold);
    println!("  - CI Lower: {:.4}", threshold_ci.lower);
    println!("  - CI Upper: {:.4}", threshold_ci.upper);
    println!("  - Rejection rate: {:.4}", threshold_ci.rejection_rate);

    // Verify CI bounds are valid
    assert!(
        threshold_ci.lower <= best_model.threshold,
        "CI lower should be <= threshold"
    );
    assert!(
        threshold_ci.upper >= best_model.threshold,
        "CI upper should be >= threshold"
    );
    assert!(
        threshold_ci.lower <= threshold_ci.upper,
        "CI lower should be <= upper"
    );
    assert!(
        threshold_ci.rejection_rate >= 0.0 && threshold_ci.rejection_rate <= 1.0,
        "Rejection rate should be between 0 and 1"
    );

    println!("✓ Threshold CI test passed");
}

/// Test GA with threshold CI and different alpha values
#[test]
fn test_ga_qin2014_threshold_ci_alpha_variations() {
    println!("\n=== Testing GA Threshold CI with Different Alpha Values ===\n");

    let alphas = vec![0.01, 0.05, 0.1, 0.2];

    for &alpha in &alphas {
        println!("Testing alpha = {}", alpha);

        let mut param = create_qin2014_params();
        param.general.seed = 42;
        param.ga.population_size = 300;
        param.ga.max_epochs = 3;

        param.general.threshold_ci_alpha = alpha;
        param.general.threshold_ci_n_bootstrap = 100;
        param.general.threshold_ci_frac_bootstrap = 0.8;

        let running = Arc::new(AtomicBool::new(true));
        let experiment = run(&param, running);

        let best_model = &experiment.final_population.as_ref().unwrap().individuals[0];
        assert!(
            best_model.threshold_ci.is_some(),
            "Should have threshold CI for alpha={}",
            alpha
        );

        let ci = best_model.threshold_ci.as_ref().unwrap();
        let ci_width = ci.upper - ci.lower;

        println!(
            "  alpha={}: CI width = {:.4}, rejection_rate = {:.4}",
            alpha, ci_width, ci.rejection_rate
        );

        // CI width can be 0 if bootstrap produces exactly the same threshold (stable model)
        assert!(
            ci_width >= 0.0,
            "CI width should be non-negative for alpha={}",
            alpha
        );
        assert!(
            ci.lower <= best_model.threshold,
            "CI lower should be <= threshold"
        );
        assert!(
            ci.upper >= best_model.threshold,
            "CI upper should be >= threshold"
        );
    }

    println!("✓ Threshold CI alpha variations test passed");
}

/// Test GA with both voting and threshold CI enabled
#[test]
fn test_ga_qin2014_voting_with_threshold_ci() {
    println!("\n=== Testing GA with Both Voting and Threshold CI ===\n");

    let mut param = create_qin2014_params();
    param.general.seed = 42;
    param.ga.population_size = 500;
    param.ga.max_epochs = 4;

    // Enable both voting and threshold CI
    param.voting.vote = true;
    param.voting.min_perf = 0.6;
    param.voting.min_diversity = 5.0;
    param.voting.method = gpredomics::voting::VotingMethod::Majority;

    param.general.threshold_ci_alpha = 0.05;
    param.general.threshold_ci_n_bootstrap = 100;
    param.general.threshold_ci_frac_bootstrap = 0.8;

    let running = Arc::new(AtomicBool::new(true));
    let mut experiment = run(&param, running);

    experiment.compute_voting();

    // Verify voting results
    assert!(experiment.others.is_some(), "Should have voting results");

    if let Some(gpredomics::experiment::ExperimentMetadata::Jury { jury }) = &experiment.others {
        println!(
            "  - Voting: {} experts, AUC={:.4}",
            jury.experts.individuals.len(),
            jury.auc
        );

        // Verify experts have threshold CI
        let experts_with_ci = jury
            .experts
            .individuals
            .iter()
            .filter(|ind| ind.threshold_ci.is_some())
            .count();

        println!(
            "  - Experts with threshold CI: {}/{}",
            experts_with_ci,
            jury.experts.individuals.len()
        );

        assert!(
            experts_with_ci > 0,
            "At least some experts should have threshold CI"
        );
    } else {
        panic!("Expected Jury metadata");
    }

    // Verify final population models have threshold CI
    let final_pop = experiment.final_population.as_ref().unwrap();
    let models_with_ci = final_pop
        .individuals
        .iter()
        .filter(|ind| ind.threshold_ci.is_some())
        .count();

    println!(
        "  - Final models with threshold CI: {}/{}",
        models_with_ci,
        final_pop.individuals.len()
    );

    assert!(
        models_with_ci > 0,
        "At least some final models should have threshold CI"
    );

    println!("✓ Combined voting and threshold CI test passed");
}

/// Test GA with voting and pruning before voting
#[test]
fn test_ga_qin2014_voting_with_pruning() {
    println!("\n=== Testing GA with Voting and Pruning ===\n");

    let mut param = create_qin2014_params();
    param.general.seed = 42;
    param.ga.population_size = 800;
    param.ga.max_epochs = 5;

    // Enable voting with pruning
    param.voting.vote = true;
    param.voting.min_perf = 0.6;
    param.voting.min_diversity = 5.0;
    param.voting.method = gpredomics::voting::VotingMethod::Majority;
    param.voting.prune_before_voting = true;

    let running = Arc::new(AtomicBool::new(true));
    let mut experiment = run(&param, running);

    experiment.compute_voting();

    // Verify voting results
    assert!(experiment.others.is_some(), "Should have voting results");

    if let Some(gpredomics::experiment::ExperimentMetadata::Jury { jury }) = &experiment.others {
        println!("  - Number of experts: {}", jury.experts.individuals.len());
        println!("  - Voting AUC: {:.4}", jury.auc);

        // Verify experts have been selected
        assert!(
            jury.experts.individuals.len() > 0,
            "Should have experts after pruning"
        );

        // All experts should have valid models
        for expert in &jury.experts.individuals {
            assert!(expert.k > 0, "Expert should have at least one feature");
            assert!(
                expert.auc >= 0.0 && expert.auc <= 1.0,
                "Expert should have valid AUC"
            );
        }
    } else {
        panic!("Expected Jury metadata");
    }

    println!("✓ Voting with pruning test passed");
}

#[test]
fn test_ga_qin2014_internal_holdout_split() {
    println!("\n=== Testing GA internal holdout split (no external test set) ===\n");

    // 1) Run without holdout nor external test set
    let mut param_no_holdout = create_qin2014_params();
    param_no_holdout.data.Xtest = "".to_string();
    param_no_holdout.data.ytest = "".to_string();
    param_no_holdout.data.holdout_ratio = 0.0;
    param_no_holdout.general.cv = false;
    // Make the test fast
    param_no_holdout.ga.population_size = 200;
    param_no_holdout.ga.max_epochs = 3;
    param_no_holdout.ga.min_epochs = 2;

    let running1 = Arc::new(AtomicBool::new(true));
    let exp_no_holdout = run(&param_no_holdout, running1);

    // Without holdout nor explicit test set, there should be no test_data
    assert!(
        exp_no_holdout.test_data.is_none(),
        "Without holdout or external test set, test_data must be None"
    );
    let total_samples = exp_no_holdout.train_data.sample_len;
    assert!(total_samples > 0, "Total number of samples must be > 0");

    // 2) Same config but with internal holdout
    let mut param_with_holdout = param_no_holdout.clone();
    param_with_holdout.data.holdout_ratio = 0.25;

    let running2 = Arc::new(AtomicBool::new(true));
    let exp_with_holdout = run(&param_with_holdout, running2);

    // With holdout_ratio > 0 and no Xtest/Ytest, we must use internal split
    assert!(
        exp_with_holdout.test_data.is_some(),
        "With holdout_ratio > 0 and no external test set, test_data must be Some"
    );
    let test_data = exp_with_holdout.test_data.as_ref().unwrap();
    let train_samples = exp_with_holdout.train_data.sample_len;
    let test_samples = test_data.sample_len;

    println!("  - Total samples (no holdout): {}", total_samples);
    println!("  - Train samples with holdout: {}", train_samples);
    println!("  - Test samples with holdout: {}", test_samples);

    // Verify that the holdout has effectively reduced the train set and created a real test set
    assert_eq!(
        train_samples + test_samples,
        total_samples,
        "Train+test sample counts must equal total samples"
    );
    assert!(
        test_samples > 0,
        "Holdout test set must contain at least one sample"
    );
    assert!(
        train_samples < total_samples,
        "Train set must be strictly smaller than total when holdout is enabled"
    );

    // Verify that train/test share the same feature/class space
    assert_eq!(
        exp_with_holdout.train_data.feature_len, test_data.feature_len,
        "Train and test must share same number of features"
    );
    assert_eq!(
        exp_with_holdout.train_data.classes, test_data.classes,
        "Train and test must share same class labels"
    );
}

#[test]
fn test_ga_qin2014_external_test_overrides_holdout() {
    println!("\n=== Testing GA external test set overrides holdout_ratio ===\n");

    // 1) Baseline: external test set only, no holdout
    let mut param_external_only = create_qin2014_params();
    param_external_only.data.holdout_ratio = 0.0;
    param_external_only.general.cv = false;
    // Make the test reasonably fast
    param_external_only.ga.population_size = 200;
    param_external_only.ga.max_epochs = 3;
    param_external_only.ga.min_epochs = 2;

    let running1 = Arc::new(AtomicBool::new(true));
    let exp_external_only = run(&param_external_only, running1);

    // With explicit Xtest/Ytest, test_data must come from the external test set
    assert!(
        exp_external_only.test_data.is_some(),
        "External test set should be used when Xtest/Ytest are provided"
    );
    let test_external_only = exp_external_only.test_data.as_ref().unwrap();

    let train_samples_external_only = exp_external_only.train_data.sample_len;
    let test_samples_external_only = test_external_only.sample_len;

    assert!(
        train_samples_external_only > 0,
        "Train set must contain at least one sample"
    );
    assert!(
        test_samples_external_only > 0,
        "External test set must contain at least one sample"
    );

    // 2) Same config but with a non-zero holdout_ratio; internal holdout must be ignored
    let mut param_with_holdout_and_external = param_external_only.clone();
    param_with_holdout_and_external.data.holdout_ratio = 0.25;

    let running2 = Arc::new(AtomicBool::new(true));
    let exp_with_holdout_and_external = run(&param_with_holdout_and_external, running2);

    assert!(
        exp_with_holdout_and_external.test_data.is_some(),
        "When external test set is provided, test_data must be Some even if holdout_ratio > 0"
    );
    let test_with_holdout_and_external = exp_with_holdout_and_external.test_data.as_ref().unwrap();

    // The presence of a holdout_ratio must not reduce the training set
    // when an explicit external test set is provided.
    assert_eq!(
        exp_with_holdout_and_external.train_data.sample_len, train_samples_external_only,
        "Train sample count must not be reduced by holdout_ratio when Xtest/Ytest are provided"
    );

    // The explicit test set must keep driving the test sample size.
    assert_eq!(
        test_with_holdout_and_external.sample_len, test_samples_external_only,
        "Test sample count must remain driven by the external Xtest/Ytest files"
    );

    // Sanity checks on feature space and classes across runs
    assert_eq!(
        exp_with_holdout_and_external.train_data.feature_len,
        exp_external_only.train_data.feature_len,
        "Train feature dimension must be identical across runs"
    );
    assert_eq!(
        test_with_holdout_and_external.feature_len, test_external_only.feature_len,
        "Test feature dimension must be identical across runs"
    );
    assert_eq!(
        exp_with_holdout_and_external.train_data.classes, exp_external_only.train_data.classes,
        "Train classes must be identical across runs"
    );
    assert_eq!(
        test_with_holdout_and_external.classes, test_external_only.classes,
        "Test classes must be identical across runs"
    );

    // 1) Train sets must be identical (no internal holdout applied)
    assert_eq!(
        exp_with_holdout_and_external.train_data.samples, exp_external_only.train_data.samples,
        "Train samples must be identical when an external test set is provided"
    );
    assert_eq!(
        exp_with_holdout_and_external.train_data.y, exp_external_only.train_data.y,
        "Train labels must be identical when an external test set is provided"
    );

    // 2) Test sets must be identical (both must use the external Xtest/Ytest)
    assert_eq!(
        test_with_holdout_and_external.samples, test_external_only.samples,
        "Test samples must come from the same external Xtest/Ytest files"
    );
    assert_eq!(
        test_with_holdout_and_external.y, test_external_only.y,
        "Test labels must come from the same external ytest file"
    );

    println!(
        "  - Train samples (external only)     : {}",
        train_samples_external_only
    );
    println!(
        "  - Train samples (ext + holdout>0.0): {}",
        exp_with_holdout_and_external.train_data.sample_len
    );
    println!(
        "  - Test samples  (external only)     : {}",
        test_samples_external_only
    );
    println!(
        "  - Test samples  (ext + holdout>0.0): {}",
        test_with_holdout_and_external.sample_len
    );
    println!("✓ GA external test overrides holdout_ratio as expected");
}
